from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from best_weights import RestoreBestWeights
from data_split import grouped_train_val_split, parse_window_timestamp


# Raw-audio HAR training configuration
SAMPLE_RATE = 16000
TARGET_SAMPLES = 32000  # 2-second windows at 16 kHz
# See train_raw_cnn.py's DATA_ROOT comment: same combined, compartment_interaction-
# excluded dataset, kept consistent across modalities so fusion scripts pair correctly.
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "windowed_data_combined_no_ci" / "right_hand_dominant" / "audio"

# STFT / mel-filterbank settings (25 ms frames, 10 ms hop @ 16 kHz)
FRAME_LENGTH = 400
FRAME_STEP = 160
FFT_LENGTH = 512
NUM_MEL_BINS = 40
FMIN_HZ = 20.0
FMAX_HZ = 8000.0


def fit_waveform(waveform: np.ndarray, target_length: int) -> np.ndarray:
    """Pad or truncate a single waveform to a fixed number of samples."""
    if waveform.shape[0] == target_length:
        return waveform.astype(np.float32)

    if waveform.shape[0] > target_length:
        return waveform[:target_length].astype(np.float32)

    padded = np.zeros(target_length, dtype=np.float32)
    padded[: waveform.shape[0]] = waveform
    return padded


def load_audio_windows(data_root: Path):
    waveforms = []
    labels = []
    timestamps = []

    for wav_path in sorted(data_root.rglob("*.wav")):
        label = wav_path.parent.name

        waveform, sample_rate = sf.read(wav_path, dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Unexpected sample rate {sample_rate} in {wav_path}")

        waveforms.append(fit_waveform(waveform, TARGET_SAMPLES))
        labels.append(label)
        timestamps.append(parse_window_timestamp(wav_path))

    if not waveforms:
        raise ValueError(f"No audio WAV files found under {data_root}")

    return np.stack(waveforms), np.array(labels), np.array(timestamps)


def compute_log_mel_spectrograms(waveforms: np.ndarray) -> np.ndarray:
    # Log-mel features are far more sample-efficient than raw waveforms for
    # small audio datasets: they collapse 32k raw samples into a compact
    # time-frequency map, which is what makes a small 2D CNN viable here.
    # Forced onto CPU: tf.signal.stft has no DirectML GPU kernel and the
    # plugin segfaults (instead of cleanly falling back) if left unpinned.
    # This runs once as a preprocessing step, so the CPU cost is negligible.
    with tf.device("/CPU:0"):
        stft = tf.signal.stft(
            waveforms,
            frame_length=FRAME_LENGTH,
            frame_step=FRAME_STEP,
            fft_length=FFT_LENGTH,
        )
        spectrograms = tf.abs(stft)

        mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=NUM_MEL_BINS,
            num_spectrogram_bins=spectrograms.shape[-1],
            sample_rate=SAMPLE_RATE,
            lower_edge_hertz=FMIN_HZ,
            upper_edge_hertz=FMAX_HZ,
        )
        mel_spectrograms = tf.tensordot(spectrograms, mel_weight_matrix, axes=1)
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

        return log_mel_spectrograms.numpy()[..., np.newaxis]  # add channel dim


def build_model(input_shape, num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model_temporal(input_shape, num_classes: int) -> tf.keras.Model:
    """Same conv frontend as build_model, but replaces GlobalAveragePooling2D with a
    GRU over the time axis instead of collapsing it. GAP averages away exactly the
    envelope/rhythm shape (periodic loud-quiet bursts vs. a continuous tone) that
    distinguishes e.g. hammering from screw_tightening -- two classes with similar
    average spectral energy but very different temporal patterns. Reshaping to
    (time, freq*channels) and feeding a GRU lets the model use that pattern instead
    of only the window's average timbre."""
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    time_steps, freq_bins, channels = x.shape[1], x.shape[2], x.shape[3]
    x = tf.keras.layers.Reshape((time_steps, freq_bins * channels))(x)
    # Keras re-checks eligibility for a fused CudnnRNN kernel inside GRU.call()
    # on every invocation (not fixed at layer-construction time, so wrapping
    # construction in tf.device("/CPU:0") does not prevent it) and DirectML
    # doesn't implement that op -- hard crash with no fallback. unroll=True
    # makes the layer emit literal per-step ops instead of the fused RNN op,
    # which sidesteps the issue entirely. Cheap here: only 49 timesteps.
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, unroll=True))(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model_hybrid(input_shape, num_classes: int) -> tf.keras.Model:
    """Concatenates GAP's average-timbre summary with the GRU's temporal-pattern
    summary instead of picking one or the other. build_model_temporal fixed
    hammering/screw_tightening (rhythm-distinguishable) by replacing GAP with a GRU,
    but that also collapsed measuring into idle (f1 0.00) -- both are quiet, low-energy
    classes with no strong rhythm, so a purely rhythm-focused representation has nothing
    to key off for them, whereas plain average-energy (what GAP captures) was already
    telling them apart, if weakly. Giving the classifier both signals at once should let
    it use whichever is actually informative per class, rather than losing one to gain
    the other."""
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    gap_branch = tf.keras.layers.GlobalAveragePooling2D()(x)

    time_steps, freq_bins, channels = x.shape[1], x.shape[2], x.shape[3]
    temporal = tf.keras.layers.Reshape((time_steps, freq_bins * channels))(x)
    gru_branch = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, unroll=True))(temporal)

    merged = tf.keras.layers.Concatenate()([gap_branch, gru_branch])
    merged = tf.keras.layers.Dense(64, activation="relu")(merged)
    merged = tf.keras.layers.Dropout(0.4)(merged)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(merged)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Which builder main() uses for standalone (single-modality) audio training. Hybrid
# won this session's architecture comparison for standalone audio (74.3% vs. GAP's
# 65.7%, on the screw_loosening-split dataset) by keeping GAP's average-timbre summary
# alongside a GRU's rhythm/envelope summary instead of replacing one with the other.
#
# NOTE: this does NOT affect fusion. train_fusion_cnn.py binds
# `build_audio_model = audio_pipeline.build_model` directly (a fixed function
# reference resolved at import time), not through this variable -- and that's
# intentional, not an oversight: swapping the hybrid backbone into fusion was tested
# and made per-class-weighted decision fusion worse (89.6% -> 88.4%), because it
# changed which mistakes the audio backbone makes in ways that overlap more with the
# IMU backbone's own mistakes, leaving less complementary signal for fusion to
# exploit. Fusion deliberately keeps using plain GAP.
MODEL_BUILDER = build_model_hybrid


def main():
    x_raw, y_raw, timestamps = load_audio_windows(DATA_ROOT)
    x_spec = compute_log_mel_spectrograms(x_raw)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    y_categorical = tf.keras.utils.to_categorical(y_encoded)

    # Grouped split: windows overlap 50%, so a plain random/stratified split can leak
    # overlapping samples from the same activity instance across train/val (see data_split.py).
    train_idx, val_idx = grouped_train_val_split(y_raw, timestamps, test_size=0.2, random_state=42)
    x_train, x_val = x_spec[train_idx], x_spec[val_idx]
    y_train, y_val = y_categorical[train_idx], y_categorical[val_idx]

    # Normalize per mel bin using only the training split.
    bin_mean = np.mean(x_train, axis=(0, 1), keepdims=True)
    bin_std = np.std(x_train, axis=(0, 1), keepdims=True)
    bin_std = np.where(bin_std == 0.0, 1.0, bin_std)

    x_train = (x_train - bin_mean) / bin_std
    x_val = (x_val - bin_mean) / bin_std

    y_train_int = np.argmax(y_train, axis=1)
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_int),
        y=y_train_int,
    )
    class_weights_dict = dict(enumerate(weights))

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(min(len(x_train), 1024))
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )

    validation_dataset = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = MODEL_BUILDER(x_spec.shape[1:], len(label_encoder.classes_))
    model.summary()

    epochs = 150
    # No early stopping -- every run goes the full epoch budget -- but the evaluated
    # model is the best-val_accuracy epoch rather than whatever epoch 150 happens to
    # land on (the audio validation curve is noisy late in training).
    callbacks = [RestoreBestWeights(monitor="val_accuracy")]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=2,
    )

    loss, accuracy = model.evaluate(validation_dataset, verbose=0)
    print(f"Validation Accuracy: {accuracy * 100:.1f}%")

    y_val_true = np.argmax(y_val, axis=1)
    y_val_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)

    print("\nClassification report:")
    print(classification_report(y_val_true, y_val_pred, target_names=label_encoder.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_val_true, y_val_pred))


if __name__ == "__main__":
    main()
