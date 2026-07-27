from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight


# Raw-audio HAR training configuration
SAMPLE_RATE = 16000
TARGET_SAMPLES = 32000  # 2-second windows at 16 kHz
DATA_ROOT = Path(__file__).resolve().parent / "data" / "windowed_data" / "right_hand_dominant" / "audio"

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

    for wav_path in sorted(data_root.rglob("*.wav")):
        label = wav_path.parent.name

        waveform, sample_rate = sf.read(wav_path, dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Unexpected sample rate {sample_rate} in {wav_path}")

        waveforms.append(fit_waveform(waveform, TARGET_SAMPLES))
        labels.append(label)

    if not waveforms:
        raise ValueError(f"No audio WAV files found under {data_root}")

    return np.stack(waveforms), np.array(labels)


def compute_log_mel_spectrograms(waveforms: np.ndarray) -> np.ndarray:
    # Log-mel features are far more sample-efficient than raw waveforms for
    # small audio datasets: they collapse 32k raw samples into a compact
    # time-frequency map, which is what makes a small 2D CNN viable here.
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


def main():
    x_raw, y_raw = load_audio_windows(DATA_ROOT)
    x_spec = compute_log_mel_spectrograms(x_raw)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    y_categorical = tf.keras.utils.to_categorical(y_encoded)

    x_train, x_val, y_train, y_val = train_test_split(
        x_spec,
        y_categorical,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42,
    )

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

    model = build_model(x_spec.shape[1:], len(label_encoder.classes_))
    model.summary()

    callbacks = [
        # tf.keras.callbacks.EarlyStopping(
        #     monitor="val_accuracy",
        #     patience=12,
        #     restore_best_weights=True,
        # ),
        # tf.keras.callbacks.ReduceLROnPlateau(
        #     monitor="val_loss",
        #     factor=0.5,
        #     patience=5,
        #     min_lr=1e-5,
        # ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=150,
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=2,
    )

    loss, accuracy = model.evaluate(validation_dataset, verbose=0)
    print(f"Validation Accuracy: {accuracy * 100:.1f}%")


if __name__ == "__main__":
    main()
