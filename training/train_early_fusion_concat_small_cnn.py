from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from data_split import grouped_train_val_split, parse_window_timestamp

import train_fusion_cnn as fusion_pipeline

audio_pipeline = fusion_pipeline.audio_pipeline
imu_pipeline = fusion_pipeline.imu_pipeline
discover_paired_windows = fusion_pipeline.discover_paired_windows
load_imu_window = fusion_pipeline.load_imu_window
load_audio_waveform = fusion_pipeline.load_audio_waveform
make_backbone_callbacks = fusion_pipeline.make_backbone_callbacks

BACKBONE_EPOCHS = fusion_pipeline.BACKBONE_EPOCHS
BATCH_SIZE = fusion_pipeline.BATCH_SIZE

# Shared time axis both modalities get resampled onto before concatenation. Reusing the
# IMU window length keeps this close to the IMU pipeline's native resolution.
FUSED_WINDOW_LENGTH = imu_pipeline.WINDOW_LENGTH

EXPORT_DIR = Path(__file__).resolve().parent.parent / "models"
REPRESENTATIVE_SAMPLE_COUNT = 200

# Same architecture as train_early_fusion_concat_cnn.py, just with roughly a quarter of
# the Conv1D filters (64->32, 128->64). On real hardware this model's TFLite Micro
# CONV_2D op is the dominant cost (see src_inference/main.cpp's Timing log --
# ~3.6s/inference at 64/128 filters, using this library's unaccelerated reference int8
# kernel, since ESP-NN hardware acceleration isn't wired up for the Arduino/PlatformIO
# build). Fewer filters means proportionally fewer MACs for that same slow kernel to
# grind through, trading some accuracy for a roughly 4x cut in per-inference latency.
CONV1_FILTERS = 32
CONV2_FILTERS = 64


def build_model(input_shape, num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(CONV1_FILTERS, kernel_size=5, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(CONV2_FILTERS, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
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
    imu_paths, audio_paths, labels = discover_paired_windows(imu_pipeline.DATA_ROOT, audio_pipeline.DATA_ROOT)
    print(f"Found {len(labels)} paired IMU/audio windows across {len(set(labels))} classes.")

    x_imu = np.stack([load_imu_window(path) for path in imu_paths])  # (N, 200, 6)

    x_audio_waveforms = np.stack([load_audio_waveform(path) for path in audio_paths])
    x_audio_spec = audio_pipeline.compute_log_mel_spectrograms(x_audio_waveforms)[..., 0]  # (N, 198, 40)

    # Put both modalities on a shared time axis before concatenating along the feature dim
    # -- this is the input-level fusion step, done before any learned representation exists.
    x_audio_aligned = np.stack(
        [imu_pipeline.resample_window(spectrogram, FUSED_WINDOW_LENGTH) for spectrogram in x_audio_spec]
    )  # (N, 200, 40)

    x_fused = np.concatenate([x_imu, x_audio_aligned], axis=-1)  # (N, 200, 46)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    y_categorical = tf.keras.utils.to_categorical(y_encoded)
    num_classes = len(label_encoder.classes_)

    timestamps = np.array([parse_window_timestamp(p) for p in imu_paths])
    train_idx, val_idx = grouped_train_val_split(labels, timestamps, test_size=0.2, random_state=42)

    x_train, x_val = x_fused[train_idx], x_fused[val_idx]
    y_train, y_val = y_categorical[train_idx], y_categorical[val_idx]
    y_train_int = y_encoded[train_idx]

    # Per-channel normalization on the training split matters a lot more here than in the
    # single-modality scripts: raw IMU units and log-mel energies sit on very different
    # scales, and each of the 46 channels needs its own mean/std to be comparable.
    channel_mean = np.mean(x_train, axis=(0, 1), keepdims=True)
    channel_std = np.std(x_train, axis=(0, 1), keepdims=True)
    channel_std = np.where(channel_std == 0.0, 1.0, channel_std)
    x_train = (x_train - channel_mean) / channel_std
    x_val = (x_val - channel_mean) / channel_std

    class_weights_dict = dict(
        enumerate(
            class_weight.compute_class_weight(
                class_weight="balanced",
                classes=np.unique(y_train_int),
                y=y_train_int,
            )
        )
    )

    model = build_model(x_fused.shape[1:], num_classes)
    model.summary()

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=BACKBONE_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights_dict,
        callbacks=make_backbone_callbacks(),
        verbose=2,
    )

    _, accuracy = model.evaluate(x_val, y_val, verbose=0)
    print(f"\nEarly Fusion (small, {CONV1_FILTERS}/{CONV2_FILTERS} filters) Validation Accuracy: {accuracy * 100:.1f}%")

    y_val_true = np.argmax(y_val, axis=1)
    y_val_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)

    print("\nClassification report:")
    print(classification_report(y_val_true, y_val_pred, target_names=label_encoder.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_val_true, y_val_pred))

    # Persist the trained model plus everything the TFLite export step needs to
    # reproduce this exact preprocessing on-device: per-channel normalization stats,
    # the class label order (must match the model's output index order), and a
    # representative sample of already-normalized training windows for int8 calibration.
    # Saved under a distinct name from train_early_fusion_concat_cnn.py's output so both
    # the full-size and small models can coexist for comparison.
    EXPORT_DIR.mkdir(exist_ok=True)
    model.save(EXPORT_DIR / "early_fusion_concat_small.keras")
    np.savez(
        EXPORT_DIR / "early_fusion_concat_small_export_data.npz",
        channel_mean=channel_mean,
        channel_std=channel_std,
        classes=label_encoder.classes_,
        representative_samples=x_train[:REPRESENTATIVE_SAMPLE_COUNT],
    )
    print(f"\nSaved model and export data to {EXPORT_DIR}")


if __name__ == "__main__":
    main()
