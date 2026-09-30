import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from data_split import grouped_train_val_split, parse_window_timestamp

import train_fusion_cnn as fusion_pipeline

audio_pipeline = fusion_pipeline.audio_pipeline
imu_pipeline = fusion_pipeline.imu_pipeline
build_audio_model = fusion_pipeline.build_audio_model
discover_paired_windows = fusion_pipeline.discover_paired_windows
load_imu_window = fusion_pipeline.load_imu_window
load_audio_waveform = fusion_pipeline.load_audio_waveform
make_backbone_callbacks = fusion_pipeline.make_backbone_callbacks

BACKBONE_EPOCHS = fusion_pipeline.BACKBONE_EPOCHS
BATCH_SIZE = fusion_pipeline.BATCH_SIZE

# IMU windows are resampled to a fixed length representing the same ~2s span as the audio
# windows, so this implied rate lets us compute a comparable per-axis STFT.
WINDOW_DURATION_SECONDS = audio_pipeline.TARGET_SAMPLES / audio_pipeline.SAMPLE_RATE
IMU_SAMPLE_RATE = imu_pipeline.WINDOW_LENGTH / WINDOW_DURATION_SECONDS
IMU_STFT_FRAME_LENGTH = 20
IMU_STFT_FRAME_STEP = 2
IMU_STFT_FFT_LENGTH = 32


def compute_imu_spectrograms(imu_windows: np.ndarray, target_shape) -> np.ndarray:
    """Per-axis STFT magnitude spectrogram, resized onto the same (time, frequency) grid
    as the audio log-mel spectrogram so every channel -- IMU or audio -- can be stacked
    together as channels of one multi-channel "image" before any modality-specific layer.

    Forced onto CPU: tf.signal.stft has no DirectML GPU kernel and the plugin segfaults
    (instead of cleanly falling back) if left unpinned -- same issue already fixed in
    train_audio_cnn.py's compute_log_mel_spectrograms, but this is a separate STFT call
    that was never touched. One-time preprocessing step, so the CPU cost is negligible."""
    with tf.device("/CPU:0"):
        axis_spectrograms = []
        for axis_index in range(imu_windows.shape[-1]):
            stft = tf.signal.stft(
                imu_windows[..., axis_index],
                frame_length=IMU_STFT_FRAME_LENGTH,
                frame_step=IMU_STFT_FRAME_STEP,
                fft_length=IMU_STFT_FFT_LENGTH,
            )
            log_magnitude = tf.math.log(tf.abs(stft) + 1e-6)
            resized = tf.image.resize(log_magnitude[..., tf.newaxis], target_shape)[..., 0]
            axis_spectrograms.append(resized)
        return tf.stack(axis_spectrograms, axis=-1).numpy()


def main():
    imu_paths, audio_paths, labels = discover_paired_windows(imu_pipeline.DATA_ROOT, audio_pipeline.DATA_ROOT)
    print(f"Found {len(labels)} paired IMU/audio windows across {len(set(labels))} classes.")

    x_imu_raw = np.stack([load_imu_window(path) for path in imu_paths])
    x_audio_raw = np.stack([load_audio_waveform(path) for path in audio_paths])
    x_audio_spec = audio_pipeline.compute_log_mel_spectrograms(x_audio_raw)  # (N, T, 40, 1)

    target_shape = x_audio_spec.shape[1:3]  # (T, 40)
    x_imu_spec = compute_imu_spectrograms(x_imu_raw, target_shape)  # (N, T, 40, 6)

    x_fused = np.concatenate([x_imu_spec, x_audio_spec], axis=-1)  # (N, T, 40, 7)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    y_categorical = tf.keras.utils.to_categorical(y_encoded)
    num_classes = len(label_encoder.classes_)

    timestamps = np.array([parse_window_timestamp(p) for p in imu_paths])
    train_idx, val_idx = grouped_train_val_split(labels, timestamps, test_size=0.2, random_state=42)

    x_train, x_val = x_fused[train_idx], x_fused[val_idx]
    y_train, y_val = y_categorical[train_idx], y_categorical[val_idx]
    y_train_int = y_encoded[train_idx]

    # Per-channel normalization: IMU-derived spectrogram magnitudes and the audio log-mel
    # channel sit on different scales, so each of the 7 channels gets its own mean/std.
    channel_mean = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
    channel_std = np.std(x_train, axis=(0, 1, 2), keepdims=True)
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

    model = build_audio_model(x_fused.shape[1:], num_classes)
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
    print(f"\nEarly Fusion (unified spectrogram) Validation Accuracy: {accuracy * 100:.1f}%")

    y_val_true = np.argmax(y_val, axis=1)
    y_val_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)

    print("\nClassification report:")
    print(classification_report(y_val_true, y_val_pred, target_names=label_encoder.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_val_true, y_val_pred))


if __name__ == "__main__":
    main()
