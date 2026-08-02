import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

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


def build_model(input_shape, num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(64, kernel_size=5, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
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

    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        stratify=y_encoded,
        random_state=42,
    )

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
    print(f"\nEarly Fusion (raw channel-stacking) Validation Accuracy: {accuracy * 100:.1f}%")

    y_val_true = np.argmax(y_val, axis=1)
    y_val_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)

    print("\nClassification report:")
    print(classification_report(y_val_true, y_val_pred, target_names=label_encoder.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_val_true, y_val_pred))


if __name__ == "__main__":
    main()
