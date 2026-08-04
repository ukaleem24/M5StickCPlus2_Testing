from pathlib import Path

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

EXPORT_DIR = Path(__file__).resolve().parent / "models"
REPRESENTATIVE_SAMPLE_COUNT = 200

# Shrunk versions of train_raw_cnn.py's / train_audio_cnn.py's backbones, shared by both
# the embedding-fusion and decision-fusion deployments. The audio backbone in particular
# needed more than a filter-count cut: its Conv2D runs over the full 198x40 log-mel grid
# (not resampled to 200 like the deployed early-fusion model), and 2D conv cost scales
# with spatial position count -- at the original 32/64 filters this measured out to an
# estimated ~26-30s/inference on this hardware's unaccelerated TFLite Micro kernel. The
# first pool going 2->4 cuts the spatial grid the second (expensive, many-channel) conv
# layer sees by 4x on top of the filter reduction.
IMU_CONV1_FILTERS = 32
IMU_CONV2_FILTERS = 64
AUDIO_CONV1_FILTERS = 16
AUDIO_CONV2_FILTERS = 32
AUDIO_FIRST_POOL_SIZE = 4


def build_imu_model_small(num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(imu_pipeline.WINDOW_LENGTH, len(imu_pipeline.CHANNEL_COLUMNS))),
            tf.keras.layers.Conv1D(IMU_CONV1_FILTERS, kernel_size=5, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(IMU_CONV2_FILTERS, kernel_size=3, padding="same", activation="relu"),
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


def build_audio_model_small(input_shape, num_classes: int) -> tf.keras.Model:
    # Strided convs, not stride-1-then-pool: a stride-1 "same" conv still computes its
    # output at the FULL 198x40 input resolution before any pooling shrinks it -- that
    # first conv's output tensor alone is 198*40*16 ~= 127KB, which blew the tensor
    # arena (and would have eaten most of the 320KB RAM budget on its own, leaving no
    # room for a second simultaneously-loaded model). strides=4 makes the downsampling
    # happen AS PART OF the conv, so that full-resolution tensor is never materialized --
    # peak activation memory drops to a few KB instead of ~127KB, an architecture-level
    # fix that no amount of arena resizing could have substituted for.
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(
                AUDIO_CONV1_FILTERS, kernel_size=3, strides=AUDIO_FIRST_POOL_SIZE, padding="same", activation="relu"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(AUDIO_CONV2_FILTERS, kernel_size=3, strides=2, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
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
    imu_paths, audio_paths, labels = discover_paired_windows(imu_pipeline.DATA_ROOT, audio_pipeline.DATA_ROOT)
    print(f"Found {len(labels)} paired IMU/audio windows across {len(set(labels))} classes.")

    x_imu_raw = np.stack([load_imu_window(path) for path in imu_paths])
    x_audio_raw = np.stack([load_audio_waveform(path) for path in audio_paths])
    x_audio_spec = audio_pipeline.compute_log_mel_spectrograms(x_audio_raw)  # (N, 198, 40, 1), native shape

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

    x_imu_train, x_imu_val = x_imu_raw[train_idx], x_imu_raw[val_idx]
    x_audio_train, x_audio_val = x_audio_spec[train_idx], x_audio_spec[val_idx]
    y_train, y_val = y_categorical[train_idx], y_categorical[val_idx]
    y_train_int = y_encoded[train_idx]

    imu_channel_mean = np.mean(x_imu_train, axis=(0, 1), keepdims=True)
    imu_channel_std = np.std(x_imu_train, axis=(0, 1), keepdims=True)
    imu_channel_std = np.where(imu_channel_std == 0.0, 1.0, imu_channel_std)
    x_imu_train = (x_imu_train - imu_channel_mean) / imu_channel_std
    x_imu_val = (x_imu_val - imu_channel_mean) / imu_channel_std

    audio_bin_mean = np.mean(x_audio_train, axis=(0, 1), keepdims=True)
    audio_bin_std = np.std(x_audio_train, axis=(0, 1), keepdims=True)
    audio_bin_std = np.where(audio_bin_std == 0.0, 1.0, audio_bin_std)
    x_audio_train = (x_audio_train - audio_bin_mean) / audio_bin_std
    x_audio_val = (x_audio_val - audio_bin_mean) / audio_bin_std

    class_weights_dict = dict(
        enumerate(
            class_weight.compute_class_weight(
                class_weight="balanced",
                classes=np.unique(y_train_int),
                y=y_train_int,
            )
        )
    )

    print("\nTraining small IMU backbone...")
    imu_model = build_imu_model_small(num_classes)
    imu_model.summary()
    imu_model.fit(
        x_imu_train,
        y_train,
        validation_data=(x_imu_val, y_val),
        epochs=BACKBONE_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights_dict,
        callbacks=make_backbone_callbacks(),
        verbose=2,
    )
    _, imu_accuracy = imu_model.evaluate(x_imu_val, y_val, verbose=0)

    print("\nTraining small audio backbone...")
    audio_model = build_audio_model_small(x_audio_spec.shape[1:], num_classes)
    audio_model.summary()
    audio_model.fit(
        x_audio_train,
        y_train,
        validation_data=(x_audio_val, y_val),
        epochs=BACKBONE_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights_dict,
        callbacks=make_backbone_callbacks(),
        verbose=2,
    )
    _, audio_accuracy = audio_model.evaluate(x_audio_val, y_val, verbose=0)

    print("\n=== Validation Accuracy Summary ===")
    print(f"IMU backbone (small):   {imu_accuracy * 100:.1f}%")
    print(f"Audio backbone (small): {audio_accuracy * 100:.1f}%")

    y_val_int = np.argmax(y_val, axis=1)
    for name, model, x_val in (("IMU", imu_model, x_imu_val), ("Audio", audio_model, x_audio_val)):
        y_val_pred = np.argmax(model.predict(x_val, verbose=0), axis=1)
        print(f"\n--- {name} backbone (small) ---")
        print("Classification report:")
        print(classification_report(y_val_int, y_val_pred, target_names=label_encoder.classes_))
        print("Confusion matrix:")
        print(confusion_matrix(y_val_int, y_val_pred))

    EXPORT_DIR.mkdir(exist_ok=True)
    imu_model.save(EXPORT_DIR / "imu_backbone_small.keras")
    audio_model.save(EXPORT_DIR / "audio_backbone_small.keras")
    np.savez(
        EXPORT_DIR / "fusion_backbones_small_export_data.npz",
        imu_channel_mean=imu_channel_mean,
        imu_channel_std=imu_channel_std,
        audio_bin_mean=audio_bin_mean,
        audio_bin_std=audio_bin_std,
        classes=label_encoder.classes_,
        imu_representative_samples=x_imu_train[:REPRESENTATIVE_SAMPLE_COUNT],
        audio_representative_samples=x_audio_train[:REPRESENTATIVE_SAMPLE_COUNT],
    )
    print(f"\nSaved both backbones and export data to {EXPORT_DIR}")


if __name__ == "__main__":
    main()
