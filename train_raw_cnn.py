import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight


# Raw-window HAR training configuration
WINDOW_LENGTH = 200
NUM_CHANNELS = 6
DATA_ROOT = Path(__file__).resolve().parent / "data" / "windowed_data" / "right_hand_dominant" / "imu"

CHANNEL_COLUMNS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
CLASS_NAMES = None


def resample_window(window: np.ndarray, target_length: int) -> np.ndarray:
    """Resample a single window to a fixed number of timesteps."""
    if window.shape[0] == target_length:
        return window.astype(np.float32)

    source_index = np.linspace(0.0, 1.0, num=window.shape[0], dtype=np.float32)
    target_index = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    resampled = np.empty((target_length, window.shape[1]), dtype=np.float32)

    for channel in range(window.shape[1]):
        resampled[:, channel] = np.interp(target_index, source_index, window[:, channel])

    return resampled


def load_raw_windows(data_root: Path):
    windows = []
    labels = []

    for csv_path in sorted(data_root.rglob("*.csv")):
        if "imu" not in csv_path.parts:
            continue

        label = csv_path.parent.name
        df = pd.read_csv(csv_path)

        missing_columns = [column for column in CHANNEL_COLUMNS if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns {missing_columns} in {csv_path}")

        window = df[CHANNEL_COLUMNS].to_numpy(dtype=np.float32)
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
        window = resample_window(window, WINDOW_LENGTH)

        windows.append(window)
        labels.append(label)

    if not windows:
        raise ValueError(f"No raw IMU CSV files found under {data_root}")

    return np.stack(windows), np.array(labels)


def build_model(num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(WINDOW_LENGTH, NUM_CHANNELS)),
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
    x_raw, y_raw = load_raw_windows(DATA_ROOT)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    y_categorical = tf.keras.utils.to_categorical(y_encoded)

    x_train, x_val, y_train, y_val = train_test_split(
        x_raw,
        y_categorical,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42,
    )

    # Normalize per channel using only the training split.
    channel_mean = np.mean(x_train, axis=(0, 1), keepdims=True)
    channel_std = np.std(x_train, axis=(0, 1), keepdims=True)
    channel_std = np.where(channel_std == 0.0, 1.0, channel_std)

    x_train = (x_train - channel_mean) / channel_std
    x_val = (x_val - channel_mean) / channel_std

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

    model = build_model(len(label_encoder.classes_))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=80,
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