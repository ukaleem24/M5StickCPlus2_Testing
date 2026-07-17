from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight


WINDOW_LENGTH = 200
NUM_CHANNELS = 6
DATA_ROOT = Path(__file__).resolve().parent / "data" / "windowed_data" / "right_hand_dominant" / "imu"
CHANNEL_COLUMNS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def resample_window(window: np.ndarray, target_length: int = WINDOW_LENGTH) -> np.ndarray:
    """Resample a raw IMU window to a fixed number of timesteps."""
    if window.shape[0] == target_length:
        return window.astype(np.float32)
    return resample(window, target_length, axis=0).astype(np.float32)


def load_windows(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    windows = []
    labels = []

    for csv_path in sorted(data_root.rglob("*.csv")):
        if "imu" not in csv_path.parts:
            continue

        df = pd.read_csv(csv_path)

        missing_columns = [column for column in CHANNEL_COLUMNS if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns {missing_columns} in {csv_path}")

        window = df[CHANNEL_COLUMNS].to_numpy(dtype=np.float32)
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
        window = resample_window(window)

        windows.append(window)
        labels.append(csv_path.parent.name)

    if not windows:
        raise ValueError(f"No raw IMU CSV files found under {data_root}")

    return np.stack(windows), np.array(labels)


def build_model(num_classes: int) -> tf.keras.Model:
    """A compact GRU classifier for raw HAR windows."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(WINDOW_LENGTH, NUM_CHANNELS)),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                    64,
                    return_sequences=True,
                    dropout=0.15,
                    recurrent_dropout=0.10,
                )
            ),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(
                    32,
                    return_sequences=False,
                    dropout=0.15,
                    recurrent_dropout=0.10,
                )
            ),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    x_raw, y_raw = load_windows(DATA_ROOT)

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

    # Per-channel standardization using only the training split.
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

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=80,
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=2,
    )

    loss, accuracy = model.evaluate(validation_dataset, verbose=0)
    print(f"Classes: {list(label_encoder.classes_)}")
    print(f"Validation Accuracy: {accuracy * 100:.1f}%")


if __name__ == "__main__":
    main()