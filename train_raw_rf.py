from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from data_split import grouped_train_val_split, parse_window_timestamp


WINDOW_LENGTH = 200
DATA_ROOT = Path(__file__).resolve().parent / "data" / "windowed_data" / "right_hand_dominant" / "imu"
CHANNEL_COLUMNS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def safe_skew(values: np.ndarray) -> float:
    centered = values - np.mean(values)
    std = np.std(centered)
    if std == 0.0:
        return 0.0
    return float(np.mean(centered**3) / (std**3))


def safe_kurtosis(values: np.ndarray) -> float:
    centered = values - np.mean(values)
    std = np.std(centered)
    if std == 0.0:
        return -3.0
    return float(np.mean(centered**4) / (std**4) - 3.0)


def zero_crossing_rate(values: np.ndarray) -> float:
    signs = np.signbit(values)
    return float(np.mean(signs[:-1] != signs[1:])) if values.size > 1 else 0.0


def spectral_features(values: np.ndarray, sampling_rate: float = 80.0) -> dict[str, float]:
    centered = values - np.mean(values)
    spectrum = np.abs(np.fft.rfft(centered))
    power = spectrum**2
    power_sum = float(np.sum(power))

    frequencies = np.fft.rfftfreq(values.size, d=1.0 / sampling_rate)
    if power_sum == 0.0:
        dominant_frequency = 0.0
        spectral_centroid = 0.0
        spectral_bandwidth = 0.0
        spectral_entropy = 0.0
        spectral_rolloff = 0.0
    else:
        dominant_frequency = float(frequencies[int(np.argmax(power))])
        spectral_centroid = float(np.sum(frequencies * power) / power_sum)
        spectral_bandwidth = float(np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * power) / power_sum))
        normalized = power / power_sum
        spectral_entropy = float(-np.sum(normalized * np.log(normalized + 1e-12)))
        cumulative = np.cumsum(power)
        spectral_rolloff = float(frequencies[np.searchsorted(cumulative, 0.85 * power_sum)])

    return {
        "dominant_frequency": dominant_frequency,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_entropy": spectral_entropy,
        "spectral_rolloff": spectral_rolloff,
        "spectral_energy": power_sum,
    }


def extract_features(window: np.ndarray) -> np.ndarray:
    features: list[float] = []

    # Per-axis statistical and frequency features.
    for axis_index, axis_name in enumerate(CHANNEL_COLUMNS):
        values = window[:, axis_index]
        features.extend(
            [
                float(np.mean(values)),
                float(np.std(values)),
                float(np.min(values)),
                float(np.max(values)),
                float(np.median(values)),
                float(np.ptp(values)),
                float(np.mean(np.abs(values))),
                float(np.sqrt(np.mean(values**2))),
                float(np.percentile(values, 25)),
                float(np.percentile(values, 75)),
                float(np.percentile(values, 75) - np.percentile(values, 25)),
                safe_skew(values),
                safe_kurtosis(values),
                zero_crossing_rate(values),
            ]
        )

        freq = spectral_features(values)
        features.extend(
            [
                freq["dominant_frequency"],
                freq["spectral_centroid"],
                freq["spectral_bandwidth"],
                freq["spectral_entropy"],
                freq["spectral_rolloff"],
                freq["spectral_energy"],
            ]
        )

    # Cross-axis relationships are useful for assembly gestures.
    for first_axis, second_axis in combinations(range(window.shape[1]), 2):
        first_values = window[:, first_axis]
        second_values = window[:, second_axis]
        if np.std(first_values) == 0.0 or np.std(second_values) == 0.0:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(first_values, second_values)[0, 1])
        features.append(correlation)

    accel_magnitude = np.linalg.norm(window[:, 0:3], axis=1)
    gyro_magnitude = np.linalg.norm(window[:, 3:6], axis=1)
    total_magnitude = np.linalg.norm(window, axis=1)

    for magnitude_values in (accel_magnitude, gyro_magnitude, total_magnitude):
        features.extend(
            [
                float(np.mean(magnitude_values)),
                float(np.std(magnitude_values)),
                float(np.min(magnitude_values)),
                float(np.max(magnitude_values)),
                float(np.sqrt(np.mean(magnitude_values**2))),
                safe_skew(magnitude_values),
                safe_kurtosis(magnitude_values),
                zero_crossing_rate(magnitude_values),
            ]
        )

    return np.asarray(features, dtype=np.float32)


def resample_window(window: np.ndarray, target_length: int = WINDOW_LENGTH) -> np.ndarray:
    if window.shape[0] == target_length:
        return window.astype(np.float32)

    return resample(window, target_length, axis=0).astype(np.float32)


def load_windows(data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows = []
    labels = []
    timestamps = []

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
        timestamps.append(parse_window_timestamp(csv_path))

    if not windows:
        raise ValueError(f"No raw IMU CSV files found under {data_root}")

    return np.stack(windows), np.array(labels), np.array(timestamps)


def main() -> None:
    windows, labels, timestamps = load_windows(DATA_ROOT)

    feature_rows = np.vstack([extract_features(window) for window in windows])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # Grouped split: windows overlap 50%, so a plain random/stratified split can leak
    # overlapping samples from the same activity instance across train/val (see data_split.py).
    train_idx, val_idx = grouped_train_val_split(labels, timestamps, test_size=0.2, random_state=42)
    x_train, x_val = feature_rows[train_idx], feature_rows[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = RandomForestClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_val)
    accuracy = accuracy_score(y_val, predictions)

    print(f"Classes: {list(label_encoder.classes_)}")
    print(f"Feature count: {feature_rows.shape[1]}")
    print(f"Validation Accuracy: {accuracy * 100:.1f}%")
    print("\nClassification report:")
    print(classification_report(y_val, predictions, target_names=label_encoder.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, predictions))


if __name__ == "__main__":
    main()