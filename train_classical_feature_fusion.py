import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import train_fusion_cnn as fusion_pipeline

audio_pipeline = fusion_pipeline.audio_pipeline
imu_pipeline = fusion_pipeline.imu_pipeline
discover_paired_windows = fusion_pipeline.discover_paired_windows
load_imu_window = fusion_pipeline.load_imu_window
load_audio_waveform = fusion_pipeline.load_audio_waveform

N_ESTIMATORS = 300
RANDOM_STATE = 42


def extract_imu_features(window: np.ndarray) -> np.ndarray:
    """window: (T, 6). Classic time-domain statistical features per axis -- the
    standard hand-crafted feature set for accelerometer/gyroscope-based HAR,
    predating deep learning approaches to the task."""
    mean = window.mean(axis=0)
    std = window.std(axis=0)
    minimum = window.min(axis=0)
    maximum = window.max(axis=0)
    rms = np.sqrt(np.mean(np.square(window), axis=0))
    zero_crossing_rate = np.mean(np.diff(np.sign(window), axis=0) != 0, axis=0)
    return np.concatenate([mean, std, minimum, maximum, rms, zero_crossing_rate])


def extract_audio_features(log_mel_spectrogram: np.ndarray) -> np.ndarray:
    """log_mel_spectrogram: (T, 40). Mean/std per mel bin over time -- the classic
    spectral summary-statistics baseline used for audio classification before
    end-to-end CNNs became standard."""
    mean = log_mel_spectrogram.mean(axis=0)
    std = log_mel_spectrogram.std(axis=0)
    return np.concatenate([mean, std])


def main():
    imu_paths, audio_paths, labels = discover_paired_windows(imu_pipeline.DATA_ROOT, audio_pipeline.DATA_ROOT)
    print(f"Found {len(labels)} paired IMU/audio windows across {len(set(labels))} classes.")

    x_imu_windows = [load_imu_window(path) for path in imu_paths]
    x_audio_waveforms = np.stack([load_audio_waveform(path) for path in audio_paths])
    x_audio_specs = audio_pipeline.compute_log_mel_spectrograms(x_audio_waveforms)[..., 0]  # (N, T, 40)

    imu_features = np.stack([extract_imu_features(window) for window in x_imu_windows])
    audio_features = np.stack([extract_audio_features(spectrogram) for spectrogram in x_audio_specs])

    # Feature-level fusion: concatenate hand-crafted features from both modalities
    # into one flat vector, before any classifier ever sees the data.
    x_features = np.concatenate([imu_features, audio_features], axis=1)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        stratify=y_encoded,
        random_state=RANDOM_STATE,
    )

    x_train, x_val = x_features[train_idx], x_features[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    # Random Forest needs no feature scaling/normalization and handles the imbalanced
    # classes via class_weight, unlike the CNN pipelines' per-channel normalization step.
    classifier = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    classifier.fit(x_train, y_train)

    val_predictions = classifier.predict(x_val)
    accuracy = accuracy_score(y_val, val_predictions)

    print(f"\nIMU feature count:   {imu_features.shape[1]}")
    print(f"Audio feature count: {audio_features.shape[1]}")
    print(
        "Classical Baseline (Random Forest, statistical feature-level fusion) "
        f"Validation Accuracy: {accuracy * 100:.1f}%"
    )

    print("\nClassification report:")
    print(classification_report(y_val, val_predictions, target_names=label_encoder.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, val_predictions))


if __name__ == "__main__":
    main()
