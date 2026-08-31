import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from data_split import grouped_train_val_split, parse_window_timestamp

import train_decision_fusion_cnn as decision_pipeline
import train_fusion_backbones_small as backbones_pipeline

audio_pipeline = backbones_pipeline.audio_pipeline
imu_pipeline = backbones_pipeline.imu_pipeline
discover_paired_windows = backbones_pipeline.discover_paired_windows
load_imu_window = backbones_pipeline.load_imu_window
load_audio_waveform = backbones_pipeline.load_audio_waveform
weighted_fusion_probs = decision_pipeline.weighted_fusion_probs
search_best_imu_weight = decision_pipeline.search_best_imu_weight

EXPORT_DIR = backbones_pipeline.EXPORT_DIR


def main():
    export_data = np.load(EXPORT_DIR / "fusion_backbones_small_export_data.npz", allow_pickle=True)
    imu_channel_mean = export_data["imu_channel_mean"]
    imu_channel_std = export_data["imu_channel_std"]
    audio_bin_mean = export_data["audio_bin_mean"]
    audio_bin_std = export_data["audio_bin_std"]
    classes = export_data["classes"]

    # Reuses the SAME backbones train_embedding_fusion_small.py loads -- both deployments
    # are built on one directly-comparable pair, not independently retrained copies.
    imu_model = tf.keras.models.load_model(EXPORT_DIR / "imu_backbone_small.keras")
    audio_model = tf.keras.models.load_model(EXPORT_DIR / "audio_backbone_small.keras")

    imu_paths, audio_paths, labels = discover_paired_windows(imu_pipeline.DATA_ROOT, audio_pipeline.DATA_ROOT)
    x_imu_raw = np.stack([load_imu_window(p) for p in imu_paths])
    x_audio_raw = np.stack([load_audio_waveform(p) for p in audio_paths])
    x_audio_spec = audio_pipeline.compute_log_mel_spectrograms(x_audio_raw)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.asarray(classes)
    y_encoded = label_encoder.transform(labels)

    timestamps = np.array([parse_window_timestamp(p) for p in imu_paths])
    _, val_idx = grouped_train_val_split(labels, timestamps, test_size=0.2, random_state=42)

    x_imu_val = (x_imu_raw[val_idx] - imu_channel_mean) / imu_channel_std
    x_audio_val = (x_audio_spec[val_idx] - audio_bin_mean) / audio_bin_std
    y_val_int = y_encoded[val_idx]

    imu_val_probs = imu_model.predict(x_imu_val, verbose=0)
    audio_val_probs = audio_model.predict(x_audio_val, verbose=0)

    best_weight, best_accuracy, weights, accuracies = search_best_imu_weight(
        imu_val_probs, audio_val_probs, y_val_int
    )

    print("\n=== Decision-Level Fusion (small backbones) Weight Search ===")
    for weight, accuracy in zip(weights, accuracies):
        marker = "  <-- best" if weight == best_weight else ""
        print(f"IMU weight {weight:.2f} / audio weight {1 - weight:.2f}: {accuracy * 100:.1f}%{marker}")

    print(
        f"\nDecision-level fused (small) Validation Accuracy: {best_accuracy * 100:.1f}% "
        f"(IMU weight={best_weight:.2f})"
    )

    fused_val_pred = np.argmax(weighted_fusion_probs(imu_val_probs, audio_val_probs, best_weight), axis=1)
    print("\nClassification report:")
    print(classification_report(y_val_int, fused_val_pred, target_names=classes))
    print("Confusion matrix:")
    print(confusion_matrix(y_val_int, fused_val_pred))

    np.savez(EXPORT_DIR / "decision_fusion_small_export_data.npz", best_imu_weight=best_weight)
    print(f"\nSaved decision fusion weight to {EXPORT_DIR}")


if __name__ == "__main__":
    main()
