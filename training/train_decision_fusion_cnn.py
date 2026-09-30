import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from data_split import grouped_train_val_split, parse_window_timestamp

import train_fusion_cnn as embed_fusion_pipeline

audio_pipeline = embed_fusion_pipeline.audio_pipeline
imu_pipeline = embed_fusion_pipeline.imu_pipeline
build_imu_model = embed_fusion_pipeline.build_imu_model
build_audio_model = embed_fusion_pipeline.build_audio_model
discover_paired_windows = embed_fusion_pipeline.discover_paired_windows
load_imu_window = embed_fusion_pipeline.load_imu_window
load_audio_waveform = embed_fusion_pipeline.load_audio_waveform
make_backbone_callbacks = embed_fusion_pipeline.make_backbone_callbacks

BACKBONE_EPOCHS = embed_fusion_pipeline.BACKBONE_EPOCHS
BATCH_SIZE = embed_fusion_pipeline.BATCH_SIZE
WEIGHT_SEARCH_STEP = 0.05


def prepare_paired_dataset():
    """Load paired IMU/audio windows and produce a single stratified train/val split
    shared by both modalities, so predictions line up sample-for-sample."""
    imu_paths, audio_paths, labels = discover_paired_windows(imu_pipeline.DATA_ROOT, audio_pipeline.DATA_ROOT)
    print(f"Found {len(labels)} paired IMU/audio windows across {len(set(labels))} classes.")

    x_imu_raw = np.stack([load_imu_window(path) for path in imu_paths])
    x_audio_raw = np.stack([load_audio_waveform(path) for path in audio_paths])
    x_audio_spec = audio_pipeline.compute_log_mel_spectrograms(x_audio_raw)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    y_categorical = tf.keras.utils.to_categorical(y_encoded)
    num_classes = len(label_encoder.classes_)

    timestamps = np.array([parse_window_timestamp(p) for p in imu_paths])
    train_idx, val_idx = grouped_train_val_split(labels, timestamps, test_size=0.2, random_state=42)

    x_imu_train, x_imu_val = x_imu_raw[train_idx], x_imu_raw[val_idx]
    x_audio_train, x_audio_val = x_audio_spec[train_idx], x_audio_spec[val_idx]
    y_train, y_val = y_categorical[train_idx], y_categorical[val_idx]
    y_train_int, y_val_int = y_encoded[train_idx], y_encoded[val_idx]

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

    return {
        "x_imu_train": x_imu_train,
        "x_imu_val": x_imu_val,
        "x_audio_train": x_audio_train,
        "x_audio_val": x_audio_val,
        "y_train": y_train,
        "y_val": y_val,
        "y_val_int": y_val_int,
        "class_weights_dict": class_weights_dict,
        "num_classes": num_classes,
        "label_encoder": label_encoder,
    }


def train_imu_backbone(dataset):
    print("\nTraining IMU backbone...")
    model = build_imu_model(dataset["num_classes"])
    model.fit(
        dataset["x_imu_train"],
        dataset["y_train"],
        validation_data=(dataset["x_imu_val"], dataset["y_val"]),
        epochs=BACKBONE_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=dataset["class_weights_dict"],
        callbacks=make_backbone_callbacks(),
        verbose=2,
    )
    return model


def train_audio_backbone(dataset):
    print("\nTraining audio backbone...")
    model = build_audio_model(dataset["x_audio_train"].shape[1:], dataset["num_classes"])
    model.fit(
        dataset["x_audio_train"],
        dataset["y_train"],
        validation_data=(dataset["x_audio_val"], dataset["y_val"]),
        epochs=BACKBONE_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=dataset["class_weights_dict"],
        callbacks=make_backbone_callbacks(),
        verbose=2,
    )
    return model


def weighted_fusion_probs(imu_probs: np.ndarray, audio_probs: np.ndarray, imu_weight: float) -> np.ndarray:
    """Weighted-sum decision rule: combine each pipeline's own class-probability
    decision rather than any shared internal representation."""
    return imu_weight * imu_probs + (1.0 - imu_weight) * audio_probs


def accuracy_at_weight(imu_probs, audio_probs, y_true_int, imu_weight) -> float:
    fused_probs = weighted_fusion_probs(imu_probs, audio_probs, imu_weight)
    predictions = np.argmax(fused_probs, axis=1)
    return float(np.mean(predictions == y_true_int))


def search_best_imu_weight(imu_val_probs, audio_val_probs, y_val_int):
    # A single scalar mixing weight, grid-searched on the validation split. Far fewer
    # degrees of freedom than training a fusion head on concatenated embeddings, so it
    # is much less likely to overfit the modest paired-window dataset.
    weights = np.arange(0.0, 1.0 + 1e-9, WEIGHT_SEARCH_STEP)
    accuracies = [accuracy_at_weight(imu_val_probs, audio_val_probs, y_val_int, w) for w in weights]
    best_index = int(np.argmax(accuracies))
    return weights[best_index], accuracies[best_index], weights, accuracies


def report_classification(name, y_true_int, y_pred_int, target_names):
    print(f"\n--- {name} ---")
    print("Classification report:")
    print(classification_report(y_true_int, y_pred_int, target_names=target_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_true_int, y_pred_int))


def main():
    dataset = prepare_paired_dataset()

    imu_model = train_imu_backbone(dataset)
    _, imu_accuracy = imu_model.evaluate(dataset["x_imu_val"], dataset["y_val"], verbose=0)

    audio_model = train_audio_backbone(dataset)
    _, audio_accuracy = audio_model.evaluate(dataset["x_audio_val"], dataset["y_val"], verbose=0)

    imu_val_probs = imu_model.predict(dataset["x_imu_val"], verbose=0)
    audio_val_probs = audio_model.predict(dataset["x_audio_val"], verbose=0)

    best_weight, best_accuracy, weights, accuracies = search_best_imu_weight(
        imu_val_probs, audio_val_probs, dataset["y_val_int"]
    )

    print("\n=== Decision-Level Fusion Weight Search ===")
    for weight, accuracy in zip(weights, accuracies):
        marker = "  <-- best" if weight == best_weight else ""
        print(f"IMU weight {weight:.2f} / audio weight {1 - weight:.2f}: {accuracy * 100:.1f}%{marker}")

    print("\n=== Validation Accuracy Summary ===")
    print(f"IMU-only:             {imu_accuracy * 100:.1f}%")
    print(f"Audio-only:           {audio_accuracy * 100:.1f}%")
    print(f"Decision-level fused: {best_accuracy * 100:.1f}%  (IMU weight={best_weight:.2f})")

    target_names = dataset["label_encoder"].classes_
    y_val_int = dataset["y_val_int"]
    imu_val_pred = np.argmax(imu_val_probs, axis=1)
    audio_val_pred = np.argmax(audio_val_probs, axis=1)
    fused_val_pred = np.argmax(weighted_fusion_probs(imu_val_probs, audio_val_probs, best_weight), axis=1)

    report_classification("IMU-only", y_val_int, imu_val_pred, target_names)
    report_classification("Audio-only", y_val_int, audio_val_pred, target_names)
    report_classification("Decision-level fused", y_val_int, fused_val_pred, target_names)


if __name__ == "__main__":
    main()
