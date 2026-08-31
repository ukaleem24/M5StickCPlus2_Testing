import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from data_split import grouped_train_val_split, parse_window_timestamp

import train_fusion_backbones_small as backbones_pipeline

fusion_pipeline = backbones_pipeline.fusion_pipeline
audio_pipeline = backbones_pipeline.audio_pipeline
imu_pipeline = backbones_pipeline.imu_pipeline
discover_paired_windows = backbones_pipeline.discover_paired_windows
load_imu_window = backbones_pipeline.load_imu_window
load_audio_waveform = backbones_pipeline.load_audio_waveform
make_fusion_callbacks = fusion_pipeline.make_fusion_callbacks
get_embedding_model = fusion_pipeline.get_embedding_model

EXPORT_DIR = backbones_pipeline.EXPORT_DIR
FUSION_EPOCHS = fusion_pipeline.FUSION_EPOCHS
BATCH_SIZE = backbones_pipeline.BATCH_SIZE
REPRESENTATIVE_SAMPLE_COUNT = backbones_pipeline.REPRESENTATIVE_SAMPLE_COUNT


def main():
    export_data = np.load(EXPORT_DIR / "fusion_backbones_small_export_data.npz", allow_pickle=True)
    imu_channel_mean = export_data["imu_channel_mean"]
    imu_channel_std = export_data["imu_channel_std"]
    audio_bin_mean = export_data["audio_bin_mean"]
    audio_bin_std = export_data["audio_bin_std"]
    classes = export_data["classes"]
    num_classes = len(classes)

    imu_model = tf.keras.models.load_model(EXPORT_DIR / "imu_backbone_small.keras")
    audio_model = tf.keras.models.load_model(EXPORT_DIR / "audio_backbone_small.keras")

    # Re-derive the exact same split the backbones were trained on (same data, same
    # random_state) -- the backbones' own saved stats/representative samples don't include
    # the full train/val arrays needed to extract embeddings here.
    imu_paths, audio_paths, labels = discover_paired_windows(imu_pipeline.DATA_ROOT, audio_pipeline.DATA_ROOT)
    x_imu_raw = np.stack([load_imu_window(p) for p in imu_paths])
    x_audio_raw = np.stack([load_audio_waveform(p) for p in audio_paths])
    x_audio_spec = audio_pipeline.compute_log_mel_spectrograms(x_audio_raw)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.asarray(classes)
    y_encoded = label_encoder.transform(labels)
    y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)

    timestamps = np.array([parse_window_timestamp(p) for p in imu_paths])
    train_idx, val_idx = grouped_train_val_split(labels, timestamps, test_size=0.2, random_state=42)

    x_imu_train = (x_imu_raw[train_idx] - imu_channel_mean) / imu_channel_std
    x_imu_val = (x_imu_raw[val_idx] - imu_channel_mean) / imu_channel_std
    x_audio_train = (x_audio_spec[train_idx] - audio_bin_mean) / audio_bin_std
    x_audio_val = (x_audio_spec[val_idx] - audio_bin_mean) / audio_bin_std
    y_train, y_val = y_categorical[train_idx], y_categorical[val_idx]
    y_train_int = y_encoded[train_idx]

    class_weights_dict = dict(
        enumerate(
            class_weight.compute_class_weight(
                class_weight="balanced", classes=np.unique(y_train_int), y=y_train_int
            )
        )
    )

    imu_embedder = get_embedding_model(imu_model)
    audio_embedder = get_embedding_model(audio_model)

    fused_train = np.concatenate(
        [imu_embedder.predict(x_imu_train, verbose=0), audio_embedder.predict(x_audio_train, verbose=0)],
        axis=1,
    )
    fused_val = np.concatenate(
        [imu_embedder.predict(x_imu_val, verbose=0), audio_embedder.predict(x_audio_val, verbose=0)],
        axis=1,
    )

    print("\nTraining fusion head (small backbones)...")
    fusion_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(fused_train.shape[1],)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    fusion_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    fusion_model.fit(
        fused_train,
        y_train,
        validation_data=(fused_val, y_val),
        epochs=FUSION_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights_dict,
        callbacks=make_fusion_callbacks(),
        verbose=2,
    )
    _, fusion_accuracy = fusion_model.evaluate(fused_val, y_val, verbose=0)
    print(f"\nEmbedding Fusion (small) Validation Accuracy: {fusion_accuracy * 100:.1f}%")

    y_val_int = np.argmax(y_val, axis=1)
    y_val_pred = np.argmax(fusion_model.predict(fused_val, verbose=0), axis=1)
    print("\nClassification report:")
    print(classification_report(y_val_int, y_val_pred, target_names=classes))
    print("Confusion matrix:")
    print(confusion_matrix(y_val_int, y_val_pred))

    fusion_model.save(EXPORT_DIR / "fusion_head_small.keras")
    np.savez(
        EXPORT_DIR / "embedding_fusion_small_export_data.npz",
        fused_representative_samples=fused_train[:REPRESENTATIVE_SAMPLE_COUNT],
    )
    print(f"\nSaved fusion head and export data to {EXPORT_DIR}")


if __name__ == "__main__":
    main()
