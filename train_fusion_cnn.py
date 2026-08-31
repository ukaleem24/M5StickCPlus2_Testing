import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from data_split import grouped_train_val_split, parse_window_timestamp

import train_audio_cnn as audio_pipeline
import train_raw_cnn as imu_pipeline

build_imu_model = imu_pipeline.build_model
build_audio_model = audio_pipeline.build_model

BACKBONE_EPOCHS = 150
FUSION_EPOCHS = 80
BATCH_SIZE = 32
FUSION_EARLY_STOPPING_PATIENCE = 15
FUSION_REDUCE_LR_PATIENCE = 6


def index_files_by_pair_key(root, prefix):
    """Map (label, timestamp, index) -> path, derived from `<prefix>_<label>_<ts>_<idx>.ext`."""
    indexed = {}
    for path in sorted(root.rglob(f"{prefix}_*")):
        if not path.is_file():
            continue
        label = path.parent.name
        timestamp, index = path.stem.split("_")[-2:]
        indexed[(label, timestamp, index)] = path
    return indexed


def discover_paired_windows(imu_root, audio_root):
    imu_files = index_files_by_pair_key(imu_root, "imu")
    audio_files = index_files_by_pair_key(audio_root, "audio")

    common_keys = sorted(set(imu_files) & set(audio_files))
    if not common_keys:
        raise ValueError(f"No matching IMU/audio window pairs found under {imu_root} and {audio_root}")

    imu_paths = [imu_files[key] for key in common_keys]
    audio_paths = [audio_files[key] for key in common_keys]
    labels = np.array([key[0] for key in common_keys])
    return imu_paths, audio_paths, labels


def load_imu_window(csv_path):
    df = pd.read_csv(csv_path)
    missing_columns = [c for c in imu_pipeline.CHANNEL_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns {missing_columns} in {csv_path}")

    window = df[imu_pipeline.CHANNEL_COLUMNS].to_numpy(dtype=np.float32)
    window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
    return imu_pipeline.resample_window(window, imu_pipeline.WINDOW_LENGTH)


def load_audio_waveform(wav_path):
    waveform, sample_rate = sf.read(wav_path, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate != audio_pipeline.SAMPLE_RATE:
        raise ValueError(f"Unexpected sample rate {sample_rate} in {wav_path}")

    return audio_pipeline.fit_waveform(waveform, audio_pipeline.TARGET_SAMPLES)


def make_backbone_callbacks():
    # No ReduceLROnPlateau and an effectively unreachable patience: both backbones train
    # at a constant 1e-3 for the full epoch budget (matches what worked empirically for
    # the noisy audio validation curve), while restore_best_weights still grabs whichever
    # epoch had the best val_accuracy instead of leaving it up to wherever training ends.
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=BACKBONE_EPOCHS,
            restore_best_weights=True,
        ),
    ]


def make_fusion_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=FUSION_EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=FUSION_REDUCE_LR_PATIENCE,
            min_lr=1e-5,
        ),
    ]


def get_embedding_model(model: tf.keras.Model) -> tf.keras.Model:
    """Wrap a trained backbone so it outputs its 64-d penultimate features instead of class scores."""
    dense_index = next(
        index
        for index, layer in enumerate(model.layers)
        if isinstance(layer, tf.keras.layers.Dense) and layer.units == 64
    )
    # Reuse the trained layer objects directly (rather than model.input/.output graph
    # introspection) since a Sequential built from Input(...) does not retain a usable
    # functional graph handle after fit() in this Keras version.
    embedder = tf.keras.Sequential(model.layers[: dense_index + 1])
    embedder.trainable = False
    return embedder


def report_classification(name, y_true_int, y_pred_int, target_names):
    print(f"\n--- {name} ---")
    print("Classification report:")
    print(classification_report(y_true_int, y_pred_int, target_names=target_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_true_int, y_pred_int))


def main():
    imu_paths, audio_paths, labels = discover_paired_windows(imu_pipeline.DATA_ROOT, audio_pipeline.DATA_ROOT)
    print(f"Found {len(labels)} paired IMU/audio windows across {len(set(labels))} classes.")

    x_imu_raw = np.stack([load_imu_window(path) for path in imu_paths])
    x_audio_raw = np.stack([load_audio_waveform(path) for path in audio_paths])
    x_audio_spec = audio_pipeline.compute_log_mel_spectrograms(x_audio_raw)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    y_categorical = tf.keras.utils.to_categorical(y_encoded)
    num_classes = len(label_encoder.classes_)

    # Single grouped split shared by both modalities so embeddings line up sample-for-sample.
    # Windows overlap 50% (src/5. windowing_script.py), so a plain random/stratified split
    # can put overlapping windows from the same activity instance on both sides of the
    # boundary and leak information; grouped_train_val_split keeps whole instances intact.
    timestamps = np.array([parse_window_timestamp(p) for p in imu_paths])
    train_idx, val_idx = grouped_train_val_split(labels, timestamps, test_size=0.2, random_state=42)

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

    print("\nTraining IMU backbone...")
    imu_model = build_imu_model(num_classes)
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

    print("\nTraining audio backbone...")
    audio_model = build_audio_model(x_audio_spec.shape[1:], num_classes)
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

    # Late fusion: freeze both backbones and train a small head on their concatenated embeddings.
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

    print("\nTraining fusion head...")
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

    print("\n=== Validation Accuracy Summary ===")
    print(f"IMU-only:   {imu_accuracy * 100:.1f}%")
    print(f"Audio-only: {audio_accuracy * 100:.1f}%")
    print(f"Fused:      {fusion_accuracy * 100:.1f}%")

    target_names = label_encoder.classes_
    y_val_int = np.argmax(y_val, axis=1)
    imu_val_pred = np.argmax(imu_model.predict(x_imu_val, verbose=0), axis=1)
    audio_val_pred = np.argmax(audio_model.predict(x_audio_val, verbose=0), axis=1)
    fusion_val_pred = np.argmax(fusion_model.predict(fused_val, verbose=0), axis=1)

    report_classification("IMU-only", y_val_int, imu_val_pred, target_names)
    report_classification("Audio-only", y_val_int, audio_val_pred, target_names)
    report_classification("Fused", y_val_int, fusion_val_pred, target_names)


if __name__ == "__main__":
    main()
