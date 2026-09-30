"""Ensembling experiment on top of the regularized/augmented small backbones.

Trains N_SEEDS independent copies of the IMU and audio backbones (same architecture,
same augmented training data, same grouped train/val split -- only the model's weight
initialization and batch shuffling vary), then compares:
  - the single-model accuracy averaged across seeds (how noisy is one run), against
  - the accuracy of averaging all seeds' predicted probabilities together (an ensemble).

Does not overwrite the canonical models/*.keras files used for firmware export --
this is a research/accuracy-ceiling experiment, not a change to the deployed model.
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

import train_decision_fusion_cnn as decision_pipeline
import train_fusion_backbones_small as backbones_pipeline
from augmentation import augment_imu_window, spec_augment
from data_split import grouped_train_val_split, parse_window_timestamp

audio_pipeline = backbones_pipeline.audio_pipeline
imu_pipeline = backbones_pipeline.imu_pipeline
discover_paired_windows = backbones_pipeline.discover_paired_windows
load_imu_window = backbones_pipeline.load_imu_window
load_audio_waveform = backbones_pipeline.load_audio_waveform
build_imu_model_small = backbones_pipeline.build_imu_model_small
build_audio_model_small = backbones_pipeline.build_audio_model_small
make_backbone_callbacks = backbones_pipeline.make_backbone_callbacks
NUM_AUGMENTED_COPIES = backbones_pipeline.NUM_AUGMENTED_COPIES
AUGMENTATION_SEED = backbones_pipeline.AUGMENTATION_SEED

BACKBONE_EPOCHS = backbones_pipeline.BACKBONE_EPOCHS
BATCH_SIZE = backbones_pipeline.BATCH_SIZE

weighted_fusion_probs = decision_pipeline.weighted_fusion_probs
search_best_imu_weight = decision_pipeline.search_best_imu_weight

SEEDS = [0, 1, 2, 3, 4]


def prepare_data():
    imu_paths, audio_paths, labels = discover_paired_windows(imu_pipeline.DATA_ROOT, audio_pipeline.DATA_ROOT)
    print(f"Found {len(labels)} paired IMU/audio windows across {len(set(labels))} classes.")

    x_imu_raw = np.stack([load_imu_window(p) for p in imu_paths])
    x_audio_raw = np.stack([load_audio_waveform(p) for p in audio_paths])
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
    y_train_int = y_encoded[train_idx]
    y_val_int = y_encoded[val_idx]

    imu_channel_mean = np.mean(x_imu_train, axis=(0, 1), keepdims=True)
    imu_channel_std = np.std(x_imu_train, axis=(0, 1), keepdims=True)
    imu_channel_std = np.where(imu_channel_std == 0.0, 1.0, imu_channel_std)

    audio_bin_mean = np.mean(x_audio_train, axis=(0, 1), keepdims=True)
    audio_bin_std = np.std(x_audio_train, axis=(0, 1), keepdims=True)
    audio_bin_std = np.where(audio_bin_std == 0.0, 1.0, audio_bin_std)

    aug_rng = np.random.RandomState(AUGMENTATION_SEED)
    x_imu_aug = np.stack(
        [augment_imu_window(w, aug_rng) for _ in range(NUM_AUGMENTED_COPIES) for w in x_imu_train]
    )
    x_audio_aug = np.stack(
        [spec_augment(s, aug_rng) for _ in range(NUM_AUGMENTED_COPIES) for s in x_audio_train]
    )
    y_train_aug = np.tile(y_train, (NUM_AUGMENTED_COPIES, 1))
    y_train_int_aug = np.tile(y_train_int, NUM_AUGMENTED_COPIES)

    x_imu_train = np.concatenate([x_imu_train, x_imu_aug], axis=0)
    x_audio_train = np.concatenate([x_audio_train, x_audio_aug], axis=0)
    y_train = np.concatenate([y_train, y_train_aug], axis=0)
    y_train_int = np.concatenate([y_train_int, y_train_int_aug], axis=0)

    x_imu_train = (x_imu_train - imu_channel_mean) / imu_channel_std
    x_imu_val = (x_imu_val - imu_channel_mean) / imu_channel_std
    x_audio_train = (x_audio_train - audio_bin_mean) / audio_bin_std
    x_audio_val = (x_audio_val - audio_bin_mean) / audio_bin_std

    class_weights_dict = dict(
        enumerate(
            class_weight.compute_class_weight(
                class_weight="balanced", classes=np.unique(y_train_int), y=y_train_int
            )
        )
    )

    return dict(
        x_imu_train=x_imu_train,
        x_imu_val=x_imu_val,
        x_audio_train=x_audio_train,
        x_audio_val=x_audio_val,
        y_train=y_train,
        y_val=y_val,
        y_val_int=y_val_int,
        class_weights_dict=class_weights_dict,
        num_classes=num_classes,
        label_encoder=label_encoder,
    )


def main():
    data = prepare_data()
    y_val_int = data["y_val_int"]

    imu_val_probs_per_seed = []
    audio_val_probs_per_seed = []
    imu_acc_per_seed = []
    audio_acc_per_seed = []

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        tf.keras.utils.set_random_seed(seed)

        imu_model = build_imu_model_small(data["num_classes"])
        imu_model.fit(
            data["x_imu_train"],
            data["y_train"],
            validation_data=(data["x_imu_val"], data["y_val"]),
            epochs=BACKBONE_EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=data["class_weights_dict"],
            callbacks=make_backbone_callbacks(),
            verbose=0,
        )
        imu_probs = imu_model.predict(data["x_imu_val"], verbose=0)
        imu_acc = float(np.mean(np.argmax(imu_probs, axis=1) == y_val_int))
        imu_val_probs_per_seed.append(imu_probs)
        imu_acc_per_seed.append(imu_acc)
        print(f"IMU seed {seed} val accuracy:   {imu_acc * 100:.1f}%")

        audio_model = build_audio_model_small(data["x_audio_train"].shape[1:], data["num_classes"])
        audio_model.fit(
            data["x_audio_train"],
            data["y_train"],
            validation_data=(data["x_audio_val"], data["y_val"]),
            epochs=BACKBONE_EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=data["class_weights_dict"],
            callbacks=make_backbone_callbacks(),
            verbose=0,
        )
        audio_probs = audio_model.predict(data["x_audio_val"], verbose=0)
        audio_acc = float(np.mean(np.argmax(audio_probs, axis=1) == y_val_int))
        audio_val_probs_per_seed.append(audio_probs)
        audio_acc_per_seed.append(audio_acc)
        print(f"Audio seed {seed} val accuracy: {audio_acc * 100:.1f}%")

    imu_ensemble_probs = np.mean(imu_val_probs_per_seed, axis=0)
    audio_ensemble_probs = np.mean(audio_val_probs_per_seed, axis=0)
    imu_ensemble_acc = float(np.mean(np.argmax(imu_ensemble_probs, axis=1) == y_val_int))
    audio_ensemble_acc = float(np.mean(np.argmax(audio_ensemble_probs, axis=1) == y_val_int))

    best_weight, best_fused_acc, weights, accuracies = search_best_imu_weight(
        imu_ensemble_probs, audio_ensemble_probs, y_val_int
    )

    print("\n=== Per-seed accuracy ===")
    print(f"IMU seeds:   {[f'{a * 100:.1f}%' for a in imu_acc_per_seed]}")
    print(f"  mean={np.mean(imu_acc_per_seed) * 100:.1f}%  std={np.std(imu_acc_per_seed) * 100:.1f}%")
    print(f"Audio seeds: {[f'{a * 100:.1f}%' for a in audio_acc_per_seed]}")
    print(f"  mean={np.mean(audio_acc_per_seed) * 100:.1f}%  std={np.std(audio_acc_per_seed) * 100:.1f}%")

    print("\n=== Ensemble (average softmax across seeds) ===")
    print(f"IMU ensemble accuracy:            {imu_ensemble_acc * 100:.1f}%")
    print(f"Audio ensemble accuracy:          {audio_ensemble_acc * 100:.1f}%")
    print(f"Decision-fused ensemble accuracy: {best_fused_acc * 100:.1f}% (IMU weight={best_weight:.2f})")


if __name__ == "__main__":
    main()
