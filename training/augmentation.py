from __future__ import annotations

import numpy as np


def _random_rotation_matrix(rng: np.random.RandomState, max_angle_deg: float = 15.0) -> np.ndarray:
    """Small random 3D rotation, applied jointly to the accel and gyro triplets to
    simulate natural variation in how the wrist-worn sensor is oriented/mounted."""
    angle = np.deg2rad(rng.uniform(-max_angle_deg, max_angle_deg))
    axis = rng.normal(size=3)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ],
        dtype=np.float32,
    )


def augment_imu_window(window: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """window: (T, 6) raw [acc_x,y,z, gyro_x,y,z]. Applies a shared small rotation of the
    accel/gyro triplets, per-channel magnitude scaling, jitter, and a small time-warp --
    label-preserving perturbations standard in HAR data augmentation, meant to partially
    compensate for the small (~1k window) training set."""
    augmented = window.copy()

    rotation = _random_rotation_matrix(rng)
    augmented[:, 0:3] = augmented[:, 0:3] @ rotation.T
    augmented[:, 3:6] = augmented[:, 3:6] @ rotation.T

    scale = rng.uniform(0.9, 1.1, size=(1, augmented.shape[1])).astype(np.float32)
    augmented = augmented * scale

    channel_std = augmented.std(axis=0, keepdims=True) + 1e-6
    noise = rng.normal(scale=0.03, size=augmented.shape).astype(np.float32) * channel_std
    augmented = augmented + noise

    # Time-warp: resample at a random speed factor, then resample back to the original
    # length so the model's fixed input shape is preserved.
    warp_factor = rng.uniform(0.9, 1.1)
    warped_length = max(2, int(round(augmented.shape[0] * warp_factor)))
    source_index = np.linspace(0.0, 1.0, num=augmented.shape[0], dtype=np.float32)
    warped_index = np.linspace(0.0, 1.0, num=warped_length, dtype=np.float32)
    warped = np.stack(
        [np.interp(warped_index, source_index, augmented[:, c]) for c in range(augmented.shape[1])],
        axis=1,
    )
    resampled_index = np.linspace(0.0, 1.0, num=warped.shape[0], dtype=np.float32)
    result = np.stack(
        [np.interp(source_index, resampled_index, warped[:, c]) for c in range(warped.shape[1])],
        axis=1,
    ).astype(np.float32)
    return result


def spec_augment(
    spectrogram: np.ndarray,
    rng: np.random.RandomState,
    max_time_mask: int = 20,
    max_freq_mask: int = 6,
    num_masks: int = 2,
) -> np.ndarray:
    """spectrogram: (T, F) or (T, F, 1) log-mel spectrogram. Applies a small circular
    time-shift plus SpecAugment-style random time/frequency masking (filled with the
    spectrogram's own mean, not zero, since these are log-energies, not linear ones)."""
    augmented = spectrogram.copy()
    has_channel_dim = augmented.ndim == 3
    if has_channel_dim:
        augmented = augmented[..., 0]

    num_frames, num_bins = augmented.shape
    fill_value = float(augmented.mean())

    shift = rng.randint(-10, 11)
    augmented = np.roll(augmented, shift, axis=0)

    for _ in range(num_masks):
        time_width = rng.randint(0, max_time_mask + 1)
        if 0 < time_width < num_frames:
            start = rng.randint(0, num_frames - time_width)
            augmented[start : start + time_width, :] = fill_value

        freq_width = rng.randint(0, max_freq_mask + 1)
        if 0 < freq_width < num_bins:
            start = rng.randint(0, num_bins - freq_width)
            augmented[:, start : start + freq_width] = fill_value

    if has_channel_dim:
        augmented = augmented[..., np.newaxis]
    return augmented.astype(np.float32)
