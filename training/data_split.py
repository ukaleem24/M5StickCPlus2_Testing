from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_window_timestamp(path) -> int:
    """Extract the window start timestamp (ms) from `<prefix>_<label>_<ts>_<idx>.ext`,
    matching the parsing convention in train_fusion_cnn.py's index_files_by_pair_key."""
    return int(Path(path).stem.split("_")[-2])


def group_windows_by_instance(
    labels: np.ndarray, timestamps: np.ndarray, stride_ms: int = 1000, gap_tolerance: float = 1.5
) -> np.ndarray:
    """Assign a group id to each window so that windows from the same contiguous,
    overlapping recording instance (same label, consecutive start timestamps within
    stride_ms * gap_tolerance of each other) share a group id.

    The windowing script (src/5. windowing_script.py) generates windows at 50% overlap
    (1s stride for a 2s window), so consecutive windows from one continuous performance
    of an activity share up to half their raw samples. Splitting by individual window
    would let those overlapping samples leak across the train/validation boundary;
    splitting by whole instance/group instead keeps every boundary between genuinely
    disjoint stretches of the recording.
    """
    labels = np.asarray(labels)
    timestamps = np.asarray(timestamps)
    order = np.argsort(timestamps, kind="stable")
    max_gap = stride_ms * gap_tolerance

    group_ids = np.empty(len(labels), dtype=np.int64)
    current_group = -1
    previous_label = None
    previous_ts = None

    for i in order:
        label = labels[i]
        ts = timestamps[i]
        if label != previous_label or previous_ts is None or (ts - previous_ts) > max_gap:
            current_group += 1
        group_ids[i] = current_group
        previous_label = label
        previous_ts = ts

    return group_ids


# Long instances are cut into blocks of at most this many consecutive windows, so a single
# long walking/carrying/measuring bout no longer has to go entirely into one side of the
# split (previously this pushed some classes to 50-70% validation share).
BLOCK_WINDOWS = 20
WINDOW_MS = 2000


def block_windows_by_instance(group_ids: np.ndarray, timestamps: np.ndarray,
                              block_windows: int = BLOCK_WINDOWS) -> np.ndarray:
    """Split each instance group into equally sized blocks of consecutive windows
    (e.g. 45 windows -> 15/15/15, not 20/20/5) and return a block id per window."""
    block_ids = np.empty(len(group_ids), dtype=np.int64)
    next_id = 0
    for group in np.unique(group_ids):
        members = np.where(group_ids == group)[0]
        members = members[np.argsort(timestamps[members], kind="stable")]
        n_blocks = int(np.ceil(len(members) / block_windows))
        for chunk in np.array_split(members, n_blocks):
            block_ids[chunk] = next_id
            next_id += 1
    return block_ids


def grouped_train_val_split(
    labels: np.ndarray,
    timestamps: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stride_ms: int = 1000,
    return_dropped: bool = False,
):
    """Per-class, group-aware alternative to sklearn's `train_test_split(..., stratify=...)`.

    Contiguous recording instances (see group_windows_by_instance) are cut into blocks of
    at most BLOCK_WINDOWS windows. Per class, shuffled blocks are added to validation only
    when that brings the class's validation count closer to test_size, so every class
    lands near the requested fraction. Finally, every training window whose time span
    overlaps a validation window (start times less than one window length apart, any
    class) is dropped from training: this keeps the block cuts leak-free and also closes
    the overlap at activity transitions, where the last window of one activity shares
    samples with the first window of the next.
    """
    labels = np.asarray(labels)
    timestamps = np.asarray(timestamps)
    group_ids = group_windows_by_instance(labels, timestamps, stride_ms=stride_ms)
    block_ids = block_windows_by_instance(group_ids, timestamps)
    rng = np.random.RandomState(random_state)

    is_val = np.zeros(len(labels), dtype=bool)
    for class_label in sorted(set(labels.tolist())):
        class_indices = np.where(labels == class_label)[0]
        class_blocks = block_ids[class_indices]
        unique_blocks = np.unique(class_blocks)
        rng.shuffle(unique_blocks)

        target_val_count = len(class_indices) * test_size
        running_count = 0
        for block in unique_blocks:
            size = int(np.sum(class_blocks == block))
            if abs(running_count + size - target_val_count) < abs(running_count - target_val_count):
                is_val[class_indices[class_blocks == block]] = True
                running_count += size

    val_timestamps = np.sort(timestamps[is_val])
    train_candidates = np.where(~is_val)[0]
    candidate_ts = timestamps[train_candidates]
    pos = np.searchsorted(val_timestamps, candidate_ts)
    nearest_left = np.abs(candidate_ts - val_timestamps[np.clip(pos - 1, 0, len(val_timestamps) - 1)])
    nearest_right = np.abs(val_timestamps[np.clip(pos, 0, len(val_timestamps) - 1)] - candidate_ts)
    overlaps_val = np.minimum(nearest_left, nearest_right) < WINDOW_MS

    train_idx = np.sort(train_candidates[~overlaps_val])
    val_idx = np.sort(np.where(is_val)[0])
    if return_dropped:
        return train_idx, val_idx, np.sort(train_candidates[overlaps_val])
    return train_idx, val_idx
