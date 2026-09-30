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


def grouped_train_val_split(
    labels: np.ndarray,
    timestamps: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stride_ms: int = 1000,
):
    """Per-class, group-aware alternative to sklearn's `train_test_split(..., stratify=...)`.

    Whole contiguous recording instances (see group_windows_by_instance) are assigned to
    train or validation as a unit -- never split across the boundary -- while still
    targeting each class's requested test_size fraction, the same guarantee `stratify=`
    gives for a plain random split.
    """
    labels = np.asarray(labels)
    timestamps = np.asarray(timestamps)
    group_ids = group_windows_by_instance(labels, timestamps, stride_ms=stride_ms)
    rng = np.random.RandomState(random_state)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for class_label in sorted(set(labels.tolist())):
        class_indices = np.where(labels == class_label)[0]
        class_groups = group_ids[class_indices]

        unique_groups = np.unique(class_groups)
        rng.shuffle(unique_groups)

        target_val_count = round(len(class_indices) * test_size)
        val_group_ids = set()
        running_count = 0
        for group in unique_groups:
            if running_count >= target_val_count:
                break
            val_group_ids.add(group)
            running_count += int(np.sum(class_groups == group))

        is_val = np.isin(class_groups, list(val_group_ids))
        val_idx.extend(class_indices[is_val].tolist())
        train_idx.extend(class_indices[~is_val].tolist())

    return np.array(sorted(train_idx)), np.array(sorted(val_idx))
