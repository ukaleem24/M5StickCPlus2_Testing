"""
Runs the same labeling -> downsampling -> audio resampling -> windowing
pipeline as scripts 2-5 (see those files for the original, single-session,
current-directory versions) against the synchronized "New data" sessions
produced by "1. correction_script.py", which live outside this repo at
Data/Synchronized/New/<session>/.

Differences from the original numbered scripts, and why:

- Operates on multiple sessions/devices in one run instead of one directory,
  since each session has 3 devices sharing one label sheet.
- The label spreadsheets here use a different format than the historical
  'Label'/'Start'/'End'-in-seconds convention (confirmed against
  Data/Synchronized/Kaleem 3 repititions/Data Labelling.xlsx): one has
  'label'/'start_time(sss.ms)'/'end_time(sss.ms)' with values in
  milliseconds, the other has no header row at all. load_labels() below
  detects and normalizes both into the Label/Start(sec)/End(sec) shape
  '2. embed_labels.py' expects.
- Device filenames are IP suffixes (e.g. imu_10_119_193_59.csv), not body
  locations -- this still satisfies '5. windowing_script.py's audio/IMU
  matching, which just checks the IMU stem is a substring of the audio
  filename, so no renaming was needed.

Everything else (downsample-by-mean+interpolate for numeric columns,
mode+ffill/bfill for the label column, librosa resampling for audio,
sliding-window + majority-vote-label windowing) is copied over unchanged
from scripts 3, 4, and 5 respectively.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy.io import wavfile

SYNCED_ROOT = Path(r"C:\Users\ukale\Desktop\Thesis\Data\Synchronized\New")

DOWNSAMPLE_TARGET_HZ = 80
AUDIO_TARGET_HZ = 16000

WINDOW_SIZE_MS = 2000
OVERLAP_RATIO = 0.5
MIN_LABEL_COVERAGE = 0.7


# ============================================================================
# Label loading (new: handles both formats found in Data/Synchronized/New)
# ============================================================================


def load_labels(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)

    if any(str(c).startswith("Unnamed") for c in df.columns):
        # No header row in the file at all -- re-read positionally.
        df = pd.read_excel(xlsx_path, header=None)
        df.columns = ["Label", "Start", "End"] + list(df.columns[3:])
    else:
        # Normalize whatever the real header names are to Label/Start/End.
        rename = {}
        for c in df.columns:
            lc = str(c).lower()
            if lc.startswith("label"):
                rename[c] = "Label"
            elif lc.startswith("start"):
                rename[c] = "Start"
            elif lc.startswith("end"):
                rename[c] = "End"
        df = df.rename(columns=rename)

    df = df[["Label", "Start", "End"]].dropna(subset=["Label", "Start", "End"])
    df["Label"] = df["Label"].astype(str).str.strip()

    # Both known new-format sheets declare "(sss.ms)" -- i.e. millisecond
    # values -- unlike the historical sheets' plain seconds. Detect by
    # magnitude as a safety net: real sessions run at most ~20 minutes
    # (1200s), so Start/End values that large can only be milliseconds.
    if df["End"].max() > 1200:
        df["Start"] = df["Start"] / 1000.0
        df["End"] = df["End"] / 1000.0

    return df.reset_index(drop=True)


def get_labels_vectorized(timestamps: pd.Series, df_labels: pd.DataFrame, min_timestamp: float) -> pd.Series:
    relative_timestamps = timestamps - min_timestamp
    labels = pd.Series(["unlabeled"] * len(relative_timestamps), index=timestamps.index)
    for _, row in df_labels.iterrows():
        mask = (relative_timestamps >= row["Start"]) & (relative_timestamps <= row["End"])
        labels[mask] = row["Label"]
    return labels


# ============================================================================
# Downsampling (copied from "3. downsample_script.py")
# ============================================================================


def downsample_imu(df: pd.DataFrame, target_hz: int) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["recv_ts"], unit="s")
    df.set_index("datetime", inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    period_us = int(1_000_000 / target_hz)
    freq_str = f"{period_us}us"

    df_numeric = df[numeric_cols].resample(freq_str).mean()
    df_numeric = df_numeric.interpolate(method="linear")

    if categorical_cols:
        def get_mode(series):
            m = series.dropna().mode()
            return m.iloc[0] if not m.empty else np.nan

        df_categorical = df[categorical_cols].resample(freq_str).agg(get_mode)
        df_categorical = df_categorical.ffill().bfill()
        df_resampled = pd.concat([df_numeric, df_categorical], axis=1)
    else:
        df_resampled = df_numeric

    df_resampled["recv_ts"] = df_resampled.index.astype("int64") / 10**9
    for col in ["seq", "raw_ts"]:
        if col in df_resampled.columns:
            df_resampled[col] = df_resampled[col].round().fillna(0).astype("int64")

    final_cols = [c for c in df.columns if c != "datetime"]
    return df_resampled[final_cols]


# ============================================================================
# Audio resampling (copied from "4. resample_audio.py")
# ============================================================================


def resample_audio(wav_path: Path, target_hz: int, out_path: Path):
    audio_data, orig_sr = librosa.load(wav_path, sr=None, mono=True)
    if orig_sr != target_hz:
        resampled = librosa.resample(y=audio_data, orig_sr=orig_sr, target_sr=target_hz)
    else:
        resampled = audio_data
    sf.write(out_path, resampled, target_hz, subtype="PCM_16")


# ============================================================================
# Windowing (copied from "5. windowing_script.py")
# ============================================================================


def get_dominant_label(labels: pd.Series, min_coverage: float):
    labeled_mask = labels != "unlabeled"
    coverage = labeled_mask.sum() / len(labels)
    if coverage < min_coverage:
        return "invalid", coverage
    labeled_labels = labels[labeled_mask]
    if len(labeled_labels) == 0:
        return "invalid", coverage
    return labeled_labels.value_counts().idxmax(), coverage


def create_imu_windows(df_imu: pd.DataFrame, window_size_ms: int, stride_ms: int, min_coverage: float):
    windows_by_label = {}
    start_ts_ms = df_imu["ts_ms"].min()
    end_ts_ms = df_imu["ts_ms"].max()

    window_start = start_ts_ms
    n_created, n_skipped = 0, 0
    while window_start + window_size_ms <= end_ts_ms:
        window_end = window_start + window_size_ms
        mask = (df_imu["ts_ms"] >= window_start) & (df_imu["ts_ms"] < window_end)
        window_data = df_imu[mask]
        if len(window_data) == 0:
            window_start += stride_ms
            continue
        label, coverage = get_dominant_label(window_data["label"], min_coverage)
        if label == "invalid":
            n_skipped += 1
            window_start += stride_ms
            continue
        windows_by_label.setdefault(label, []).append(
            {"start_ts": window_start, "end_ts": window_end, "coverage": coverage,
             "sample_count": len(window_data), "data": window_data.copy()}
        )
        n_created += 1
        window_start += stride_ms
    return windows_by_label, n_created, n_skipped


def load_audio_for_windowing(wav_path: Path, imu_start_ts_ms: int):
    sample_rate, audio_data = wavfile.read(wav_path)
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    num_samples = len(audio_data)
    sample_duration_ms = 1000.0 / sample_rate
    timestamps_ms = imu_start_ts_ms + np.arange(num_samples) * sample_duration_ms
    # .astype(int) maps to 32-bit on Windows and silently overflows/wraps for
    # epoch-millisecond values (~1.7e12) -- must be explicit about width here.
    df = pd.DataFrame({"ts_ms": timestamps_ms.astype("int64"), "audio_value": audio_data})
    return sample_rate, df


def create_audio_windows(df_audio: pd.DataFrame, imu_windows: dict):
    windows_by_label = {}
    for label, imu_window_list in imu_windows.items():
        windows_by_label[label] = []
        for w in imu_window_list:
            mask = (df_audio["ts_ms"] >= w["start_ts"]) & (df_audio["ts_ms"] < w["end_ts"])
            wd = df_audio[mask]
            if len(wd) == 0:
                continue
            windows_by_label[label].append({"start_ts": w["start_ts"], "data": wd})
    return windows_by_label


def save_windows(imu_windows, audio_windows, sample_rate, out_dir: Path, device: str):
    imu_out = out_dir / "imu"
    audio_out = out_dir / "audio"
    for label, window_list in imu_windows.items():
        label_dir = imu_out / label
        label_dir.mkdir(parents=True, exist_ok=True)
        for idx, w in enumerate(window_list):
            save_cols = [c for c in w["data"].columns if c not in ["ts_ms"]]
            w["data"][save_cols].to_csv(label_dir / f"imu_{label}_{w['start_ts']}_{idx:04d}.csv", index=False)
    for label, window_list in audio_windows.items():
        label_dir = audio_out / label
        label_dir.mkdir(parents=True, exist_ok=True)
        for idx, w in enumerate(window_list):
            wavfile.write(label_dir / f"audio_{label}_{w['start_ts']}_{idx:04d}.wav",
                           sample_rate, w["data"]["audio_value"].values)


# ============================================================================
# Main
# ============================================================================


def process_device(session_dir: Path, out_root: Path, device: str, df_labels: pd.DataFrame, report: dict):
    imu_path = session_dir / f"imu_{device}.csv"
    audio_path = session_dir / f"audio_{device}.wav"
    if not imu_path.exists() or not audio_path.exists():
        print(f"    [SKIP] {device}: missing imu/audio file")
        return

    df_imu = pd.read_csv(imu_path)
    min_ts = df_imu["recv_ts"].min()
    df_imu["label"] = get_labels_vectorized(df_imu["recv_ts"], df_labels, min_ts)
    labeled_frac = (df_imu["label"] != "unlabeled").mean()

    df_ds = downsample_imu(df_imu, DOWNSAMPLE_TARGET_HZ)
    ds_dir = out_root / f"imu_downsampled_{DOWNSAMPLE_TARGET_HZ}_audio_{AUDIO_TARGET_HZ // 1000}k"
    ds_dir.mkdir(parents=True, exist_ok=True)
    df_ds.to_csv(ds_dir / f"imu_{device}.csv", index=False)

    resampled_audio_path = ds_dir / f"audio_{device}.wav"
    resample_audio(audio_path, AUDIO_TARGET_HZ, resampled_audio_path)

    # Same 32-bit .astype(int) overflow risk as above -- must be int64.
    df_ds["ts_ms"] = (df_ds["recv_ts"] * 1000).astype("int64")
    imu_rate = len(df_ds) / ((df_ds["ts_ms"].max() - df_ds["ts_ms"].min()) / 1000.0)
    stride_ms = int(WINDOW_SIZE_MS * (1 - OVERLAP_RATIO))
    imu_windows, n_created, n_skipped = create_imu_windows(df_ds, WINDOW_SIZE_MS, stride_ms, MIN_LABEL_COVERAGE)

    imu_start_ts_ms = int(df_ds["ts_ms"].iloc[0])
    sample_rate, df_audio = load_audio_for_windowing(resampled_audio_path, imu_start_ts_ms)
    audio_windows = create_audio_windows(df_audio, imu_windows)

    windowed_dir = ds_dir / "windowed_data" / device
    windowed_dir.mkdir(parents=True, exist_ok=True)
    save_windows(imu_windows, audio_windows, sample_rate, windowed_dir, device)

    counts = {label: len(w) for label, w in imu_windows.items()}
    dev_report = {
        "device": device,
        "labeled_fraction": round(float(labeled_frac), 4),
        "imu_rate_after_downsample_hz": round(imu_rate, 2),
        "windows_created": n_created,
        "windows_skipped_low_coverage": n_skipped,
        "windows_by_label": counts,
    }
    report["devices"].append(dev_report)
    print(f"    device {device}: {dev_report}")

    stats_path = windowed_dir / "windowing_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(dev_report, f, indent=2)


def discover_devices(session_dir: Path):
    return sorted(p.stem[len("imu_"):] for p in session_dir.glob("imu_10_*.csv"))


def main():
    session_dirs = [p for p in sorted(SYNCED_ROOT.iterdir()) if p.is_dir()]
    for session_dir in session_dirs:
        label_files = list(session_dir.glob("*label*.xlsx"))
        if not label_files:
            print(f"\n=== Session: {session_dir.name} -- SKIPPED (no label file found) ===")
            continue

        print(f"\n=== Session: {session_dir.name} ===")
        df_labels = load_labels(label_files[0])
        print(f"  Loaded {len(df_labels)} label ranges from {label_files[0].name}")
        print(df_labels.to_string(index=False))

        devices = discover_devices(session_dir)
        report = {"session": session_dir.name, "label_ranges": len(df_labels), "devices": []}
        for device in devices:
            process_device(session_dir, session_dir, device, df_labels, report)

        if not report["devices"]:
            print(f"  [SKIP] no devices processed for {session_dir.name}, not writing a report")
            continue

        ds_dir = session_dir / f"imu_downsampled_{DOWNSAMPLE_TARGET_HZ}_audio_{AUDIO_TARGET_HZ // 1000}k"
        ds_dir.mkdir(parents=True, exist_ok=True)
        report_path = ds_dir / "windowing_run_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  -> wrote {report_path}")


if __name__ == "__main__":
    main()
