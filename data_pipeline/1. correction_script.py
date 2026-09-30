"""
Synchronizes raw IMU/audio captures from test.py into labeling-ready files.

For each device found in a raw-capture session folder, this:
  - Discovers device IPs dynamically (no hardcoded list).
  - Detects mid-recording device reboots (the ESP32's own clock and packet
    sequence counters reset to 0 on reboot, but the receiving PC's recv_ts
    for IMU never does -- so recv_ts is the only reliable ground truth for
    how long the device was actually offline).
  - Gap-fills dropped audio packets with silence, sized correctly across a
    reboot using the IMU-measured real-world downtime (audio has no
    independent PC-side timestamp per packet, only the device clock, which
    resets at exactly the same instant as the IMU's does).
  - Recomputes audio's true sample rate from real elapsed time (device
    packet loss + clock drift both eat into the naive 16 kHz assumption)
    and writes the WAV at that corrected rate.
  - Aligns the IMU stream's start to the audio stream's start (device-clock
    based, matching how the two are recorded from the same board).
  - Writes a verification report (recorded duration, raw vs corrected
    sample rate, reboot events) per device so data quality is auditable
    before labeling.

recv_ts is the same column '2. embed_labels.py' and '5. windowing_script.py'
already treat as ground truth, so no reboot-specific handling is needed on
the IMU side -- only audio's extrapolated-timestamp reconstruction cares.
"""

from pathlib import Path
import csv
import json

import numpy as np
from scipy.io import wavfile

RAW_ROOT = Path(r"C:\Users\ukale\Desktop\Thesis\Data\Raw Data\New data")
OUT_ROOT = Path(r"C:\Users\ukale\Desktop\Thesis\Data\Synchronized\New")

CHUNK_SAMPLES = 512
NOMINAL_AUDIO_RATE = 16000
EXPECTED_GAP_US = CHUNK_SAMPLES / NOMINAL_AUDIO_RATE * 1e6  # 32000us per packet

IMU_HEADER = ["recv_ts", "src", "raw_ts", "seq", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def discover_devices(session_dir: Path):
    ips = []
    for p in sorted(session_dir.glob("imu_*.csv")):
        ips.append(p.stem[len("imu_"):])
    return ips


def parse_imu(path: Path):
    recv_ts, raw_ts, seq, rows = [], [], [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            if len(r) < 10:
                continue
            try:
                recv_ts.append(float(r[0]))
                raw_ts.append(int(r[2]))
                seq.append(int(r[3]))
            except ValueError:
                continue
            rows.append(r)
    return recv_ts, raw_ts, seq, rows


def parse_audio_ts(path: Path):
    seq, ts, counts = [], [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            seq.append(int(r[0]))
            ts.append(int(r[1]))
            counts.append(int(r[2]))
    return seq, ts, counts


def find_resets(seq):
    """Row indices where the device rebooted: packetSequence is a global that
    only ever starts at 0 at boot, so a genuine reboot drops the counter to
    (near) zero. A small backward step is just UDP packet reordering --
    harmless, since downstream scripts re-sort by recv_ts anyway -- and must
    not be confused with an actual reboot."""
    return [i for i in range(1, len(seq)) if seq[i] < seq[i - 1] and seq[i] < 50]


def gapfill_span(seq, ts, counts, audio_data, sample_offset, expected_gap_us):
    """Gap-fill packet loss within one continuous (no-reboot) span of audio."""
    pieces = []
    idx = sample_offset
    for i in range(len(seq)):
        if i > 0:
            gap = ts[i] - ts[i - 1]
            missing_chunks = round((gap - expected_gap_us) / expected_gap_us)
            if missing_chunks > 0:
                pieces.append(np.zeros(missing_chunks * CHUNK_SAMPLES, dtype=audio_data.dtype))
        c = counts[i]
        pieces.append(audio_data[idx : idx + c])
        idx += c
    return pieces, idx


def process_device(session_dir: Path, out_dir: Path, ip: str, report: dict):
    imu_path = session_dir / f"imu_{ip}.csv"
    audio_ts_path = session_dir / f"audio_ts_{ip}.csv"
    audio_wav_path = session_dir / f"audio_{ip}.wav"

    recv_ts, raw_ts, seq, imu_rows = parse_imu(imu_path)
    a_seq, a_ts, a_counts = parse_audio_ts(audio_ts_path)
    _, audio_data = wavfile.read(audio_wav_path)

    imu_resets = find_resets(seq)
    audio_resets = find_resets(a_seq)

    raw_imu_duration = recv_ts[-1] - recv_ts[0]
    raw_imu_rate = len(seq) / raw_imu_duration
    raw_audio_samples = sum(a_counts)
    raw_audio_span_s = (a_ts[-1] - a_ts[0]) / 1e6 if not audio_resets else None

    dev_report = {
        "device": ip,
        "recorded_duration_s": round(raw_imu_duration, 2),
        "imu_rows_received": len(seq),
        "imu_rate_hz_raw": round(raw_imu_rate, 2),
        "audio_packets_received": len(a_seq),
        "audio_samples_received": raw_audio_samples,
        "reboot_events": len(imu_resets),
    }

    if len(imu_resets) != len(audio_resets):
        dev_report["warning"] = (
            f"IMU detected {len(imu_resets)} reboot(s) but audio detected {len(audio_resets)} -- "
            "reboot boundaries could not be reliably cross-matched; treating each stream's own "
            "resets independently, cross-stream alignment after the first mismatch may be off."
        )

    # --- Reconstruct continuous audio, bridging any reboot with a correctly
    # sized silence gap measured from the IMU's own (never-resetting) recv_ts ---
    audio_bounds = [0] + audio_resets + [len(a_seq)]
    imu_bounds = [0] + imu_resets + [len(seq)]

    pieces = []
    sample_cursor = 0
    for seg_i in range(len(audio_bounds) - 1):
        s, e = audio_bounds[seg_i], audio_bounds[seg_i + 1]
        seg_pieces, sample_cursor = gapfill_span(
            a_seq[s:e], a_ts[s:e], a_counts[s:e], audio_data, sample_cursor, EXPECTED_GAP_US
        )
        pieces.extend(seg_pieces)

        # Bridge to the next segment (a reboot) using the IMU-measured downtime.
        if seg_i < len(audio_resets):
            if seg_i < len(imu_resets):
                imu_boundary_row = imu_bounds[seg_i + 1]
                downtime_s = recv_ts[imu_boundary_row] - recv_ts[imu_boundary_row - 1]
            else:
                downtime_s = 0.0
            downtime_samples = int(round(downtime_s * NOMINAL_AUDIO_RATE))
            if downtime_samples > 0:
                pieces.append(np.zeros(downtime_samples, dtype=audio_data.dtype))
            dev_report.setdefault("reboot_downtime_s", []).append(round(downtime_s, 2))

    continuous_audio = np.concatenate(pieces) if pieces else np.array([], dtype=audio_data.dtype)

    # --- True sample rate from real elapsed time (IMU recv_ts span is the
    # only wall-clock reference that survives a reboot uncorrupted) ---
    true_sample_rate = int(round(len(continuous_audio) / raw_imu_duration))
    wavfile.write(out_dir / f"audio_{ip}.wav", true_sample_rate, continuous_audio)

    dev_report["audio_samples_after_gapfill"] = int(len(continuous_audio))
    dev_report["audio_rate_declared_raw_hz"] = NOMINAL_AUDIO_RATE
    dev_report["audio_rate_corrected_hz"] = true_sample_rate
    dev_report["synced_audio_duration_s"] = round(len(continuous_audio) / true_sample_rate, 2)

    # --- Align IMU start to audio start (device-clock based, segment 0 only --
    # both clocks share the same epoch since boot within an uninterrupted span) ---
    audio_start_us = a_ts[0]
    imu_start_us = raw_ts[0]
    offset_sec = (audio_start_us - imu_start_us) / 1e6
    if offset_sec > 0 and raw_imu_rate > 0:
        drop_rows = int(offset_sec * raw_imu_rate)
        synced_rows = imu_rows[drop_rows:]
    else:
        synced_rows = imu_rows
    dev_report["imu_start_trim_s"] = round(max(offset_sec, 0.0), 3)
    dev_report["imu_rows_after_sync"] = len(synced_rows)

    with open(out_dir / f"imu_{ip}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(IMU_HEADER)
        writer.writerows(synced_rows)

    report["devices"].append(dev_report)
    print(f"  device {ip}: {dev_report}")


def main():
    if not RAW_ROOT.is_dir():
        print(f"Raw data root not found: {RAW_ROOT}")
        return

    session_dirs = [p for p in sorted(RAW_ROOT.iterdir()) if p.is_dir()]
    for session_dir in session_dirs:
        ips = discover_devices(session_dir)
        if not ips:
            continue
        print(f"\n=== Session: {session_dir.name} ({len(ips)} device(s)) ===")
        out_dir = OUT_ROOT / session_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        report = {"session": session_dir.name, "devices": []}
        for ip in ips:
            process_device(session_dir, out_dir, ip, report)

        with open(out_dir / "sync_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"  -> wrote {out_dir / 'sync_report.json'}")


if __name__ == "__main__":
    main()
