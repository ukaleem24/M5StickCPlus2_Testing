import pandas as pd
import numpy as np
from scipy.io import wavfile
from pathlib import Path

# 1. Configuration
sensor_ips = ["10_90_50_49", "10_90_50_59", "10_90_50_196", "10_90_50_218"]
script_dir = Path(__file__).resolve().parent

# The exact visual offset established via cross-referencing with Audacity
trim_duration_sec = 3.857 

print(f"Starting Hard Sync: Trimming {trim_duration_sec} seconds from all files...\n")

for ip in sensor_ips:
    print(f"--- Processing Sensor {ip} ---")
    
    audio_file = script_dir / f"synced_audio_{ip}.wav"
    imu_file = script_dir / f"synced_imu_{ip}.csv"
    
    # ---------------------------------------------------------
    # 1. Trim Audio File
    # ---------------------------------------------------------
    if audio_file.exists():
        sample_rate, audio_data = wavfile.read(audio_file)
        
        # Calculate exactly how many frames equal 3.857 seconds at this specific sample rate
        frames_to_drop = int(trim_duration_sec * sample_rate)
        
        # Slice the numpy array
        trimmed_audio = audio_data[frames_to_drop:]
        
        # Save the new trimmed file
        out_audio_file = script_dir / f"trimmed_synced_audio_{ip}.wav"
        wavfile.write(out_audio_file, sample_rate, trimmed_audio)
        print(f"  -> Audio: Dropped {frames_to_drop} frames. Saved {out_audio_file.name}")
    else:
        print(f"  -> Warning: {audio_file.name} not found.")

    # ---------------------------------------------------------
    # 2. Trim IMU Data
    # ---------------------------------------------------------
    if imu_file.exists():
        imu_df = pd.read_csv(imu_file)
        
        # Get the very first timestamp in seconds
        start_time_sec = imu_df['recv_ts'].iloc[0]
        target_start_time = start_time_sec + trim_duration_sec
        
        # Drop all rows that occurred before the target start time
        trimmed_imu_df = imu_df[imu_df['recv_ts'] >= target_start_time].reset_index(drop=True)
        
        # Save the new trimmed file
        out_imu_file = script_dir / f"trimmed_synced_imu_{ip}.csv"
        trimmed_imu_df.to_csv(out_imu_file, index=False)
        
        rows_dropped = len(imu_df) - len(trimmed_imu_df)
        print(f"  -> IMU: Dropped {rows_dropped} rows. Saved {out_imu_file.name}")
    else:
        print(f"  -> Warning: {imu_file.name} not found.")
        
print("\nTrimming complete. The 'trimmed_' files are now perfectly aligned with your 0.000s video track.")