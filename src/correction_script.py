from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
# 1. Load Data
print("Loading files...")
script_dir = Path(__file__).resolve().parent
audio_ts = pd.read_csv(script_dir / "audio_ts_10_90_50_49.csv")
sample_rate, audio_data = wavfile.read(script_dir / "audio_10_90_50_49.wav")

# Load IMU data (assigning column names based on your structure)
imu_cols = ['recv_ts', 'src', 'raw_ts', 'seq', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
imu_df = pd.read_csv(script_dir / "imu_10_90_50_49.csv", skiprows=1, names=imu_cols)

# 2. Fix Audio Dropouts by Zero-Padding
print("Fixing audio dropouts...")
chunk_size = 512
expected_gap_us = 32000 # 32ms per chunk

gaps = audio_ts['timestamp_us'].diff()
# Calculate missing chunks (rounding to nearest whole chunk)
missing_chunks = np.round((gaps - expected_gap_us) / expected_gap_us).fillna(0).astype(int)

# Create a new, continuous audio array
padded_audio = []
for i in range(len(audio_data) // chunk_size):
    # If chunks were missed before this one, insert silence
    if missing_chunks.iloc[i] > 0:
        silence = np.zeros(missing_chunks.iloc[i] * chunk_size, dtype=audio_data.dtype)
        padded_audio.append(silence)
    
    # Append the actual audio chunk
    start_idx = i * chunk_size
    end_idx = start_idx + chunk_size
    padded_audio.append(audio_data[start_idx:end_idx])

continuous_audio = np.concatenate(padded_audio)

# 3. Calculate True Sample Rate to fix Clock Drift
# Use the first and last timestamps to find the exact real-world duration
audio_start_us = audio_ts['timestamp_us'].min()
audio_end_us = audio_ts['timestamp_us'].max()
real_duration_sec = (audio_end_us - audio_start_us) / 1e6

# True sample rate = total samples / real world time
true_sample_rate = int(len(continuous_audio) / real_duration_sec)
print(f"Corrected Sample Rate: {true_sample_rate} Hz")

# Save the synchronized, real-time audio file
wavfile.write(script_dir / "synced_audio.wav", true_sample_rate, continuous_audio)

# 4. Calculate the Start Offset
imu_start_us = imu_df['raw_ts'].min()

offset_us = audio_start_us - imu_start_us
offset_sec = offset_us / 1e6

print("\n--- Synchronization Complete ---")
print(f"1. Saved 'synced_audio.wav'. It is now perfectly continuous in real-time.")
print(f"2. ALIGNMENT: The IMU sensor started {offset_sec} seconds BEFORE the audio sensor.")
print("   To align them in your analysis, shift your audio data forward by this offset.")