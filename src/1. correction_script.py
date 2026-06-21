import pandas as pd

import numpy as np

from scipy.io import wavfile

from pathlib import Path



# 1. Define all 4 sensor IPs

sensor_ips = ["10_90_50_49", "10_90_50_59", "10_90_50_196", "10_90_50_218"]

script_dir = Path(__file__).resolve().parent



for ip in sensor_ips:

    print(f"\n--- Processing Sensor {ip} ---")

   

    # 2. Load Data for this specific sensor

    audio_ts = pd.read_csv(script_dir / f"audio_ts_{ip}.csv")

    sample_rate, audio_data = wavfile.read(script_dir / f"audio_{ip}.wav")

   

    imu_cols = ['recv_ts', 'src', 'raw_ts', 'seq', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

    imu_df = pd.read_csv(script_dir / f"imu_{ip}.csv", skiprows=1, names=imu_cols)



    # 3. Fix Audio Dropouts

    chunk_size = 512

    expected_gap_us = 32000

    gaps = audio_ts['timestamp_us'].diff()

    missing_chunks = np.round((gaps - expected_gap_us) / expected_gap_us).fillna(0).astype(int)



    padded_audio = []

    for i in range(len(audio_data) // chunk_size):

        if missing_chunks.iloc[i] > 0:

            silence = np.zeros(missing_chunks.iloc[i] * chunk_size, dtype=audio_data.dtype)

            padded_audio.append(silence)

       

        start_idx = i * chunk_size

        end_idx = start_idx + chunk_size

        padded_audio.append(audio_data[start_idx:end_idx])



    continuous_audio = np.concatenate(padded_audio)



    # 4. Fix Clock Drift

    audio_start_us = audio_ts['timestamp_us'].min()

    audio_end_us = audio_ts['timestamp_us'].max()

    real_duration_sec = (audio_end_us - audio_start_us) / 1e6

    true_sample_rate = int(len(continuous_audio) / real_duration_sec)

   

    wavfile.write(script_dir / f"synced_audio_{ip}.wav", true_sample_rate, continuous_audio)



    # 5. Internal IMU Alignment

    # We trim the beginning of the IMU file so it starts at the exact same microsecond as the audio

    imu_start_us = imu_df['raw_ts'].min()

    offset_sec = (audio_start_us - imu_start_us) / 1e6

   

    # Calculate average IMU rate to trim rows accurately

    imu_rate = 1e6 / imu_df['raw_ts'].diff().mean()

   

    if offset_sec > 0:

        imu_drop_samples = int(offset_sec * imu_rate)

        synced_imu = imu_df.iloc[imu_drop_samples:].reset_index(drop=True)

    else:

        synced_imu = imu_df

   

    # Save the aligned IMU data

    synced_imu.to_csv(script_dir / f"synced_imu_{ip}.csv", index=False)

   

    print(f"Saved 'synced_audio_{ip}.wav' and 'synced_imu_{ip}.csv'.")

    print(f"Internal Offset of {offset_sec:.4f}s was corrected.")