# Data Synchronization and Correction

## 1. Overview

The first critical step in this research involved synchronizing and correcting multi-modal sensor data collected from four distributed M5StickC Plus 2 devices. Each sensor simultaneously captured audio and inertial measurement unit (IMU) data, but due to network latency, clock drift, and audio codec buffering, the data streams required careful alignment before downstream processing and model training.

## 2. Problem Statement

When collecting synchronized sensor data across multiple distributed devices, three primary synchronization challenges arise:

### 2.1 Audio Dropouts
- The audio codec on each M5StickC Plus 2 device buffers data in 512-sample chunks
- Network congestion and variable transmission latency cause sporadic loss of audio chunks
- Missing chunks result in non-continuous audio streams with temporal gaps
- These gaps invalidate downstream audio processing and frequency analysis

### 2.2 Clock Drift
- Each device runs an independent local clock with inherent frequency drift
- While the audio timestamp column (`timestamp_us`) records microsecond precision, the actual sample rate may deviate from the nominal rate (e.g., 16 kHz)
- Without correction, audio resampling and audio-IMU synchronization becomes inaccurate

### 2.3 Temporal Misalignment (Audio-IMU Offset)
- Audio and IMU modules on each device initialize independently with different latencies
- The audio stream and IMU stream do not start at the same microsecond timestamp
- Downstream fusion models require aligned data with matching temporal references

## 3. Methodology

### 3.1 Data Processing Pipeline

The synchronization and correction workflow processes each of the four sensors independently through the following steps:

#### **Step 1: Data Loading**
```
Input Files (per sensor):
  - audio_ts_{IP}.csv       → Audio chunk timestamps (microseconds)
  - audio_{IP}.wav          → Raw audio waveform
  - imu_{IP}.csv            → IMU measurements (acceleration, gyroscope)
```

Each sensor's data is loaded with explicit path resolution to ensure the script runs correctly regardless of the working directory.

#### **Step 2: Audio Dropout Detection and Padding**
**Problem:** Audio is transmitted in 512-sample chunks with an expected gap of 32,000 microseconds between consecutive chunks (at 16 kHz, 512 samples = 32 ms = 32,000 µs).

**Solution:**
1. Calculate inter-chunk gaps from the `audio_ts` timestamps
2. Identify missing chunks: `missing_chunks = (gaps - expected_gap) / expected_gap`
3. Insert silence (zeros) in the audio waveform for each missing chunk
4. Result: A continuous audio stream with no temporal gaps, preserving the original sample rate relationship

**Mathematical Basis:**
$$\text{expected\_gap} = \frac{\text{chunk\_size}}{\text{nominal\_sample\_rate}} \times 10^6 = \frac{512}{16000} \times 10^6 = 32000 \text{ µs}$$

#### **Step 3: Clock Drift Correction**
**Problem:** The nominal sample rate (e.g., 16 kHz) may differ from the actual rate due to crystal oscillator drift on the device.

**Solution:**
1. Extract the audio timestamp range: `[audio_start_us, audio_end_us]`
2. Calculate the real duration: `real_duration_sec = (audio_end_us - audio_start_us) / 10^6`
3. Calculate true sample rate: `true_sample_rate = len(continuous_audio) / real_duration_sec`
4. Resample the audio to the corrected rate
5. Save the corrected audio with the updated sample rate

**Rationale:** The number of samples is fixed after dropout correction. By measuring the actual elapsed time between the first and last timestamp, we derive the true sample rate.

#### **Step 4: Internal IMU Alignment**
**Problem:** The IMU data stream starts at a different microsecond timestamp than the audio stream, even though collection was simultaneous.

**Solution:**
1. Identify temporal offset: `offset_sec = (audio_start_us - imu_start_us) / 10^6`
2. Estimate IMU sampling rate: `imu_rate = 1 / (mean(diff(raw_ts)) × 10^-6)` 
3. Calculate samples to drop: `imu_drop_samples = offset_sec × imu_rate`
4. Trim the IMU dataframe to start at the same time reference as the audio
5. Reset the index for downstream processing

**Result:** Both audio and IMU streams now share the same temporal origin (0 µs reference).

### 3.2 Output

For each sensor (identified by IP address `{IP}`), the following synchronized files are generated:

| File | Description |
|------|-------------|
| `synced_audio_{IP}.wav` | Continuous audio with dropouts corrected and clock drift compensated |
| `synced_imu_{IP}.csv` | IMU data temporally aligned with the audio start time |

## 4. Implementation Details

### 4.1 Key Parameters

- **Chunk Size:** 512 samples (fixed by audio codec)
- **Expected Chunk Gap:** 32,000 µs (for 16 kHz nominal rate)
- **Nominal Sample Rate:** 16,000 Hz
- **IMU Columns:** `recv_ts`, `src`, `raw_ts`, `seq`, `acc_x`, `acc_y`, `acc_z`, `gyro_x`, `gyro_y`, `gyro_z`

### 4.2 Path Resolution

The script uses Python's `pathlib.Path` to resolve file paths relative to the script location, ensuring robustness when executed from different working directories. This is critical when processing data across multiple folders.

```python
script_dir = Path(__file__).resolve().parent
audio_ts = pd.read_csv(script_dir / f"audio_ts_{ip}.csv")
```

## 5. Results and Outputs

### 5.1 Processed Datasets

For each of the 4 sensors, the synchronization pipeline produces:

- **Synced Audio:** Dropout-corrected, clock-drift-compensated audio files ready for feature extraction (MFCC, spectrogram, etc.)
- **Synced IMU:** Temporally aligned IMU streams with matching time reference to audio

### 5.2 Verification Steps

After processing, verification should confirm:
1. ✅ Audio files have no discontinuities (no silent gaps from missing chunks)
2. ✅ Audio duration matches the corrected sample rate × number of samples
3. ✅ IMU and audio streams start at the same time reference (0 offset)
4. ✅ All four sensors' data are independently synchronized

## 6. Significance for Downstream Processing

Accurate data synchronization is foundational for:
- **Multi-modal Feature Fusion:** Combining audio and IMU features requires precise temporal alignment
- **Machine Learning Models:** Training models on misaligned data introduces systematic errors and reduces model generalization
- **Cross-sensor Comparisons:** Synchronized data enables valid comparative analysis across the four sensors
- **Time-Series Analysis:** Continuous, drift-corrected data preserves the true temporal relationships necessary for signal processing

## 7. Files and Code References

- **Main Processing Script:** `script.py`
- **Edge Impulse Formatter:** `edge_impulse_script.py` (formats IMU data for Edge Impulse platform)
- **Input Data Directory:** Contains raw `audio_ts_*.csv`, `audio_*.wav`, and `imu_*.csv` files for each sensor

---

**Author:** [Your Name]  
**Date:** [Date of Execution]  
**Thesis:** Master's Thesis - [Title]
