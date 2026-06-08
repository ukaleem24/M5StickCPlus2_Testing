# Data Collection and Experimental Design

## 1. Objective

This chapter documents the multi-modal sensor data collection campaign conducted at the Fraunhofer IPA Stuttgart Future Works Lab. The goal was to capture synchronized audio and inertial measurement unit (IMU) data from participants performing a realistic assembly task while wearing multiple wearable sensors at different body locations.

## 2. Experimental Environment

### 2.1 Location
- **Facility:** Fraunhofer IPA Stuttgart Future Works Lab
- **Type:** Industrial/manufacturing simulation laboratory equipped for human-machine interaction research

### 2.2 Rationale for Multi-Location Sensors
Different body locations provide complementary information about human motion and activity:
- **Wrists (×2):** Capture fine-grained hand and arm movements, tremor, gestures
- **Waist:** Represent trunk and torso movement, postural stability
- **Leg:** Capture lower-body dynamics, balance, and foot movement

This distributed sensor placement enables:
- Activity recognition across the entire body (not just localized motion)
- Cross-body correlation analysis
- Robustness to single-point sensor failures
- Comprehensive biomechanical characterization

## 3. Hardware Configuration

### 3.1 Sensing Devices

**Device Model:** M5StickC Plus 2 (×4)

**Specifications per Device:**
| Component | Specification |
|-----------|---------------|
| Processor | ESP32-S3 (240 MHz dual-core) |
| Connectivity | WiFi 802.11 b/g/n |
| IMU Sensor | MPU-6886 (3-axis accelerometer + 3-axis gyroscope) |
| Microphone | Analog input (integrated on-board) |
| Power | 1350 mAh Li-Po Battery |

### 3.2 Sensor Placement

```
Participant Body Map:
┌─────────────────────────┐
│   Left Wrist (Device 1) │  <- Accelerometer + Gyroscope + Audio
│         |               │
│    Waist (Device 3)     │  <- Accelerometer + Gyroscope + Audio
│     (center)            │
│         |               │
│ Right Wrist (Device 2)  │  <- Accelerometer + Gyroscope + Audio
│         |               │
│   Leg/Ankle (Device 4)  │  <- Accelerometer + Gyroscope + Audio
└─────────────────────────┘
```

**Mounting Method:**
- Devices secured to body with elastic straps or adhesive bands
- Sensors oriented with consistent x, y, z axes relative to body frame
- Placement ensures comfort and minimizes slipping during task execution

### 3.3 Network Configuration

**WiFi Network:**
- SSID: `Redmi Note 11`
- Security: WPA2 (password: passpass123)
- Host IP: `10.90.50.46`
- Network Role: Local area network (LAN) for real-time UDP streaming

**Data Transmission Ports:**
- **Port 1234:** IMU data (6-DoF motion)
- **Port 1235:** Audio data (spatially distributed microphones)

**Transmission Method:** UDP (User Datagram Protocol)
- **Rationale:** Low-latency, connectionless streaming suitable for real-time wearable sensor networks
- **Trade-off:** Occasional packet loss accepted in exchange for minimal transmission delay

## 4. Firmware and Data Acquisition

### 4.1 Overview

Each M5StickC Plus 2 runs custom firmware (see [main.cpp](../src/main.cpp)) that simultaneously captures and streams two independent data modalities:

#### **Inertial Measurement Unit (IMU)**
- **Sampling Rate:** 250 Hz
- **Axes:** 3-axis acceleration + 3-axis angular velocity (6 degrees of freedom)
- **Data Format per packet:**
  ```
  timestamp_us, seq, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
  [microseconds], [count], [g-units], [g-units], [g-units], [°/s], [°/s], [°/s]
  ```
- **Update Frequency:** Sent on every IMU data availability event
- **Timestamp Precision:** Microsecond-level (from `micros()` call at packet generation)

#### **Audio**
- **Sampling Rate:** 16,000 Hz (16 kHz, telephony grade)
- **Resolution:** 16-bit signed integers (PCM)
- **Channels:** Mono (single on-device microphone)
- **Chunk Size:** 512 samples per packet (= 32 milliseconds of audio)
- **Data Format per packet:**
  ```
  [seq (4B)][count (2B)][timestamp_us (8B)][audio_samples (1024B)]
  ```
- **Packet Size:** 14 bytes header + 1024 bytes audio = 1038 bytes (well below 1500 byte MTU)

### 4.2 Real-Time Streaming Architecture

**Key Design Features:**

1. **Double Buffering (Audio):**
   - Two 512-sample circular buffers prevent race conditions
   - One buffer fills while the other transmits
   - Ensures no audio samples are overwritten before transmission

2. **Microsecond Timestamps:**
   - IMU: Captured at packet generation (`micros()` call)
   - Audio: Captured at microphone read completion
   - Enables post-hoc synchronization across devices

3. **Sequence Counters:**
   - Both IMU and audio packets include sequence numbers
   - Detects packet loss during transmission
   - Enables reconstruction of dropped packets in post-processing

4. **WiFi Optimization:**
   - WiFi power saving disabled (`WiFi.setSleep(false)`)
   - Reduces latency variance from aggressive power management
   - Critical for real-time streaming applications

5. **UDP Atomic Transactions:**
   - Each audio packet transmitted in single `udp.write()` call
   - Prevents fragmentation and out-of-order delivery
   - Entire packet arrives intact or is lost (no partial packets)

## 5. Data Collection Protocol

### 5.1 Experimental Task

Participants performed a **structured assembly task** involving manipulation of physical parts, tools, and fasteners. The task was designed to generate realistic, naturalistic motion patterns while maintaining reproducibility across multiple participants.

### 5.2 Task Sequence

**Phase 1: Preparation & Positioning**
1. Participant stands at workbench in relaxed posture
2. All 4 sensors activated and confirmed online
3. Participant reads task instructions
4. Experimenter confirms recording start on host computer

**Phase 2: Assembly Operations** (primary data collection)

The participant performs the following steps in sequence:

| Step | Action | Duration | Body Regions Active |
|------|--------|----------|---------------------|
| 1-3 | Idle, walk, reach for parts | Setup | Legs, Waist |
| 4-6 | Pick up parts, place on table | ~30s | Wrists, Waist |
| 7-14 | Reach for hammer, measure, smash | ~1m | Wrists (dominant), Waist |
| 15-19 | Reach for screwdriver, grab items | ~45s | Wrists, Waist |
| 20-23 | Repeat screw insertion & measurement | ~1m30s | Wrists, Waist |
| 24-32 | Final assembly, remove screws, store | ~1m | Wrists, Waist |

**Key Characteristics:**
- **Repetitive movements:** Multiple reaching, grasping, and insertion cycles
- **Tool usage:** Hammer strikes, screwdriver rotation capture high-frequency dynamics
- **Precision tasks:** Measurement and fastening require fine motor control
- **Naturalistic:** Task resembles real-world assembly work

**Total Duration per Participant:** ~5-7 minutes of continuous data

### 5.3 Data Validation During Collection

**Real-Time Monitoring:**
- Host computer displays incoming UDP packets from all 4 devices
- Experimenter verifies:
  - ✅ All 4 devices connected and transmitting
  - ✅ No prolonged packet loss (sequence counter jumps)
  - ✅ Audio levels within acceptable range (no clipping or silence)
  - ✅ IMU accelerometer readings sensible (±2g during normal movement)

**Interrupt Protocol:**
- If any device disconnects mid-task, experiment paused and restarted
- If audio/IMU severely corrupted, task repeated with adjusted sensor placement

## 6. Data Streams Collected

### 6.1 Raw Data Files

**Per Participant, Per Device:**

| Filename | Content | Size (est.) |
|----------|---------|-------------|
| `imu_{IP}.csv` | 250 Hz IMU (6-DoF) for ~5-7 min | 300-400 KB |
| `audio_ts_{IP}.csv` | Audio chunk timestamps, metadata | 5-10 KB |
| `audio_{IP}.wav` | 16 kHz mono PCM audio | 5-7 MB |

**Total Data per Participant:** ~20-28 MB (4 devices)

### 6.2 Data Format Details

**IMU Data (`imu_{IP}.csv`):**
```
recv_ts,src,raw_ts,seq,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
1622000001234567,10_90_50_49,1622000001234500,0,-0.1234,0.9876,0.0156,2.45,-1.20,0.08
...
```

**Audio Metadata (`audio_ts_{IP}.csv`):**
```
chunk_index,timestamp_us
0,1622000001234500
1,1622000001266500
2,1622000001298500
...
```

**Audio Waveform (`audio_{IP}.wav`):**
- RIFF WAV format, 16-bit PCM
- Single channel (mono)
- 16,000 Hz sample rate (may vary slightly per device)

## 7. Technical Specifications Summary

### 7.1 Sensor Specifications

| Parameter | IMU (250 Hz) | Audio (16 kHz) |
|-----------|--------------|----------------|
| **Sampling Rate** | 250 Hz | 16,000 Hz |
| **Resolution** | 16-bit accelerometer, 16-bit gyroscope | 16-bit signed integer |
| **Range** | ±8g accel, ±2000°/s gyro | Microphone dependent |
| **Axes** | 3 acceleration + 3 rotation | 1 (mono) |
| **Timestamp** | Microsecond precision | Microsecond precision |
| **Latency** | ~10 ms from sample to transmit | ~10-20 ms (buffered) |

### 7.2 Network Specifications

| Parameter | Value |
|-----------|-------|
| **Protocol** | UDP over WiFi 802.11n |
| **Host** | 10.90.50.46 |
| **IMU Port** | 1234 |
| **Audio Port** | 1235 |
| **Packet Size (IMU)** | ~100 bytes |
| **Packet Size (Audio)** | 1038 bytes |
| **Max Throughput** | ~8 Mbps (250 Hz × 4 devices IMU + 16 kHz × 4 audio streams) |

### 7.3 Expected Data Quality Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| **WiFi Link Quality** | >80% signal strength | TBD |
| **IMU Packet Loss** | <1% | TBD |
| **Audio Packet Loss** | <2% | TBD |
| **Clock Synchronization Error** | <100 µs inter-device | Corrected in post-processing |
| **Audio Dropout Rate** | <1% of chunks | Corrected in sync script |

## 8. Quality Assurance & Mitigation

### 8.1 Potential Issues and Mitigations

**Issue:** WiFi packet loss due to interference
- **Mitigation:** Conducted experiment in isolated lab; used 2.4 GHz band; verified no other networks on same SSID

**Issue:** Audio clipping or saturation
- **Mitigation:** Calibrated microphone gain before each participant; monitored peak levels during task

**Issue:** IMU cross-axis interference (accelerometer + gyroscope coupling)
- **Mitigation:** Devices mounted orthogonal to body; used factory calibration data from MPU-6886

**Issue:** Battery depletion mid-task
- **Mitigation:** All devices fully charged before experiment; estimated 2+ hours runtime per charge

**Issue:** Temporal desynchronization between devices
- **Mitigation:** Post-hoc synchronization algorithms (dropout padding, clock drift correction) applied in signal processing pipeline

## 9. Participants

**Sample Size:** [To be filled: Number of participants]

**Demographics:** [To be filled: Age range, gender distribution, prior experience if relevant]

**Ethical Approval:** [To be filled: Ethics committee approval reference, informed consent procedures]

## 10. Data Storage and Organization

All collected data organized in directory structure:

```
Data Thesis/
├── Florian 4 and 1 invalid sequence for validation/
│   ├── audio_ts_10_90_50_*.csv
│   ├── imu_10_90_50_*.csv
│   └── ... (4 sensor sets)
├── Florian First 3 wrists ankle waist/
│   └── ... (4 sensor sets)
├── Kaleem 3 repititions/
│   ├── audio_ts_10_90_50_*.csv
│   ├── audio_10_90_50_*.wav
│   ├── imu_10_90_50_*.csv
│   ├── script.py (synchronization)
│   ├── edge_impulse_script.py (formatting)
│   └── synced_*.csv (post-processing outputs)
```

Each subdirectory represents one participant collection session, with data from all 4 devices present.

## 11. Next Steps

1. ✅ **Data Collection Complete**
2. ✅ **Synchronization & Dropout Correction** (see [01_DATA_SYNCHRONIZATION_DOCUMENTATION.md](01_DATA_SYNCHRONIZATION_DOCUMENTATION.md))
3. → **Feature Extraction** (audio spectrograms, IMU statistics)
4. → **Machine Learning Model Training**
5. → **Activity Recognition & Classification**

---

**Author:** [Your Name]  
**Date:** [Date of Experiment]  
**Facility:** Fraunhofer IPA Stuttgart Future Works Lab  
**Thesis:** Master's Thesis - [Title]
