#pragma once

#include <Arduino.h>
#include <M5StickCPlus2.h>

#include "model_settings.h"

// The live IMU output rate on this hardware/library combo measures ~250 Hz (confirmed via
// the data-collection firmware), not the ~100 Hz this was originally sized for -- at
// 250 Hz a 2s window is ~500 raw samples. Sized with real margin so interleaved polling
// (see pollImuSample) never hits this cap early and silently truncates the window.
constexpr int IMU_MAX_RAW_SAMPLES = 550;

// src/3. downsample_script.py mean-bins the raw ~250 Hz capture down to 80 Hz before
// windowing -- confirmed against data/windowed_data, where every raw IMU window CSV has
// exactly 160 rows (80 Hz x 2s). That averaging is a deliberate anti-aliasing low-pass
// filter, not just a rate change: skipping it here would feed the model live windows
// that are far noisier/higher-bandwidth than anything it saw in training, even once the
// duration and final resample-to-IMU_WINDOW_LENGTH step (below) are otherwise correct.
constexpr int IMU_DOWNSAMPLE_BIN_COUNT = 160;

struct ImuCaptureBuffer {
  float samples[IMU_MAX_RAW_SAMPLES * IMU_CHANNELS];
  int sampleCount = 0;
};

// One non-blocking check: appends a sample only if the IMU driver has a fresh reading
// ready right now. Meant to be called repeatedly, interleaved with the audio capture
// steps in main.cpp, so IMU and audio are sampled across the *same* physical time window
// instead of as two sequential ~2s phases -- both backbones were trained on IMU and audio
// paired by matching timestamp, so capturing them from different moments in time (even if
// each is internally correct) is a real distribution mismatch, not just noise.
inline void pollImuSample(ImuCaptureBuffer &buffer) {
  if (buffer.sampleCount >= IMU_MAX_RAW_SAMPLES) {
    return;
  }
  if (StickCP2.Imu.update()) {
    auto data = StickCP2.Imu.getImuData();
    float *row = &buffer.samples[buffer.sampleCount * IMU_CHANNELS];
    row[0] = data.accel.x;
    row[1] = data.accel.y;
    row[2] = data.accel.z;
    row[3] = data.gyro.x;
    row[4] = data.gyro.y;
    row[5] = data.gyro.z;
    buffer.sampleCount++;
  }
}

// Mean-bins the raw captured samples into IMU_DOWNSAMPLE_BIN_COUNT contiguous groups,
// mirroring downsample_script.py's `.resample(freq).mean()` low-pass step (time-based
// there; index-based here, which is an equivalent close approximation since the capture
// loop above polls at a roughly regular interval). Safe to call with
// binnedSamples == sourceSamples (in place): bin k only ever reads source indices
// >= rangeStart(k) >= k, so it never reads a position an earlier bin has already
// overwritten, and IMU_DOWNSAMPLE_BIN_COUNT < IMU_MAX_RAW_SAMPLES keeps every write
// behind the next bin's read range.
inline void meanBinImuSamples(const float *sourceSamples, int sourceLength, float *binnedSamples) {
  for (int binIndex = 0; binIndex < IMU_DOWNSAMPLE_BIN_COUNT; binIndex++) {
    int rangeStart = (int)((long)binIndex * sourceLength / IMU_DOWNSAMPLE_BIN_COUNT);
    int rangeEnd = (int)((long)(binIndex + 1) * sourceLength / IMU_DOWNSAMPLE_BIN_COUNT);
    rangeEnd = max(rangeEnd, rangeStart + 1);
    rangeEnd = min(rangeEnd, sourceLength);

    float sums[IMU_CHANNELS] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int count = 0;
    for (int i = rangeStart; i < rangeEnd; i++) {
      for (int channel = 0; channel < IMU_CHANNELS; channel++) {
        sums[channel] += sourceSamples[i * IMU_CHANNELS + channel];
      }
      count++;
    }
    for (int channel = 0; channel < IMU_CHANNELS; channel++) {
      binnedSamples[binIndex * IMU_CHANNELS + channel] = sums[channel] / (float)count;
    }
  }
}

// Linear-interpolation resample along the time axis (mirroring train_raw_cnn.py's
// resample_window), normalizes with the training-time per-channel stats, quantizes with
// the IMU model's own input scale/zero-point, and writes straight into its input tensor
// data -- this model gets its own dedicated (IMU_WINDOW_LENGTH, IMU_CHANNELS) tensor, not
// a channel-offset slice of a shared one like the single-model early-fusion deployment.
inline void resampleAndWriteImuWindow(const float *sourceSamples, int sourceLength, int8_t *inputData) {
  for (int destIndex = 0; destIndex < IMU_WINDOW_LENGTH; destIndex++) {
    float sourcePosition = (sourceLength == 1)
                                ? 0.0f
                                : (float)destIndex * (sourceLength - 1) / (float)(IMU_WINDOW_LENGTH - 1);
    int lowerIndex = (int)floorf(sourcePosition);
    int upperIndex = min(lowerIndex + 1, sourceLength - 1);
    float fraction = sourcePosition - lowerIndex;

    for (int channel = 0; channel < IMU_CHANNELS; channel++) {
      float lowerValue = sourceSamples[lowerIndex * IMU_CHANNELS + channel];
      float upperValue = sourceSamples[upperIndex * IMU_CHANNELS + channel];
      float value = lowerValue + fraction * (upperValue - lowerValue);

      float normalized = (value - IMU_CHANNEL_MEAN[channel]) / IMU_CHANNEL_STD[channel];
      int quantized = (int)lroundf(normalized / IMU_INPUT_SCALE) + IMU_INPUT_ZERO_POINT;
      quantized = constrain(quantized, -128, 127);
      inputData[destIndex * IMU_CHANNELS + channel] = (int8_t)quantized;
    }
  }
}
