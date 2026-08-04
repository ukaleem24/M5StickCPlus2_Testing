#pragma once

#include <Arduino.h>
#include <M5StickCPlus2.h>
#include <arduinoFFT.h>
#include <math.h>

#include "model_settings.h"

// Reproduces train_audio_cnn.py's compute_log_mel_spectrograms on-device, streaming the
// mic capture frame-by-frame (matching src/main.cpp's chunked Mic.record() usage) rather
// than buffering the full 2s/32000-sample clip, which does not fit the ESP32's DRAM
// budget alongside the TFLite tensor arena. Only the AUDIO_RAW_FRAMES x AUDIO_MEL_BINS
// mel energies (not the raw audio) are buffered, for the time-axis resample step.
class AudioFeatureExtractor {
 public:
  // Split into per-step calls (rather than one call that blocks for the whole ~2s clip)
  // so the caller can interleave IMU polling between each audio chunk -- the two
  // modalities need to span the same physical time window, matching how the training
  // data pairs them by timestamp, not run as two back-to-back independent captures.

  // Records the first AUDIO_FRAME_LENGTH-sample frame and computes its mel energies.
  // Returns false on capture failure (see recordWithRetry).
  bool beginCapture() {
    rawMin_ = 32767;
    rawMax_ = -32768;
    if (!recordWithRetry(rollingBuffer_, AUDIO_FRAME_LENGTH)) {
      Serial.println("Mic capture failed (initial frame).");
      return false;
    }
    trackRawRange(rollingBuffer_, AUDIO_FRAME_LENGTH);
    computeFrameMel(0);
    return true;
  }

  // Records one more AUDIO_FRAME_STEP-sample chunk, slides the rolling window, and
  // computes mel energies for raw frame index frameIndex (1..AUDIO_RAW_FRAMES-1).
  bool captureNextFrame(int frameIndex) {
    if (!recordWithRetry(chunkBuffer_, AUDIO_FRAME_STEP)) {
      Serial.printf("Mic capture failed (frame %d).\n", frameIndex);
      return false;
    }
    trackRawRange(chunkBuffer_, AUDIO_FRAME_STEP);
    memmove(rollingBuffer_, rollingBuffer_ + AUDIO_FRAME_STEP,
            (AUDIO_FRAME_LENGTH - AUDIO_FRAME_STEP) * sizeof(int16_t));
    memcpy(rollingBuffer_ + (AUDIO_FRAME_LENGTH - AUDIO_FRAME_STEP), chunkBuffer_,
           AUDIO_FRAME_STEP * sizeof(int16_t));
    computeFrameMel(frameIndex);
    return true;
  }

  // Diagnostic: raw int16 sample range seen across the whole capture, and a couple of raw
  // mel values, so we can tell whether the mic is picking up real varying signal or
  // returning near-silent/constant data despite Mic.record() reporting success.
  void printDiagnostics() {
    Serial.printf("Audio raw sample range: [%d, %d]\n", rawMin_, rawMax_);
    Serial.printf("First raw mel frame [0..4]: %.3f %.3f %.3f %.3f %.3f\n", rawMel_[0][0], rawMel_[0][1],
                  rawMel_[0][2], rawMel_[0][3], rawMel_[0][4]);
    Serial.printf("Last raw mel frame [0..4]: %.3f %.3f %.3f %.3f %.3f\n", rawMel_[AUDIO_RAW_FRAMES - 1][0],
                  rawMel_[AUDIO_RAW_FRAMES - 1][1], rawMel_[AUDIO_RAW_FRAMES - 1][2],
                  rawMel_[AUDIO_RAW_FRAMES - 1][3], rawMel_[AUDIO_RAW_FRAMES - 1][4]);
  }

  // Resamples the buffered raw mel frames onto FUSED_WINDOW_LENGTH and writes the
  // normalized + int8-quantized result directly into the model's input tensor data,
  // at channelOffset (audio channels start right after the IMU_CHANNELS columns).
  void writeResampledIntoTensor(int8_t *inputData, int channelOffset) {
    for (int destIndex = 0; destIndex < FUSED_WINDOW_LENGTH; destIndex++) {
      float sourcePosition = (float)destIndex * (AUDIO_RAW_FRAMES - 1) / (float)(FUSED_WINDOW_LENGTH - 1);
      int lowerIndex = (int)floorf(sourcePosition);
      int upperIndex = min(lowerIndex + 1, AUDIO_RAW_FRAMES - 1);
      float fraction = sourcePosition - lowerIndex;

      for (int melBin = 0; melBin < AUDIO_MEL_BINS; melBin++) {
        float lowerValue = rawMel_[lowerIndex][melBin];
        float upperValue = rawMel_[upperIndex][melBin];
        float value = lowerValue + fraction * (upperValue - lowerValue);

        int channelIndex = channelOffset + melBin;
        float normalized = (value - CHANNEL_MEAN[channelIndex]) / CHANNEL_STD[channelIndex];
        int quantized = (int)lroundf(normalized / INPUT_SCALE) + INPUT_ZERO_POINT;
        quantized = constrain(quantized, -128, 127);
        inputData[destIndex * FUSED_CHANNELS + channelIndex] = (int8_t)quantized;
      }
    }
  }

 private:
  bool recordWithRetry(int16_t *buffer, int sampleCount, int maxAttempts = 5) {
    for (int attempt = 0; attempt < maxAttempts; attempt++) {
      if (StickCP2.Mic.record(buffer, sampleCount)) {
        return true;
      }
      delay(1);
    }
    return false;
  }

  void trackRawRange(const int16_t *buffer, int sampleCount) {
    for (int i = 0; i < sampleCount; i++) {
      rawMin_ = min(rawMin_, buffer[i]);
      rawMax_ = max(rawMax_, buffer[i]);
    }
  }

  int16_t rawMin_ = 32767;
  int16_t rawMax_ = -32768;
  int16_t rollingBuffer_[AUDIO_FRAME_LENGTH];
  int16_t chunkBuffer_[AUDIO_FRAME_STEP];
  float vReal_[AUDIO_FFT_LENGTH];
  float vImag_[AUDIO_FFT_LENGTH];
  float magnitude_[AUDIO_SPECTROGRAM_BINS];
  float rawMel_[AUDIO_RAW_FRAMES][AUDIO_MEL_BINS];
  ArduinoFFT<float> fft_ = ArduinoFFT<float>(vReal_, vImag_, AUDIO_FFT_LENGTH, (float)AUDIO_SAMPLE_RATE);

  void computeFrameMel(int frameIndex) {
    // soundfile (training) reads WAV samples normalized to [-1, 1]; matching that here
    // keeps the FFT input on the same numeric scale the mel filterbank was fit against.
    for (int i = 0; i < AUDIO_FRAME_LENGTH; i++) {
      vReal_[i] = ((float)rollingBuffer_[i] / 32768.0f) * HANN_WINDOW[i];
    }
    for (int i = AUDIO_FRAME_LENGTH; i < AUDIO_FFT_LENGTH; i++) {
      vReal_[i] = 0.0f;
    }
    for (int i = 0; i < AUDIO_FFT_LENGTH; i++) {
      vImag_[i] = 0.0f;
    }

    fft_.compute(FFTDirection::Forward);

    for (int i = 0; i < AUDIO_SPECTROGRAM_BINS; i++) {
      magnitude_[i] = sqrtf(vReal_[i] * vReal_[i] + vImag_[i] * vImag_[i]);
    }

    for (int melBin = 0; melBin < AUDIO_MEL_BINS; melBin++) {
      float energy = 0.0f;
      for (int freqBin = 0; freqBin < AUDIO_SPECTROGRAM_BINS; freqBin++) {
        energy += magnitude_[freqBin] * MEL_FILTERBANK[freqBin * AUDIO_MEL_BINS + melBin];
      }
      rawMel_[frameIndex][melBin] = logf(energy + 1e-6f);
    }
  }
};
