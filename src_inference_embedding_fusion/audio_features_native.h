#pragma once

#include <Arduino.h>
#include <M5StickCPlus2.h>
#include <arduinoFFT.h>
#include <math.h>

#include "model_settings.h"

// Same STFT -> mel-filterbank pipeline as src_inference/audio_features.h (streamed
// frame-by-frame to fit the ESP32's DRAM budget -- see that file's header comment), but
// outputs the *native* (AUDIO_RAW_FRAMES x AUDIO_MEL_BINS) grid directly instead of
// resampling onto a shared 200-step time axis: this backbone was trained on the real
// 198-frame spectrogram (train_audio_cnn.py's Conv2D architecture), not the raw
// early-fusion deployment's resampled-to-200 version.
class AudioFeatureExtractor {
 public:
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

  // Writes the normalized + int8-quantized native mel grid straight into the audio
  // model's own input tensor -- no resample, no channel offset, since this model has its
  // own dedicated tensor rather than sharing one with the IMU branch.
  void writeIntoTensor(int8_t *inputData) {
    for (int frameIndex = 0; frameIndex < AUDIO_RAW_FRAMES; frameIndex++) {
      for (int melBin = 0; melBin < AUDIO_MEL_BINS; melBin++) {
        float value = rawMel_[frameIndex][melBin];
        float normalized = (value - AUDIO_BIN_MEAN[melBin]) / AUDIO_BIN_STD[melBin];
        int quantized = (int)lroundf(normalized / AUDIO_INPUT_SCALE) + AUDIO_INPUT_ZERO_POINT;
        quantized = constrain(quantized, -128, 127);
        inputData[frameIndex * AUDIO_MEL_BINS + melBin] = (int8_t)quantized;
      }
    }
  }

  void printDiagnostics() { Serial.printf("Audio raw sample range: [%d, %d]\n", rawMin_, rawMax_); }

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
