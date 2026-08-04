#include <Arduino.h>
#include <M5StickCPlus2.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "audio_features_native.h"
#include "audio_model_data.h"
#include "imu_features.h"
#include "imu_model_data.h"
#include "model_settings.h"
#include "test_vectors.h"

namespace {
tflite::ErrorReporter *errorReporter = nullptr;

const tflite::Model *imuModel = nullptr;
tflite::MicroInterpreter *imuInterpreter = nullptr;
TfLiteTensor *imuInputTensor = nullptr;
TfLiteTensor *imuOutputTensor = nullptr;
constexpr int kImuArenaSize = 20 * 1024;
uint8_t imuArena[kImuArenaSize];

const tflite::Model *audioModel = nullptr;
tflite::MicroInterpreter *audioInterpreter = nullptr;
TfLiteTensor *audioInputTensor = nullptr;
TfLiteTensor *audioOutputTensor = nullptr;
constexpr int kAudioArenaSize = 20 * 1024;
uint8_t audioArena[kAudioArenaSize];

ImuCaptureBuffer imuCapture;
AudioFeatureExtractor audioFeatureExtractor;
}  // namespace

void haltWithError(const char *message) {
  Serial.println(message);
  StickCP2.Display.fillScreen(BLACK);
  StickCP2.Display.setCursor(0, 10);
  StickCP2.Display.println(message);
  while (true) {
    delay(1000);
  }
}

void registerOps(tflite::MicroMutableOpResolver<9> &resolver) {
  resolver.AddExpandDims();
  resolver.AddConv2D();
  resolver.AddMul();
  resolver.AddAdd();
  resolver.AddReshape();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
}

void setupModels() {
  static tflite::MicroErrorReporter microErrorReporter;
  errorReporter = &microErrorReporter;

  imuModel = tflite::GetModel(g_imu_model_data);
  if (imuModel->version() != TFLITE_SCHEMA_VERSION) {
    haltWithError("IMU model schema mismatch!");
  }
  static tflite::MicroMutableOpResolver<9> imuResolver;
  registerOps(imuResolver);
  static tflite::MicroInterpreter imuStaticInterpreter(imuModel, imuResolver, imuArena, kImuArenaSize, errorReporter);
  imuInterpreter = &imuStaticInterpreter;
  if (imuInterpreter->AllocateTensors() != kTfLiteOk) {
    haltWithError("IMU AllocateTensors failed!");
  }
  imuInputTensor = imuInterpreter->input(0);
  imuOutputTensor = imuInterpreter->output(0);
  Serial.printf("IMU arena used: %u / %d bytes\n", (unsigned)imuInterpreter->arena_used_bytes(), kImuArenaSize);

  audioModel = tflite::GetModel(g_audio_model_data);
  if (audioModel->version() != TFLITE_SCHEMA_VERSION) {
    haltWithError("Audio model schema mismatch!");
  }
  static tflite::MicroMutableOpResolver<9> audioResolver;
  registerOps(audioResolver);
  static tflite::MicroInterpreter audioStaticInterpreter(audioModel, audioResolver, audioArena, kAudioArenaSize,
                                                           errorReporter);
  audioInterpreter = &audioStaticInterpreter;
  if (audioInterpreter->AllocateTensors() != kTfLiteOk) {
    haltWithError("Audio AllocateTensors failed!");
  }
  audioInputTensor = audioInterpreter->input(0);
  audioOutputTensor = audioInterpreter->output(0);
  Serial.printf("Audio arena used: %u / %d bytes\n", (unsigned)audioInterpreter->arena_used_bytes(), kAudioArenaSize);
}

// Runs one model and dequantizes its softmax output into probsOut[0..NUM_CLASSES-1],
// using that model's own (independently-calibrated) output scale/zero-point.
void runImuClassifier(float *probsOut) {
  imuInterpreter->Invoke();
  for (int i = 0; i < NUM_CLASSES; i++) {
    probsOut[i] = (imuOutputTensor->data.int8[i] - IMU_OUTPUT_ZERO_POINT) * IMU_OUTPUT_SCALE;
  }
}

void runAudioClassifier(float *probsOut) {
  audioInterpreter->Invoke();
  for (int i = 0; i < NUM_CLASSES; i++) {
    probsOut[i] = (audioOutputTensor->data.int8[i] - AUDIO_OUTPUT_ZERO_POINT) * AUDIO_OUTPUT_SCALE;
  }
}

// Weighted-sum decision rule: fused = w*imu_probs + (1-w)*audio_probs, matching
// weighted_fusion_probs in train_decision_fusion_cnn.py, with BEST_IMU_WEIGHT the same
// grid-searched constant train_decision_fusion_small.py found on the shared backbones.
int argmaxWeightedFusion(const float *imuProbs, const float *audioProbs, float *confidenceOut) {
  int bestClassIndex = 0;
  float bestProbability = -1.0f;
  for (int classIndex = 0; classIndex < NUM_CLASSES; classIndex++) {
    float probability = BEST_IMU_WEIGHT * imuProbs[classIndex] + (1.0f - BEST_IMU_WEIGHT) * audioProbs[classIndex];
    if (probability > bestProbability) {
      bestProbability = probability;
      bestClassIndex = classIndex;
    }
  }
  *confidenceOut = bestProbability;
  return bestClassIndex;
}

// Runs real, labeled validation windows (pre-quantized in Python for each classifier's
// own input scale/zero-point) through both models plus the weighted combination, exactly
// like a live classification would, but bypassing IMU/mic capture entirely.
void runSelfTest() {
  Serial.println("\n=== Self-test: known-label vectors, no live capture ===");
  int correct = 0;
  float imuProbs[NUM_CLASSES];
  float audioProbs[NUM_CLASSES];

  for (int i = 0; i < NUM_TEST_VECTORS; i++) {
    memcpy(imuInputTensor->data.int8, TEST_VECTOR_IMU[i], IMU_WINDOW_LENGTH * IMU_CHANNELS * sizeof(int8_t));
    runImuClassifier(imuProbs);

    memcpy(audioInputTensor->data.int8, TEST_VECTOR_AUDIO[i], AUDIO_RAW_FRAMES * AUDIO_MEL_BINS * sizeof(int8_t));
    runAudioClassifier(audioProbs);

    float confidence;
    int bestClassIndex = argmaxWeightedFusion(imuProbs, audioProbs, &confidence);

    bool isCorrect = bestClassIndex == TEST_VECTOR_LABELS[i];
    correct += isCorrect ? 1 : 0;
    Serial.printf("Vector %d: predicted=%s (%.1f%%) true=%s [%s]\n", i, CLASS_NAMES[bestClassIndex],
                  confidence * 100.0f, CLASS_NAMES[TEST_VECTOR_LABELS[i]], isCorrect ? "OK" : "WRONG");
  }
  Serial.printf("Self-test: %d/%d correct\n\n", correct, NUM_TEST_VECTORS);

  StickCP2.Display.fillScreen(BLACK);
  StickCP2.Display.setCursor(0, 10);
  StickCP2.Display.setTextSize(2);
  StickCP2.Display.printf("Self-test\n%d/%d\n", correct, NUM_TEST_VECTORS);
  StickCP2.Display.setTextSize(1);
  delay(3000);
}

void captureAndClassify() {
  imuCapture.sampleCount = 0;

  pollImuSample(imuCapture);
  bool audioOk = audioFeatureExtractor.beginCapture();
  pollImuSample(imuCapture);

  for (int frameIndex = 1; audioOk && frameIndex < AUDIO_RAW_FRAMES; frameIndex++) {
    pollImuSample(imuCapture);
    audioOk = audioFeatureExtractor.captureNextFrame(frameIndex);
    pollImuSample(imuCapture);
  }

  Serial.printf("IMU samples captured: %d (cap: %d)\n", imuCapture.sampleCount, IMU_MAX_RAW_SAMPLES);
  audioFeatureExtractor.printDiagnostics();

  if (!audioOk) {
    Serial.println("Skipping this cycle due to mic capture failure.");
    return;
  }

  meanBinImuSamples(imuCapture.samples, imuCapture.sampleCount, imuCapture.samples);
  resampleAndWriteImuWindow(imuCapture.samples, IMU_DOWNSAMPLE_BIN_COUNT, imuInputTensor->data.int8);
  audioFeatureExtractor.writeIntoTensor(audioInputTensor->data.int8);

  unsigned long invokeStartMs = millis();
  float imuProbs[NUM_CLASSES];
  float audioProbs[NUM_CLASSES];
  runImuClassifier(imuProbs);
  runAudioClassifier(audioProbs);

  float confidence;
  int bestClassIndex = argmaxWeightedFusion(imuProbs, audioProbs, &confidence);
  Serial.printf("Timing: invoke=%lums\n", millis() - invokeStartMs);

  Serial.printf("Predicted: %s (%.1f%%)\n", CLASS_NAMES[bestClassIndex], confidence * 100.0f);

  StickCP2.Display.fillScreen(BLACK);
  StickCP2.Display.setCursor(0, 10);
  StickCP2.Display.setTextSize(2);
  StickCP2.Display.println(CLASS_NAMES[bestClassIndex]);
  StickCP2.Display.setTextSize(1);
  StickCP2.Display.printf("%.1f%%\n", confidence * 100.0f);
}

void setup() {
  auto cfg = M5.config();
  StickCP2.begin(cfg);
  Serial.begin(115200);

  StickCP2.Display.setRotation(1);
  StickCP2.Display.setTextSize(2);
  StickCP2.Display.println("Loading models...");

  auto micCfg = StickCP2.Mic.config();
  micCfg.sample_rate = AUDIO_SAMPLE_RATE;
  StickCP2.Mic.config(micCfg);
  StickCP2.Mic.begin();

  setupModels();
  runSelfTest();

  StickCP2.Display.fillScreen(BLACK);
  StickCP2.Display.setCursor(0, 10);
  StickCP2.Display.println("Ready");
}

void loop() { captureAndClassify(); }
