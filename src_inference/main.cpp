#include <Arduino.h>
#include <M5StickCPlus2.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "audio_features.h"
#include "imu_features.h"
#include "model_data.h"
#include "model_settings.h"
#include "test_vectors.h"

namespace {
tflite::ErrorReporter *errorReporter = nullptr;
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *inputTensor = nullptr;
TfLiteTensor *outputTensor = nullptr;

// The ESP32's DRAM budget for static globals is tight (~160KB total, shared with the
// framework/WiFi/BT stack and every other global below) -- this is sized to fit
// alongside those, not by the model's own requirement (~65KB weights, which live in
// flash, not this arena). Actual usage is printed at startup via
// interpreter->arena_used_bytes() to tune this further on real hardware.
constexpr int kTensorArenaSize = 40 * 1024;
uint8_t tensorArena[kTensorArenaSize];

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

void setupModel() {
  static tflite::MicroErrorReporter microErrorReporter;
  errorReporter = &microErrorReporter;

  model = tflite::GetModel(g_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    haltWithError("Model schema mismatch!");
  }

  // Explicit op list instead of AllOpsResolver (~130 builtin ops) -- this model only
  // uses these 9 (confirmed by inspecting the .tflite directly), and avoiding the much
  // larger op table cuts per-node dispatch lookup overhead in Invoke(). Unlikely to be
  // the dominant cost (the real bottleneck is almost certainly the CONV_2D kernel itself
  // being unoptimized int8 reference code in this older TFLite Micro port -- no ESP-NN
  // hardware acceleration here), but it's free and worth ruling in or out.
  static tflite::MicroMutableOpResolver<9> resolver;
  resolver.AddExpandDims();
  resolver.AddConv2D();
  resolver.AddMul();
  resolver.AddAdd();
  resolver.AddReshape();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();

  static tflite::MicroInterpreter staticInterpreter(model, resolver, tensorArena, kTensorArenaSize, errorReporter);
  interpreter = &staticInterpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    haltWithError("AllocateTensors failed!");
  }

  inputTensor = interpreter->input(0);
  outputTensor = interpreter->output(0);

  Serial.printf("Tensor arena used: %u / %d bytes\n", (unsigned)interpreter->arena_used_bytes(), kTensorArenaSize);
}

// Runs inference on real, labeled validation windows that were normalized and quantized
// in Python exactly like the live capture pipeline is supposed to -- isolates whether the
// TFLite Micro runtime itself classifies correctly from whether live sensor capture is
// producing correct input. If this reports high accuracy but live predictions are still
// bad, the bug is in capture/feature-extraction; if this ALSO fails, the runtime itself
// (op kernels, quantization handling, arena sizing) is the problem, since these vectors
// bypass IMU/mic capture entirely and write straight into the input tensor.
void runSelfTest() {
  Serial.println("\n=== Self-test: known-label vectors, no live capture ===");
  int correct = 0;
  for (int i = 0; i < NUM_TEST_VECTORS; i++) {
    memcpy(inputTensor->data.int8, TEST_VECTORS[i], FUSED_WINDOW_LENGTH * FUSED_CHANNELS * sizeof(int8_t));

    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.printf("Vector %d: Invoke failed!\n", i);
      continue;
    }

    int bestClassIndex = 0;
    float bestProbability = -1.0f;
    for (int classIndex = 0; classIndex < NUM_CLASSES; classIndex++) {
      float probability = (outputTensor->data.int8[classIndex] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
      if (probability > bestProbability) {
        bestProbability = probability;
        bestClassIndex = classIndex;
      }
    }

    bool isCorrect = bestClassIndex == TEST_VECTOR_LABELS[i];
    correct += isCorrect ? 1 : 0;
    Serial.printf("Vector %d: predicted=%s (%.1f%%) true=%s [%s]\n", i, CLASS_NAMES[bestClassIndex],
                  bestProbability * 100.0f, CLASS_NAMES[TEST_VECTOR_LABELS[i]], isCorrect ? "OK" : "WRONG");
  }
  Serial.printf("Self-test: %d/%d correct\n\n", correct, NUM_TEST_VECTORS);

  StickCP2.Display.fillScreen(BLACK);
  StickCP2.Display.setCursor(0, 10);
  StickCP2.Display.setTextSize(2);
  StickCP2.Display.printf("Self-test\n%d/%d\n", correct, NUM_TEST_VECTORS);
  StickCP2.Display.setTextSize(1);
  delay(3000);
}

// Captures both modalities INTERLEAVED -- IMU is polled (non-blocking) right around
// each audio chunk capture, so both streams span the same ~2s of real time instead of
// running as two sequential ~2s phases. The model was trained on IMU and audio paired
// by matching timestamp (see discover_paired_windows in train_fusion_cnn.py); feeding
// it two unrelated moments in time glued together at matching array positions is a much
// bigger distribution mismatch than any per-modality preprocessing detail. Writes the
// concatenated, normalized, int8-quantized window straight into the model's input
// tensor (channel order: IMU_CHANNELS then AUDIO_MEL_BINS, matching
// train_early_fusion_concat_cnn.py's np.concatenate order).
void captureAndClassify() {
  unsigned long cycleStartMs = millis();
  imuCapture.sampleCount = 0;

  pollImuSample(imuCapture);
  bool audioOk = audioFeatureExtractor.beginCapture();
  pollImuSample(imuCapture);

  for (int frameIndex = 1; audioOk && frameIndex < AUDIO_RAW_FRAMES; frameIndex++) {
    pollImuSample(imuCapture);
    audioOk = audioFeatureExtractor.captureNextFrame(frameIndex);
    pollImuSample(imuCapture);
  }
  unsigned long captureDoneMs = millis();

  // Diagnostic: at the ~250 Hz this hardware actually runs at, expect somewhat under the
  // ~500 samples/2s a dedicated capture would get, since polling here is opportunistic
  // (squeezed around the audio chunk reads) rather than continuous -- but it should be
  // in the hundreds, not near zero, and should never hit IMU_MAX_RAW_SAMPLES.
  Serial.printf("IMU samples captured: %d (cap: %d)\n", imuCapture.sampleCount, IMU_MAX_RAW_SAMPLES);

  // Target for a still/idle hold, from the actual training data (data/windowed_data/
  // right_hand_dominant/imu/idle, 45 windows): accel mean ~= (-0.19, -0.57, 0.77).
  // Match all three axes simultaneously -- a single-axis match isn't enough, since it's
  // the full 3D gravity direction that encodes device orientation.
  float accelSum[3] = {0.0f, 0.0f, 0.0f};
  for (int i = 0; i < imuCapture.sampleCount; i++) {
    accelSum[0] += imuCapture.samples[i * IMU_CHANNELS + 0];
    accelSum[1] += imuCapture.samples[i * IMU_CHANNELS + 1];
    accelSum[2] += imuCapture.samples[i * IMU_CHANNELS + 2];
  }
  int n = max(imuCapture.sampleCount, 1);
  Serial.printf("IMU accel mean (x,y,z): (%.4f, %.4f, %.4f)  -- target idle ~= (-0.19, -0.57, 0.77)\n",
                accelSum[0] / n, accelSum[1] / n, accelSum[2] / n);
  audioFeatureExtractor.printDiagnostics();

  if (!audioOk) {
    Serial.println("Skipping this cycle due to mic capture failure.");
    return;
  }

  // In-place: see meanBinImuSamples's comment for why this doesn't corrupt unread data.
  meanBinImuSamples(imuCapture.samples, imuCapture.sampleCount, imuCapture.samples);
  resampleAndWriteImuWindow(imuCapture.samples, IMU_DOWNSAMPLE_BIN_COUNT, inputTensor->data.int8);
  audioFeatureExtractor.writeResampledIntoTensor(inputTensor->data.int8, IMU_CHANNELS);

  unsigned long featuresDoneMs = millis();
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }
  unsigned long invokeDoneMs = millis();
  Serial.printf("Timing: capture=%lums features=%lums invoke=%lums total=%lums\n",
                captureDoneMs - cycleStartMs, featuresDoneMs - captureDoneMs, invokeDoneMs - featuresDoneMs,
                invokeDoneMs - cycleStartMs);

  int bestClassIndex = 0;
  float bestProbability = -1.0f;
  for (int classIndex = 0; classIndex < NUM_CLASSES; classIndex++) {
    int8_t quantizedOutput = outputTensor->data.int8[classIndex];
    float probability = (quantizedOutput - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
    if (probability > bestProbability) {
      bestProbability = probability;
      bestClassIndex = classIndex;
    }
  }

  Serial.printf("Predicted: %s (%.1f%%)\n", CLASS_NAMES[bestClassIndex], bestProbability * 100.0f);

  StickCP2.Display.fillScreen(BLACK);
  StickCP2.Display.setCursor(0, 10);
  StickCP2.Display.setTextSize(2);
  StickCP2.Display.println(CLASS_NAMES[bestClassIndex]);
  StickCP2.Display.setTextSize(1);
  StickCP2.Display.printf("%.1f%%\n", bestProbability * 100.0f);
}

void setup() {
  auto cfg = M5.config();
  StickCP2.begin(cfg);
  Serial.begin(115200);

  StickCP2.Display.setRotation(1);
  StickCP2.Display.setTextSize(2);
  StickCP2.Display.println("Loading model...");

  // Note: unlike src/main.cpp's data-collection firmware, this environment resolves a
  // newer M5Unified (0.2.19) whose IMU_Class dropped setODR() entirely -- there's no
  // equivalent call to pin the ODR here. StickCP2.begin() above already initializes the
  // IMU at the driver's default rate (measured ~250 Hz); meanBinImuSamples +
  // resampleAndWriteImuWindow (imu_features.h) reproduce the same mean-bin-then-resample
  // pipeline the training data went through regardless of the exact live rate.

  auto micCfg = StickCP2.Mic.config();
  micCfg.sample_rate = AUDIO_SAMPLE_RATE;
  StickCP2.Mic.config(micCfg);
  StickCP2.Mic.begin();

  setupModel();
  runSelfTest();

  StickCP2.Display.fillScreen(BLACK);
  StickCP2.Display.setCursor(0, 10);
  StickCP2.Display.println("Ready");
}

void loop() {
  captureAndClassify();
}
