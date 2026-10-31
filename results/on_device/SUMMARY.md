# On-device measurements (2026-10-29)

Device: M5StickC Plus2 (ESP32-PICO-V3-02), COM5, firmware built from the new-split models
(`models/`, exported on CPU). Raw serial logs in this folder (`<variant>.log`,
first column = seconds since reset).

## Per variant

| Variant | Networks | Model size (int8) | Tensor arena used / reserved | Inference per window | Full cycle (capture + features + inference) |
|---|---|---|---|---|---|
| Early fusion (small) | 1 | 29.9 KB | 22.3 / 40.0 KB | 1417 ms | 3544 ms (capture 2097 ms, features 30 ms) |
| Decision fusion (small, per-class) | 2 | 23.5 + 14.2 KB | 15.9 / 20.0 KB (IMU), 18.1 / 20.0 KB (audio) | 1066 ms | ~3190 ms (from log spacing) |
| Embedding fusion (small) | 3 | 22.2 + 12.9 + 11.1 KB | 15.6 / 20.0, 17.8 / 20.0, 0.9 / 3.0 KB | 1068 ms | ~3190 ms (from log spacing) |

Firmware image (PlatformIO build): RAM 118-122 KB of 320 KB (36-37 %), flash 691-708 KB of
1280 KB (53-54 %).

Timing was identical in every cycle (+-2 ms), since the reference kernels do a fixed amount of
work per window. The decision/embedding firmware only prints the invoke time; their cycle time
is the spacing between consecutive predictions in the log.

## Self-test (one labelled validation window per class, embedded in firmware)

| Variant | Correct vs. label | Agreement with Python int8 reference |
|---|---|---|
| Early fusion | 6/9 | 9/9 |
| Decision fusion | 7/9 | 9/9 |
| Embedding fusion | 7/9 | 9/9 |

The device reproduces the Python int8 predictions exactly, including the misclassified windows
(checked by running the same quantized vectors through `tf.lite.Interpreter`). The self-test
windows are the first validation window of each class, which often lie at an activity boundary,
so the "correct vs. label" figure is not an accuracy estimate. Accuracy on the full validation
set (float vs. int8) is in `models/*_quantization_report.json`:
early fusion 93.2 / 93.2 %, decision fusion 90.2 / 89.1 %, embedding fusion 90.0 / 89.2 %.

Note: during the captures above the device lay still on the desk (not worn), so the logged
live predictions carry no information about recognition accuracy. Only timing, memory and the
self-test from these logs are valid results.

## Live tests (performed and reported by the author, not logged)

No labelled live recording was made, so live accuracy is not quantified.

- Normal wearing orientation (as in the training data): activities performed individually
  while wearing the device (walking, hammering, carrying a container, and others) were
  recognised. The delay between predictions was noticeable (one prediction every
  ~3.2-3.5 s: 2 s capture + ~1.1-1.4 s inference).
- Orientation test, embedding-fusion firmware (the variant flashed last): the device was
  still worn on top of the wrist, but with the strap direction reversed, i.e. the device
  turned by 180 degrees on the back of the wrist (horizontal IMU axes inverted relative to the
  training data; vertical axis unchanged). Activities performed: carry_container, hammering,
  walk, idle. Only hammering was still recognised correctly; the other three were mostly
  misclassified.
- Interpretation for the thesis: hammering has the most distinctive sound, so the acoustic
  branch (independent of device orientation) can carry the decision; the other activities
  rely mainly on the inertial pattern, which changes sign under the rotation. Training
  augmentation only covers rotations up to +-15 degrees, far from 180 degrees. Links to the
  orientation problem stated in Chapter 1 (problem statement) and to RQ4; belongs in
  Chapter 5 (observation) and Chapter 6 (limitation, remedies: larger-angle rotation
  augmentation, orientation-invariant features such as acceleration magnitude, recording in
  several orientations, calibration at start-up).
