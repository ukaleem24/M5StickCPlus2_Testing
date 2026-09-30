import argparse
import importlib
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from data_split import grouped_train_val_split, parse_window_timestamp

FIRMWARE_DIR = Path(__file__).resolve().parent.parent / "src_inference"

# All of these are resolved in main() once --model is known, since which training
# script's saved artifacts we're exporting is a runtime choice (e.g. the full-size
# early_fusion_concat vs. the fewer-filters early_fusion_concat_small), not something
# fixed at import time. Every function below reads them as module globals.
fusion_model_pipeline = None
audio_pipeline = None
imu_pipeline = None
EXPORT_DIR = None
MODEL_PATH = None
EXPORT_DATA_PATH = None
AUDIO_RAW_FRAMES = None
AUDIO_SPECTROGRAM_BINS = None


def configure_for_model(model_name: str):
    """Imports train_<model_name>_cnn.py and resolves every path/constant the rest of
    this script needs from it, matching the naming convention each training script
    follows (e.g. early_fusion_concat -> train_early_fusion_concat_cnn.py, saving
    models/early_fusion_concat.keras)."""
    global fusion_model_pipeline, audio_pipeline, imu_pipeline, EXPORT_DIR
    global MODEL_PATH, EXPORT_DATA_PATH, AUDIO_RAW_FRAMES, AUDIO_SPECTROGRAM_BINS

    fusion_model_pipeline = importlib.import_module(f"train_{model_name}_cnn")
    audio_pipeline = fusion_model_pipeline.audio_pipeline
    imu_pipeline = fusion_model_pipeline.imu_pipeline

    EXPORT_DIR = fusion_model_pipeline.EXPORT_DIR
    MODEL_PATH = EXPORT_DIR / f"{model_name}.keras"
    EXPORT_DATA_PATH = EXPORT_DIR / f"{model_name}_export_data.npz"

    # Number of raw STFT frames before resampling onto the shared 200-step time axis --
    # same arithmetic tf.signal.stft uses internally.
    AUDIO_RAW_FRAMES = (audio_pipeline.TARGET_SAMPLES - audio_pipeline.FRAME_LENGTH) // audio_pipeline.FRAME_STEP + 1
    AUDIO_SPECTROGRAM_BINS = audio_pipeline.FFT_LENGTH // 2 + 1


def build_representative_dataset(representative_samples: np.ndarray):
    def representative_dataset():
        for sample in representative_samples:
            yield [sample[np.newaxis, ...].astype(np.float32)]

    return representative_dataset


def convert_to_int8_tflite(model: tf.keras.Model, representative_samples: np.ndarray) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = build_representative_dataset(representative_samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def get_io_quantization(tflite_model: bytes) -> dict:
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]
    return {
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zero_point),
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zero_point),
    }


def compute_mel_filterbank() -> np.ndarray:
    # Fixed, data-independent matrix -- precomputing it here and embedding it means the
    # firmware only needs an FFT + matmul, guaranteeing exact parity with the mel
    # spectrogram computed at training time instead of re-deriving the mel-scale formula.
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=audio_pipeline.NUM_MEL_BINS,
        num_spectrogram_bins=AUDIO_SPECTROGRAM_BINS,
        sample_rate=audio_pipeline.SAMPLE_RATE,
        lower_edge_hertz=audio_pipeline.FMIN_HZ,
        upper_edge_hertz=audio_pipeline.FMAX_HZ,
    )
    return mel_weight_matrix.numpy()


def compute_hann_window() -> np.ndarray:
    # tf.signal.stft applies a periodic Hann window of length frame_length to each frame
    # before zero-padding to fft_length -- this is TF's default window_fn, not something
    # train_audio_cnn.py opts into explicitly, so it's easy to miss when reimplementing
    # the STFT on-device. Precompute it here so the firmware applies the identical window.
    return tf.signal.hann_window(audio_pipeline.FRAME_LENGTH, periodic=True).numpy()


def format_c_array(name: str, values, dtype: str, values_per_line: int = 12) -> str:
    flat = np.asarray(values).flatten()
    if dtype == "float":
        # A bare "0" needs a decimal point to be a valid C++ float literal ("0f" is not
        # legal -- the compiler parses it as an integer with an unknown literal suffix).
        formatted = []
        for v in flat:
            literal = f"{v:.8g}"
            if "." not in literal and "e" not in literal and "E" not in literal:
                literal += ".0"
            formatted.append(f"{literal}f")
    else:
        formatted = [str(int(v)) for v in flat]

    lines = []
    for i in range(0, len(formatted), values_per_line):
        chunk = formatted[i : i + values_per_line]
        lines.append("    " + ", ".join(chunk) + ",")
    body = "\n".join(lines)
    return f"const {dtype} {name}[{len(flat)}] = {{\n{body}\n}};\n"


def write_model_data_header(tflite_model: bytes, output_path: Path, variable_name: str = "g_model_data"):
    # variable_name matters once a firmware needs more than one model header included in
    # the same translation unit (e.g. decision/embedding fusion's separate IMU and audio
    # models) -- the default keeps existing single-model callers unchanged.
    lines = []
    for i in range(0, len(tflite_model), 12):
        chunk = tflite_model[i : i + 12]
        lines.append("    " + ", ".join(str(b) for b in chunk) + ",")
    body = "\n".join(lines)

    content = (
        "// Auto-generated by export_tflite_model.py -- do not edit by hand.\n"
        "#pragma once\n\n"
        f"alignas(8) const unsigned char {variable_name}[] = {{\n"
        f"{body}\n"
        "};\n"
        f"const int {variable_name}_len = {len(tflite_model)};\n"
    )
    output_path.write_text(content)


def write_model_settings_header(
    output_path: Path,
    classes,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    mel_filterbank: np.ndarray,
    hann_window: np.ndarray,
    quantization: dict,
):
    class_name_array = ",\n    ".join(f'"{name}"' for name in classes)

    content = f"""// Auto-generated by export_tflite_model.py -- do not edit by hand.
#pragma once

// --- Class labels (index order matches the model's output tensor) ---
constexpr int NUM_CLASSES = {len(classes)};
const char *const CLASS_NAMES[NUM_CLASSES] = {{
    {class_name_array}
}};

// --- Window / feature dimensions ---
constexpr int FUSED_WINDOW_LENGTH = {fusion_model_pipeline.FUSED_WINDOW_LENGTH};
constexpr int IMU_CHANNELS = {len(imu_pipeline.CHANNEL_COLUMNS)};
constexpr int AUDIO_MEL_BINS = {audio_pipeline.NUM_MEL_BINS};
constexpr int FUSED_CHANNELS = IMU_CHANNELS + AUDIO_MEL_BINS;

// --- Audio capture / STFT parameters (must match train_audio_cnn.py exactly) ---
constexpr int AUDIO_SAMPLE_RATE = {audio_pipeline.SAMPLE_RATE};
constexpr int AUDIO_TARGET_SAMPLES = {audio_pipeline.TARGET_SAMPLES};
constexpr int AUDIO_FRAME_LENGTH = {audio_pipeline.FRAME_LENGTH};
constexpr int AUDIO_FRAME_STEP = {audio_pipeline.FRAME_STEP};
constexpr int AUDIO_FFT_LENGTH = {audio_pipeline.FFT_LENGTH};
constexpr int AUDIO_SPECTROGRAM_BINS = {AUDIO_SPECTROGRAM_BINS};
constexpr int AUDIO_RAW_FRAMES = {AUDIO_RAW_FRAMES};

// --- TFLite int8 quantization parameters ---
constexpr float INPUT_SCALE = {quantization["input_scale"]:.10g}f;
constexpr int INPUT_ZERO_POINT = {quantization["input_zero_point"]};
constexpr float OUTPUT_SCALE = {quantization["output_scale"]:.10g}f;
constexpr int OUTPUT_ZERO_POINT = {quantization["output_zero_point"]};

// --- Per-channel normalization stats from the training split (order: {imu_pipeline.CHANNEL_COLUMNS}
// followed by {audio_pipeline.NUM_MEL_BINS} mel bins, matching the concatenation order in
// train_early_fusion_concat_cnn.py) ---
{format_c_array("CHANNEL_MEAN", channel_mean, "float")}
{format_c_array("CHANNEL_STD", channel_std, "float")}

// --- Precomputed mel filterbank matrix, shape [AUDIO_SPECTROGRAM_BINS][AUDIO_MEL_BINS],
// row-major flattened. Identical to tf.signal.linear_to_mel_weight_matrix at training time. ---
{format_c_array("MEL_FILTERBANK", mel_filterbank, "float")}

// --- Periodic Hann window (length AUDIO_FRAME_LENGTH), applied to each frame before the
// FFT -- tf.signal.stft's default window_fn, must be replicated exactly on-device. ---
{format_c_array("HANN_WINDOW", hann_window, "float")}
"""
    output_path.write_text(content)


def build_self_test_vectors(channel_mean, channel_std, quantization, classes, samples_per_class=1):
    """Re-derives the same val split train_early_fusion_concat_cnn.py used, and quantizes
    a handful of real labeled windows exactly as the device pipeline would. These get
    embedded in the firmware so it can run inference on known-good, known-label input at
    boot -- if the device gets these right but live capture still doesn't work, the bug is
    in the capture/feature pipeline, not the TFLite Micro runtime itself. That's the one
    thing this export script can't verify from the Python side."""
    imu_paths, audio_paths, labels = fusion_model_pipeline.discover_paired_windows(
        imu_pipeline.DATA_ROOT, audio_pipeline.DATA_ROOT
    )
    x_imu = np.stack([fusion_model_pipeline.load_imu_window(p) for p in imu_paths])
    x_audio_waveforms = np.stack([fusion_model_pipeline.load_audio_waveform(p) for p in audio_paths])
    x_audio_spec = audio_pipeline.compute_log_mel_spectrograms(x_audio_waveforms)[..., 0]
    x_audio_aligned = np.stack(
        [imu_pipeline.resample_window(s, fusion_model_pipeline.FUSED_WINDOW_LENGTH) for s in x_audio_spec]
    )
    x_fused = np.concatenate([x_imu, x_audio_aligned], axis=-1)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.asarray(classes)
    y_encoded = label_encoder.transform(labels)

    timestamps = np.array([parse_window_timestamp(p) for p in imu_paths])
    _, val_idx = grouped_train_val_split(labels, timestamps, test_size=0.2, random_state=42)

    chosen_indices = []
    seen_classes = set()
    for idx in val_idx:
        class_index = int(y_encoded[idx])
        if class_index in seen_classes:
            continue
        seen_classes.add(class_index)
        chosen_indices.append(idx)
        if len(seen_classes) >= len(classes) * samples_per_class:
            break

    vectors = []
    for idx in chosen_indices:
        normalized = (x_fused[idx] - channel_mean) / channel_std
        quantized = np.round(normalized / quantization["input_scale"] + quantization["input_zero_point"])
        quantized = np.clip(quantized, -128, 127).astype(np.int8)
        vectors.append((quantized, int(y_encoded[idx])))
    return vectors


def write_test_vectors_header(output_path: Path, vectors):
    blocks = []
    for i, (quantized, label) in enumerate(vectors):
        flat = quantized.flatten()
        lines = []
        for j in range(0, len(flat), 20):
            lines.append("    " + ", ".join(str(int(v)) for v in flat[j : j + 20]) + ",")
        body = "\n".join(lines)
        blocks.append(
            f"const int8_t TEST_VECTOR_{i}_DATA[{len(flat)}] = {{\n{body}\n}};\nconst int TEST_VECTOR_{i}_LABEL = {label};\n"
        )

    pointer_list = ",\n    ".join(f"TEST_VECTOR_{i}_DATA" for i in range(len(vectors)))
    label_list = ", ".join(str(label) for _, label in vectors)

    content = f"""// Auto-generated by export_tflite_model.py -- do not edit by hand.
// Real, labeled validation windows, pre-normalized and quantized exactly like the live
// capture pipeline would. Used by runSelfTest() in main.cpp to check the TFLite Micro
// runtime in isolation from live sensor capture -- see build_self_test_vectors's comment.
#pragma once

#include <cstdint>

constexpr int NUM_TEST_VECTORS = {len(vectors)};

{"".join(blocks)}
const int8_t *const TEST_VECTORS[NUM_TEST_VECTORS] = {{
    {pointer_list}
}};
const int TEST_VECTOR_LABELS[NUM_TEST_VECTORS] = {{{label_list}}};
"""
    output_path.write_text(content)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a trained early-fusion model to int8 TFLite and generate firmware headers."
    )
    parser.add_argument(
        "--model",
        choices=["early_fusion_concat", "early_fusion_concat_small"],
        default="early_fusion_concat",
        help="Which trained model to export (default: %(default)s). Must have already been "
        "trained via the matching train_<model>_cnn.py script.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configure_for_model(args.model)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found -- run train_{args.model}_cnn.py first to produce it.")

    model = tf.keras.models.load_model(MODEL_PATH)
    export_data = np.load(EXPORT_DATA_PATH, allow_pickle=True)

    channel_mean = export_data["channel_mean"]
    channel_std = export_data["channel_std"]
    classes = export_data["classes"]
    representative_samples = export_data["representative_samples"]

    print("Converting to int8 TFLite...")
    tflite_model = convert_to_int8_tflite(model, representative_samples)
    quantization = get_io_quantization(tflite_model)
    print(f"Quantized model size: {len(tflite_model)} bytes")
    print(f"Input quantization:  scale={quantization['input_scale']}, zero_point={quantization['input_zero_point']}")
    print(f"Output quantization: scale={quantization['output_scale']}, zero_point={quantization['output_zero_point']}")

    mel_filterbank = compute_mel_filterbank()
    hann_window = compute_hann_window()

    FIRMWARE_DIR.mkdir(exist_ok=True)

    tflite_path = EXPORT_DIR / f"{args.model}_int8.tflite"
    tflite_path.write_bytes(tflite_model)
    print(f"Saved raw TFLite model to {tflite_path}")

    write_model_data_header(tflite_model, FIRMWARE_DIR / "model_data.h")
    write_model_settings_header(
        FIRMWARE_DIR / "model_settings.h",
        classes,
        channel_mean,
        channel_std,
        mel_filterbank,
        hann_window,
        quantization,
    )
    print(f"Wrote {FIRMWARE_DIR / 'model_data.h'} and {FIRMWARE_DIR / 'model_settings.h'}")

    print("Building self-test vectors from real labeled validation windows...")
    test_vectors = build_self_test_vectors(channel_mean, channel_std, quantization, classes)
    write_test_vectors_header(FIRMWARE_DIR / "test_vectors.h", test_vectors)
    print(f"Wrote {FIRMWARE_DIR / 'test_vectors.h'} ({len(test_vectors)} vectors)")


if __name__ == "__main__":
    main()
