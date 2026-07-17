import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight  # <-- Added for class imbalance
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, Reshape, Permute, Flatten
from tensorflow.keras.optimizers import Adam

# --- DSP Configuration ---
FS = 80.0  # Sampling frequency in Hz
CUTOFF = 25.0  # Cut-off frequency in Hz
FILTER_ORDER = 4
FFT_LENGTH = 128
AXES = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
FEATURES_PER_AXIS = 45


def edge_impulse_rms(data):
    return np.sqrt(np.mean(np.square(data)))


def edge_impulse_skew(data):
    centered = data - np.mean(data)
    stddev = edge_impulse_rms(centered)
    if stddev == 0.0:
        stddev = 1e-10
    return np.mean(centered ** 3) / (stddev ** 3)


def edge_impulse_kurtosis(data):
    centered = data - np.mean(data)
    stddev = edge_impulse_rms(centered)
    if stddev == 0.0:
        stddev = 1e-10
    return (np.mean(centered ** 4) / (stddev ** 4)) - 3.0


def edge_impulse_power_spectrum(frame, fft_points):
    fft = np.fft.rfft(frame, n=fft_points)
    magnitude = np.abs(fft)
    return (magnitude ** 2) / float(fft_points)


def edge_impulse_welch_max_hold(axis_data, fft_points=FFT_LENGTH, do_overlap=True, start_bin=1, stop_bin=41):
    output = np.zeros(stop_bin - start_bin, dtype=np.float32)
    input_ix = 0

    while input_ix < len(axis_data):
        end_ix = min(input_ix + fft_points, len(axis_data))
        frame = axis_data[input_ix:end_ix]
        power_spectrum = edge_impulse_power_spectrum(frame, fft_points)
        output = np.maximum(output, power_spectrum[start_bin:stop_bin])

        input_ix += fft_points // 2 if do_overlap else fft_points

    return output


def edge_impulse_start_stop_bin(sampling_freq=FS, fft_length=FFT_LENGTH, filter_cutoff=CUTOFF):
    bin_index = filter_cutoff * fft_length / sampling_freq
    start_bin = 1
    stop_bin = int(bin_index + 0.5) + 1
    return start_bin, stop_bin

def apply_lowpass_filter(data, cutoff, fs, order):
    """Applies a 4th order low-pass Butterworth filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Apply filter along the time axis (axis 0)
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

def extract_spectral_features(window_data):
    """
    Extracts Edge Impulse v4-style spectral features from a 2-second window.
    """
    # 1. Apply the low-pass filter
    filtered_window = apply_lowpass_filter(window_data, CUTOFF, FS, FILTER_ORDER)
    filtered_window = filtered_window - np.mean(filtered_window, axis=0, keepdims=True)

    start_bin, stop_bin = edge_impulse_start_stop_bin()
    
    features = []
    # 2. Extract features per axis
    for i in range(filtered_window.shape[1]):
        axis_data = filtered_window[:, i]
        
        # Time-domain features used by the exported Edge Impulse DSP block.
        features.append(edge_impulse_rms(axis_data))
        features.append(edge_impulse_skew(axis_data))
        features.append(edge_impulse_kurtosis(axis_data))

        # FFT-derived features use max-hold Welch power bins.
        fft_bins = edge_impulse_welch_max_hold(axis_data, FFT_LENGTH, True, start_bin, stop_bin)
        fft_bins = np.where(fft_bins == 0.0, 1e-10, fft_bins)
        features.append(edge_impulse_skew(fft_bins))
        features.append(edge_impulse_kurtosis(fft_bins))
        features.extend(np.log10(fft_bins))
        
    return np.array(features)


def load_and_process_data(base_directory):
    X = []
    y = []
    
    # Iterate through each folder (class label)
    for label in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, label)
        
        if not os.path.isdir(folder_path):
            continue
            
        # Iterate through each 2-second window file in the folder
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                file_path = os.path.join(folder_path, file)
                
                # Load the window (assuming columns match the AXES list)
                df = pd.read_csv(file_path)
                window_data = df[AXES].values
                
                # Extract features
                features = extract_spectral_features(window_data)
                if features.shape[0] != FEATURES_PER_AXIS * len(AXES):
                    raise ValueError(
                        f"Expected {FEATURES_PER_AXIS * len(AXES)} features, got {features.shape[0]} from {file_path}"
                    )
                
                X.append(features)
                y.append(label)
                
    return np.array(X), np.array(y)


base_dir = r".\data\windowed_data\right_hand_dominant\imu"

X_raw, y_raw = load_and_process_data(base_dir)

# Check if data loaded successfully to prevent cryptic numpy errors
if len(y_raw) == 0:
    raise ValueError(f"No .csv files found in the subfolders of {base_dir}!")

# Encode labels to integers, then to one-hot vectors
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
y_categorical = tf.keras.utils.to_categorical(y_encoded)

# Split the data (Validation set size: 20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_raw,
    y_categorical,
    test_size=0.20,
    stratify=y_encoded,
    random_state=42
)

# --- NEW: Standardize/Scale the features ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val) # Note: We only transform validation data to prevent data leakage

# --- Model Architecture ---
model = Sequential()

model.add(Input(shape=(X_train.shape[1],)))

# Reshape into (axes, features_per_axis) and then permute to (features_per_axis, axes)
# so Conv1D learns feature-type interactions across the six IMU channels.
model.add(Reshape((len(AXES), FEATURES_PER_AXIS)))
model.add(Permute((2, 1)))
model.add(Conv1D(32, kernel_size=3, padding="same", activation="relu"))
model.add(Conv1D(64, kernel_size=3, padding="same", activation="relu"))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(
    Dense(
        128,
        activation="relu",
        activity_regularizer=tf.keras.regularizers.l1(1e-5)
    )
)

model.add(Dropout(0.4))

model.add(
    Dense(
        64,
        activation="relu",
        activity_regularizer=tf.keras.regularizers.l1(1e-5)
    )
)

model.add(
    Dense(
        len(label_encoder.classes_),
        activation="softmax",
        name="y_pred"
    )
)

# --- Compilation ---
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999
)
model.compile(optimizer=optimizer, 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

# --- NEW: Compute Class Weights for Imbalanced Data ---
y_integers = np.argmax(y_train, axis=1)
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
class_weights_dict = dict(enumerate(weights))

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_train, y_train))
    .shuffle(32 * 4)
    .batch(32, drop_remainder=False)
)

validation_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_val, y_val))
    .batch(32, drop_remainder=False)
)

# --- Training ---
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=60,
    class_weight=class_weights_dict,  # <-- Added class weights here
    verbose=2
)

# Optional: Evaluate the model
loss, accuracy = model.evaluate(validation_dataset, verbose=0)
print(f"Validation Accuracy: {accuracy*100:.1f}%")