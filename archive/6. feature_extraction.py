import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from pathlib import Path

def extract_features(df, axes=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']):
    features = {}
    
    for axis in axes:
        if axis not in df.columns:
            continue
            
        data = df[axis].values
        
        # Time-Domain Features
        features[f'{axis}_mean'] = np.mean(data)
        features[f'{axis}_std'] = np.std(data)
        features[f'{axis}_max'] = np.max(data)
        features[f'{axis}_min'] = np.min(data)
        features[f'{axis}_skewness'] = skew(data) if len(data) > 0 else 0
        features[f'{axis}_kurtosis'] = kurtosis(data) if len(data) > 0 else 0
        
        # Frequency-Domain Features
        # Compute FFT
        fft_values = np.fft.fft(data)
        fft_magnitudes = np.abs(fft_values)
        
        # Spectral Energy (Sum of squared magnitudes / N)
        spectral_energy = np.sum(fft_magnitudes ** 2) / len(data)
        features[f'{axis}_spectral_energy'] = spectral_energy
        
        # FFT Peak (excluding DC component at index 0)
        if len(fft_magnitudes) > 1:
            peak_magnitude = np.max(fft_magnitudes[1:])
        else:
            peak_magnitude = 0
        features[f'{axis}_fft_peak'] = peak_magnitude
        
    return features

def main():
    # Base directory
    base_dir = Path(r"..\Data Thesis\Florian First 3 wrists ankle waist")
    
    if not base_dir.exists():
        # Fallback to absolute path if relative fails
        base_dir = Path(r"c:\Users\pag-ku\Documents\PlatformIO\Projects\M5StickCPlus2_Testing\Data Thesis\Florian First 3 wrists ankle waist")
        
    if not base_dir.exists():
        print(f"Error: Directory not found at {base_dir}")
        return

    # Find all IMU csv files (in subdirectories like left_hand/imu/hammering)
    search_pattern = str(base_dir / "**" / "imu" / "**" / "*.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(csv_files)} IMU CSV files. Processing...")
    
    dataset_features = []
    
    for i, file_path in enumerate(csv_files):
        try:
            df = pd.read_csv(file_path)
            
            # Check if dataframe has any data
            if df.empty:
                continue
                
            # Extract features
            file_features = extract_features(df)
            
            # Add metadata
            path_parts = Path(file_path).parts
            # Extracting label and sensor location from the path structure
            # Example path: ...\left_hand\imu\hammering\imu_hammering_...csv
            label = path_parts[-2] # e.g., hammering
            sensor_location = path_parts[-4] # e.g., left_hand
            
            file_features['label'] = label
            file_features['sensor_location'] = sensor_location
            
            dataset_features.append(file_features)
            
            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{len(csv_files)} files...")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    if dataset_features:
        # Convert to DataFrame
        features_df = pd.DataFrame(dataset_features)
        
        # Reorder columns to put metadata first
        cols = features_df.columns.tolist()
        metadata_cols = ['sensor_location', 'label']
        feature_cols = [c for c in cols if c not in metadata_cols]
        features_df = features_df[metadata_cols + feature_cols]
        
        # Save to CSV
        output_file = Path(__file__).parent / "extracted_features.csv"
        features_df.to_csv(output_file, index=False)
        print(f"Successfully extracted features from {len(dataset_features)} files.")
        print(f"Features saved to {output_file}")
    else:
        print("No valid data found to extract features from.")

if __name__ == "__main__":
    main()
