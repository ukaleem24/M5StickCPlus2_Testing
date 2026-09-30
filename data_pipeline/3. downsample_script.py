"""
IMU Downsampling Utility
=========================================================================
Downsamples high-frequency IMU data (e.g., 250Hz) to a lower target 
frequency (e.g., 80Hz) to improve machine learning generalization and 
reduce model memory bloat.

Features:
- Configurable target frequency
- Applies average binning (acts as a low-pass filter to prevent aliasing)
- Preserves majority voting for categorical labels 
- Processes all IMU CSVs in the current directory
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

class DownsampleConfig:
    # MAIN CONTROL VARIABLE: Change this to 50, 60, 80, 100, etc.
    TARGET_FREQ_HZ = 80 
    
    # The folder where the downsampled files will be saved
    OUTPUT_FOLDER = f"downsampled_{TARGET_FREQ_HZ}hz"
    
    @classmethod
    def get_period_us(cls) -> int:
        """Calculate the exact period in microseconds based on frequency."""
        return int(1000000 / cls.TARGET_FREQ_HZ)

# ============================================================================
# DOWNSAMPLING LOGIC
# ============================================================================

def process_imu_file(filepath: Path, output_dir: Path):
    """Load, downsample, and save a single IMU file."""
    try:
        df = pd.read_csv(filepath)
        print(f"\nProcessing: {filepath.name}")
        
        total_duration_sec = df['recv_ts'].max() - df['recv_ts'].min()
        orig_freq = len(df) / total_duration_sec if total_duration_sec > 0 else 0
        print(f"  [INFO] Original Frequency: ~{orig_freq:.2f} Hz | Samples: {len(df)}")
        
        # 1. Convert timestamps to a Datetime Index
        df['datetime'] = pd.to_datetime(df['recv_ts'], unit='s')
        df.set_index('datetime', inplace=True)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        target_period_us = DownsampleConfig.get_period_us()
        freq_str = f"{target_period_us}us" 
        
        # 2. Downsample Numeric Data (Mean) + INTERPOLATE MISSING
        df_numeric = df[numeric_cols].resample(freq_str).mean()
        df_numeric = df_numeric.interpolate(method='linear') # <--- FIX: Fill the sensor gaps
        
        # 3. Downsample Categorical Data (Majority Vote) + FORWARD FILL
        if categorical_cols:
            def get_mode(series):
                # Drop NaNs before calculating mode to avoid errors on empty bins
                m = series.dropna().mode()
                return m.iloc[0] if not m.empty else np.nan
            
            df_categorical = df[categorical_cols].resample(freq_str).agg(get_mode)
            df_categorical = df_categorical.ffill().bfill() # <--- FIX: Fill missing labels
            
            df_resampled = pd.concat([df_numeric, df_categorical], axis=1)
        else:
            df_resampled = df_numeric
            
        # 4. Cleanup (NO DROPNA)
        # Reconstruct the absolute UNIX timestamp from the perfect datetime index
        # This replaces the averaged 'recv_ts' with a mathematically perfect timeline
        df_resampled['recv_ts'] = df_resampled.index.astype('int64') / 10**9
        
        # Restore integer formatting for system sequence IDs
        for col in ['seq', 'raw_ts']:
            if col in df_resampled.columns:
                # Use round and fillna in case interpolation created floats/nans on the edges
                df_resampled[col] = df_resampled[col].round().fillna(0).astype(int)
        
        # Ensure column order perfectly matches the original file
        final_cols = [c for c in df.columns if c != 'datetime']
        df_resampled = df_resampled[final_cols]
        
        # Save to output folder
        output_filepath = output_dir / filepath.name
        df_resampled.to_csv(output_filepath, index=False)
        
        new_duration_sec = df_resampled['recv_ts'].max() - df_resampled['recv_ts'].min()
        new_freq = len(df_resampled) / new_duration_sec if new_duration_sec > 0 else 0
        
        print(f"  [OK] Downsampled to: ~{new_freq:.2f} Hz | Samples: {len(df_resampled)}")
        print(f"  [OK] Saved to {output_filepath.parent.name}/{output_filepath.name}")
        
    except Exception as e:
        print(f"  [ERR] Failed to process {filepath.name}: {e}")

# ============================================================================
# MAIN EXECUTOR
# ============================================================================

def main():
    script_dir = Path(__file__).parent
    
    print(f"\n{'='*60}")
    print(f"IMU DOWNSAMPLING PIPELINE")
    print(f"Target Frequency: {DownsampleConfig.TARGET_FREQ_HZ} Hz")
    print(f"{'='*60}")
    
    # Find all IMU files
    imu_files = list(script_dir.glob("*imu*.csv"))
    if not imu_files:
        print("[ERR] No IMU CSV files found matching pattern '*imu*.csv'")
        return
        
    # Create output directory based on the target frequency
    output_dir = script_dir / DownsampleConfig.OUTPUT_FOLDER
    output_dir.mkdir(exist_ok=True)
    
    # Process each file
    for imu_file in imu_files:
        process_imu_file(imu_file, output_dir)
        
    print(f"\n{'='*60}")
    print("[DONE] Downsampling Complete!")
    print(f"Next Step: Copy the files from '{output_dir.name}/' and use them with your Windowing Script.\n")

if __name__ == "__main__":
    main()