"""
Modular Windowing Script for HAR (Human Activity Recognition) Pipeline
=========================================================================
Creates sliding windows of IMU and audio data with activity labels.

Features:
- Configurable window size and overlap for ablation studies
- Synchronized IMU and audio windowing
- Label assignment based on majority voting
- Modular structure for easy parameter tuning
- Statistics and validation reporting
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from collections import Counter
from typing import Tuple, Dict, List
import json


# ============================================================================
# CONFIGURATION - EASILY ADJUSTABLE PARAMETERS FOR ABLATION STUDIES
# ============================================================================

class WindowConfig:
    """Configuration class for windowing parameters."""
    
    # Window size in milliseconds - MAIN PARAMETER FOR ABLATION STUDIES
    WINDOW_SIZE_MS = 2000  # Change this for ablation: 1000, 2000, 3000, etc.
    
    # Overlap as a fraction (0.0 = no overlap, 0.5 = 50% overlap)
    OVERLAP_RATIO = 0.5  # 50% overlap = 1 second stride for 2-second window
    
    # Minimum percentage of labeled data needed for a window to be valid
    MIN_LABEL_COVERAGE = 0.7  # 70% of window must have a label (not 'unlabeled')
    
    # Output format configuration
    SAVE_WINDOWS = True
    SAVE_STATISTICS = True
    CREATE_LABEL_FOLDERS = True
    
    # IMU sampling rate (approximately 100 Hz based on your data)
    IMU_SAMPLE_RATE_HZ = 100
    
    @classmethod
    def get_stride_ms(cls) -> int:
        """Calculate stride in milliseconds based on overlap."""
        return int(cls.WINDOW_SIZE_MS * (1 - cls.OVERLAP_RATIO))
    
    @classmethod
    def get_samples_per_window(cls) -> int:
        """Calculate expected number of IMU samples per window."""
        return int(cls.WINDOW_SIZE_MS / 1000 * cls.IMU_SAMPLE_RATE_HZ)
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("\n" + "="*60)
        print("WINDOWING CONFIGURATION")
        print("="*60)
        print(f"Window Size: {cls.WINDOW_SIZE_MS} ms")
        print(f"Stride (overlap): {cls.get_stride_ms()} ms")
        print(f"Overlap Ratio: {cls.OVERLAP_RATIO * 100:.0f}%")
        print(f"Expected samples/window: {cls.get_samples_per_window()}")
        print(f"Min Label Coverage: {cls.MIN_LABEL_COVERAGE * 100:.0f}%")
        print("="*60 + "\n")


# ============================================================================
# IMU WINDOWING FUNCTIONS
# ============================================================================

def load_imu_file(filepath: Path) -> pd.DataFrame:
    """Load and validate IMU CSV file."""
    try:
        df = pd.read_csv(filepath)
        
        # Verify required columns
        required_cols = ['recv_ts', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  ✗ Missing columns in {filepath.name}: {missing_cols}")
            return None
        
        # Sort by timestamp
        df = df.sort_values('recv_ts').reset_index(drop=True)
        
        # Convert recv_ts to milliseconds for easier window calculation
        df['ts_ms'] = (df['recv_ts'] * 1000).astype(int)
        
        print(f"  ✓ Loaded {filepath.name}: {len(df)} samples")
        return df
    
    except Exception as e:
        print(f"  ✗ Error loading {filepath.name}: {e}")
        return None


def get_dominant_label(labels: pd.Series, min_coverage: float) -> Tuple[str, float]:
    """
    Determine dominant label using majority voting.
    
    Returns:
        Tuple of (dominant_label, coverage_ratio)
        coverage_ratio = fraction of non-unlabeled samples in window
    """
    # Count labeled samples (exclude 'unlabeled')
    labeled_mask = labels != 'unlabeled'
    labeled_count = labeled_mask.sum()
    coverage = labeled_count / len(labels)
    
    if coverage < min_coverage:
        return 'invalid', coverage
    
    # Get most common label among labeled samples
    labeled_labels = labels[labeled_mask]
    if len(labeled_labels) == 0:
        return 'invalid', coverage
    
    dominant = labeled_labels.value_counts().idxmax()
    return dominant, coverage


def create_imu_windows(df_imu: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    Create sliding windows from IMU data.
    
    Returns:
        Dictionary with format:
        {
            'label1': [
                {'start_ts': int, 'end_ts': int, 'coverage': float, 'data': pd.DataFrame},
                ...
            ],
            ...
        }
    """
    windows_by_label = {}
    
    # Get time range
    start_ts_ms = df_imu['ts_ms'].min()
    end_ts_ms = df_imu['ts_ms'].max()
    window_size = WindowConfig.WINDOW_SIZE_MS
    stride = WindowConfig.get_stride_ms()
    
    print(f"\n  Creating IMU windows:")
    print(f"    Time range: {start_ts_ms} - {end_ts_ms} ms")
    print(f"    Window size: {window_size} ms")
    print(f"    Stride: {stride} ms")
    
    window_start = start_ts_ms
    window_count = 0
    skipped_count = 0
    
    while window_start + window_size <= end_ts_ms:
        window_end = window_start + window_size
        
        # Extract data for this window
        mask = (df_imu['ts_ms'] >= window_start) & (df_imu['ts_ms'] < window_end)
        window_data = df_imu[mask].copy()
        
        if len(window_data) == 0:
            window_start += stride
            continue
        
        # Determine label for this window
        label, coverage = get_dominant_label(
            window_data['label'], 
            WindowConfig.MIN_LABEL_COVERAGE
        )
        
        if label == 'invalid':
            skipped_count += 1
            window_start += stride
            continue
        
        # Store window
        window_info = {
            'start_ts': window_start,
            'end_ts': window_end,
            'coverage': coverage,
            'sample_count': len(window_data),
            'data': window_data.copy()
        }
        
        if label not in windows_by_label:
            windows_by_label[label] = []
        windows_by_label[label].append(window_info)
        
        window_count += 1
        window_start += stride
    
    print(f"    Created {window_count} windows, skipped {skipped_count} (low label coverage)")
    return windows_by_label


# ============================================================================
# AUDIO WINDOWING FUNCTIONS
# ============================================================================

def load_audio_file(filepath: Path) -> pd.DataFrame:
    """Load and validate audio CSV file."""
    try:
        df = pd.read_csv(filepath)
        
        # Audio files typically have timestamp and audio sample columns
        if 'ts' not in df.columns and 'recv_ts' not in df.columns:
            print(f"  ✗ No timestamp column found in {filepath.name}")
            return None
        
        # Identify timestamp column
        ts_col = 'ts' if 'ts' in df.columns else 'recv_ts'
        df = df.sort_values(ts_col).reset_index(drop=True)
        
        # Rename for consistency
        df.rename(columns={ts_col: 'recv_ts'}, inplace=True)
        
        # Convert to milliseconds if needed
        if df['recv_ts'].dtype in ['float64', 'float32']:
            # If already in seconds (Unix timestamp), convert to ms
            if df['recv_ts'].iloc[0] > 1e10:  # Unix timestamp threshold
                df['ts_ms'] = (df['recv_ts'] * 1000).astype(int)
            else:
                df['ts_ms'] = df['recv_ts'].astype(int)
        else:
            df['ts_ms'] = df['recv_ts'].astype(int)
        
        print(f"  ✓ Loaded {filepath.name}: {len(df)} samples")
        return df
    
    except Exception as e:
        print(f"  ✗ Error loading {filepath.name}: {e}")
        return None


def create_audio_windows(df_audio: pd.DataFrame, imu_windows: Dict) -> Dict[str, List[Dict]]:
    """
    Create sliding windows from audio data aligned with IMU windows.
    
    Since audio and IMU are synchronized, we use the same window boundaries.
    """
    windows_by_label = {}
    
    print(f"\n  Creating audio windows (aligned with IMU):")
    
    for label, imu_window_list in imu_windows.items():
        windows_by_label[label] = []
        
        for imu_window in imu_window_list:
            window_start = imu_window['start_ts']
            window_end = imu_window['end_ts']
            
            # Extract audio data for same timestamp range
            mask = (df_audio['ts_ms'] >= window_start) & (df_audio['ts_ms'] < window_end)
            window_data = df_audio[mask].copy()
            
            if len(window_data) == 0:
                continue
            
            window_info = {
                'start_ts': window_start,
                'end_ts': window_end,
                'sample_count': len(window_data),
                'data': window_data.copy()
            }
            
            windows_by_label[label].append(window_info)
    
    # Print summary
    total_audio_windows = sum(len(windows) for windows in windows_by_label.values())
    print(f"    Created {total_audio_windows} audio windows")
    return windows_by_label


# ============================================================================
# WINDOW SAVING FUNCTIONS
# ============================================================================

def save_imu_windows(windows_by_label: Dict, output_dir: Path):
    """Save IMU windows as CSV files, organized by label."""
    
    print(f"\n  Saving IMU windows to {output_dir.name}/")
    
    for label, window_list in windows_by_label.items():
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, window in enumerate(window_list):
            # Create filename with timestamp
            start_ts = window['start_ts']
            filename = f"imu_{label}_{start_ts}_{idx:04d}.csv"
            filepath = label_dir / filename
            
            # Save window data (without the extra columns we added)
            save_cols = [col for col in window['data'].columns 
                        if col not in ['ts_ms']]
            window['data'][save_cols].to_csv(filepath, index=False)
        
        print(f"    ✓ {label}: {len(window_list)} windows saved")


def save_audio_windows(windows_by_label: Dict, output_dir: Path):
    """Save audio windows as CSV files, organized by label."""
    
    print(f"\n  Saving audio windows to {output_dir.name}/")
    
    for label, window_list in windows_by_label.items():
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, window in enumerate(window_list):
            # Create filename with timestamp
            start_ts = window['start_ts']
            filename = f"audio_{label}_{start_ts}_{idx:04d}.csv"
            filepath = label_dir / filename
            
            # Save window data
            save_cols = [col for col in window['data'].columns 
                        if col not in ['ts_ms']]
            window['data'][save_cols].to_csv(filepath, index=False)
        
        print(f"    ✓ {label}: {len(window_list)} windows saved")


def save_statistics(imu_windows: Dict, audio_windows: Dict, output_dir: Path):
    """Save windowing statistics as JSON."""
    
    stats = {
        'config': {
            'window_size_ms': WindowConfig.WINDOW_SIZE_MS,
            'overlap_ratio': WindowConfig.OVERLAP_RATIO,
            'stride_ms': WindowConfig.get_stride_ms(),
            'min_label_coverage': WindowConfig.MIN_LABEL_COVERAGE,
        },
        'imu_windows': {},
        'audio_windows': {},
        'summary': {}
    }
    
    # IMU statistics
    total_imu_windows = 0
    for label, windows in imu_windows.items():
        stats['imu_windows'][label] = {
            'count': len(windows),
            'avg_coverage': np.mean([w['coverage'] for w in windows]),
            'avg_samples': np.mean([w['sample_count'] for w in windows])
        }
        total_imu_windows += len(windows)
    
    # Audio statistics
    total_audio_windows = 0
    for label, windows in audio_windows.items():
        stats['audio_windows'][label] = {
            'count': len(windows),
            'avg_samples': np.mean([w['sample_count'] for w in windows])
        }
        total_audio_windows += len(windows)
    
    # Summary
    stats['summary'] = {
        'total_imu_windows': total_imu_windows,
        'total_audio_windows': total_audio_windows,
        'labels': list(imu_windows.keys())
    }
    
    # Save
    stats_file = output_dir / 'windowing_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n  ✓ Statistics saved to windowing_statistics.json")
    
    # Print summary
    print(f"\n  SUMMARY:")
    print(f"    Total IMU windows: {total_imu_windows}")
    print(f"    Total audio windows: {total_audio_windows}")
    print(f"    Labels: {', '.join(imu_windows.keys())}")
    for label in imu_windows.keys():
        print(f"      - {label}: {stats['imu_windows'][label]['count']} windows")


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def main():
    """Main function to orchestrate the windowing pipeline."""
    
    # Get script directory
    script_dir = Path(__file__).parent
    print(f"\n{'='*60}")
    print("HAR WINDOWING PIPELINE")
    print(f"{'='*60}")
    print(f"Script directory: {script_dir}\n")
    
    # Print configuration
    WindowConfig.print_config()
    
    # Find IMU files
    imu_files = list(script_dir.glob("*imu*.csv"))
    if not imu_files:
        print("✗ No IMU CSV files found matching pattern '*imu*.csv'")
        return
    
    print(f"Found {len(imu_files)} IMU file(s):")
    for f in imu_files:
        print(f"  - {f.name}")
    
    # Find audio files
    audio_files = list(script_dir.glob("*audio_ts*.csv"))
    if not audio_files:
        print("\n✗ No audio CSV files found matching pattern '*audio_ts*.csv'")
        return
    
    print(f"\nFound {len(audio_files)} audio file(s):")
    for f in audio_files:
        print(f"  - {f.name}")
    
    # Create output directory
    output_dir = script_dir / "windowed_data"
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")
    
    # =====================================================================
    # PROCESS EACH IMU/AUDIO PAIR
    # =====================================================================
    
    for imu_file in imu_files:
        print(f"\n{'='*60}")
        print(f"Processing: {imu_file.name}")
        print(f"{'='*60}")
        
        # Load IMU data
        print(f"\nLoading data:")
        df_imu = load_imu_file(imu_file)
        if df_imu is None:
            print(f"  ✗ Skipping {imu_file.name}")
            continue
        
        # Find corresponding audio file (matching pattern)
        # Audio files like audio_ts_10_90_50_196.csv, IMU like imu_left_hand.csv
        # For now, we'll try to match by finding audio files in the same directory
        matching_audio = [f for f in audio_files if f.parent == imu_file.parent]
        
        if not matching_audio:
            print(f"  ⚠ No audio files found in same directory, skipping audio windowing")
            df_audio = None
        else:
            # For multiple audio files, process the first one (or you could process all)
            audio_file = matching_audio[0]
            df_audio = load_audio_file(audio_file)
        
        # Create IMU windows
        print(f"\nWindowing:")
        imu_windows = create_imu_windows(df_imu)
        
        # Create audio windows (if audio is available)
        audio_windows = {}
        if df_audio is not None:
            audio_windows = create_audio_windows(df_audio, imu_windows)
        
        # Save windows
        if WindowConfig.SAVE_WINDOWS:
            print(f"\nSaving:")
            save_imu_windows(imu_windows, output_dir / "imu")
            if df_audio is not None:
                save_audio_windows(audio_windows, output_dir / "audio")
        
        # Save statistics
        if WindowConfig.SAVE_STATISTICS:
            save_statistics(imu_windows, audio_windows, output_dir)
    
    print(f"\n{'='*60}")
    print("✓ WINDOWING COMPLETE")
    print(f"{'='*60}\n")
    print(f"To modify window parameters, edit the WindowConfig class at the top of this script.")
    print(f"For ablation studies, change WINDOW_SIZE_MS and re-run this script.\n")


if __name__ == "__main__":
    main()
