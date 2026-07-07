"""
Modular Chunking Script for HAR (Human Activity Recognition) Pipeline
=========================================================================
Extracts continuous blocks of IMU and audio data based on activity labels.

Features:
- Extracts full continuous sequences of an activity as a single file.
- Handles repeating activities by numbering them sequentially.
- Filters out sequences that are too short for downstream windowing.
- Synchronized IMU and audio chunking.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from collections import Counter
from typing import Tuple, Dict, List
import json
from scipy.io import wavfile

# ============================================================================
# CONFIGURATION
# ============================================================================

class ChunkConfig:
    """Configuration class for chunking parameters."""
    
    # Filter out activities shorter than this duration. 
    # Set this to your future Edge Impulse window size to avoid zero-padding!
    MIN_CHUNK_DURATION_MS = 2000 
    
    # Output format configuration
    SAVE_CHUNKS = True
    SAVE_STATISTICS = True
    CREATE_LABEL_FOLDERS = True
    
    # IMU sampling rate (default 250 Hz, updated dynamically per file)
    IMU_SAMPLE_RATE_HZ = 250
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("\n" + "="*60)
        print("CHUNKING CONFIGURATION")
        print("="*60)
        print(f"Minimum Duration: {cls.MIN_CHUNK_DURATION_MS} ms")
        print("="*60 + "\n")


# ============================================================================
# IMU CHUNKING FUNCTIONS
# ============================================================================

def load_imu_file(filepath: Path) -> pd.DataFrame:
    """Load and validate IMU CSV file."""
    try:
        df = pd.read_csv(filepath)
        
        required_cols = ['recv_ts', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  [ERR] Missing columns in {filepath.name}: {missing_cols}")
            return None
        
        df = df.sort_values('recv_ts').reset_index(drop=True)
        df['ts_ms'] = (df['recv_ts'] * 1000).astype(int)
        
        print(f"  [OK] Loaded {filepath.name}: {len(df)} samples")
        return df
    
    except Exception as e:
        print(f"  [ERR] Error loading {filepath.name}: {e}")
        return None


def extract_imu_chunks(df_imu: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    Extract continuous blocks of identical labels into discrete files.
    """
    chunks_by_label = {}
    
    print(f"\n  Extracting continuous IMU chunks:")
    
    # Identify continuous blocks of identical labels by looking for changes
    # This creates a unique ID for every continuous segment
    block_ids = (df_imu['label'] != df_imu['label'].shift()).cumsum()
    
    skipped_count = 0
    chunk_count = 0
    
    for block_id, group in df_imu.groupby(block_ids):
        label = group['label'].iloc[0]
        
        # Skip unlabeled or invalid sections entirely
        if label in ['unlabeled', 'invalid']:
            continue
            
        start_ts = group['ts_ms'].min()
        end_ts = group['ts_ms'].max()
        duration_ms = end_ts - start_ts
        
        # Filter out chunks that are too short
        if duration_ms < ChunkConfig.MIN_CHUNK_DURATION_MS:
            skipped_count += 1
            continue
            
        chunk_info = {
            'start_ts': start_ts,
            'end_ts': end_ts,
            'duration_ms': duration_ms,
            'sample_count': len(group),
            'data': group.copy()
        }
        
        if label not in chunks_by_label:
            chunks_by_label[label] = []
        
        chunks_by_label[label].append(chunk_info)
        chunk_count += 1
        
    print(f"    Created {chunk_count} valid files, skipped {skipped_count} (duration < {ChunkConfig.MIN_CHUNK_DURATION_MS}ms)")
    return chunks_by_label


# ============================================================================
# AUDIO CHUNKING FUNCTIONS
# ============================================================================

def load_audio_wav(filepath: Path, imu_start_ts_ms: int) -> Tuple[np.ndarray, int, pd.DataFrame]:
    """Load and synchronize audio WAV file with IMU timestamps."""
    try:
        sample_rate, audio_data = wavfile.read(filepath)
        
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
            
        num_samples = len(audio_data)
        sample_duration_ms = 1000.0 / sample_rate 
        timestamps_ms = imu_start_ts_ms + np.arange(num_samples) * sample_duration_ms
        
        df = pd.DataFrame({
            'ts_ms': timestamps_ms.astype(int),
            'audio_sample': np.arange(num_samples),
            'audio_value': audio_data
        })
        
        print(f"  [OK] Loaded {filepath.name}: {num_samples} samples @ {sample_rate} Hz")
        return audio_data, sample_rate, df
    
    except Exception as e:
        print(f"  [ERR] Error loading {filepath.name}: {e}")
        return None, None, None


def extract_audio_chunks(df_audio: pd.DataFrame, imu_chunks: Dict) -> Dict[str, List[Dict]]:
    """Extract audio data corresponding to the exact times of the IMU chunks."""
    audio_chunks = {}
    print(f"\n  Creating audio chunks (aligned with IMU):")
    
    if df_audio is None or len(df_audio) == 0:
        print(f"    [WARNING] No audio data available. Skipping audio.")
        return {}
        
    for label, chunk_list in imu_chunks.items():
        audio_chunks[label] = []
        
        for chunk in chunk_list:
            start_ts = chunk['start_ts']
            end_ts = chunk['end_ts']
            
            mask = (df_audio['ts_ms'] >= start_ts) & (df_audio['ts_ms'] <= end_ts)
            chunk_data = df_audio[mask].copy()
            
            if len(chunk_data) == 0:
                continue
                
            audio_chunks[label].append({
                'start_ts': start_ts,
                'end_ts': end_ts,
                'duration_ms': end_ts - start_ts,
                'sample_count': len(chunk_data),
                'data': chunk_data
            })
            
    total_audio_chunks = sum(len(chunks) for chunks in audio_chunks.values())
    print(f"    Created {total_audio_chunks} audio chunks")
    return audio_chunks


# ============================================================================
# SAVING FUNCTIONS
# ============================================================================

def save_imu_chunks(chunks_by_label: Dict, output_dir: Path):
    """Save IMU chunks as CSV files, organized by label."""
    print(f"\n  Saving IMU chunks to {output_dir.name}/")
    
    for label, chunk_list in chunks_by_label.items():
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, chunk in enumerate(chunk_list):
            filename = f"imu_{label}_{idx:04d}.csv"
            filepath = label_dir / filename
            
            save_cols = [col for col in chunk['data'].columns if col not in ['ts_ms']]
            chunk['data'][save_cols].to_csv(filepath, index=False)
            
        print(f"    [OK] {label}: {len(chunk_list)} files saved")


def save_audio_chunks(chunks_by_label: Dict, output_dir: Path, sample_rate: int):
    """Save audio chunks as WAV files, organized by label."""
    print(f"\n  Saving audio chunks to {output_dir.name}/")
    
    for label, chunk_list in chunks_by_label.items():
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, chunk in enumerate(chunk_list):
            filename = f"audio_{label}_{idx:04d}.wav"
            filepath = label_dir / filename
            
            audio_array = chunk['data']['audio_value'].values
            wavfile.write(filepath, sample_rate, audio_array)
            
        print(f"    [OK] {label}: {len(chunk_list)} files saved")


def save_statistics(imu_chunks: Dict, audio_chunks: Dict, output_dir: Path):
    """Save chunking statistics as JSON."""
    stats = {
        'config': {
            'min_chunk_duration_ms': ChunkConfig.MIN_CHUNK_DURATION_MS,
            'imu_sample_rate_hz': ChunkConfig.IMU_SAMPLE_RATE_HZ,
        },
        'imu_chunks': {},
        'audio_chunks': {},
        'summary': {}
    }
    
    total_imu = 0
    for label, chunks in imu_chunks.items():
        stats['imu_chunks'][label] = {
            'count': len(chunks),
            'avg_duration_ms': np.mean([c['duration_ms'] for c in chunks]),
            'avg_samples': np.mean([c['sample_count'] for c in chunks])
        }
        total_imu += len(chunks)
        
    total_audio = 0
    for label, chunks in audio_chunks.items():
        stats['audio_chunks'][label] = {
            'count': len(chunks),
            'avg_samples': np.mean([c['sample_count'] for c in chunks])
        }
        total_audio += len(chunks)
        
    stats['summary'] = {
        'total_imu_files': total_imu,
        'total_audio_files': total_audio,
        'labels': list(imu_chunks.keys())
    }
    
    stats_file = output_dir / 'chunking_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    script_dir = Path(__file__).parent
    print(f"\n{'='*60}")
    print("HAR CONTINUOUS CHUNKING PIPELINE")
    print(f"{'='*60}")
    
    ChunkConfig.print_config()
    
    imu_files = list(script_dir.glob("*imu*.csv"))
    if not imu_files:
        print("[ERR] No IMU CSV files found matching pattern '*imu*.csv'")
        return
        
    audio_wav_files = list(script_dir.glob("audio_*.wav"))
    output_dir = script_dir / "chunked_data"
    output_dir.mkdir(exist_ok=True)
    
    for imu_file in imu_files:
        print(f"\n{'='*60}")
        print(f"Processing: {imu_file.name}")
        
        sensor_loc = imu_file.stem.replace('imu_', '')
        df_imu = load_imu_file(imu_file)
        if df_imu is None:
            continue
            
        total_duration_sec = (df_imu['ts_ms'].max() - df_imu['ts_ms'].min()) / 1000.0
        if total_duration_sec > 0:
            ChunkConfig.IMU_SAMPLE_RATE_HZ = len(df_imu) / total_duration_sec
            
        imu_start_ts_ms = df_imu['ts_ms'].iloc[0]
        
        df_audio, sample_rate = None, None
        matching_audio = next((f for f in audio_wav_files if sensor_loc in f.name), None)
        
        if matching_audio:
            audio_data, sample_rate, df_audio = load_audio_wav(matching_audio, imu_start_ts_ms)
            
        # Extract the contiguous segments
        imu_chunks = extract_imu_chunks(df_imu)
        
        audio_chunks = {}
        if df_audio is not None:
            audio_chunks = extract_audio_chunks(df_audio, imu_chunks)
            
        # Save output files
        if ChunkConfig.SAVE_CHUNKS:
            save_imu_chunks(imu_chunks, output_dir / sensor_loc / "imu")
            if df_audio is not None and sample_rate is not None:
                save_audio_chunks(audio_chunks, output_dir / sensor_loc / "audio", sample_rate)
                
        if ChunkConfig.SAVE_STATISTICS:
            save_statistics(imu_chunks, audio_chunks, output_dir / sensor_loc)

    print(f"\n{'='*60}\n[OK] CHUNKING COMPLETE\n{'='*60}\n")

if __name__ == "__main__":
    main()