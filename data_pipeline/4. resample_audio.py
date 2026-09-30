"""
Audio Resampling Utility
=========================================================================
Resamples varying-frequency audio files to a strict target sample rate
(e.g., 16,000 Hz) to ensure perfect multimodal synchronization.
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

class AudioConfig:
    # Target Sample Rate for standard ML/Spectrogram models
    TARGET_SR = 16000 
    
    # The folder where the perfectly synced files will be saved
    OUTPUT_FOLDER = f"resampled_{TARGET_SR}hz"

# ============================================================================
# PROCESSING LOGIC
# ============================================================================

def process_audio_file(filepath: Path, output_dir: Path):
    """Load, resample, and save a single WAV file."""
    try:
        print(f"\nProcessing: {filepath.name}")
        
        # 1. Load the audio file at its NATIVE sample rate
        # sr=None prevents librosa from automatically resampling on load
        audio_data, orig_sr = librosa.load(filepath, sr=None, mono=True)
        
        # Calculate original duration
        orig_frames = len(audio_data)
        orig_duration = orig_frames / orig_sr
        print(f"  [INFO] Original SR: {orig_sr} Hz | Duration: {orig_duration:.6f}s")
        
        # 2. Resample to the target frequency
        if orig_sr != AudioConfig.TARGET_SR:
            # librosa's resampler uses high-quality mathematical interpolation
            resampled_data = librosa.resample(y=audio_data, orig_sr=orig_sr, target_sr=AudioConfig.TARGET_SR)
        else:
            print("  [INFO] File is already at target sample rate. Skipping resampling.")
            resampled_data = audio_data
            
        # Calculate new duration to verify the timeline hasn't shifted
        new_frames = len(resampled_data)
        new_duration = new_frames / AudioConfig.TARGET_SR
        
        # 3. Save the new WAV file
        output_filepath = output_dir / filepath.name
        # Note: subtype='PCM_16' saves it as a standard 16-bit WAV file
        sf.write(output_filepath, resampled_data, AudioConfig.TARGET_SR, subtype='PCM_16')
        
        print(f"  [OK] New SR: {AudioConfig.TARGET_SR} Hz | Duration: {new_duration:.6f}s")
        
        # Flag any tiny floating-point duration drift
        drift = abs(orig_duration - new_duration)
        if drift > 0.001:
            print(f"  [WARN] Slight duration shift detected: {drift:.6f}s")
            
    except Exception as e:
        print(f"  [ERR] Failed to process {filepath.name}: {e}")

# ============================================================================
# MAIN EXECUTOR
# ============================================================================

def main():
    script_dir = Path(__file__).parent
    
    print(f"\n{'='*60}")
    print(f"AUDIO DOWNSAMPLING PIPELINE")
    print(f"Target Frequency: {AudioConfig.TARGET_SR} Hz")
    print(f"{'='*60}")
    
    # Find all WAV files
    wav_files = list(script_dir.glob("*.wav"))
    if not wav_files:
        print("[ERR] No .wav files found in the directory.")
        return
        
    # Create output directory
    output_dir = script_dir / AudioConfig.OUTPUT_FOLDER
    output_dir.mkdir(exist_ok=True)
    
    # Process each file
    for wav_file in wav_files:
        process_audio_file(wav_file, output_dir)
        
    print(f"\n{'='*60}")
    print(f"[DONE] All audio files locked to {AudioConfig.TARGET_SR} Hz.")
    print("You are now ready to generate your sliding windows!")

if __name__ == "__main__":
    main()