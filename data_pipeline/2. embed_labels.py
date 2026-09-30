"""
Script to embed labels from Excel file into IMU CSV files.
Labels are matched based on timestamp ranges.
"""

import pandas as pd
import glob
import os
from pathlib import Path

def load_labels(excel_file):
    """
    Load labels from Excel file.
    Expected columns: Label, Start, End
    """
    try:
        df_labels = pd.read_excel(excel_file)
        print(f"✓ Loaded labels from {excel_file}")
        print(f"  Found {len(df_labels)} label ranges")
        return df_labels
    except FileNotFoundError:
        print(f"✗ Error: {excel_file} not found")
        return None
    except Exception as e:
        print(f"✗ Error loading Excel file: {e}")
        return None


def get_labels_vectorized(timestamps, df_labels, min_timestamp):
    """
    Vectorized label assignment for efficiency.
    Converts absolute timestamps to relative time from start, then matches against label ranges.
    """
    # Convert absolute timestamps to relative time (seconds from start)
    relative_timestamps = timestamps - min_timestamp
    
    # Initialize all as 'unlabeled'
    labels = pd.Series(['unlabeled'] * len(relative_timestamps), index=timestamps.index)
    
    # For each label range, mark all timestamps that fall within it
    for idx, row in df_labels.iterrows():
        start = row['Start']
        end = row['End']
        # Skip rows with NaN values
        if pd.isna(start) or pd.isna(end):
            continue
        mask = (relative_timestamps >= start) & (relative_timestamps <= end)
        labels[mask] = row['Label']
    
    return labels


def embed_labels_in_csv(csv_file, df_labels):
    """
    Load CSV file, add label column based on timestamps, and save.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)
        
        # Check if required columns exist
        if 'recv_ts' not in df.columns:
            print(f"✗ {csv_file}: Missing 'recv_ts' column, skipping...")
            return False
        
        # Get minimum timestamp to convert to relative time
        min_ts = df['recv_ts'].min()
        
        # Apply label assignment using vectorized operation (much faster)
        print(f"  Processing {os.path.basename(csv_file)}...", end=" ", flush=True)
        df['label'] = get_labels_vectorized(df['recv_ts'], df_labels, min_ts)
        
        # Count labeled vs unlabeled
        labeled_count = (df['label'] != 'unlabeled').sum()
        total_count = len(df)
        
        # Save the updated CSV
        df.to_csv(csv_file, index=False)
        print(f"✓ ({labeled_count}/{total_count} rows labeled)")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {csv_file}: {e}")
        return False


def main():
    """
    Main function to process all IMU CSV files in the script's directory.
    """
    # Get script directory (not current working directory)
    script_dir = Path(__file__).parent
    print(f"Script directory: {script_dir}\n")
    
    # Look for the Excel label file (search for any Data Labelling*.xlsx file)
    excel_files = list(script_dir.glob("*Data Labelling*.xlsx"))
    
    if not excel_files:
        print("✗ Error: No Excel file matching '*Data Labelling*.xlsx' found in script directory")
        print(f"  Please place the Excel file in: {script_dir}")
        return
    
    excel_file = excel_files[0]
    print(f"Using label file: {excel_file.name}\n")
    
    # Load labels from Excel
    df_labels = load_labels(excel_file)
    if df_labels is None:
        return
    
    print("\nLabel ranges:")
    for idx, row in df_labels.iterrows():
        print(f"  {row['Label']}: {row['Start']:.3f} - {row['End']:.3f}")
    
    # Find all IMU CSV files (typically named *imu*.csv)
    imu_files = list(script_dir.glob("*imu*.csv"))
    
    if not imu_files:
        print("\n✗ No IMU CSV files found matching pattern '*imu*.csv'")
        return
    
    print(f"\n\nProcessing {len(imu_files)} IMU file(s):")
    print("-" * 60)
    
    # Show sample timestamp range from first file for debugging
    if imu_files:
        df_sample = pd.read_csv(imu_files[0])
        if 'recv_ts' in df_sample.columns:
            min_ts = df_sample['recv_ts'].min()
            max_ts = df_sample['recv_ts'].max()
            relative_min = 0
            relative_max = max_ts - min_ts
            print(f"Sample timestamp range (relative): {relative_min:.3f} - {relative_max:.3f} seconds")
            print(f"Label ranges: {df_labels['Start'].min():.3f} - {df_labels['End'].max():.3f} seconds\n")
    
    success_count = 0
    for csv_file in imu_files:
        if embed_labels_in_csv(csv_file, df_labels):
            success_count += 1
    
    print("-" * 60)
    print(f"\n✓ Successfully processed {success_count}/{len(imu_files)} files")
    print("\nLabel column 'label' has been added to each CSV file.")


if __name__ == "__main__":
    main()
