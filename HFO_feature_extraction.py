#!/usr/bin/env python3
"""
IMPROVED HFO EXTRACTION: Proper folder-per-subject structure

The issue: all_subs_preprocessed_data has folders for each subject,
each containing multiple .mat files.

Solution: Find one .mat file per subject folder and extract from it.
"""

import scipy.io as sio
import numpy as np
import pandas as pd
from scipy.signal import medfilt, welch, butter, filtfilt
import os
from pathlib import Path

print("=" * 80)
print("IMPROVED HFO EXTRACTION: Folder-per-subject structure")
print("=" * 80)

# ============================================================================
# STEP 1: Find all subject folders
# ============================================================================

print("\n1. Finding subject folders...")

data_folder = '/Users/olivialiau/Downloads/data for olivia/all_subs_preprocessed_data'

if not os.path.exists(data_folder):
    print(f"✗ Folder not found: {data_folder}")
    exit()

# List all folders in all_subs_preprocessed_data
subject_folders = {}

for item in os.listdir(data_folder):
    item_path = os.path.join(data_folder, item)
    
    # Check if it's a directory
    if os.path.isdir(item_path):
        # Try to extract subject ID from folder name
        folder_name = item.strip()
        
        # Common patterns: DBS_bG01, DBS_bG01_processed, etc.
        # Extract the base subject ID
        subject_id = folder_name
        
        # Try to get cleaner ID (remove _processed, _data, etc.)
        for suffix in ['_processed', '_data', '_preproc', '_prep']:
            if suffix in subject_id:
                subject_id = subject_id.replace(suffix, '')
        
        subject_folders[subject_id] = item_path

print(f"✓ Found {len(subject_folders)} subject folders")
print(f"\n  Sample folders:")
for subj in sorted(list(subject_folders.keys()))[:10]:
    print(f"    {subj}")

# ============================================================================
# STEP 2: Find .mat files in each folder
# ============================================================================

print(f"\n2. Finding .mat files in each subject folder...")

subject_files = {}

for subject_id, folder_path in sorted(subject_folders.items()):
    # Find all .mat files in this folder
    mat_files = list(Path(folder_path).glob('*.mat'))
    
    if len(mat_files) == 0:
        continue
    
    # Prefer certain files (look for preprocessed/filtered versions)
    preferred_order = [
        'filtered_downsampled',
        'filtered',
        'preprocessed',
        'data',
        ''  # any .mat file
    ]
    
    selected_file = None
    for pattern in preferred_order:
        for mat_file in mat_files:
            if pattern.lower() in mat_file.name.lower() or pattern == '':
                selected_file = str(mat_file)
                break
        if selected_file:
            break
    
    if selected_file:
        subject_files[subject_id] = selected_file

print(f"✓ Found .mat files for {len(subject_files)} subjects")

if len(subject_files) != len(subject_folders):
    missing = set(subject_folders.keys()) - set(subject_files.keys())
    print(f"\n⚠ {len(missing)} folders without .mat files:")
    for subj in list(missing)[:5]:
        print(f"    {subj}")

# ============================================================================
# STEP 3: Helper functions
# ============================================================================

def extract_hfo_features(signal, fs):
    """Extract HFO features from a single signal"""
    
    if len(signal) == 0 or np.isnan(signal).all():
        return 0, 0
    
    try:
        # Bandpass filter 80-500 Hz
        nyquist = fs / 2
        if 80/nyquist >= 1 or 500/nyquist >= 1:
            return 0, 0
        
        b, a = butter(4, [80/nyquist, 500/nyquist], btype='band')
        signal_hfo = filtfilt(b, a, signal)
    except:
        signal_hfo = signal
    
    try:
        # Compute PSD
        frequencies, psd_hfo = welch(
            signal_hfo, fs=fs, window='hamming',
            nperseg=int(fs*1), noverlap=int(fs*0.5), nfft=int(fs*10)
        )
        
        # Extract HFO band
        hfo_band_idx = (frequencies >= 80) & (frequencies <= 500)
        
        if np.sum(hfo_band_idx) == 0:
            return 0, 0
        
        hfo_power = np.trapz(psd_hfo[hfo_band_idx], frequencies[hfo_band_idx])
        hfo_peak_freq = frequencies[hfo_band_idx][np.argmax(psd_hfo[hfo_band_idx])]
        
        return float(hfo_power), float(hfo_peak_freq)
    except:
        return 0, 0

# ============================================================================
# STEP 4: Extract HFO from each subject
# ============================================================================

print(f"\n3. Extracting HFO features from each subject...")
print("=" * 80)

all_hfo_features = []
successful_subjects = 0
failed_subjects = []

for file_idx, (subject_id, mat_filepath) in enumerate(sorted(subject_files.items())):
    if (file_idx + 1) % 10 == 0 or (file_idx + 1) == len(subject_files):
        print(f"   Processing {file_idx + 1}/{len(subject_files)}: {subject_id}")
    
    try:
        # Load subject's data
        mat_data = sio.loadmat(mat_filepath)
        
        # Find the trial data
        trial = None
        fs = 1200  # default
        labels = None
        
        # Try different common structures
        if 'data' in mat_data:
            data = mat_data['data']
            if isinstance(data, np.ndarray) and data.shape == (1, 1):
                data = data[0, 0]
                if hasattr(data, 'dtype') and 'trial' in data.dtype.names:
                    trial = data['trial'][0, 0]
                    if 'fsample' in data.dtype.names:
                        fs = int(data['fsample'][0, 0])
                    if 'label' in data.dtype.names:
                        labels = data['label']
            elif isinstance(data, np.ndarray):
                trial = data
        elif 'trial' in mat_data:
            trial = mat_data['trial']
            if 'fsample' in mat_data:
                fs = int(mat_data['fsample'][0, 0]) if isinstance(mat_data['fsample'], np.ndarray) else int(mat_data['fsample'])
        
        if trial is None:
            failed_subjects.append((subject_id, "No trial data"))
            continue
        
        # Ensure trial is 2D
        if len(trial.shape) == 1:
            trial = trial.reshape(1, -1)
        elif len(trial.shape) > 2:
            trial = trial.reshape(trial.shape[0], -1)
        
        # Extract labels if available
        if labels is not None:
            channel_labels = []
            for i in range(min(labels.shape[0], trial.shape[0])):
                try:
                    if isinstance(labels[i, 0], np.ndarray):
                        ch_label = labels[i, 0][0]
                    else:
                        ch_label = str(labels[i, 0])
                    channel_labels.append(ch_label)
                except:
                    channel_labels.append(f"ch_{i}")
        else:
            channel_labels = [f"ch_{i}" for i in range(trial.shape[0])]
        
        # Extract HFO from each channel
        for ch_idx in range(trial.shape[0]):
            signal = trial[ch_idx, :]
            
            hfo_power, hfo_peak_freq = extract_hfo_features(signal, fs)
            
            all_hfo_features.append({
                'subject': subject_id,
                'channel': channel_labels[ch_idx] if ch_idx < len(channel_labels) else f"ch_{ch_idx}",
                'hfo_power': hfo_power,
                'hfo_peak_freq': hfo_peak_freq,
                'artifacts_detected': 0,
                'artifact_percentage': 0.0
            })
        
        successful_subjects += 1
        
    except Exception as e:
        failed_subjects.append((subject_id, str(e)[:50]))

print(f"\n✓ Successfully extracted from {successful_subjects}/{len(subject_files)} subjects")

if failed_subjects:
    print(f"\n⚠ Failed subjects ({len(failed_subjects)}):")
    for subj, error in failed_subjects[:10]:
        print(f"  • {subj}: {error}")

# ============================================================================
# STEP 5: Save results
# ============================================================================

print(f"\n4. Saving results...")

hfo_df = pd.DataFrame(all_hfo_features)
output_file = '/Users/olivialiau/Downloads/data for olivia/HFO_features.csv'
hfo_df.to_csv(output_file, index=False)

print(f"✓ Saved to: {output_file}")

# ============================================================================
# STEP 6: Verification
# ============================================================================

print(f"\n5. Verification:")

print(f"\n   Total records: {len(hfo_df)}")
print(f"   Unique subjects: {hfo_df['subject'].nunique()}")
print(f"   Unique channels: {hfo_df['channel'].nunique()}")

print(f"\n   HFO Power statistics:")
print(f"     Mean: {hfo_df['hfo_power'].mean():.4f}")
print(f"     Std: {hfo_df['hfo_power'].std():.4f}")
print(f"     Min: {hfo_df['hfo_power'].min():.4f}")
print(f"     Max: {hfo_df['hfo_power'].max():.4f}")
print(f"     Unique values: {hfo_df['hfo_power'].nunique()}")

# Check for outliers/issues
print(f"\n   Data quality check:")
zero_power = (hfo_df['hfo_power'] == 0).sum()
print(f"     Zero power values: {zero_power} ({100*zero_power/len(hfo_df):.1f}%)")

if hfo_df['hfo_power'].std() > hfo_df['hfo_power'].mean() * 0.1:
    print(f"   GOOD: HFO Power has substantial VARIANCE across subjects!")
else:
    print(f"   WARNING: HFO Power has low variance")

print(f"\n   HFO Peak Frequency statistics:")
print(f"     Mean: {hfo_df['hfo_peak_freq'].mean():.4f}")
print(f"     Std: {hfo_df['hfo_peak_freq'].std():.4f}")
print(f"     Unique values: {hfo_df['hfo_peak_freq'].nunique()}")

if hfo_df['hfo_peak_freq'].std() > 1.0:
    print(f"   GOOD: HFO Peak Freq has VARIANCE across subjects!")
else:
    print(f"   WARNING: HFO Peak Freq has low variance")

# Show sample data
print(f"\n   Sample features (first 10 unique subjects):")
for subj in sorted(hfo_df['subject'].unique())[:10]:
    subj_data = hfo_df[hfo_df['subject'] == subj].iloc[0]
    print(f"     {subj:15s}: power={subj_data['hfo_power']:10.4f}, freq={subj_data['hfo_peak_freq']:7.2f}")


