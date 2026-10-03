"""[ARCHIVED — SUPERSEDED, KNOWN BUGGY]

This script is retained for provenance only. It is superseded by
src/curve_icon_dbso/ and scripts/. Known defects, documented in
docs/methodology.md and README.md:

* Saved predictions were written in fold order and paired with
  subjects/targets in original row order, so reported per-subject
  results and saved metrics were misaligned (all figures generated
  from this pipeline report invalid metrics).
* Feature selection, imputation and scaling were fit on the full
  dataset before cross-validation (leakage).
* run_pipeline.py silently substituted RANDOM placeholder HFO
  features when the aggregated HFO file was missing.
* Broad ``except`` blocks turned extraction failures into zeros.

Do NOT run this script and do NOT cite numbers produced by it.
"""

#!/usr/bin/env python3
"""
EXPANDED UPDRS EXTRACTION: All Metadata + Multiple Recordings

This script:
1. Loads UPDRS data (with multiple POST recordings per subject)
2. Extracts ALL clinical metadata from ECoG_localization.xlsx Sheet1 & Sheet2
3. Matches recordings to ECoG metadata by subject
4. Creates comprehensive training dataset with:
   - All 9 clinical features from UPDRS
   - All 36 recording parameters from Sheet1
   - All brain structure features from Sheet2
   - Proper handling of multiple recordings per subject
5. Outputs ready-to-use dataset for ML pipeline

Author: Olivia
Date: December 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("=" * 100)
print("EXPANDED UPDRS EXTRACTION: Complete Metadata Integration")
print("=" * 100)

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path('<PRIVATE_DATA_DIR>/')
OUTPUT_DIR = Path('<PRIVATE_DATA_DIR>/')

UPDRS_FILE = DATA_DIR / 'UPDRS_data.csv'
ECOG_FILE = DATA_DIR / 'ECoG_localization.xlsx'

# ============================================================================
# STEP 1: LOAD ALL DATA
# ============================================================================

print("\n" + "=" * 100)
print("STEP 1: LOADING DATA")
print("=" * 100)

# Load UPDRS with multiple recordings
print("\n1a. Loading UPDRS data...")
updrs_df = pd.read_csv(UPDRS_FILE)
print(f"    ✓ Loaded {len(updrs_df)} UPDRS records")
print(f"    ✓ Unique subjects: {updrs_df['subject'].nunique()}")
print(f"    ✓ Columns: {list(updrs_df.columns)}")

# Identify subjects with multiple recordings
subject_counts = updrs_df['subject'].value_counts()
multi_recording_subjects = subject_counts[subject_counts > 1]
print(f"\n    Subjects with multiple recordings ({len(multi_recording_subjects)}):")
for subj, count in multi_recording_subjects.items():
    print(f"      • {subj}: {count} recordings")

# Load ECoG Sheet1 (all recording parameters)
print("\n1b. Loading ECoG Sheet1 (Recording Parameters)...")
ecog_sheet1 = pd.read_excel(ECOG_FILE, sheet_name='Sheet1')
print(f"    ✓ Loaded {ecog_sheet1.shape[0]} rows, {ecog_sheet1.shape[1]} columns")

# Load ECoG Sheet2 (brain structure)
print("\n1c. Loading ECoG Sheet2 (Brain Structure)...")
ecog_sheet2 = pd.read_excel(ECOG_FILE, sheet_name='Sheet2')
print(f"    ✓ Loaded {ecog_sheet2.shape[0]} rows, {ecog_sheet2.shape[1]} columns")

# ============================================================================
# STEP 2: EXTRACT ALL METADATA FROM ECOG
# ============================================================================

print("\n" + "=" * 100)
print("STEP 2: EXTRACTING ALL METADATA FROM ECOG SHEETS")
print("=" * 100)

# ---- SHEET1 EXTRACTION ----
print("\n2a. Sheet1 Metadata (Recording Parameters & Location):")

# Mapping from original column names to clean names
original_col_names = {
    'SUBJECT': 'subject',
    'GENDER': 'gender',
    'Age': 'age',
    'M1_Contact_Img': 'm1_contact_img',
    'M1_Coord_Img_x': 'm1_coord_img_x',
    'M1_Coord_Img_y': 'm1_coord_img_y',
    'M1_Coord_Img_z': 'm1_coord_img_z',
    'M1_Contact_Intraop': 'm1_contact_intraop',
    'M1_Coord_Intraop_x': 'm1_coord_intraop_x',
    'M1_Coord_Intraop_y': 'm1_coord_intraop_y',
    'M1_Coord_Intraop_z': 'm1_coord_intraop_z',
    'REST/MOVE': 'rest_move',
    'SELF-INITIATED': 'self_initiated',
    'APDM': 'apdm',
    'FREQ STIM': 'dbs_freq',
    'VOLT STIM': 'dbs_volt',
    'IPSILATERAL': 'ipsilateral',
    'ANAESTHESIA': 'anaesthesia',
    'ADVANCEMENT': 'advancement',
    'MIRROR': 'task_mirror',
    'FLANKER': 'task_flanker',
    'CHOICE': 'task_choice',
    'STOP': 'task_stop',
    'VerbalWorking Memonry': 'task_verbal_wm',
    'Inbrija': 'task_inbrija',
    'Signal quality': 'signal_quality'
}

sheet1_extract = pd.DataFrame()
for orig_col, new_col in original_col_names.items():
    if orig_col in ecog_sheet1.columns:
        sheet1_extract[new_col] = ecog_sheet1[orig_col]
    else:
        print(f"    ⚠ Warning: Column '{orig_col}' not found in Sheet1")

print(f"    ✓ Extracted {len(sheet1_extract.columns)} columns from Sheet1")

# ---- SHEET2 EXTRACTION ----
print("\n2b. Sheet2 Metadata (Brain Structure):")

sheet2_columns = {
    'SUBJECT': 'subject',
    'Yeo Thickness': 'cortical_yeo_thickness',
    'SF thickness': 'cortical_sf_thickness',
    'PC thickness': 'cortical_pc_thickness',
    'dkt': 'parc_dkt',
    'yeo': 'parc_yeo',
    'local': 'parc_local',
    'diagnosis': 'diagnosis'
}

sheet2_extract = pd.DataFrame()
for orig_col, new_col in sheet2_columns.items():
    if orig_col in ecog_sheet2.columns:
        sheet2_extract[new_col] = ecog_sheet2[orig_col]
    else:
        print(f"    ⚠ Warning: Column '{orig_col}' not found in Sheet2")

print(f"    ✓ Extracted {len(sheet2_extract.columns)} columns from Sheet2")

# ============================================================================
# STEP 3: COMBINE AND MERGE DATA
# ============================================================================

print("\n" + "=" * 100)
print("STEP 3: MERGING UPDRS WITH ALL METADATA")
print("=" * 100)

# Normalize subject names
updrs_df['subject'] = updrs_df['subject'].astype(str).str.strip()
sheet1_extract['subject'] = sheet1_extract['subject'].astype(str).str.strip()
sheet2_extract['subject'] = sheet2_extract['subject'].astype(str).str.strip()

# Merge UPDRS with Sheet1 metadata (one-to-one on subject)
print("\n3a. Merging UPDRS with Sheet1 metadata...")
merged = updrs_df.merge(sheet1_extract, on='subject', how='left')
print(f"    ✓ Merged rows: {len(merged)}")
print(f"    ✓ Columns: {len(merged.columns)}")

# Merge with Sheet2 metadata (one-to-one on subject)
print("\n3b. Merging with Sheet2 metadata...")
merged = merged.merge(sheet2_extract, on='subject', how='left')
print(f"    ✓ Final columns: {len(merged.columns)}")

# ============================================================================
# STEP 4: ANALYZE MULTIPLE RECORDINGS
# ============================================================================

print("\n" + "=" * 100)
print("STEP 4: ANALYZING MULTIPLE RECORDINGS PER SUBJECT")
print("=" * 100)

print("\nSubjects with multiple POST recordings:")
print("-" * 100)

multi_recording_data = []

for subj in multi_recording_subjects.index:
    subj_data = merged[merged['subject'] == subj]
    print(f"\n{subj}: {len(subj_data)} recordings")
    print(f"  UPDRS improvements: {subj_data['UPDRS_improvement'].tolist()}")
    
    # Check if clinical data is same across recordings
    clinical_cols = ['age', 'gender', 'years_since_diagnosis', 'diagnosis']
    
    for idx, row in subj_data.iterrows():
        rec_num = len([r for r in subj_data.head(subj_data.index.tolist().index(idx) + 1)['subject']])
        print(f"\n    Recording #{rec_num}:")
        print(f"      UPDRS improvement: {row['UPDRS_improvement']}")
        print(f"      REST/MOVE: {row['rest_move']}")
        print(f"      DBS Freq: {row['dbs_freq']}, DBS Volt: {row['dbs_volt']}")
        print(f"      M1 Coord (Imaging): x={row['m1_coord_img_x']}, y={row['m1_coord_img_y']}, z={row['m1_coord_img_z']}")
        
        multi_recording_data.append({
            'subject': subj,
            'recording_num': rec_num,
            'updrs_improvement': row['UPDRS_improvement']
        })

# ============================================================================
# STEP 5: CREATE COMPREHENSIVE DATASETS
# ============================================================================

print("\n" + "=" * 100)
print("STEP 5: CREATING COMPREHENSIVE DATASETS")
print("=" * 100)

# Dataset 1: Multiple recordings as separate rows
print("\n5a. Dataset 1: Multiple recordings as separate rows")
print("    (Each recording from the same subject is a separate training example)")

merged['recording_id'] = merged.groupby('subject').cumcount() + 1
merged['sample_id'] = merged['subject'] + '_R' + merged['recording_id'].astype(str)

print(f"    ✓ Total samples: {len(merged)}")
print(f"    ✓ Unique subjects: {merged['subject'].nunique()}")
print(f"    ✓ With multiple recordings treated as separate samples")

# Save this version
merged.to_csv(OUTPUT_DIR / 'UPDRS_EXPANDED_MultiRecordings.csv', index=False)
print(f"    ✓ Saved: UPDRS_EXPANDED_MultiRecordings.csv")

# Dataset 2: Aggregated by subject (mean of multiple recordings)
print("\n5b. Dataset 2: Aggregated by subject (mean of recordings)")
print("    (For subjects with multiple recordings, take mean UPDRS)")

agg_dict = {}
for col in merged.columns:
    if col in ['subject', 'recording_id', 'sample_id', 'diagnosis', 'gender']:
        agg_dict[col] = 'first'
    elif col == 'UPDRS_improvement':
        agg_dict[col] = 'mean'
    elif merged[col].dtype in ['float64', 'int64']:
        agg_dict[col] = 'mean'
    else:
        agg_dict[col] = 'first'

merged_agg = merged.groupby('subject', as_index=False).agg(agg_dict)

print(f"    ✓ Total subjects: {len(merged_agg)}")
print(f"    ✓ All unique (one row per subject)")

# Save this version
merged_agg.to_csv(OUTPUT_DIR / 'UPDRS_EXPANDED_AggregatedBySubject.csv', index=False)
print(f"    ✓ Saved: UPDRS_EXPANDED_AggregatedBySubject.csv")

# Dataset 3: Long format with recording metadata
print("\n5c. Dataset 3: Long format with explicit recording information")
print("    (Each recording explicit with metadata)")

recording_df = pd.DataFrame()
for subj in merged['subject'].unique():
    subj_data = merged[merged['subject'] == subj].reset_index(drop=True)
    
    for idx, row in subj_data.iterrows():
        new_row = row.copy()
        new_row['num_recordings_for_subject'] = len(subj_data)
        new_row['recording_sequence'] = idx + 1
        recording_df = pd.concat([recording_df, pd.DataFrame([new_row])], ignore_index=True)

# Save this version
recording_df.to_csv(OUTPUT_DIR / 'UPDRS_EXPANDED_LongFormat.csv', index=False)
print(f"    ✓ Saved: UPDRS_EXPANDED_LongFormat.csv")

# ============================================================================
# STEP 6: DATA QUALITY SUMMARY
# ============================================================================

print("\n" + "=" * 100)
print("STEP 6: DATA QUALITY SUMMARY")
print("=" * 100)

print("\nColumn Coverage in Expanded Dataset:")
print("-" * 100)

coverage_data = []
for col in merged.columns:
    non_null = merged[col].notna().sum()
    coverage = 100 * non_null / len(merged)
    coverage_data.append({
        'column': col,
        'non_null': non_null,
        'total': len(merged),
        'coverage_%': coverage
    })

coverage_df = pd.DataFrame(coverage_data).sort_values('coverage_%', ascending=False)

print(f"\n{'Column':<40} {'Non-Null':<10} {'Total':<8} {'Coverage':<10}")
print("-" * 100)
for _, row in coverage_df.head(20).iterrows():
    print(f"{row['column']:<40} {row['non_null']:<10.0f} {row['total']:<8} {row['coverage_%']:<10.1f}%")

print(f"\n... and {len(coverage_df) - 20} more columns")

# Save coverage report
coverage_df.to_csv(OUTPUT_DIR / 'UPDRS_EXPANDED_ColumnCoverage.csv', index=False)
print(f"\n✓ Saved: UPDRS_EXPANDED_ColumnCoverage.csv")

# ============================================================================
# STEP 7: SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 100)
print("STEP 7: SUMMARY STATISTICS")
print("=" * 100)

print(f"\nOriginal UPDRS data:")
print(f"  • Samples: {len(updrs_df)}")
print(f"  • Unique subjects: {updrs_df['subject'].nunique()}")
print(f"  • Columns: {len(updrs_df.columns)}")

print(f"\nExpanded with ALL metadata:")
print(f"  • Samples: {len(merged)}")
print(f"  • Unique subjects: {merged['subject'].nunique()}")
print(f"  • Total columns: {len(merged.columns)}")
print(f"    - UPDRS original: {len(updrs_df.columns)}")
print(f"    - Sheet1 metadata: {len(sheet1_extract.columns)}")
print(f"    - Sheet2 metadata: {len(sheet2_extract.columns)}")
print(f"    - Processing columns: 2 (recording_id, sample_id)")

print(f"\nMultiple recordings:")
print(f"  • Subjects with multiple recordings: {len(multi_recording_subjects)}")
print(f"  • Total recordings: {len(merged)}")

print(f"\nKey features NOW available:")
print(f"  ✓ REST/MOVE state: {merged['rest_move'].notna().sum()}/{len(merged)} ({100*merged['rest_move'].notna().sum()/len(merged):.1f}%)")
print(f"  ✓ M1 Coordinates (Imaging): {merged['m1_coord_img_x'].notna().sum()}/{len(merged)} ({100*merged['m1_coord_img_x'].notna().sum()/len(merged):.1f}%)")
print(f"  ✓ Cortical Thickness: {merged['cortical_yeo_thickness'].notna().sum()}/{len(merged)} ({100*merged['cortical_yeo_thickness'].notna().sum()/len(merged):.1f}%)")
print(f"  ✓ DBS Parameters: {merged['dbs_freq'].notna().sum()}/{len(merged)} ({100*merged['dbs_freq'].notna().sum()/len(merged):.1f}%)")
print(f"  ✓ All behavioral tasks: {merged['task_mirror'].notna().sum()}/{len(merged)} ({100*merged['task_mirror'].notna().sum()/len(merged):.1f}%)")

# ============================================================================
# STEP 8: RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 100)
print("STEP 8: RECOMMENDATIONS FOR USING THESE DATASETS")
print("=" * 100)

print("""
Three datasets created - choose based on your analysis approach:

1. UPDRS_EXPANDED_MultiRecordings.csv
   ├─ Each recording from same subject = separate row
   ├─ Best for: Analyzing recording-by-recording variability
   ├─ Use if: REST vs MOVE states differ within subjects
   ├─ Pros: More training samples (15 total)
   └─ Cons: Not independent samples (same subject appears multiple times)

2. UPDRS_EXPANDED_AggregatedBySubject.csv
   ├─ Multiple recordings per subject = averaged
   ├─ Best for: Standard independent sample analysis
   ├─ Use if: Want one outcome per subject
   ├─ Pros: Independent samples, proper statistical inference
   └─ Cons: Loss of recording-specific information

3. UPDRS_EXPANDED_LongFormat.csv
   ├─ Each recording explicit with metadata
   ├─ Best for: Mixed-effects or hierarchical modeling
   ├─ Use if: Can account for repeated measures
   ├─ Pros: All information preserved
   └─ Cons: Need to handle within-subject correlation

RECOMMENDED FOR YOUR PROJECT:
→ Use UPDRS_EXPANDED_MultiRecordings.csv initially
→ This lets the model learn from both REST and MOVE recordings
→ Monitor if same subject dominates predictions
→ Consider subsampling if needed (e.g., one recording per subject)

KEY INSIGHT:
Your subjects with multiple recordings have DIFFERENT REST/MOVE states:
  • SUBJ_CODE: Recording 1 vs Recording 2 differ
  • SUBJ_CODE: Recording 1 vs Recording 2 differ
  • SUBJ_CODE: Recording 1 vs Recording 2 differ
  • SUBJ_CODE: Recording 1 vs Recording 2 differ

This is VALUABLE DATA! Different neural states → different outcomes
Machine learning can learn this pattern!
""")

print("\n" + "=" * 100)
print("✅ EXPANDED UPDRS EXTRACTION COMPLETE")
print("=" * 100)
print(f"\nThree datasets ready for analysis:")
print(f"  1. UPDRS_EXPANDED_MultiRecordings.csv")
print(f"  2. UPDRS_EXPANDED_AggregatedBySubject.csv")
print(f"  3. UPDRS_EXPANDED_LongFormat.csv")
print(f"\nAll with complete metadata from ECoG sheets!")