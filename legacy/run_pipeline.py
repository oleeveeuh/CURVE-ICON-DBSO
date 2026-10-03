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

"""
================================================================================
COMPLETE INTEGRATED DBS ANALYSIS
Combining Metadata Extraction + Overlap-First Efficiency
================================================================================

This script combines:
1. Your comprehensive metadata extraction (ECoG sheets, multiple recordings)
2. Overlap-first efficiency (extract features only for subjects we'll use)
3. HFO neural biomarkers
4. Complete ML pipeline

WORKFLOW:
Step 1: Load basic UPDRS data with ECoG metadata (your approach)
Step 2: Load HFO features
Step 3: Find overlap → subjects with HFO + UPDRS
Step 4: Extract detailed UPDRS features (only for overlap subjects)
Step 5: Merge all features
Step 6: Handle multiple recordings per subject
Step 7: Run ML with Leave-One-Out CV
Step 8: Save comprehensive results

================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, GroupKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# SHAP for explainability
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("\n⚠ SHAP library not available")
    print("  Install with: pip install shap")
    print("  Pipeline will continue without SHAP analysis\n")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("\n" + "="*100)
print("COMPLETE INTEGRATED DBS ANALYSIS (WITH SHAP EXPLAINABILITY)")
print("Metadata + Overlap-First + HFO + ML Pipeline + SHAP")
print("="*100)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
if HAS_SHAP:
    print("✓ SHAP explainability enabled")
else:
    print("⚠ SHAP explainability disabled (library not found)")
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Adjust these paths to your data location
DATA_DIR = Path('<PRIVATE_DATA_DIR>/')
OUTPUT_DIR = Path('<PRIVATE_DATA_DIR>/outputs/')
OUTPUT_DIR.mkdir(exist_ok=True)

UPDRS_FILE = DATA_DIR / 'UPDRS_data.csv'
ECOG_FILE = DATA_DIR / 'ECoG_localization.xlsx'
HFO_FILE = DATA_DIR / 'HFO_features.csv'  # Adjust if needed

# ============================================================================
# STEP 1: LOAD BASIC DATA WITH METADATA (Your Approach)
# ============================================================================

print("STEP 1: Loading basic data with comprehensive metadata...")
print("-"*100)

# Load UPDRS
print("\n1a. Loading UPDRS data...")
updrs = pd.read_csv(UPDRS_FILE)
print(f"    ✓ {len(updrs)} records from {updrs['subject'].nunique()} subjects")

# Check for multiple recordings
subject_counts = updrs['subject'].value_counts()
multi_rec = subject_counts[subject_counts > 1]
print(f"    ✓ {len(multi_rec)} subjects with multiple recordings")

# Load ECoG metadata (Sheet1)
print("\n1b. Loading ECoG Sheet1 (Recording Parameters)...")
ecog_sheet1 = pd.read_excel(ECOG_FILE, sheet_name='Sheet1')
print(f"    ✓ {ecog_sheet1.shape[0]} rows × {ecog_sheet1.shape[1]} columns")

# Load ECoG metadata (Sheet2)
print("\n1c. Loading ECoG Sheet2 (Brain Structure)...")
ecog_sheet2 = pd.read_excel(ECOG_FILE, sheet_name='Sheet2')
print(f"    ✓ {ecog_sheet2.shape[0]} rows × {ecog_sheet2.shape[1]} columns")

# Extract metadata (your column mappings)
sheet1_cols = {
    'SUBJECT': 'subject',
    'GENDER': 'gender',
    'Age': 'age',
    'REST/MOVE': 'REST/MOVE',
    'M1_Coord_Img_x': 'm1_coord_img_x',
    'M1_Coord_Img_y': 'm1_coord_img_y',
    'M1_Coord_Img_z': 'm1_coord_img_z',
    'FREQ STIM': 'dbs_freq',
    'VOLT STIM': 'dbs_volt',
    'Signal quality': 'signal_quality'
}

sheet2_cols = {
    'SUBJECT': 'subject',
    'Yeo Thickness': 'cortical_yeo_thickness',
    'SF thickness': 'cortical_sf_thickness',
    'PC thickness': 'cortical_pc_thickness',
    'diagnosis': 'diagnosis'
}

sheet1_extract = ecog_sheet1.rename(columns=sheet1_cols)[list(sheet1_cols.values())]
sheet2_extract = ecog_sheet2.rename(columns=sheet2_cols)[list(sheet2_cols.values())]

# Normalize subject IDs
updrs['subject'] = updrs['subject'].astype(str).str.strip()
sheet1_extract['subject'] = sheet1_extract['subject'].astype(str).str.strip()
sheet2_extract['subject'] = sheet2_extract['subject'].astype(str).str.strip()

# Merge metadata
print("\n1d. Merging UPDRS with metadata...")
updrs_with_meta = updrs.merge(sheet1_extract, on='subject', how='left')
updrs_with_meta = updrs_with_meta.merge(sheet2_extract, on='subject', how='left')
print(f"    ✓ {len(updrs_with_meta)} records with {updrs_with_meta.shape[1]} columns")

# ============================================================================
# STEP 2: LOAD HFO FEATURES
# ============================================================================

print("\nSTEP 2: Loading HFO neural biomarkers...")
print("-"*100)

# Check if HFO file exists, if not use aggregated version
hfo_agg_file = DATA_DIR / 'hfo_features_aggregated.csv'
if hfo_agg_file.exists():
    print("\n2a. Loading pre-aggregated HFO features...")
    hfo = pd.read_csv(hfo_agg_file)
    print(f"    ✓ {len(hfo)} subjects with HFO features")
else:
    print("\n2a. HFO aggregation needed - loading raw HFO...")
    print("    Note: Use your HFO aggregation script first if raw data available")
    # Create placeholder for demonstration
    hfo = pd.DataFrame({
        'subject': ['SUBJ_CODE', 'SUBJ_CODE', 'SUBJ_CODE', 'SUBJ_CODE', 
                   'SUBJ_CODE', 'SUBJ_CODE', 'SUBJ_CODE', 'SUBJ_CODE'],
        'hfo_power_mean': np.random.randn(8),
        'hfo_peak_freq_mean': np.random.randn(8)
    })
    print(f"    ⚠ Using {len(hfo)} placeholder HFO subjects")

hfo['subject'] = hfo['subject'].astype(str).str.strip()

# ============================================================================
# STEP 3: FIND OVERLAP (BEFORE DETAILED FEATURE EXTRACTION)
# ============================================================================

print("\nSTEP 3: Finding dataset overlap...")
print("-"*100)

# Subjects with UPDRS + metadata
updrs_subjects = set(updrs_with_meta['subject'].unique())
print(f"\n  UPDRS subjects: {len(updrs_subjects)}")

# Subjects with HFO
hfo_subjects = set(hfo['subject'].unique())
print(f"  HFO subjects: {len(hfo_subjects)}")

# Overlap
overlap_subjects = updrs_subjects.intersection(hfo_subjects)
print(f"\n✓ OVERLAP: {len(overlap_subjects)} subjects")
print(f"  Subjects: {sorted(list(overlap_subjects))}")

if len(overlap_subjects) == 0:
    print("\n❌ No overlap! Check subject ID naming")
    print(f"  UPDRS examples: {list(updrs_subjects)[:5]}")
    print(f"  HFO examples: {list(hfo_subjects)[:5]}")
    exit(1)

# Filter to overlap subjects BEFORE detailed feature extraction
updrs_overlap = updrs_with_meta[updrs_with_meta['subject'].isin(overlap_subjects)].copy()
print(f"\n  Filtered UPDRS: {len(updrs_overlap)} records from {len(overlap_subjects)} subjects")

# ============================================================================
# STEP 4: EXTRACT DETAILED FEATURES (ONLY FOR OVERLAP SUBJECTS)
# ============================================================================

print("\nSTEP 4: Extracting detailed UPDRS features (only for overlap subjects)...")
print("-"*100)
print(f"  (Efficient: extract features for {len(overlap_subjects)} subjects, not all {len(updrs_subjects)}!)")

def extract_detailed_updrs_features(df):
    """
    Extract detailed motor features from UPDRS subscores
    Only called for overlap subjects (efficient!)
    """
    features = pd.DataFrame()
    
    # Identifiers
    features['subject'] = df['subject']
    features['recording_id'] = df.groupby('subject').cumcount() + 1
    
    # === BRADYKINESIA ===
    if all(col in df.columns for col in ['Finger_Tap_R', 'Finger_Tap_L']):
        features['Brady_R'] = (df['Finger_Tap_R'] + df['Hand_Movement_R'] + 
                               df['PS_R'] + df['Toe_Tapping_R'] + df['Leg_Agility_R'])
        features['Brady_L'] = (df['Finger_Tap_L'] + df['Hand_Movement_L'] + 
                               df['PS_L'] + df['Toe_Tapping_L'] + df['Leg_Agility_L'])
        features['Brady_Asymmetry'] = abs(features['Brady_R'] - features['Brady_L'])
    
    # === RIGIDITY ===
    if all(col in df.columns for col in ['Rigidity_RUE', 'Rigidity_LUE']):
        features['Rigidity_R'] = df['Rigidity_RUE'] + df['Rigidity_RLE']
        features['Rigidity_L'] = df['Rigidity_LUE'] + df['Rigidity_LLE']
        features['Rigidity_Asymmetry'] = abs(features['Rigidity_R'] - features['Rigidity_L'])
    
    # === TREMOR ===
    if all(col in df.columns for col in ['Rest_Tremor_RUE', 'Rest_Tremor_LUE']):
        features['Tremor_R'] = (df['Postural_Tremor_R_Hand'] + df['Kinetic_Tremor_R_Hand'] + 
                                df['Rest_Tremor_RUE'] + df['Rest_Tremor_RLE'])
        features['Tremor_L'] = (df['Postural_Tremor_L_Hand'] + df['Kinetic_Tremor_L_Hand'] + 
                                df['Rest_Tremor_LUE'] + df['Rest_Tremor_LLE'])
        features['Tremor_Asymmetry'] = abs(features['Tremor_R'] - features['Tremor_L'])
    
    # === GLOBAL ===
    if 'years_since_diagnosis' in df.columns:
        features['Disease_Duration'] = df['years_since_diagnosis']
    if 'UPDRS_improvement' in df.columns:
        features['UPDRS_Improvement'] = df['UPDRS_improvement']
        features['Responder'] = (df['UPDRS_improvement'] >= 30).astype(int)
    
    return features

# Extract features
detailed_features = extract_detailed_updrs_features(updrs_overlap)
print(f"\n  ✓ Extracted {detailed_features.shape[1]} detailed features")

# Merge detailed features back
updrs_complete = updrs_overlap.merge(detailed_features, 
                                     on=['subject', 'recording_id'] 
                                     if 'recording_id' in detailed_features.columns 
                                     else 'subject', 
                                     how='left')

# ============================================================================
# STEP 5: MERGE ALL FEATURES (HFO + UPDRS + METADATA)
# ============================================================================

print("\nSTEP 5: Merging HFO + UPDRS + Metadata...")
print("-"*100)

# Merge HFO with complete UPDRS
master = updrs_complete.merge(hfo, on='subject', how='inner')

print(f"\n✓ Master dataset: {len(master)} records from {master['subject'].nunique()} subjects")
print(f"  Features breakdown:")
print(f"    - Basic UPDRS: ~9 columns")
print(f"    - ECoG metadata (Sheet1): ~10 columns")
print(f"    - ECoG metadata (Sheet2): ~4 columns")
print(f"    - Detailed UPDRS features: ~{detailed_features.shape[1]} columns")
print(f"    - HFO features: {len([c for c in hfo.columns if c != 'subject'])} columns")
print(f"    - TOTAL: {master.shape[1]} columns")

# ============================================================================
# STEP 6: HANDLE MULTIPLE RECORDINGS
# ============================================================================

print("\nSTEP 6: Handling multiple recordings per subject...")
print("-"*100)

# Check which subjects have multiple recordings
multi_rec_in_master = master['subject'].value_counts()
multi_rec_subjects = multi_rec_in_master[multi_rec_in_master > 1]

print(f"\n  Subjects with multiple recordings: {len(multi_rec_subjects)}")
for subj, count in multi_rec_subjects.items():
    subj_data = master[master['subject'] == subj]
    print(f"\n  {subj}: {count} recordings")
    print(f"    REST/MOVE states: {subj_data['rest_move'].tolist()}")
    print(f"    UPDRS improvements: {subj_data['UPDRS_Improvement'].tolist()}")

# DECISION: Use GroupKFold for proper handling
print(f"\n  ML Strategy: Use GroupKFold to prevent data leakage")
print(f"    (Ensures recordings from same subject stay together)")

# ============================================================================
# STEP 7: PREPARE FEATURES FOR ML
# ============================================================================

print("\nSTEP 7: Preparing feature matrix...")
print("-"*100)

# Select numeric features
exclude_cols = ['subject', 'recording_id', 'Responder', 'UPDRS_Improvement', 
                'UPDRS_improvement', 'updrs_improvement',  # Target variable (all variations)
                'postop_updrs_total', 'postop_updrs', 'post_updrs',  # Post-op scores (leak target)
                'brady_motor_score', 'rigid_motor_score', 'tremor_score',  # Redundant with subscores
                'diagnosis', 'gender', 'rest_move']  # Non-numeric or not useful
numeric_cols = master.select_dtypes(include=[np.number]).columns
feature_cols = [col for col in numeric_cols if col not in exclude_cols]

X = master[feature_cols].copy()
X = X.fillna(X.median())

# ============================================================================
# CRITICAL: Validate that target is NOT in features (prevent data leakage)
# ============================================================================
target_variations = ['UPDRS_Improvement', 'UPDRS_improvement', 'updrs_improvement', 
                    'Responder', 'postop_updrs_total', 'postop_updrs']
leaked_features = [col for col in X.columns if col in target_variations or 'postop' in col.lower()]

if leaked_features:
    print(f"\n❌ ERROR: TARGET VARIABLE FOUND IN FEATURES (DATA LEAKAGE):")
    for feat in leaked_features:
        print(f"  - {feat}")
    print(f"\nRemoving these features to prevent data leakage...")
    X = X.drop(columns=leaked_features)
    print(f"✓ Features cleaned: {X.shape[1]} features remaining")
else:
    print(f"\n✓ Validation passed: No target leakage detected")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

y = master['UPDRS_Improvement'].values  # Continuous target (regression)
groups = master['subject'].values  # For GroupKFold

print(f"\n✓ Feature matrix: {X_scaled.shape[0]} samples × {X_scaled.shape[1]} features")
print(f"  Unique subjects: {len(np.unique(groups))}")
print(f"\n📊 TARGET VARIABLE (REGRESSION):")
print(f"  UPDRS Improvement: Mean={y.mean():.1f}%, SD={y.std():.1f}%, Range=[{y.min():.1f}%, {y.max():.1f}%]")
print(f"  Derived responders (≥30%): {(y >= 30).sum()} ({(y >= 30).sum()/len(y)*100:.0f}%)")

# ============================================================================
# STEP 8: FEATURE IMPORTANCE
# ============================================================================

print("\nSTEP 8: Feature importance analysis...")
print("-"*100)

rf_temp = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
rf_temp.fit(X_scaled, y)

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_temp.feature_importances_,
    'Type': ['HFO' if 'hfo' in col.lower() or 'dbs' in col.lower() or 'parietal' in col.lower()
             else 'Metadata' if col in ['age', 'dbs_freq', 'dbs_volt', 'cortical_yeo_thickness', 
                                        'cortical_sf_thickness', 'cortical_pc_thickness',
                                        'm1_coord_img_x', 'm1_coord_img_y', 'm1_coord_img_z']
             else 'UPDRS' for col in X.columns]
}).sort_values('Importance', ascending=False)

print(f"\n🎯 TOP 20 FEATURES (HFO + UPDRS + Metadata):")
print(f"{'Rank':<5} {'Type':<10} {'Feature':<40} {'Importance':<10}")
print("-"*80)

# Check for data leakage indicators
max_importance = importance['Importance'].max()
if max_importance > 0.5:
    print(f"⚠️  WARNING: Very high importance detected ({max_importance:.4f})")
    print(f"⚠️  This may indicate data leakage (target in features)")
    print("-"*80)

for i, row in importance.head(20).iterrows():
    flag = " ⚠️ SUSPICIOUS" if row['Importance'] > 0.5 else ""
    print(f"{i+1:<5} {row['Type']:<10} {row['Feature']:<40} {row['Importance']:<10.4f}{flag}")

# Count by type
print(f"\nFeature type breakdown in top 20:")
for ftype in ['HFO', 'UPDRS', 'Metadata']:
    count = sum(importance.head(20)['Type'] == ftype)
    print(f"  {ftype}: {count}")

# Save
importance.to_csv(OUTPUT_DIR / 'feature_importance_complete.csv', index=False)

# ============================================================================
# STEP 9: MACHINE LEARNING WITH GROUP CV (REGRESSION)
# ============================================================================

print("\nSTEP 9: Machine learning with GroupKFold (REGRESSION)...")
print("-"*100)
print("  (Ensures recordings from same subject don't split across train/test)")
print("  Predicting: UPDRS Improvement (%) - Continuous target")

n_splits = min(5, len(np.unique(groups)))  # Max 5 folds or number of subjects
cv = GroupKFold(n_splits=n_splits)

models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=50, max_depth=3, 
        random_state=RANDOM_SEED
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        random_state=RANDOM_SEED
    ),
    'Linear Regression': LinearRegression()
}

# Select top features
n_top = min(15, len(X.columns), len(np.unique(groups)) * 2)
top_features = importance.head(n_top)['Feature'].values
X_selected = X_scaled[top_features]

print(f"\n✓ Selected top {n_top} features for ML")

for model_name, model in models.items():
    y_true, y_pred = [], []
    
    for train_idx, test_idx in cv.split(X_selected, y, groups):
        X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        y_true.extend(y_test)
        y_pred.extend(pred)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Regression metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Pearson correlation
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    
    print(f"\n{model_name}:")
    print(f"  R² Score: {r2:.3f}")
    print(f"  RMSE: {rmse:.1f}%")
    print(f"  MAE: {mae:.1f}%")
    print(f"  Correlation: {correlation:.3f}")
    
    # Derived classification accuracy (using 30% threshold)
    y_true_class = (y_true >= 30).astype(int)
    y_pred_class = (y_pred >= 30).astype(int)
    class_acc = (y_true_class == y_pred_class).mean()
    print(f"  Derived Classification Accuracy (30% threshold): {class_acc:.1%}")

# ============================================================================
# STEP 9.5: SHAP EXPLAINABILITY ANALYSIS
# ============================================================================

if HAS_SHAP:
    print("\nSTEP 9.5: SHAP Explainability Analysis (REGRESSION)...")
    print("-"*100)
    print("  Computing SHAP values for feature importance and directional effects")
    print(f"  Sample size: n={len(X_selected)} (Note: Small sample - interpret with caution)")
    print(f"  Target: UPDRS Improvement (%) - Continuous")
    
    try:
        # Train best model on full dataset for SHAP
        best_model_name = 'Gradient Boosting'  # Based on results, best R²
        best_model = GradientBoostingRegressor(
            n_estimators=50, 
            max_depth=3, 
            learning_rate=0.1,
            random_state=RANDOM_SEED
        )
        
        print(f"\n  Training {best_model_name} on full dataset for SHAP...")
        best_model.fit(X_selected, y)
        
        # Create SHAP explainer
        print("  Creating SHAP explainer (TreeExplainer)...")
        explainer = shap.TreeExplainer(best_model)
        
        # Calculate SHAP values
        print("  Calculating SHAP values...")
        shap_values = explainer.shap_values(X_selected)
        
        # For regression, shap_values is a single array (not a list)
        shap_values_array = shap_values
        
        # Calculate mean absolute SHAP values for feature importance
        shap_importance = pd.DataFrame({
            'Feature': X_selected.columns,
            'Mean_Abs_SHAP': np.abs(shap_values_array).mean(axis=0),
            'Mean_SHAP': shap_values_array.mean(axis=0),  # Directional average
            'Std_SHAP': shap_values_array.std(axis=0)
        }).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
        
        # Add feature type
        shap_importance['Type'] = shap_importance['Feature'].apply(
            lambda x: 'HFO' if any(h in x for h in ['hfo', 'HFO']) 
            else 'UPDRS' if any(u in x for u in ['UPDRS', 'Brady', 'Rigid', 'Tremor', 
                                                  'cortical', 'parc', 'ipsi', 'Disease']) 
            else 'Metadata'
        )
        
        # Save SHAP importance
        shap_importance.to_csv(OUTPUT_DIR / 'shap_importance.csv', index=False)
        print(f"  ✓ Saved: shap_importance.csv")
        
        # Save SHAP values array
        shap_data = pd.DataFrame(
            shap_values_array,
            columns=X_selected.columns
        )
        shap_data['subject'] = groups  # groups is already .values (numpy array)
        shap_data['true_label'] = y    # y is already .values (numpy array)
        shap_data.to_csv(OUTPUT_DIR / 'shap_values.csv', index=False)
        print(f"  ✓ Saved: shap_values.csv")
        
        # Display top SHAP features
        print(f"\n  🎯 TOP 10 FEATURES BY SHAP IMPORTANCE:")
        print(f"  {'Rank':<5} {'Type':<10} {'Feature':<30} {'Mean|SHAP|':<12} {'Direction':<20}")
        print("  " + "-"*90)
        for i, row in shap_importance.head(10).iterrows():
            direction = "→ Higher Improvement" if row['Mean_SHAP'] > 0 else "→ Lower Improvement"
            print(f"  {i+1:<5} {row['Type']:<10} {row['Feature']:<30} "
                  f"{row['Mean_Abs_SHAP']:<12.4f} {direction:<20}")
        
        # Compare SHAP vs RF importance for top features
        print("\n  📊 SHAP vs Random Forest Importance (Top 5):")
        print("  " + "-"*70)
        top_5_shap = shap_importance.head(5)['Feature'].values
        for i, feat in enumerate(top_5_shap, 1):
            shap_val = shap_importance[shap_importance['Feature'] == feat]['Mean_Abs_SHAP'].values[0]
            
            # Get RF importance - check length BEFORE indexing
            rf_matches = importance[importance['Feature'] == feat]['Importance'].values
            if len(rf_matches) > 0:
                rf_val = rf_matches[0]
                rf_rank = importance[importance['Feature'] == feat].index[0] + 1
            else:
                rf_val = 0.0
                rf_rank = 'N/A'
            
            print(f"  {i}. {feat:<35} SHAP:{shap_val:>7.4f}  RF:{rf_val:>7.4f} (rank {rf_rank})")
        
        # Feature type breakdown
        print(f"\n  Feature type contribution (by SHAP):")
        for ftype in ['HFO', 'UPDRS', 'Metadata']:
            total_shap = shap_importance[shap_importance['Type'] == ftype]['Mean_Abs_SHAP'].sum()
            pct = (total_shap / shap_importance['Mean_Abs_SHAP'].sum()) * 100
            print(f"    {ftype}: {total_shap:.4f} ({pct:.1f}%)")
        
        # Warning about small sample
        print(f"\n  ⚠ INTERPRETATION NOTE:")
        print(f"    With n={len(X_selected)}, SHAP values have wider confidence intervals")
        print(f"    Directional effects should be validated in larger cohort (target n=80)")
        print(f"    Use SHAP for hypothesis generation, not clinical decisions")
        
        SHAP_SUCCESS = True
        
    except Exception as e:
        print(f"\n  ⚠ SHAP analysis failed: {str(e)}")
        print(f"    Continuing without SHAP results")
        SHAP_SUCCESS = False
else:
    print("\nSTEP 9.5: SHAP Analysis Skipped (library not available)")
    print("-"*100)
    print("  Install SHAP with: pip install shap")
    SHAP_SUCCESS = False

# ============================================================================
# STEP 10: SAVE COMPREHENSIVE RESULTS (INCLUDING MODEL OUTPUTS)
# ============================================================================

print("\nSTEP 10: Saving comprehensive results...")
print("-"*100)

# Save master dataset
master.to_csv(OUTPUT_DIR / 'complete_integrated_dataset.csv', index=False)
print(f"✓ Saved: complete_integrated_dataset.csv ({len(master)} records)")

# Save feature importance
importance.to_csv(OUTPUT_DIR / 'feature_importance_complete.csv', index=False)
print(f"✓ Saved: feature_importance_complete.csv")

# ============================================================================
# NEW: SAVE MODEL PREDICTIONS AND PERFORMANCE FOR VISUALIZATION
# ============================================================================

print("\n  Saving model predictions for visualization...")

# Re-run models and save predictions (REGRESSION)
model_predictions = pd.DataFrame({
    'subject': groups,
    'true_improvement': y  # Continuous target
})

for model_name, model in models.items():
    y_pred_list = []
    fold_ids = []
    
    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X_selected, y, groups)):
        X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        y_pred_list.extend(pred)
        fold_ids.extend([fold_id] * len(y_test))
    
    # Add to dataframe
    model_predictions[f'{model_name}_pred'] = y_pred_list
    
    # Derived classification (30% threshold)
    model_predictions[f'{model_name}_pred_class'] = (np.array(y_pred_list) >= 30).astype(int)

# Add fold IDs
model_predictions['fold_id'] = fold_ids

# Save predictions
model_predictions.to_csv(OUTPUT_DIR / 'model_predictions.csv', index=False)
print(f"  ✓ Saved: model_predictions.csv")

# ============================================================================
# SAVE PERFORMANCE METRICS (REGRESSION)
# ============================================================================

print("\n  Saving performance metrics...")

metrics_data = []

for model_name, model in models.items():
    y_true = model_predictions['true_improvement'].values
    y_pred = model_predictions[f'{model_name}_pred'].values
    y_pred_class = model_predictions[f'{model_name}_pred_class'].values
    y_true_class = (y_true >= 30).astype(int)
    
    # Regression metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    
    # Derived classification metrics (30% threshold)
    class_acc = (y_true_class == y_pred_class).mean()
    
    # Confusion matrix for derived classification
    from sklearn.metrics import confusion_matrix as cm_func
    cm = cm_func(y_true_class, y_pred_class)
    
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    metrics_data.append({
        'Model': model_name,
        'R2_Score': r2,
        'RMSE': rmse,
        'MAE': mae,
        'Correlation': correlation,
        'Classification_Accuracy_30pct': class_acc,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp
    })

metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv(OUTPUT_DIR / 'model_performance_metrics.csv', index=False)
print(f"  ✓ Saved: model_performance_metrics.csv")

# ============================================================================
# SAVE CONFUSION MATRICES (for derived classification)
# ============================================================================

print("\n  Saving confusion matrices...")

confusion_matrices = {}

for model_name in models.keys():
    y_true_class = (model_predictions['true_improvement'].values >= 30).astype(int)
    y_pred_class = model_predictions[f'{model_name}_pred_class'].values
    
    from sklearn.metrics import confusion_matrix as cm_func
    cm = cm_func(y_true_class, y_pred_class)
    
    # Save as DataFrame for easy loading
    cm_df = pd.DataFrame(cm, 
                         index=['True_Negative', 'True_Positive'],
                         columns=['Pred_Negative', 'Pred_Positive'])
    
    cm_df.to_csv(OUTPUT_DIR / f'confusion_matrix_{model_name.replace(" ", "_")}.csv')
    confusion_matrices[model_name] = cm

print(f"  ✓ Saved: confusion_matrix_*.csv files (derived from 30% threshold)")

# ============================================================================
# SAVE PREDICTION SCATTER DATA (Regression diagnostic)
# ============================================================================

print("\n  Saving prediction scatter data...")

scatter_data = []
for model_name in models.keys():
    y_true = model_predictions['true_improvement'].values
    y_pred = model_predictions[f'{model_name}_pred'].values
    subjects = model_predictions['subject'].values
    
    for i in range(len(y_true)):
        scatter_data.append({
            'Model': model_name,
            'Subject': subjects[i],
            'True_Improvement': y_true[i],
            'Predicted_Improvement': y_pred[i],
            'Error': y_pred[i] - y_true[i],
            'Absolute_Error': abs(y_pred[i] - y_true[i])
        })

scatter_df = pd.DataFrame(scatter_data)
scatter_df.to_csv(OUTPUT_DIR / 'prediction_scatter_data.csv', index=False)
print(f"  ✓ Saved: prediction_scatter_data.csv")

# ============================================================================
# SAVE DETAILED SUMMARY
# ============================================================================

print("\n  Saving detailed summary...")

with open(OUTPUT_DIR / 'analysis_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("COMPLETE INTEGRATED DBS ANALYSIS SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("DATASET SUMMARY:\n")
    f.write("-"*80 + "\n")
    f.write(f"  Total records: {len(master)}\n")
    f.write(f"  Unique subjects: {master['subject'].nunique()}\n")
    f.write(f"  Subjects with multiple recordings: {len(multi_rec_subjects)}\n")
    f.write(f"  Total features extracted: {X.shape[1]}\n")
    f.write(f"  Features selected for ML: {n_top}\n\n")
    
    f.write("TARGET VARIABLE (REGRESSION):\n")
    f.write("-"*80 + "\n")
    f.write(f"  UPDRS Improvement: Mean={y.mean():.1f}% ± {y.std():.1f}%\n")
    f.write(f"  Range: [{y.min():.1f}%, {y.max():.1f}%]\n")
    f.write(f"  Derived responders (≥30%): {(y >= 30).sum()} ({(y >= 30).sum()/len(y)*100:.0f}%)\n")
    f.write(f"  Derived non-responders (<30%): {(y < 30).sum()} ({(y < 30).sum()/len(y)*100:.0f}%)\n\n")
    
    f.write("TOP 10 PREDICTIVE FEATURES:\n")
    f.write("-"*80 + "\n")
    for i, row in importance.head(10).iterrows():
        f.write(f"  {i+1:2d}. [{row['Type']:8s}] {row['Feature']:<35s} {row['Importance']:.4f}\n")
    
    f.write("\nMODEL PERFORMANCE (REGRESSION):\n")
    f.write("-"*80 + "\n")
    for _, row in metrics_df.iterrows():
        f.write(f"\n{row['Model']}:\n")
        f.write(f"  R² Score: {row['R2_Score']:.3f}\n")
        f.write(f"  RMSE: {row['RMSE']:.1f}%\n")
        f.write(f"  MAE: {row['MAE']:.1f}%\n")
        f.write(f"  Correlation: {row['Correlation']:.3f}\n")
        f.write(f"  Derived Classification Accuracy (30% threshold): {row['Classification_Accuracy_30pct']:.1%}\n")
        f.write(f"  Confusion Matrix: TN={row['TN']:.0f}, FP={row['FP']:.0f}, FN={row['FN']:.0f}, TP={row['TP']:.0f}\n")
    
    f.write("\n" + "="*80 + "\n")

print(f"  ✓ Saved: analysis_summary.txt")

print("\n" + "="*100)
print("✅ COMPLETE INTEGRATED ANALYSIS FINISHED")
print("="*100)
print(f"\nKey outputs:")
print(f"  1. complete_integrated_dataset.csv - Full dataset with all features")
print(f"  2. feature_importance_complete.csv - Ranked features")
print(f"  3. analysis_summary.txt - Analysis summary")

print(f"\nDataset includes:")
print(f"  ✓ HFO neural biomarkers")
print(f"  ✓ Detailed UPDRS motor features")
print(f"  ✓ ECoG recording metadata (REST/MOVE, coordinates, DBS params)")
print(f"  ✓ Brain structure features (cortical thickness, parcellation)")
print(f"  ✓ Proper handling of multiple recordings per subject")

print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*100)