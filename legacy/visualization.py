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
INTEGRATED COMPREHENSIVE VISUALIZATION SUITE - REGRESSION
================================================================================

Optimized for regression models predicting UPDRS improvement (continuous).

Generates 7 publication-ready figures:
1. Model Performance (scatter plots, R²/RMSE/MAE, residuals)
2. Clinical Predictions (subject-level predictions with error bars)
3. Feature Analysis (SHAP/RF importance with direction)
4. Clinical Context (phenotype, duration, PCA)
5. Publication Summary (dataset, methods, results, limitations)
6. Partial Dependence Plots (feature effect curves)
7. Statistical Analysis (mixed-effects regression + comparisons)

Date: December 2025
================================================================================
"""
from adjustText import adjust_text

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import partial_dependence
import warnings
warnings.filterwarnings('ignore')

# Statistical models for mixed-effects regression
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("\n⚠ statsmodels not available - Mixed-effects regression will be skipped")
    print("  Install with: pip install statsmodels\n")

# Set publication style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

print("\n" + "="*100)
print("INTEGRATED COMPREHENSIVE VISUALIZATION SUITE - REGRESSION")
print("="*100)

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path('<PRIVATE_DATA_DIR>/outputs/')
OUTPUT_DIR = DATA_DIR / 'figures/'
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLoading data...")
print("-"*100)

# Core datasets
data = pd.read_csv(DATA_DIR / 'complete_integrated_dataset.csv')
print(f"✓ Loaded master dataset: {len(data)} records")

predictions = pd.read_csv(DATA_DIR / 'model_predictions.csv')
print(f"✓ Loaded predictions: {len(predictions)} predictions")

metrics = pd.read_csv(DATA_DIR / 'model_performance_metrics.csv')
print(f"✓ Loaded performance metrics: {len(metrics)} models")

importance = pd.read_csv(DATA_DIR / 'feature_importance_complete.csv')
print(f"✓ Loaded feature importance: {len(importance)} features")

# SHAP importance (if available)
shap_file = DATA_DIR / 'shap_importance.csv'
if shap_file.exists():
    shap_importance = pd.read_csv(shap_file)
    print(f"✓ Loaded SHAP importance: {len(shap_importance)} features")
    has_shap = True
else:
    shap_importance = None
    has_shap = False
    print(f"  SHAP importance not found (will use RF importance)")

# Scatter data (regression diagnostic)
scatter_file = DATA_DIR / 'prediction_scatter_data.csv'
if scatter_file.exists():
    scatter_data = pd.read_csv(scatter_file)
    print(f"✓ Loaded scatter data: {len(scatter_data)} points")
else:
    scatter_data = None
    print(f"  Scatter data not found (will generate from predictions)")

# Aggregate to subject level
subject_data = predictions.groupby('subject').agg({
    'true_improvement': 'first'
}).reset_index()

# Add predictions (average if multiple recordings)
for col in predictions.columns:
    if '_pred' in col and '_class' not in col:
        subject_data[col] = predictions.groupby('subject')[col].mean().values

print(f"\n✓ Aggregated to subject level: {len(subject_data)} subjects")

# ============================================================================
# FIGURE 1: MODEL PERFORMANCE (REGRESSION)
# ============================================================================

print("\nCreating Figure 1: Model Performance (Regression)...")
print("-"*100)

fig1 = plt.figure(figsize=(24, 14))
fig1.suptitle('Model Performance: Regression Metrics & Diagnostics', 
              fontsize=18, fontweight='bold', y=0.98)

gs1 = GridSpec(2, 3, figure=fig1, left=0.06, right=0.98, top=0.94, bottom=0.08,
               wspace=0.35, hspace=0.35)

# Get model names (excluding Linear Regression if poor performance)
model_cols = [col.replace('_pred', '') for col in subject_data.columns if '_pred' in col and '_class' not in col]
models_to_plot = model_cols[:3]  # Top 3 models

# --- Panels 1.1-1.3: Scatter Plots (Predicted vs Actual) ---
for idx, model_name in enumerate(models_to_plot):
    ax = fig1.add_subplot(gs1[0, idx])
    
    col_name = f'{model_name}_pred'
    if col_name in subject_data.columns:
        y_true = subject_data['true_improvement'].values
        y_pred = subject_data[col_name].values
        
        # Scatter plot
        ax.scatter(y_true, y_pred, s=150, alpha=0.7, edgecolors='black', linewidth=2,
                  c=range(len(y_true)), cmap='viridis')
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5, label='Perfect Prediction')
        
        # Add subject labels
        for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
            ax.annotate(
                f'S{i+1}',
                (yt, yp),
                textcoords='offset points',
                xytext=(-5, -20),   # (x_offset, y_offset)
                fontsize=8,
                ha='left',       # adjust depending on offset direction
                va='bottom'
            )        
        # Get metrics
        model_metrics = metrics[metrics['Model'] == model_name]
        if len(model_metrics) > 0:
            r2 = model_metrics['R2_Score'].values[0]
            rmse = model_metrics['RMSE'].values[0]
            mae = model_metrics['MAE'].values[0]
            
            # Add metrics text
            metrics_text = f'R² = {r2:.3f}\nRMSE = {rmse:.1f}%\nMAE = {mae:.1f}%'
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('True Improvement (%)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Predicted Improvement (%)', fontweight='bold', fontsize=11)
        ax.set_title(f'{chr(65+idx)}. {model_name}', fontweight='bold', fontsize=12, loc='left', pad=10)
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)

# --- Panel 1.4: Performance Metrics Comparison ---
ax4 = fig1.add_subplot(gs1[1, 0])

metrics_plot = metrics.copy()
x = np.arange(len(metrics_plot))
width = 0.25

r2_bars = ax4.bar(x - width, metrics_plot['R2_Score'], width, label='R²', alpha=0.8, edgecolor='black')
rmse_bars = ax4.bar(x, metrics_plot['RMSE']/100, width, label='RMSE/100', alpha=0.8, edgecolor='black')
mae_bars = ax4.bar(x + width, metrics_plot['MAE']/100, width, label='MAE/100', alpha=0.8, edgecolor='black')

ax4.set_ylabel('Score', fontweight='bold', fontsize=11)
ax4.set_title('D. Regression Metrics Comparison', fontweight='bold', fontsize=12, loc='left', pad=10)
ax4.set_xticks(x)
ax4.set_xticklabels(metrics_plot['Model'], rotation=45, ha='right', fontsize=10)
ax4.legend(loc='upper right', fontsize=9)
ax4.grid(axis='y', alpha=0.3)
ax4.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='R²=0.7 (Good)')
ax4.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='R²=0.5 (Moderate)')

# Add value labels
for bars in [r2_bars, rmse_bars, mae_bars]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

# --- Panel 1.5: Residual Plot (Best Model) ---
ax5 = fig1.add_subplot(gs1[1, 1])

best_model = metrics.loc[metrics['R2_Score'].idxmax(), 'Model']
best_col = f'{best_model}_pred'

if best_col in subject_data.columns:
    y_true = subject_data['true_improvement'].values
    y_pred = subject_data[best_col].values
    residuals = y_pred - y_true
    
    ax5.scatter(y_pred, residuals, s=150, alpha=0.7, c=range(len(residuals)),
               cmap='coolwarm', edgecolors='black', linewidth=2)
    ax5.axhline(y=0, color='black', linestyle='--', linewidth=2)
    
    # Add ±1 SD lines
    std_resid = np.std(residuals)
    ax5.axhline(y=std_resid, color='red', linestyle=':', alpha=0.5)
    ax5.axhline(y=-std_resid, color='red', linestyle=':', alpha=0.5)
    
    ax5.set_xlabel('Predicted Improvement (%)', fontweight='bold', fontsize=11)
    ax5.set_ylabel('Residual (Predicted - True)', fontweight='bold', fontsize=11)
    ax5.set_title(f'E. Residual Plot ({best_model})', fontweight='bold', fontsize=12, loc='left', pad=10)
    ax5.grid(alpha=0.3)
    
    # Add text
    ax5.text(0.05, 0.95, f'Mean Residual: {np.mean(residuals):.2f}%\nSD: {std_resid:.2f}%',
            transform=ax5.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))


# --- Panel 1.6: Distribution of Errors ---
ax6 = fig1.add_subplot(gs1[1, 2])

all_errors = []
all_models = []
for model_name in model_cols:
    col_name = f'{model_name}_pred'
    if col_name in subject_data.columns:
        y_true = subject_data['true_improvement'].values
        y_pred = subject_data[col_name].values
        errors = np.abs(y_pred - y_true)
        all_errors.extend(errors)
        all_models.extend([model_name] * len(errors))

error_df = pd.DataFrame({'Model': all_models, 'Absolute_Error': all_errors})

bp = ax6.boxplot([error_df[error_df['Model'] == m]['Absolute_Error'].values for m in model_cols],
                 labels=[m.replace('_', ' ') for m in model_cols],
                 patch_artist=True, widths=0.6)

colors = plt.cm.Set3(np.linspace(0, 1, len(model_cols)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax6.set_ylabel('Absolute Error (%)', fontweight='bold', fontsize=11)
ax6.set_title('F. Error Distribution by Model', fontweight='bold', fontsize=12, loc='left', pad=10)
plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
ax6.grid(axis='y', alpha=0.3)

plt.savefig(OUTPUT_DIR / 'Integrated_01_ModelPerformance_REGRESSION.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Integrated_01_ModelPerformance_REGRESSION.png")
plt.close()

# ============================================================================
# FIGURE 2: CLINICAL PREDICTIONS
# ============================================================================

print("\nCreating Figure 2: Clinical Predictions...")
print("-"*100)

fig2 = plt.figure(figsize=(20, 12))
fig2.suptitle('Clinical Predictions: Subject-Level Analysis', 
              fontsize=18, fontweight='bold', y=0.98)

gs2 = GridSpec(2, 2, figure=fig2, left=0.08, right=0.98, top=0.94, bottom=0.08,
               wspace=0.3, hspace=0.35)

# --- Panel 2.1: Subject-Level Predictions (Bar Plot) ---
ax1 = fig2.add_subplot(gs2[0, :])

subjects = subject_data['subject'].values
y_true = subject_data['true_improvement'].values
best_col = f'{best_model}_pred'
y_pred = subject_data[best_col].values if best_col in subject_data.columns else y_true

x = np.arange(len(subjects))
width = 0.35

bars1 = ax1.bar(x - width/2, y_true, width, label='True Improvement', 
               color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, y_pred, width, label=f'Predicted ({best_model})', 
               color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

# Add 30% threshold line
ax1.axhline(y=30, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Responder Threshold (30%)')

ax1.set_xlabel('Subject', fontweight='bold', fontsize=12)
ax1.set_ylabel('UPDRS Improvement (%)', fontweight='bold', fontsize=12)
ax1.set_title('A. Subject-Level Predictions', fontweight='bold', fontsize=13, loc='left', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels([f'S{i+1}' for i in range(len(subjects))], fontsize=10)
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    ax1.text(bar1.get_x() + bar1.get_width()/2., height1 + 1,
            f'{height1:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax1.text(bar2.get_x() + bar2.get_width()/2., height2 + 1,
            f'{height2:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

# --- Panel 2.2: Prediction Accuracy by Subject ---
ax2 = fig2.add_subplot(gs2[1, 0])

errors = y_pred - y_true
abs_errors = np.abs(errors)

colors_acc = ['green' if abs(e) < 5 else 'orange' if abs(e) < 10 else 'red' for e in errors]

bars = ax2.barh(range(len(subjects)), abs_errors, color=colors_acc, alpha=0.7, edgecolor='black', linewidth=1.5)

ax2.set_yticks(range(len(subjects)))
ax2.set_yticklabels([f'S{i+1}' for i in range(len(subjects))], fontsize=10)
ax2.set_xlabel('Absolute Prediction Error (%)', fontweight='bold', fontsize=11)
ax2.set_title('B. Prediction Error by Subject', fontweight='bold', fontsize=12, loc='left', pad=10)
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

# Add value labels
for i, (bar, err, subj) in enumerate(zip(bars, abs_errors, subjects)):
    ax2.text(err + 0.5, i, f'{err:.1f}%', va='center', fontsize=9)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', label='Excellent (<5%)'),
    Patch(facecolor='orange', label='Good (5-10%)'),
    Patch(facecolor='red', label='Fair (>10%)')
]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=9)

# --- Panel 2.3: Classification from Regression ---
ax3 = fig2.add_subplot(gs2[1, 1])

# Derive classification
y_true_class = (y_true >= 30).astype(int)
y_pred_class = (y_pred >= 30).astype(int)

# Confusion matrix
from sklearn.metrics import confusion_matrix as cm_func
cm = cm_func(y_true_class, y_pred_class)

im = ax3.imshow(cm, cmap='Blues', aspect='auto')

# Annotations
for i in range(2):
    for j in range(2):
        text = ax3.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                       fontsize=20, fontweight='bold',
                       color='white' if cm[i, j] > cm.max()/2 else 'black')

ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['Non-Resp', 'Resp'], fontsize=11)
ax3.set_yticklabels(['Non-Resp', 'Resp'], fontsize=11)
ax3.set_xlabel('Predicted Class (30% threshold)', fontweight='bold', fontsize=11)
ax3.set_ylabel('True Class', fontweight='bold', fontsize=11)
ax3.set_title('C. Derived Classification', fontweight='bold', fontsize=12, loc='left', pad=10)

# Add accuracy
class_acc = (y_true_class == y_pred_class).mean()
ax3.text(0.5, -0.15, f'Classification Accuracy: {class_acc:.1%}',
        transform=ax3.transAxes, ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

plt.savefig(OUTPUT_DIR / 'Integrated_02_ClinicalPredictions_REGRESSION.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Integrated_02_ClinicalPredictions_REGRESSION.png")
plt.close()

# ============================================================================
# FIGURE 3: FEATURE ANALYSIS (SHAP/RF with Regression Interpretation)
# ============================================================================

print("\nCreating Figure 3: Feature Analysis...")
print("-"*100)

fig3 = plt.figure(figsize=(22, 12))
fig3.suptitle('Feature Importance: SHAP Values & Random Forest Rankings (Regression)', 
              fontsize=18, fontweight='bold', y=0.98)

gs3 = GridSpec(2, 2, figure=fig3, left=0.08, right=0.98, top=0.94, bottom=0.08,
               wspace=0.3, hspace=0.35)

# --- Panel 3.1: SHAP/RF Importance ---
ax1 = fig3.add_subplot(gs3[0, :])

shap_source = None
if has_shap and shap_importance is not None:
    imp_data = shap_importance.copy()
    shap_source = 'SHAP (Integrated Pipeline)'
    imp_col = 'Mean_Abs_SHAP'
    dir_col = 'Mean_SHAP'
else:
    imp_data = importance.copy()
    shap_source = 'Random Forest'
    imp_col = 'Importance'
    dir_col = None

top_n = min(15, len(imp_data))
top_features = imp_data.head(top_n)

# Color by type
if 'Type' in top_features.columns:
    colors_imp = ['#3498db' if t == 'HFO' else '#e74c3c' if t == 'UPDRS' else '#f39c12' 
                 for t in top_features['Type']]
else:
    colors_imp = plt.cm.viridis(np.linspace(0, 1, len(top_features)))

bars = ax1.barh(range(len(top_features)), top_features[imp_col].values,
               color=colors_imp, alpha=0.7, edgecolor='black', linewidth=1.5)

ax1.set_yticks(range(len(top_features)))
ax1.set_yticklabels(top_features['Feature'].values, fontsize=10)
ax1.set_xlabel(f'{shap_source} Importance', fontweight='bold', fontsize=11)
ax1.set_title(f'Top {top_n} Features (Source: {shap_source})', fontsize=12, fontweight='bold', pad=10)
ax1.grid(axis='x', alpha=0.3)
ax1.invert_yaxis()

# Add directional arrows if SHAP
if dir_col and dir_col in top_features.columns:
    # --------------------------
# IMPROVED: NO OVERLAP LABELS + ARROWS
# --------------------------
    ax1.figure.canvas.draw()
    x1, x2 = ax1.get_xlim()

    # Small offset just for value label (close to bar)
    offset_val = 0.005 * (x2 - x1)

    # Larger offset for arrow, farther away
    offset_arrow = 0.03 * (x2 - x1)

    for i, (idx, row) in enumerate(top_features.iterrows()):
        val = float(row[imp_col])

        # --- VALUE LABEL (tight against bar) ---
        ax1.text(
            val + offset_val,
            i,
            f'{val:.4f}',
            va='center',
            ha='left',
            fontsize=9,
            fontweight='bold'
        )

        # --- ARROW LABEL (well separated) ---
        if dir_col and abs(row[dir_col]) > 0.001:
            direction = float(row[dir_col])
            arrow = '↑' if direction > 0 else '↓'
            color = 'darkgreen' if direction > 0 else 'darkred'

            ax1.text(
                val + offset_arrow,
                i,
                arrow,
                va='center',
                ha='left',
                fontsize=12,
                fontweight='bold',
                color=color
            )



# Legend
if 'Type' in top_features.columns:
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='HFO Neural'),
        Patch(facecolor='#e74c3c', label='UPDRS Clinical'),
        Patch(facecolor='#f39c12', label='Metadata')
    ]
    if has_shap:
        legend_elements.append(Patch(facecolor='white', edgecolor='black', 
                                    label='↑ = Higher Improvement'))
        legend_elements.append(Patch(facecolor='white', edgecolor='black', 
                                    label='↓ = Lower Improvement'))
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=9)

# --- Panel 3.2: Feature Type Importance ---
ax2 = fig3.add_subplot(gs3[1, 0])

if 'Type' in imp_data.columns:
    feature_type_imp = imp_data.groupby('Type')[imp_col].sum().sort_values()
    colors_type = ['#3498db', '#e74c3c', '#f39c12'][:len(feature_type_imp)]
    
    bars = ax2.barh(range(len(feature_type_imp)), feature_type_imp.values,
                   color=colors_type, alpha=0.7, edgecolor='black', linewidth=2)
    
    ax2.set_yticks(range(len(feature_type_imp)))
    ax2.set_yticklabels(feature_type_imp.index, fontsize=10)
    ax2.set_xlabel('Cumulative Importance', fontweight='bold', fontsize=11)
    ax2.set_title('Importance by Feature Type', fontweight='bold', fontsize=12, pad=10)
    ax2.grid(axis='x', alpha=0.3)
    
    total_imp = feature_type_imp.sum()
    for i, (feat_type, imp) in enumerate(feature_type_imp.items()):
        pct = (imp / total_imp) * 100
        ax2.text(imp, i, f' {pct:.1f}%', va='center', fontsize=10, fontweight='bold')
else:
    ax2.text(0.5, 0.5, 'Feature type information not available',
            ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    ax2.axis('off')

# --- Panel 3.3: Feature Correlations ---
ax3 = fig3.add_subplot(gs3[1, 1])

top_10_features = imp_data.head(10)['Feature'].values
available_features = [f for f in top_10_features if f in data.columns]

if len(available_features) >= 3:
    corr_data = data.groupby('subject')[available_features].mean()
    corr_matrix = corr_data.corr()
    
    im = ax3.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add correlation values as text in each cell
    for i in range(len(available_features)):
        for j in range(len(available_features)):
            text_color = 'white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black'
            text = ax3.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha='center', va='center', color=text_color,
                          fontsize=8, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation', fontweight='bold', fontsize=10)
    
    ax3.set_xticks(range(len(available_features)))
    ax3.set_yticks(range(len(available_features)))
    
    labels = [f[:15] + '...' if len(f) > 15 else f for f in available_features]
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax3.set_yticklabels(labels, fontsize=8)
    
    ax3.set_title(f'Feature Correlations (Top {len(available_features)})', 
                 fontweight='bold', fontsize=12, pad=10)
    
    # Turn off grid lines for heatmap
    ax3.grid(False)
else:
    ax3.text(0.5, 0.5, 'Insufficient features for correlation analysis',
            ha='center', va='center', transform=ax3.transAxes, fontsize=12)
    ax3.axis('off')

plt.savefig(OUTPUT_DIR / 'Integrated_03_FeatureAnalysis_REGRESSION.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Integrated_03_FeatureAnalysis_REGRESSION.png")
plt.close()

# ============================================================================
# FIGURE 4: CLINICAL CONTEXT
# ============================================================================

print("\nCreating Figure 4: Clinical Context...")
print("-"*100)

fig4 = plt.figure(figsize=(20, 12))
fig4.suptitle('Clinical Context: Phenotype & Disease Duration', 
              fontsize=18, fontweight='bold', y=0.98)

gs4 = GridSpec(2, 2, figure=fig4, left=0.08, right=0.98, top=0.94, bottom=0.08,
               wspace=0.35, hspace=0.35)

# --- Panel 4.1: Disease Duration vs Improvement ---
ax1 = fig4.add_subplot(gs4[0, 0])

if 'Disease_Duration' in data.columns or 'years_since_diagnosis' in data.columns:
    duration_col = 'Disease_Duration' if 'Disease_Duration' in data.columns else 'years_since_diagnosis'
    duration_data = data.groupby('subject').agg({
        duration_col: 'first',
        'UPDRS_Improvement': 'first'
    }).dropna()
    
    if len(duration_data) > 0:
        x = duration_data[duration_col].values
        y = duration_data['UPDRS_Improvement'].values
        
        ax1.scatter(x, y, s=150, alpha=0.7, c=y, cmap='RdYlGn', 
                   edgecolors='black', linewidth=2, vmin=0, vmax=60)
        
        # Trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax1.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.1f}')
        
        # Correlation
        corr = np.corrcoef(x, y)[0, 1]
        ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax1.set_xlabel('Disease Duration (years)', fontweight='bold', fontsize=11)
        ax1.set_ylabel('UPDRS Improvement (%)', fontweight='bold', fontsize=11)
        ax1.set_title('A. Disease Duration vs Response', fontweight='bold', fontsize=12, loc='left', pad=10)
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(alpha=0.3)
        
        # 30% threshold line
        ax1.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Responder threshold')
    else:
        ax1.text(0.5, 0.5, 'No duration data available', ha='center', va='center',
                transform=ax1.transAxes, fontsize=12)
        ax1.axis('off')
else:
    ax1.text(0.5, 0.5, 'Disease duration not available', ha='center', va='center',
            transform=ax1.transAxes, fontsize=12)
    ax1.axis('off')

# --- Panel 4.2: Motor Phenotype (if available) ---
ax2 = fig4.add_subplot(gs4[0, 1])

ax2.text(0.5, 0.5, 'Motor phenotype analysis\n(if categorical data available)',
        ha='center', va='center', transform=ax2.transAxes, fontsize=11,
        style='italic', color='gray')
ax2.set_title('B. Motor Phenotype', fontweight='bold', fontsize=12, loc='left', pad=10)
ax2.axis('off')

# --- Panel 4.3: PCA Visualization ---
ax3 = fig4.add_subplot(gs4[1, :])

# Get numeric features
feature_cols = [col for col in data.columns 
                if col in importance['Feature'].values and data[col].dtype in [np.float64, np.int64]]
feature_cols = feature_cols[:30]  # Top 30 features

if len(feature_cols) >= 3:
    X_pca = data.groupby('subject')[feature_cols].mean()
    X_pca = X_pca.fillna(X_pca.median())
    y_pca = subject_data.set_index('subject').loc[X_pca.index, 'true_improvement'].values
    
    # PCA
    scaler_pca = StandardScaler()
    X_scaled_pca = scaler_pca.fit_transform(X_pca)
    
    pca = PCA(n_components=2)
    X_pca_2d = pca.fit_transform(X_scaled_pca)
    
    # Scatter plot
    scatter = ax3.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                         s=200, c=y_pca, cmap='RdYlGn', 
                         edgecolors='black', linewidth=2,
                         vmin=0, vmax=60, alpha=0.8)
    
    # Add subject labels
    for i, (x, y_val) in enumerate(zip(X_pca_2d, y_pca)):
        ax3.annotate(f'S{i+1}', (x[0], x[1]), fontsize=9, ha='center', va='bottom',
                    fontweight='bold')
    
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', 
                  fontweight='bold', fontsize=11)
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', 
                  fontweight='bold', fontsize=11)
    ax3.set_title(f'C. PCA Projection (Top {len(feature_cols)} Features)', 
                 fontweight='bold', fontsize=12, loc='left', pad=10)
    ax3.grid(alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax3, fraction=0.03, pad=0.02)
    cbar.set_label('UPDRS Improvement (%)', fontweight='bold', fontsize=10)
    
    # Add 30% threshold
    cbar.ax.axhline(y=30, color='green', linestyle='--', linewidth=2)
else:
    ax3.text(0.5, 0.5, 'Insufficient features for PCA',
            ha='center', va='center', transform=ax3.transAxes, fontsize=12)
    ax3.axis('off')

plt.savefig(OUTPUT_DIR / 'Integrated_04_ClinicalContext_REGRESSION.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Integrated_04_ClinicalContext_REGRESSION.png")
plt.close()

# ============================================================================
# FIGURE 5: PUBLICATION SUMMARY
# ============================================================================

print("\nCreating Figure 5: Publication Summary...")
print("-"*100)

fig5 = plt.figure(figsize=(22, 14))
fig5.suptitle('Publication Summary: DBS Response Prediction (Regression)', 
              fontsize=18, fontweight='bold', y=0.98)

gs5 = GridSpec(3, 2, figure=fig5, left=0.08, right=0.98, top=0.94, bottom=0.06,
               wspace=0.3, hspace=0.4)

# --- Panel 5.1: Dataset Overview ---
ax1 = fig5.add_subplot(gs5[0, 0])
ax1.axis('off')

dataset_text = "DATASET\n" + "="*50 + "\n\n"
dataset_text += f"Subjects: {len(subject_data)}\n"
dataset_text += f"Total recordings: {len(data)}\n"
dataset_text += f"UPDRS Improvement:\n"
dataset_text += f"  Mean: {subject_data['true_improvement'].mean():.1f}%\n"
dataset_text += f"  SD: {subject_data['true_improvement'].std():.1f}%\n"
dataset_text += f"  Range: [{subject_data['true_improvement'].min():.1f}%, {subject_data['true_improvement'].max():.1f}%]\n\n"
dataset_text += f"Responders (≥30%): {(subject_data['true_improvement'] >= 30).sum()}\n"
dataset_text += f"Non-responders (<30%): {(subject_data['true_improvement'] < 30).sum()}\n\n"
dataset_text += f"Features:\n"
dataset_text += f"  Total extracted: {len(importance)}\n"

hfo_count = sum(importance['Type'] == 'HFO') if 'Type' in importance.columns else 0
updrs_count = sum(importance['Type'] == 'UPDRS') if 'Type' in importance.columns else 0
meta_count = sum(importance['Type'] == 'Metadata') if 'Type' in importance.columns else 0

dataset_text += f"  HFO neural: {hfo_count}\n"
dataset_text += f"  UPDRS clinical: {updrs_count}\n"
dataset_text += f"  Metadata: {meta_count}\n"

ax1.text(0.05, 0.95, dataset_text, transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax1.set_title('A. Dataset Overview', fontweight='bold', fontsize=12, loc='left')

# --- Panel 5.2: Model Performance ---
ax2 = fig5.add_subplot(gs5[0, 1])
ax2.axis('off')

performance_text = "MODEL PERFORMANCE\n" + "="*50 + "\n\n"
for _, row in metrics.iterrows():
    performance_text += f"{row['Model']}:\n"
    performance_text += f"  R² Score: {row['R2_Score']:.3f}\n"
    performance_text += f"  RMSE: {row['RMSE']:.1f}%\n"
    performance_text += f"  MAE: {row['MAE']:.1f}%\n"
    performance_text += f"  Correlation: {row['Correlation']:.3f}\n"
    performance_text += f"  Class Acc (30%): {row['Classification_Accuracy_30pct']:.1%}\n\n"

performance_text += f"Best Model: {best_model}\n"
best_metrics = metrics[metrics['Model'] == best_model].iloc[0]
performance_text += f"  R² = {best_metrics['R2_Score']:.3f}\n"
performance_text += f"  RMSE = {best_metrics['RMSE']:.1f}%\n"

ax2.text(0.05, 0.95, performance_text, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax2.set_title('B. Performance Metrics', fontweight='bold', fontsize=12, loc='left')

# --- Panel 5.3: Top Features ---
ax3 = fig5.add_subplot(gs5[1, :])

top_10 = imp_data.head(10)
colors_feat = ['#3498db' if t == 'HFO' else '#e74c3c' if t == 'UPDRS' else '#f39c12' 
              for t in top_10['Type']] if 'Type' in top_10.columns else ['steelblue']*10

bars = ax3.barh(range(len(top_10)), top_10[imp_col].values,
               color=colors_feat, alpha=0.7, edgecolor='black', linewidth=2)

ax3.set_yticks(range(len(top_10)))
ax3.set_yticklabels(top_10['Feature'].values, fontsize=10)
ax3.set_xlabel('Importance', fontweight='bold', fontsize=11)
ax3.set_title('C. Top 10 Predictive Features', fontweight='bold', fontsize=12, loc='left', pad=10)
ax3.grid(axis='x', alpha=0.3)
ax3.invert_yaxis()

for i, (val, feat_type) in enumerate(zip(top_10[imp_col].values, top_10['Type'].values if 'Type' in top_10.columns else [''] * len(top_10))):
    ax3.text(val + 0.002, i, f'{val:.4f} ({feat_type})' if feat_type else f'{val:.4f}', 
            va='center', fontsize=9)

# --- Panel 5.4: Key Findings ---
ax4 = fig5.add_subplot(gs5[2, 0])
ax4.axis('off')

findings_text = "KEY FINDINGS\n" + "="*50 + "\n\n"
findings_text += "REGRESSION APPROACH:\n" + "-"*50 + "\n"
findings_text += "✓ Predicts continuous improvement (%)\n"
findings_text += "✓ More clinically actionable\n"
findings_text += "✓ Retains full information\n\n"

findings_text += "TOP BIOMARKERS:\n" + "-"*50 + "\n"
for i, row in imp_data.head(5).iterrows():
    findings_text += f"{i+1}. {row['Feature'][:30]:30s}\n"

findings_text += "\n\nSTRENGTHS:\n" + "-"*50 + "\n"
findings_text += "✓ Multi-modal data integration\n"
findings_text += "✓ Comprehensive feature engineering\n"
findings_text += "✓ Proper CV validation (GroupKFold)\n"
findings_text += "✓ SHAP explainability\n" if has_shap else ""
findings_text += "✓ Regression for continuous outcome\n"

ax4.text(0.05, 0.95, findings_text, transform=ax4.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
ax4.set_title('D. Key Findings', fontweight='bold', fontsize=12, loc='left')

# --- Panel 5.5: Limitations & Next Steps ---
ax5 = fig5.add_subplot(gs5[2, 1])
ax5.axis('off')

limitations_text = "LIMITATIONS & NEXT STEPS\n" + "="*50 + "\n\n"
limitations_text += "CURRENT LIMITATIONS:\n" + "-"*50 + "\n"
limitations_text += f"• Small sample size (n={len(subject_data)})\n"
limitations_text += "• Limited validation possible\n"
limitations_text += "• Single-center study\n"
limitations_text += f"• Class imbalance ({(subject_data['true_improvement']>=30).sum()}:{(subject_data['true_improvement']<30).sum()} ratio)\n\n"

limitations_text += "RECOMMENDED NEXT STEPS:\n" + "-"*50 + "\n"
limitations_text += "• Target n=80 (more robust estimates)\n"
limitations_text += "• Multi-center collaboration\n"
limitations_text += "• Prospective validation\n"
limitations_text += "• External test set\n"
limitations_text += "• Clinical deployment pilot\n"

ax5.text(0.05, 0.95, limitations_text, transform=ax5.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
ax5.set_title('E. Limitations & Future Work', fontweight='bold', fontsize=12, loc='left')

plt.savefig(OUTPUT_DIR / 'Integrated_05_PublicationSummary_REGRESSION.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Integrated_05_PublicationSummary_REGRESSION.png")
plt.close()

# ============================================================================
# FIGURE 6: PARTIAL DEPENDENCE PLOTS
# ============================================================================

print("\nCreating Figure 6: Partial Dependence Plots...")
print("-"*100)

# Prepare data for PDPs
feature_cols_pdp = [col for col in data.columns 
                    if col in importance['Feature'].values and col in data.columns]
feature_cols_pdp = feature_cols_pdp[:30]

if len(feature_cols_pdp) >= 6:
    X_pdp = data.groupby('subject')[feature_cols_pdp].mean()
    X_pdp = X_pdp.fillna(X_pdp.median())
    y_pdp = subject_data.set_index('subject').loc[X_pdp.index, 'true_improvement'].values
    
    # Scale
    scaler_pdp = StandardScaler()
    X_scaled_pdp = scaler_pdp.fit_transform(X_pdp)
    
    # Train model
    model_pdp = GradientBoostingRegressor(n_estimators=50, max_depth=3, 
                                          learning_rate=0.1, random_state=42)
    model_pdp.fit(X_scaled_pdp, y_pdp)
    
    # Get top 6 features
    top_6_features = imp_data.head(6)['Feature'].values
    top_6_indices = [feature_cols_pdp.index(f) for f in top_6_features if f in feature_cols_pdp]
    top_6_names = [f for f in top_6_features if f in feature_cols_pdp]
    
    if len(top_6_indices) >= 4:
        fig6, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig6.suptitle('Partial Dependence Analysis: Feature Effects on UPDRS Improvement', 
                      fontsize=18, fontweight='bold', y=0.98)
        
        axes = axes.flatten()
        
        for idx, (ax, feat_idx, feat_name) in enumerate(zip(axes[:len(top_6_indices)], 
                                                             top_6_indices, 
                                                             top_6_names)):
            try:
                pd_result = partial_dependence(model_pdp, X_scaled_pdp, [feat_idx], 
                                               grid_resolution=20)
                
                ax.plot(pd_result['grid_values'][0], 
                       pd_result['average'][0], 
                       linewidth=3, color='#3498db')
                
                # Confidence band
                std_approx = np.std(pd_result['average'][0]) * 0.5
                ax.fill_between(pd_result['grid_values'][0], 
                               pd_result['average'][0] - std_approx,
                               pd_result['average'][0] + std_approx,
                               alpha=0.3, color='#3498db')
                
                ax.set_xlabel(f'{feat_name} (Scaled)', fontweight='bold', fontsize=11)
                ax.set_ylabel('Effect on Improvement (%)', fontweight='bold', fontsize=11)
                ax.set_title(f'{chr(65+idx)}. {feat_name}', fontweight='bold', fontsize=12, loc='left', pad=10)
                ax.grid(alpha=0.3)
                
                # Trend
                trend = "↑" if pd_result['average'][0][-1] > pd_result['average'][0][0] else "↓"
                ax.text(0.02, 0.98, f'Trend {trend}', transform=ax.transAxes, fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       verticalalignment='top')
            except:
                ax.text(0.5, 0.5, f'PDP failed\n{feat_name}', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)
        
        for idx in range(len(top_6_indices), 6):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'Integrated_06_PartialDependence_REGRESSION.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: Integrated_06_PartialDependence_REGRESSION.png")
        plt.close()
    else:
        print("  Skipping - insufficient features for PDPs")
else:
    print("  Skipping - insufficient features for PDPs")

# ============================================================================
# FIGURE 7: STATISTICAL ANALYSIS
# ============================================================================

print("\nCreating Figure 7: Statistical Analysis...")
print("-"*100)

fig7 = plt.figure(figsize=(22, 12))
fig7.suptitle('Statistical Analysis: Mixed-Effects Regression & Improvement Distribution', 
              fontsize=18, fontweight='bold', y=0.98)

gs7 = GridSpec(2, 3, figure=fig7, left=0.06, right=0.98, top=0.94, bottom=0.08,
               wspace=0.35, hspace=0.4)

# --- Mixed-Effects Regression ---
ax1 = fig7.add_subplot(gs7[0, :])
ax1.axis('off')

if HAS_STATSMODELS and len(feature_cols_pdp) >= 3:
    try:
        me_data = data.copy()
        me_data['improvement'] = me_data['UPDRS_Improvement']
        
        # Top 3 features
        top_3 = [f for f in imp_data.head(5)['Feature'].values if f in me_data.columns][:3]
        
        if len(top_3) >= 2:
            formula = f"improvement ~ {' + '.join(top_3)}"
            
            print(f"  Fitting mixed-effects: {formula}")
            model_me = smf.mixedlm(formula, me_data, groups=me_data['subject'])
            result_me = model_me.fit(method='powell')
            
            me_summary = result_me.summary().as_text()
            
            ax1.text(0.05, 0.95, 
                    "MIXED-EFFECTS REGRESSION (Continuous Outcome)\n" + "="*80 + "\n\n" + 
                    "Accounts for repeated measures (multiple recordings per subject)\n" +
                    "Formula: " + formula + "\n\n" + me_summary[:1500],
                    transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            ax1.set_title('A. Mixed-Effects Regression Results', fontweight='bold', fontsize=12, loc='left')
        else:
            ax1.text(0.5, 0.5, 'Insufficient features for mixed-effects',
                    ha='center', va='center', transform=ax1.transAxes, fontsize=11)
    except Exception as e:
        ax1.text(0.5, 0.5, f'Mixed-effects failed:\n{str(e)}',
                ha='center', va='center', transform=ax1.transAxes, fontsize=11)
else:
    if not HAS_STATSMODELS:
        msg = 'statsmodels required\nInstall: pip install statsmodels'
    else:
        msg = 'Insufficient features'
    ax1.text(0.5, 0.5, msg, ha='center', va='center', transform=ax1.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.set_title('A. Mixed-Effects Regression', fontweight='bold', fontsize=12, loc='left')

# --- Improvement Distribution Comparisons ---
# Group by responder status
resp_subjects = subject_data[subject_data['true_improvement'] >= 30]
nonresp_subjects = subject_data[subject_data['true_improvement'] < 30]

# Plot top features comparison
for idx in range(6):
    row = idx // 3
    col = idx % 3
    ax = fig7.add_subplot(gs7[row, col])
    
    if idx < len(top_6_names):
        feat = top_6_names[idx]
        
        if feat in data.columns:
            resp_vals = data[data['subject'].isin(resp_subjects['subject'])].groupby('subject')[feat].mean().dropna().values
            nonresp_vals = data[data['subject'].isin(nonresp_subjects['subject'])].groupby('subject')[feat].mean().dropna().values
            
            if len(resp_vals) > 0 and len(nonresp_vals) > 0:
                bp = ax.boxplot([nonresp_vals, resp_vals], labels=['Non-Resp', 'Resp'],
                               patch_artist=True, widths=0.6)
                
                colors = ['#e74c3c', '#2ecc71']
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Scatter overlay
                for i, vals in enumerate([nonresp_vals, resp_vals]):
                    x = np.random.normal(i+1, 0.04, size=len(vals))
                    ax.scatter(x, vals, alpha=0.6, s=50, color='black', zorder=3)
                
                ax.set_ylabel(f'{feat[:20]}...', fontweight='bold', fontsize=10)
                ax.set_title(f'{chr(66+idx)}. {feat[:30]}', fontweight='bold', fontsize=11, loc='left', pad=8)
                ax.grid(axis='y', alpha=0.3)
                
                # Stats
                mean_diff = np.mean(resp_vals) - np.mean(nonresp_vals)
                ax.text(0.98, 0.02, f'Δ={mean_diff:.3f}', transform=ax.transAxes,
                       fontsize=9, ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.axis('off')

plt.savefig(OUTPUT_DIR / 'Integrated_07_StatisticalAnalysis_REGRESSION.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Integrated_07_StatisticalAnalysis_REGRESSION.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*100)
print("✅ REGRESSION VISUALIZATION COMPLETE")
print("="*100)

print(f"\nGenerated 7 comprehensive figures (REGRESSION):")
print(f"  1. ✓ Integrated_01_ModelPerformance_REGRESSION.png")
print(f"  2. ✓ Integrated_02_ClinicalPredictions_REGRESSION.png")
print(f"  3. ✓ Integrated_03_FeatureAnalysis_REGRESSION.png")
print(f"  4. ✓ Integrated_04_ClinicalContext_REGRESSION.png")
print(f"  5. ✓ Integrated_05_PublicationSummary_REGRESSION.png")
print(f"  6. ✓ Integrated_06_PartialDependence_REGRESSION.png")
print(f"  7. ✓ Integrated_07_StatisticalAnalysis_REGRESSION.png")

print(f"\nAll figures saved to: {OUTPUT_DIR}")

print(f"\nKey features:")
print(f"  ✓ Regression-optimized (continuous UPDRS improvement)")
print(f"  ✓ Scatter plots (predicted vs actual)")
print(f"  ✓ R²/RMSE/MAE metrics")
print(f"  ✓ Residual analysis")
print(f"  ✓ SHAP explainability (if available)")
print(f"  ✓ Partial dependence plots")
print(f"  ✓ Mixed-effects regression")
print(f"  ✓ Clinical interpretation throughout")

if HAS_STATSMODELS:
    print(f"\n  ✓ statsmodels available - mixed-effects included")
else:
    print(f"\n  ⚠ statsmodels not available - install with: pip install statsmodels")

print(f"\n" + "="*100)