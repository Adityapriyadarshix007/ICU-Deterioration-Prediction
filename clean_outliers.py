"""
Outlier Detection and Cleaning for ICU Data
Identifies and handles extreme values that degrade model performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs/plots", exist_ok=True)

print("="*60)
print("OUTLIER DETECTION AND CLEANING")
print("="*60)

# Load data
X = pd.read_csv("outputs/tables/X_features.csv")
y = pd.read_csv("outputs/tables/y_target.csv")

print(f"\n📊 Original Data Shape: {X.shape}")

# ============================================================
# STEP 1: Define Clinical Ranges
# ============================================================
print("\n[1] Defining clinical reference ranges...")

clinical_ranges = {
    'heart_rate': (30, 220),
    'sbp': (60, 250),
    'dbp': (30, 150),
    'gcs': (3, 15),
    'lactate': (0.1, 15),
    'urine_output': (0, 500),
    'fio2': (21, 100),
    'creatinine': (0.1, 10),
}

def clean_outliers(df, ranges, method='clip'):
    """Clean outliers using clinical ranges"""
    cleaned_df = df.copy()
    
    for col, (min_val, max_val) in ranges.items():
        feature_cols = [c for c in df.columns if col in c]
        for fcol in feature_cols:
            if method == 'clip':
                cleaned_df[fcol] = df[fcol].clip(min_val, max_val)
    return cleaned_df

def detect_outliers(df, ranges):
    """Detect outliers in the dataset"""
    outlier_counts = {}
    
    for col, (min_val, max_val) in ranges.items():
        feature_cols = [c for c in df.columns if col in c]
        for fcol in feature_cols:
            if fcol in df.columns:
                outliers = ((df[fcol] < min_val) | (df[fcol] > max_val)).sum()
                if outliers > 0:
                    outlier_counts[fcol] = {
                        'outliers': outliers,
                        'percentage': (outliers / len(df)) * 100,
                        'range': f'[{min_val}, {max_val}]'
                    }
    
    return outlier_counts

# ============================================================
# STEP 2: Detect Outliers
# ============================================================
print("\n[2] Detecting outliers...")

outliers = detect_outliers(X, clinical_ranges)

print("\n📊 Outliers Detected:")
for col, info in outliers.items():
    print(f"   {col}: {info['outliers']} outliers ({info['percentage']:.2f}%) outside {info['range']}")

# ============================================================
# STEP 3: Visualize Outliers (Skip empty columns)
# ============================================================
print("\n[3] Visualizing outliers...")

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()

plot_idx = 0
for col, (min_val, max_val) in clinical_ranges.items():
    if plot_idx >= len(axes):
        break
        
    feature_cols = [c for c in X.columns if col in c]
    if feature_cols:
        # Get first valid column
        valid_col = None
        for fc in feature_cols:
            if fc in X.columns and X[fc].notna().sum() > 0:
                valid_col = fc
                break
        
        if valid_col:
            data = X[valid_col].dropna()
            if len(data) > 0 and not np.isnan(data).all():
                axes[plot_idx].hist(data, bins=50, alpha=0.7, color='blue', edgecolor='black')
                axes[plot_idx].axvline(min_val, color='red', linestyle='--', label=f'Min: {min_val}')
                axes[plot_idx].axvline(max_val, color='red', linestyle='--', label=f'Max: {max_val}')
                axes[plot_idx].set_title(f'{col}\n({data.min():.1f} - {data.max():.1f})')
                axes[plot_idx].set_xlabel('Value')
                axes[plot_idx].set_ylabel('Frequency')
                axes[plot_idx].legend(fontsize=8)
            else:
                axes[plot_idx].text(0.5, 0.5, f'No valid data for {col}', 
                                   ha='center', va='center', transform=axes[plot_idx].transAxes)
                axes[plot_idx].set_title(f'{col} - No Data')
        else:
            axes[plot_idx].text(0.5, 0.5, f'No valid columns for {col}', 
                               ha='center', va='center', transform=axes[plot_idx].transAxes)
            axes[plot_idx].set_title(f'{col} - Not Found')
    
    plot_idx += 1

# Hide unused subplots
for idx in range(plot_idx, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Feature Distributions with Clinical Ranges', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/plots/outlier_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Outlier visualization saved: outputs/plots/outlier_analysis.png")

# ============================================================
# STEP 4: Clean Outliers
# ============================================================
print("\n[4] Cleaning outliers...")

# Use clipping method (preserves data shape)
X_cleaned = clean_outliers(X, clinical_ranges, method='clip')

# Check how many values were changed
modified_cols = []
for col in X.columns:
    if col in X_cleaned.columns:
        diff = (X[col] != X_cleaned[col]).sum()
        if diff > 0:
            modified_cols.append((col, diff))

print(f"\n📊 Values Modified:")
for col, diff in modified_cols[:10]:
    print(f"   {col}: {diff} values modified ({diff/len(X)*100:.2f}%)")

if len(modified_cols) > 10:
    print(f"   ... and {len(modified_cols)-10} more columns")

# ============================================================
# STEP 5: Save Cleaned Data
# ============================================================
print("\n[5] Saving cleaned data...")

X_cleaned.to_csv("outputs/tables/X_features_cleaned.csv", index=False)
print("✅ Cleaned features saved: outputs/tables/X_features_cleaned.csv")

# ============================================================
# STEP 6: Summary
# ============================================================
print("\n" + "="*60)
print("OUTLIER CLEANING SUMMARY")
print("="*60)
print(f"Original shape: {X.shape}")
print(f"Cleaned shape: {X_cleaned.shape}")
print(f"Columns modified: {len(modified_cols)}")
print(f"Total values modified: {sum([d for _, d in modified_cols])}")
print("\n✅ Outlier cleaning complete!")

# Show sample of cleaned data
print("\n📊 Sample of cleaned data (first 5 rows):")
print(X_cleaned.head())
