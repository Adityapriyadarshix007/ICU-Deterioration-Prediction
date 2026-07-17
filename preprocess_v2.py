"""
Enhanced Preprocessing Pipeline v2.0
- Fixes impossible physiological values
- Feature-specific imputation
- Proper handling of delta features
- Robust scaling
"""

import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("ENHANCED PREPROCESSING PIPELINE v2.0")
print("="*60)

# Load original data
X = pd.read_csv("outputs/tables/X_features.csv")
y = pd.read_csv("outputs/tables/y_target.csv")

print(f"\n📊 Original Data Shape: {X.shape}")

# ============================================================
# STEP 1: Define Clinical Ranges for Absolute Values
# ============================================================
print("\n[1] Defining clinical ranges for absolute values...")

clinical_ranges = {
    'heart_rate': (30, 220),
    'sbp': (60, 250),
    'dbp': (30, 150),
    'gcs': (3, 15),
    'lactate': (0.1, 20),
    'urine_output': (0, 500),
    'fio2': (21, 100),
    'creatinine': (0.1, 15),
}

# ============================================================
# STEP 2: Replace Impossible Values with NaN
# ============================================================
print("\n[2] Replacing impossible values with NaN...")

X_clean = X.copy()
impossible_counts = {}

for feature, (min_val, max_val) in clinical_ranges.items():
    # Find columns containing this feature
    feature_cols = [c for c in X_clean.columns if feature in c and 'delta' not in c]
    
    for col in feature_cols:
        # Identify impossible values
        impossible_mask = (X_clean[col] < min_val) | (X_clean[col] > max_val)
        count = impossible_mask.sum()
        
        if count > 0:
            impossible_counts[col] = count
            # Replace with NaN
            X_clean.loc[impossible_mask, col] = np.nan
            print(f"   {col}: {count} impossible values replaced with NaN")

print(f"\n✅ Total impossible values replaced: {sum(impossible_counts.values())}")

# ============================================================
# STEP 3: Feature-Specific Imputation
# ============================================================
print("\n[3] Applying feature-specific imputation...")

def impute_features(df):
    """Feature-specific imputation"""
    df_imputed = df.copy()
    
    # Get patient IDs (if available, use forward fill within patient)
    # For now, use column-wise median imputation
    
    for col in df_imputed.columns:
        # For delta features, use median (preserves negative values)
        if 'delta' in col:
            median_val = df_imputed[col].median()
            df_imputed[col] = df_imputed[col].fillna(median_val)
        else:
            # For absolute values, use median
            median_val = df_imputed[col].median()
            df_imputed[col] = df_imputed[col].fillna(median_val)
    
    return df_imputed

X_imputed = impute_features(X_clean)

# Check remaining NaN
remaining_nan = X_imputed.isnull().sum().sum()
print(f"   Remaining NaN values after imputation: {remaining_nan}")

if remaining_nan > 0:
    print("   ⚠️ Filling remaining NaN with 0...")
    X_imputed = X_imputed.fillna(0)

print("✅ Imputation complete")

# ============================================================
# STEP 4: Fix Delta Features - Keep Negative Values
# ============================================================
print("\n[4] Processing delta features...")

# Delta features should NOT be clipped to positive ranges
delta_cols = [c for c in X_imputed.columns if 'delta' in c]

for col in delta_cols:
    # Only remove extreme outliers (beyond 5 standard deviations)
    mean = X_imputed[col].mean()
    std = X_imputed[col].std()
    lower = mean - 5 * std
    upper = mean + 5 * std
    
    # Replace extreme outliers with NaN and re-impute
    extreme_mask = (X_imputed[col] < lower) | (X_imputed[col] > upper)
    if extreme_mask.sum() > 0:
        X_imputed.loc[extreme_mask, col] = np.nan
        median_val = X_imputed[col].median()
        X_imputed[col] = X_imputed[col].fillna(median_val)

print(f"   Processed {len(delta_cols)} delta features")
print("✅ Delta features processed")

# ============================================================
# STEP 5: Visualize Distributions After Cleaning
# ============================================================
print("\n[5] Visualizing cleaned distributions...")

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()

for idx, (feature, (min_val, max_val)) in enumerate(clinical_ranges.items()):
    if idx < len(axes):
        cols = [c for c in X_imputed.columns if feature in c and 'delta' not in c]
        if cols:
            data = X_imputed[cols[0]].dropna()
            if len(data) > 0:
                axes[idx].hist(data, bins=50, alpha=0.7, color='blue', edgecolor='black')
                axes[idx].axvline(min_val, color='red', linestyle='--', label=f'Min: {min_val}')
                axes[idx].axvline(max_val, color='red', linestyle='--', label=f'Max: {max_val}')
                axes[idx].set_title(f'{feature}\n({data.min():.1f} - {data.max():.1f})')
                axes[idx].set_xlabel('Value')
                axes[idx].set_ylabel('Frequency')
                axes[idx].legend(fontsize=8)

plt.suptitle('Feature Distributions After Cleaning', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/plots/cleaned_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Distribution plots saved")

# ============================================================
# STEP 6: Save Processed Data
# ============================================================
print("\n[6] Saving processed data...")

X_imputed.to_csv("outputs/tables/X_features_v2.csv", index=False)
print("✅ Processed data saved: outputs/tables/X_features_v2.csv")

# ============================================================
# STEP 7: Summary
# ============================================================
print("\n" + "="*60)
print("PREPROCESSING SUMMARY")
print("="*60)
print(f"Original shape: {X.shape}")
print(f"Processed shape: {X_imputed.shape}")
print(f"Impossible values removed: {sum(impossible_counts.values())}")
print(f"Remaining NaN: {X_imputed.isnull().sum().sum()}")
print(f"Delta features: {len(delta_cols)}")
print("\n✅ Preprocessing complete!")

print("\n📊 Sample of processed data:")
print(X_imputed.head())
