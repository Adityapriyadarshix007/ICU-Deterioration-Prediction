"""
Missing Value Analysis - Fixed with Clarification
"""

import pandas as pd
import numpy as np

print("="*60)
print("MISSING VALUE ANALYSIS (FIXED)")
print("="*60)

# Load original data
X = pd.read_csv("outputs/tables/X_features.csv")
y = pd.read_csv("outputs/tables/y_target.csv")

print(f"\n📊 Original Data Shape: {X.shape}")

# Calculate missing percentages for each feature
missing_pct = (X.isnull().sum() / len(X) * 100).sort_values(ascending=False)

# Focus on key features
key_features = [
    'urine_output_hour0', 'lactate_hour0', 'fio2_hour0',
    'creatinine_hour0', 'gcs_hour0', 'heart_rate_hour0',
    'sbp_hour0', 'dbp_hour0'
]

missing_data = []
for feat in key_features:
    if feat in missing_pct.index:
        missing_data.append({
            'Feature': feat.replace('_hour0', '').replace('_', ' ').title(),
            'Missing %': round(missing_pct[feat], 1)
        })

missing_df = pd.DataFrame(missing_data)

print("\n📊 Missing Data by Feature:")
print("-"*40)
print(missing_df.to_string(index=False))

total_missing = X.isnull().sum().sum()
total_cells = X.shape[0] * X.shape[1]
missing_rate = total_missing / total_cells * 100

print(f"\n📊 Overall Missing Values:")
print(f"   Total missing: {total_missing:,}")
print(f"   Total cells: {total_cells:,}")
print(f"   Missing rate: {missing_rate:.1f}%")

print("\n📊 CLARIFICATION:")
print("-"*40)
print("   Across all feature cells, {:.1f}% were missing.".format(missing_rate))
print("   Missingness was concentrated in a small number of laboratory")
print("   variables (e.g., lactate, creatinine, urine output), whereas")
print("   most vital signs had substantially lower missingness.")
print("   Median imputation was applied using training-set statistics.")

# Save
missing_df.to_csv('publication/missing_value_analysis_fixed.csv', index=False)

print("\n✅ Missing value analysis complete")
