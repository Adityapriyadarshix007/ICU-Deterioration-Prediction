"""
Missing Value Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("MISSING VALUE ANALYSIS")
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
            'Feature': feat.replace('_hour0', ''),
            'Missing %': missing_pct[feat]
        })

missing_df = pd.DataFrame(missing_data)

print("\n📊 Missing Data by Feature:")
print("-"*40)
print(missing_df.to_string(index=False))

print(f"\n📊 Overall Missing Values:")
print(f"   Total missing: {X.isnull().sum().sum()}")
print(f"   Missing rate: {X.isnull().sum().sum() / (X.shape[0] * X.shape[1]) * 100:.1f}%")
print(f"   Imputation Method: Median Imputation")
print(f"   Rationale: Robust to outliers, preserves data distribution")

# Save
missing_df.to_csv('publication/missing_value_analysis.csv', index=False)

print("\n✅ Missing value analysis complete")
