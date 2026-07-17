"""
Enhanced Feature Engineering
Adds clinically meaningful derived features
"""

import pandas as pd
import numpy as np
import os

print("="*60)
print("ENHANCED FEATURE ENGINEERING")
print("="*60)

# Load original data
X = pd.read_csv("outputs/tables/X_features.csv")
y = pd.read_csv("outputs/tables/y_target.csv")

print(f"\n📊 Original Features: {X.shape[1]}")

# ============================================================
# STEP 1: Identify hour0 and hour6 columns
# ============================================================
hour0_cols = {col.replace('_hour0', ''): col for col in X.columns if '_hour0' in col}
hour6_cols = {col.replace('_hour6', ''): col for col in X.columns if '_hour6' in col}

print(f"   Hour 0 features: {len(hour0_cols)}")
print(f"   Hour 6 features: {len(hour6_cols)}")

# ============================================================
# STEP 2: Create Derived Clinical Features
# ============================================================
print("\n[1] Creating derived clinical features...")

# Shock Index = Heart Rate / SBP
if 'heart_rate' in hour0_cols and 'sbp' in hour0_cols:
    X['shock_index_hour0'] = X[hour0_cols['heart_rate']] / X[hour0_cols['sbp']]
    X['shock_index_hour6'] = X[hour6_cols['heart_rate']] / X[hour6_cols['sbp']]
    print("   ✅ Shock Index added")

# Mean Arterial Pressure (MAP) = (2*DBP + SBP) / 3
if 'sbp' in hour0_cols and 'dbp' in hour0_cols:
    X['map_hour0'] = (2 * X[hour0_cols['dbp']] + X[hour0_cols['sbp']]) / 3
    X['map_hour6'] = (2 * X[hour6_cols['dbp']] + X[hour6_cols['sbp']]) / 3
    print("   ✅ MAP added")

# Pulse Pressure = SBP - DBP
if 'sbp' in hour0_cols and 'dbp' in hour0_cols:
    X['pulse_pressure_hour0'] = X[hour0_cols['sbp']] - X[hour0_cols['dbp']]
    X['pulse_pressure_hour6'] = X[hour6_cols['sbp']] - X[hour6_cols['dbp']]
    print("   ✅ Pulse Pressure added")

# ============================================================
# STEP 3: Calculate Rate of Change (Slope) Features
# ============================================================
print("\n[2] Calculating slope features...")

# Slope = (hour6 - hour0) / 6 (per hour change)
for feature in ['heart_rate', 'sbp', 'dbp', 'gcs', 'lactate', 'creatinine']:
    if feature in hour0_cols and feature in hour6_cols:
        X[f'{feature}_slope'] = (X[hour6_cols[feature]] - X[hour0_cols[feature]]) / 6
        X[f'{feature}_pct_change'] = ((X[hour6_cols[feature]] - X[hour0_cols[feature]]) / 
                                      (X[hour0_cols[feature]] + 0.001)) * 100

print(f"   ✅ Slope and percentage change features added")

# ============================================================
# STEP 4: Variability Features
# ============================================================
print("\n[3] Calculating variability features...")

# Standard deviation between hour0 and hour6 (approximated)
for feature in ['heart_rate', 'sbp', 'dbp', 'lactate', 'creatinine']:
    if feature in hour0_cols and feature in hour6_cols:
        X[f'{feature}_variability'] = np.abs(X[hour6_cols[feature]] - X[hour0_cols[feature]])

print("   ✅ Variability features added")

# ============================================================
# STEP 5: Missing Value Indicators
# ============================================================
print("\n[4] Creating missing value indicators...")

# For each feature, create a binary indicator if value was missing
for feature in ['heart_rate', 'sbp', 'dbp', 'gcs', 'lactate', 'urine_output', 'fio2', 'creatinine']:
    for time in ['hour0', 'hour6']:
        col_name = f'{feature}_{time}'
        if col_name in X.columns:
            X[f'{feature}_{time}_missing'] = X[col_name].isna().astype(int)

print("   ✅ Missing value indicators added")

# ============================================================
# STEP 6: Clinical Risk Scores (Simplified)
# ============================================================
print("\n[5] Creating clinical risk scores...")

# Simple composite score based on vital signs
X['vital_risk_score_hour0'] = (
    (X[hour0_cols['heart_rate']] > 100).astype(int) +
    (X[hour0_cols['heart_rate']] < 60).astype(int) +
    (X[hour0_cols['sbp']] < 90).astype(int) +
    (X[hour0_cols['gcs']] < 13).astype(int) +
    (X[hour0_cols['lactate']] > 2).astype(int)
)

X['vital_risk_score_hour6'] = (
    (X[hour6_cols['heart_rate']] > 100).astype(int) +
    (X[hour6_cols['heart_rate']] < 60).astype(int) +
    (X[hour6_cols['sbp']] < 90).astype(int) +
    (X[hour6_cols['gcs']] < 13).astype(int) +
    (X[hour6_cols['lactate']] > 2).astype(int)
)

print("   ✅ Clinical risk scores added")

# ============================================================
# STEP 7: Handle Infinite and NaN Values
# ============================================================
print("\n[6] Cleaning derived features...")

# Replace infinite values with NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Fill remaining NaN with median for each column
for col in X.columns:
    if X[col].isnull().any():
        median_val = X[col].median()
        if pd.isna(median_val):
            median_val = 0
        X[col] = X[col].fillna(median_val)

print("   ✅ Derived features cleaned")

# ============================================================
# STEP 8: Save Enhanced Dataset
# ============================================================
print("\n[7] Saving enhanced dataset...")

X.to_csv("outputs/tables/X_features_enhanced.csv", index=False)
print(f"✅ Enhanced features saved: {X.shape[1]} features")

print("\n📊 Feature Summary:")
print(f"   Original features: 24")
print(f"   Enhanced features: {X.shape[1]}")
print(f"   New features added: {X.shape[1] - 24}")

print("\n📊 Sample of new features:")
new_cols = [c for c in X.columns if c not in pd.read_csv("outputs/tables/X_features.csv").columns]
print(new_cols[:10])

print("\n✅ Feature enhancement complete!")
