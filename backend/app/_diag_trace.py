#!/usr/bin/env python3
"""
Full trace of the prediction path for one patient.
Prints: what the API receives, what ml_model does, what the model outputs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

print("=" * 78)
print("TRACE: form input -> API -> ml_model -> pickle -> output")
print("=" * 78)

# ----------------------------------------------------------------
# 1. What does ml_model load?
# ----------------------------------------------------------------
from app.ml_model import ml_model

print("\n[1] MODEL LOADED")
print(f"    MODEL_PATH  = {ml_model.__class__.__module__}")
print(f"    type(model) = {type(ml_model.model).__name__ if ml_model.model else 'None'}")
print(f"    n_features  = {len(ml_model.feature_cols)}")
print(f"    n_masks     = {len(ml_model.kept_mask_cols)}")
print(f"    mask_variant= {ml_model.mask_variant}")
print(f"    imputation  = {ml_model.imputation}")

# ----------------------------------------------------------------
# 2. Simulate the exact dict the API would build from the form
# ----------------------------------------------------------------
healthy = {
    'patient_id': 'TEST-001', 'patient_name': 'Test Healthy',
    'age': 35, 'gender': 'Female', 'room': 'ICU-999', 'diagnosis': 'Observation',
    'heart_rate': 72, 'sbp': 118, 'dbp': 76, 'gcs': 15,
    'lactate': 0.9, 'urine_output': 120, 'fio2': 21, 'creatinine': 0.8,
    'status': 'Active',
}
critical = {
    'patient_id': 'TEST-002', 'patient_name': 'Test Critical',
    'age': 78, 'gender': 'Male', 'room': 'ICU-998', 'diagnosis': 'Septic shock',
    'heart_rate': 140, 'sbp': 70, 'dbp': 40, 'gcs': 4,
    'lactate': 8.0, 'urine_output': 5, 'fio2': 100, 'creatinine': 6.0,
    'status': 'Active',
}

# ----------------------------------------------------------------
# 3. What does _map_api_to_features produce?
# ----------------------------------------------------------------
print("\n[2] FEATURE VECTOR BUILT FROM FORM INPUT")
for tag, patient in [('HEALTHY', healthy), ('CRITICAL', critical)]:
    X_raw = ml_model._map_api_to_features(patient)
    n_pop = int(X_raw.notna().sum(axis=1).values[0])
    print(f"\n  {tag}: populated {n_pop} of {len(ml_model.feature_cols)} base features")
    row = X_raw.iloc[0]
    for k, v in row[row.notna()].items():
        print(f"        {k:40s} = {v}")

# ----------------------------------------------------------------
# 4. What does the FULL pipeline (impute + mask) produce?
# ----------------------------------------------------------------
print("\n[3] FINAL INPUT VECTOR AFTER IMPUTE + MASKS")
for tag, patient in [('HEALTHY', healthy), ('CRITICAL', critical)]:
    X_raw = ml_model._map_api_to_features(patient)
    X_imp = ml_model._apply_imputation(X_raw)
    X_masks = ml_model._add_masks(X_raw)
    for col in ml_model.feature_cols:
        if col in X_masks.columns:
            X_masks[col] = X_imp[col].values

    expected = (ml_model.feature_cols + ml_model.kept_mask_cols
                if ml_model.mask_variant == 'with_masks' and ml_model.kept_mask_cols
                else ml_model.feature_cols)
    for c in expected:
        if c not in X_masks.columns:
            X_masks[c] = 0
    X_final = X_masks[expected]

    # Count non-zero columns in the final vector
    nonzero = (X_final.iloc[0] != 0).sum()
    print(f"\n  {tag}: {nonzero} of {len(expected)} features are non-zero")
    print(f"        shape: {X_final.shape}")

    # Show the non-zero base features (not masks)
    row = X_final.iloc[0]
    nz_base = row[row.index.isin(ml_model.feature_cols) & (row != 0)]
    print(f"        non-zero base features ({len(nz_base)}):")
    for k, v in nz_base.items():
        print(f"          {k:40s} = {v}")

# ----------------------------------------------------------------
# 5. What does the model output?
# ----------------------------------------------------------------
print("\n[4] MODEL OUTPUT")
for tag, patient in [('HEALTHY', healthy), ('CRITICAL', critical)]:
    p = ml_model.predict(patient)
    print(f"    {tag}:  {p:.4f}")

# ----------------------------------------------------------------
# 6. What is the base rate the model returns for "no info"?
# ----------------------------------------------------------------
print("\n[5] BASE RATE TEST (empty input)")
p_empty = ml_model.predict({})
print(f"    Empty dict -> {p_empty:.4f}")
print("    If this is close to the HEALTHY and CRITICAL scores,")
print("    the model is ignoring the inputs entirely.")
