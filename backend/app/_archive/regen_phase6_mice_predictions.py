#!/usr/bin/env python3
"""
Regenerate Phase 6 MICE + LightGBM predictions.
Same as regen_best_predictions.py but with MICE imputation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score, recall_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
from config import TABLE_DIR, RANDOM_STATE, LGBM_PARAMS

# --- Load ---
print("Loading data...")
X_train = pd.read_csv(TABLE_DIR / "X_train.csv")
X_test  = pd.read_csv(TABLE_DIR / "X_test.csv")

y_train = X_train['outcome'].values
y_test  = X_test['outcome'].values

# Drop masks
mask_cols = [c for c in X_train.columns if c.endswith('_missing')]
X_train = X_train.drop(columns=mask_cols + ['outcome'])
X_test  = X_test.drop(columns=[c for c in X_test.columns if c.endswith('_missing')] + ['outcome'])

print(f"  X_train: {X_train.shape}")
print(f"  X_test:  {X_test.shape}")

# --- MICE ---
print("Running MICE imputation...")
imputer = IterativeImputer(
    max_iter=10,
    random_state=RANDOM_STATE,
    n_nearest_features=5
)
X_train_imp = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns, index=X_train.index
)
X_test_imp = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns, index=X_test.index
)
print("  MICE done.")

# --- LightGBM ---
params = {k: v for k, v in LGBM_PARAMS.items()
          if k != 'early_stopping_rounds'}
params['verbose'] = -1

print("Training LightGBM...")
model = lgb.LGBMClassifier(**params)
model.fit(X_train_imp, y_train)

y_prob = model.predict_proba(X_test_imp)[:, 1]

# --- Sanity check ---
auc = roc_auc_score(y_test, y_prob)
rec_50 = recall_score(y_test, (y_prob >= 0.50).astype(int))
auprc = average_precision_score(y_test, y_prob)

print()
print("=" * 60)
print("SANITY CHECK (should match Phase 6 MICE + LightGBM)")
print("=" * 60)
print(f"  AUC:             {auc:.4f}    (Phase 6 said 0.8020)")
print(f"  Recall @ 0.50:   {rec_50:.4f}    (Phase 6 said 0.2389)")
print(f"  AUPRC:           {auprc:.4f}")
print("=" * 60)

# --- Save ---
out = pd.DataFrame({'y_true': y_test, 'y_prob': y_prob})
out.to_csv(TABLE_DIR / 'phase6_mice_predictions.csv', index=False)
print(f"  Saved: {TABLE_DIR / 'phase6_mice_predictions.csv'}")
