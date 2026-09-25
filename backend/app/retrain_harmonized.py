#!/usr/bin/env python3
"""
Retrain LightGBM on the harmonized MIMIC feature set.
"""

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score,
    precision_score, f1_score, brier_score_loss, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent))

from config import TABLE_DIR, MODEL_DIR, RANDOM_STATE, LGBM_PARAMS

import lightgbm as lgb


def evaluate(model, X, y):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        'recall': float(recall_score(y, y_pred, zero_division=0)),
        'auroc': float(roc_auc_score(y, y_prob)),
        'auprc': float(average_precision_score(y, y_prob)),
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'f1': float(f1_score(y, y_pred, zero_division=0)),
        'brier': float(brier_score_loss(y, y_prob)),
        'specificity': float(spec),
    }


def main():
    print("=" * 78)
    print("RETRAIN LIGHTGBM ON HARMONIZED FEATURES")
    print("=" * 78)

    train = pd.read_csv(TABLE_DIR / 'X_train_harmonized.csv')
    test = pd.read_csv(TABLE_DIR / 'X_test_harmonized.csv')
    print(f"Train: {train.shape}, Test: {test.shape}")

    y_train = train['outcome'].values
    y_test = test['outcome'].values
    X_train = train.drop(columns=['stay_id', 'outcome'])
    X_test = test.drop(columns=['stay_id', 'outcome'])
    feature_cols = list(X_train.columns)
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Train positives: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
    print(f"Test positives:  {y_test.sum()} ({y_test.mean()*100:.2f}%)")

    params = {k: v for k, v in LGBM_PARAMS.items()
              if k not in ('early_stopping_rounds',)}
    params['class_weight'] = 'balanced'
    params['verbose'] = -1

    model = lgb.LGBMClassifier(**params)

    X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
        X_train, y_train, test_size=0.15,
        stratify=y_train, random_state=RANDOM_STATE,
    )

    print("\nTraining...")
    model.fit(
        X_tr2, y_tr2,
        eval_set=[(X_val2, y_val2)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(0)],
    )

    test_metrics = evaluate(model, X_test, y_test)
    print(f"\nHarmonized model — MIMIC test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    original_metrics_path = TABLE_DIR / 'phase6_test_metrics.csv'
    if original_metrics_path.exists():
        original = pd.read_csv(original_metrics_path).iloc[0]
        print(f"\nOriginal 120-feature model (MIMIC test):")
        for k in ['auroc', 'auprc', 'recall', 'precision', 'f1', 'brier']:
            if k in original.index:
                print(f"  {k:12s}: {float(original[k]):.4f}")

        print(f"\nDelta (harmonized - original):")
        for k in ['auroc', 'auprc', 'recall', 'precision', 'f1', 'brier']:
            if k in original.index:
                delta = test_metrics.get(k, 0) - float(original[k])
                print(f"  {k:12s}: {delta:+.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / 'phase6_harmonized_model.pkl'
    config_path = MODEL_DIR / 'phase6_harmonized_config.pkl'

    joblib.dump(model, model_path)
    joblib.dump({
        'imputation': 'zero',
        'model': 'lightgbm',
        'mask_variant': 'with_masks',
        'feature_cols': feature_cols,
        'source': 'harmonized MIMIC/eICU intersection',
    }, config_path)

    pd.DataFrame([test_metrics]).to_csv(
        TABLE_DIR / 'phase6_harmonized_test_metrics.csv', index=False)

    print(f"\nSaved:")
    print(f"  {model_path}")
    print(f"  {config_path}")
    print(f"  phase6_harmonized_test_metrics.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
