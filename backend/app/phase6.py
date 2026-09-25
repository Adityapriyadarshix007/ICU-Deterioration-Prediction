#!/usr/bin/env python3
"""
Phase 6: Imputation × Model Interaction with Mask Variants (FINAL + PATCHED)

PURPOSE
-------
Systematically compare tabular imputation strategies and test whether
missingness-indicator features (masks) improve prediction of ΔSOFA ≥ 2.

DESIGN
------
Tier 1 (12 combos): 4 imputations × 3 models × WITHOUT masks
Tier 2 (12 combos): 4 imputations × 3 models × WITH masks (deduplicated)
Tier 3 (4 combos):  MAR/MNAR sensitivity analysis
Tier 4:             Final model refit + recommendation report

All combinations use 5-fold StratifiedKFold on train.
Imputation is fit PER FOLD (no leakage).
Final model is refit on full train and evaluated once on test.
Tier 3 uses a train-only validation split for early stopping.

SELECTION CRITERION
-------------------
Best combination selected by CV AUPRC (average precision), because:
  - Positive class (ΔSOFA progression) is the minority (18.9%)
  - AUPRC directly measures positive-class ranking quality
  - AUROC can appear strong while precision is poor at low prevalence
  - Threshold tuning is performed separately in Phase 7

PATCHES APPLIED
---------------
1. Imputer closures enforce column order (X_new = X_new[cols])
2. XGBoost early stopping uses version check (no bare TypeError fallback)
3. Tier 3 uses train-only validation split for early stopping
4. Final model refit + save to MODEL_DIR for Phase 7
5. MNAR fill guards against NaN quantiles
6. Selection criterion = AUPRC (was recall)
7. --refit-only flag to reuse cached grid results

DATA
----
X_train.csv (46,327 × 212), X_test.csv (8,176 × 212)

NO DATA LEAKAGE
---------------
Features from 0-6h window; outcome from 6-18h.
Imputation fit inside CV folds; scaler fit inside folds.
Early stopping uses train-only validation splits.
Test set evaluated exactly once per tier.

USAGE
-----
    python3 phase6.py --tier 1        # without masks only
    python3 phase6.py --tier 12       # without + with masks
    python3 phase6.py --tier all      # full pipeline
    python3 phase6.py --refit-only    # reuse cached grid; refit only
"""

import sys
import json
import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    recall_score, roc_auc_score, average_precision_score,
    precision_score, f1_score, accuracy_score,
    brier_score_loss, confusion_matrix,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

import joblib
import catboost as cb
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TABLE_DIR, FIG_DIR, LOGS_DIR, MODEL_DIR,
    RANDOM_STATE,
    CATBOOST_PARAMS, XGB_PARAMS, LGBM_PARAMS,
    FEATURE_WINDOW_HOURS, PREDICTION_START_HOUR,
    PREDICTION_END_HOUR, SOFA_CHANGE_THRESHOLD,
    N_FOLDS,
)


# ============================================================
# CONSTANTS
# ============================================================

METADATA_COLS = {'stay_id', 'hadm_id', 'subject_id', 'intime'}
TARGET_COL = 'outcome'
MASK_SUFFIX = '_missing'

IMPUTATION_METHODS = ['zero', 'median', 'knn', 'ice']
MODEL_NAMES = ['catboost', 'xgboost', 'lightgbm']
MASK_VARIANTS = ['without_masks', 'with_masks']

# ICE runtime controls
ICE_MAX_ITER = 5
ICE_SAMPLE_POSTERIOR = False
ICE_N_NEAREST_FEATURES = 10

# Class weight (matches (1-p)/p for p=0.189)
POS_WEIGHT = 4.3

# Early stopping
EARLY_STOPPING_ROUNDS = 50

# Validation fraction for early stopping (used in Tier 3 and final refit)
EARLY_STOP_VAL_FRAC = 0.15

# XGBoost version (for API routing)
XGB_VERSION = tuple(int(x) for x in xgb.__version__.split('.')[:2])


# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"phase6_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def print_header(tier: str):
    print("=" * 80)
    print(" PHASE 6: IMPUTATION × MODEL INTERACTION")
    print("=" * 80)
    print(f" Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Tier: {tier}")
    print("-" * 80)
    print(f" Feature window: 0-{FEATURE_WINDOW_HOURS}h")
    print(f" Outcome: ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD} at "
          f"{PREDICTION_START_HOUR}-{PREDICTION_END_HOUR}h")
    print(f" CV: {N_FOLDS}-fold StratifiedKFold (imputation fit per fold)")
    print(f" Imputations: {IMPUTATION_METHODS}")
    print(f" Models: {MODEL_NAMES}")
    print(f" Mask variants: {MASK_VARIANTS}")
    print(f" Selection criterion: AUPRC (positive-class ranking)")
    print(f" XGBoost version: {XGB_VERSION}")
    print("-" * 80)
    print(" TIER PLAN:")
    print("   Tier 1: 4 imp × 3 models × without_masks = 12 combos")
    print("   Tier 2: 4 imp × 3 models × with_masks    = 12 combos")
    print("   Tier 3: MAR vs MNAR sensitivity (4 assumptions)")
    print("   Tier 4: Final model refit + recommendation report")
    print("=" * 80)
    print()


# ============================================================
# DATA LOADING
# ============================================================

def load_raw_data(logger):
    logger.info("\n" + "=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)

    train_path = TABLE_DIR / "X_train.csv"
    test_path = TABLE_DIR / "X_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Missing X_train.csv or X_test.csv. Run Phase 4 first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info(f"Loaded X_train: {train_df.shape}")
    logger.info(f"Loaded X_test:  {test_df.shape}")

    if TARGET_COL not in train_df.columns:
        raise ValueError(f"'{TARGET_COL}' not found in X_train.csv")

    y_train = train_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    exclude = set(METADATA_COLS) | {TARGET_COL}
    mask_cols_train = [c for c in train_df.columns if c.endswith(MASK_SUFFIX)]

    if mask_cols_train:
        logger.info(f"Dropping {len(mask_cols_train)} existing masks "
                    f"(recreated from raw NaN pattern)")

    feature_cols = [c for c in train_df.columns
                    if c not in exclude and c not in mask_cols_train]

    X_train_raw = train_df[feature_cols].copy()
    X_test_raw = test_df[feature_cols].copy()

    logger.info(f"Feature matrix shape: {X_train_raw.shape}")
    logger.info(f"Missing cells (train): {int(X_train_raw.isnull().sum().sum()):,}")
    logger.info(f"Missing cells (test):  {int(X_test_raw.isnull().sum().sum()):,}")
    logger.info(f"Class balance (train): {np.bincount(y_train).tolist()}")
    logger.info(f"Class balance (test):  {np.bincount(y_test).tolist()}")

    return X_train_raw, X_test_raw, y_train, y_test


# ============================================================
# MASK UTILITIES
# ============================================================

def deduplicate_masks(X_raw: pd.DataFrame, logger) -> List[str]:
    mask_df = X_raw.isnull().astype(int).add_suffix(MASK_SUFFIX)
    mask_cols = mask_df.columns.tolist()
    if not mask_cols:
        return []

    dup = mask_df.T.duplicated(keep='first')
    kept = [c for c, is_dup in zip(mask_cols, dup) if not is_dup]
    dropped = [c for c, is_dup in zip(mask_cols, dup) if is_dup]

    logger.info(f"  Masks: {len(mask_cols)} total → "
                f"{len(kept)} unique "
                f"({len(dropped)} duplicates dropped)")
    if dropped:
        logger.info(f"  Sample dropped: {dropped[:8]}")
    return kept


def build_masks(X_raw: pd.DataFrame, kept_mask_cols: List[str]) -> pd.DataFrame:
    mask_df = X_raw.isnull().astype(int).add_suffix(MASK_SUFFIX)
    return mask_df[kept_mask_cols].reset_index(drop=True)


# ============================================================
# IMPUTATION (fit per fold) — PATCH 1
# ============================================================

def fit_imputer(method: str, X_fit: pd.DataFrame):
    """
    Fit imputer on X_fit. Returns transform closure.
    All closures enforce column order to prevent silent misalignment.
    """
    cols = list(X_fit.columns)

    if method == 'zero':
        def transform(X_new):
            return X_new[cols].fillna(0).reset_index(drop=True)
        return transform

    if method == 'median':
        imputer = SimpleImputer(strategy='median').fit(X_fit)

        def transform(X_new):
            arr = imputer.transform(X_new[cols])
            return pd.DataFrame(arr, columns=cols)
        return transform

    if method == 'knn':
        scaler = StandardScaler().fit(X_fit)
        imputer = KNNImputer(n_neighbors=5, weights='uniform')
        imputer.fit(scaler.transform(X_fit))

        def transform(X_new):
            arr = scaler.transform(X_new[cols])
            arr = imputer.transform(arr)
            return pd.DataFrame(arr, columns=cols)
        return transform

    if method == 'ice':
        imputer = IterativeImputer(
            max_iter=ICE_MAX_ITER,
            random_state=RANDOM_STATE,
            sample_posterior=ICE_SAMPLE_POSTERIOR,
            n_nearest_features=ICE_N_NEAREST_FEATURES,
        ).fit(X_fit)

        def transform(X_new):
            arr = imputer.transform(X_new[cols])
            return pd.DataFrame(arr, columns=cols)
        return transform

    raise ValueError(f"Unknown imputation method: {method}")


# ============================================================
# MODEL FACTORY — PATCH 2 (XGB version routing)
# ============================================================

def build_model(model_name: str):
    if model_name == 'catboost':
        params = {k: v for k, v in CATBOOST_PARAMS.items()
                  if k not in ('early_stopping_rounds', 'verbose')}
        params['class_weights'] = {0: 1.0, 1: POS_WEIGHT}
        params['verbose'] = False
        params['od_type'] = 'Iter'
        params['od_wait'] = EARLY_STOPPING_ROUNDS
        return cb.CatBoostClassifier(**params)

    if model_name == 'xgboost':
        params = {k: v for k, v in XGB_PARAMS.items()
                  if k not in ('early_stopping_rounds',)}
        params['scale_pos_weight'] = POS_WEIGHT
        params['eval_metric'] = 'aucpr'
        params['verbosity'] = 0
        if XGB_VERSION >= (2, 0):
            params['early_stopping_rounds'] = EARLY_STOPPING_ROUNDS
        return xgb.XGBClassifier(**params)

    if model_name == 'lightgbm':
        params = {k: v for k, v in LGBM_PARAMS.items()
                  if k not in ('early_stopping_rounds',)}
        params['class_weight'] = 'balanced'
        params['verbose'] = -1
        return lgb.LGBMClassifier(**params)

    raise ValueError(f"Unknown model: {model_name}")


def fit_model_with_early_stopping(model_name: str, model,
                                  X_tr, y_tr, X_val, y_val):
    """Fit with early stopping. Version-safe for XGBoost."""
    if model_name == 'catboost':
        model.fit(X_tr, y_tr,
                  eval_set=(X_val, y_val),
                  use_best_model=True,
                  verbose=False)
    elif model_name == 'xgboost':
        if XGB_VERSION >= (2, 0):
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
        else:
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                      verbose=False)
    elif model_name == 'lightgbm':
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS,
                                                verbose=False),
                             lgb.log_evaluation(0)])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, X, y):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {
        'recall': float(recall_score(y, y_pred, zero_division=0)),
        'auroc': float(roc_auc_score(y, y_prob)),
        'auprc': float(average_precision_score(y, y_prob)),
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'f1': float(f1_score(y, y_pred, zero_division=0)),
        'accuracy': float(accuracy_score(y, y_pred)),
        'brier': float(brier_score_loss(y, y_prob)),
        'specificity': float(specificity),
        'npv': float(npv),
    }


# ============================================================
# CV RUNNER
# ============================================================

def run_combination(X_raw, y, imp_method, model_name,
                    use_masks, kept_mask_cols, logger,
                    n_folds=N_FOLDS):
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                         random_state=RANDOM_STATE)
    fold_metrics = []
    n_features_observed = None

    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_raw, y)):
        X_tr_raw = X_raw.iloc[tr_idx].reset_index(drop=True)
        X_val_raw = X_raw.iloc[val_idx].reset_index(drop=True)
        y_tr = y[tr_idx]
        y_val = y[val_idx]

        # Fit imputer on train-fold ONLY
        transform = fit_imputer(imp_method, X_tr_raw)
        X_tr = transform(X_tr_raw)
        X_val = transform(X_val_raw)

        if use_masks and kept_mask_cols:
            mask_tr = build_masks(X_tr_raw, kept_mask_cols)
            mask_val = build_masks(X_val_raw, kept_mask_cols)
            X_tr = pd.concat([X_tr, mask_tr], axis=1)
            X_val = pd.concat([X_val, mask_val], axis=1)

        n_features_observed = X_tr.shape[1]

        model = build_model(model_name)
        model = fit_model_with_early_stopping(
            model_name, model, X_tr, y_tr, X_val, y_val
        )
        m = evaluate(model, X_val, y_val)
        m['fold'] = fold_idx
        fold_metrics.append(m)

    fold_df = pd.DataFrame(fold_metrics)
    result = {
        'imputation': imp_method,
        'model': model_name,
        'mask_variant': 'with_masks' if use_masks else 'without_masks',
        'n_features': int(n_features_observed),
    }
    for col in [c for c in fold_df.columns if c != 'fold']:
        result[f'{col}_mean'] = float(fold_df[col].mean())
        result[f'{col}_std'] = float(fold_df[col].std())
    return result


# ============================================================
# TIER 1 + 2: GRID
# ============================================================

def run_grid(X_raw, y_train, mask_variant, kept_mask_cols, logger):
    use_masks = (mask_variant == 'with_masks')
    logger.info(f"\n{'='*60}")
    logger.info(f"GRID: {mask_variant.upper()}")
    logger.info(f"{'='*60}")

    results = []
    total = len(IMPUTATION_METHODS) * len(MODEL_NAMES)
    count = 0

    for imp in IMPUTATION_METHODS:
        for mdl in MODEL_NAMES:
            count += 1
            logger.info(f"\n  [{count}/{total}] {imp.upper()} + "
                        f"{mdl.upper()} ({mask_variant})")
            try:
                r = run_combination(
                    X_raw, y_train, imp, mdl,
                    use_masks=use_masks,
                    kept_mask_cols=kept_mask_cols,
                    logger=logger,
                )
                results.append(r)
                logger.info(f"    CV Recall: {r['recall_mean']:.4f} ± "
                            f"{r['recall_std']:.4f}")
                logger.info(f"    CV AUROC:  {r['auroc_mean']:.4f} ± "
                            f"{r['auroc_std']:.4f}")
                logger.info(f"    CV AUPRC:  {r['auprc_mean']:.4f}")
                logger.info(f"    n_features: {r['n_features']}")
            except Exception as e:
                logger.error(f"    FAILED: {e}")
                results.append({
                    'imputation': imp, 'model': mdl,
                    'mask_variant': mask_variant,
                    'error': str(e),
                })

    return pd.DataFrame(results)


# ============================================================
# TIER 3: SENSITIVITY — PATCH 3 (train-only val split) + PATCH 5 (NaN guard)
# ============================================================

def _fit_and_eval(model_name, X_tr, y_tr, X_te, y_te):
    """Split train, fit with early stopping, evaluate on test."""
    X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
        X_tr, y_tr, test_size=EARLY_STOP_VAL_FRAC,
        stratify=y_tr, random_state=RANDOM_STATE,
    )
    model = build_model(model_name)
    model = fit_model_with_early_stopping(
        model_name, model, X_tr2, y_tr2, X_val2, y_val2,
    )
    return evaluate(model, X_te, y_te)


def run_sensitivity(X_raw, X_test_raw, y_train, y_test, logger):
    logger.info(f"\n{'='*60}")
    logger.info("TIER 3: SENSITIVITY ANALYSIS (MAR vs MNAR)")
    logger.info(f"{'='*60}")

    results = []
    model_name = 'catboost'

    # --- MAR: ICE ---
    logger.info("\n  [1/4] MAR (ICE):")
    transform = fit_imputer('ice', X_raw)
    X_tr = transform(X_raw)
    X_te = transform(X_test_raw)
    m = _fit_and_eval(model_name, X_tr, y_train, X_te, y_test)
    results.append({'assumption': 'MAR_ICE', **m})
    logger.info(f"    Recall: {m['recall']:.4f}, AUROC: {m['auroc']:.4f}")

    # --- MAR: Median ---
    logger.info("\n  [2/4] MAR (Median):")
    transform = fit_imputer('median', X_raw)
    X_tr = transform(X_raw)
    X_te = transform(X_test_raw)
    m = _fit_and_eval(model_name, X_tr, y_train, X_te, y_test)
    results.append({'assumption': 'MAR_Median', **m})
    logger.info(f"    Recall: {m['recall']:.4f}, AUROC: {m['auroc']:.4f}")

    # --- MNAR: Pessimistic ---
    logger.info("\n  [3/4] MNAR (Pessimistic — 1.2 × train p75):")
    X_tr = X_raw.copy()
    X_te = X_test_raw.copy()
    for col in X_tr.columns:
        if X_tr[col].isnull().any():
            p75 = X_tr[col].quantile(0.75)
            if pd.isna(p75):
                p75 = 0.0
            fill = 1.2 * p75
            X_tr[col] = X_tr[col].fillna(fill)
            X_te[col] = X_te[col].fillna(fill)
    m = _fit_and_eval(model_name, X_tr, y_train, X_te, y_test)
    results.append({'assumption': 'MNAR_Pessimistic', **m})
    logger.info(f"    Recall: {m['recall']:.4f}, AUROC: {m['auroc']:.4f}")

    # --- MNAR: Optimistic ---
    logger.info("\n  [4/4] MNAR (Optimistic — 0.8 × train p25):")
    X_tr = X_raw.copy()
    X_te = X_test_raw.copy()
    for col in X_tr.columns:
        if X_tr[col].isnull().any():
            p25 = X_tr[col].quantile(0.25)
            if pd.isna(p25):
                p25 = 0.0
            fill = 0.8 * p25
            X_tr[col] = X_tr[col].fillna(fill)
            X_te[col] = X_te[col].fillna(fill)
    m = _fit_and_eval(model_name, X_tr, y_train, X_te, y_test)
    results.append({'assumption': 'MNAR_Optimistic', **m})
    logger.info(f"    Recall: {m['recall']:.4f}, AUROC: {m['auroc']:.4f}")

    sens_df = pd.DataFrame(results)
    auroc_range = sens_df['auroc'].max() - sens_df['auroc'].min()
    recall_range = sens_df['recall'].max() - sens_df['recall'].min()
    logger.info(f"\n  AUROC range:  {auroc_range:.4f}")
    logger.info(f"  Recall range: {recall_range:.4f}")

    if auroc_range < 0.05 and recall_range < 0.05:
        robustness = "Robust"
    elif auroc_range < 0.10 and recall_range < 0.10:
        robustness = "Moderately Robust"
    else:
        robustness = "Sensitive"
    logger.info(f"  Conclusion: {robustness}")

    return sens_df


# ============================================================
# TIER 4: RECOMMENDATION + FINAL MODEL REFIT — PATCH 4 + PATCH 6 (AUPRC)
# ============================================================

def _successful_rows(df):
    if 'error' not in df.columns:
        return df
    return df[df['error'].isna()]


def build_recommendation(all_df, sens_df, logger):
    logger.info(f"\n{'='*60}")
    logger.info("TIER 4: RECOMMENDATION")
    logger.info(f"{'='*60}")

    ok = _successful_rows(all_df)
    if ok.empty:
        logger.error("No successful combinations — cannot recommend")
        return {}

    # ------------------------------------------------------------
    # SELECTION CRITERION: AUPRC
    # ------------------------------------------------------------
    # AUPRC (average precision) is the primary selection metric because:
    #   (a) the positive class (ΔSOFA progression) is the minority (18.9%)
    #   (b) AUPRC directly measures positive-class ranking quality
    #   (c) AUROC can appear "good" while precision is poor at low prevalence
    # Threshold tuning is performed separately in Phase 7.
    best = ok.loc[ok['auprc_mean'].idxmax()]

    rec = {
        'best_imputation': best['imputation'],
        'best_model': best['model'],
        'best_mask_variant': best['mask_variant'],
        'best_recall': float(best['recall_mean']),
        'best_recall_std': float(best['recall_std']),
        'best_auroc': float(best['auroc_mean']),
        'best_auprc': float(best['auprc_mean']),
        'best_f1': float(best['f1_mean']),
        'n_features': int(best['n_features']),
        'selection_metric': 'auprc',
    }

    best_imp = best['imputation']
    best_mdl = best['model']

    wom = ok[(ok['imputation'] == best_imp) &
             (ok['model'] == best_mdl) &
             (ok['mask_variant'] == 'without_masks')]
    wm = ok[(ok['imputation'] == best_imp) &
            (ok['model'] == best_mdl) &
            (ok['mask_variant'] == 'with_masks')]

    if not wom.empty and not wm.empty:
        rec['mask_delta_recall'] = float(
            wm.iloc[0]['recall_mean'] - wom.iloc[0]['recall_mean'])
        rec['mask_delta_auroc'] = float(
            wm.iloc[0]['auroc_mean'] - wom.iloc[0]['auroc_mean'])
        rec['mask_delta_auprc'] = float(
            wm.iloc[0]['auprc_mean'] - wom.iloc[0]['auprc_mean'])
    else:
        rec['mask_delta_recall'] = None
        rec['mask_delta_auroc'] = None
        rec['mask_delta_auprc'] = None

    if sens_df is not None and not sens_df.empty:
        auroc_range = sens_df['auroc'].max() - sens_df['auroc'].min()
        recall_range = sens_df['recall'].max() - sens_df['recall'].min()
        if auroc_range < 0.05 and recall_range < 0.05:
            rec['robustness'] = "Robust"
        elif auroc_range < 0.10 and recall_range < 0.10:
            rec['robustness'] = "Moderately Robust"
        else:
            rec['robustness'] = "Sensitive"
        rec['sensitivity_auroc_range'] = float(auroc_range)
        rec['sensitivity_recall_range'] = float(recall_range)
    else:
        rec['robustness'] = "Not evaluated"

    logger.info(f"\n  Selection criterion: AUPRC")
    logger.info(f"  Best combination:")
    logger.info(f"    Imputation: {rec['best_imputation'].upper()}")
    logger.info(f"    Model:      {rec['best_model'].upper()}")
    logger.info(f"    Masks:      {rec['best_mask_variant']}")
    logger.info(f"    AUPRC:      {rec['best_auprc']:.4f}")
    logger.info(f"    AUROC:      {rec['best_auroc']:.4f}")
    logger.info(f"    Recall:     {rec['best_recall']:.4f} ± "
                f"{rec['best_recall_std']:.4f}")
    logger.info(f"    F1:         {rec['best_f1']:.4f}")
    logger.info(f"    n_features: {rec['n_features']}")
    if rec['mask_delta_auprc'] is not None:
        logger.info(f"\n  Mask effect (same imp/model):")
        logger.info(f"    ΔAUPRC:  {rec['mask_delta_auprc']:+.4f}")
        logger.info(f"    ΔAUROC:  {rec['mask_delta_auroc']:+.4f}")
        logger.info(f"    ΔRecall: {rec['mask_delta_recall']:+.4f}")
    logger.info(f"\n  Robustness: {rec['robustness']}")

    return rec


def refit_final_model(rec, X_train_raw, X_test_raw, y_train, y_test,
                      kept_mask_cols, logger):
    """
    PATCH 4: Refit the best config on full train, save for Phase 7.
    Uses train-only validation split for early stopping.
    """
    logger.info(f"\n{'='*60}")
    logger.info("FINAL MODEL REFIT (for Phase 7)")
    logger.info(f"{'='*60}")

    best_imp = rec['best_imputation']
    best_mdl = rec['best_model']
    use_masks = rec['best_mask_variant'] == 'with_masks'

    logger.info(f"  Imputation: {best_imp}")
    logger.info(f"  Model:      {best_mdl}")
    logger.info(f"  Masks:      {use_masks}")

    # Imputer on full train
    transform = fit_imputer(best_imp, X_train_raw)
    X_tr_full = transform(X_train_raw)
    X_te_full = transform(X_test_raw)

    if use_masks and kept_mask_cols:
        X_tr_full = pd.concat(
            [X_tr_full, build_masks(X_train_raw, kept_mask_cols)], axis=1)
        X_te_full = pd.concat(
            [X_te_full, build_masks(X_test_raw, kept_mask_cols)], axis=1)

    # Early-stopping split from train
    X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
        X_tr_full, y_train, test_size=EARLY_STOP_VAL_FRAC,
        stratify=y_train, random_state=RANDOM_STATE,
    )

    final_model = build_model(best_mdl)
    final_model = fit_model_with_early_stopping(
        best_mdl, final_model, X_tr2, y_tr2, X_val2, y_val2,
    )

    # Test set metrics
    test_metrics = evaluate(final_model, X_te_full, y_test)
    logger.info(f"\n  Test set metrics:")
    for k, v in test_metrics.items():
        logger.info(f"    {k:12s}: {v:.4f}")

    # Save model + config
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_DIR / 'phase6_final_model.pkl')
    joblib.dump({
        'imputation': best_imp,
        'model': best_mdl,
        'mask_variant': rec['best_mask_variant'],
        'kept_mask_cols': kept_mask_cols,
        'feature_cols': list(X_train_raw.columns),
    }, MODEL_DIR / 'phase6_final_config.pkl')
    logger.info(f"\n  Saved: {MODEL_DIR}/phase6_final_model.pkl")
    logger.info(f"  Saved: {MODEL_DIR}/phase6_final_config.pkl")

    # Save test metrics
    pd.DataFrame([test_metrics]).to_csv(
        TABLE_DIR / 'phase6_test_metrics.csv', index=False)
    logger.info(f"  Saved: {TABLE_DIR}/phase6_test_metrics.csv")

    return test_metrics


# ============================================================
# FIGURES
# ============================================================

def make_figures(all_df, logger):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ok = _successful_rows(all_df)

    if ok.empty:
        logger.warning("No results to plot")
        return

    for variant in MASK_VARIANTS:
        sub = ok[ok['mask_variant'] == variant]
        if sub.empty:
            continue
        pivot = sub.pivot(index='imputation', columns='model',
                          values='auprc_mean')
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax,
                    cbar_kws={'label': 'CV AUPRC (mean)'})
        ax.set_title(f'CV AUPRC — {variant}')
        ax.set_xlabel('Model')
        ax.set_ylabel('Imputation')
        plt.tight_layout()
        out = FIG_DIR / f'phase6_heatmap_auprc_{variant}.png'
        plt.savefig(out, dpi=150)
        plt.close()
        logger.info(f"  Figure saved: {out.name}")

    # Top-5 by AUPRC: mask effect comparison
    top = ok.groupby(['imputation', 'model'])['auprc_mean'].mean() \
            .sort_values(ascending=False).head(5).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(top))
    width = 0.35
    wom_vals, wm_vals = [], []
    for _, r in top.iterrows():
        w = ok[(ok['imputation'] == r['imputation']) &
               (ok['model'] == r['model']) &
               (ok['mask_variant'] == 'without_masks')]
        m = ok[(ok['imputation'] == r['imputation']) &
               (ok['model'] == r['model']) &
               (ok['mask_variant'] == 'with_masks')]
        wom_vals.append(float(w.iloc[0]['auprc_mean']) if not w.empty else 0)
        wm_vals.append(float(m.iloc[0]['auprc_mean']) if not m.empty else 0)

    labels = [f"{r['imputation']}\n{r['model']}" for _, r in top.iterrows()]
    ax.bar(x - width/2, wom_vals, width, label='without_masks')
    ax.bar(x + width/2, wm_vals, width, label='with_masks')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('CV AUPRC (mean)')
    ax.set_title('Mask Effect on Top-5 (Imputation, Model) Pairs — AUPRC')
    ax.legend()
    plt.tight_layout()
    out = FIG_DIR / 'phase6_mask_comparison.png'
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"  Figure saved: {out.name}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Phase 6: Imputation × Model with mask variants'
    )
    parser.add_argument('--tier', type=str, default='all',
                        choices=['1', '12', 'all'])
    parser.add_argument('--refit-only', action='store_true',
                        help='Skip the grid; refit final model using '
                             'cached grid results (AUPRC-selected)')
    args = parser.parse_args()

    logger = setup_logging()
    print_header(args.tier)

    t0 = datetime.now()

    # ------------------------------------------------------------
    # REFIT-ONLY MODE
    # ------------------------------------------------------------
    if args.refit_only:
        logger.info("\n" + "=" * 60)
        logger.info("REFIT-ONLY MODE")
        logger.info("=" * 60)
        logger.info("Loading cached grid results...")

        grid_path = TABLE_DIR / 'interaction_results_all.csv'
        sens_path = TABLE_DIR / 'sensitivity_analysis_results.csv'

        if not grid_path.exists():
            logger.error(f"Cached grid not found: {grid_path}")
            logger.error("Run full Phase 6 first: python3 phase6.py --tier all")
            return 1

        all_df = pd.read_csv(grid_path)
        logger.info(f"Loaded grid results: {all_df.shape}")

        if sens_path.exists():
            sens_df = pd.read_csv(sens_path)
            logger.info(f"Loaded sensitivity results: {sens_df.shape}")
        else:
            sens_df = None
            logger.warning(f"Sensitivity file not found: {sens_path}")

        # Load raw data (for mask deduplication and refit)
        X_train_raw, X_test_raw, y_train, y_test = load_raw_data(logger)

        logger.info("\n" + "-" * 60)
        logger.info("MASK DEDUPLICATION")
        logger.info("-" * 60)
        kept_mask_cols = deduplicate_masks(X_train_raw, logger)

        # Re-select best combination (AUPRC)
        logger.info("\n" + "-" * 60)
        logger.info("RE-SELECTING BEST COMBINATION (AUPRC)")
        logger.info("-" * 60)
        rec = build_recommendation(all_df, sens_df, logger)

        if not rec:
            logger.error("Recommendation failed — cannot refit")
            return 1

        with open(TABLE_DIR / 'phase6_recommendation.json', 'w') as f:
            json.dump(rec, f, indent=2, default=str)

        # Refit final model
        test_metrics = refit_final_model(
            rec, X_train_raw, X_test_raw, y_train, y_test,
            kept_mask_cols, logger,
        )

        # Regenerate figures with AUPRC metric
        make_figures(all_df, logger)

        # Summary
        elapsed = datetime.now() - t0
        print("\n" + "=" * 80)
        print(" PHASE 6 REFIT COMPLETED")
        print("=" * 80)
        print(f" Elapsed: {elapsed}")
        print(f" Selected by: AUPRC")
        print(f" Best: {rec.get('best_imputation', 'N/A').upper()} + "
              f"{rec.get('best_model', 'N/A').upper()} "
              f"({rec.get('best_mask_variant', 'N/A')})")
        print(f" CV AUPRC:  {rec.get('best_auprc', 0):.4f}")
        print(f" CV AUROC:  {rec.get('best_auroc', 0):.4f}")
        print(f" CV Recall: {rec.get('best_recall', 0):.4f} ± "
              f"{rec.get('best_recall_std', 0):.4f}")
        if test_metrics:
            print("-" * 80)
            print(f" Test AUPRC:  {test_metrics['auprc']:.4f}")
            print(f" Test AUROC:  {test_metrics['auroc']:.4f}")
            print(f" Test Recall: {test_metrics['recall']:.4f}")
            print(f" Test F1:     {test_metrics['f1']:.4f}")
        print("-" * 80)
        print(" Updated artifacts:")
        print(f"   {TABLE_DIR}/phase6_recommendation.json")
        print(f"   {TABLE_DIR}/phase6_test_metrics.csv")
        print(f"   {MODEL_DIR}/phase6_final_model.pkl")
        print(f"   {MODEL_DIR}/phase6_final_config.pkl")
        print("=" * 80)
        print()
        return 0

    # ------------------------------------------------------------
    # NORMAL FULL RUN (or tiered run)
    # ------------------------------------------------------------
    X_train_raw, X_test_raw, y_train, y_test = load_raw_data(logger)

    # Deduplicate masks
    logger.info("\n" + "-" * 60)
    logger.info("MASK DEDUPLICATION")
    logger.info("-" * 60)
    kept_mask_cols = deduplicate_masks(X_train_raw, logger)

    # ---- Tier 1 ----
    results_t1 = None
    if args.tier in ('1', '12', 'all'):
        logger.info(f"\n{'#'*70}\n# TIER 1: WITHOUT MASKS\n{'#'*70}")
        results_t1 = run_grid(X_train_raw, y_train,
                              'without_masks', [], logger)
        results_t1.to_csv(TABLE_DIR / 'interaction_results_without_masks.csv',
                          index=False)
        logger.info(f"\n  Saved: interaction_results_without_masks.csv")

    # ---- Tier 2 ----
    results_t2 = None
    if args.tier in ('12', 'all'):
        logger.info(f"\n{'#'*70}\n# TIER 2: WITH MASKS\n{'#'*70}")
        results_t2 = run_grid(X_train_raw, y_train,
                              'with_masks', kept_mask_cols, logger)
        results_t2.to_csv(TABLE_DIR / 'interaction_results_with_masks.csv',
                          index=False)
        logger.info(f"\n  Saved: interaction_results_with_masks.csv")

    # ---- Combine ----
    all_df = pd.concat([d for d in [results_t1, results_t2]
                        if d is not None], ignore_index=True)
    all_df.to_csv(TABLE_DIR / 'interaction_results_all.csv', index=False)
    logger.info(f"\n  Saved: interaction_results_all.csv "
                f"({len(all_df)} rows)")

    # ---- Tier 3 ----
    sens_df = None
    if args.tier == 'all':
        logger.info(f"\n{'#'*70}\n# TIER 3: SENSITIVITY\n{'#'*70}")
        sens_df = run_sensitivity(X_train_raw, X_test_raw,
                                  y_train, y_test, logger)
        sens_df.to_csv(TABLE_DIR / 'sensitivity_analysis_results.csv',
                       index=False)
        logger.info(f"\n  Saved: sensitivity_analysis_results.csv")

    # ---- Tier 4 ----
    rec = build_recommendation(all_df, sens_df, logger)
    with open(TABLE_DIR / 'phase6_recommendation.json', 'w') as f:
        json.dump(rec, f, indent=2, default=str)

    # Final model refit (only on 'all' tier)
    test_metrics = None
    if args.tier == 'all' and rec:
        test_metrics = refit_final_model(
            rec, X_train_raw, X_test_raw, y_train, y_test,
            kept_mask_cols, logger,
        )

    # Figures
    make_figures(all_df, logger)

    # Summary
    elapsed = datetime.now() - t0
    print("\n" + "=" * 80)
    print(" PHASE 6 COMPLETED")
    print("=" * 80)
    print(f" Elapsed: {elapsed}")
    print(f" Combinations tested: {len(all_df)}")
    print(f" Selection: AUPRC")
    print(f" Best: {rec.get('best_imputation', 'N/A').upper()} + "
          f"{rec.get('best_model', 'N/A').upper()} "
          f"({rec.get('best_mask_variant', 'N/A')})")
    print(f" CV AUPRC:  {rec.get('best_auprc', 0):.4f}")
    print(f" CV AUROC:  {rec.get('best_auroc', 0):.4f}")
    print(f" CV Recall: {rec.get('best_recall', 0):.4f} ± "
          f"{rec.get('best_recall_std', 0):.4f}")
    print(f" Robustness: {rec.get('robustness', 'N/A')}")
    if test_metrics:
        print(f" Test AUPRC:  {test_metrics['auprc']:.4f}")
        print(f" Test AUROC:  {test_metrics['auroc']:.4f}")
        print(f" Test Recall: {test_metrics['recall']:.4f}")
    print("-" * 80)
    print(" Outputs:")
    print(f"   {TABLE_DIR}/interaction_results_all.csv")
    print(f"   {TABLE_DIR}/sensitivity_analysis_results.csv")
    print(f"   {TABLE_DIR}/phase6_recommendation.json")
    if test_metrics:
        print(f"   {TABLE_DIR}/phase6_test_metrics.csv")
        print(f"   {MODEL_DIR}/phase6_final_model.pkl")
        print(f"   {MODEL_DIR}/phase6_final_config.pkl")
    print("=" * 80)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
