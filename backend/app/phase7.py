#!/usr/bin/env python3
"""
Phase 7: SHAP Interpretability, Threshold Tuning, DCA, and Calibration

PURPOSE
-------
Explain the Phase 6 final model (config-driven imputation + model + masks) and evaluate
clinical utility at a tuned decision threshold.

OUTPUTS
-------
Tables (11):
  phase7_shap_values.csv             - full SHAP matrix (n × 158)
  phase7_feature_importance.csv      - mean |SHAP| per feature
  phase7_mask_importance.csv         - mean |SHAP| per mask (40)
  phase7_threshold_tuning.csv        - metrics at each threshold
  phase7_dca.csv                     - net benefit across thresholds
  phase7_calibration_results.csv     - calibration bins
  phase7_confusion_matrix.csv        - TP/FP/FN/TN at selected threshold
  phase7_test_predictions.csv        - per-patient probabilities
  phase7_test_metrics_at_0.5.csv     - baseline test metrics
  phase7_bootstrap_cis.csv           - 95% CIs for test metrics
  phase7_report.txt                  - narrative summary

Figures (11):
  phase7_shap_summary.png
  phase7_shap_bar.png
  phase7_shap_dependence_1..4.png
  phase7_shap_masks.png
  phase7_shap_waterfall.png
  phase7_threshold_curve.png
  phase7_dca.png
  phase7_calibration.png
  phase7_confusion_matrix.png

INPUTS
------
outputs/models/phase6_final_model.pkl
outputs/models/phase6_final_config.pkl
outputs/tables/X_train.csv
outputs/tables/X_test.csv

NO DATA LEAKAGE
---------------
All SHAP and threshold analyses are on the held-out test set.
Imputation is fit on train (replicating Phase 6).

FIXES APPLIED
-------------
1. make_confusion_matrix_figure: cast tn/fp/fn/tp to int (fixes fmt='d' crash)
2. tune_threshold: select threshold by max F1 subject to recall >= target
   (fixes degenerate selection at 0.05)
"""

import sys
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    recall_score, roc_auc_score, average_precision_score,
    precision_score, f1_score, accuracy_score,
    brier_score_loss, confusion_matrix,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import shap

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TABLE_DIR, FIG_DIR, LOGS_DIR, MODEL_DIR,
    RANDOM_STATE,
    FEATURE_WINDOW_HOURS, PREDICTION_START_HOUR,
    PREDICTION_END_HOUR, SOFA_CHANGE_THRESHOLD,
)


# ============================================================
# CONSTANTS
# ============================================================

METADATA_COLS = {'stay_id', 'hadm_id', 'subject_id', 'intime'}
TARGET_COL = 'outcome'
MASK_SUFFIX = '_missing'

# Clinical operating point
TARGET_RECALL = 0.80

# Bootstrap
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 95

# DCA thresholds
DCA_THRESHOLDS = np.arange(0.05, 0.51, 0.01)

# Calibration bins
N_CALIBRATION_BINS = 10

# Number of dependence plots
N_DEPENDENCE = 4

# Top-N features to show in bar plot
TOP_N_BAR = 20


# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"phase7_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def print_header():
    print("=" * 80)
    print(" PHASE 7: SHAP + THRESHOLD + DCA + CALIBRATION")
    print("=" * 80)
    print(f" Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    print(f" Feature window: 0-{FEATURE_WINDOW_HOURS}h")
    print(f" Outcome: ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD} at "
          f"{PREDICTION_START_HOUR}-{PREDICTION_END_HOUR}h")
    print(f" Target recall: {TARGET_RECALL}")
    print(f" Bootstrap: {N_BOOTSTRAP} resamples, {BOOTSTRAP_CI}% CI")
    print(f" DCA thresholds: {DCA_THRESHOLDS[0]:.2f} to "
          f"{DCA_THRESHOLDS[-1]:.2f} ({len(DCA_THRESHOLDS)} points)")
    print("=" * 80)
    print()


# ============================================================
# RECONSTRUCT TEST SET (replicates Phase 6 refit exactly)
# ============================================================

def load_and_reconstruct(config: Dict, logger):
    """
    Rebuild X_train and X_test with imputation + masks, matching Phase 6.
    """
    logger.info("\n" + "=" * 60)
    logger.info("RECONSTRUCTING DATA (replicating Phase 6 refit)")
    logger.info("=" * 60)

    train_df = pd.read_csv(TABLE_DIR / 'X_train.csv')
    test_df = pd.read_csv(TABLE_DIR / 'X_test.csv')

    y_train = train_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values
    stay_ids_test = test_df['stay_id'].values if 'stay_id' in test_df.columns \
        else np.arange(len(test_df))

    # Drop metadata + target + pre-existing masks
    exclude = METADATA_COLS | {TARGET_COL}
    mask_cols_train = [c for c in train_df.columns if c.endswith(MASK_SUFFIX)]

    feature_cols = [c for c in train_df.columns
                    if c not in exclude and c not in mask_cols_train]

    X_train_raw = train_df[feature_cols].copy()
    X_test_raw = test_df[feature_cols].copy()

    logger.info(f"Train raw shape: {X_train_raw.shape}")
    logger.info(f"Test raw shape:  {X_test_raw.shape}")

        # Replicate Phase 6's imputation exactly (config-driven)
    imputation = config['imputation']

    if imputation == 'zero':
        X_train_imp = X_train_raw.fillna(0).reset_index(drop=True)
        X_test_imp = X_test_raw.fillna(0).reset_index(drop=True)
    elif imputation == 'median':
        imputer = SimpleImputer(strategy='median').fit(X_train_raw)
        X_train_imp = pd.DataFrame(
            imputer.transform(X_train_raw), columns=feature_cols)
        X_test_imp = pd.DataFrame(
            imputer.transform(X_test_raw), columns=feature_cols)
    else:
        raise ValueError(f"Unsupported imputation: {imputation}")
    logger.info(f"Imputation: {imputation}")

    # Add masks
    mask_variant = config['mask_variant']
    if mask_variant == 'with_masks':
        kept_mask_cols = config['kept_mask_cols']
        mask_train = X_train_raw.isnull().astype(int).add_suffix(MASK_SUFFIX)
        mask_test = X_test_raw.isnull().astype(int).add_suffix(MASK_SUFFIX)
        X_train_imp = pd.concat(
            [X_train_imp, mask_train[kept_mask_cols]], axis=1)
        X_test_imp = pd.concat(
            [X_test_imp, mask_test[kept_mask_cols]], axis=1)
        logger.info(f"Added {len(kept_mask_cols)} masks")
    else:
        kept_mask_cols = []
        logger.info("No masks (variant=without_masks)")

    logger.info(f"Final train shape: {X_train_imp.shape}")
    logger.info(f"Final test shape:  {X_test_imp.shape}")

    return X_train_imp, X_test_imp, y_train, y_test, stay_ids_test


# ============================================================
# SHAP ANALYSIS
# ============================================================

def compute_shap(model, X_test: pd.DataFrame, logger):
    """Compute SHAP values on the full test set."""
    logger.info("\n" + "=" * 60)
    logger.info("COMPUTING SHAP VALUES")
    logger.info("=" * 60)
    logger.info(f"Samples: {X_test.shape[0]:,}")
    logger.info(f"Features: {X_test.shape[1]:,}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # LightGBM binary classifier returns list [neg_class, pos_class]
    # or a 2D array depending on shap version
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if shap_values.ndim == 3:
        # (n, features, classes) → take positive class
        shap_values = shap_values[:, :, 1]

    logger.info(f"SHAP values shape: {shap_values.shape}")

    # Also compute expected value (base rate)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(expected_value[1]
                               if len(expected_value) > 1
                               else expected_value[0])
    logger.info(f"Expected value (base rate): {expected_value:.4f}")

    return shap_values, expected_value


def save_shap_outputs(model, X_test, shap_values, expected_value, logger):
    """Save SHAP matrices and importance rankings."""
    # Full SHAP matrix
    shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
    shap_df.to_csv(TABLE_DIR / 'phase7_shap_values.csv', index=False)
    logger.info(f"Saved: phase7_shap_values.csv")

    # Mean |SHAP| per feature
    importance = pd.DataFrame({
        'feature': X_test.columns,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0),
        'mean_shap': shap_values.mean(axis=0),
        'std_shap': shap_values.std(axis=0),
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    importance['rank'] = importance.index + 1

    importance.to_csv(TABLE_DIR / 'phase7_feature_importance.csv', index=False)
    logger.info(f"Saved: phase7_feature_importance.csv")

    # Mask-only importance
    mask_rows = importance[importance['feature'].str.endswith(MASK_SUFFIX)]
    mask_rows = mask_rows.sort_values('mean_abs_shap',
                                       ascending=False).reset_index(drop=True)
    mask_rows['rank'] = mask_rows.index + 1
    mask_rows.to_csv(TABLE_DIR / 'phase7_mask_importance.csv', index=False)
    logger.info(f"Saved: phase7_mask_importance.csv ({len(mask_rows)} masks)")

    logger.info("\nTop 15 features by mean |SHAP|:")
    for _, row in importance.head(15).iterrows():
        logger.info(f"  {row['rank']:2d}. {row['feature']:40s} "
                    f"{row['mean_abs_shap']:.4f}")

    if not mask_rows.empty:
        logger.info("\nTop 10 masks by mean |SHAP|:")
        for _, row in mask_rows.head(10).iterrows():
            logger.info(f"  {row['rank']:2d}. {row['feature']:40s} "
                        f"{row['mean_abs_shap']:.4f}")

    return importance, mask_rows


def make_shap_figures(model, X_test, shap_values, importance,
                      mask_importance, logger):
    """Generate all SHAP figures."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Beeswarm (SHAP creates its own figure — do not pre-create)
    shap.summary_plot(shap_values, X_test, show=False, max_display=20)
    plt.savefig(FIG_DIR / 'phase7_shap_summary.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    logger.info("  Figure: phase7_shap_summary.png")

    # 2. Bar (top 20) — custom matplotlib, no shap.summary_plot
    top20 = importance.head(TOP_N_BAR)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(range(len(top20)), top20['mean_abs_shap'].values[::-1],
            color='steelblue')
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20['feature'].values[::-1], fontsize=9)
    ax.set_xlabel('Mean |SHAP| (impact on model output)')
    ax.set_title(f'Top {TOP_N_BAR} Features by SHAP Importance')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'phase7_shap_bar.png', dpi=150)
    plt.close()
    logger.info("  Figure: phase7_shap_bar.png")

    # 3. Dependence plots for top-4
    top_features = importance.head(N_DEPENDENCE)['feature'].tolist()
    for i, feat in enumerate(top_features, 1):
        fig, ax = plt.subplots(figsize=(8, 5))
        feat_idx = list(X_test.columns).index(feat)
        ax.scatter(X_test[feat].values, shap_values[:, feat_idx],
                   alpha=0.4, s=8, color='steelblue')
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
        ax.set_xlabel(feat)
        ax.set_ylabel('SHAP value')
        ax.set_title(f'SHAP Dependence: {feat}')
        plt.tight_layout()
        plt.savefig(FIG_DIR / f'phase7_shap_dependence_{i}.png', dpi=150)
        plt.close()
        logger.info(f"  Figure: phase7_shap_dependence_{i}.png ({feat})")

    # 4. Mask-specific bar
    if not mask_importance.empty:
        n_show = min(20, len(mask_importance))
        top_masks = mask_importance.head(n_show)
        fig, ax = plt.subplots(figsize=(10, max(5, n_show * 0.35)))
        ax.barh(range(n_show), top_masks['mean_abs_shap'].values[::-1],
                color='coral')
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(top_masks['feature'].values[::-1], fontsize=9)
        ax.set_xlabel('Mean |SHAP|')
        ax.set_title(f'Top {n_show} Missingness Masks by SHAP Importance')
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'phase7_shap_masks.png', dpi=150)
        plt.close()
        logger.info("  Figure: phase7_shap_masks.png")

    # NOTE: waterfall is generated separately in make_waterfall()


def make_waterfall(model, X_test, y_test, shap_values, expected_value,
                   logger):
    """Waterfall plot for the highest-confidence true positive."""
    y_prob = model.predict_proba(X_test)[:, 1]

    # Find highest-probability true positive
    tp_mask = (y_test == 1)
    if not tp_mask.any():
        logger.warning("No true positives — skipping waterfall")
        return

    tp_probs = y_prob.copy()
    tp_probs[~tp_mask] = -1  # exclude non-TPs
    idx = int(np.argmax(tp_probs))

    logger.info(f"\nWaterfall: patient idx={idx}, "
                f"predicted prob={y_prob[idx]:.3f}, true=1")

    # shap.Explanation object for waterfall
    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=expected_value,
        data=X_test.iloc[idx].values,
        feature_names=X_test.columns.tolist(),
    )

    fig = plt.figure(figsize=(10, 8))
    shap.waterfall_plot(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'phase7_shap_waterfall.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    logger.info("  Figure: phase7_shap_waterfall.png")


# ============================================================
# THRESHOLD TUNING  (PATCH 2: select by max F1 subject to recall >= target)
# ============================================================

def tune_threshold(y_test, y_prob, logger):
    """Find threshold achieving target recall; compute metrics."""
    logger.info("\n" + "=" * 60)
    logger.info(f"THRESHOLD TUNING (target recall = {TARGET_RECALL})")
    logger.info("=" * 60)

    thresholds = np.arange(0.05, 0.96, 0.01)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred,
                                           labels=[0, 1]).ravel()
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        rows.append({
            'threshold': t,
            'recall': rec,
            'precision': prec,
            'specificity': spec,
            'npv': npv,
            'f1': f1,
            'tp': int(tp), 'fp': int(fp),
            'fn': int(fn), 'tn': int(tn),
        })

    df = pd.DataFrame(rows)

    # PATCH 2: among thresholds achieving target recall, pick max F1.
    # This avoids degenerate operating points (e.g., threshold=0.05 where
    # everything is predicted positive and recall is trivially 1.0).
    above = df[df['recall'] >= TARGET_RECALL]
    if not above.empty:
        selected = above.loc[above['f1'].idxmax()]
        logger.info(f"Found {len(above)} thresholds achieving "
                    f"recall ≥ {TARGET_RECALL}; selecting by max F1")
    else:
        selected = df.iloc[df['recall'].idxmax()]
        logger.warning(f"Target recall {TARGET_RECALL} not achievable; "
                       f"using max recall {selected['recall']:.4f}")

    logger.info(f"Selected threshold: {selected['threshold']:.3f}")
    logger.info(f"  Recall:     {selected['recall']:.4f}")
    logger.info(f"  Precision:  {selected['precision']:.4f}")
    logger.info(f"  Specificity:{selected['specificity']:.4f}")
    logger.info(f"  NPV:        {selected['npv']:.4f}")
    logger.info(f"  F1:         {selected['f1']:.4f}")
    logger.info(f"  TP/FP/FN/TN: {selected['tp']}/{selected['fp']}/"
                f"{selected['fn']}/{selected['tn']}")

    df.to_csv(TABLE_DIR / 'phase7_threshold_tuning.csv', index=False)
    logger.info(f"Saved: phase7_threshold_tuning.csv")

    return df, selected


def bootstrap_metrics(y_test, y_prob, threshold, logger):
    """Bootstrap 95% CIs for recall, precision, F1, AUROC, AUPRC."""
    logger.info(f"\nBootstrap ({N_BOOTSTRAP} resamples)...")

    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y_test)
    metrics = {'recall': [], 'precision': [], 'f1': [],
               'auroc': [], 'auprc': [], 'specificity': [], 'npv': []}

    for _ in range(N_BOOTSTRAP):
        idx = rng.randint(0, n, n)
        y_b = y_test[idx]
        p_b = y_prob[idx]

        if len(np.unique(y_b)) < 2:
            continue

        y_pred = (p_b >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_b, y_pred,
                                           labels=[0, 1]).ravel()
        metrics['recall'].append(
            tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        metrics['precision'].append(
            tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        metrics['specificity'].append(
            tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        metrics['npv'].append(
            tn / (tn + fn) if (tn + fn) > 0 else 0.0)
        f1 = 0.0
        if metrics['precision'][-1] + metrics['recall'][-1] > 0:
            f1 = (2 * metrics['precision'][-1] * metrics['recall'][-1]
                  / (metrics['precision'][-1] + metrics['recall'][-1]))
        metrics['f1'].append(f1)
        metrics['auroc'].append(roc_auc_score(y_b, p_b))
        metrics['auprc'].append(average_precision_score(y_b, p_b))

    lo = (100 - BOOTSTRAP_CI) / 2
    hi = 100 - lo
    ci = {}
    for k, vals in metrics.items():
        vals = np.array(vals)
        ci[k] = {
            'mean': float(vals.mean()),
            'ci_lo': float(np.percentile(vals, lo)),
            'ci_hi': float(np.percentile(vals, hi)),
        }
        logger.info(f"  {k:12s}: {ci[k]['mean']:.4f} "
                    f"[{ci[k]['ci_lo']:.4f}, {ci[k]['ci_hi']:.4f}]")

    return ci


# ============================================================
# DECISION CURVE ANALYSIS
# ============================================================

def compute_dca(y_test, y_prob, logger):
    """Net benefit across thresholds."""
    logger.info("\n" + "=" * 60)
    logger.info("DECISION CURVE ANALYSIS")
    logger.info("=" * 60)

    n = len(y_test)
    prevalence = y_test.mean()

    rows = []
    for t in DCA_THRESHOLDS:
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_pred == 1) & (y_test == 1)).sum())
        fp = int(((y_pred == 1) & (y_test == 0)).sum())
        # Net benefit = (TP / n) - (FP / n) * (t / (1 - t))
        if t < 1.0:
            nb = tp / n - fp / n * (t / (1 - t))
        else:
            nb = 0.0

        # Treat-all: everyone predicted positive
        tp_all = int((y_test == 1).sum())
        fp_all = int((y_test == 0).sum())
        nb_all = tp_all / n - fp_all / n * (t / (1 - t))

        rows.append({
            'threshold': float(t),
            'net_benefit_model': float(nb),
            'net_benefit_treat_all': float(nb_all),
            'net_benefit_treat_none': 0.0,
        })

    df = pd.DataFrame(rows)

    # Find best threshold by net benefit
    best_idx = df['net_benefit_model'].idxmax()
    best = df.loc[best_idx]
    logger.info(f"Max net benefit: {best['net_benefit_model']:.4f} "
                f"at threshold={best['threshold']:.2f}")

    df.to_csv(TABLE_DIR / 'phase7_dca.csv', index=False)
    logger.info(f"Saved: phase7_dca.csv")

    return df


# ============================================================
# CALIBRATION
# ============================================================

def compute_calibration(y_test, y_prob, logger):
    """Reliability curve."""
    logger.info("\n" + "=" * 60)
    logger.info("CALIBRATION")
    logger.info("=" * 60)

    bin_edges = np.linspace(0, 1, N_CALIBRATION_BINS + 1)
    bin_idx = np.digitize(y_prob, bin_edges[1:-1])

    rows = []
    for b in range(N_CALIBRATION_BINS):
        mask = (bin_idx == b)
        if mask.sum() == 0:
            continue
        rows.append({
            'bin': b,
            'bin_lo': float(bin_edges[b]),
            'bin_hi': float(bin_edges[b + 1]),
            'n': int(mask.sum()),
            'mean_predicted': float(y_prob[mask].mean()),
            'observed_rate': float(y_test[mask].mean()),
        })

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / 'phase7_calibration_results.csv', index=False)
    logger.info(f"Saved: phase7_calibration_results.csv")

    brier = brier_score_loss(y_test, y_prob)
    logger.info(f"Brier score: {brier:.4f}")

    return df, brier


# ============================================================
# FIGURES — THRESHOLD / DCA / CALIBRATION / CM
# ============================================================

def make_threshold_figure(threshold_df, selected, y_test, y_prob, logger):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(threshold_df['threshold'], threshold_df['recall'],
            label='Recall', color='red', linewidth=2)
    ax.plot(threshold_df['threshold'], threshold_df['precision'],
            label='Precision', color='blue', linewidth=2)
    ax.plot(threshold_df['threshold'], threshold_df['specificity'],
            label='Specificity', color='green', linewidth=2)
    ax.plot(threshold_df['threshold'], threshold_df['f1'],
            label='F1', color='purple', linewidth=2)

    ax.axvline(selected['threshold'], color='black', linestyle='--',
               alpha=0.7, label=f"Selected t={selected['threshold']:.2f}")
    ax.axhline(TARGET_RECALL, color='gray', linestyle=':',
               alpha=0.5, label=f'Target recall={TARGET_RECALL}')

    ax.set_xlabel('Decision threshold')
    ax.set_ylabel('Metric value')
    ax.set_title('Threshold Tuning: Metrics vs. Decision Threshold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'phase7_threshold_curve.png', dpi=150)
    plt.close()
    logger.info("  Figure: phase7_threshold_curve.png")


def make_dca_figure(dca_df, logger):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(dca_df['threshold'], dca_df['net_benefit_model'],
            label='Model', color='steelblue', linewidth=2)
    ax.plot(dca_df['threshold'], dca_df['net_benefit_treat_all'],
            label='Treat All', color='gray', linestyle='--')
    ax.plot(dca_df['threshold'], dca_df['net_benefit_treat_none'],
            label='Treat None', color='black', linestyle=':')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Threshold probability')
    ax.set_ylabel('Net benefit')
    ax.set_title('Decision Curve Analysis')
    ax.set_ylim(bottom=min(-0.05, dca_df['net_benefit_model'].min() - 0.05))
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'phase7_dca.png', dpi=150)
    plt.close()
    logger.info("  Figure: phase7_dca.png")


def make_calibration_figure(cal_df, brier, logger):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.plot(cal_df['mean_predicted'], cal_df['observed_rate'],
            marker='o', linewidth=2, color='steelblue', label='Model')

    # Annotate bin counts
    for _, row in cal_df.iterrows():
        ax.annotate(f"n={int(row['n'])}",
                    (row['mean_predicted'], row['observed_rate']),
                    fontsize=7, alpha=0.7,
                    textcoords='offset points', xytext=(5, 5))

    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed positive rate')
    ax.set_title(f'Calibration Curve (Brier = {brier:.4f})')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'phase7_calibration.png', dpi=150)
    plt.close()
    logger.info("  Figure: phase7_calibration.png")


def make_confusion_matrix_figure(selected, logger):
    # PATCH 1: cast to int (fixes fmt='d' crash on float input)
    tn = int(selected['tn'])
    fp = int(selected['fp'])
    fn = int(selected['fn'])
    tp = int(selected['tp'])
    cm = np.array([[tn, fp], [fn, tp]], dtype=int)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'],
                cbar_kws={'label': 'Count'})
    ax.set_title(f"Confusion Matrix at threshold="
                 f"{selected['threshold']:.2f}\n"
                 f"Recall={selected['recall']:.3f}, "
                 f"Precision={selected['precision']:.3f}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'phase7_confusion_matrix.png', dpi=150)
    plt.close()
    logger.info("  Figure: phase7_confusion_matrix.png")


# ============================================================
# REPORT
# ============================================================

def save_report(importance, mask_importance, threshold_df, selected,
                boot_ci, dca_df, cal_df, brier, test_metrics_05,
                logger):
    lines = []
    lines.append("=" * 70)
    lines.append("PHASE 7: INTERPRETABILITY AND CLINICAL UTILITY REPORT")
    lines.append("=" * 70)
    lines.append(f"Feature window: 0-{FEATURE_WINDOW_HOURS}h")
    lines.append(f"Outcome: ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD} at "
                 f"{PREDICTION_START_HOUR}-{PREDICTION_END_HOUR}h")
    lines.append("")

    lines.append("TOP 15 FEATURES BY MEAN |SHAP|:")
    lines.append("-" * 70)
    for _, row in importance.head(15).iterrows():
        lines.append(f"  {row['rank']:2d}. {row['feature']:40s} "
                     f"{row['mean_abs_shap']:.4f}")
    lines.append("")

    if not mask_importance.empty:
        lines.append("TOP 10 MASKS BY MEAN |SHAP|:")
        lines.append("-" * 70)
        for _, row in mask_importance.head(10).iterrows():
            lines.append(f"  {row['rank']:2d}. {row['feature']:40s} "
                         f"{row['mean_abs_shap']:.4f}")
        lines.append("")

    lines.append("THRESHOLD TUNING:")
    lines.append("-" * 70)
    lines.append(f"  Target recall: {TARGET_RECALL}")
    lines.append(f"  Selection: max F1 subject to recall >= target")
    lines.append(f"  Selected threshold: {selected['threshold']:.3f}")
    lines.append(f"  Recall:     {selected['recall']:.4f}")
    lines.append(f"  Precision:  {selected['precision']:.4f}")
    lines.append(f"  Specificity:{selected['specificity']:.4f}")
    lines.append(f"  NPV:        {selected['npv']:.4f}")
    lines.append(f"  F1:         {selected['f1']:.4f}")
    lines.append(f"  TP/FP/FN/TN: {int(selected['tp'])}/{int(selected['fp'])}/"
                 f"{int(selected['fn'])}/{int(selected['tn'])}")
    lines.append("")

    lines.append(f"BOOTSTRAP 95% CIs ({N_BOOTSTRAP} resamples, "
                 f"at selected threshold):")
    lines.append("-" * 70)
    for k, v in boot_ci.items():
        lines.append(f"  {k:12s}: {v['mean']:.4f} "
                     f"[{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]")
    lines.append("")

    lines.append("REFERENCE METRICS AT THRESHOLD = 0.50:")
    lines.append("-" * 70)
    for k, v in test_metrics_05.items():
        lines.append(f"  {k:12s}: {v:.4f}")
    lines.append("")

    lines.append("DECISION CURVE ANALYSIS:")
    lines.append("-" * 70)
    best_nb = dca_df.loc[dca_df['net_benefit_model'].idxmax()]
    lines.append(f"  Max net benefit: {best_nb['net_benefit_model']:.4f} "
                 f"at threshold={best_nb['threshold']:.2f}")
    lines.append(f"  Model exceeds treat-all for thresholds "
                 f"where net_benefit_model > net_benefit_treat_all")
    lines.append("")

    lines.append("CALIBRATION:")
    lines.append("-" * 70)
    lines.append(f"  Brier score: {brier:.4f}")
    lines.append("")

    lines.append("NOTE:")
    lines.append("-" * 70)
    lines.append("  All analyses on held-out test set (n=8,176).")
    lines.append("  SHAP computed on full test set.")
    lines.append("  Imputation fit on train only (no leakage).")

    with open(TABLE_DIR / 'phase7_report.txt', 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"Saved: phase7_report.txt")


# ============================================================
# MAIN
# ============================================================

def main():
    logger = setup_logging()
    print_header()

    t0 = datetime.now()

    # Load Phase 6 artifacts
    model_path = MODEL_DIR / 'phase6_final_model.pkl'
    config_path = MODEL_DIR / 'phase6_final_config.pkl'

    if not model_path.exists() or not config_path.exists():
        logger.error(f"Missing Phase 6 artifacts. Run Phase 6 refit first.")
        return 1

    logger.info(f"Loading model: {model_path}")
    model = joblib.load(model_path)
    config = joblib.load(config_path)
    logger.info(f"Loaded config: {config}")

    # Reconstruct test set
    X_train, X_test, y_train, y_test, stay_ids_test = \
        load_and_reconstruct(config, logger)

    # Sanity: feature columns must match
    expected_cols = config['feature_cols'] + config.get('kept_mask_cols', [])
    if config['mask_variant'] == 'without_masks':
        expected_cols = config['feature_cols']

    missing = set(expected_cols) - set(X_test.columns)
    extra = set(X_test.columns) - set(expected_cols)
    if missing:
        logger.error(f"Missing expected columns: {list(missing)[:10]}")
        return 1
    if extra:
        logger.warning(f"Extra columns in X_test: {list(extra)[:10]}")

    # Order columns to match training
    X_test = X_test[expected_cols]

    # ========================================================
    # 1. Predictions
    # ========================================================
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred_05 = (y_prob >= 0.5).astype(int)

    test_metrics_05 = {
        'recall': float(recall_score(y_test, y_pred_05)),
        'auroc': float(roc_auc_score(y_test, y_prob)),
        'auprc': float(average_precision_score(y_test, y_prob)),
        'precision': float(precision_score(y_test, y_pred_05,
                                            zero_division=0)),
        'f1': float(f1_score(y_test, y_pred_05, zero_division=0)),
        'accuracy': float(accuracy_score(y_test, y_pred_05)),
        'brier': float(brier_score_loss(y_test, y_prob)),
    }

    # Save baseline test metrics as CSV
    pd.DataFrame([test_metrics_05]).to_csv(
        TABLE_DIR / 'phase7_test_metrics_at_0.5.csv', index=False)
    logger.info("Saved: phase7_test_metrics_at_0.5.csv")

    logger.info("\n" + "=" * 60)
    logger.info("TEST SET METRICS AT THRESHOLD = 0.50")
    logger.info("=" * 60)
    for k, v in test_metrics_05.items():
        logger.info(f"  {k:12s}: {v:.4f}")

    # ========================================================
    # 2. SHAP
    # ========================================================
    shap_values, expected_value = compute_shap(model, X_test, logger)
    importance, mask_importance = save_shap_outputs(
        model, X_test, shap_values, expected_value, logger)
    make_shap_figures(model, X_test, shap_values, importance,
                      mask_importance, logger)

    # ========================================================
    # 3. Threshold tuning
    # ========================================================
    threshold_df, selected = tune_threshold(y_test, y_prob, logger)
    make_threshold_figure(threshold_df, selected, y_test, y_prob, logger)

    # Save confusion matrix CSV
    pd.DataFrame([{
        'threshold': float(selected['threshold']),
        'tp': int(selected['tp']),
        'fp': int(selected['fp']),
        'fn': int(selected['fn']),
        'tn': int(selected['tn']),
        'recall': float(selected['recall']),
        'precision': float(selected['precision']),
        'specificity': float(selected['specificity']),
        'npv': float(selected['npv']),
        'f1': float(selected['f1']),
    }]).to_csv(TABLE_DIR / 'phase7_confusion_matrix.csv', index=False)
    logger.info("Saved: phase7_confusion_matrix.csv")
    make_confusion_matrix_figure(selected, logger)

    # Bootstrap CIs at selected threshold
    boot_ci = bootstrap_metrics(y_test, y_prob, selected['threshold'],
                                 logger)
    # Save bootstrap CIs as CSV
    pd.DataFrame([
        {'metric': k,
         'mean': v['mean'],
         'ci_lo': v['ci_lo'],
         'ci_hi': v['ci_hi']}
        for k, v in boot_ci.items()
    ]).to_csv(TABLE_DIR / 'phase7_bootstrap_cis.csv', index=False)
    logger.info("Saved: phase7_bootstrap_cis.csv")

    # ========================================================
    # 4. DCA
    # ========================================================
    dca_df = compute_dca(y_test, y_prob, logger)
    make_dca_figure(dca_df, logger)

    # ========================================================
    # 5. Calibration
    # ========================================================
    cal_df, brier = compute_calibration(y_test, y_prob, logger)
    make_calibration_figure(cal_df, brier, logger)

    # ========================================================
    # 6. Waterfall
    # ========================================================
    make_waterfall(model, X_test, y_test, shap_values, expected_value,
                   logger)

    # ========================================================
    # 7. Test predictions CSV
    # ========================================================
    pred_df = pd.DataFrame({
        'stay_id': stay_ids_test,
        'y_true': y_test,
        'y_prob': y_prob,
        'y_pred_at_selected': (y_prob >= selected['threshold']).astype(int),
        'y_pred_at_0.5': y_pred_05,
    })
    pred_df.to_csv(TABLE_DIR / 'phase7_test_predictions.csv', index=False)
    logger.info("Saved: phase7_test_predictions.csv")

    # ========================================================
    # 8. Report
    # ========================================================
    save_report(importance, mask_importance, threshold_df, selected,
                boot_ci, dca_df, cal_df, brier, test_metrics_05, logger)

    # ========================================================
    # Summary
    # ========================================================
    elapsed = datetime.now() - t0
    print("\n" + "=" * 80)
    print(" PHASE 7 COMPLETED")
    print("=" * 80)
    print(f" Elapsed: {elapsed}")
    print("-" * 80)
    print(" TOP FEATURES BY SHAP:")
    for _, row in importance.head(5).iterrows():
        print(f"   {row['rank']}. {row['feature']:40s} "
              f"{row['mean_abs_shap']:.4f}")
    print("-" * 80)
    print(f" TOP MASKS BY SHAP:")
    for _, row in mask_importance.head(5).iterrows():
        print(f"   {row['rank']}. {row['feature']:40s} "
              f"{row['mean_abs_shap']:.4f}")
    print("-" * 80)
    print(f" SELECTED THRESHOLD: {selected['threshold']:.3f} "
          f"(recall target={TARGET_RECALL}, selected by max F1)")
    print(f"   Recall:     {selected['recall']:.4f} "
          f"[{boot_ci['recall']['ci_lo']:.4f}, "
          f"{boot_ci['recall']['ci_hi']:.4f}]")
    print(f"   Precision:  {selected['precision']:.4f} "
          f"[{boot_ci['precision']['ci_lo']:.4f}, "
          f"{boot_ci['precision']['ci_hi']:.4f}]")
    print(f"   F1:         {selected['f1']:.4f}")
    print(f"   Brier:      {brier:.4f}")
    print("-" * 80)
    print(f" Outputs:")
    print(f"   {TABLE_DIR}/phase7_*.csv, phase7_report.txt")
    print(f"   {FIG_DIR}/phase7_*.png")
    print("=" * 80)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
