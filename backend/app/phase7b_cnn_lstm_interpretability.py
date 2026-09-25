#!/usr/bin/env python3
"""
Phase 7b: CNN-LSTM Interpretability + Clinical Utility

PURPOSE
-------
Explain the Phase 6b final CNN-LSTM and evaluate its clinical utility
at a tuned decision threshold. Mirrors Phase 7 (tree) for the DL branch.

METHODS
-------
- Permutation importance (per input channel, over test set)
- Attention weights (mean over test patients, per timestep)
- Mask ablation (performance with mask channels zeroed)
- Threshold tuning (same target: recall >= 0.80)
- Decision curve analysis (same threshold grid as Phase 7)
- Calibration (reliability + Brier)
- Confusion matrix at selected threshold

INPUTS
------
outputs/models/phase6b/cnn_lstm_final.pt      (single final model)
outputs/sequences/sequences_test.npz

OUTPUTS
-------
Tables:
    phase7b_permutation_importance.csv
    phase7b_attention_summary.csv
    phase7b_mask_ablation.csv
    phase7b_threshold_tuning.csv
    phase7b_dca.csv
    phase7b_calibration_results.csv
    phase7b_confusion_matrix.csv
    phase7b_test_metrics_at_0.5.csv
    phase7b_bootstrap_cis.csv
    phase7b_test_predictions.csv
    phase7b_report.txt

Figures:
    phase7b_permutation_bar.png
    phase7b_attention_heatmap.png
    phase7b_mask_ablation.png
    phase7b_threshold_curve.png
    phase7b_dca.png
    phase7b_calibration.png
    phase7b_confusion_matrix.png

FIXES APPLIED
-------------
- attention_summary: cast timestep to int for log formatting
- save_report: consistent int/float formatting with attention_summary
"""

import sys
import json
import time
import logging
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    recall_score, roc_auc_score, average_precision_score,
    precision_score, f1_score, accuracy_score,
    brier_score_loss, confusion_matrix,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TABLE_DIR, FIG_DIR, LOGS_DIR, MODEL_DIR, SEQUENCE_DIR,
    DEEP_LEARNING_PARAMS as P,
    RANDOM_STATE,
    FEATURE_WINDOW_HOURS, PREDICTION_START_HOUR,
    PREDICTION_END_HOUR, SOFA_CHANGE_THRESHOLD,
)

from phase6b_cnn_lstm import (
    get_device, check_mps_bce, set_seeds,
    SequenceDataset, CNNLSTMAttention, AttentionPooling,
)


# ============================================================
# CONSTANTS
# ============================================================

TARGET_RECALL = 0.80
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 95
DCA_THRESHOLDS = np.arange(0.05, 0.51, 0.01)
N_CALIBRATION_BINS = 10
N_PERMUTATION_REPEATS = 5
TOP_N_BAR = 20

FEATURE_NAMES = [
    'heart_rate', 'respiratory_rate', 'spo2', 'map', 'gcs_total',
    'fio2', 'pao2',
    'creatinine', 'lactate', 'bilirubin', 'platelets', 'wbc',
    'hemoglobin', 'sodium', 'potassium', 'bun', 'glucose',
    'norepinephrine', 'epinephrine', 'dopamine',
]

assert len(FEATURE_NAMES) == P['n_features'], (
    f"Feature name count {len(FEATURE_NAMES)} != P['n_features'] {P['n_features']}"
)


# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"phase7b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    print(" PHASE 7b: CNN-LSTM INTERPRETABILITY + CLINICAL UTILITY")
    print("=" * 80)
    print(f" Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    print(f" Feature window: 0-{FEATURE_WINDOW_HOURS}h")
    print(f" Outcome: ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD} at "
          f"{PREDICTION_START_HOUR}-{PREDICTION_END_HOUR}h")
    print(f" Target recall: {TARGET_RECALL}")
    print(f" Bootstrap: {N_BOOTSTRAP} resamples, {BOOTSTRAP_CI}% CI")
    print(f" Permutation: {N_PERMUTATION_REPEATS} repeats per channel")
    print("=" * 80)
    print()


# ============================================================
# MODEL LOADING
# ============================================================

def load_final_dl_model(device, logger):
    ckpt_path = Path(P['checkpoint_dir']) / 'cnn_lstm_final.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing {ckpt_path}. Run Phase 6b first."
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CNNLSTMAttention(P).to(device)
    model.load_state_dict({
        k: v.to(device) for k, v in ckpt['model_state_dict'].items()
    })
    model.eval()
    logger.info(f"Loaded: {ckpt_path}")
    logger.info(f"  best_epoch: {ckpt.get('best_epoch')}")
    logger.info(f"  test_metrics: {ckpt.get('test_metrics')}")
    return model


# ============================================================
# INFERENCE HELPERS
# ============================================================

@torch.no_grad()
def predict_probs(model, X, device, batch_size=128):
    model.eval()
    ds = SequenceDataset(X, np.zeros(len(X), dtype=np.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=P['num_workers'],
                        pin_memory=P['pin_memory'])
    probs = []
    for Xb, _ in loader:
        Xb = Xb.to(device)
        logits = model(Xb)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred,
                                       labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'auroc': float(roc_auc_score(y_true, y_prob)),
        'auprc': float(average_precision_score(y_true, y_prob)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'brier': float(brier_score_loss(y_true, y_prob)),
        'specificity': float(spec),
        'npv': float(npv),
    }


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

def permutation_importance(model, X, y, device, logger,
                            n_repeats=N_PERMUTATION_REPEATS):
    logger.info("\n" + "=" * 60)
    logger.info(f"PERMUTATION IMPORTANCE ({n_repeats} repeats/channel)")
    logger.info("=" * 60)

    rng = np.random.RandomState(RANDOM_STATE)

    base_probs = predict_probs(model, X, device)
    base_auprc = average_precision_score(y, base_probs)
    base_auroc = roc_auc_score(y, base_probs)
    logger.info(f"  Baseline AUPRC: {base_auprc:.4f}")
    logger.info(f"  Baseline AUROC: {base_auroc:.4f}")

    n_features = P['n_features']
    n_channels = P['n_channels']

    rows = []
    for k in range(n_channels):
        is_mask = (k >= n_features)
        feat_idx = k - n_features if is_mask else k
        feat_name = FEATURE_NAMES[feat_idx]
        label = f"{feat_name}_mask" if is_mask else feat_name

        auprc_drops = []
        auroc_drops = []

        for _ in range(n_repeats):
            X_perm = X.copy()
            perm = rng.permutation(len(X))
            X_perm[:, :, k] = X_perm[perm, :, k]

            probs = predict_probs(model, X_perm, device)
            auprc = average_precision_score(y, probs)
            auroc = roc_auc_score(y, probs)

            auprc_drops.append(base_auprc - auprc)
            auroc_drops.append(base_auroc - auroc)

        rows.append({
            'channel_index': k,
            'label': label,
            'is_mask': is_mask,
            'auprc_drop_mean': float(np.mean(auprc_drops)),
            'auprc_drop_std': float(np.std(auprc_drops)),
            'auroc_drop_mean': float(np.mean(auroc_drops)),
            'auroc_drop_std': float(np.std(auroc_drops)),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('auprc_drop_mean', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1

    df.to_csv(TABLE_DIR / 'phase7b_permutation_importance.csv', index=False)
    logger.info(f"  Saved: phase7b_permutation_importance.csv")

    logger.info("\n  Top 15 channels by AUPRC drop:")
    for _, r in df.head(15).iterrows():
        logger.info(f"    {int(r['rank']):2d}. {r['label']:35s} "
                    f"ΔAUPRC={r['auprc_drop_mean']:+.4f}")

    return df, base_auprc, base_auroc


# ============================================================
# ATTENTION VISUALIZATION
# ============================================================

def extract_attention_weights(model, X, device, batch_size=128):
    model.eval()
    captured = {}

    def hook(module, inp, out):
        x = inp[0]
        scores = module.v(torch.tanh(module.W(x)))
        weights = torch.softmax(scores, dim=1)
        captured['weights'] = weights.detach().cpu().numpy()

    handle = model.attention.register_forward_hook(hook)

    ds = SequenceDataset(X, np.zeros(len(X), dtype=np.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=P['num_workers'],
                        pin_memory=P['pin_memory'])

    all_weights = []
    with torch.no_grad():
        for Xb, _ in loader:
            Xb = Xb.to(device)
            _ = model(Xb)
            w = captured['weights'].squeeze(-1)
            all_weights.append(w)

    handle.remove()
    return np.concatenate(all_weights, axis=0)


def attention_summary(model, X, y, device, logger):
    logger.info("\n" + "=" * 60)
    logger.info("ATTENTION WEIGHTS")
    logger.info("=" * 60)

    weights = extract_attention_weights(model, X, device)
    T = weights.shape[1]

    rows = []
    for t in range(T):
        rows.append({
            'timestep': int(t),
            'bin_start_min': int(t * 30),
            'bin_end_min': int((t + 1) * 30),
            'mean_weight': float(weights[:, t].mean()),
            'std_weight': float(weights[:, t].std()),
            'mean_weight_progressor': float(weights[y == 1, t].mean()),
            'mean_weight_stable': float(weights[y == 0, t].mean()),
        })

    df = pd.DataFrame(rows)
    df['timestep'] = df['timestep'].astype(int)
    df['bin_start_min'] = df['bin_start_min'].astype(int)
    df['bin_end_min'] = df['bin_end_min'].astype(int)
    df.to_csv(TABLE_DIR / 'phase7b_attention_summary.csv', index=False)
    logger.info(f"  Saved: phase7b_attention_summary.csv")

    logger.info("\n  Mean attention per timestep:")
    for _, r in df.iterrows():
        logger.info(f"    t={int(r['timestep']):2d} "
                    f"({int(r['bin_start_min']):3d}-{int(r['bin_end_min']):3d} min): "
                    f"{r['mean_weight']:.4f} "
                    f"(prog={r['mean_weight_progressor']:.4f}, "
                    f"stable={r['mean_weight_stable']:.4f})")

    return df, weights


def make_attention_heatmap(weights, y, logger):
    fig, ax = plt.subplots(figsize=(10, 6))

    order = np.argsort(-y)
    W_sorted = weights[order]

    sns.heatmap(W_sorted[:500], cmap='viridis',
                cbar_kws={'label': 'Attention weight'},
                ax=ax,
                xticklabels=[f"t{t}" for t in range(weights.shape[1])])
    ax.set_xlabel('Timestep (30-min bins)')
    ax.set_ylabel('Patients (progressors first)')
    ax.set_title('Attention Weights per Patient (first 500)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'phase7b_attention_heatmap.png', dpi=150)
    plt.close()
    logger.info("  Figure: phase7b_attention_heatmap.png")


# ============================================================
# MASK ABLATION
# ============================================================

def mask_ablation(model, X_test, y_test, device, logger):
    logger.info("\n" + "=" * 60)
    logger.info("MASK ABLATION")
    logger.info("=" * 60)

    n_features = P['n_features']

    probs_full = predict_probs(model, X_test, device)
    metrics_full = compute_metrics(y_test, probs_full)

    X_no_masks = X_test.copy()
    X_no_masks[:, :, n_features:] = 0.0
    probs_no_masks = predict_probs(model, X_no_masks, device)
    metrics_no_masks = compute_metrics(y_test, probs_no_masks)

    rows = [
        {'condition': 'full', **metrics_full},
        {'condition': 'masks_zeroed', **metrics_no_masks},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / 'phase7b_mask_ablation.csv', index=False)
    logger.info(f"  Saved: phase7b_mask_ablation.csv")

    logger.info("\n  Mask ablation results:")
    for _, r in df.iterrows():
        logger.info(f"    {r['condition']:15s}: "
                    f"AUPRC={r['auprc']:.4f}  AUROC={r['auroc']:.4f}  "
                    f"Recall={r['recall']:.4f}")

    logger.info(f"\n  ΔAUPRC (full - masks_zeroed) = "
                f"{metrics_full['auprc'] - metrics_no_masks['auprc']:+.4f}")
    logger.info(f"  ΔAUROC (full - masks_zeroed) = "
                f"{metrics_full['auroc'] - metrics_no_masks['auroc']:+.4f}")

    return df


def make_mask_ablation_figure(ablation_df, logger):
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = ['auprc', 'auroc', 'recall', 'precision', 'f1']
    x = np.arange(len(metrics))
    width = 0.35

    full_vals = [ablation_df[ablation_df['condition'] == 'full'][m].iloc[0]
                 for m in metrics]
    no_vals = [ablation_df[ablation_df['condition'] == 'masks_zeroed'][m].iloc[0]
               for m in metrics]

    ax.bar(x - width/2, full_vals, width, label='Full (values + masks)',
           color='steelblue')
    ax.bar(x + width/2, no_vals, width, label='Masks zeroed',
           color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Value')
    ax.set_title('Mask Ablation: Effect of Zeroing Mask Channels')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    for i, (f, n) in enumerate(zip(full_vals, no_vals)):
        ax.text(i - width/2, f + 0.005, f'{f:.3f}', ha='center', fontsize=8)
        ax.text(i + width/2, n + 0.005, f'{n:.3f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'phase7b_mask_ablation.png', dpi=150)
    plt.close()
    logger.info("  Figure: phase7b_mask_ablation.png")


# ============================================================
# THRESHOLD TUNING
# ============================================================

def tune_threshold(y_test, y_prob, logger):
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
            'threshold': t, 'recall': rec, 'precision': prec,
            'specificity': spec, 'npv': npv, 'f1': f1,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        })

    df = pd.DataFrame(rows)

    above = df[df['recall'] >= TARGET_RECALL]
    if not above.empty:
        selected = above.loc[above['f1'].idxmax()]
        logger.info(f"Found {len(above)} thresholds achieving recall ≥ "
                    f"{TARGET_RECALL}; selecting by max F1")
    else:
        selected = df.iloc[df['recall'].idxmax()]
        logger.warning(f"Target recall not achievable; using max recall "
                       f"{selected['recall']:.4f}")

    logger.info(f"Selected threshold: {selected['threshold']:.3f}")
    logger.info(f"  Recall:     {selected['recall']:.4f}")
    logger.info(f"  Precision:  {selected['precision']:.4f}")
    logger.info(f"  Specificity:{selected['specificity']:.4f}")
    logger.info(f"  NPV:        {selected['npv']:.4f}")
    logger.info(f"  F1:         {selected['f1']:.4f}")

    df.to_csv(TABLE_DIR / 'phase7b_threshold_tuning.csv', index=False)
    logger.info(f"Saved: phase7b_threshold_tuning.csv")
    return df, selected


def make_threshold_figure(threshold_df, selected, logger):
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
    ax.axhline(TARGET_RECALL, color='gray', linestyle=':', alpha=0.5,
               label=f'Target recall={TARGET_RECALL}')
    ax.set_xlabel('Decision threshold')
    ax.set_ylabel('Metric value')
    ax.set_title('CNN-LSTM Threshold Tuning')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'phase7b_threshold_curve.png', dpi=150)
    plt.close()
    logger.info("  Figure: phase7b_threshold_curve.png")


# ============================================================
# BOOTSTRAP CIs
# ============================================================

def bootstrap_metrics(y_test, y_prob, threshold, logger):
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
        metrics['recall'].append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        metrics['precision'].append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        metrics['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        metrics['npv'].append(tn / (tn + fn) if (tn + fn) > 0 else 0.0)
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
        ci[k] = {'mean': float(vals.mean()),
                 'ci_lo': float(np.percentile(vals, lo)),
                 'ci_hi': float(np.percentile(vals, hi))}
        logger.info(f"  {k:12s}: {ci[k]['mean']:.4f} "
                    f"[{ci[k]['ci_lo']:.4f}, {ci[k]['ci_hi']:.4f}]")
    return ci


# ============================================================
# DCA
# ============================================================

def compute_dca(y_test, y_prob, logger):
    logger.info("\n" + "=" * 60)
    logger.info("DECISION CURVE ANALYSIS")
    logger.info("=" * 60)

    n = len(y_test)
    rows = []
    for t in DCA_THRESHOLDS:
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_pred == 1) & (y_test == 1)).sum())
        fp = int(((y_pred == 1) & (y_test == 0)).sum())
        nb = tp / n - fp / n * (t / (1 - t)) if t < 1.0 else 0.0
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
    best = df.loc[df['net_benefit_model'].idxmax()]
    logger.info(f"Max net benefit: {best['net_benefit_model']:.4f} "
                f"at threshold={best['threshold']:.2f}")
    df.to_csv(TABLE_DIR / 'phase7b_dca.csv', index=False)
    logger.info(f"Saved: phase7b_dca.csv")
    return df


def make_dca_figure(dca_df, logger):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(dca_df['threshold'], dca_df['net_benefit_model'],
            label='CNN-LSTM', color='crimson', linewidth=2)
    ax.plot(dca_df['threshold'], dca_df['net_benefit_treat_all'],
            label='Treat All', color='gray', linestyle='--')
    ax.plot(dca_df['threshold'], dca_df['net_benefit_treat_none'],
            label='Treat None', color='black', linestyle=':')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Threshold probability')
    ax.set_ylabel('Net benefit')
    ax.set_title('CNN-LSTM Decision Curve Analysis')
    ax.set_ylim(bottom=min(-0.05, dca_df['net_benefit_model'].min() - 0.05))
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'phase7b_dca.png', dpi=150)
    plt.close()
    logger.info("  Figure: phase7b_dca.png")


# ============================================================
# CALIBRATION
# ============================================================

def compute_calibration(y_test, y_prob, logger):
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
    df.to_csv(TABLE_DIR / 'phase7b_calibration_results.csv', index=False)
    logger.info(f"Saved: phase7b_calibration_results.csv")
    brier = brier_score_loss(y_test, y_prob)
    logger.info(f"Brier score: {brier:.4f}")
    return df, brier


def make_calibration_figure(cal_df, brier, logger):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.plot(cal_df['mean_predicted'], cal_df['observed_rate'],
            marker='o', linewidth=2, color='crimson', label='CNN-LSTM')
    for _, row in cal_df.iterrows():
        ax.annotate(f"n={int(row['n'])}",
                    (row['mean_predicted'], row['observed_rate']),
                    fontsize=7, alpha=0.7,
                    textcoords='offset points', xytext=(5, 5))
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed positive rate')
    ax.set_title(f'CNN-LSTM Calibration (Brier = {brier:.4f})')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'phase7b_calibration.png', dpi=150)
    plt.close()
    logger.info("  Figure: phase7b_calibration.png")


# ============================================================
# CONFUSION MATRIX
# ============================================================

def make_confusion_matrix_figure(selected, logger):
    tn = int(selected['tn']); fp = int(selected['fp'])
    fn = int(selected['fn']); tp = int(selected['tp'])
    cm = np.array([[tn, fp], [fn, tp]], dtype=int)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'],
                cbar_kws={'label': 'Count'})
    ax.set_title(f"CNN-LSTM Confusion Matrix at threshold="
                 f"{selected['threshold']:.2f}\n"
                 f"Recall={selected['recall']:.3f}, "
                 f"Precision={selected['precision']:.3f}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'phase7b_confusion_matrix.png', dpi=150)
    plt.close()
    logger.info("  Figure: phase7b_confusion_matrix.png")


# ============================================================
# PERMUTATION FIGURE
# ============================================================

def make_permutation_figure(perm_df, logger):
    top = perm_df.head(TOP_N_BAR).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['coral' if m else 'steelblue' for m in top['is_mask']]
    ax.barh(range(len(top)), top['auprc_drop_mean'],
            xerr=top['auprc_drop_std'], color=colors, alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['label'], fontsize=9)
    ax.set_xlabel('AUPRC drop when permuted')
    ax.set_title(f'Top {TOP_N_BAR} Channels by Permutation Importance')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', label='Value channel'),
                       Patch(facecolor='coral', label='Mask channel')]
    ax.legend(handles=legend_elements, loc='lower right')
    ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'phase7b_permutation_bar.png', dpi=150)
    plt.close()
    logger.info("  Figure: phase7b_permutation_bar.png")


# ============================================================
# REPORT
# ============================================================

def save_report(perm_df, attn_df, ablation_df, selected, boot_ci,
                dca_df, cal_df, brier, test_metrics_05, logger):
    lines = []
    lines.append("=" * 70)
    lines.append("PHASE 7b: CNN-LSTM INTERPRETABILITY REPORT")
    lines.append("=" * 70)
    lines.append(f"Feature window: 0-{FEATURE_WINDOW_HOURS}h")
    lines.append(f"Outcome: ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD} at "
                 f"{PREDICTION_START_HOUR}-{PREDICTION_END_HOUR}h")
    lines.append("")

    lines.append("TOP 15 CHANNELS BY PERMUTATION IMPORTANCE (AUPRC drop):")
    lines.append("-" * 70)
    for _, row in perm_df.head(15).iterrows():
        lines.append(f"  {int(row['rank']):2d}. {row['label']:35s} "
                     f"{row['auprc_drop_mean']:+.4f}")

    lines.append("")
    lines.append("ATTENTION BY TIMESTEP:")
    lines.append("-" * 70)
    for _, row in attn_df.iterrows():
        lines.append(f"  t={int(row['timestep']):2d} "
                     f"({int(row['bin_start_min']):3d}-"
                     f"{int(row['bin_end_min']):3d} min): "
                     f"{row['mean_weight']:.4f}")

    lines.append("")
    lines.append("MASK ABLATION:")
    lines.append("-" * 70)
    for _, row in ablation_df.iterrows():
        lines.append(f"  {row['condition']:15s}: AUPRC={row['auprc']:.4f} "
                     f"AUROC={row['auroc']:.4f} Recall={row['recall']:.4f}")

    lines.append("")
    lines.append("THRESHOLD TUNING:")
    lines.append("-" * 70)
    lines.append(f"  Target recall: {TARGET_RECALL}")
    lines.append(f"  Selected threshold: {selected['threshold']:.3f}")
    lines.append(f"  Recall:     {selected['recall']:.4f}")
    lines.append(f"  Precision:  {selected['precision']:.4f}")
    lines.append(f"  Specificity:{selected['specificity']:.4f}")
    lines.append(f"  NPV:        {selected['npv']:.4f}")
    lines.append(f"  F1:         {selected['f1']:.4f}")

    lines.append("")
    lines.append(f"BOOTSTRAP 95% CIs ({N_BOOTSTRAP} resamples):")
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
    lines.append("DCA:")
    lines.append("-" * 70)
    best_nb = dca_df.loc[dca_df['net_benefit_model'].idxmax()]
    lines.append(f"  Max net benefit: {best_nb['net_benefit_model']:.4f} "
                 f"at threshold={best_nb['threshold']:.2f}")

    lines.append("")
    lines.append("CALIBRATION:")
    lines.append("-" * 70)
    lines.append(f"  Brier score: {brier:.4f}")

    with open(TABLE_DIR / 'phase7b_report.txt', 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"Saved: phase7b_report.txt")


# ============================================================
# MAIN
# ============================================================

def main():
    logger = setup_logging()
    print_header()

    t0 = time.time()

    set_seeds(P['seed'], deterministic=P.get('deterministic', True))
    device = get_device()
    logger.info(f"  Device: {device}")
    if device.type == 'mps' and not check_mps_bce():
        logger.warning("MPS fallback → CPU")
        device = torch.device('cpu')

    train_data = np.load(SEQUENCE_DIR / 'sequences_train.npz')
    test_data = np.load(SEQUENCE_DIR / 'sequences_test.npz')
    X_test = test_data['X']
    y_test = test_data['y'].astype(int)
    stay_ids_test = test_data['stay_ids']

    logger.info(f"  X_test: {X_test.shape}")
    logger.info(f"  y_test: {int(y_test.sum())} pos / {len(y_test)}")

    model = load_final_dl_model(device, logger)

    # 1. Predictions + baseline metrics
    y_prob = predict_probs(model, X_test, device)
    test_metrics_05 = compute_metrics(y_test, y_prob, threshold=0.5)

    pd.DataFrame([test_metrics_05]).to_csv(
        TABLE_DIR / 'phase7b_test_metrics_at_0.5.csv', index=False)
    logger.info(f"  Saved: phase7b_test_metrics_at_0.5.csv")

    logger.info("\n" + "=" * 60)
    logger.info("TEST METRICS AT THRESHOLD = 0.50")
    logger.info("=" * 60)
    for k, v in test_metrics_05.items():
        logger.info(f"  {k:12s}: {v:.4f}")

    # 2. Permutation importance
    perm_df, base_auprc, base_auroc = permutation_importance(
        model, X_test, y_test, device, logger,
    )
    make_permutation_figure(perm_df, logger)

    # 3. Attention weights
    attn_df, attn_weights = attention_summary(model, X_test, y_test,
                                                device, logger)
    make_attention_heatmap(attn_weights, y_test, logger)

    # 4. Mask ablation
    ablation_df = mask_ablation(model, X_test, y_test, device, logger)
    make_mask_ablation_figure(ablation_df, logger)

    # 5. Threshold tuning
    threshold_df, selected = tune_threshold(y_test, y_prob, logger)
    make_threshold_figure(threshold_df, selected, logger)

    pd.DataFrame([{
        'threshold': float(selected['threshold']),
        'tp': int(selected['tp']), 'fp': int(selected['fp']),
        'fn': int(selected['fn']), 'tn': int(selected['tn']),
        'recall': float(selected['recall']),
        'precision': float(selected['precision']),
        'specificity': float(selected['specificity']),
        'npv': float(selected['npv']),
        'f1': float(selected['f1']),
    }]).to_csv(TABLE_DIR / 'phase7b_confusion_matrix.csv', index=False)
    logger.info("Saved: phase7b_confusion_matrix.csv")
    make_confusion_matrix_figure(selected, logger)

    # 6. Bootstrap CIs
    boot_ci = bootstrap_metrics(y_test, y_prob, selected['threshold'], logger)
    pd.DataFrame([{'metric': k, 'mean': v['mean'],
                   'ci_lo': v['ci_lo'], 'ci_hi': v['ci_hi']}
                  for k, v in boot_ci.items()]).to_csv(
        TABLE_DIR / 'phase7b_bootstrap_cis.csv', index=False)
    logger.info("Saved: phase7b_bootstrap_cis.csv")

    # 7. DCA
    dca_df = compute_dca(y_test, y_prob, logger)
    make_dca_figure(dca_df, logger)

    # 8. Calibration
    cal_df, brier = compute_calibration(y_test, y_prob, logger)
    make_calibration_figure(cal_df, brier, logger)

    # 9. Predictions CSV
    pred_df = pd.DataFrame({
        'stay_id': stay_ids_test,
        'y_true': y_test,
        'y_prob_cnn_lstm': y_prob,
        'y_pred_at_selected': (y_prob >= selected['threshold']).astype(int),
        'y_pred_at_0.5': (y_prob >= 0.5).astype(int),
    })
    pred_df.to_csv(TABLE_DIR / 'phase7b_test_predictions.csv', index=False)
    logger.info("Saved: phase7b_test_predictions.csv")

    # 10. Report
    save_report(perm_df, attn_df, ablation_df, selected, boot_ci,
                dca_df, cal_df, brier, test_metrics_05, logger)

    # Summary
    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(" PHASE 7b COMPLETED")
    print("=" * 80)
    print(f" Elapsed: {elapsed/60:.1f} min")
    print("-" * 80)
    print(" TOP 5 CHANNELS (permutation):")
    for _, row in perm_df.head(5).iterrows():
        print(f"   {int(row['rank'])}. {row['label']:35s} "
              f"{row['auprc_drop_mean']:+.4f}")
    print("-" * 80)
    print(" SELECTED THRESHOLD:")
    print(f"   threshold:  {selected['threshold']:.3f}")
    print(f"   Recall:     {selected['recall']:.4f} "
          f"[{boot_ci['recall']['ci_lo']:.4f}, "
          f"{boot_ci['recall']['ci_hi']:.4f}]")
    print(f"   Precision:  {selected['precision']:.4f} "
          f"[{boot_ci['precision']['ci_lo']:.4f}, "
          f"{boot_ci['precision']['ci_hi']:.4f}]")
    print(f"   F1:         {selected['f1']:.4f}")
    print(f"   Brier:      {brier:.4f}")
    print("-" * 80)
    print(" MASK ABLATION:")
    for _, row in ablation_df.iterrows():
        print(f"   {row['condition']:15s}: AUPRC={row['auprc']:.4f} "
              f"AUROC={row['auroc']:.4f}")
    print("=" * 80)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
