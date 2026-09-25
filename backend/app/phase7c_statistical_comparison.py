#!/usr/bin/env python3
"""
Phase 7c: Statistical Comparison of Tree vs. DL

Primary comparison:   Tree vs. single final CNN-LSTM
Secondary comparison: Tree vs. 5-fold CNN-LSTM ensemble

Tests:
    1. DeLong's test for AUROC difference (+ 95% Wald CI)
    2. Bootstrap SE-based z-test for AUPRC difference (+ 95% CI)
    3. McNemar's test at matched-sensitivity operating points

INPUTS
------
    outputs/tables/phase7_test_predictions.csv
    outputs/tables/phase6b_test_predictions.csv

OUTPUTS
-------
    outputs/tables/phase7c_auroc_delong.csv
    outputs/tables/phase7c_auprc_bootstrap.csv
    outputs/tables/phase7c_mcnemar.csv
    outputs/tables/phase7c_operating_points.csv
    outputs/tables/phase7c_ensemble_comparison.csv
    outputs/tables/phase7c_report.txt
    outputs/figures/phase7c_roc_comparison.png
    outputs/figures/phase7c_pr_comparison.png

USAGE
-----
    python3 phase7c_statistical_comparison.py             # full analysis
    python3 phase7c_statistical_comparison.py --selftest  # sanity check
"""

import sys
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from config import TABLE_DIR, FIG_DIR, LOGS_DIR, RANDOM_STATE


# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"phase7c_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    print(" PHASE 7c: STATISTICAL COMPARISON (TREE vs. DL)")
    print("=" * 80)
    print(f" Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    print(" Primary:   Tree vs. single CNN-LSTM")
    print(" Secondary: Tree vs. 5-fold CNN-LSTM ensemble")
    print(" Tests:     DeLong (AUROC) | Bootstrap z-test (AUPRC) | McNemar")
    print("=" * 80)
    print()


def fmt_p(p):
    """Human-readable p-value formatting."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


# ============================================================
# DELONG'S TEST
# ============================================================

def compute_midrank(x):
    """Midrank computation for DeLong (Sun & Xu 2014)."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fast_delong(predictions_sorted_transposed, label_1_count):
    """
    Fast DeLong. predictions_sorted_transposed shape (k, n), positives first.
    Returns (aucs, covariance) with covariance shape (k, k).
    """
    m = int(label_1_count)
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]

    if m < 2 or n < 2:
        raise ValueError(f"DeLong requires m>=2 and n>=2 (got m={m}, n={n})")

    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive[r, :])
        ty[r, :] = compute_midrank(negative[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    if sx.shape != (k, k) or sy.shape != (k, k):
        raise ValueError(f"Unexpected covariance shape: {sx.shape}, {sy.shape}")

    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(y_true, prob_a, prob_b, logger=None, tol=1e-4):
    """
    Two-sided DeLong test with 95% Wald CI on ΔAUROC.
    Returns dict: auroc_a, auroc_b, delta, ci_lo, ci_hi, z, p, var.
    On degenerate input (perfect separation), z/p/ci/var are NaN.
    """
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true, kind='stable')
    label_1_count = int(y_true.sum())

    preds = np.vstack([prob_a[order], prob_b[order]])

    fail = dict(
        auroc_a=float(roc_auc_score(y_true, prob_a)),
        auroc_b=float(roc_auc_score(y_true, prob_b)),
        delta=np.nan, ci_lo=np.nan, ci_hi=np.nan,
        z=np.nan, p=np.nan, var=np.nan,
    )

    try:
        aucs, cov = fast_delong(preds, label_1_count)
    except Exception as e:
        if logger:
            logger.error(f"DeLong failed: {e}")
        return fail

    if cov.shape != (2, 2):
        if logger:
            logger.error(f"DeLong covariance shape {cov.shape} != (2, 2)")
        return fail

    # Sanity check vs. sklearn
    auc_a_check = float(roc_auc_score(y_true, prob_a))
    auc_b_check = float(roc_auc_score(y_true, prob_b))
    if abs(aucs[0] - auc_a_check) > tol or abs(aucs[1] - auc_b_check) > tol:
        if logger:
            logger.error(
                f"DeLong/sklearn AUROC mismatch "
                f"(DeLong={aucs[0]:.6f}/{aucs[1]:.6f}, "
                f"sklearn={auc_a_check:.6f}/{auc_b_check:.6f})"
            )
        return fail

    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var <= 1e-12:
        if logger:
            logger.warning(
                f"Non-positive DeLong variance ({var:.2e}). "
                f"This typically means both models perfectly separate the "
                f"classes on this sample (AUROC = 1.0 for both). "
                f"DeLong test is undefined in that case."
            )
        return dict(
            auroc_a=float(aucs[0]), auroc_b=float(aucs[1]),
            delta=float(aucs[0] - aucs[1]),
            ci_lo=np.nan, ci_hi=np.nan, z=np.nan, p=np.nan, var=var,
        )

    delta = float(aucs[0] - aucs[1])
    se = float(np.sqrt(var))
    z = delta / se
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    ci_lo = delta - 1.96 * se
    ci_hi = delta + 1.96 * se

    return dict(
        auroc_a=float(aucs[0]), auroc_b=float(aucs[1]),
        delta=delta, ci_lo=float(ci_lo), ci_hi=float(ci_hi),
        z=float(z), p=p, var=var,
    )


# ============================================================
# BOOTSTRAP AUPRC
# ============================================================

def bootstrap_auprc_diff(y_true, prob_a, prob_b,
                          n_boot=2000, seed=RANDOM_STATE, logger=None):
    """
    Returns dict: delta, ci_lo, ci_hi, se, p, n_used, n_skipped.
    p is a normal-approximation two-sided p from bootstrap SE.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []
    n_skipped = 0

    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            n_skipped += 1
            continue
        diffs.append(
            average_precision_score(yb, prob_a[idx]) -
            average_precision_score(yb, prob_b[idx])
        )

    point = float(average_precision_score(y_true, prob_a) -
                  average_precision_score(y_true, prob_b))

    if len(diffs) == 0:
        if logger:
            logger.error("Bootstrap produced zero valid resamples")
        return dict(delta=point, ci_lo=np.nan, ci_hi=np.nan, se=np.nan,
                    p=np.nan, n_used=0, n_skipped=n_skipped)

    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    se = float(diffs.std(ddof=1))
    p = float(2 * (1 - stats.norm.cdf(abs(point / se)))) if se > 0 else 1.0

    if logger and n_skipped > 0:
        pct = 100 * n_skipped / n_boot
        logger.info(f"    Bootstrap: {n_skipped}/{n_boot} skipped ({pct:.1f}%)")

    return dict(
        delta=point, ci_lo=float(lo), ci_hi=float(hi),
        se=se, p=p, n_used=int(len(diffs)), n_skipped=n_skipped,
    )


# ============================================================
# MCNEMAR
# ============================================================

def mcnemar_test(y_true, pred_a, pred_b):
    """Exact binomial two-sided McNemar. Returns (b, c, chi2, p)."""
    y_true = np.asarray(y_true).astype(int)
    pred_a = np.asarray(pred_a).astype(int)
    pred_b = np.asarray(pred_b).astype(int)

    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())

    if b + c == 0:
        return b, c, 0.0, 1.0

    try:
        p = float(stats.binomtest(b, b + c, 0.5,
                                   alternative='two-sided').pvalue)
    except AttributeError:
        p = float(stats.binom_test(b, b + c, 0.5))

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, float(chi2), p


# ============================================================
# OPERATING POINTS
# ============================================================

def threshold_at_sensitivity(y_true, y_prob, target_sens):
    """Threshold whose empirical sensitivity is closest to target."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    idx = int(np.argmin(np.abs(tpr - target_sens)))
    return float(thr[idx])


def operating_point(y_true, y_prob, threshold):
    """Sens/spec/prec/confusion at a threshold."""
    y_prob = np.asarray(y_prob, dtype=float)
    if np.isnan(y_prob).any():
        raise ValueError("NaN in predicted probabilities")

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        'threshold': float(threshold),
        'sensitivity': float(sens), 'specificity': float(spec),
        'precision': float(prec),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }


def load_tree_threshold(logger):
    """Load tree threshold from Phase 7 output, else default 0.40."""
    for name in ('phase7_confusion_matrix.csv',
                 'phase7_threshold_selection.csv'):
        path = TABLE_DIR / name
        if path.exists():
            df = pd.read_csv(path)
            if 'threshold' in df.columns:
                t = float(df['threshold'].iloc[0])
                logger.info(f"  Loaded tree threshold from {name}: {t}")
                return t
    logger.warning("  Using default tree threshold = 0.40")
    return 0.40


# ============================================================
# COMPARISON HELPER (single or ensemble)
# ============================================================

def run_full_comparison(y_true, prob_tree, prob_dl, dl_label,
                         tree_threshold, logger):
    """Run DeLong + bootstrap AUPRC + McNemar for one DL variant."""
    logger.info("\n" + "=" * 60)
    logger.info(f"COMPARISON: Tree vs. {dl_label}")
    logger.info("=" * 60)

    # ---- DeLong ----
    d = delong_test(y_true, prob_tree, prob_dl, logger)
    logger.info("  DeLong (AUROC):")
    logger.info(f"    Tree AUROC:  {d['auroc_a']:.4f}")
    logger.info(f"    {dl_label} AUROC: {d['auroc_b']:.4f}")
    logger.info(f"    ΔAUROC:      {d['delta']:+.4f}  "
                f"(95% CI [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}])")
    logger.info(f"    Z = {d['z']:.4f},  p = {fmt_p(d['p'])}")

    # ---- Bootstrap AUPRC ----
    ap_t = float(average_precision_score(y_true, prob_tree))
    ap_d = float(average_precision_score(y_true, prob_dl))
    b_ap = bootstrap_auprc_diff(y_true, prob_tree, prob_dl,
                                 n_boot=2000, logger=logger)
    logger.info("  Bootstrap (AUPRC):")
    logger.info(f"    Tree AUPRC:  {ap_t:.4f}")
    logger.info(f"    {dl_label} AUPRC: {ap_d:.4f}")
    logger.info(f"    ΔAUPRC:      {b_ap['delta']:+.4f}  "
                f"(95% CI [{b_ap['ci_lo']:+.4f}, {b_ap['ci_hi']:+.4f}])")
    logger.info(f"    SE = {b_ap['se']:.6f},  p = {fmt_p(b_ap['p'])}")

    # ---- McNemar at matched sensitivity ----
    tree_op = operating_point(y_true, prob_tree, tree_threshold)
    target_sens = tree_op['sensitivity']
    dl_threshold = threshold_at_sensitivity(y_true, prob_dl, target_sens)
    dl_op = operating_point(y_true, prob_dl, dl_threshold)

    sens_gap = abs(tree_op['sensitivity'] - dl_op['sensitivity'])
    logger.info("  McNemar (matched sensitivity):")
    logger.info(f"    Tree t={tree_op['threshold']:.4f}: "
                f"sens={tree_op['sensitivity']:.4f}, "
                f"spec={tree_op['specificity']:.4f}, "
                f"prec={tree_op['precision']:.4f}")
    logger.info(f"    {dl_label} t={dl_op['threshold']:.4f}: "
                f"sens={dl_op['sensitivity']:.4f}, "
                f"spec={dl_op['specificity']:.4f}, "
                f"prec={dl_op['precision']:.4f}")
    if sens_gap > 0.01:
        logger.warning(f"    ⚠️  Sensitivity gap = {sens_gap:.4f}")

    pred_tree = (prob_tree >= tree_op['threshold']).astype(int)
    pred_dl = (prob_dl >= dl_op['threshold']).astype(int)
    b, c, chi2, p_mc = mcnemar_test(y_true, pred_tree, pred_dl)
    logger.info(f"    b (tree only correct) = {b}")
    logger.info(f"    c ({dl_label} only)   = {c}")
    logger.info(f"    chi2 = {chi2:.4f},  p = {fmt_p(p_mc)}")

    return {
        'dl_label': dl_label,
        'tree_op': tree_op, 'dl_op': dl_op,
        'mcnemar_b': b, 'mcnemar_c': c, 'mcnemar_p': p_mc,
        'delta_auroc': d['delta'],
        'delta_auroc_ci_lo': d['ci_lo'], 'delta_auroc_ci_hi': d['ci_hi'],
        'auroc_a': d['auroc_a'], 'auroc_b': d['auroc_b'],
        'auroc_z': d['z'], 'auroc_p': d['p'], 'auroc_var': d['var'],
        'auprc_a': ap_t, 'auprc_b': ap_d,
        'delta_auprc': b_ap['delta'],
        'delta_auprc_ci_lo': b_ap['ci_lo'],
        'delta_auprc_ci_hi': b_ap['ci_hi'],
        'auprc_se': b_ap['se'], 'auprc_p': b_ap['p'],
        'auprc_n_used': b_ap['n_used'],
        'auprc_n_skipped': b_ap['n_skipped'],
    }


# ============================================================
# FIGURES
# ============================================================

def plot_roc_comparison(y_true, prob_tree, prob_dl,
                         auc_t, auc_d, tree_op, dl_op,
                         dl_label, logger):
    fpr_t, tpr_t, _ = roc_curve(y_true, prob_tree)
    fpr_d, tpr_d, _ = roc_curve(y_true, prob_dl)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr_t, tpr_t, label=f'Tree (AUC={auc_t:.4f})',
            color='steelblue', linewidth=2)
    ax.plot(fpr_d, tpr_d, label=f'{dl_label} (AUC={auc_d:.4f})',
            color='crimson', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)

    ax.scatter([1 - tree_op['specificity']], [tree_op['sensitivity']],
               color='steelblue', s=120, marker='o', zorder=5,
               edgecolors='black',
               label=f"Tree op (t={tree_op['threshold']:.2f})")
    ax.scatter([1 - dl_op['specificity']], [dl_op['sensitivity']],
               color='crimson', s=120, marker='o', zorder=5,
               edgecolors='black',
               label=f"{dl_label} op (t={dl_op['threshold']:.2f})")

    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(f'ROC: Tree vs. {dl_label}')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'phase7c_roc_comparison.png', dpi=150)
    plt.close()
    logger.info("  Figure: phase7c_roc_comparison.png")


def plot_pr_comparison(y_true, prob_tree, prob_dl,
                        ap_t, ap_d, tree_op, dl_op,
                        dl_label, logger):
    prec_t, rec_t, _ = precision_recall_curve(y_true, prob_tree)
    prec_d, rec_d, _ = precision_recall_curve(y_true, prob_dl)
    prevalence = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(rec_t, prec_t, label=f'Tree (AP={ap_t:.4f})',
            color='steelblue', linewidth=2)
    ax.plot(rec_d, prec_d, label=f'{dl_label} (AP={ap_d:.4f})',
            color='crimson', linewidth=2)
    ax.axhline(prevalence, color='gray', linestyle=':', alpha=0.6,
               label=f'Baseline (prev={prevalence:.3f})')

    # Real operating points
    ax.scatter([tree_op['sensitivity']], [tree_op['precision']],
               color='steelblue', s=120, marker='o', zorder=5,
               edgecolors='black',
               label=f"Tree op (t={tree_op['threshold']:.2f})")
    ax.scatter([dl_op['sensitivity']], [dl_op['precision']],
               color='crimson', s=120, marker='o', zorder=5,
               edgecolors='black',
               label=f"{dl_label} op (t={dl_op['threshold']:.2f})")

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'PR: Tree vs. {dl_label}')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'phase7c_pr_comparison.png', dpi=150)
    plt.close()
    logger.info("  Figure: phase7c_pr_comparison.png")


# ============================================================
# INTERPRETATION
# ============================================================

def _sig(p):
    return (p is not None) and (not np.isnan(p)) and (p < 0.05)


def build_interpretation(res):
    """Direction-aware interpretation for one comparison result."""
    lines = []
    auroc_tree_better = res['delta_auroc'] > 0
    auprc_tree_better = res['delta_auprc'] > 0
    mc_tree_better = res['mcnemar_b'] > res['mcnemar_c']

    sig_auroc = _sig(res['auroc_p'])
    sig_auprc = _sig(res['auprc_p'])
    sig_mc = _sig(res['mcnemar_p'])

    parts = []
    if sig_auroc:
        parts.append(f"DeLong favors {'tree' if auroc_tree_better else 'DL'}")
    if sig_auprc:
        parts.append(f"AUPRC favors {'tree' if auprc_tree_better else 'DL'}")
    if sig_mc:
        parts.append(f"McNemar favors {'tree' if mc_tree_better else 'DL'}")

    if (sig_auroc and sig_auprc and sig_mc and
            auroc_tree_better and auprc_tree_better and mc_tree_better):
        lines.append("→ Tree significantly better on all three tests.")
    elif parts:
        lines.append("→ " + "; ".join(parts))
    else:
        lines.append("→ No statistically significant differences.")
    return lines


# ============================================================
# SELF-TEST
# ============================================================

def _selftest():
    """Verify DeLong against a non-degenerate synthetic case."""
    rng = np.random.RandomState(0)
    n = 500
    y = rng.binomial(1, 0.3, n).astype(int)

    # Overlapping class-conditional distributions → AUROC ~0.75
    a = rng.normal(loc=y * 0.8, scale=1.0, size=n)   # better model
    b = rng.normal(loc=y * 0.5, scale=1.0, size=n)   # weaker model

    d = delong_test(y, a, b, logger=None, tol=1e-3)
    auc_a = float(roc_auc_score(y, a))
    auc_b = float(roc_auc_score(y, b))

    print("Self-test DeLong:")
    print(f"  auc_a = {d['auroc_a']:.4f}  (sklearn {auc_a:.4f})")
    print(f"  auc_b = {d['auroc_b']:.4f}  (sklearn {auc_b:.4f})")
    print(f"  delta = {d['delta']:+.4f}  z = {d['z']:.3f}  p = {fmt_p(d['p'])}")
    print(f"  CI    = [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]")

    assert abs(d['auroc_a'] - auc_a) < 1e-6, "AUROC mismatch vs sklearn"
    assert abs(d['auroc_b'] - auc_b) < 1e-6, "AUROC mismatch vs sklearn"
    assert not np.isnan(d['p']), "p-value is NaN (degenerate data?)"
    assert d['delta'] > 0, "Model A should beat Model B"

    # Independent bootstrap cross-check
    boot_diffs = []
    for _ in range(500):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boot_diffs.append(
            roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx])
        )
    boot_diffs = np.asarray(boot_diffs)
    boot_se = float(boot_diffs.std(ddof=1))
    boot_z = (auc_a - auc_b) / boot_se
    boot_p = 2 * (1 - stats.norm.cdf(abs(boot_z)))

    print(f"  Bootstrap cross-check: z={boot_z:.3f}  p={fmt_p(boot_p)}")

    # DeLong and bootstrap should agree within ~20% relative
    if not np.isnan(d['z']):
        rel_gap = abs(d['z'] - boot_z) / max(abs(boot_z), 1e-9)
        print(f"  Relative Z gap: {rel_gap:.2%}")
        assert rel_gap < 0.30, "DeLong and bootstrap Z disagree"

    print("  OK")


# ============================================================
# MAIN
# ============================================================

def main():
    logger = setup_logging()
    print_header()

    tree_path = TABLE_DIR / 'phase7_test_predictions.csv'
    dl_path = TABLE_DIR / 'phase6b_test_predictions.csv'

    if not tree_path.exists():
        logger.error(f"Missing {tree_path}. Run Phase 7 first.")
        return 1
    if not dl_path.exists():
        logger.error(f"Missing {dl_path}. Run Phase 6b first.")
        return 1

    tree = pd.read_csv(tree_path)
    dl = pd.read_csv(dl_path)

    logger.info(f"Tree predictions: {tree.shape}")
    logger.info(f"DL predictions:   {dl.shape}")
    logger.info(f"Tree columns: {list(tree.columns)}")
    logger.info(f"DL columns:   {list(dl.columns)}")

    assert tree['stay_id'].is_unique, "Duplicate stay_id in tree"
    assert dl['stay_id'].is_unique, "Duplicate stay_id in DL"

    if 'y_prob_cnn_lstm_single' not in dl.columns:
        logger.error("Missing required DL column: y_prob_cnn_lstm_single")
        logger.error(f"Available DL columns: {list(dl.columns)}")
        return 1
    has_ensemble = 'y_prob_cnn_lstm_ensemble' in dl.columns
    logger.info(f"Ensemble column present: {has_ensemble}")

    # --- Normalize column names BEFORE merge (avoid pandas suffix surprises) ---
    # Tree file uses 'y_prob' and 'y_true' (no suffix). Rename to *_tree.
    if 'y_prob' in tree.columns and 'y_prob_tree' not in tree.columns:
        tree = tree.rename(columns={'y_prob': 'y_prob_tree'})
    if 'y_true' in tree.columns and 'y_true_tree' not in tree.columns:
        tree = tree.rename(columns={'y_true': 'y_true_tree'})

    # DL file uses 'y_true' (no suffix). Rename to y_true_dl.
    if 'y_true' in dl.columns and 'y_true_dl' not in dl.columns:
        dl = dl.rename(columns={'y_true': 'y_true_dl'})

    # --- Merge (no suffixes needed — columns are now unique) ---
    df = tree.merge(dl, on='stay_id', how='inner')
    logger.info(f"Merged: {len(df)} rows")
    logger.info(f"Merged columns: {list(df.columns)}")

    if len(df) != len(tree) or len(df) != len(dl):
        logger.warning(
            f"Merge changed row count: tree={len(tree)}, "
            f"dl={len(dl)}, merged={len(df)}"
        )

    # --- Verify expected columns exist (loud failure > silent KeyError) ---
    required_merged = ['y_true_tree', 'y_true_dl',
                       'y_prob_tree', 'y_prob_cnn_lstm_single']
    missing = [c for c in required_merged if c not in df.columns]
    if missing:
        logger.error(f"Missing expected columns after merge: {missing}")
        logger.error(f"Available columns: {list(df.columns)}")
        return 1

    y_true = df['y_true_tree'].values.astype(int)
    assert (y_true == df['y_true_dl'].values.astype(int)).all(), \
        "y_true mismatch between branches"

    prob_tree = df['y_prob_tree'].values.astype(float)
    prob_dl_single = df['y_prob_cnn_lstm_single'].values.astype(float)

    assert not np.isnan(prob_tree).any()
    assert not np.isnan(prob_dl_single).any()

    logger.info(f"n = {len(y_true)}, prevalence = {y_true.mean():.4f}")

    tree_threshold = load_tree_threshold(logger)

    # ---------- PRIMARY ----------
    logger.info("\n" + "#" * 70)
    logger.info("# PRIMARY: Tree vs. single CNN-LSTM")
    logger.info("#" * 70)
    single_res = run_full_comparison(
        y_true, prob_tree, prob_dl_single, "CNN-LSTM (single)",
        tree_threshold, logger,
    )

    pd.DataFrame([{
        'test': 'delong',
        'auroc_tree': single_res['auroc_a'],
        'auroc_dl': single_res['auroc_b'],
        'delta': single_res['delta_auroc'],
        'ci_lo': single_res['delta_auroc_ci_lo'],
        'ci_hi': single_res['delta_auroc_ci_hi'],
        'z_stat': single_res['auroc_z'],
        'p_value': single_res['auroc_p'],
        'var': single_res['auroc_var'],
    }]).to_csv(TABLE_DIR / 'phase7c_auroc_delong.csv', index=False)

    pd.DataFrame([{
        'test': 'auprc_bootstrap',
        'auprc_tree': single_res['auprc_a'],
        'auprc_dl': single_res['auprc_b'],
        'delta': single_res['delta_auprc'],
        'ci_lo': single_res['delta_auprc_ci_lo'],
        'ci_hi': single_res['delta_auprc_ci_hi'],
        'se': single_res['auprc_se'],
        'p_value': single_res['auprc_p'],
        'n_used': single_res['auprc_n_used'],
        'n_skipped': single_res['auprc_n_skipped'],
    }]).to_csv(TABLE_DIR / 'phase7c_auprc_bootstrap.csv', index=False)

    pd.DataFrame([{
        'test': 'mcnemar',
        'tree_threshold': single_res['tree_op']['threshold'],
        'dl_threshold': single_res['dl_op']['threshold'],
        'b_tree_only_correct': single_res['mcnemar_b'],
        'c_dl_only_correct': single_res['mcnemar_c'],
        'p_value': single_res['mcnemar_p'],
    }]).to_csv(TABLE_DIR / 'phase7c_mcnemar.csv', index=False)

    pd.DataFrame([
        {'model': 'tree',
         **{k: single_res['tree_op'][k]
            for k in ('threshold', 'sensitivity', 'specificity', 'precision',
                      'tp', 'fp', 'fn', 'tn')}},
        {'model': 'dl_single',
         **{k: single_res['dl_op'][k]
            for k in ('threshold', 'sensitivity', 'specificity', 'precision',
                      'tp', 'fp', 'fn', 'tn')}},
    ]).to_csv(TABLE_DIR / 'phase7c_operating_points.csv', index=False)

    # ---------- SECONDARY ----------
    ensemble_res = None
    if has_ensemble:
        logger.info("\n" + "#" * 70)
        logger.info("# SECONDARY: Tree vs. 5-fold ensemble")
        logger.info("#" * 70)

        prob_dl_ens = df['y_prob_cnn_lstm_ensemble'].values.astype(float)
        assert not np.isnan(prob_dl_ens).any()

        ensemble_res = run_full_comparison(
            y_true, prob_tree, prob_dl_ens, "CNN-LSTM (ensemble)",
            tree_threshold, logger,
        )

        pd.DataFrame([{
            'test': 'ensemble',
            'auroc_tree': ensemble_res['auroc_a'],
            'auroc_ensemble': ensemble_res['auroc_b'],
            'delta_auroc': ensemble_res['delta_auroc'],
            'delta_auroc_ci_lo': ensemble_res['delta_auroc_ci_lo'],
            'delta_auroc_ci_hi': ensemble_res['delta_auroc_ci_hi'],
            'auroc_z': ensemble_res['auroc_z'],
            'auroc_p': ensemble_res['auroc_p'],
            'auprc_tree': ensemble_res['auprc_a'],
            'auprc_ensemble': ensemble_res['auprc_b'],
            'delta_auprc': ensemble_res['delta_auprc'],
            'delta_auprc_ci_lo': ensemble_res['delta_auprc_ci_lo'],
            'delta_auprc_ci_hi': ensemble_res['delta_auprc_ci_hi'],
            'auprc_se': ensemble_res['auprc_se'],
            'auprc_p': ensemble_res['auprc_p'],
            'tree_threshold': ensemble_res['tree_op']['threshold'],
            'dl_threshold': ensemble_res['dl_op']['threshold'],
            'mcnemar_b': ensemble_res['mcnemar_b'],
            'mcnemar_c': ensemble_res['mcnemar_c'],
            'mcnemar_p': ensemble_res['mcnemar_p'],
        }]).to_csv(TABLE_DIR / 'phase7c_ensemble_comparison.csv', index=False)
    else:
        logger.warning("Ensemble column missing; skipping ensemble comparison")

    # ---------- FIGURES ----------
    logger.info("\n" + "=" * 60)
    logger.info("FIGURES")
    logger.info("=" * 60)
    plot_roc_comparison(
        y_true, prob_tree, prob_dl_single,
        single_res['auroc_a'], single_res['auroc_b'],
        single_res['tree_op'], single_res['dl_op'],
        "CNN-LSTM (single)", logger,
    )
    plot_pr_comparison(
        y_true, prob_tree, prob_dl_single,
        single_res['auprc_a'], single_res['auprc_b'],
        single_res['tree_op'], single_res['dl_op'],
        "CNN-LSTM (single)", logger,
    )

    # ---------- REPORT ----------
    report = []
    report.append("=" * 70)
    report.append("PHASE 7c: STATISTICAL COMPARISON")
    report.append("=" * 70)
    report.append(f"n = {len(y_true)}  |  prevalence = {y_true.mean():.4f}")
    report.append("")
    report.append("PRIMARY: Tree vs. single CNN-LSTM")
    report.append(f"  AUROC   tree={single_res['auroc_a']:.4f}  "
                  f"dl={single_res['auroc_b']:.4f}  "
                  f"Δ={single_res['delta_auroc']:+.4f}  "
                  f"[{single_res['delta_auroc_ci_lo']:+.4f}, "
                  f"{single_res['delta_auroc_ci_hi']:+.4f}]  "
                  f"p={fmt_p(single_res['auroc_p'])}")
    report.append(f"  AUPRC   tree={single_res['auprc_a']:.4f}  "
                  f"dl={single_res['auprc_b']:.4f}  "
                  f"Δ={single_res['delta_auprc']:+.4f}  "
                  f"[{single_res['delta_auprc_ci_lo']:+.4f}, "
                  f"{single_res['delta_auprc_ci_hi']:+.4f}]  "
                  f"p={fmt_p(single_res['auprc_p'])}")
    report.append(f"  McNemar b={single_res['mcnemar_b']}, "
                  f"c={single_res['mcnemar_c']}, "
                  f"p={fmt_p(single_res['mcnemar_p'])}")
    report.append("")
    report.append("INTERPRETATION:")
    for line in build_interpretation(single_res):
        report.append(f"  {line}")

    if ensemble_res is not None:
        report.append("")
        report.append("SECONDARY: Tree vs. 5-fold ensemble")
        report.append(f"  AUROC   Δ={ensemble_res['delta_auroc']:+.4f}  "
                      f"p={fmt_p(ensemble_res['auroc_p'])}")
        report.append(f"  AUPRC   Δ={ensemble_res['delta_auprc']:+.4f}  "
                      f"p={fmt_p(ensemble_res['auprc_p'])}")
        report.append(f"  McNemar b={ensemble_res['mcnemar_b']}, "
                      f"c={ensemble_res['mcnemar_c']}, "
                      f"p={fmt_p(ensemble_res['mcnemar_p'])}")

    report_path = TABLE_DIR / 'phase7c_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    logger.info(f"\n  Saved: {report_path}")

    # ---------- SUMMARY ----------
    print("\n" + "=" * 80)
    print(" PHASE 7c COMPLETED")
    print("=" * 80)
    print(" PRIMARY: Tree vs. single CNN-LSTM")
    print(f"   ΔAUROC = {single_res['delta_auroc']:+.4f} "
          f"[{single_res['delta_auroc_ci_lo']:+.4f}, "
          f"{single_res['delta_auroc_ci_hi']:+.4f}]  "
          f"p = {fmt_p(single_res['auroc_p'])}")
    print(f"   ΔAUPRC = {single_res['delta_auprc']:+.4f} "
          f"[{single_res['delta_auprc_ci_lo']:+.4f}, "
          f"{single_res['delta_auprc_ci_hi']:+.4f}]  "
          f"p = {fmt_p(single_res['auprc_p'])}")
    print(f"   McNemar b={single_res['mcnemar_b']}, "
          f"c={single_res['mcnemar_c']}, "
          f"p = {fmt_p(single_res['mcnemar_p'])}")
    if ensemble_res is not None:
        print("-" * 80)
        print(" SECONDARY: Tree vs. 5-fold ensemble")
        print(f"   ΔAUROC = {ensemble_res['delta_auroc']:+.4f}  "
              f"p = {fmt_p(ensemble_res['auroc_p'])}")
        print(f"   ΔAUPRC = {ensemble_res['delta_auprc']:+.4f}  "
              f"p = {fmt_p(ensemble_res['auprc_p'])}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
        sys.exit(0)
    sys.exit(main())
