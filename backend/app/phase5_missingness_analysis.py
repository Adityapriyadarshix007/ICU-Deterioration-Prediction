#!/usr/bin/env python3
"""
Phase 5: Missingness Characterization (CORRECTED + UPGRADED)

PURPOSE:
    Describe missingness. Run one MCAR test. Test whether missingness
    is associated with the outcome.

    This phase is DIAGNOSTIC ONLY. It does not impute, drop, or modify data.

    We do NOT test MAR vs MNAR (impossible).
    We do NOT classify mechanisms per feature.
    We do NOT use Fisher's method, logistic regression, or rate-based rules.

UPGRADES (this version):
    - MCAR test scale caveat added (Little's test is hypersensitive at N)
    - Cramér's V is now the PRIMARY metric for missingness-vs-outcome
    - Effect size classification (negligible/small/medium/large)
    - Report ranks by V, not by p-value
    - Interpretive guidance added to log and report

DATA:
    Reads outputs/tables/X_train.csv from Phase 4.
    - Metadata columns: stay_id, hadm_id, subject_id, intime
    - Feature columns: numeric features (may include missing)
    - Mask columns: *_missing (from Phase 4)
    - Target: outcome

    Analysis focuses on the NUMERIC features only (metadata and masks excluded).

OUTPUTS:
    - phase5_descriptive_stats.csv       : per-feature missing rate
    - phase5_mcar_test.json              : MCAR test result + caveat
    - phase5_missingness_vs_outcome.csv  : chi-square + BH-FDR + Cramér's V
    - 5 visualization PNGs
    - phase5_report.txt                  : one-page summary

REQUIRES:
    pandas, numpy, scipy, matplotlib, seaborn, statsmodels, pyampute
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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

# pyampute is REQUIRED — we do not substitute a fake MCAR test.
try:
    from pyampute.exploration.mcar_statistical_tests import MCARTest
    import pyampute
    PYAMPUTE_VERSION = getattr(pyampute, '__version__', 'unknown')
except ImportError:
    print("=" * 80)
    print(" ERROR: pyampute is required for Phase 5.")
    print(" Install with: pip install pyampute")
    print("=" * 80)
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TABLE_DIR, FIG_DIR, LOGS_DIR, RANDOM_STATE,
    FEATURE_WINDOW_HOURS, PREDICTION_START_HOUR,
    PREDICTION_END_HOUR, SOFA_CHANGE_THRESHOLD,
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================
# COLUMN PARTITIONS
# ============================================================
# Phase 4 output contains these columns:
#   - stay_id, hadm_id, subject_id, intime  (metadata)
#   - outcome                                 (target)
#   - ~118 numeric features                   (analyzed here)
#   - ~89 *_missing masks                     (from Phase 4)

METADATA_COLS = {'stay_id', 'hadm_id', 'subject_id', 'intime'}
TARGET_COL = 'outcome'
MASK_SUFFIX = '_missing'

# Cramér's V thresholds for a 2×2 table (df=1)
# (Cohen's standard thresholds)
CV_NEGLIGIBLE = 0.10   # below this: negligible
CV_SMALL = 0.30        # 0.10–0.30: small
CV_MEDIUM = 0.50       # 0.30–0.50: medium
                       # ≥0.50: large


# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                LOGS_DIR / f"phase5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def print_header():
    print("=" * 80)
    print(" PHASE 5: MISSINGNESS CHARACTERIZATION")
    print("=" * 80)
    print(f" Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    print(f" Features: 0-{FEATURE_WINDOW_HOURS}h (from Phase 3)")
    print(f" Outcome:  {PREDICTION_START_HOUR}-{PREDICTION_END_HOUR}h "
          f"(pure ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD})")
    print(f" Data:     X_train.csv from Phase 4 (NO DATA LEAKAGE)")
    print(f" pyampute: {PYAMPUTE_VERSION}")
    print("-" * 80)
    print(" STEPS:")
    print("   1. Descriptive audit (missing rates, patterns)")
    print("   2. One MCAR test (Little's test via pyampute)")
    print("   3. Missingness vs outcome (chi-square + BH-FDR + Cramér's V)")
    print("   4. Visualizations")
    print("   5. Report")
    print("-" * 80)
    print(" INTERPRETATION NOTES:")
    print("   • Cramér's V is the PRIMARY metric for association strength.")
    print("     Chi-square p-values are inflated by sample size and are")
    print("     reported for completeness only.")
    print("   • Little's MCAR test is hypersensitive at large N. Rejection")
    print("     is expected and does NOT constitute strong evidence against MCAR.")
    print("   • MAR vs MNAR cannot be distinguished statistically.")
    print("=" * 80)
    print()


# ============================================================
# ANALYZER
# ============================================================

class MissingnessAnalyzer:
    """Missingness characterization on train data (diagnostic only)."""

    def __init__(self, logger):
        self.logger = logger
        self.X_train = None
        self.y_train = None

        # Column partitions
        self.numeric_cols: List[str] = []
        self.mask_cols: List[str] = []

        # Results
        self.missing_rates = None
        self.missingness_matrix = None
        self.mcar_result: Dict = {}
        self.outcome_assoc = None

        # Pattern statistics
        self.n_patterns: int = 0
        self.n_complete: int = 0
        self.pattern_counts = None

    # ------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------

    def load_data(self) -> bool:
        self.logger.info("\n" + "=" * 60)
        self.logger.info("LOADING DATA FROM PHASE 4")
        self.logger.info("=" * 60)

        path = TABLE_DIR / "X_train.csv"
        if not path.exists():
            self.logger.error(f"X_train.csv not found: {path}")
            self.logger.error("Run Phase 4 first: python3 phase4.py")
            return False

        df = pd.read_csv(path)
        self.logger.info(f"Loaded X_train: {df.shape}")

        if TARGET_COL not in df.columns:
            self.logger.error(f"'{TARGET_COL}' column not found in X_train.csv")
            return False

        # Extract target
        self.y_train = df[TARGET_COL].copy()

        # Partition columns
        meta_present = [c for c in df.columns if c in METADATA_COLS]
        self.mask_cols = [c for c in df.columns if c.endswith(MASK_SUFFIX)]
        numeric_candidates = [
            c for c in df.columns
            if c not in METADATA_COLS
            and c != TARGET_COL
            and c not in self.mask_cols
        ]

        # Keep only numeric columns (defensive)
        self.numeric_cols = [
            c for c in numeric_candidates
            if pd.api.types.is_numeric_dtype(df[c])
        ]
        dropped_non_numeric = [
            c for c in numeric_candidates
            if c not in self.numeric_cols
        ]
        if dropped_non_numeric:
            self.logger.warning(
                f"Non-numeric columns excluded: {dropped_non_numeric[:5]}"
                f"{'...' if len(dropped_non_numeric) > 5 else ''}"
            )

        # Store X (metadata + features + masks, excluding target)
        self.X_train = df.drop(columns=[TARGET_COL])

        # Missingness matrix over NUMERIC features only
        self.missingness_matrix = self.X_train[self.numeric_cols].isnull().astype(int)
        self.missing_rates = self.missingness_matrix.mean()

        n_cells = self.missingness_matrix.size
        n_missing = int(self.missingness_matrix.values.sum())

        self.logger.info(f"  Metadata columns: {meta_present}")
        self.logger.info(f"  Numeric features: {len(self.numeric_cols)}")
        self.logger.info(f"  Mask columns:     {len(self.mask_cols)}")
        self.logger.info(f"  Overall missingness (numeric features): "
                         f"{n_missing:,}/{n_cells:,} "
                         f"({n_missing / n_cells * 100:.2f}%)")
        self.logger.info(f"  Target distribution:")
        counts = self.y_train.value_counts()
        self.logger.info(f"    Stable (0): {counts.get(0, 0):,}")
        self.logger.info(f"    Progression (1): {counts.get(1, 0):,}")

        return True

    # ------------------------------------------------------------
    # STEP 1: DESCRIPTIVE AUDIT
    # ------------------------------------------------------------

    def step1_descriptive_audit(self):
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 1: DESCRIPTIVE AUDIT")
        self.logger.info("=" * 60)

        rates = self.missing_rates[self.missing_rates > 0].sort_values(
            ascending=False
        )

        self.logger.info(f"Features with missingness: {len(rates)} / "
                         f"{len(self.missing_rates)}")

        if len(rates) == 0:
            self.logger.info("No missing values found.")
            return

        # Descriptive rate distribution
        low = rates[rates < 0.05]
        med = rates[(rates >= 0.05) & (rates < 0.15)]
        high = rates[rates >= 0.15]

        self.logger.info(f"  <5%:    {len(low)}")
        self.logger.info(f"  5–15%:  {len(med)}")
        self.logger.info(f"  >15%:   {len(high)}")

        self.logger.info("\nTop 20 features by missing rate:")
        for feat, rate in rates.head(20).items():
            self.logger.info(f"  {feat}: {rate * 100:.2f}%")

        # Pattern statistics
        unique_patterns = self.missingness_matrix.drop_duplicates()
        self.n_patterns = len(unique_patterns)

        n_complete = int((self.missingness_matrix.sum(axis=1) == 0).sum())
        self.n_complete = n_complete

        self.logger.info(f"\nPattern statistics:")
        self.logger.info(f"  Unique missingness patterns: {self.n_patterns}")
        self.logger.info(f"  Samples with complete data: {n_complete} "
                         f"({n_complete / len(self.X_train) * 100:.1f}%)")

        # Top 5 patterns
        pattern_series = self.missingness_matrix.apply(
            lambda row: tuple(row), axis=1
        )
        pattern_counts = pattern_series.value_counts()
        self.pattern_counts = pattern_counts

        self.logger.info(f"\nTop 5 most common missingness patterns:")
        for i, (pattern, count) in enumerate(pattern_counts.head(5).items()):
            pct = count / len(self.X_train) * 100
            n_missing_in_pattern = sum(pattern)
            self.logger.info(
                f"  Pattern {i+1}: {count:,} samples ({pct:.1f}%), "
                f"{n_missing_in_pattern} features missing"
            )

        # Save
        desc = pd.DataFrame({
            'feature': self.missing_rates.index,
            'missing_rate': self.missing_rates.values,
            'missing_count': self.missingness_matrix.sum().values,
            'n_samples': len(self.missingness_matrix),
        }).sort_values('missing_rate', ascending=False)
        desc.to_csv(TABLE_DIR / 'phase5_descriptive_stats.csv', index=False)
        self.logger.info(f"\nSaved: phase5_descriptive_stats.csv")

    # ------------------------------------------------------------
    # STEP 2: MCAR TEST
    # ------------------------------------------------------------

    def step2_mcar_test(self):
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 2: MCAR TEST (Little's test via pyampute)")
        self.logger.info("=" * 60)

        cols = [c for c in self.numeric_cols
                if self.missingness_matrix[c].any()]

        if len(cols) < 2:
            self.logger.info("Not enough features with missingness. Skipping.")
            self.mcar_result = {'status': 'skipped',
                                'reason': 'insufficient_features'}
            self._save_mcar_result()
            return

        self.logger.info(f"Testing {len(cols)} features with missingness")

        try:
            data = self.X_train[cols]
            mcar = MCARTest(method="little")
            p_value = mcar.little_mcar_test(data)

            self.mcar_result = {
                'status': 'completed',
                'method': 'little_pyampute',
                'pyampute_version': PYAMPUTE_VERSION,
                'p_value': float(p_value),
                'n_features_tested': len(cols),
                'n_samples': len(data),
                'conclusion': (
                    'reject_MCAR' if p_value <= 0.05
                    else 'cannot_reject_MCAR'
                ),
                'scale_caveat': (
                    "Little's test is hypersensitive at large sample sizes. "
                    "Rejection is expected and does not constitute strong "
                    "evidence against MCAR. MAR and MNAR remain "
                    "indistinguishable."
                ),
            }
            self.logger.info(f"Little's MCAR test p-value: {p_value:.4g}")
            if p_value > 0.05:
                self.logger.info("  Cannot reject MCAR (p > 0.05)")
                self.logger.info("  → Missingness is consistent with MCAR")
            else:
                self.logger.info("  Reject MCAR (p ≤ 0.05)")
                self.logger.info("  → Missingness is NOT exactly MCAR")
                self.logger.info("  → It is either MAR or MNAR "
                                 "(cannot distinguish)")

            # Always print the scale caveat after reporting the result
            self.logger.info("")
            self.logger.info("  ⚠️  SCALE CAVEAT:")
            self.logger.info(f"      Little's test at n={len(data):,} is "
                             f"hypersensitive.")
            self.logger.info("      Rejection indicates non-exact-MCAR, "
                             "not strong")
            self.logger.info("      evidence against MCAR. MAR vs MNAR "
                             "remains indistinguishable.")

        except Exception as e:
            self.logger.error(f"MCAR test failed: {e}")
            self.mcar_result = {'status': 'failed', 'reason': str(e)}

        self._save_mcar_result()

    def _save_mcar_result(self):
        with open(TABLE_DIR / 'phase5_mcar_test.json', 'w') as f:
            json.dump(self.mcar_result, f, indent=2)
        self.logger.info(f"Saved: phase5_mcar_test.json")

    # ------------------------------------------------------------
    # STEP 3: MISSINGNESS vs OUTCOME
    # ------------------------------------------------------------

    def step3_missingness_vs_outcome(self):
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 3: MISSINGNESS vs OUTCOME")
        self.logger.info("=" * 60)
        self.logger.info("PRIMARY METRIC: Cramér's V (effect size)")
        self.logger.info("Chi-square p-values reported for reference only.")

        cols = [c for c in self.numeric_cols
                if self.missingness_matrix[c].any()]
        if not cols:
            self.logger.info("No features with missingness.")
            return

        rows = []
        for col in cols:
            is_missing = self.missingness_matrix[col]
            ct = pd.crosstab(is_missing, self.y_train)
            if ct.shape != (2, 2):
                continue
            try:
                chi2, p, _, _ = stats.chi2_contingency(ct)
                n = ct.values.sum()
                cramer_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

                # Effect size classification (df=1, 2×2 table)
                if cramer_v < CV_NEGLIGIBLE:
                    effect = 'negligible'
                elif cramer_v < CV_SMALL:
                    effect = 'small'
                elif cramer_v < CV_MEDIUM:
                    effect = 'medium'
                else:
                    effect = 'large'

                rows.append({
                    'feature': col,
                    'missing_rate': float(is_missing.mean()),
                    'chi2': float(chi2),
                    'p_value': float(p),
                    'cramers_v': float(cramer_v),
                    'effect_size': effect,
                })
            except Exception:
                continue

        if not rows:
            self.logger.info("No valid contingency tables.")
            return

        res = pd.DataFrame(rows)

        # BH-FDR on p-values (kept for reference)
        _, res['p_bh'], _, _ = multipletests(
            res['p_value'].values, alpha=0.05, method='fdr_bh'
        )
        res['significant_bh'] = res['p_bh'] < 0.05

        # Rank by Cramér's V (primary), not by p-value
        res = res.sort_values('cramers_v', ascending=False).reset_index(drop=True)

        # Counts
        n_total = len(res)
        n_bh_sig = int(res['significant_bh'].sum())
        n_v_small = int((res['cramers_v'] >= CV_NEGLIGIBLE).sum())
        n_v_medium = int((res['cramers_v'] >= CV_SMALL).sum())
        n_v_large = int((res['cramers_v'] >= CV_MEDIUM).sum())

        self.logger.info("")
        self.logger.info(f"Total features tested: {n_total}")
        self.logger.info(f"BH-FDR significant (p_bh < 0.05): {n_bh_sig}/{n_total}")
        self.logger.info("")
        self.logger.info("Effect size distribution (Cramér's V):")
        self.logger.info(f"  V ≥ 0.10 (small or larger):   {n_v_small}")
        self.logger.info(f"  V ≥ 0.30 (medium or larger):  {n_v_medium}")
        self.logger.info(f"  V ≥ 0.50 (large):             {n_v_large}")
        self.logger.info("")
        self.logger.info("⚠️  p-values are inflated by sample size; treat V as")
        self.logger.info("    the primary measure of association strength.")
        self.logger.info("")
        self.logger.info("Top 15 features by Cramér's V:")

        for _, row in res.head(15).iterrows():
            self.logger.info(
                f"  V={row['cramers_v']:.3f} ({row['effect_size']:10s}) "
                f"{row['feature']}: "
                f"p_bh={row['p_bh']:.4g}, "
                f"missing={row['missing_rate'] * 100:.1f}%"
            )

        self.outcome_assoc = res
        res.to_csv(TABLE_DIR / 'phase5_missingness_vs_outcome.csv', index=False)
        self.logger.info(f"\nSaved: phase5_missingness_vs_outcome.csv")

    # ------------------------------------------------------------
    # STEP 4: VISUALIZATIONS
    # ------------------------------------------------------------

    def step4_visualizations(self):
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 4: VISUALIZATIONS")
        self.logger.info("=" * 60)

        FIG_DIR.mkdir(parents=True, exist_ok=True)
        rates_sorted = self.missing_rates[self.missing_rates > 0].sort_values()

        if len(rates_sorted) == 0:
            self.logger.info("No missingness to visualize.")
            return

        # 1. Bar chart
        fig, ax = plt.subplots(figsize=(10, max(6, len(rates_sorted) * 0.15)))
        ax.barh(range(len(rates_sorted)), rates_sorted.values, color='steelblue')
        ax.set_yticks(range(len(rates_sorted)))
        ax.set_yticklabels(rates_sorted.index, fontsize=7)
        ax.axvline(0.05, color='green', linestyle='--', alpha=0.6, label='5%')
        ax.axvline(0.15, color='orange', linestyle='--', alpha=0.6, label='15%')
        ax.set_xlabel('Missing rate')
        ax.set_title('Missing Rates by Feature')
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'phase5_missing_rates_bar.png', dpi=150)
        plt.close()
        self.logger.info("  phase5_missing_rates_bar.png")

        # 2. Distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(rates_sorted.values, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(0.05, color='green', linestyle='--', alpha=0.6, label='5%')
        ax.axvline(0.15, color='orange', linestyle='--', alpha=0.6, label='15%')
        ax.set_xlabel('Missing rate')
        ax.set_ylabel('Number of features')
        ax.set_title('Distribution of Missing Rates')
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'phase5_missing_rates_distribution.png', dpi=150)
        plt.close()
        self.logger.info("  phase5_missing_rates_distribution.png")

        # 3. Heatmap
        cols_with_missing = [c for c in self.numeric_cols
                             if self.missingness_matrix[c].any()]
        if len(cols_with_missing) > 1:
            M = self.missingness_matrix[cols_with_missing]
            if M.shape[0] > 500:
                M = M.sample(500, random_state=RANDOM_STATE)
            if M.shape[1] > 60:
                top = M.mean().sort_values(ascending=False).head(60).index
                M = M[top]
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.heatmap(M.T, cbar=True, cmap='RdBu_r', ax=ax,
                        xticklabels=False, yticklabels=True,
                        cbar_kws={'label': '1 = missing'})
            ax.set_title('Missingness Heatmap (sampled rows, top features)')
            ax.set_xlabel('Sample index')
            ax.set_ylabel('Feature')
            plt.tight_layout()
            plt.savefig(FIG_DIR / 'phase5_missingness_heatmap.png', dpi=150)
            plt.close()
            self.logger.info("  phase5_missingness_heatmap.png")

        # 4. Normalized co-occurrence
        if len(cols_with_missing) > 1:
            M = self.missingness_matrix[cols_with_missing]
            joint = M.T.dot(M).values.astype(float)
            diag = np.diag(joint).copy()
            with np.errstate(divide='ignore', invalid='ignore'):
                cond = joint / diag[:, None]
            cond = np.nan_to_num(cond, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(cond, 0)

            if cond.shape[0] > 15:
                idx = np.argsort(-diag)[:15]
                cond_small = cond[np.ix_(idx, idx)]
                labels = [cols_with_missing[i] for i in idx]
            else:
                cond_small = cond
                labels = cols_with_missing

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cond_small, annot=False, cmap='YlOrRd', ax=ax,
                        xticklabels=labels, yticklabels=labels,
                        vmin=0, vmax=1,
                        cbar_kws={'label': 'P(col missing | row missing)'})
            ax.set_title('Normalized Missingness Co-occurrence')
            plt.xticks(rotation=90, fontsize=7)
            plt.yticks(rotation=0, fontsize=7)
            plt.tight_layout()
            plt.savefig(FIG_DIR / 'phase5_missingness_cooccurrence.png', dpi=150)
            plt.close()
            self.logger.info("  phase5_missingness_cooccurrence.png")

        # 5. Missingness vs outcome scatter (primary metric: V)
        if self.outcome_assoc is not None and len(self.outcome_assoc) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            meaningful = self.outcome_assoc['cramers_v'] >= CV_NEGLIGIBLE
            ax.scatter(
                self.outcome_assoc.loc[~meaningful, 'missing_rate'],
                self.outcome_assoc.loc[~meaningful, 'cramers_v'],
                alpha=0.5, s=25, color='gray',
                label=f"Negligible (V < {CV_NEGLIGIBLE})",
            )
            ax.scatter(
                self.outcome_assoc.loc[meaningful, 'missing_rate'],
                self.outcome_assoc.loc[meaningful, 'cramers_v'],
                alpha=0.7, s=35, color='red',
                label=f"Meaningful (V ≥ {CV_NEGLIGIBLE})",
            )
            ax.axhline(CV_NEGLIGIBLE, color='black', linestyle='--',
                       alpha=0.4, linewidth=1)
            ax.set_xlabel('Missing rate')
            ax.set_ylabel("Cramér's V (missingness vs outcome)")
            ax.set_title('Missingness–Outcome Association (Cramér\'s V)')
            ax.legend()
            plt.tight_layout()
            plt.savefig(FIG_DIR / 'phase5_missingness_vs_outcome.png', dpi=150)
            plt.close()
            self.logger.info("  phase5_missingness_vs_outcome.png")

    # ------------------------------------------------------------
    # STEP 5: REPORT
    # ------------------------------------------------------------

    def step5_report(self):
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 5: REPORT")
        self.logger.info("=" * 60)

        rates = self.missing_rates[self.missing_rates > 0]
        total_cells = self.missingness_matrix.size
        n_missing = int(self.missingness_matrix.values.sum())

        lines = []
        lines.append("=" * 70)
        lines.append("PHASE 5: MISSINGNESS CHARACTERIZATION REPORT")
        lines.append("=" * 70)
        lines.append(f"Feature window: 0-{FEATURE_WINDOW_HOURS}h from ICU admission")
        lines.append(f"Outcome: pure ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD} at "
                     f"{PREDICTION_START_HOUR}-{PREDICTION_END_HOUR}h")
        lines.append(f"Data: X_train.csv (train only, from Phase 4)")
        lines.append(f"Samples: {len(self.X_train):,}")
        lines.append(f"Numeric features: {len(self.numeric_cols)}")
        lines.append(f"Mask columns (from Phase 4): {len(self.mask_cols)}")
        lines.append(f"pyampute version: {PYAMPUTE_VERSION}")
        lines.append(f"Overall missingness: {n_missing:,}/{total_cells:,} "
                     f"({n_missing / total_cells * 100:.2f}%)")

        lines.append("\n" + "-" * 70)
        lines.append("STEP 1: DESCRIPTIVE")
        lines.append("-" * 70)
        lines.append(f"Features with missingness: {len(rates)}")
        if len(rates) > 0:
            lines.append(f"  <5%:    {int((rates < 0.05).sum())}")
            lines.append(f"  5–15%:  {int(((rates >= 0.05) & (rates < 0.15)).sum())}")
            lines.append(f"  >15%:   {int((rates >= 0.15).sum())}")
        lines.append(f"Unique missingness patterns: {self.n_patterns}")
        lines.append(f"Samples with complete data: {self.n_complete} "
                     f"({self.n_complete / len(self.X_train) * 100:.1f}%)")

        lines.append("\n" + "-" * 70)
        lines.append("STEP 2: MCAR TEST")
        lines.append("-" * 70)
        for k, v in self.mcar_result.items():
            lines.append(f"  {k}: {v}")

        lines.append("\n" + "-" * 70)
        lines.append("STEP 3: MISSINGNESS vs OUTCOME")
        lines.append("-" * 70)
        if self.outcome_assoc is not None and len(self.outcome_assoc) > 0:
            n_total = len(self.outcome_assoc)
            n_bh_sig = int(self.outcome_assoc['significant_bh'].sum())
            n_v_small = int((self.outcome_assoc['cramers_v'] >= CV_NEGLIGIBLE).sum())
            n_v_medium = int((self.outcome_assoc['cramers_v'] >= CV_SMALL).sum())
            n_v_large = int((self.outcome_assoc['cramers_v'] >= CV_MEDIUM).sum())

            lines.append("PRIMARY METRIC: Cramér's V (effect size)")
            lines.append("Chi-square p-values reported for reference only.")
            lines.append("")
            lines.append(f"Total features tested: {n_total}")
            lines.append(f"BH-FDR significant (p_bh < 0.05): {n_bh_sig}/{n_total}")
            lines.append("")
            lines.append("Effect size distribution (Cramér's V):")
            lines.append(f"  V ≥ 0.10 (small or larger):   {n_v_small}")
            lines.append(f"  V ≥ 0.30 (medium or larger):  {n_v_medium}")
            lines.append(f"  V ≥ 0.50 (large):             {n_v_large}")
            lines.append("")
            lines.append("⚠️  Given the large sample size, chi-square p-values")
            lines.append("    are expected to be significant for trivial")
            lines.append("    associations. We therefore report Cramér's V as")
            lines.append("    the primary measure of effect size and interpret")
            lines.append("    V ≥ 0.10 as potentially meaningful.")
            lines.append("")
            lines.append("Cramér's V interpretation (2×2 table, df=1):")
            lines.append("  V < 0.10 : negligible")
            lines.append("  0.10–0.30: small")
            lines.append("  0.30–0.50: medium")
            lines.append("  V ≥ 0.50 : large")
            lines.append("")
            lines.append("Top 10 features by Cramér's V:")

            for _, row in self.outcome_assoc.head(10).iterrows():
                lines.append(
                    f"  V={row['cramers_v']:.3f} ({row['effect_size']:10s}) "
                    f"{row['feature']}: "
                    f"p_bh={row['p_bh']:.4g}, "
                    f"missing={row['missing_rate'] * 100:.1f}%"
                )

        lines.append("\n" + "-" * 70)
        lines.append("LIMITATIONS")
        lines.append("-" * 70)
        lines.append("  1. MAR vs MNAR cannot be distinguished statistically.")
        lines.append("     Any method claiming to do so makes untestable")
        lines.append("     assumptions. We assume MAR for imputation purposes")
        lines.append("     and acknowledge this as a limitation.")
        lines.append("")
        lines.append("  2. If MCAR was rejected, we know only that missingness")
        lines.append("     is not completely random. We do not know whether it")
        lines.append("     is MAR or MNAR.")
        lines.append("")
        lines.append("  3. Little's MCAR test is hypersensitive at large N.")
        lines.append("     Rejection indicates non-exact-MCAR, not strong")
        lines.append("     evidence against MCAR.")
        lines.append("")
        lines.append("  4. Chi-square p-values are inflated by sample size.")
        lines.append("     Cramér's V is reported as the primary effect size")
        lines.append("     and should be used to prioritize features.")

        lines.append("\n" + "-" * 70)
        lines.append("IMPLICATIONS FOR PHASE 6")
        lines.append("-" * 70)
        lines.append("  • The MCAR test result determines whether simple")
        lines.append("    imputation (e.g., median) is justified. If MCAR is")
        lines.append("    rejected, do not assume MCAR for imputation.")
        lines.append("  • Test multiple imputation methods empirically in")
        lines.append("    Phase 6 (the phase already does this: 4 tabular")
        lines.append("    strategies × 3 tree models).")
        lines.append("  • Missingness masks are already present in X_train.csv")
        lines.append("    from Phase 4. Phase 6 decides whether to include them.")
        lines.append("  • Prioritize features with high Cramér's V when")
        lines.append("    deciding which masks to include.")
        lines.append("  • Include sensitivity analysis under MAR and MNAR")
        lines.append("    assumptions in Phase 6.")

        path = TABLE_DIR / 'phase5_report.txt'
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        self.logger.info(f"Saved: {path}")

    # ------------------------------------------------------------
    # RUN ALL
    # ------------------------------------------------------------

    def run_all(self) -> bool:
        if not self.load_data():
            return False
        self.step1_descriptive_audit()
        self.step2_mcar_test()
        self.step3_missingness_vs_outcome()
        self.step4_visualizations()
        self.step5_report()
        return True


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Phase 5: Missingness Characterization (Corrected)'
    )
    args = parser.parse_args()  # capture for future-proofing

    logger = setup_logging()
    print_header()

    try:
        analyzer = MissingnessAnalyzer(logger)
        success = analyzer.run_all()
        if not success:
            return 1

        print("\n" + "=" * 80)
        print(" PHASE 5 COMPLETED")
        print("=" * 80)
        print(f" Outputs: {TABLE_DIR}")
        print(f" Figures: {FIG_DIR}")
        print("=" * 80)
        return 0

    except Exception as e:
        logger.error(f"Error in Phase 5: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
