#!/usr/bin/env python3
"""
Phase 4: Feature Filtering and Preprocessing (CORRECTED)

- Whitelist-based feature selection (base-name matching)
- Train/Test split FIRST (stratified 85/15) — KEEPS MISSING VALUES
- Missingness characterization on TRAIN ONLY (no leakage)
- Missingness filter (>60% dropped, with SOFA-relevant exemptions)
- Create missingness masks (train-only rates, applied to both)
- Winsorization (0.5-99.5 percentile, fit on train)
- Z-score standardization (fit on train)

FIXES APPLIED:
1. _reduce_aggregations: clean if/else (was crashing on .add() called on a list)
2. Split before missingness characterization (train-only stats)
3. Preserve stay_id in X_train.csv / X_test.csv
4. NO outcome re-merge: Phase 3 already provides 'outcome' in the features file

INPUT:
  - outputs/tables/features_0_6h_none.csv  (includes 'outcome')

OUTPUT:
  - outputs/tables/X_train.csv
  - outputs/tables/X_test.csv
  - outputs/tables/feature_cols.txt
  - outputs/tables/scaler_params.csv
"""

import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TABLE_DIR, LOGS_DIR,
    MAX_MISSING_RATE,
    OUTLIER_LOW,
    OUTLIER_HIGH,
    RANDOM_STATE,
    TEST_SIZE,
    FEATURE_WINDOW_HOURS,
    PREDICTION_START_HOUR,
    PREDICTION_END_HOUR,
    SOFA_CHANGE_THRESHOLD,
)


# ============================================================
# SOFA-RELEVANT FEATURES (exempted from missingness filter)
# ============================================================
SOFA_RELEVANT_FEATURES = {
    'bilirubin_worst', 'bilirubin_last',
    'platelets_worst', 'platelets_last',
    'creatinine_worst', 'creatinine_last',
    'pao2_min', 'pao2_max', 'pao2_mean', 'pao2_last',
    'fio2_min', 'fio2_max', 'fio2_mean', 'fio2_last',
    'pao2_fio2_ratio',
}

METADATA_COLS = {'stay_id', 'hadm_id', 'subject_id', 'intime'}
OUTCOME_COLS = {'outcome', 'sofa_feature', 'sofa_outcome', 'sofa_change'}


# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    phase4_log = LOGS_DIR / f"phase4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(phase4_log),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


def print_header():
    print("=" * 80)
    print(" PHASE 4: FEATURE FILTERING AND PREPROCESSING")
    print("=" * 80)
    print(f" Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    print(f" Feature Window: 0-{FEATURE_WINDOW_HOURS}h from ICU admission")
    print(f" Outcome Window: {PREDICTION_START_HOUR}-{PREDICTION_END_HOUR}h from ICU admission")
    print(f" Target: Pure ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD}")
    print("-" * 80)
    print(f" Max Missing Rate: {MAX_MISSING_RATE*100:.0f}%")
    print(f" Outlier Percentiles: {OUTLIER_LOW*100:.1f}% - {OUTLIER_HIGH*100:.1f}%")
    print(f" Test Size: {TEST_SIZE*100:.0f}%")
    print("=" * 80)
    print()


class FeaturePreprocessor:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.features = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.feature_cols = None
        self.missingness_report = None

    # ------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------

    def load_features(self, features_path: str, cohort_path: str = None):
        """
        Load features from Phase 3. 'outcome' is already in the file.
        Do NOT re-merge outcomes (creates outcome_x/outcome_y duplicates).
        cohort_path is accepted for backward compat but not used.
        """
        self.logger.info(f"Loading features from: {features_path}")
        self.features = pd.read_csv(features_path)
        self.logger.info(f"Loaded {self.features.shape[0]:,} rows, "
                         f"{self.features.shape[1]} columns")
        self.logger.info(f"⚠️  Features from 0-{FEATURE_WINDOW_HOURS}h window")

        # Drop pandas merge collision artifacts
        collision_cols = [c for c in self.features.columns
                          if c.endswith('_x') or c.endswith('_y')]
        if collision_cols:
            self.logger.warning(
                f"⚠️  Detected {len(collision_cols)} columns ending in _x or _y. "
                f"Dropping."
            )
            for c in collision_cols[:10]:
                self.logger.warning(f"    {c}")
            self.features = self.features.drop(columns=collision_cols)

        # Verify outcome is present
        if 'outcome' not in self.features.columns:
            self.logger.error(
                "'outcome' not found in features file. "
                "Phase 3 must have merged it from the cohort."
            )
            return

        # Whitelist reduction
        self._reduce_aggregations()

        # Log target distribution
        counts = self.features['outcome'].value_counts()
        self.logger.info(f"\nTarget (ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD}):")
        self.logger.info(f"  Stable (0): {counts.get(0, 0):,} "
                         f"({counts.get(0, 0)/len(self.features)*100:.1f}%)")
        self.logger.info(f"  Progression (1): {counts.get(1, 0):,} "
                         f"({counts.get(1, 0)/len(self.features)*100:.1f}%)")

    # ------------------------------------------------------------
    # WHITELIST REDUCTION
    # ------------------------------------------------------------

    def _get_base_and_suffix(self, col):
        suffixes = [
            '_time_since_last_measure',
            '_max_rate', '_mean_rate',
            '_slope', '_delta', '_range',
            '_worst',
            '_mean', '_median', '_min', '_max', '_std', '_last', '_count',
            '_any',
        ]
        for s in suffixes:
            if col.endswith(s):
                return col[:-len(s)], s
        return col, None

    def _reduce_aggregations(self):
        self.logger.info("\n" + "=" * 60)
        self.logger.info("WHITELIST FEATURE REDUCTION")
        self.logger.info("=" * 60)

        always_keep = METADATA_COLS | OUTCOME_COLS

        VITAL_BASES = {
            'heart_rate', 'hr',
            'respiratory_rate', 'rr',
            'spo2', 'map', 'temperature', 'sbp', 'dbp',
        }
        LAB_BASES = {
            'creatinine', 'bilirubin', 'platelets', 'bun', 'lactate',
            'wbc', 'hemoglobin', 'hematocrit', 'sodium', 'potassium',
            'chloride', 'bicarbonate', 'glucose', 'albumin', 'alt',
            'ast', 'alkaline_phosphatase', 'inr', 'ptt', 'pt',
        }
        GCS_BASES = {'gcs_eyes', 'gcs_verbal', 'gcs_motor', 'gcs_total'}
        RESP_BASES = {'pao2', 'fio2', 'paco2'}
        VASO_BASES = {
            'norepinephrine', 'epinephrine', 'dopamine',
            'dobutamine', 'vasopressin', 'phenylephrine',
        }

        VITAL_SUFFIXES = {'_mean', '_slope', '_delta', '_range',
                          '_time_since_last_measure'}
        LAB_SUFFIXES = {'_worst', '_last', '_slope', '_delta'}
        GCS_SUFFIXES = {'_min', '_last', '_slope', '_delta'}
        RESP_SUFFIXES = {'_min', '_max', '_mean', '_last', '_slope', '_delta'}
        VASO_SUFFIXES = {'_any', '_max_rate', '_mean_rate'}

        cols_to_keep = set(always_keep)
        cols_to_drop = []

        for col in self.features.columns:
            if col in always_keep:
                continue
            if col.endswith('_missing'):
                cols_to_keep.add(col)
                continue
            if col in ('age', 'gender_male') or col.startswith('admission_'):
                cols_to_keep.add(col)
                continue

            base, suffix = self._get_base_and_suffix(col)

            if suffix is None:
                cols_to_keep.add(col)
                continue

            if base in VITAL_BASES:
                if suffix in VITAL_SUFFIXES:
                    cols_to_keep.add(col)
                else:
                    cols_to_drop.append(col)
            elif base in LAB_BASES:
                if suffix in LAB_SUFFIXES:
                    cols_to_keep.add(col)
                else:
                    cols_to_drop.append(col)
            elif base in GCS_BASES:
                if suffix in GCS_SUFFIXES:
                    cols_to_keep.add(col)
                else:
                    cols_to_drop.append(col)
            elif base in RESP_BASES:
                if suffix in RESP_SUFFIXES:
                    cols_to_keep.add(col)
                else:
                    cols_to_drop.append(col)
            elif base in VASO_BASES:
                if suffix in VASO_SUFFIXES:
                    cols_to_keep.add(col)
                else:
                    cols_to_drop.append(col)
            else:
                all_allowed = (VITAL_SUFFIXES | LAB_SUFFIXES | GCS_SUFFIXES
                               | RESP_SUFFIXES | VASO_SUFFIXES)
                if suffix in all_allowed:
                    cols_to_keep.add(col)
                else:
                    cols_to_drop.append(col)

        if cols_to_drop:
            self.features = self.features.drop(columns=cols_to_drop)
            self.logger.info(f"Dropped {len(cols_to_drop)} non-whitelisted features")
            self.logger.info(f"  Sample dropped: {cols_to_drop[:8]}")

        self.features = self.features[[c for c in self.features.columns
                                       if c in cols_to_keep]]
        self.logger.info(f"Remaining features: {self.features.shape[1]} columns")

        numeric_count = sum(1 for c in self.features.columns
                            if c not in always_keep and not c.endswith('_missing'))
        mask_count = sum(1 for c in self.features.columns if c.endswith('_missing'))
        self.logger.info(f"  Numeric features: {numeric_count}")
        self.logger.info(f"  Missingness masks: {mask_count}")

        temporal = [c for c in self.features.columns
                    if any(c.endswith(s) for s in
                           ['_slope', '_delta', '_range',
                            '_time_since_last_measure'])]
        self.logger.info(f"  Temporal features retained: {len(temporal)}")

    # ------------------------------------------------------------
    # SPLIT
    # ------------------------------------------------------------

    def split_data(self):
        self.logger.info("\n" + "=" * 60)
        self.logger.info("TRAIN/TEST SPLIT")
        self.logger.info("=" * 60)

        if 'outcome' not in self.features.columns:
            self.logger.error("No 'outcome' column — cannot split.")
            return False

        feature_cols = [c for c in self.features.columns if c not in OUTCOME_COLS]
        X = self.features[feature_cols].copy()
        y = self.features['outcome'].copy()

        self.logger.info(f"Total samples: {len(X):,}")
        self.logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        self.logger.info(f"⚠️  Missing values in X: {X.isnull().sum().sum():,}")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        self.logger.info(f"\nTraining set: {len(self.X_train):,} samples")
        self.logger.info(f"  Stable (0): {sum(self.y_train==0):,} "
                         f"({sum(self.y_train==0)/len(self.y_train)*100:.1f}%)")
        self.logger.info(f"  Progression (1): {sum(self.y_train==1):,} "
                         f"({sum(self.y_train==1)/len(self.y_train)*100:.1f}%)")
        self.logger.info(f"  Missing values in train: "
                         f"{self.X_train.isnull().sum().sum():,}")

        self.logger.info(f"\nTest set: {len(self.X_test):,} samples")
        self.logger.info(f"  Stable (0): {sum(self.y_test==0):,} "
                         f"({sum(self.y_test==0)/len(self.y_test)*100:.1f}%)")
        self.logger.info(f"  Progression (1): {sum(self.y_test==1):,} "
                         f"({sum(self.y_test==1)/len(self.y_test)*100:.1f}%)")
        self.logger.info(f"  Missing values in test: "
                         f"{self.X_test.isnull().sum().sum():,}")

        return True

    # ------------------------------------------------------------
    # MISSINGNESS (TRAIN ONLY)
    # ------------------------------------------------------------

    def characterize_missingness(self):
        self.logger.info("\n" + "=" * 60)
        self.logger.info("MISSINGNESS CHARACTERIZATION (TRAIN ONLY)")
        self.logger.info("=" * 60)

        candidate_cols = [c for c in self.X_train.columns
                          if c not in METADATA_COLS
                          and not c.endswith('_missing')]

        missing_counts = self.X_train[candidate_cols].isnull().sum()
        missing_rates = missing_counts / len(self.X_train)

        self.missingness_report = pd.DataFrame({
            'feature': candidate_cols,
            'missing_count': missing_counts,
            'missing_rate': missing_rates,
        }).sort_values('missing_rate', ascending=False)

        total = len(candidate_cols)
        with_missing = (missing_rates > 0).sum()
        high_missing = (missing_rates > MAX_MISSING_RATE).sum()

        self.logger.info(f"\nTotal candidate features: {total}")
        self.logger.info(f"Features with missing values: {with_missing}")
        self.logger.info(f"Features with >{MAX_MISSING_RATE*100:.0f}% missing: "
                         f"{high_missing}")

        self.logger.info("\nTop 10 features by missing rate:")
        for _, row in self.missingness_report.head(10).iterrows():
            self.logger.info(f"  {row['feature']}: {row['missing_rate']*100:.1f}%")

        features_to_drop = self.missingness_report[
            self.missingness_report['missing_rate'] > MAX_MISSING_RATE
        ]['feature'].tolist()

        exempted = [f for f in features_to_drop if f in SOFA_RELEVANT_FEATURES]
        if exempted:
            self.logger.info(
                f"\n⚠️  Exempting {len(exempted)} SOFA-relevant features:"
            )
            for f in exempted:
                rate = self.missingness_report[
                    self.missingness_report['feature'] == f
                ]['missing_rate'].iloc[0]
                self.logger.info(f"    {f} ({rate*100:.1f}% missing)")
            features_to_drop = [f for f in features_to_drop
                                if f not in SOFA_RELEVANT_FEATURES]

        self.logger.info(f"\nDropping {len(features_to_drop)} features "
                         f"(>{MAX_MISSING_RATE*100:.0f}% missing)")

        self.logger.info("\nCreating missingness masks...")
        mask_count = 0
        final_feature_cols = ['stay_id']

        for col in candidate_cols:
            if col in features_to_drop:
                continue
            final_feature_cols.append(col)
            if self.X_train[col].isnull().any():
                mask_col = f'{col}_missing'
                self.X_train[mask_col] = self.X_train[col].isnull().astype(int)
                self.X_test[mask_col] = self.X_test[col].isnull().astype(int)
                final_feature_cols.append(mask_col)
                mask_count += 1

        drop_cols = [c for c in features_to_drop if c in self.X_train.columns]
        if drop_cols:
            self.X_train = self.X_train.drop(columns=drop_cols)
            self.X_test = self.X_test.drop(columns=drop_cols)

        self.feature_cols = final_feature_cols

        self.logger.info(f"Dropped {len(features_to_drop)} features")
        self.logger.info(f"Kept {len(candidate_cols) - len(features_to_drop)} "
                         f"original features")
        self.logger.info(f"Created {mask_count} missingness masks")
        self.logger.info(f"Total feature columns (incl. stay_id): "
                         f"{len(self.feature_cols)}")

        retained = [f for f in SOFA_RELEVANT_FEATURES if f in self.feature_cols]
        missing = SOFA_RELEVANT_FEATURES - set(retained)
        self.logger.info(f"\nSOFA-relevant retained: "
                         f"{len(retained)}/{len(SOFA_RELEVANT_FEATURES)}")
        if missing:
            self.logger.warning(
                f"  ⚠️  Not present in features: {sorted(missing)}"
            )

        return self.missingness_report

    # ------------------------------------------------------------
    # WINSORIZE
    # ------------------------------------------------------------

    def winsorize_outliers(self):
        self.logger.info("\n" + "=" * 60)
        self.logger.info("WINSORIZING OUTLIERS (fit on train)")
        self.logger.info("=" * 60)

        numeric_cols = []
        for col in self.feature_cols:
            if col == 'stay_id':
                continue
            if col.endswith('_missing'):
                continue
            if self.X_train[col].dtype not in ['float64', 'int64']:
                continue
            if self.X_train[col].nunique() <= 2:
                continue
            numeric_cols.append(col)

        self.logger.info(f"Winsorizing {len(numeric_cols)} numeric features...")

        for col in numeric_cols:
            if self.X_train[col].isnull().all():
                continue
            lower = self.X_train[col].quantile(OUTLIER_LOW)
            upper = self.X_train[col].quantile(OUTLIER_HIGH)
            self.X_train[col] = self.X_train[col].clip(lower, upper)
            self.X_test[col] = self.X_test[col].clip(lower, upper)

        self.logger.info("✅ Outliers winsorized (train quantiles → both)")

    # ------------------------------------------------------------
    # STANDARDIZE
    # ------------------------------------------------------------

    def standardize_features(self):
        self.logger.info("\n" + "=" * 60)
        self.logger.info("FEATURE STANDARDIZATION (fit on train)")
        self.logger.info("=" * 60)

        numeric_cols = []
        for col in self.feature_cols:
            if col == 'stay_id':
                continue
            if col.endswith('_missing'):
                continue
            if self.X_train[col].dtype not in ['float64', 'int64']:
                continue
            if self.X_train[col].nunique() <= 2:
                continue
            if self.X_train[col].isnull().all():
                continue
            numeric_cols.append(col)

        self.logger.info(f"Standardizing {len(numeric_cols)} numeric features...")

        if numeric_cols:
            self.scaler = StandardScaler()
            self.X_train[numeric_cols] = self.scaler.fit_transform(
                self.X_train[numeric_cols])
            self.X_test[numeric_cols] = self.scaler.transform(
                self.X_test[numeric_cols])
            pd.DataFrame({
                'feature': numeric_cols,
                'mean': self.scaler.mean_,
                'scale': self.scaler.scale_,
            }).to_csv(TABLE_DIR / 'scaler_params.csv', index=False)
            self.logger.info(f"✅ Scaler params: {TABLE_DIR / 'scaler_params.csv'}")
        else:
            self.logger.warning("No valid numeric features to standardize")
            self.scaler = None

        self.logger.info("✅ Features standardized")

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    def save_preprocessed_data(self, output_dir=None):
        if output_dir is None:
            output_dir = TABLE_DIR

        train_df = self.X_train.copy()
        train_df['outcome'] = self.y_train.values
        train_df.to_csv(f"{output_dir}/X_train.csv", index=False)

        test_df = self.X_test.copy()
        test_df['outcome'] = self.y_test.values
        test_df.to_csv(f"{output_dir}/X_test.csv", index=False)

        self.logger.info(f"\n✅ Saved to: {output_dir}")
        self.logger.info(f"  X_train: {train_df.shape}")
        self.logger.info(f"  X_test:  {test_df.shape}")

        mask_count = sum(1 for c in self.feature_cols if c.endswith('_missing'))
        numeric_count = len(self.feature_cols) - mask_count - 1
        temporal_count = sum(1 for c in self.feature_cols
                             if any(c.endswith(s) for s in
                                    ['_slope', '_delta', '_range',
                                     '_time_since_last_measure']))
        sofa_retained = [f for f in SOFA_RELEVANT_FEATURES
                         if f in self.feature_cols]

        self.logger.info(f"\n📊 Feature Summary:")
        self.logger.info(f"  Total columns (excl. outcome): {len(self.feature_cols)}")
        self.logger.info(f"    stay_id: 1")
        self.logger.info(f"    Numeric features: {numeric_count}")
        self.logger.info(f"    Of which temporal: {temporal_count}")
        self.logger.info(f"    Missingness masks: {mask_count}")
        self.logger.info(f"  SOFA-relevant retained: "
                         f"{len(sofa_retained)}/{len(SOFA_RELEVANT_FEATURES)}")

        with open(f"{output_dir}/feature_cols.txt", 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("FEATURE LIST (After Phase 4)\n")
            f.write("=" * 60 + "\n")
            f.write(f"Total columns (excl. outcome): {len(self.feature_cols)}\n")
            f.write(f"  stay_id: 1\n")
            f.write(f"  Numeric features: {numeric_count}\n")
            f.write(f"    Of which temporal: {temporal_count}\n")
            f.write(f"  Missingness masks: {mask_count}\n")
            f.write(f"SOFA-relevant retained: "
                    f"{len(sofa_retained)}/{len(SOFA_RELEVANT_FEATURES)}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"TARGET: Pure ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD}\n")
            f.write("=" * 60 + "\n\n")
            for col in self.feature_cols:
                f.write(f"{col}\n")

        self.logger.info(f"  Feature list: {output_dir}/feature_cols.txt")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Phase 4: Feature Filtering and Preprocessing'
    )
    parser.add_argument('--features', type=str, default=None,
                        help='Path to features CSV '
                             '(default: features_0_6h_none.csv)')
    parser.add_argument('--cohort', type=str, default=None,
                        help='(Deprecated; kept for compatibility)')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--no-split', action='store_true')

    args = parser.parse_args()

    logger = setup_logging()
    print_header()

    features_path = args.features or TABLE_DIR / "features_0_6h_none.csv"

    if not Path(features_path).exists():
        logger.error(f"Features file not found: {features_path}")
        logger.error("Run Phase 3 first: python3 phase3.py --method none")
        return 1

    try:
        preprocessor = FeaturePreprocessor(logger)
        preprocessor.load_features(features_path)   # no cohort merge

        if args.no_split:
            logger.info("--no-split given — skipping split and preprocessing")
            return 0

        if not preprocessor.split_data():
            return 1

        preprocessor.characterize_missingness()
        preprocessor.winsorize_outliers()
        preprocessor.standardize_features()
        preprocessor.save_preprocessed_data(args.output)

        print("\n" + "=" * 80)
        print(" PHASE 4 COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f" ✅ Total columns (incl. stay_id): {len(preprocessor.feature_cols)}")
        print(f" ✅ Training samples: {len(preprocessor.X_train):,}")
        print(f" ✅ Test samples: {len(preprocessor.X_test):,}")
        print("-" * 80)
        print(" ⚠️  NO DATA LEAKAGE:")
        print(f"   • Features from 0-{FEATURE_WINDOW_HOURS}h ONLY")
        print(f"   • Missingness characterized on TRAIN only")
        print(f"   • Winsorization and scaling fit on TRAIN only")
        print("-" * 80)
        print(" ⚠️  MISSING VALUES KEPT for Phase 5 & 6")
        print("=" * 80)

        print("\n📝 Next Steps:")
        print("  1. Run Phase 5: Missingness Characterization")
        print("  2. Run Phase 6: Imputation × Model Interaction")
        print()

        return 0

    except Exception as e:
        logger.error(f"Error in Phase 4: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
