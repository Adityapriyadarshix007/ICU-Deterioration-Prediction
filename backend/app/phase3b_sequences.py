#!/usr/bin/env python3
"""
Phase 3b: Sequence Extraction for CNN-LSTM

PURPOSE
-------
Build time-series sequences (T=12 bins over 0-6h) for the CNN-LSTM model.
Complements the tabular pipeline (Phase 3) with sequential features.

FEATURES (20)
-------------
    Vitals (5):       heart_rate, respiratory_rate, spo2, map, gcs_total
    Respiratory (2):  fio2, pao2
    Labs (10):        creatinine, lactate, bilirubin, platelets, wbc,
                      hemoglobin, sodium, potassium, bun, glucose
    Vasopressors (3): norepinephrine, epinephrine, dopamine

DESIGN
------
1. Time bins: 30-min intervals over 0-6h from ICU intime -> T=12.
2. Per (patient, feature, bin): mean of all measurements in that bin.
3. GCS total: sum of eyes+verbal+motor when all 3 present in same 2-min
   bucket (from Step 0 logic). Stored as a single feature.
4. Tensor shape: (n_patients, 12, 40).
     - Channels 0-19  : values (normalized)
     - Channels 20-39 : masks (1 if observed, 0 if missing)
5. Cleaning pipeline:
     a. Apply physiological bounds -> drop out-of-range values.
     b. Convert FiO2 from % to fraction.
     c. Winsorize at 1%/99% (thresholds from train, applied to both).
     d. Z-score using train mean/std only.
6. Missing handling:
     - Observed bin: (normalized_value, mask=1)
     - Missing bin:  (0.0, mask=0)
7. Split: same stay_ids as Phase 4 (X_train.csv, X_test.csv).

FIXES vs previous version
-------------------------
1. GCS_ITEM_IDS_COMPONENT moved above FEATURES (was NameError).
2. Vasopressor rate fallback uses amount/duration when rate is NaN.
3. Input events row-filtered before bin loop.
4. Lab events deduplicated on (stay_id, charttime, itemid).
5. Physiological bounds filter (removes extreme outliers).
6. FiO2 % -> fraction conversion.
7. Winsorization at 1%/99% (train thresholds applied to test).
8. Consistent cleaning for both train and test.

OUTPUT
------
    outputs/sequences/sequences_train.npz   (X, stay_ids, y)
    outputs/sequences/sequences_test.npz    (X, stay_ids, y)
    outputs/tables/phase3b_feature_stats.csv
    outputs/tables/phase3b_coverage.csv
    outputs/tables/phase3b_report.txt

RUNTIME
-------
    ~5-10 minutes
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TABLE_DIR, LOGS_DIR, DB_NAME,
    HR_ITEMID, MAP_ITEMIDS,
    RESPIRATORY_ITEM_IDS,
    GCS_ITEM_IDS,
    VASOPRESSORS, LAB_ITEM_IDS,
)
from database import DatabaseManager


# ============================================================
# CONFIG
# ============================================================

WINDOW_START_H = 0
WINDOW_END_H = 6
BIN_MINUTES = 30
N_BINS = int((WINDOW_END_H - WINDOW_START_H) * 60 / BIN_MINUTES)  # = 12

GCS_ROUND_MIN = 2

OUTPUT_DIR = Path(__file__).parent / "outputs" / "sequences"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PHYSIOLOGICAL BOUNDS
# ============================================================

# Values outside these ranges are dropped as data-entry errors.
PHYSIOLOGICAL_BOUNDS = {
    'heart_rate':       (20, 250),
    'respiratory_rate': (5, 60),
    'spo2':             (50, 100),
    'map':              (20, 200),
    'gcs_total':        (3, 15),
    'fio2':             (21, 100),    # percentage; converted to fraction later
    'pao2':             (30, 700),
    'creatinine':       (0.1, 30),
    'lactate':          (0.1, 40),
    'bilirubin':        (0.1, 60),
    'platelets':        (1, 1500),
    'wbc':              (0.1, 200),
    'hemoglobin':       (3, 25),
    'sodium':           (90, 200),
    'potassium':        (1.5, 10),
    'bun':              (1, 300),
    'glucose':          (20, 1500),
    'norepinephrine':   (0, 5),
    'epinephrine':      (0, 5),
    'dopamine':         (0, 50),
}

# Features requiring % -> fraction conversion
FIO2_FEATURES = {'fio2'}

# Winsorization percentiles (train only)
WINSOR_LOW = 1.0
WINSOR_HIGH = 99.0


# ============================================================
# GCS COMPONENT IDS
# ============================================================

GCS_ITEM_IDS_COMPONENT = {
    'eyes':   GCS_ITEM_IDS['gcs_eyes'],
    'verbal': GCS_ITEM_IDS['gcs_verbal'],
    'motor':  GCS_ITEM_IDS['gcs_motor'],
}


# ============================================================
# FEATURE REGISTRY
# ============================================================

FEATURES = {
    # ---- Vitals ----
    'heart_rate':        {'itemids': [HR_ITEMID], 'source': 'chartevents', 'kind': 'single'},
    'respiratory_rate':  {'itemids': [RESPIRATORY_ITEM_IDS['rr']], 'source': 'chartevents', 'kind': 'single'},
    'spo2':              {'itemids': [RESPIRATORY_ITEM_IDS['spo2']], 'source': 'chartevents', 'kind': 'single'},
    'map':               {'itemids': list(MAP_ITEMIDS), 'source': 'chartevents', 'kind': 'mean_of_multiple'},
    'gcs_total':         {'itemids': [
                             GCS_ITEM_IDS_COMPONENT['eyes'],
                             GCS_ITEM_IDS_COMPONENT['verbal'],
                             GCS_ITEM_IDS_COMPONENT['motor'],
                         ], 'source': 'chartevents', 'kind': 'gcs_components'},

    # ---- Respiratory ----
    'fio2':              {'itemids': [RESPIRATORY_ITEM_IDS['fio2']], 'source': 'chartevents', 'kind': 'single'},
    'pao2':              {'itemids': [RESPIRATORY_ITEM_IDS['pao2']], 'source': 'chartevents', 'kind': 'single'},

    # ---- Labs ----
    'creatinine':        {'itemids': [LAB_ITEM_IDS['creatinine']], 'source': 'labevents', 'kind': 'single'},
    'lactate':           {'itemids': [LAB_ITEM_IDS['lactate']], 'source': 'labevents', 'kind': 'single'},
    'bilirubin':         {'itemids': [LAB_ITEM_IDS['bilirubin']], 'source': 'labevents', 'kind': 'single'},
    'platelets':         {'itemids': [LAB_ITEM_IDS['platelets']], 'source': 'labevents', 'kind': 'single'},
    'wbc':               {'itemids': [LAB_ITEM_IDS['wbc']], 'source': 'labevents', 'kind': 'single'},
    'hemoglobin':        {'itemids': [LAB_ITEM_IDS['hemoglobin']], 'source': 'labevents', 'kind': 'single'},
    'sodium':            {'itemids': [LAB_ITEM_IDS['sodium']], 'source': 'labevents', 'kind': 'single'},
    'potassium':         {'itemids': [LAB_ITEM_IDS['potassium']], 'source': 'labevents', 'kind': 'single'},
    'bun':               {'itemids': [LAB_ITEM_IDS['bun']], 'source': 'labevents', 'kind': 'single'},
    'glucose':           {'itemids': [LAB_ITEM_IDS['glucose']], 'source': 'labevents', 'kind': 'single'},

    # ---- Vasopressors ----
    'norepinephrine':    {'itemids': [VASOPRESSORS['norepinephrine']], 'source': 'inputevents', 'kind': 'vaso_rate'},
    'epinephrine':       {'itemids': [VASOPRESSORS['epinephrine']], 'source': 'inputevents', 'kind': 'vaso_rate'},
    'dopamine':          {'itemids': [VASOPRESSORS['dopamine']], 'source': 'inputevents', 'kind': 'vaso_rate'},
}

FEATURE_NAMES = list(FEATURES.keys())
N_FEATURES = len(FEATURE_NAMES)  # 20
N_CHANNELS = 2 * N_FEATURES      # 40


# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"phase3b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    import logging
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
    print(" PHASE 3b: SEQUENCE EXTRACTION FOR CNN-LSTM")
    print("=" * 80)
    print(f" Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    print(f" Window: 0-{WINDOW_END_H}h from ICU intime")
    print(f" Bins: {BIN_MINUTES} min | T = {N_BINS}")
    print(f" Features: {N_FEATURES}")
    print(f" Channels: {N_CHANNELS} (values + masks)")
    print(f" Tensor shape: (n_patients, {N_BINS}, {N_CHANNELS})")
    print(f" Cleaning: bounds + winsorize (1/99) + FiO2 conversion")
    print("=" * 80)
    print()


# ============================================================
# COHORT
# ============================================================

def load_cohort_split(logger):
    logger.info("\n" + "=" * 60)
    logger.info("LOADING COHORT SPLIT FROM PHASE 4")
    logger.info("=" * 60)

    train = pd.read_csv(TABLE_DIR / 'X_train.csv',
                        usecols=['stay_id', 'outcome'])
    test = pd.read_csv(TABLE_DIR / 'X_test.csv',
                       usecols=['stay_id', 'outcome'])

    logger.info(f"  Train: {len(train):,}  "
                f"(progressors={int(train['outcome'].sum()):,})")
    logger.info(f"  Test:  {len(test):,}  "
                f"(progressors={int(test['outcome'].sum()):,})")

    return train, test


# ============================================================
# EVENT EXTRACTION
# ============================================================

def query_chart_events(logger, stay_ids):
    itemids = set()
    for feat, spec in FEATURES.items():
        if spec['source'] == 'chartevents':
            itemids.update(spec['itemids'])
    itemids = sorted(itemids)

    stay_id_list = ','.join(map(str, stay_ids))
    id_list = ','.join(map(str, itemids))

    db = DatabaseManager(db_path=DB_NAME)
    db.create_views()

    query = f"""
    WITH target_stays AS (
        SELECT stay_id, intime
        FROM icustays
        WHERE stay_id IN ({stay_id_list})
    )
    SELECT
        ce.stay_id,
        ce.charttime,
        ce.itemid,
        ce.valuenum,
        ts.intime
    FROM chartevents ce
    INNER JOIN target_stays ts ON ce.stay_id = ts.stay_id
    WHERE ce.itemid IN ({id_list})
      AND ce.valuenum IS NOT NULL
      AND ce.charttime >= ts.intime
      AND ce.charttime <  ts.intime + INTERVAL '{WINDOW_END_H} HOURS'
    """
    logger.info(f"  Querying chartevents ({len(itemids)} itemids)...")
    df = db.fetch_df(query)
    db.close()
    logger.info(f"    Chartevents: {len(df):,} rows")
    return df


def query_lab_events(logger, stay_ids):
    itemids = set()
    for feat, spec in FEATURES.items():
        if spec['source'] == 'labevents':
            itemids.update(spec['itemids'])
    itemids = sorted(itemids)

    stay_id_list = ','.join(map(str, stay_ids))
    id_list = ','.join(map(str, itemids))

    db = DatabaseManager(db_path=DB_NAME)
    db.create_views()

    query = f"""
    WITH target_stays AS (
        SELECT stay_id, hadm_id, intime
        FROM icustays
        WHERE stay_id IN ({stay_id_list})
    )
    SELECT
        ts.stay_id,
        le.charttime,
        le.itemid,
        le.valuenum,
        ts.intime
    FROM labevents le
    INNER JOIN target_stays ts ON le.hadm_id = ts.hadm_id
    WHERE le.itemid IN ({id_list})
      AND le.valuenum IS NOT NULL
      AND le.charttime >= ts.intime
      AND le.charttime <  ts.intime + INTERVAL '{WINDOW_END_H} HOURS'
    """
    logger.info(f"  Querying labevents ({len(itemids)} itemids)...")
    df = db.fetch_df(query)
    db.close()
    logger.info(f"    Labevents (raw): {len(df):,} rows")

    df = df.drop_duplicates(subset=['stay_id', 'charttime', 'itemid'])
    logger.info(f"    Labevents (deduped): {len(df):,} rows")
    return df


def query_input_events(logger, stay_ids):
    itemids = set()
    for feat, spec in FEATURES.items():
        if spec['source'] == 'inputevents':
            itemids.update(spec['itemids'])
    itemids = sorted(itemids)

    stay_id_list = ','.join(map(str, stay_ids))
    id_list = ','.join(map(str, itemids))

    db = DatabaseManager(db_path=DB_NAME)
    db.create_views()

    query = f"""
    WITH target_stays AS (
        SELECT stay_id, intime
        FROM icustays
        WHERE stay_id IN ({stay_id_list})
    )
    SELECT
        ie.stay_id,
        ie.starttime,
        ie.endtime,
        ie.itemid,
        ie.rate,
        ie.amount,
        ts.intime
    FROM inputevents ie
    INNER JOIN target_stays ts ON ie.stay_id = ts.stay_id
    WHERE ie.itemid IN ({id_list})
      AND (ie.rate IS NOT NULL OR ie.amount IS NOT NULL)
      AND ie.starttime >= ts.intime
      AND ie.starttime <  ts.intime + INTERVAL '{WINDOW_END_H} HOURS'
    """
    logger.info(f"  Querying inputevents ({len(itemids)} itemids)...")
    df = db.fetch_df(query)
    db.close()
    logger.info(f"    Inputevents: {len(df):,} rows")
    return df


# ============================================================
# BINNING
# ============================================================

def assign_bin(charttime, intime):
    hours = (charttime - intime).dt.total_seconds() / 3600
    return np.floor(hours * 60 / BIN_MINUTES).astype(int)


def bin_chart_events(df, logger):
    logger.info("\n  Binning chart events...")

    itemid_to_feat = {}
    for feat, spec in FEATURES.items():
        if spec['source'] != 'chartevents':
            continue
        for i in spec['itemids']:
            itemid_to_feat[i] = feat

    df = df.copy()
    df['charttime'] = pd.to_datetime(df['charttime'])
    df['intime'] = pd.to_datetime(df['intime'])
    df['feat'] = df['itemid'].map(itemid_to_feat)
    df = df.dropna(subset=['feat'])

    gcs_ids = set(GCS_ITEM_IDS_COMPONENT.values())
    is_gcs = df['itemid'].isin(gcs_ids)

    gcs_df = df[is_gcs].copy()
    other_df = df[~is_gcs].copy()

    # --- Non-GCS ---
    other_df = other_df.copy()
    other_df['bin_idx'] = assign_bin(other_df['charttime'], other_df['intime'])
    other_df = other_df[
        (other_df['bin_idx'] >= 0) & (other_df['bin_idx'] < N_BINS)
    ]
    other_agg = (other_df
                 .groupby(['stay_id', 'feat', 'bin_idx'], as_index=False)
                 ['valuenum'].mean())

    # --- GCS ---
    gcs_valid = {
        GCS_ITEM_IDS_COMPONENT['eyes']:   (1, 4),
        GCS_ITEM_IDS_COMPONENT['verbal']: (1, 5),
        GCS_ITEM_IDS_COMPONENT['motor']:  (1, 6),
    }
    gcs_df = gcs_df.copy()
    gcs_df['valid'] = gcs_df.apply(
        lambda r: gcs_valid.get(r['itemid'], (0, -1))[0]
                  <= r['valuenum']
                  <= gcs_valid.get(r['itemid'], (0, -1))[1],
        axis=1
    )
    gcs_df = gcs_df[gcs_df['valid']].drop(columns='valid')
    gcs_df['charttime_rounded'] = (
        gcs_df['charttime'].dt.round(f'{GCS_ROUND_MIN}min')
    )
    gcs_df = gcs_df.sort_values('charttime')
    gcs_dedup = (gcs_df
                 .groupby(['stay_id', 'charttime_rounded', 'itemid', 'intime'],
                          as_index=False)
                 .last())
    gcs_sum = (gcs_dedup
               .groupby(['stay_id', 'charttime_rounded', 'intime'])
               .agg(value=('valuenum', 'sum'),
                    n_components=('itemid', 'nunique'))
               .reset_index())
    gcs_sum = gcs_sum[gcs_sum['n_components'] == 3].copy()
    gcs_sum = gcs_sum.drop(columns='n_components')
    gcs_sum['bin_idx'] = assign_bin(gcs_sum['charttime_rounded'],
                                     gcs_sum['intime'])
    gcs_sum = gcs_sum[
        (gcs_sum['bin_idx'] >= 0) & (gcs_sum['bin_idx'] < N_BINS)
    ].copy()
    gcs_sum['feat'] = 'gcs_total'
    gcs_sum = gcs_sum.rename(columns={'value': 'valuenum'})
    gcs_agg = gcs_sum[['stay_id', 'feat', 'bin_idx', 'valuenum']].groupby(
        ['stay_id', 'feat', 'bin_idx'], as_index=False
    )['valuenum'].mean()

    logger.info(f"    Chart bins (non-GCS): {len(other_agg):,}")
    logger.info(f"    Chart bins (GCS):     {len(gcs_agg):,}")

    combined = pd.concat([other_agg, gcs_agg], ignore_index=True)
    logger.info(f"    Total chart bins:     {len(combined):,}")
    return combined


def bin_labs(df, logger):
    logger.info("\n  Binning lab events...")

    itemid_to_feat = {}
    for feat, spec in FEATURES.items():
        if spec['source'] != 'labevents':
            continue
        for i in spec['itemids']:
            itemid_to_feat[i] = feat

    df = df.copy()
    df['charttime'] = pd.to_datetime(df['charttime'])
    df['intime'] = pd.to_datetime(df['intime'])
    df['feat'] = df['itemid'].map(itemid_to_feat)
    df = df.dropna(subset=['feat'])
    df['bin_idx'] = assign_bin(df['charttime'], df['intime'])
    df = df[(df['bin_idx'] >= 0) & (df['bin_idx'] < N_BINS)]

    agg = (df
           .groupby(['stay_id', 'feat', 'bin_idx'], as_index=False)
           ['valuenum'].mean())
    logger.info(f"    Lab bins: {len(agg):,}")
    return agg


def bin_input_events(df, logger):
    logger.info("\n  Binning vasopressor input events...")

    itemid_to_feat = {}
    for feat, spec in FEATURES.items():
        if spec['source'] != 'inputevents':
            continue
        for i in spec['itemids']:
            itemid_to_feat[i] = feat

    df = df.copy()
    df['starttime'] = pd.to_datetime(df['starttime'])
    df['endtime'] = pd.to_datetime(df['endtime'])
    df['intime'] = pd.to_datetime(df['intime'])
    df['feat'] = df['itemid'].map(itemid_to_feat)
    df = df.dropna(subset=['feat'])

    window_end = df['intime'] + pd.Timedelta(hours=WINDOW_END_H)
    df = df[df['starttime'] < window_end].copy()

    rows = []
    for _, r in df.iterrows():
        start_bin = int(np.floor(
            (r['starttime'] - r['intime']).total_seconds() / 3600
            * 60 / BIN_MINUTES))
        end_bin = int(np.floor(
            (r['endtime'] - r['intime']).total_seconds() / 3600
            * 60 / BIN_MINUTES))

        if pd.notna(r['rate']):
            val = float(r['rate'])
        elif pd.notna(r['amount']):
            duration_h = max(
                (r['endtime'] - r['starttime']).total_seconds() / 3600,
                0.1,
            )
            val = float(r['amount']) / duration_h
        else:
            val = 1.0

        for b in range(max(start_bin, 0), min(end_bin + 1, N_BINS)):
            rows.append({
                'stay_id': r['stay_id'],
                'feat': r['feat'],
                'bin_idx': b,
                'valuenum': val,
            })

    if not rows:
        logger.info("    Vasopressor bins: 0")
        return pd.DataFrame(columns=['stay_id', 'feat', 'bin_idx', 'valuenum'])

    vdf = pd.DataFrame(rows)
    agg = (vdf
           .groupby(['stay_id', 'feat', 'bin_idx'], as_index=False)
           ['valuenum'].max())
    logger.info(f"    Vasopressor bins: {len(agg):,}")
    return agg


# ============================================================
# CLEANING (PATCH: bounds + FiO2 conversion)
# ============================================================

def apply_bounds_and_convert(binned_df, logger):
    """
    Apply physiological bounds and FiO2 conversion to binned values.
    Operates on the combined binned DataFrame BEFORE tensor assembly.
    """
    logger.info("\n" + "=" * 60)
    logger.info("APPLYING PHYSIOLOGICAL BOUNDS AND FIO2 CONVERSION")
    logger.info("=" * 60)

    binned_df = binned_df.copy()
    before = len(binned_df)

    # Physiological bounds
    total_dropped = 0
    for feat, (lo, hi) in PHYSIOLOGICAL_BOUNDS.items():
        m = binned_df['feat'] == feat
        n_feat = int(m.sum())
        if n_feat == 0:
            continue
        in_bounds = (binned_df.loc[m, 'valuenum'] >= lo) & \
                    (binned_df.loc[m, 'valuenum'] <= hi)
        n_dropped = int((~in_bounds).sum())
        if n_dropped > 0:
            logger.info(f"    {feat:20s}: dropped {n_dropped:>7,} / "
                        f"{n_feat:>9,} out-of-bounds rows "
                        f"({100*n_dropped/n_feat:.2f}%)")
        total_dropped += n_dropped
        binned_df.loc[m & ~in_bounds, 'valuenum'] = np.nan

    binned_df = binned_df.dropna(subset=['valuenum']).reset_index(drop=True)
    after = len(binned_df)
    logger.info(f"\n    Total dropped: {total_dropped:,} / {before:,} "
                f"({100*total_dropped/before:.2f}%)")
    logger.info(f"    Remaining rows: {after:,}")

    # FiO2 conversion: % -> fraction
    for feat in FIO2_FEATURES:
        m = binned_df['feat'] == feat
        n_fio2 = int(m.sum())
        if n_fio2 > 0:
            binned_df.loc[m, 'valuenum'] = binned_df.loc[m, 'valuenum'] / 100.0
            logger.info(f"    Converted {n_fio2:,} {feat} rows "
                        f"from % to fraction")

    return binned_df


# ============================================================
# TENSOR ASSEMBLY
# ============================================================

def build_tensors(binned, split_df, logger, tag):
    stay_ids = split_df['stay_id'].values
    n = len(stay_ids)
    sid_to_row = {sid: i for i, sid in enumerate(stay_ids)}

    values = np.zeros((n, N_BINS, N_FEATURES), dtype=np.float32)
    masks = np.zeros((n, N_BINS, N_FEATURES), dtype=np.float32)

    feat_to_col = {f: i for i, f in enumerate(FEATURE_NAMES)}

    for _, r in binned.iterrows():
        i = sid_to_row.get(r['stay_id'])
        if i is None:
            continue
        j = int(r['bin_idx'])
        k = feat_to_col.get(r['feat'])
        if k is None:
            continue
        if 0 <= j < N_BINS:
            values[i, j, k] = r['valuenum']
            masks[i, j, k] = 1.0

    logger.info(f"  [{tag}] Tensor shape: {values.shape}")
    logger.info(f"  [{tag}] Observed cells: {int(masks.sum()):,} / "
                f"{masks.size:,} ({100*masks.mean():.1f}%)")

    return values, masks


# ============================================================
# NORMALIZATION (PATCH: winsorization with train-only thresholds)
# ============================================================

def compute_feature_stats(values_train, masks_train, logger):
    """
    Compute per-feature mean/std using observed values in TRAINING.
    Also returns winsorization thresholds (1%, 99%) per feature.
    Applies winsorization IN PLACE to values_train.
    """
    logger.info("\n" + "=" * 60)
    logger.info("COMPUTING NORMALIZATION STATS (TRAIN ONLY)")
    logger.info("=" * 60)
    logger.info(f"  Winsorization: [{WINSOR_LOW}%, {WINSOR_HIGH}%]")

    means = np.zeros(N_FEATURES, dtype=np.float32)
    stds = np.ones(N_FEATURES, dtype=np.float32)
    counts = np.zeros(N_FEATURES, dtype=int)
    lo_bounds = np.zeros(N_FEATURES, dtype=np.float32)
    hi_bounds = np.zeros(N_FEATURES, dtype=np.float32)

    for k, feat in enumerate(FEATURE_NAMES):
        obs = masks_train[:, :, k] == 1
        vals = values_train[:, :, k][obs]

        if len(vals) < 2:
            means[k] = 0.0
            stds[k] = 1.0
            counts[k] = len(vals)
            lo_bounds[k] = 0.0
            hi_bounds[k] = 0.0
            continue

        # Winsorization thresholds from train
        p1 = float(np.percentile(vals, WINSOR_LOW))
        p99 = float(np.percentile(vals, WINSOR_HIGH))
        lo_bounds[k] = p1
        hi_bounds[k] = p99

        # Apply winsorization IN PLACE to train tensor
        clipped_flat = np.clip(values_train[:, :, k], p1, p99)
        values_train[:, :, k] = np.where(obs, clipped_flat,
                                          values_train[:, :, k])

        vals_clipped = values_train[:, :, k][obs]
        means[k] = float(vals_clipped.mean())
        s = float(vals_clipped.std())
        stds[k] = s if s > 1e-6 else 1.0
        counts[k] = len(vals_clipped)

    return means, stds, counts, lo_bounds, hi_bounds


def apply_winsor_to_test(values_test, masks_test, lo_bounds, hi_bounds):
    """Apply winsorization to test using TRAIN-derived thresholds."""
    for k in range(N_FEATURES):
        obs = masks_test[:, :, k] == 1
        clipped = np.clip(values_test[:, :, k], lo_bounds[k], hi_bounds[k])
        values_test[:, :, k] = np.where(obs, clipped, values_test[:, :, k])
    return values_test


def normalize(values, masks, means, stds):
    """Apply per-feature z-score to observed cells only."""
    out = np.zeros_like(values)
    for k in range(N_FEATURES):
        m = masks[:, :, k] == 1
        out[:, :, k][m] = (values[:, :, k][m] - means[k]) / stds[k]
    return out


# ============================================================
# MAIN
# ============================================================

def main():
    logger = setup_logging()
    print_header()

    t0 = datetime.now()

    # ---------- Cohort split ----------
    train_df, test_df = load_cohort_split(logger)
    all_ids = np.concatenate([train_df['stay_id'].values,
                              test_df['stay_id'].values])

    # ---------- Query raw events ----------
    logger.info("\n" + "=" * 60)
    logger.info("QUERYING RAW EVENTS")
    logger.info("=" * 60)

    chart_df = query_chart_events(logger, all_ids)
    lab_df = query_lab_events(logger, all_ids)
    input_df = query_input_events(logger, all_ids)

    # ---------- Bin ----------
    logger.info("\n" + "=" * 60)
    logger.info("BINNING INTO 30-MIN INTERVALS")
    logger.info("=" * 60)

    chart_binned = bin_chart_events(chart_df, logger)
    lab_binned = bin_labs(lab_df, logger)
    input_binned = bin_input_events(input_df, logger)

    all_binned = pd.concat([chart_binned, lab_binned, input_binned],
                            ignore_index=True)
    logger.info(f"\n  Total binned rows (pre-clean): {len(all_binned):,}")

    # ---------- Clean (bounds + FiO2) ----------
    all_binned = apply_bounds_and_convert(all_binned, logger)
    logger.info(f"\n  Total binned rows (post-clean): {len(all_binned):,}")

    # ---------- Build tensors ----------
    logger.info("\n" + "=" * 60)
    logger.info("BUILDING TENSORS")
    logger.info("=" * 60)

    train_values, train_masks = build_tensors(all_binned, train_df,
                                                logger, 'train')
    test_values, test_masks = build_tensors(all_binned, test_df,
                                              logger, 'test')

    # ---------- Normalization (train-only stats) ----------
    means, stds, counts, lo_bounds, hi_bounds = compute_feature_stats(
        train_values, train_masks, logger
    )

    # Apply the same winsorization thresholds to test (no leakage)
    test_values = apply_winsor_to_test(test_values, test_masks,
                                        lo_bounds, hi_bounds)

    # Save stats
    stats_df = pd.DataFrame({
        'feature': FEATURE_NAMES,
        'mean': means,
        'std': stds,
        'winsor_lo': lo_bounds,
        'winsor_hi': hi_bounds,
        'n_observed_train': counts,
    })
    stats_path = TABLE_DIR / 'phase3b_feature_stats.csv'
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"\n  Saved feature stats: {stats_path}")

    logger.info("\n  Final stats:")
    for _, r in stats_df.iterrows():
        logger.info(f"    {r['feature']:20s}  "
                    f"mean={r['mean']:>10.3f}  "
                    f"std={r['std']:>9.3f}  "
                    f"n={int(r['n_observed_train']):>8,}")

    # Normalize (z-score)
    train_values_norm = normalize(train_values, train_masks, means, stds)
    test_values_norm = normalize(test_values, test_masks, means, stds)

    # ---------- Coverage report ----------
    coverage_rows = []
    for k, feat in enumerate(FEATURE_NAMES):
        tr_cov = float(train_masks[:, :, k].mean()) * 100
        te_cov = float(test_masks[:, :, k].mean()) * 100
        coverage_rows.append({
            'feature': feat,
            'train_coverage_pct': tr_cov,
            'test_coverage_pct': te_cov,
        })
    cov_df = pd.DataFrame(coverage_rows)
    cov_path = TABLE_DIR / 'phase3b_coverage.csv'
    cov_df.to_csv(cov_path, index=False)
    logger.info(f"\n  Saved coverage report: {cov_path}")

    # ---------- Concatenate values + masks ----------
    X_train = np.concatenate([train_values_norm, train_masks], axis=2)
    X_test = np.concatenate([test_values_norm, test_masks], axis=2)

    y_train = train_df['outcome'].values.astype(np.float32)
    y_test = test_df['outcome'].values.astype(np.float32)
    stay_train = train_df['stay_id'].values
    stay_test = test_df['stay_id'].values

    logger.info(f"\n  X_train shape: {X_train.shape}")
    logger.info(f"  X_test shape:  {X_test.shape}")
    logger.info(f"  y_train: {y_train.sum():.0f} pos / {len(y_train)} total")
    logger.info(f"  y_test:  {y_test.sum():.0f} pos / {len(y_test)} total")

    # Sanity checks
    logger.info("\n  Sanity checks:")
    v_tr = X_train[:, :, :N_FEATURES]
    v_te = X_test[:, :, :N_FEATURES]
    m_tr = X_train[:, :, N_FEATURES:]
    m_te = X_test[:, :, N_FEATURES:]
    logger.info(f"    Train values: min={v_tr.min():.2f}, "
                f"max={v_tr.max():.2f}, "
                f"NaN={np.isnan(v_tr).any()}, "
                f"Inf={np.isinf(v_tr).any()}")
    logger.info(f"    Test values:  min={v_te.min():.2f}, "
                f"max={v_te.max():.2f}, "
                f"NaN={np.isnan(v_te).any()}, "
                f"Inf={np.isinf(v_te).any()}")
    logger.info(f"    Masks: unique values = "
                f"{np.unique(m_tr)}")

    # ---------- Save ----------
    train_path = OUTPUT_DIR / 'sequences_train.npz'
    test_path = OUTPUT_DIR / 'sequences_test.npz'
    np.savez_compressed(train_path,
                        X=X_train.astype(np.float32),
                        stay_ids=stay_train,
                        y=y_train)
    np.savez_compressed(test_path,
                        X=X_test.astype(np.float32),
                        stay_ids=stay_test,
                        y=y_test)
    logger.info(f"\n  Saved: {train_path}")
    logger.info(f"  Saved: {test_path}")

    # ---------- Report ----------
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("PHASE 3b: SEQUENCE EXTRACTION REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Window: 0-{WINDOW_END_H}h | Bins: {BIN_MINUTES} min "
                        f"| T = {N_BINS}")
    report_lines.append(f"Features: {N_FEATURES} | Channels: {N_CHANNELS}")
    report_lines.append(f"Tensor shape: (n, {N_BINS}, {N_CHANNELS})")
    report_lines.append("")
    report_lines.append("Cleaning pipeline:")
    report_lines.append("  1. Physiological bounds (drop out-of-range)")
    report_lines.append("  2. FiO2 % -> fraction conversion")
    report_lines.append("  3. Winsorization at 1%/99% (train thresholds)")
    report_lines.append("  4. Z-score normalization (train stats)")
    report_lines.append("")
    report_lines.append(f"Train: {len(train_df):,} patients")
    report_lines.append(f"Test:  {len(test_df):,} patients")
    report_lines.append("")
    report_lines.append("Coverage (% of bins observed):")
    report_lines.append(cov_df.to_string(index=False))
    report_lines.append("")
    report_lines.append("Normalization stats (from train observations):")
    report_lines.append(stats_df.to_string(index=False))
    report_lines.append("")
    report_lines.append("Notes:")
    report_lines.append("- Missing bin: value=0.0, mask=0.0")
    report_lines.append("- Observed bin: value=z-score, mask=1.0")
    report_lines.append("- GCS total computed only when all 3 components "
                        "present in same 2-min bucket")
    report_lines.append("- Vasopressor bins: value = max rate in bin "
                        "(or amount/duration if rate missing)")
    report_path = TABLE_DIR / 'phase3b_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    logger.info(f"\n  Report: {report_path}")

    print(f"\n Done in {datetime.now() - t0}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
