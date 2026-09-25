#!/usr/bin/env python3
"""
Step 0: Feasibility test for CNN-LSTM.

QUESTION
--------
Does the 0-6h ICU window contain temporal signal beyond what aggregates
(mean, slope, delta, range) capture?

If progressor and non-progressor trajectories diverge over time in a
way that aggregates miss, CNN-LSTM is worth building. If trajectories
are flat or parallel, aggregates suffice and CNN-LSTM adds little.

METHOD
------
1. Query raw chartevents from DuckDB (0-6h from ICU intime).
2. Bin into 30-min intervals -> T = 12 bins per patient.
3. For each patient and feature, take the mean within each bin.
4. Group by outcome (progressor vs. non-progressor).
5. Plot mean trajectories with 95% CI.
6. Compute two temporal-signal scores per feature:
     - level_separation: standardized mean difference (SMD) per bin
     - trajectory_divergence: slope of the inter-group gap over bins

FEATURES TESTED (6)
-------------------
    heart_rate, respiratory_rate, spo2, gcs_total, map, fio2

GCS HANDLING (FIXED)
--------------------
GCS total is not stored as a single itemid; it is charted as three
components (eyes, verbal, motor) at slightly different timestamps.
Summing requires:
  1. Filter to valid ranges (eyes 1-4, verbal 1-5, motor 1-6).
  2. Round charttime to 2-min buckets.
  3. Deduplicate: keep LAST value per (stay_id, rounded_ct, itemid).
  4. Require all 3 components present in the bucket before summing.
Without step 4, partial buckets inflate the "total" upward (e.g., only
eyes+motor summed = 10 instead of a valid 15).

FIXES vs previous version
-------------------------
1. GCS: valid-range filter + dedup + require n_components == 3 before
   summing (was producing totals from partial buckets).
2. hours_since_icu recomputed from rounded charttime.
3. _as_list() helper handles config values that may be list or scalar.
4. Cohort stay_ids passed into SQL (3-5x speedup).

OUTPUT
------
    outputs/figures/step0_trajectories.png
    outputs/tables/step0_temporal_signal.csv
    outputs/tables/step0_report.txt

RUNTIME
-------
    ~2-4 minutes
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TABLE_DIR, FIG_DIR, LOGS_DIR, DB_NAME,
    HR_ITEMID, MAP_ITEMIDS,
    RESPIRATORY_ITEM_IDS, GCS_ITEM_IDS,
)
from database import DatabaseManager


# ============================================================
# CONFIG
# ============================================================

WINDOW_START_H = 0
WINDOW_END_H = 6
BIN_MINUTES = 30
N_BINS = int((WINDOW_END_H - WINDOW_START_H) * 60 / BIN_MINUTES)  # = 12

# Charttime rounding for GCS component summing (minutes)
GCS_ROUND_MIN = 2

# Valid value ranges for each GCS component
GCS_VALID_RANGES = {
    GCS_ITEM_IDS['gcs_eyes']:   (1, 4),
    GCS_ITEM_IDS['gcs_verbal']: (1, 5),
    GCS_ITEM_IDS['gcs_motor']:  (1, 6),
}


def _as_list(x):
    """Normalize scalar or list/tuple to a flat list."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [i for i in x if i is not None]
    return [x]


FEATURE_ITEMIDS = {
    'heart_rate':        _as_list(HR_ITEMID),
    'respiratory_rate':  _as_list(RESPIRATORY_ITEM_IDS['rr']),
    'spo2':              _as_list(RESPIRATORY_ITEM_IDS['spo2']),
    'gcs_total':         [_as_list(GCS_ITEM_IDS['gcs_eyes'])[0],
                          _as_list(GCS_ITEM_IDS['gcs_verbal'])[0],
                          _as_list(GCS_ITEM_IDS['gcs_motor'])[0]],
    'map':               _as_list(MAP_ITEMIDS),
    'fio2':              _as_list(RESPIRATORY_ITEM_IDS['fio2']),
}

# Reverse lookup: itemid -> feature name
ITEMID_TO_FEATURE = {}
for feat, ids in FEATURE_ITEMIDS.items():
    for i in ids:
        ITEMID_TO_FEATURE[i] = feat

ALL_ITEM_IDS = sorted(ITEMID_TO_FEATURE.keys())

# GCS component itemids set
GCS_COMPONENT_IDS = set(FEATURE_ITEMIDS['gcs_total'])


# ============================================================
# COHORT + OUTCOME
# ============================================================

def load_cohort(logger):
    train = pd.read_csv(TABLE_DIR / 'X_train.csv',
                        usecols=['stay_id', 'outcome'])
    test = pd.read_csv(TABLE_DIR / 'X_test.csv',
                       usecols=['stay_id', 'outcome'])
    cohort = pd.concat([train, test], ignore_index=True)

    logger(f"  Cohort: {len(cohort):,} patients")
    logger(f"  Progressors: {int(cohort['outcome'].sum()):,}")
    logger(f"  Stable:      {int((cohort['outcome'] == 0).sum()):,}")

    return cohort


# ============================================================
# LOAD EVENTS (DuckDB)
# ============================================================

def load_events(logger, cohort):
    """
    Query raw chartevents for the 6 target features in the 0-6h window
    from ICU intime, restricted to the cohort.

    Returns DataFrame with columns:
        stay_id | charttime | intime | hours_since_icu | feat | value
    """
    logger("Connecting to DuckDB...")
    db = DatabaseManager(db_path=DB_NAME)
    db.create_views()

    # Cohort stay_ids for SQL filter
    cohort_ids = ','.join(map(str, cohort['stay_id'].unique()))
    logger(f"  Filtering to {len(cohort)} cohort stays...")

    id_list = ','.join(map(str, ALL_ITEM_IDS))
    query = f"""
    WITH target_stays AS (
        SELECT stay_id, intime
        FROM icustays
        WHERE stay_id IN ({cohort_ids})
    )
    SELECT
        ce.stay_id,
        ce.charttime,
        ce.itemid,
        ce.valuenum,
        ts.intime,
        EXTRACT(EPOCH FROM (ce.charttime - ts.intime)) / 3600.0
            AS hours_since_icu
    FROM chartevents ce
    INNER JOIN target_stays ts ON ce.stay_id = ts.stay_id
    WHERE ce.itemid IN ({id_list})
      AND ce.valuenum IS NOT NULL
      AND ce.charttime >= ts.intime
      AND ce.charttime <  ts.intime + INTERVAL '{WINDOW_END_H} HOURS'
    """
    logger(f"  Querying chartevents for {len(ALL_ITEM_IDS)} itemids...")
    df = db.fetch_df(query)
    db.close()

    logger(f"  Raw events: {len(df):,}")

    # Map itemid -> feature
    df['feat'] = df['itemid'].map(ITEMID_TO_FEATURE)

    # Datetime conversion
    df['charttime'] = pd.to_datetime(df['charttime'])
    df['intime'] = pd.to_datetime(df['intime'])

    # Split GCS vs. others
    is_gcs = df['itemid'].isin(GCS_COMPONENT_IDS)

    gcs_df = df[is_gcs].copy()
    other_df = df[~is_gcs].copy()

    # --- Non-GCS ---
    other_df = other_df.rename(columns={'valuenum': 'value'})
    other_df = other_df[['stay_id', 'charttime', 'intime',
                          'hours_since_icu', 'feat', 'value']]

    # --- GCS: valid-range filter + dedup + sum ---
    gcs_agg = _process_gcs(gcs_df, logger)

    events = pd.concat([other_df, gcs_agg], ignore_index=True)

    logger(f"  Total events: {len(events):,}")
    logger(f"  Breakdown by feature:")
    for feat, count in events['feat'].value_counts().items():
        logger(f"    {feat:20s}: {count:,}")

    # Sanity: GCS total range
    gcs_check = events[events['feat'] == 'gcs_total']['value']
    if len(gcs_check) > 0:
        logger(f"  GCS total range: [{gcs_check.min():.0f}, "
               f"{gcs_check.max():.0f}]  (expect [3, 15])")
        if gcs_check.min() < 3 or gcs_check.max() > 15:
            logger("  ⚠️  GCS range outside [3, 15] — investigate")

    return events


def _process_gcs(gcs_df, logger):
    """
    Process GCS components:
      1. Filter to valid ranges (eyes 1-4, verbal 1-5, motor 1-6).
      2. Round charttime to GCS_ROUND_MIN-minute buckets.
      3. Deduplicate: keep LAST value per (stay_id, rounded_ct, itemid).
      4. Require all 3 components present before summing.
    """
    if gcs_df.empty:
        return pd.DataFrame(columns=['stay_id', 'charttime', 'intime',
                                      'hours_since_icu', 'feat', 'value'])

    n_before = len(gcs_df)

    # 1. Valid-range filter
    def _valid(row):
        lo, hi = GCS_VALID_RANGES.get(row['itemid'], (0, -1))
        return lo <= row['valuenum'] <= hi

    gcs_df = gcs_df[gcs_df.apply(_valid, axis=1)].copy()
    n_valid = len(gcs_df)
    logger(f"  GCS: {n_before:,} raw → {n_valid:,} valid-range rows")

    # 2. Round charttime
    gcs_df['charttime_rounded'] = (
        gcs_df['charttime'].dt.round(f'{GCS_ROUND_MIN}min')
    )

    # 3. Deduplicate: keep LAST value per (stay_id, ct_rounded, itemid)
    gcs_df = gcs_df.sort_values('charttime')
    gcs_dedup = (gcs_df
                 .groupby(['stay_id', 'charttime_rounded', 'itemid',
                           'intime'], as_index=False)
                 .last())
    n_dedup = len(gcs_dedup)
    logger(f"  GCS: dedup → {n_dedup:,} rows")

    # 4. Require all 3 components before summing
    gcs_agg = (gcs_dedup
               .groupby(['stay_id', 'charttime_rounded', 'intime'])
               .agg(value=('valuenum', 'sum'),
                    n_components=('itemid', 'nunique'))
               .reset_index())
    n_before_filter = len(gcs_agg)
    gcs_agg = gcs_agg[gcs_agg['n_components'] == 3].copy()
    gcs_agg = gcs_agg.drop(columns=['n_components'])
    gcs_agg = gcs_agg.rename(columns={'charttime_rounded': 'charttime'})
    logger(f"  GCS: complete triples → {len(gcs_agg):,} rows "
           f"(dropped {n_before_filter - len(gcs_agg):,} partial buckets)")

    # Recompute hours_since_icu from rounded charttime
    gcs_agg['hours_since_icu'] = (
        (gcs_agg['charttime'] - gcs_agg['intime'])
        .dt.total_seconds() / 3600
    )
    gcs_agg['feat'] = 'gcs_total'
    gcs_agg = gcs_agg[['stay_id', 'charttime', 'intime',
                       'hours_since_icu', 'feat', 'value']]

    # Sanity check
    if len(gcs_agg) > 0:
        rng = (gcs_agg['value'].min(), gcs_agg['value'].max())
        logger(f"  GCS aggregated range: [{rng[0]:.0f}, {rng[1]:.0f}]")

    return gcs_agg


# ============================================================
# BIN INTO 30-MIN INTERVALS
# ============================================================

def bin_events(events, logger):
    """
    Assign each event to a 30-min bin (0..11) and take the mean value
    within (stay_id, feat, bin_idx).
    """
    # Defensive: ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(events['charttime']):
        events = events.copy()
        events['charttime'] = pd.to_datetime(events['charttime'])
    if not pd.api.types.is_datetime64_any_dtype(events['intime']):
        events = events.copy()
        events['intime'] = pd.to_datetime(events['intime'])

    # Ensure hours_since_icu
    if 'hours_since_icu' not in events.columns:
        events = events.copy()
        events['hours_since_icu'] = (
            events['charttime'] - events['intime']
        ).dt.total_seconds() / 3600

    # Filter to window (defensive — SQL already restricts)
    events = events[
        (events['hours_since_icu'] >= WINDOW_START_H) &
        (events['hours_since_icu'] <  WINDOW_END_H)
    ].copy()

    # Assign bin index
    events['bin_idx'] = (
        (events['hours_since_icu'] * 60) // BIN_MINUTES
    ).astype(int)

    # Aggregate: mean within (stay_id, feat, bin_idx)
    binned = (events
              .groupby(['stay_id', 'feat', 'bin_idx'], as_index=False)['value']
              .mean())

    logger(f"  Binned cells: {len(binned):,}")

    return binned


# ============================================================
# BUILD TENSORS
# ============================================================

def build_tensors(binned, cohort):
    """
    dict: feat -> array (n_patients, N_BINS).
    Row i corresponds to cohort['stay_id'].iloc[i].
    """
    stay_ids = cohort['stay_id'].values
    n = len(stay_ids)
    sid_to_row = {sid: i for i, sid in enumerate(stay_ids)}

    tensors = {feat: np.full((n, N_BINS), np.nan) for feat in FEATURE_ITEMIDS}

    for _, row in binned.iterrows():
        i = sid_to_row.get(row['stay_id'])
        if i is None:
            continue
        j = int(row['bin_idx'])
        if 0 <= j < N_BINS:
            tensors[row['feat']][i, j] = row['value']

    print("  Coverage per feature:")
    for feat, arr in tensors.items():
        obs = np.sum(~np.isnan(arr))
        tot = arr.size
        pct = 100 * obs / tot if tot > 0 else 0.0
        print(f"    {feat:20s}: {pct:5.1f}%")

    return tensors


# ============================================================
# SIGNAL SCORES
# ============================================================

def level_separation_score(arr, y):
    """Standardized mean difference per bin; aggregate mean/max/std."""
    prog = arr[y == 1]
    nonp = arr[y == 0]
    diffs = []
    for b in range(arr.shape[1]):
        a = prog[:, b]
        c = nonp[:, b]
        a = a[~np.isnan(a)]
        c = c[~np.isnan(c)]
        if len(a) < 30 or len(c) < 30:
            continue
        pooled_std = np.sqrt((a.var() + c.var()) / 2) + 1e-8
        diffs.append(abs(a.mean() - c.mean()) / pooled_std)
    if not diffs:
        return 0.0, 0.0, 0.0
    d = np.array(diffs)
    return float(d.mean()), float(d.max()), float(d.std())


def trajectory_divergence_score(arr, y):
    """
    Slope of the inter-group gap over bins.
    High absolute slope -> trajectories diverge (DL may help).
    Low slope           -> parallel trajectories (aggregates suffice).
    """
    prog = arr[y == 1]
    nonp = arr[y == 0]
    gaps = []
    bin_idx = []
    for b in range(arr.shape[1]):
        a = prog[:, b]
        c = nonp[:, b]
        a = a[~np.isnan(a)]
        c = c[~np.isnan(c)]
        if len(a) < 30 or len(c) < 30:
            continue
        gaps.append(a.mean() - c.mean())
        bin_idx.append(b)
    if len(gaps) < 3:
        return 0.0, 0.0
    gaps = np.array(gaps)
    bin_idx = np.array(bin_idx)
    slope = float(np.polyfit(bin_idx, gaps, 1)[0])
    gap_scale = float(np.std(gaps)) + 1e-8
    return slope, abs(slope) / gap_scale


# ============================================================
# PLOT
# ============================================================

def plot_trajectories(tensors, y, out_path):
    feats = list(FEATURE_ITEMIDS.keys())
    n_feats = len(feats)

    fig, axes = plt.subplots(n_feats, 1,
                              figsize=(9, 2.8 * n_feats),
                              sharex=True)
    if n_feats == 1:
        axes = [axes]

    bin_centers_min = (np.arange(N_BINS) + 0.5) * BIN_MINUTES

    for ax, feat in zip(axes, feats):
        arr = tensors[feat]
        for group_val, color, label in [(0, 'steelblue', 'Stable'),
                                         (1, 'crimson',   'Progressor')]:
            sub = arr[y == group_val]
            if len(sub) == 0:
                continue
            counts = np.sum(~np.isnan(sub), axis=0)
            mean = np.nanmean(sub, axis=0)
            sem = np.nanstd(sub, axis=0) / np.sqrt(np.maximum(counts, 1))
            ax.plot(bin_centers_min, mean, color=color, label=label,
                    linewidth=2, marker='o', markersize=4)
            ax.fill_between(bin_centers_min,
                            mean - 1.96 * sem,
                            mean + 1.96 * sem,
                            color=color, alpha=0.15)

        ax.set_ylabel(feat, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=8)

    axes[-1].set_xlabel('Time since ICU admission (minutes)', fontsize=10)
    fig.suptitle('Per-30min Trajectories: Progressors vs. Stable (0-6h)',
                 fontsize=13, y=1.002)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = datetime.now()
    print("=" * 70)
    print(" STEP 0: FEASIBILITY TEST FOR CNN-LSTM")
    print("=" * 70)
    print(f" Bins: {BIN_MINUTES} min | T = {N_BINS} | Window: 0-6h")
    print(f" Features: {list(FEATURE_ITEMIDS.keys())}")
    print("-" * 70)

    # 1. Cohort
    print("\n[1/6] Loading cohort...")
    cohort = load_cohort(print)

    # 2. Events
    print("\n[2/6] Loading raw events from DuckDB...")
    events = load_events(print, cohort)

    # 3. Bin
    print("\n[3/6] Binning into 30-min intervals...")
    binned = bin_events(events, print)

    # 4. Tensors
    print("\n[4/6] Building per-patient tensors...")
    tensors = build_tensors(binned, cohort)

    y = cohort['outcome'].values

    # 5. Signal scores
    print("\n[5/6] Computing temporal signal scores...")
    rows = []
    for feat, arr in tensors.items():
        mean_s, max_s, std_s = level_separation_score(arr, y)
        slope_raw, slope_norm = trajectory_divergence_score(arr, y)
        rows.append({
            'feature': feat,
            'level_sep_mean': mean_s,
            'level_sep_max': max_s,
            'level_sep_std': std_s,
            'traj_slope_raw': slope_raw,
            'traj_slope_norm': slope_norm,
        })
        print(f"    {feat:20s}  "
              f"level_sep_mean={mean_s:.4f}  "
              f"traj_slope_norm={slope_norm:.4f}")

    signal_df = pd.DataFrame(rows).sort_values('level_sep_mean',
                                                ascending=False)
    signal_df.to_csv(TABLE_DIR / 'step0_temporal_signal.csv', index=False)

    # 6. Plot
    print("\n[6/6] Plotting trajectories...")
    out_fig = FIG_DIR / 'step0_trajectories.png'
    plot_trajectories(tensors, y, out_fig)
    print(f"    Figure: {out_fig}")

    # Verdict
    max_level = signal_df['level_sep_mean'].max()
    max_traj = signal_df['traj_slope_norm'].max()

    if max_level > 0.30 or max_traj > 0.10:
        verdict = ("STRONG temporal signal — CNN-LSTM is worth building.")
        decision = "build_dl"
    elif max_level > 0.15 or max_traj > 0.05:
        verdict = ("MODERATE temporal signal — CNN-LSTM may help, "
                   "but aggregates may already capture most of it.")
        decision = "consider_dl"
    else:
        verdict = ("WEAK temporal signal — aggregates likely suffice.")
        decision = "skip_dl"

    print("\n" + "=" * 70)
    print(f" Max level separation:  {max_level:.4f}")
    print(f" Max trajectory slope:   {max_traj:.4f}")
    print(f" VERDICT: {verdict}")
    print("=" * 70)

    # Report
    report_path = TABLE_DIR / 'step0_report.txt'
    with open(report_path, 'w') as f:
        f.write("Step 0: Feasibility Test for CNN-LSTM\n")
        f.write("=" * 60 + "\n")
        f.write(f"Bins: {BIN_MINUTES} min | T = {N_BINS} | Window: 0-6h\n")
        f.write(f"Cohort: {len(cohort):,} patients\n")
        f.write(f"  Progressors: {int(cohort['outcome'].sum()):,}\n")
        f.write(f"  Stable:      {int((cohort['outcome']==0).sum()):,}\n\n")
        f.write("Signal scores per feature:\n")
        f.write(signal_df.to_string(index=False))
        f.write(f"\n\nMax level separation: {max_level:.4f}\n")
        f.write(f"Max trajectory slope:  {max_traj:.4f}\n")
        f.write(f"Verdict: {verdict}\n")
        f.write(f"Decision: {decision}\n")
    print(f"    Report: {report_path}")

    print(f"\n Done in {datetime.now() - t0}")
    print(f" Outputs:")
    print(f"   {out_fig}")
    print(f"   {TABLE_DIR}/step0_temporal_signal.csv")
    print(f"   {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
