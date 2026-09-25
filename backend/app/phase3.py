#!/usr/bin/env python3
"""
Phase 3: Feature Extraction with Time-Series Imputation (FULLY CORRECTED)

Extracts features from the 0-6h ICU window. Applies time-series
imputation (none / locf / linear) to raw time series before aggregation.

DESIGN:
- Three time-series methods: 'none', 'locf', 'linear'
- Time-series imputation on: vitals, labs, GCS, respiratory
  (excluded: vasopressors — interval-based)
- BP coalescing: arterial (220050/51/52) first, NIBP (220179/80/81) fallback
- Order: extract -> coalesce BP -> capture raw hours -> impute -> aggregate

LEAKAGE GUARANTEE:
- All extraction from [intime, intime + FEATURE_WINDOW_HOURS)
- All imputation bounded to same window
- No outcome-window data ever touches features

FIXES APPLIED:
1. Pre-grouped by stay_id in ALL aggregation methods (was O(N*M))
2. Handles both 'hours_since_icu' and 'hours_since_admission' internally
3. time_since_last_measure computed on RAW (pre-imputation) hours
4. PaO2/FiO2 ratio paired by charttime (30min tolerance)
5. Lab 'worst' computed per-lab direction (min or max)
6. Empty-DataFrame guards in all aggregation methods
7. Linear imputation does NOT extrapolate at edges
8. LOGS_DIR used from config (not hardcoded)
9. Slope returns NaN for single time point (not 0.0)
10. GCS total: verbal=0 (intubated) treated as verbal=1
11. BP raw hours keyed by (stay_id, bp_type) after coalescing

OUTPUT:
- outputs/tables/features_0_6h_none.csv
- outputs/tables/features_0_6h_locf.csv
- outputs/tables/features_0_6h_linear.csv
"""

import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TABLE_DIR, LOGS_DIR, DB_NAME,
    FEATURE_WINDOW_HOURS,
    SOFA_CHANGE_THRESHOLD,
    PREDICTION_START_HOUR,
    PREDICTION_END_HOUR,
    LAB_ITEM_IDS,
    RESPIRATORY_ITEM_IDS,
    SOFA_RESPIRATORY_ITEM_IDS,
    GCS_ITEM_IDS,
    HR_ITEMID,
    SBP_ITEMIDS, DBP_ITEMIDS, MAP_ITEMIDS,
    VASOPRESSORS,
    TIME_SERIES_IMPUTATION_STRATEGIES,
)
from database import DatabaseManager
from data_loader import DataLoader


# ============================================================
# ITEM ID CONSTANTS
# ============================================================

ARTERIAL_BP = {220050: 'sbp', 220051: 'dbp', 220052: 'map'}
NIBP_BP = {220179: 'sbp', 220180: 'dbp', 220181: 'map'}
ALL_BP_ITEMIDS = list(ARTERIAL_BP.keys()) + list(NIBP_BP.keys())

OTHER_VITAL_ITEMIDS = {
    HR_ITEMID: 'heart_rate',
    RESPIRATORY_ITEM_IDS['rr']: 'respiratory_rate',
    RESPIRATORY_ITEM_IDS['spo2']: 'spo2',
    223761: 'temperature',
}

LAB_ITEM_ID_LIST = list(LAB_ITEM_IDS.values())
GCS_ITEM_ID_LIST = [v for v in GCS_ITEM_IDS.values() if v is not None]
RESP_ITEM_ID_LIST = list(SOFA_RESPIRATORY_ITEM_IDS.values())

LAB_WORST_DIRECTION = {
    'creatinine': 'max', 'bilirubin': 'max', 'bun': 'max', 'lactate': 'max',
    'platelets': 'min', 'hemoglobin': 'min', 'hematocrit': 'min',
    'inr': 'max', 'ptt': 'max', 'pt': 'max',
    'alt': 'max', 'ast': 'max', 'alkaline_phosphatase': 'max',
    'albumin': 'min',
    'wbc': 'max', 'sodium': 'max', 'potassium': 'max',
    'chloride': 'max', 'bicarbonate': 'min', 'glucose': 'max',
}


# ============================================================
# TIME COLUMN RESOLUTION
# ============================================================

def _resolve_time_col(df: pd.DataFrame) -> str:
    for col in ('hours_since_admission', 'hours_since_icu'):
        if col in df.columns:
            return col
    raise KeyError(
        f"No time column found. Expected 'hours_since_admission' or "
        f"'hours_since_icu'. Got: {list(df.columns)}"
    )


# ============================================================
# TIME-SERIES IMPUTATION
# ============================================================

def _check_window(hours: np.ndarray, window_end: float):
    if len(hours) == 0:
        return
    if (hours > window_end + 1e-6).any():
        raise ValueError(
            f"Data outside feature window: max hour {hours.max():.2f} > "
            f"{window_end}"
        )


def locf_time_impute(values: np.ndarray, hours: np.ndarray,
                     window_end: float) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    hours = np.asarray(hours, dtype=float)
    if len(values) == 0:
        return values
    _check_window(hours, window_end)

    order = np.argsort(hours)
    v = values[order]
    last = np.nan
    for i in range(len(v)):
        if np.isnan(v[i]):
            v[i] = last
        else:
            last = v[i]

    out = np.empty_like(v)
    out[order] = v
    return out


def linear_time_impute(values: np.ndarray, hours: np.ndarray,
                       window_end: float) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    hours = np.asarray(hours, dtype=float)
    if len(values) == 0:
        return values
    _check_window(hours, window_end)

    valid = ~np.isnan(values)
    if valid.sum() < 2:
        return values

    order = np.argsort(hours)
    v = values[order]
    h = hours[order]
    ok = ~np.isnan(v)

    nan_mask = ~ok
    if nan_mask.sum() > 0:
        v[nan_mask] = np.interp(
            h[nan_mask],
            h[ok],
            v[ok],
            left=np.nan,
            right=np.nan,
        )

    out = np.empty_like(v)
    out[order] = v
    return out


def apply_time_series_imputation(df: pd.DataFrame, method: str,
                                 group_col: str = 'stay_id',
                                 window_end: float = FEATURE_WINDOW_HOURS
                                 ) -> pd.DataFrame:
    if df.empty or method == 'none':
        return df.copy()

    if method not in ('locf', 'linear'):
        raise ValueError(f"Unknown method: {method}")

    func = locf_time_impute if method == 'locf' else linear_time_impute

    time_col = _resolve_time_col(df)
    vaso_ids = set(VASOPRESSORS.values())

    df = df.copy()
    chunks = []
    for (gid, itemid), group in df.groupby([group_col, 'itemid']):
        if itemid in vaso_ids:
            chunks.append(group)
            continue
        group = group.sort_values(time_col).copy()
        v = group['valuenum'].values
        h = group[time_col].values
        group['valuenum'] = func(v, h, window_end=window_end)
        chunks.append(group)

    if not chunks:
        return df.iloc[0:0]
    return pd.concat(chunks, ignore_index=True)


# ============================================================
# BP COALESCING
# ============================================================

def coalesce_bp(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df['bp_type'] = df['itemid'].map({**ARTERIAL_BP, **NIBP_BP})
    df['is_arterial'] = df['itemid'].isin(ARTERIAL_BP).astype(int)

    df = df.sort_values(
        ['stay_id', 'charttime', 'bp_type', 'is_arterial'],
        ascending=[True, True, True, False],
    )
    df = df.drop_duplicates(
        subset=['stay_id', 'charttime', 'bp_type'],
        keep='first',
    )
    return df.reset_index(drop=True)


# ============================================================
# TEMPORAL HELPERS
# ============================================================

def compute_slope(values: np.ndarray, hours: np.ndarray) -> float:
    if len(values) < 2:
        return np.nan
    valid = ~np.isnan(values)
    if valid.sum() < 2:
        return np.nan
    v = values[valid]
    h = hours[valid]
    if len(np.unique(h)) < 2:
        return np.nan
    try:
        return float(np.polyfit(h, v, 1)[0])
    except Exception:
        return np.nan


def compute_delta(values: np.ndarray) -> float:
    v = values[~np.isnan(values)]
    if len(v) < 2:
        return np.nan
    return float(v[-1] - v[0])


def compute_range(values: np.ndarray) -> float:
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return np.nan
    return float(v.max() - v.min())


def time_since_last_measure(hours: np.ndarray,
                            total: float = FEATURE_WINDOW_HOURS) -> float:
    h = hours[~np.isnan(hours)]
    if len(h) == 0:
        return np.nan
    return float(total - h.max())


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_series(series_df: pd.DataFrame, name_prefix: str,
                     agg_config: str, raw_hours: np.ndarray = None) -> dict:
    row = {}
    if series_df.empty:
        return row

    time_col = _resolve_time_col(series_df)
    series_df = series_df.sort_values(time_col)

    values = series_df['valuenum'].values
    hours = series_df[time_col].values
    v = values[~np.isnan(values)]
    h = hours[~np.isnan(values)]

    if len(v) == 0:
        return row

    hours_for_recency = raw_hours if raw_hours is not None else hours

    if agg_config == 'vitals':
        row[f'{name_prefix}_min'] = float(v.min())
        row[f'{name_prefix}_max'] = float(v.max())
        row[f'{name_prefix}_mean'] = float(v.mean())
        row[f'{name_prefix}_median'] = float(np.median(v))
        row[f'{name_prefix}_std'] = float(v.std())
        row[f'{name_prefix}_last'] = float(v[-1])
        row[f'{name_prefix}_count'] = int(len(v))
        row[f'{name_prefix}_slope'] = compute_slope(values, hours)
        row[f'{name_prefix}_delta'] = compute_delta(values)
        row[f'{name_prefix}_range'] = compute_range(values)
        row[f'{name_prefix}_time_since_last_measure'] = \
            time_since_last_measure(hours_for_recency)

    elif agg_config == 'labs':
        direction = LAB_WORST_DIRECTION.get(name_prefix, 'max')
        if direction == 'min':
            row[f'{name_prefix}_worst'] = float(v.min())
        else:
            row[f'{name_prefix}_worst'] = float(v.max())
        row[f'{name_prefix}_min'] = float(v.min())
        row[f'{name_prefix}_max'] = float(v.max())
        row[f'{name_prefix}_last'] = float(v[-1])
        row[f'{name_prefix}_count'] = int(len(v))
        row[f'{name_prefix}_slope'] = compute_slope(values, hours)
        row[f'{name_prefix}_delta'] = compute_delta(values)

    elif agg_config == 'gcs':
        row[f'{name_prefix}_min'] = float(v.min())
        row[f'{name_prefix}_last'] = float(v[-1])
        row[f'{name_prefix}_slope'] = compute_slope(values, hours)
        row[f'{name_prefix}_delta'] = compute_delta(values)

    elif agg_config == 'respiratory':
        row[f'{name_prefix}_min'] = float(v.min())
        row[f'{name_prefix}_max'] = float(v.max())
        row[f'{name_prefix}_mean'] = float(v.mean())
        row[f'{name_prefix}_last'] = float(v[-1])
        row[f'{name_prefix}_slope'] = compute_slope(values, hours)
        row[f'{name_prefix}_delta'] = compute_delta(values)

    return row


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

class FeatureExtractor:

    def __init__(self, db_manager, logger, method='none'):
        self.db = db_manager
        self.logger = logger
        self.loader = DataLoader(db_manager)
        self.method = method
        self.features_df = None

    # ---------- extraction ----------

    def _extract_vitals_and_bp(self, stay_ids):
        ids = list(OTHER_VITAL_ITEMIDS.keys()) + ALL_BP_ITEMIDS
        return self.loader.get_feature_window_chart_events(
            stay_ids=stay_ids, item_ids=ids)

    def _extract_labs(self, stay_ids):
        return self.loader.get_feature_window_labs(
            stay_ids=stay_ids, item_ids=LAB_ITEM_ID_LIST)

    def _extract_gcs(self, stay_ids):
        return self.loader.get_feature_window_chart_events(
            stay_ids=stay_ids, item_ids=GCS_ITEM_ID_LIST)

    def _extract_respiratory(self, stay_ids):
        return self.loader.get_feature_window_chart_events(
            stay_ids=stay_ids, item_ids=RESP_ITEM_ID_LIST)

    def _extract_vasopressors(self, stay_ids):
        return self.loader.get_feature_window_vasopressors(stay_ids=stay_ids)

    # ---------- aggregation ----------

    def _aggregate_vitals(self, df, stay_ids, raw_hours_by_stay_item=None):
        if df.empty:
            return pd.DataFrame(columns=['stay_id'])
        raw_hours_by_stay_item = raw_hours_by_stay_item or {}
        groups = {sid: g for sid, g in df.groupby('stay_id')}
        empty = pd.DataFrame()
        rows = []
        for sid in stay_ids:
            stay_df = groups.get(sid, empty)
            row = {'stay_id': sid}
            for itemid, name in OTHER_VITAL_ITEMIDS.items():
                item_df = stay_df[stay_df['itemid'] == itemid] \
                    if not stay_df.empty else empty
                raw_h = raw_hours_by_stay_item.get((sid, itemid))
                row.update(aggregate_series(
                    item_df, name, 'vitals', raw_hours=raw_h))
            rows.append(row)
        return pd.DataFrame(rows)

    def _aggregate_bp(self, df, stay_ids, bp_raw_hours=None):
        """
        bp_raw_hours keyed by (stay_id, bp_type) — captured AFTER coalescing,
        BEFORE imputation.
        """
        if df.empty:
            return pd.DataFrame(columns=['stay_id'])
        bp_raw_hours = bp_raw_hours or {}
        groups = {sid: g for sid, g in df.groupby('stay_id')}
        empty = pd.DataFrame()
        rows = []
        for sid in stay_ids:
            stay_df = groups.get(sid, empty)
            row = {'stay_id': sid}
            for bp_type in ['sbp', 'dbp', 'map']:
                bp_df = stay_df[stay_df['bp_type'] == bp_type] \
                    if not stay_df.empty else empty
                raw_h = bp_raw_hours.get((sid, bp_type))
                row.update(aggregate_series(
                    bp_df, bp_type, 'vitals', raw_hours=raw_h))
            rows.append(row)
        return pd.DataFrame(rows)

    def _aggregate_labs(self, df, stay_ids):
        if df.empty:
            return pd.DataFrame(columns=['stay_id'])
        groups = {sid: g for sid, g in df.groupby('stay_id')}
        empty = pd.DataFrame()
        labs_map = {v: k for k, v in LAB_ITEM_IDS.items()}
        rows = []
        for sid in stay_ids:
            stay_df = groups.get(sid, empty)
            row = {'stay_id': sid}
            for itemid, name in labs_map.items():
                item_df = stay_df[stay_df['itemid'] == itemid] \
                    if not stay_df.empty else empty
                row.update(aggregate_series(item_df, name, 'labs'))
            rows.append(row)
        return pd.DataFrame(rows)

    def _aggregate_gcs(self, df, stay_ids):
        if df.empty:
            return pd.DataFrame(columns=['stay_id'])
        groups = {sid: g for sid, g in df.groupby('stay_id')}
        empty = pd.DataFrame()

        eyes_id = GCS_ITEM_IDS.get('gcs_eyes')
        verbal_id = GCS_ITEM_IDS.get('gcs_verbal')
        motor_id = GCS_ITEM_IDS.get('gcs_motor')

        rows = []
        for sid in stay_ids:
            stay_df = groups.get(sid, empty)
            row = {'stay_id': sid}
            for comp, iid in [('gcs_eyes', eyes_id),
                              ('gcs_verbal', verbal_id),
                              ('gcs_motor', motor_id)]:
                if iid is None:
                    continue
                comp_df = stay_df[stay_df['itemid'] == iid] \
                    if not stay_df.empty else empty
                row.update(aggregate_series(comp_df, comp, 'gcs'))

            # ---- GCS total: verbal=0 (intubated) treated as verbal=1 ----
            def _gcs_total(e_key, v_key, m_key):
                if not all(k in row for k in [e_key, v_key, m_key]):
                    return None
                e = row[e_key]
                v = row[v_key]
                m = row[m_key]
                if not (1 <= e <= 4 and 1 <= m <= 6):
                    return None
                if v == 0:
                    v = 1  # intubated → treat as 1
                if not (1 <= v <= 5):
                    return None
                return e + v + m

            total_min = _gcs_total(
                'gcs_eyes_min', 'gcs_verbal_min', 'gcs_motor_min')
            if total_min is not None:
                row['gcs_total_min'] = total_min

            total_last = _gcs_total(
                'gcs_eyes_last', 'gcs_verbal_last', 'gcs_motor_last')
            if total_last is not None:
                row['gcs_total_last'] = total_last

            if 'gcs_total_min' in row and 'gcs_total_last' in row:
                row['gcs_total_delta'] = \
                    row['gcs_total_last'] - row['gcs_total_min']

            rows.append(row)
        return pd.DataFrame(rows)

    def _aggregate_respiratory(self, df, stay_ids):
        if df.empty:
            return pd.DataFrame(columns=['stay_id'])
        groups = {sid: g for sid, g in df.groupby('stay_id')}
        empty = pd.DataFrame()
        resp_map = {
            RESPIRATORY_ITEM_IDS['pao2']: 'pao2',
            RESPIRATORY_ITEM_IDS['fio2']: 'fio2',
        }
        rows = []
        for sid in stay_ids:
            stay_df = groups.get(sid, empty)
            row = {'stay_id': sid}

            for itemid, name in resp_map.items():
                item_df = stay_df[stay_df['itemid'] == itemid] \
                    if not stay_df.empty else empty
                row.update(aggregate_series(item_df, name, 'respiratory'))

            if not stay_df.empty:
                pao2 = stay_df[stay_df['itemid'] == RESPIRATORY_ITEM_IDS['pao2']]
                fio2 = stay_df[stay_df['itemid'] == RESPIRATORY_ITEM_IDS['fio2']]
                if not pao2.empty and not fio2.empty:
                    pf = self._compute_pf_ratio(pao2, fio2)
                    if pf is not None:
                        row['pao2_fio2_ratio'] = pf

            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _compute_pf_ratio(pao2_df, fio2_df) -> float:
        try:
            pao2 = pao2_df[['charttime', 'valuenum']].rename(
                columns={'valuenum': 'pao2'}).sort_values('charttime')
            fio2 = fio2_df[['charttime', 'valuenum']].rename(
                columns={'valuenum': 'fio2'}).sort_values('charttime')

            if not pd.api.types.is_datetime64_any_dtype(pao2['charttime']):
                pao2['charttime'] = pd.to_datetime(pao2['charttime'])
            if not pd.api.types.is_datetime64_any_dtype(fio2['charttime']):
                fio2['charttime'] = pd.to_datetime(fio2['charttime'])

            merged = pd.merge_asof(
                pao2, fio2,
                on='charttime',
                direction='nearest',
                tolerance=pd.Timedelta('30min'),
            )
            merged = merged.dropna(subset=['pao2', 'fio2'])
            if merged.empty:
                return None

            merged['fio2_norm'] = merged['fio2'].apply(
                lambda x: x / 100.0 if x > 1.0 else x
            )
            valid = merged[merged['fio2_norm'] > 0]
            if valid.empty:
                return None
            valid = valid.assign(pf=valid['pao2'] / valid['fio2_norm'])
            return float(valid['pf'].min())
        except Exception:
            return None

    def _aggregate_vasopressors(self, df, stay_ids):
        if df.empty:
            return pd.DataFrame(columns=['stay_id'])
        groups = {sid: g for sid, g in df.groupby('stay_id')}
        empty = pd.DataFrame()
        vaso_map = {v: k for k, v in VASOPRESSORS.items()}
        rows = []
        for sid in stay_ids:
            stay_df = groups.get(sid, empty)
            row = {'stay_id': sid}
            for itemid, name in vaso_map.items():
                item_df = stay_df[stay_df['itemid'] == itemid] \
                    if not stay_df.empty else empty
                if item_df.empty:
                    row[f'{name}_any'] = 0
                    row[f'{name}_max_rate'] = 0.0
                    row[f'{name}_mean_rate'] = 0.0
                else:
                    row[f'{name}_any'] = 1
                    rates = item_df['rate'].dropna()
                    row[f'{name}_max_rate'] = float(rates.max()) if len(rates) else 0.0
                    row[f'{name}_mean_rate'] = float(rates.mean()) if len(rates) else 0.0
            rows.append(row)
        return pd.DataFrame(rows)

    # ---------- main ----------

    def extract_all_features(self, cohort_df):
        self.logger.info("=" * 60)
        self.logger.info(f"FEATURE EXTRACTION — method = {self.method.upper()}")
        self.logger.info("=" * 60)
        self.logger.info(f"⚠️  Bounded to [0, {FEATURE_WINDOW_HOURS}h) from ICU intime")
        self.logger.info("⚠️  NO DATA LEAKAGE")

        stay_ids = cohort_df['stay_id'].tolist()
        self.logger.info(f"Extracting for {len(stay_ids):,} stays...")

        vitals_df = self._extract_vitals_and_bp(stay_ids)
        self.logger.info(f"  Vitals+BP: {len(vitals_df):,}")
        labs_df = self._extract_labs(stay_ids)
        self.logger.info(f"  Labs: {len(labs_df):,}")
        gcs_df = self._extract_gcs(stay_ids)
        self.logger.info(f"  GCS: {len(gcs_df):,}")
        resp_df = self._extract_respiratory(stay_ids)
        self.logger.info(f"  Respiratory: {len(resp_df):,}")
        vaso_df = self._extract_vasopressors(stay_ids)
        self.logger.info(f"  Vasopressors: {len(vaso_df):,}")

        # Split BP from other vitals
        bp_df = vitals_df[vitals_df['itemid'].isin(ALL_BP_ITEMIDS)].copy()
        other_vitals_df = vitals_df[
            vitals_df['itemid'].isin(OTHER_VITAL_ITEMIDS.keys())
        ].copy()

        # Coalesce BP
        self.logger.info("Coalescing BP (arterial-first)...")
        bp_df = coalesce_bp(bp_df)
        self.logger.info(f"  BP after coalescing: {len(bp_df):,}")

        # ---- Capture raw hours BEFORE imputation ----
        # Keyed by (stay_id, itemid) for vitals/labs/GCS/respiratory
        # Keyed by (stay_id, bp_type) for BP (post-coalesce, pre-impute)
        raw_hours_by_stay_item = {}
        bp_raw_hours = {}
        if self.method != 'none':
            for df in [other_vitals_df, labs_df, gcs_df, resp_df]:
                if df.empty:
                    continue
                time_col = _resolve_time_col(df)
                for (sid, iid), g in df.groupby(['stay_id', 'itemid']):
                    raw_hours_by_stay_item[(sid, iid)] = g[time_col].values
            # BP: keyed by (stay_id, bp_type)
            if not bp_df.empty:
                time_col = _resolve_time_col(bp_df)
                for (sid, bp_type), g in bp_df.groupby(['stay_id', 'bp_type']):
                    bp_raw_hours[(sid, bp_type)] = g[time_col].values

        # Apply time-series imputation
        if self.method != 'none':
            self.logger.info(f"Applying {self.method} imputation...")
            other_vitals_df = apply_time_series_imputation(
                other_vitals_df, self.method, 'stay_id')
            bp_df = apply_time_series_imputation(bp_df, self.method, 'stay_id')
            labs_df = apply_time_series_imputation(labs_df, self.method, 'stay_id')
            gcs_df = apply_time_series_imputation(gcs_df, self.method, 'stay_id')
            resp_df = apply_time_series_imputation(resp_df, self.method, 'stay_id')
            self.logger.info("  ✅ Imputation applied")

        # Aggregate
        self.logger.info("Aggregating...")
        vitals_agg = self._aggregate_vitals(
            other_vitals_df, stay_ids, raw_hours_by_stay_item)
        bp_agg = self._aggregate_bp(
            bp_df, stay_ids, bp_raw_hours)
        labs_agg = self._aggregate_labs(labs_df, stay_ids)
        gcs_agg = self._aggregate_gcs(gcs_df, stay_ids)
        resp_agg = self._aggregate_respiratory(resp_df, stay_ids)
        vaso_agg = self._aggregate_vasopressors(vaso_df, stay_ids)

        # Merge
        self.logger.info("Merging features...")
        features = cohort_df[
            ['stay_id', 'hadm_id', 'subject_id', 'intime']
        ].copy()
        for agg in [vitals_agg, bp_agg, labs_agg, gcs_agg, resp_agg, vaso_agg]:
            if agg is not None and not agg.empty:
                if not agg['stay_id'].is_unique:
                    self.logger.warning(
                        "Duplicate stay_id in aggregation — deduplicating")
                    agg = agg.drop_duplicates('stay_id', keep='first')
                features = features.merge(agg, on='stay_id', how='left')

        # Demographics and target
        demo_cols = ['stay_id', 'anchor_age', 'gender', 'admission_type']
        if 'outcome' in cohort_df.columns:
            demo_cols.append('outcome')
        demo = cohort_df[demo_cols].copy()
        features = features.merge(demo, on='stay_id', how='left')

        features['age'] = features['anchor_age']
        features['gender_male'] = (features['gender'] == 'M').astype(int)
        adm_dummies = pd.get_dummies(
            features['admission_type'], prefix='admission')
        features = pd.concat([features, adm_dummies], axis=1)

        features = features.drop(
            columns=['anchor_age', 'gender', 'admission_type'],
            errors='ignore')

        # GCS coverage log (verifies intubation fix)
        if 'gcs_total_min' in features.columns:
            gcs_cov = features['gcs_total_min'].notna().mean() * 100
            n_have = features['gcs_total_min'].notna().sum()
            self.logger.info(
                f"  GCS total coverage: {gcs_cov:.1f}% "
                f"({n_have:,}/{len(features):,})"
            )

        self.features_df = features
        self.logger.info(f"Final feature count: {features.shape[1]}")
        return features

    def save_features(self, output_dir=None):
        if self.features_df is None:
            raise ValueError("No features extracted.")
        output_dir = output_dir or TABLE_DIR
        out_path = Path(output_dir) / f"features_0_6h_{self.method}.csv"
        self.features_df.to_csv(out_path, index=False)
        self.logger.info(f"✅ Saved: {out_path}")


# ============================================================
# MAIN
# ============================================================

def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                LOGS_DIR / f"phase3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            ),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


def print_header():
    print("=" * 80)
    print(" PHASE 3: FEATURE EXTRACTION + TIME-SERIES IMPUTATION")
    print("=" * 80)
    print(f" Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Output: {TABLE_DIR}")
    print("-" * 80)
    print(f" Feature Window: 0-{FEATURE_WINDOW_HOURS}h from ICU admission")
    print(f" Target: Pure ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD}")
    print(" ⚠️  NO DATA LEAKAGE")
    print("-" * 80)
    print(" Methods: none, locf, linear")
    print(" TS imputation on: vitals, labs, GCS, respiratory")
    print(" TS imputation OFF: vasopressors (interval-based)")
    print(" BP coalescing: arterial-first, NIBP fallback")
    print(" GCS total: verbal=0 (intubated) treated as verbal=1")
    print("=" * 80)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str,
                        choices=TIME_SERIES_IMPUTATION_STRATEGIES,
                        default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--cohort', type=str, default=None)
    args = parser.parse_args()

    logger = setup_logging()
    print_header()

    cohort_path = args.cohort or TABLE_DIR / "cohort.csv"
    if not Path(cohort_path).exists():
        logger.error(f"Cohort not found: {cohort_path}")
        return 1

    cohort_df = pd.read_csv(cohort_path)
    logger.info(f"Loaded cohort: {len(cohort_df):,} stays")

    db_manager = DatabaseManager(db_path=DB_NAME)
    db_manager.create_views()

    if args.all:
        methods = TIME_SERIES_IMPUTATION_STRATEGIES
    elif args.method is None:
        logger.info("No --method given and no --all — running 'none' only.")
        logger.info("Use --all to run all three methods.")
        methods = ['none']
    else:
        methods = [args.method]

    for method in methods:
        logger.info(f"\n{'#'*70}\n# {method.upper()}\n{'#'*70}")
        extractor = FeatureExtractor(db_manager, logger, method=method)
        extractor.extract_all_features(cohort_df)
        extractor.save_features()

    db_manager.close()

    print("\n" + "=" * 80)
    print(" PHASE 3 COMPLETED")
    print("=" * 80)
    for m in methods:
        p = TABLE_DIR / f"features_0_6h_{m}.csv"
        if p.exists():
            print(f"  ✅ {p.name}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
