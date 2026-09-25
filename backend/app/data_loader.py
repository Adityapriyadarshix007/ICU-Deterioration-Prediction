"""
Data loading and validation utilities
UPDATED: 0-6h feature window, 6-18h outcome window (NO DATA LEAKAGE)

FIXES APPLIED:
1. Option A — Labs use ICU intime as reference (consistent with vitals).
2. Issue B — create_feature_datasets loads all items in ONE query,
   then splits by item-id groups in pandas.
3. Issue C — stay_ids / hadm_ids are required (no full-table fallbacks).
4. Lab queries now keyed by stay_id (not hadm_id). Fixes multi-ICU-stay
   admissions where MIN(intime) misattributed labs.
5. Lab results include stay_id for unambiguous attribution.
6. Validation methods filter to ALL_CHART_ITEM_IDS.
7. Naming: hours_since_icu everywhere (was inconsistent).
8. Inclusivity: both windows use [start, end).

LEAKAGE GUARANTEE:
- Feature window: [intime, intime + FEATURE_WINDOW_HOURS)
- Outcome window: [intime + PREDICTION_START_HOUR, intime + PREDICTION_END_HOUR)
- Both anchored to ICU intime.
- No overlap.
"""

import pandas as pd
from typing import List, Optional, Dict, Any

from database import DatabaseManager
from config import (
    DATA_DIR,
    VASOPRESSORS, LAB_ITEM_IDS,
    RESPIRATORY_ITEM_IDS, GCS_ITEM_IDS,
    HR_ITEMID,
    SBP_ITEMIDS, DBP_ITEMIDS, MAP_ITEMIDS,
    FEATURE_WINDOW_HOURS,
    PREDICTION_START_HOUR,
    PREDICTION_END_HOUR,
)


# ============================================================
# ITEM-ID GROUPS (used for splitting a single big chart query)
# ============================================================

VITAL_ITEM_IDS = [
    HR_ITEMID,
    RESPIRATORY_ITEM_IDS['rr'],
    RESPIRATORY_ITEM_IDS['spo2'],
    *SBP_ITEMIDS,
    *DBP_ITEMIDS,
    *MAP_ITEMIDS,
    223761,  # Temperature F
    223762,  # Temperature C
]

RESPIRATORY_ITEM_ID_LIST = list(RESPIRATORY_ITEM_IDS.values())

GCS_ITEM_ID_LIST = [v for v in GCS_ITEM_IDS.values() if v is not None]

# Every chart item we need, deduplicated.
# NOTE: rr and spo2 appear in both VITAL_ITEM_IDS and RESPIRATORY_ITEM_ID_LIST.
# When splitting, they will appear in both feature_vitals and
# feature_respiratory — this is intentional, don't double-count.
ALL_CHART_ITEM_IDS = sorted(set(
    VITAL_ITEM_IDS + RESPIRATORY_ITEM_ID_LIST + GCS_ITEM_ID_LIST
))


class DataLoader:
    """Load and validate MIMIC-IV data with window-specific methods."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.connection = db_manager.connection

    # ============================================================
    # Validation Methods
    # ============================================================

    def validate_data_presence(self) -> Dict[str, bool]:
        required_files = [
            "icu/icustays.csv.gz",
            "icu/chartevents.csv.gz",
            "icu/inputevents.csv.gz",
            "icu/d_items.csv.gz",
            "hosp/admissions.csv.gz",
            "hosp/patients.csv.gz",
            "hosp/labevents.csv.gz",
            "hosp/d_labitems.csv.gz",
        ]

        results = {}
        base_dir = DATA_DIR

        for file_path in required_files:
            full_path = base_dir / file_path
            exists = full_path.exists()
            results[file_path] = exists
            status = "✅" if exists else "❌"
            print(f"{status} {file_path}: {'Found' if exists else 'MISSING'}")

        return results

    # ============================================================
    # Core Data Loading Methods
    # ============================================================

    def get_icu_stays(self) -> pd.DataFrame:
        query = """
        SELECT
            ic.stay_id,
            ic.hadm_id,
            ic.subject_id,
            ic.intime,
            ic.outtime,
            ic.los,
            p.anchor_age,
            p.gender,
            p.dod,
            a.admission_type,
            a.admission_location,
            a.discharge_location,
            a.hospital_expire_flag,
            a.deathtime,
            a.dischtime
        FROM icustays ic
        INNER JOIN patients p ON ic.subject_id = p.subject_id
        INNER JOIN admissions a ON ic.hadm_id = a.hadm_id
        """
        return self.db.fetch_df(query)

    def get_chart_events(self,
                         stay_ids: List[int],
                         item_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Load chart events for explicit stay_ids."""
        if not stay_ids:
            raise ValueError("stay_ids is required (no full-table fallback)")

        stay_str = ','.join(map(str, stay_ids))
        query = f"""
        SELECT
            stay_id,
            itemid,
            charttime,
            valuenum,
            valueuom
        FROM chartevents
        WHERE valuenum IS NOT NULL
          AND stay_id IN ({stay_str})
        """

        if item_ids:
            id_str = ','.join(map(str, item_ids))
            query += f" AND itemid IN ({id_str})"

        return self.db.fetch_df(query)

    def get_lab_events(self,
                       hadm_ids: List[int],
                       item_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Load lab events for explicit hadm_ids."""
        if not hadm_ids:
            raise ValueError("hadm_ids is required (no full-table fallback)")

        hadm_str = ','.join(map(str, hadm_ids))
        query = f"""
        SELECT
            hadm_id,
            itemid,
            charttime,
            valuenum,
            valueuom,
            flag,
            ref_range_lower,
            ref_range_upper
        FROM labevents
        WHERE valuenum IS NOT NULL
          AND hadm_id IN ({hadm_str})
        """

        if item_ids:
            id_str = ','.join(map(str, item_ids))
            query += f" AND itemid IN ({id_str})"

        return self.db.fetch_df(query)

    def get_input_events(self,
                         stay_ids: List[int],
                         item_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Load input events for explicit stay_ids."""
        if not stay_ids:
            raise ValueError("stay_ids is required (no full-table fallback)")

        stay_str = ','.join(map(str, stay_ids))
        query = f"""
        SELECT
            stay_id,
            itemid,
            starttime,
            endtime,
            amount,
            amountuom,
            rate,
            rateuom,
            orderid,
            linkorderid
        FROM inputevents
        WHERE (amount IS NOT NULL OR rate IS NOT NULL)
          AND stay_id IN ({stay_str})
        """

        if item_ids:
            id_str = ','.join(map(str, item_ids))
            query += f" AND itemid IN ({id_str})"

        return self.db.fetch_df(query)

    # ============================================================
    # Specialized Data Loading Methods
    # ============================================================

    def get_vasopressors(self, stay_ids: List[int]) -> pd.DataFrame:
        vaso_ids = list(VASOPRESSORS.values())
        return self.get_input_events(stay_ids, vaso_ids)

    def get_all_labs(self, hadm_ids: List[int]) -> pd.DataFrame:
        """Load ALL labs (not just SOFA-relevant)."""
        lab_ids = list(LAB_ITEM_IDS.values())
        return self.get_lab_events(hadm_ids, lab_ids)

    def get_respiratory_events(self, stay_ids: List[int]) -> pd.DataFrame:
        return self.get_chart_events(stay_ids, RESPIRATORY_ITEM_ID_LIST)

    def get_gcs_events(self, stay_ids: List[int]) -> pd.DataFrame:
        return self.get_chart_events(stay_ids, GCS_ITEM_ID_LIST)

    def get_sofa_map_events(self, stay_ids: List[int]) -> pd.DataFrame:
        """Load MAP from both arterial and NIBP."""
        return self.get_chart_events(stay_ids, MAP_ITEMIDS)

    def get_vital_signs(self, stay_ids: List[int]) -> pd.DataFrame:
        """Load all vital signs (includes both arterial and NIBP BP)."""
        return self.get_chart_events(stay_ids, VITAL_ITEM_IDS)

    # ============================================================
    # FEATURE WINDOW METHODS (0-6 hours from ICU intime)
    # ============================================================

    def get_feature_window_chart_events(self,
                                        stay_ids: List[int],
                                        item_ids: Optional[List[int]] = None,
                                        window_hours: Optional[int] = None) -> pd.DataFrame:
        if not stay_ids:
            raise ValueError("stay_ids is required (no full-table fallback)")
        if window_hours is None:
            window_hours = FEATURE_WINDOW_HOURS

        stay_str = ','.join(map(str, stay_ids))
        query = f"""
        WITH feature_stays AS (
            SELECT
                stay_id,
                intime,
                intime + INTERVAL '{window_hours} HOURS' AS feature_end
            FROM icustays
            WHERE stay_id IN ({stay_str})
        )
        SELECT
            ce.stay_id,
            ce.itemid,
            ce.charttime,
            ce.valuenum,
            ce.valueuom,
            EXTRACT(EPOCH FROM (ce.charttime - fs.intime)) / 3600 AS hours_since_icu
        FROM chartevents ce
        INNER JOIN feature_stays fs ON ce.stay_id = fs.stay_id
        WHERE ce.charttime >= fs.intime
          AND ce.charttime < fs.feature_end
          AND ce.valuenum IS NOT NULL
        """

        if item_ids:
            id_str = ','.join(map(str, item_ids))
            query += f" AND ce.itemid IN ({id_str})"

        return self.db.fetch_df(query)

    def get_feature_window_labs(self,
                                stay_ids: List[int],
                                item_ids: Optional[List[int]] = None,
                                window_hours: Optional[int] = None) -> pd.DataFrame:
        """
        Load labs within [ICU intime, ICU intime + window_hours).

        Keyed by stay_id: each lab is attributed to the ICU stay whose
        window contains its charttime. Handles multi-ICU-stay admissions
        correctly (does NOT collapse to a single MIN(intime)).
        """
        if not stay_ids:
            raise ValueError("stay_ids is required (no full-table fallback)")
        if window_hours is None:
            window_hours = FEATURE_WINDOW_HOURS

        stay_str = ','.join(map(str, stay_ids))
        query = f"""
        WITH feature_stays AS (
            SELECT
                stay_id,
                hadm_id,
                intime,
                intime + INTERVAL '{window_hours} HOURS' AS feature_end
            FROM icustays
            WHERE stay_id IN ({stay_str})
        )
        SELECT
            le.hadm_id,
            fs.stay_id,
            le.itemid,
            le.charttime,
            le.valuenum,
            le.valueuom,
            le.flag,
            EXTRACT(EPOCH FROM (le.charttime - fs.intime)) / 3600 AS hours_since_icu
        FROM labevents le
        INNER JOIN feature_stays fs
            ON le.hadm_id = fs.hadm_id
           AND le.charttime >= fs.intime
           AND le.charttime < fs.feature_end
        WHERE le.valuenum IS NOT NULL
        """

        if item_ids:
            id_str = ','.join(map(str, item_ids))
            query += f" AND le.itemid IN ({id_str})"

        return self.db.fetch_df(query)

    def get_feature_window_vasopressors(self,
                                        stay_ids: List[int],
                                        window_hours: Optional[int] = None) -> pd.DataFrame:
        if not stay_ids:
            raise ValueError("stay_ids is required (no full-table fallback)")
        if window_hours is None:
            window_hours = FEATURE_WINDOW_HOURS

        vaso_ids = list(VASOPRESSORS.values())
        stay_str = ','.join(map(str, stay_ids))
        id_str = ','.join(map(str, vaso_ids))

        query = f"""
        WITH feature_stays AS (
            SELECT
                stay_id,
                intime,
                intime + INTERVAL '{window_hours} HOURS' AS feature_end
            FROM icustays
            WHERE stay_id IN ({stay_str})
        )
        SELECT
            ie.stay_id,
            ie.itemid,
            ie.starttime,
            ie.endtime,
            ie.amount,
            ie.amountuom,
            ie.rate,
            ie.rateuom,
            EXTRACT(EPOCH FROM (ie.starttime - fs.intime)) / 3600 AS hours_since_icu
        FROM inputevents ie
        INNER JOIN feature_stays fs ON ie.stay_id = fs.stay_id
        WHERE ie.itemid IN ({id_str})
          AND (ie.amount IS NOT NULL OR ie.rate IS NOT NULL)
          AND ie.starttime >= fs.intime
          AND ie.starttime < fs.feature_end
        """

        return self.db.fetch_df(query)

    # ============================================================
    # OUTCOME WINDOW METHODS (6-18 hours from ICU intime)
    # ============================================================

    def get_outcome_window_chart_events(self,
                                        stay_ids: List[int],
                                        item_ids: Optional[List[int]] = None,
                                        start_hour: Optional[int] = None,
                                        end_hour: Optional[int] = None) -> pd.DataFrame:
        if not stay_ids:
            raise ValueError("stay_ids is required (no full-table fallback)")
        if start_hour is None:
            start_hour = PREDICTION_START_HOUR
        if end_hour is None:
            end_hour = PREDICTION_END_HOUR

        stay_str = ','.join(map(str, stay_ids))
        query = f"""
        WITH outcome_stays AS (
            SELECT
                stay_id,
                intime,
                intime + INTERVAL '{start_hour} HOURS' AS outcome_start,
                intime + INTERVAL '{end_hour} HOURS' AS outcome_end
            FROM icustays
            WHERE stay_id IN ({stay_str})
        )
        SELECT
            ce.stay_id,
            ce.itemid,
            ce.charttime,
            ce.valuenum,
            ce.valueuom,
            EXTRACT(EPOCH FROM (ce.charttime - os.intime)) / 3600 AS hours_since_icu
        FROM chartevents ce
        INNER JOIN outcome_stays os ON ce.stay_id = os.stay_id
        WHERE ce.charttime >= os.outcome_start
          AND ce.charttime < os.outcome_end
          AND ce.valuenum IS NOT NULL
        """

        if item_ids:
            id_str = ','.join(map(str, item_ids))
            query += f" AND ce.itemid IN ({id_str})"

        return self.db.fetch_df(query)

    def get_outcome_window_labs(self,
                                stay_ids: List[int],
                                item_ids: Optional[List[int]] = None,
                                start_hour: Optional[int] = None,
                                end_hour: Optional[int] = None) -> pd.DataFrame:
        """
        Load labs within [ICU intime + start_hour, ICU intime + end_hour).

        Keyed by stay_id (same fix as feature window).
        """
        if not stay_ids:
            raise ValueError("stay_ids is required (no full-table fallback)")
        if start_hour is None:
            start_hour = PREDICTION_START_HOUR
        if end_hour is None:
            end_hour = PREDICTION_END_HOUR

        stay_str = ','.join(map(str, stay_ids))
        query = f"""
        WITH outcome_stays AS (
            SELECT
                stay_id,
                hadm_id,
                intime,
                intime + INTERVAL '{start_hour} HOURS' AS outcome_start,
                intime + INTERVAL '{end_hour} HOURS' AS outcome_end
            FROM icustays
            WHERE stay_id IN ({stay_str})
        )
        SELECT
            le.hadm_id,
            os.stay_id,
            le.itemid,
            le.charttime,
            le.valuenum,
            le.valueuom,
            le.flag,
            EXTRACT(EPOCH FROM (le.charttime - os.intime)) / 3600 AS hours_since_icu
        FROM labevents le
        INNER JOIN outcome_stays os
            ON le.hadm_id = os.hadm_id
           AND le.charttime >= os.outcome_start
           AND le.charttime < os.outcome_end
        WHERE le.valuenum IS NOT NULL
        """

        if item_ids:
            id_str = ','.join(map(str, item_ids))
            query += f" AND le.itemid IN ({id_str})"

        return self.db.fetch_df(query)

    def get_outcome_window_vasopressors(self,
                                        stay_ids: List[int],
                                        start_hour: Optional[int] = None,
                                        end_hour: Optional[int] = None) -> pd.DataFrame:
        if not stay_ids:
            raise ValueError("stay_ids is required (no full-table fallback)")
        if start_hour is None:
            start_hour = PREDICTION_START_HOUR
        if end_hour is None:
            end_hour = PREDICTION_END_HOUR

        vaso_ids = list(VASOPRESSORS.values())
        stay_str = ','.join(map(str, stay_ids))
        id_str = ','.join(map(str, vaso_ids))

        query = f"""
        WITH outcome_stays AS (
            SELECT
                stay_id,
                intime,
                intime + INTERVAL '{start_hour} HOURS' AS outcome_start,
                intime + INTERVAL '{end_hour} HOURS' AS outcome_end
            FROM icustays
            WHERE stay_id IN ({stay_str})
        )
        SELECT
            ie.stay_id,
            ie.itemid,
            ie.starttime,
            ie.endtime,
            ie.amount,
            ie.amountuom,
            ie.rate,
            ie.rateuom,
            EXTRACT(EPOCH FROM (ie.starttime - os.intime)) / 3600 AS hours_since_icu
        FROM inputevents ie
        INNER JOIN outcome_stays os ON ie.stay_id = os.stay_id
        WHERE ie.itemid IN ({id_str})
          AND (ie.amount IS NOT NULL OR ie.rate IS NOT NULL)
          AND ie.starttime >= os.outcome_start
          AND ie.starttime < os.outcome_end
        """

        return self.db.fetch_df(query)

    # ============================================================
    # COMPREHENSIVE FEATURE DATASETS (Issue B fix preserved)
    # ============================================================

    def create_feature_datasets(self,
                                cohort_df: pd.DataFrame,
                                include_all: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Build all feature/outcome datasets for a cohort.

        Loads all chart events in ONE query, then splits by item-id group.
        Labs and vasopressors come from their own tables.
        """
        stay_ids = cohort_df['stay_id'].tolist()

        print("=" * 60)
        print("CREATING FEATURE DATASETS")
        print("=" * 60)
        print(f"Total stays: {len(stay_ids):,}")
        print(f"FEATURE WINDOW: 0-{FEATURE_WINDOW_HOURS}h from ICU admission")
        print(f"OUTCOME WINDOW: {PREDICTION_START_HOUR}-{PREDICTION_END_HOUR}h from ICU admission")
        print("⚠️  NO DATA LEAKAGE: Features END before Outcome BEGINS")
        print("=" * 60)

        # ------ FEATURE WINDOW ------
        print("\n📊 Loading FEATURE WINDOW data (0-6h ICU)...")

        feature_chart = self.get_feature_window_chart_events(
            stay_ids=stay_ids,
            item_ids=ALL_CHART_ITEM_IDS,
        )
        print(f"  ✅ feature_chart (all items): {len(feature_chart):,} records")

        feature_vitals = feature_chart[feature_chart['itemid'].isin(VITAL_ITEM_IDS)]
        feature_respiratory = feature_chart[
            feature_chart['itemid'].isin(RESPIRATORY_ITEM_ID_LIST)
        ]
        feature_gcs = feature_chart[feature_chart['itemid'].isin(GCS_ITEM_ID_LIST)]

        feature_labs = self.get_feature_window_labs(
            stay_ids=stay_ids,
            item_ids=list(LAB_ITEM_IDS.values()),
        )
        feature_vasopressors = self.get_feature_window_vasopressors(stay_ids=stay_ids)

        feature_features = {
            'feature_vitals': feature_vitals,
            'feature_labs': feature_labs,
            'feature_vasopressors': feature_vasopressors,
            'feature_respiratory': feature_respiratory,
            'feature_gcs': feature_gcs,
        }
        for name, df in feature_features.items():
            print(f"  ✅ {name}: {len(df):,} records")

        # ------ OUTCOME WINDOW ------
        print("\n📊 Loading OUTCOME WINDOW data (6-18h ICU)...")

        outcome_chart = self.get_outcome_window_chart_events(
            stay_ids=stay_ids,
            item_ids=ALL_CHART_ITEM_IDS,
        )
        print(f"  ✅ outcome_chart (all items): {len(outcome_chart):,} records")

        outcome_vitals = outcome_chart[outcome_chart['itemid'].isin(VITAL_ITEM_IDS)]
        outcome_respiratory = outcome_chart[
            outcome_chart['itemid'].isin(RESPIRATORY_ITEM_ID_LIST)
        ]
        outcome_gcs = outcome_chart[outcome_chart['itemid'].isin(GCS_ITEM_ID_LIST)]

        outcome_labs = self.get_outcome_window_labs(
            stay_ids=stay_ids,
            item_ids=list(LAB_ITEM_IDS.values()),
        )
        outcome_vasopressors = self.get_outcome_window_vasopressors(stay_ids=stay_ids)

        outcome_features = {
            'outcome_vitals': outcome_vitals,
            'outcome_labs': outcome_labs,
            'outcome_vasopressors': outcome_vasopressors,
            'outcome_respiratory': outcome_respiratory,
            'outcome_gcs': outcome_gcs,
        }
        for name, df in outcome_features.items():
            print(f"  ✅ {name}: {len(df):,} records")

        datasets = {
            'stays': cohort_df,
            **feature_features,
            **outcome_features,
        }

        if include_all:
            print("\n📊 Loading additional datasets...")
            datasets['all_labs'] = self.get_all_labs(
                cohort_df['hadm_id'].tolist()
            )
            print(f"  ✅ all_labs: {len(datasets['all_labs']):,} records")

        print("\n" + "=" * 60)
        print(f"✅ Loaded {len(datasets)} datasets")
        print("⚠️  NO DATA LEAKAGE: Features (0-6h) END before Outcome (6-18h) BEGINS")
        print("=" * 60)

        return datasets

    # ============================================================
    # DATA QUALITY METHODS
    # ============================================================

    def check_data_quality(self, df: pd.DataFrame, name: str) -> Dict[str, Any]:
        missing = df.isnull().sum().sum()
        denom = len(df) * len(df.columns)
        return {
            'name': name,
            'rows': len(df),
            'columns': len(df.columns),
            'missing_values': int(missing),
            'missing_percentage': (missing / denom * 100) if denom > 0 else 0.0,
            'duplicate_rows': int(df.duplicated().sum()),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,
            'column_types': df.dtypes.value_counts().to_dict(),
        }

    # ============================================================
    # WINDOW SUMMARY / VALIDATION
    # ============================================================

    def _window_report(self, stay_ids: List[int]) -> Dict[str, Any]:
        """Shared logic for get_window_summary and validate_window_data."""
        if not stay_ids:
            raise ValueError("stay_ids is required")

        n = len(stay_ids)

        # Feature window
        fv = self.get_feature_window_chart_events(stay_ids, ALL_CHART_ITEM_IDS)
        fl = self.get_feature_window_labs(stay_ids)
        fvp = self.get_feature_window_vasopressors(stay_ids)
        fg = self.get_feature_window_chart_events(stay_ids, GCS_ITEM_ID_LIST)

        # Outcome window
        ov = self.get_outcome_window_chart_events(stay_ids, ALL_CHART_ITEM_IDS)
        ol = self.get_outcome_window_labs(stay_ids)
        ovp = self.get_outcome_window_vasopressors(stay_ids)
        og = self.get_outcome_window_chart_events(stay_ids, GCS_ITEM_ID_LIST)

        f_stays = set(fv['stay_id'].unique()) if not fv.empty else set()
        o_stays = set(ov['stay_id'].unique()) if not ov.empty else set()

        return {
            'total_stays': n,
            'feature_window': {
                'window': f'0-{FEATURE_WINDOW_HOURS}h from ICU admission',
                'vitals_count': len(fv),
                'stays_with_vitals': fv['stay_id'].nunique() if not fv.empty else 0,
                'labs_count': len(fl),
                'stays_with_labs': fl['stay_id'].nunique() if not fl.empty else 0,
                'vasopressors_count': len(fvp),
                'stays_with_vasopressors': fvp['stay_id'].nunique() if not fvp.empty else 0,
                'gcs_count': len(fg),
                'stays_with_gcs': fg['stay_id'].nunique() if not fg.empty else 0,
            },
            'outcome_window': {
                'window': f'{PREDICTION_START_HOUR}-{PREDICTION_END_HOUR}h from ICU admission',
                'vitals_count': len(ov),
                'stays_with_vitals': ov['stay_id'].nunique() if not ov.empty else 0,
                'labs_count': len(ol),
                'stays_with_labs': ol['stay_id'].nunique() if not ol.empty else 0,
                'vasopressors_count': len(ovp),
                'stays_with_vasopressors': ovp['stay_id'].nunique() if not ovp.empty else 0,
                'gcs_count': len(og),
                'stays_with_gcs': og['stay_id'].nunique() if not og.empty else 0,
            },
            'both_windows': len(f_stays.intersection(o_stays)),
        }

    def get_window_summary(self, stay_ids: List[int]) -> Dict[str, Any]:
        """Silent summary (no printing)."""
        return self._window_report(stay_ids)

    def validate_window_data(self, stay_ids: List[int]) -> Dict[str, Any]:
        """Print-and-return window validation report."""
        report = self._window_report(stay_ids)
        n = report['total_stays']

        print("=" * 60)
        print("VALIDATING WINDOW DATA AVAILABILITY")
        print("=" * 60)
        print(f"Checking {n} stays...")
        print(f"Feature Window: {report['feature_window']['window']}")
        print(f"Outcome Window: {report['outcome_window']['window']}")
        print("-" * 60)

        fw = report['feature_window']
        ow = report['outcome_window']

        print("\n📊 Feature Window:")
        print(f"  Stays with vitals: {fw['stays_with_vitals']}/{n} ({fw['stays_with_vitals']/n*100:.1f}%)")
        print(f"  Stays with labs: {fw['stays_with_labs']}/{n} ({fw['stays_with_labs']/n*100:.1f}%)")
        print(f"  Stays with vasopressors: {fw['stays_with_vasopressors']}/{n} ({fw['stays_with_vasopressors']/n*100:.1f}%)")
        print(f"  Stays with GCS: {fw['stays_with_gcs']}/{n} ({fw['stays_with_gcs']/n*100:.1f}%)")

        print("\n📊 Outcome Window:")
        print(f"  Stays with vitals: {ow['stays_with_vitals']}/{n} ({ow['stays_with_vitals']/n*100:.1f}%)")
        print(f"  Stays with labs: {ow['stays_with_labs']}/{n} ({ow['stays_with_labs']/n*100:.1f}%)")
        print(f"  Stays with vasopressors: {ow['stays_with_vasopressors']}/{n} ({ow['stays_with_vasopressors']/n*100:.1f}%)")
        print(f"  Stays with GCS: {ow['stays_with_gcs']}/{n} ({ow['stays_with_gcs']/n*100:.1f}%)")

        print(f"\n✅ Stays with data in BOTH windows: {report['both_windows']}/{n} ({report['both_windows']/n*100:.1f}%)")
        print("⚠️  NO DATA LEAKAGE: Both windows anchored to ICU intime")
        print("=" * 60)

        return report
