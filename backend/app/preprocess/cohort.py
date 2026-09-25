"""
Cohort Construction: MIMIC-IV Multi-Organ Failure Prediction

Pure ΔSOFA Target (NO DATA LEAKAGE, NO MORTALITY)

DESIGN:
1. Target: pure ΔSOFA ≥ 2 at 6-18h. Mortality is NOT included.
2. Both early deaths (≤6h) and late deaths (6-18h) are excluded.
3. Labs use ICU intime as reference.
4. All queries keyed by stay_id.
5. Excludes inter-hospital transfers IN (Req 1) and OUT (Req 3).

This is a library module. It defines CohortBuilder and nothing else.
To build the cohort, run: python phase2.py
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import logging

from database import DatabaseManager
from data_loader import DataLoader
from config import (
    MIN_ICU_STAY_HOURS,
    FEATURE_WINDOW_HOURS,
    PREDICTION_START_HOUR,
    PREDICTION_END_HOUR,
    SOFA_CHANGE_THRESHOLD,
    RANDOM_STATE,
    MAP_ITEMIDS,
    SBP_ITEMIDS,
    DBP_ITEMIDS,
    HR_ITEMID,
    VASOPRESSORS,
    LAB_ITEM_IDS,
    SOFA_LAB_ITEM_IDS,
    RESPIRATORY_ITEM_IDS,
    SOFA_RESPIRATORY_ITEM_IDS,
    GCS_ITEM_IDS,
    # ── Cohort exclusion config (from config.py)
    EXCLUDE_INTERHOSPITAL_TRANSFERS,
    MIMIC_TRANSFER_IN_LOCATIONS,
    MIMIC_TRANSFER_OUT_LOCATIONS,
)

logger = logging.getLogger(__name__)


class CohortBuilder:
    """
    Build MIMIC-IV cohort for pure ΔSOFA prediction.

    Excludes inter-hospital transfers IN (Requirement 1) and OUT
    (Requirement 3) so the 0-6h feature window represents the patient's
    first 6 hours of ICU-level illness. Ward-to-ICU (FLOOR) patients
    are kept (Requirement 2, documented caveat).
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.data_loader = DataLoader(db_manager)
        self.cohort = None
        self.exclusion_counts = {}

    # ============================================================
    # BUILD COHORT
    # ============================================================

    def build_cohort(self) -> pd.DataFrame:
        logger.info("Starting cohort construction...")
        logger.info("=" * 60)
        logger.info(f"FEATURE WINDOW: 0-{FEATURE_WINDOW_HOURS}h from ICU admission")
        logger.info(f"OUTCOME WINDOW: {PREDICTION_START_HOUR}-"
                    f"{PREDICTION_END_HOUR}h from ICU admission")
        logger.info(f"TARGET: Pure ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD} (NO mortality)")
        logger.info(f"TRANSFER EXCLUSION (Req 1 & 3): "
                    f"{EXCLUDE_INTERHOSPITAL_TRANSFERS}")
        logger.info("⚠️  NO DATA LEAKAGE: Features END before Outcome BEGINS")
        logger.info("=" * 60)

        initial_stays = self._get_initial_stays()
        logger.info(f"Initial ICU stays: {len(initial_stays):,}")

        cohort = self._apply_inclusion_criteria(initial_stays)
        logger.info(f"After inclusion criteria: {len(cohort):,}")

        cohort = self._get_first_stay_only(cohort)
        logger.info(f"After first stay only: {len(cohort):,}")

        cohort = self._calculate_feature_sofa(cohort)
        logger.info(f"With feature SOFA: {len(cohort):,}")

        cohort = self._calculate_outcome_sofa(cohort)
        logger.info(f"With outcome SOFA: {len(cohort):,}")

        cohort = self._create_outcome_labels(cohort)
        logger.info(f"With outcome labels: {len(cohort):,}")

        cohort, _ = self._exclude_early_death(cohort)
        logger.info(f"After excluding deaths ≤{PREDICTION_END_HOUR}h: "
                    f"{len(cohort):,}")

        self._analyze_cohort(cohort)
        self.cohort = cohort
        return cohort

    # ============================================================
    # INITIAL LOAD
    # ============================================================

    def _get_initial_stays(self) -> pd.DataFrame:
        query = """
        SELECT
            ic.stay_id, ic.hadm_id, ic.subject_id,
            ic.intime, ic.outtime, ic.los,
            p.anchor_age, p.gender, p.dod,
            a.admission_type, a.admission_location, a.discharge_location,
            a.hospital_expire_flag, a.deathtime, a.dischtime
        FROM icustays ic
        INNER JOIN patients p ON ic.subject_id = p.subject_id
        INNER JOIN admissions a ON ic.hadm_id = a.hadm_id
        """
        return self.db.fetch_df(query)

    # ============================================================
    # INCLUSION CRITERIA (age, LOS, dates, Req 1, Req 3)
    # ============================================================

    def _apply_inclusion_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        # ---- Age ----
        before = len(df)
        df = df[df['anchor_age'] >= 18]
        self.exclusion_counts['age_lt_18'] = before - len(df)
        logger.info(f"    Excluded {self.exclusion_counts['age_lt_18']:,} age < 18")

        # ---- ICU LOS ----
        before = len(df)
        df = df[df['los'] * 24 >= MIN_ICU_STAY_HOURS]
        self.exclusion_counts['stay_lt_min_hours'] = before - len(df)
        logger.info(f"    Excluded {self.exclusion_counts['stay_lt_min_hours']:,} "
                    f"stay < {MIN_ICU_STAY_HOURS}h")

        # ---- Valid dates ----
        before = len(df)
        df = df[df['intime'].notna() & df['outtime'].notna()]
        self.exclusion_counts['invalid_dates'] = before - len(df)
        if self.exclusion_counts['invalid_dates'] > 0:
            logger.info(f"    Excluded {self.exclusion_counts['invalid_dates']:,} "
                        f"invalid dates")

        # ---- Requirement 1 & 3: transfers in / out ----
        if EXCLUDE_INTERHOSPITAL_TRANSFERS:
            # Req 1: transfer IN
            before = len(df)
            df = df[~df['admission_location'].isin(MIMIC_TRANSFER_IN_LOCATIONS)]
            self.exclusion_counts['req1_transfer_in'] = before - len(df)
            logger.info(f"    [Req 1] Excluded "
                        f"{self.exclusion_counts['req1_transfer_in']:,} "
                        f"inter-hospital transfers IN "
                        f"({MIMIC_TRANSFER_IN_LOCATIONS})")

            # Req 3: transfer OUT
            before = len(df)
            df = df[~df['discharge_location'].isin(MIMIC_TRANSFER_OUT_LOCATIONS)]
            self.exclusion_counts['req3_transfer_out'] = before - len(df)
            logger.info(f"    [Req 3] Excluded "
                        f"{self.exclusion_counts['req3_transfer_out']:,} "
                        f"transfers OUT "
                        f"({MIMIC_TRANSFER_OUT_LOCATIONS})")

            # Req 2: documented as kept
            n_ward = int((df['admission_location'] == 'FLOOR').sum())
            logger.info(f"    [Req 2] KEPT {n_ward:,} ward-to-ICU patients "
                        f"(documented caveat)")

        # ---- Kept cohort composition ----
        logger.info(f"    KEPT {len(df):,} direct ICU admissions")
        if 'admission_location' in df.columns:
            kept_locs = df['admission_location'].value_counts().to_dict()
            top_locs = sorted(kept_locs.items(), key=lambda x: -x[1])[:6]
            logger.info(f"    Composition of kept cohort (top 6 sources):")
            for loc, cnt in top_locs:
                logger.info(f"      {loc}: {cnt:,}")

        return df.reset_index(drop=True)

    # ============================================================
    # FIRST STAY ONLY
    # ============================================================

    def _get_first_stay_only(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['subject_id', 'intime'])
        first_stay = df.drop_duplicates('subject_id', keep='first').reset_index(drop=True)
        self.exclusion_counts['not_first_stay'] = len(df) - len(first_stay)
        logger.info(f"    Excluded {self.exclusion_counts['not_first_stay']:,} "
                    f"non-first stays")
        return first_stay

    # ============================================================
    # FEATURE SOFA (0-6h)
    # ============================================================

    def _calculate_feature_sofa(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Calculating feature SOFA (0-{FEATURE_WINDOW_HOURS}h)...")

        stay_ids = df['stay_id'].tolist()

        map_data = self.data_loader.get_feature_window_chart_events(
            stay_ids=stay_ids, item_ids=MAP_ITEMIDS,
        )
        lab_data = self.data_loader.get_feature_window_labs(
            stay_ids=stay_ids, item_ids=list(SOFA_LAB_ITEM_IDS.values()),
        )
        vaso_data = self.data_loader.get_feature_window_vasopressors(
            stay_ids=stay_ids,
        )
        resp_item_ids = list(SOFA_RESPIRATORY_ITEM_IDS.values()) + [
            RESPIRATORY_ITEM_IDS['spo2']
        ]
        resp_data = self.data_loader.get_feature_window_chart_events(
            stay_ids=stay_ids, item_ids=resp_item_ids,
        )
        gcs_ids = [v for v in GCS_ITEM_IDS.values() if v is not None]
        gcs_data = self.data_loader.get_feature_window_chart_events(
            stay_ids=stay_ids, item_ids=gcs_ids,
        )

        logger.info(f"      MAP: {len(map_data):,}")
        logger.info(f"      Labs: {len(lab_data):,}")
        logger.info(f"      Vasopressors: {len(vaso_data):,}")
        logger.info(f"      Respiratory: {len(resp_data):,}")
        logger.info(f"      GCS: {len(gcs_data):,}")

        map_g = {sid: g for sid, g in map_data.groupby('stay_id')}
        lab_g = {sid: g for sid, g in lab_data.groupby('stay_id')}
        vaso_g = {sid: g for sid, g in vaso_data.groupby('stay_id')}
        resp_g = {sid: g for sid, g in resp_data.groupby('stay_id')}
        gcs_g = {sid: g for sid, g in gcs_data.groupby('stay_id')}

        empty = pd.DataFrame()
        sofa_scores = []
        for sid in df['stay_id']:
            sofa = self._calculate_sofa_components(
                map_data=map_g.get(sid, empty),
                lab_data=lab_g.get(sid, empty),
                vaso_data=vaso_g.get(sid, empty),
                resp_data=resp_g.get(sid, empty),
                gcs_data=gcs_g.get(sid, empty),
            )
            complete_count = sum([
                1 if sid in map_g else 0,
                1 if sid in lab_g else 0,
                1 if sid in vaso_g else 0,
                1 if sid in resp_g else 0,
                1 if sid in gcs_g else 0,
            ])
            sofa_scores.append({
                'stay_id': sid,
                'sofa_feature': sum(sofa.values()),
                'sofa_feature_complete': complete_count,
                **{f'sofa_feature_{k}': v for k, v in sofa.items()},
            })

        sofa_df = pd.DataFrame(sofa_scores)
        df = df.merge(sofa_df, on='stay_id', how='left')

        sofa_cols = [c for c in df.columns
                     if c.startswith('sofa_feature_')
                     and c != 'sofa_feature_complete']
        for col in sofa_cols:
            df[col] = df[col].fillna(0)
        df['sofa_feature'] = df['sofa_feature'].fillna(0)
        df['sofa_feature_complete'] = (df['sofa_feature_complete']
                                       .fillna(0).astype(int))
        return df

    # ============================================================
    # OUTCOME SOFA (6-18h)
    # ============================================================

    def _calculate_outcome_sofa(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Calculating outcome SOFA "
                    f"({PREDICTION_START_HOUR}-{PREDICTION_END_HOUR}h)...")

        stay_ids = df['stay_id'].tolist()

        map_data = self.data_loader.get_outcome_window_chart_events(
            stay_ids=stay_ids, item_ids=MAP_ITEMIDS,
        )
        lab_data = self.data_loader.get_outcome_window_labs(
            stay_ids=stay_ids, item_ids=list(SOFA_LAB_ITEM_IDS.values()),
        )
        vaso_data = self.data_loader.get_outcome_window_vasopressors(
            stay_ids=stay_ids,
        )
        resp_item_ids = list(SOFA_RESPIRATORY_ITEM_IDS.values()) + [
            RESPIRATORY_ITEM_IDS['spo2']
        ]
        resp_data = self.data_loader.get_outcome_window_chart_events(
            stay_ids=stay_ids, item_ids=resp_item_ids,
        )
        gcs_ids = [v for v in GCS_ITEM_IDS.values() if v is not None]
        gcs_data = self.data_loader.get_outcome_window_chart_events(
            stay_ids=stay_ids, item_ids=gcs_ids,
        )

        logger.info(f"      MAP: {len(map_data):,}")
        logger.info(f"      Labs: {len(lab_data):,}")
        logger.info(f"      Vasopressors: {len(vaso_data):,}")
        logger.info(f"      Respiratory: {len(resp_data):,}")
        logger.info(f"      GCS: {len(gcs_data):,}")

        map_g = {sid: g for sid, g in map_data.groupby('stay_id')}
        lab_g = {sid: g for sid, g in lab_data.groupby('stay_id')}
        vaso_g = {sid: g for sid, g in vaso_data.groupby('stay_id')}
        resp_g = {sid: g for sid, g in resp_data.groupby('stay_id')}
        gcs_g = {sid: g for sid, g in gcs_data.groupby('stay_id')}

        empty = pd.DataFrame()
        sofa_scores = []
        for sid in df['stay_id']:
            sofa = self._calculate_sofa_components(
                map_data=map_g.get(sid, empty),
                lab_data=lab_g.get(sid, empty),
                vaso_data=vaso_g.get(sid, empty),
                resp_data=resp_g.get(sid, empty),
                gcs_data=gcs_g.get(sid, empty),
            )
            complete_count = sum([
                1 if sid in map_g else 0,
                1 if sid in lab_g else 0,
                1 if sid in vaso_g else 0,
                1 if sid in resp_g else 0,
                1 if sid in gcs_g else 0,
            ])
            sofa_scores.append({
                'stay_id': sid,
                'sofa_outcome': sum(sofa.values()),
                'sofa_outcome_complete': complete_count,
                **{f'sofa_outcome_{k}': v for k, v in sofa.items()},
            })

        sofa_df = pd.DataFrame(sofa_scores)
        df = df.merge(sofa_df, on='stay_id', how='left')

        sofa_cols = [c for c in df.columns
                     if c.startswith('sofa_outcome_')
                     and c != 'sofa_outcome_complete']
        for col in sofa_cols:
            df[col] = df[col].fillna(0)
        df['sofa_outcome'] = df['sofa_outcome'].fillna(0)
        df['sofa_outcome_complete'] = (df['sofa_outcome_complete']
                                       .fillna(0).astype(int))
        return df

    # ============================================================
    # SOFA COMPONENT CALCULATION
    # ============================================================

    def _calculate_sofa_components(self,
                                   map_data: pd.DataFrame,
                                   lab_data: pd.DataFrame,
                                   vaso_data: pd.DataFrame,
                                   resp_data: Optional[pd.DataFrame] = None,
                                   gcs_data: Optional[pd.DataFrame] = None) -> Dict[str, int]:
        sofa = {
            'cardiovascular': 0,
            'respiratory': 0,
            'renal': 0,
            'liver': 0,
            'coagulation': 0,
            'neurological': 0,
        }

        if map_data is None or not isinstance(map_data, pd.DataFrame) or map_data.empty:
            map_data = pd.DataFrame(columns=['itemid', 'valuenum'])
        if lab_data is None or not isinstance(lab_data, pd.DataFrame) or lab_data.empty:
            lab_data = pd.DataFrame(columns=['itemid', 'valuenum'])
        if vaso_data is None or not isinstance(vaso_data, pd.DataFrame) or vaso_data.empty:
            vaso_data = pd.DataFrame(columns=['itemid', 'rate'])
        if resp_data is None or not isinstance(resp_data, pd.DataFrame) or resp_data.empty:
            resp_data = pd.DataFrame(columns=['itemid', 'valuenum'])
        if gcs_data is None or not isinstance(gcs_data, pd.DataFrame) or gcs_data.empty:
            gcs_data = pd.DataFrame(columns=['itemid', 'valuenum'])

        # ---- 1. Cardiovascular ----
        scores = []
        if not vaso_data.empty:
            vaso_types = vaso_data['itemid'].unique()
            NE = VASOPRESSORS['norepinephrine']
            EPI = VASOPRESSORS['epinephrine']
            DOPA = VASOPRESSORS['dopamine']
            DOBU = VASOPRESSORS['dobutamine']

            if NE in vaso_types or EPI in vaso_types:
                scores.append(3)
            if DOPA in vaso_types:
                dopa_rows = vaso_data[vaso_data['itemid'] == DOPA]
                dopa_rates = dopa_rows['rate'].dropna()
                if len(dopa_rates) == 0:
                    scores.append(2)
                elif (dopa_rates > 15).any():
                    scores.append(4)
                elif (dopa_rates > 5).any():
                    scores.append(3)
                else:
                    scores.append(2)
            if DOBU in vaso_types:
                scores.append(2)

        if scores:
            sofa['cardiovascular'] = max(scores)
        elif not map_data.empty:
            min_map = map_data['valuenum'].min()
            if min_map < 70:
                sofa['cardiovascular'] = 1

        # ---- 2. Respiratory ----
        if resp_data is not None and not resp_data.empty:
            PAO2 = RESPIRATORY_ITEM_IDS['pao2']
            FIO2 = RESPIRATORY_ITEM_IDS['fio2']
            SPO2 = RESPIRATORY_ITEM_IDS['spo2']

            pao2 = resp_data[resp_data['itemid'] == PAO2]
            fio2 = resp_data[resp_data['itemid'] == FIO2]
            spo2 = resp_data[resp_data['itemid'] == SPO2]

            fio2_val = None
            if not fio2.empty:
                fio2_val = fio2['valuenum'].max()
                if fio2_val > 1.0:
                    fio2_val /= 100.0

            if fio2_val and fio2_val > 0:
                if not pao2.empty:
                    pao2_val = pao2['valuenum'].min()
                    pf = pao2_val / fio2_val
                    if pf >= 400: sofa['respiratory'] = 0
                    elif pf >= 300: sofa['respiratory'] = 1
                    elif pf >= 200: sofa['respiratory'] = 2
                    elif pf >= 100: sofa['respiratory'] = 3
                    else: sofa['respiratory'] = 4
                elif not spo2.empty:
                    spo2_val = spo2['valuenum'].min()
                    sf = spo2_val / fio2_val
                    if sf >= 235: sofa['respiratory'] = 0
                    elif sf >= 200: sofa['respiratory'] = 1
                    elif sf >= 150: sofa['respiratory'] = 2
                    elif sf >= 100: sofa['respiratory'] = 3
                    else: sofa['respiratory'] = 4

        # ---- 3. Renal ----
        creat = lab_data[lab_data['itemid'] == SOFA_LAB_ITEM_IDS['creatinine']]
        if not creat.empty:
            v = creat['valuenum'].max()
            if v < 1.2: sofa['renal'] = 0
            elif v < 2.0: sofa['renal'] = 1
            elif v < 3.5: sofa['renal'] = 2
            elif v < 5.0: sofa['renal'] = 3
            else: sofa['renal'] = 4

        # ---- 4. Liver ----
        bili = lab_data[lab_data['itemid'] == SOFA_LAB_ITEM_IDS['bilirubin']]
        if not bili.empty:
            v = bili['valuenum'].max()
            if v < 1.2: sofa['liver'] = 0
            elif v < 2.0: sofa['liver'] = 1
            elif v < 6.0: sofa['liver'] = 2
            elif v < 12.0: sofa['liver'] = 3
            else: sofa['liver'] = 4

        # ---- 5. Coagulation ----
        plt = lab_data[lab_data['itemid'] == SOFA_LAB_ITEM_IDS['platelets']]
        if not plt.empty:
            v = plt['valuenum'].min()
            if v >= 150: sofa['coagulation'] = 0
            elif v >= 100: sofa['coagulation'] = 1
            elif v >= 50: sofa['coagulation'] = 2
            elif v >= 20: sofa['coagulation'] = 3
            else: sofa['coagulation'] = 4

        # ---- 6. Neurological ----
        if gcs_data is not None and not gcs_data.empty:
            eyes_id = GCS_ITEM_IDS.get('gcs_eyes')
            verbal_id = GCS_ITEM_IDS.get('gcs_verbal')
            motor_id = GCS_ITEM_IDS.get('gcs_motor')

            eyes = gcs_data[gcs_data['itemid'] == eyes_id] if eyes_id else pd.DataFrame()
            verbal = gcs_data[gcs_data['itemid'] == verbal_id] if verbal_id else pd.DataFrame()
            motor = gcs_data[gcs_data['itemid'] == motor_id] if motor_id else pd.DataFrame()

            if not eyes.empty and not verbal.empty and not motor.empty:
                e_min = eyes['valuenum'].min()
                v_min = verbal['valuenum'].min()
                m_min = motor['valuenum'].min()

                if v_min == 0:
                    v_min = 1

                if 1 <= e_min <= 4 and 1 <= v_min <= 5 and 1 <= m_min <= 6:
                    total = e_min + v_min + m_min
                    if total >= 15: sofa['neurological'] = 0
                    elif total >= 13: sofa['neurological'] = 1
                    elif total >= 10: sofa['neurological'] = 2
                    elif total >= 6: sofa['neurological'] = 3
                    else: sofa['neurological'] = 4

        return sofa

    # ============================================================
    # OUTCOME LABELS
    # ============================================================

    def _create_outcome_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating outcome labels...")
        logger.info(f"  TARGET: Pure ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD} at "
                    f"{PREDICTION_START_HOUR}-{PREDICTION_END_HOUR}h "
                    f"(NO mortality)")

        df['sofa_change'] = df['sofa_outcome'] - df['sofa_feature']
        df['outcome'] = (df['sofa_change'] >= SOFA_CHANGE_THRESHOLD).astype(int)

        counts = df['outcome'].value_counts()
        logger.info(f"\n  OUTCOME DISTRIBUTION (pre-exclusion):")
        logger.info(f"    Stable (0): {counts.get(0, 0):,} "
                    f"({counts.get(0, 0) / len(df) * 100:.1f}%)")
        logger.info(f"    Progression (1): {counts.get(1, 0):,} "
                    f"({counts.get(1, 0) / len(df) * 100:.1f}%)")

        return df

    # ============================================================
    # DEATH EXCLUSION
    # ============================================================

    def _exclude_early_death(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Excluding deaths ≤ {PREDICTION_END_HOUR}h...")

        intime_dt = pd.to_datetime(df['intime'], errors='coerce')
        deathtime_dt = pd.to_datetime(df['deathtime'], errors='coerce')

        early_mask = (
            (df['hospital_expire_flag'] == 1) &
            (deathtime_dt.notna()) &
            (deathtime_dt <= intime_dt + pd.Timedelta(hours=PREDICTION_START_HOUR))
        )
        late_mask = (
            (df['hospital_expire_flag'] == 1) &
            (deathtime_dt.notna()) &
            (deathtime_dt > intime_dt + pd.Timedelta(hours=PREDICTION_START_HOUR)) &
            (deathtime_dt <= intime_dt + pd.Timedelta(hours=PREDICTION_END_HOUR))
        )

        n_early = int(early_mask.sum())
        n_late = int(late_mask.sum())
        logger.info(f"      Early deaths (≤{PREDICTION_START_HOUR}h): {n_early:,}")
        logger.info(f"      Late deaths "
                    f"({PREDICTION_START_HOUR}-{PREDICTION_END_HOUR}h): {n_late:,}")

        high_risk = df[df['sofa_feature'] >= 4]
        logger.info(f"      HIGH-RISK (SOFA ≥ 4): {len(high_risk):,} (NOT excluded)")

        all_deaths = early_mask | late_mask
        excluded = df[all_deaths].copy()
        kept = df[~all_deaths].copy()

        self.exclusion_counts['early_death'] = n_early
        self.exclusion_counts['late_death'] = n_late
        self.exclusion_counts['total_deaths_excluded'] = n_early + n_late

        logger.info(f"      Total deaths excluded: {n_early + n_late:,}")

        if 'outcome' in kept.columns:
            counts = kept['outcome'].value_counts()
            logger.info(f"\n  OUTCOME DISTRIBUTION (post-exclusion):")
            logger.info(f"    Stable (0): {counts.get(0, 0):,} "
                        f"({counts.get(0, 0) / len(kept) * 100:.1f}%)")
            logger.info(f"    Progression (1): {counts.get(1, 0):,} "
                        f"({counts.get(1, 0) / len(kept) * 100:.1f}%)")

        return kept, excluded

    # ============================================================
    # ANALYSIS
    # ============================================================

    def _analyze_cohort(self, cohort: pd.DataFrame):
        logger.info("\n" + "=" * 60)
        logger.info("COHORT CHARACTERISTICS")
        logger.info("=" * 60)

        logger.info(f"\n[COHORT SIZE]")
        logger.info(f"  Total stays: {len(cohort):,}")
        logger.info(f"  Unique patients: {cohort['subject_id'].nunique():,}")

        if 'outcome' in cohort.columns:
            rate = cohort['outcome'].mean() * 100
            logger.info(f"\n[TARGET]")
            logger.info(f"  Progression (ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD}): {rate:.1f}%")

        if 'sofa_feature' in cohort.columns:
            logger.info(f"\n[SOFA SCORES]")
            logger.info(f"  Feature SOFA (0-{FEATURE_WINDOW_HOURS}h): "
                        f"{cohort['sofa_feature'].mean():.1f} ± "
                        f"{cohort['sofa_feature'].std():.1f}")

        if 'sofa_outcome' in cohort.columns:
            logger.info(f"  Outcome SOFA "
                        f"({PREDICTION_START_HOUR}-{PREDICTION_END_HOUR}h): "
                        f"{cohort['sofa_outcome'].mean():.1f} ± "
                        f"{cohort['sofa_outcome'].std():.1f}")

        if 'sofa_change' in cohort.columns:
            logger.info(f"  SOFA Change: "
                        f"{cohort['sofa_change'].mean():.1f} ± "
                        f"{cohort['sofa_change'].std():.1f}")

        for window_label, prefix in [("FEATURE WINDOW", 'sofa_feature_'),
                                     ("OUTCOME WINDOW", 'sofa_outcome_')]:
            logger.info(f"\n[SOFA COMPONENTS — {window_label}]")
            for organ in ['cardiovascular', 'respiratory', 'renal',
                          'liver', 'coagulation', 'neurological']:
                col = f'{prefix}{organ}'
                if col in cohort.columns:
                    nonzero = (cohort[col] > 0).sum() / len(cohort) * 100
                    logger.info(f"  {organ:15s}: mean={cohort[col].mean():.2f}, "
                                f"nonzero={nonzero:.1f}%")

        logger.info(f"\n[SOFA DATA COMPLETENESS]")
        for window_label, prefix in [('Feature', 'sofa_feature_'),
                                     ('Outcome', 'sofa_outcome_')]:
            col = f'{prefix}complete'
            if col in cohort.columns:
                logger.info(f"  {window_label} window (0-5 sources):")
                dist = cohort[col].value_counts().sort_index()
                for n_src, count in dist.items():
                    logger.info(f"    {n_src}/5 sources: {count:,} "
                                f"({count / len(cohort) * 100:.1f}%)")

        for window_label, prefix in [('Feature', 'sofa_feature_'),
                                     ('Outcome', 'sofa_outcome_')]:
            col = f'{prefix}complete'
            if col in cohort.columns:
                n_none = int((cohort[col] == 0).sum())
                pct = n_none / len(cohort) * 100
                if pct > 1:
                    logger.warning(f"  ⚠️ {window_label}: {n_none:,} stays "
                                   f"({pct:.1f}%) have NO SOFA data "
                                   f"(filled with 0)")

        logger.info("\n[EXCLUSIONS SUMMARY]")
        for key, value in self.exclusion_counts.items():
            logger.info(f"  {key}: {value:,}")

        logger.info("=" * 60)

    # ============================================================
    # SAVE
    # ============================================================

    def save_cohort(self, output_path: Optional[str] = None):
        if self.cohort is None:
            raise ValueError("Cohort not built yet.")

        if output_path is None:
            from config import TABLE_DIR
            output_path = TABLE_DIR / "cohort.csv"

        self.cohort.to_csv(output_path, index=False)
        logger.info(f"✅ Cohort saved: {output_path}")

        summary_path = str(output_path).replace('.csv', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("COHORT SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"FEATURE WINDOW: 0-{FEATURE_WINDOW_HOURS}h from ICU admission\n")
            f.write(f"OUTCOME WINDOW: {PREDICTION_START_HOUR}-{PREDICTION_END_HOUR}h "
                    f"from ICU admission\n")
            f.write(f"TARGET: Pure ΔSOFA ≥ {SOFA_CHANGE_THRESHOLD} "
                    f"(NO mortality)\n")
            f.write(f"TRANSFER EXCLUSION (Req 1 & 3): "
                    f"{EXCLUDE_INTERHOSPITAL_TRANSFERS}\n")
            f.write("⚠️  NO DATA LEAKAGE\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total stays: {len(self.cohort):,}\n")
            f.write(f"Unique patients: {self.cohort['subject_id'].nunique():,}\n")
            if 'outcome' in self.cohort.columns:
                rate = self.cohort['outcome'].mean() * 100
                f.write(f"\nProgression Rate (pure ΔSOFA): {rate:.1f}%\n")
            if 'sofa_feature' in self.cohort.columns:
                f.write(f"\nFeature SOFA: "
                        f"{self.cohort['sofa_feature'].mean():.1f} "
                        f"± {self.cohort['sofa_feature'].std():.1f}\n")
            f.write("\nExclusions:\n")
            for k, v in self.exclusion_counts.items():
                f.write(f"  {k}: {v:,}\n")
            f.write("\nMETHODS NOTES:\n")
            f.write("  - Inter-hospital transfers excluded (Req 1 & 3).\n")
            f.write("  - Ward-to-ICU patients KEPT (Req 2, documented caveat).\n")
            f.write("  - SOFA uses 'worst value in window' rule.\n")
            f.write("  - GCS total: verbal=0 (intubated) treated as verbal=1.\n")
            f.write("\nSURVIVORSHIP BIAS:\n")
            f.write("  Patients who died on or before hour 18 were excluded.\n")
            f.write("  This introduces a survivorship bias documented in the paper.\n")

        logger.info(f"✅ Summary saved: {summary_path}")
