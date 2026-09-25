"""
Inference for MIMIC-IV Multi-Organ Failure prediction.

Exposes two interfaces:

1. MOFPredictor (class) — low-level pipeline for a 120-feature dict
2. ml_model (singleton) — high-level adapter that maps clinical form fields
   onto the model's expected feature set, leaving unspecified features NaN.

Both share the same underlying LGBM model and preprocessing pipeline.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).parent / "outputs" / "models"

MODEL_PATH    = MODEL_DIR / "phase6_final_model.pkl"
MANIFEST_PATH = MODEL_DIR / "serving_manifest.json"
PIPELINE_PATH = MODEL_DIR / "serving_pipeline.json"


# ==================================================================
# Low-level predictor
# ==================================================================

class MOFPredictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.manifest = json.load(open(MANIFEST_PATH))
        self.pipe = json.load(open(PIPELINE_PATH))

        self.base = self.pipe["feature_cols"]       # 120
        self.masks = self.pipe["mask_cols"]         # 40
        self.all_cols = self.base + self.masks      # 160

        assert len(self.all_cols) == self.model.n_features_in_, \
            f"column count mismatch: {len(self.all_cols)} vs {self.model.n_features_in_}"

    def predict(self, raw: dict) -> dict:
        # 1. build 120-vector (missing -> NaN)
        x = pd.DataFrame([{c: raw.get(c, np.nan) for c in self.base}])

        # 2. masks from PRE-imputation NaN pattern
        for c in self.masks:
            src = c.replace("_missing", "")
            if src not in x.columns:
                src = src.replace("_", " ")
            x[c] = x[src].isna().astype(int) if src in x.columns else 1

        # 3. winsorize
        for c in self.base:
            lo = self.pipe["winsor_lo"].get(c)
            hi = self.pipe["winsor_hi"].get(c)
            if lo is not None:
                x[c] = x[c].clip(lo, hi)

        # 4. standardize (fit on training data)
        for c in self.base:
            if c in self.pipe["scaler"]:
                m = self.pipe["scaler"][c]["mean"]
                s = self.pipe["scaler"][c]["scale"]
                x[c] = (x[c] - m) / s

        # 5. ZERO-IMPUTE in standardized space (matches deployed model config)
        #    The model was trained with phase6 config imputation='zero':
        #    missing values were filled with 0 in standardized space,
        #    equivalent to the training mean in raw units.
        x[self.base] = x[self.base].fillna(0.0)

        # 6. reorder + predict
        X = x[self.all_cols].astype(float).values
        prob = float(self.model.predict_proba(X)[0, 1])
        return {
            "probability": round(prob, 4),
            "prediction": int(prob >= 0.5),
            "threshold": 0.5,
        }

    def predict_batch(self, rows: list) -> list:
        return [self.predict(r) for r in rows]


_predictor = None


def get_predictor() -> MOFPredictor:
    global _predictor
    if _predictor is None:
        _predictor = MOFPredictor()
    return _predictor


# ==================================================================
# High-level adapter (matches legacy `ml_model.predict(patient_data)`)
# ==================================================================

# Form field -> (model_feature, transform_fn)
# Every form field is OPTIONAL. Unmapped model features are left NaN.

def _identity(x):
    return float(x)

def _fio2_frac(x):
    v = float(x)
    return v / 100.0 if v > 1.0 else v

def _any_to_1_0(x):
    if x is None:
        return np.nan
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    s = str(x).strip().lower()
    return 1.0 if s in ("1", "true", "yes", "y", "on") else 0.0


# each entry: form_field -> (model_feature_name, transform)
FIELD_MAP = {
    # Demographics
    "age":              ("age",              _identity),
    "gender_male":      ("gender_male",      _identity),   # set by adapter from `gender`

    # Vitals -> mean
    "heart_rate":        ("heart_rate_mean",        _identity),
    "respiratory_rate":  ("respiratory_rate_mean",  _identity),
    "spo2":              ("spo2_mean",              _identity),
    "temperature":       ("temperature_mean",       _identity),
    "sbp":               ("sbp_mean",               _identity),
    "dbp":               ("dbp_mean",               _identity),
    "map":               ("map_mean",               _identity),

    # Respiratory
    "fio2":              ("fio2_mean",              _fio2_frac),
    "pao2":              ("pao2_mean",              _identity),

    # Neuro
    "gcs_eyes":          ("gcs_eyes_min",           _identity),
    "gcs_verbal":        ("gcs_verbal_min",         _identity),
    "gcs_motor":         ("gcs_motor_min",          _identity),
    "gcs_total":         ("gcs_total_min",          _identity),

    # Labs -> worst
    "creatinine":        ("creatinine_worst",       _identity),
    "bun":               ("bun_worst",              _identity),
    "sodium":            ("sodium_worst",           _identity),
    "potassium":         ("potassium_worst",        _identity),
    "chloride":          ("chloride_worst",         _identity),
    "bicarbonate":       ("bicarbonate_worst",      _identity),
    "glucose":           ("glucose_worst",          _identity),
    "hemoglobin":        ("hemoglobin_worst",       _identity),
    "hematocrit":        ("hematocrit_worst",       _identity),
    "wbc":               ("wbc_worst",              _identity),
    "platelets":         ("platelets_worst",        _identity),
    "inr":               ("inr_worst",              _identity),
    "ptt":               ("ptt_worst",              _identity),
    "bilirubin":         ("bilirubin_worst",        _identity),
    "alt":               ("alt_worst",              _identity),
    "ast":               ("ast_worst",              _identity),
    "lactate":           ("lactate_worst",          _identity),

    # Vasopressors (any flag + max rate)
    "norepinephrine_any":       ("norepinephrine_any",       _any_to_1_0),
    "norepinephrine_max_rate":  ("norepinephrine_max_rate",  _identity),
    "epinephrine_any":          ("epinephrine_any",          _any_to_1_0),
    "epinephrine_max_rate":     ("epinephrine_max_rate",     _identity),
    "dopamine_any":             ("dopamine_any",             _any_to_1_0),
    "dopamine_max_rate":        ("dopamine_max_rate",        _identity),
    "dobutamine_any":           ("dobutamine_any",           _any_to_1_0),
    "dobutamine_max_rate":      ("dobutamine_max_rate",      _identity),
    "vasopressin_any":          ("vasopressin_any",          _any_to_1_0),
    "vasopressin_max_rate":     ("vasopressin_max_rate",     _identity),
    "phenylephrine_any":        ("phenylephrine_any",        _any_to_1_0),
    "phenylephrine_max_rate":   ("phenylephrine_max_rate",   _identity),
}

# All admission_* dummy columns the model has — matched by dropdown value
ADMISSION_TYPES = [
    "AMBULATORY OBSERVATION",
    "DIRECT EMER.",
    "DIRECT OBSERVATION",
    "ELECTIVE",
    "EU OBSERVATION",
    "EW EMER.",
    "OBSERVATION ADMIT",
    "SURGICAL SAME DAY ADMISSION",
    "URGENT",
]


class MLModel:
    """Adapter: clinical form fields -> model feature dict -> probability."""

    def __init__(self):
        self._predictor = None
        self.model = None   # attribute exposed for main_api.py compatibility

    def _ensure_loaded(self):
        if self._predictor is None:
            self._predictor = get_predictor()
            self.model = self._predictor.model

    @property
    def is_loaded(self) -> bool:
        try:
            self._ensure_loaded()
            return self.model is not None
        except Exception:
            return False

    # ---------------------------------------------------------
    def _to_dict(self, patient_data) -> dict:
        """Normalize Pydantic model / dict / object into a plain dict."""
        if patient_data is None:
            return {}
        if isinstance(patient_data, dict):
            return dict(patient_data)
        # Pydantic v1/v2
        if hasattr(patient_data, "model_dump"):
            return patient_data.model_dump()
        if hasattr(patient_data, "dict"):
            return patient_data.dict()
        return {k: v for k, v in vars(patient_data).items()}

    # ---------------------------------------------------------
    def _build_model_input(self, data: dict) -> dict:
        """Map form fields to the 120 base model features."""
        out = {}

        # --- demographics ---
        if "age" in data and data["age"] is not None:
            out["age"] = float(data["age"])

        gender = data.get("gender")
        if gender is not None:
            out["gender_male"] = 1.0 if str(gender).strip().lower() == "male" else 0.0

        admission = data.get("admission_type") or data.get("admission")
        if admission:
            adm = str(admission).strip().upper()
            for at in ADMISSION_TYPES:
                key = f"admission_{at}"
                out[key] = 1.0 if adm == at.upper() else 0.0

        # --- field map ---
        for field, (model_feat, tfm) in FIELD_MAP.items():
            v = data.get(field)
            if v is None or v == "":
                continue
            try:
                out[model_feat] = tfm(v)
            except Exception:
                # malformed value -> skip, pipeline will impute
                continue

        # --- GCS total derivation if components given but total not ---
        if "gcs_total_min" not in out:
            e = out.get("gcs_eyes_min")
            v = out.get("gcs_verbal_min")
            m = out.get("gcs_motor_min")
            if e is not None and v is not None and m is not None:
                out["gcs_total_min"] = float(e) + float(v) + float(m)
                # also mirror into gcs_total_last and delta for consistency
                out["gcs_total_last"] = out["gcs_total_min"]
                out["gcs_total_delta"] = 0.0

        # --- vasopressor derivation ---
        for drug in ["norepinephrine", "epinephrine", "dopamine",
                     "dobutamine", "vasopressin", "phenylephrine"]:
            rate_key = f"{drug}_max_rate"
            any_key = f"{drug}_any"
            if rate_key in out and any_key not in out:
                out[any_key] = 1.0 if out[rate_key] > 0 else 0.0
            if any_key in out and rate_key not in out and out[any_key] > 0:
                out[rate_key] = np.nan  # flag set, rate unknown

        return out

    # ---------------------------------------------------------
    def predict(self, patient_data) -> float:
        """
        Legacy interface. Returns probability as a float (0-1).

        Now routes through the unified predict_from_dict so the manual form
        and stay-ID flow share the exact same serving pipeline.
        """
        result = self.predict_full(patient_data)
        return float(result["probability"])

    def predict_full(self, patient_data) -> dict:
        """
        Same as predict() but returns the full dict.

        Routes through the unified SinglePatientPredictor.predict_from_dict
        so the manual form and stay-ID flow produce identical results for
        identical inputs.
        """
        self._ensure_loaded()

        # Build model-feature dict from the patient's form data
        data = self._to_dict(patient_data)
        model_input = self._build_model_input(data)

        # Delegate to the unified predictor
        try:
            from serve_single_patient import get_single_patient_predictor
            unified = get_single_patient_predictor()
            return unified.predict_from_dict(model_input)
        except ImportError:
            # Fall back to the old path if serve_single_patient is unavailable
            return self._predictor.predict(model_input)


# Module-level singleton for `from ml_model import ml_model`
ml_model = MLModel()
