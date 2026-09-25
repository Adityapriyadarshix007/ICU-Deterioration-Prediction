"""
Serve a single patient prediction.

Two entry points share ONE pipeline:

1. predict(stay_id)           — extract features from DuckDB for that ICU stay
2. predict_from_dict(features) — take an already-built feature dict (manual form)

Both go through _build_input() → _predict_from_matrix() → result dict.
This guarantees consistency: the same feature dict always yields the same
probability, regardless of where the features came from.
"""

from pathlib import Path
import json
import logging
import joblib
import numpy as np
import pandas as pd

from database import DatabaseManager
from phase3 import FeatureExtractor

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "outputs" / "models"


# ---------------------------------------------------------------
# DuckDB metadata lookup
# ---------------------------------------------------------------
def load_cohort_row(stay_id: int) -> pd.DataFrame:
    """Query DuckDB for one stay's metadata in the shape FeatureExtractor expects."""
    db = DatabaseManager(db_path=str(APP_DIR.parent / "mods.duckdb"))
    query = f"""
        SELECT
            ic.stay_id, ic.hadm_id, ic.subject_id,
            ic.intime, ic.outtime, ic.los,
            p.anchor_age, p.gender,
            a.admission_type, a.admission_location, a.discharge_location
        FROM icustays ic
        INNER JOIN patients   p ON ic.subject_id = p.subject_id
        INNER JOIN admissions a ON ic.hadm_id   = a.hadm_id
        WHERE ic.stay_id = {int(stay_id)}
    """
    df = db.fetch_df(query)
    db.close()
    if df.empty:
        raise ValueError(f"stay_id {stay_id} not found in DuckDB")
    df["outcome"] = None
    return df


def extract_features_single(stay_id: int, logger=None) -> pd.DataFrame:
    """Run Phase 3's extractor on one stay -> 1-row DataFrame of raw features."""
    cohort_row = load_cohort_row(stay_id)
    if logger is None:
        logger = logging.getLogger("serve_single")
        logger.setLevel(logging.WARNING)

    db = DatabaseManager(db_path=str(APP_DIR.parent / "mods.duckdb"))
    db.create_views()
    extractor = FeatureExtractor(db, logger, method="none")
    features_df = extractor.extract_all_features(cohort_row)
    db.close()
    return features_df


# ---------------------------------------------------------------
# The unified predictor
# ---------------------------------------------------------------
class SinglePatientPredictor:
    def __init__(self):
        self.model = joblib.load(MODEL_DIR / "phase6_final_model.pkl")
        with open(MODEL_DIR / "serving_pipeline.json") as f:
            self.pipe = json.load(f)
        with open(MODEL_DIR / "serving_manifest.json") as f:
            self.manifest = json.load(f)

        self.base = self.pipe["feature_cols"]     # 120
        self.masks = self.pipe["mask_cols"]       # 40

        try:
            import shap
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            self.explainer = None

    # -----------------------------------------------------------
    def _features_to_dataframe(self, features: dict) -> pd.DataFrame:
        """Convert a flat feature dict into a 1-row DataFrame."""
        return pd.DataFrame([{c: features.get(c, np.nan) for c in self.base}])

    # -----------------------------------------------------------
    def _build_input(self, features_df: pd.DataFrame) -> np.ndarray:
        """Raw features -> standardized 160-vector (120 base + 40 masks)."""
        # Align columns
        missing_cols = [c for c in self.base if c not in features_df.columns]
        for c in missing_cols:
            features_df[c] = np.nan

        x = features_df[self.base].copy()

        # Masks: 1 if feature is missing, 0 otherwise
        for mc in self.masks:
            src = mc[:-len("_missing")]
            x[mc] = x[src].isna().astype(int) if src in x.columns else 1

        # Winsorize
        for c in self.base:
            lo = self.pipe["winsor_lo"].get(c)
            hi = self.pipe["winsor_hi"].get(c)
            if lo is not None:
                x[c] = x[c].clip(lo, hi)

        # Standardize
        for c in self.base:
            if c in self.pipe["scaler"]:
                pr = self.pipe["scaler"][c]
                x[c] = (x[c] - pr["mean"]) / pr["scale"]

        # Zero-impute in standardized space
        # (matches phase6 config: imputation='zero')
        x[self.base] = x[self.base].fillna(0.0)

        return x[self.base + self.masks].astype(float).values

    # -----------------------------------------------------------
    def _predict_from_matrix(self, X: np.ndarray, top_k_shap: int = 10) -> dict:
        """160-vector -> probability, risk band, top SHAP contributions."""
        prob = float(self.model.predict_proba(X)[0, 1])
        pred = int(prob >= 0.5)

        top_features = []
        if self.explainer is not None:
            try:
                shap_vals = self.explainer.shap_values(X)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                if shap_vals.ndim == 3:
                    shap_vals = shap_vals[:, :, 1]
                contrib = shap_vals[0]
                names = self.base + self.masks
                idx = np.argsort(-np.abs(contrib))[:top_k_shap]
                top_features = [
                    {
                        "feature": names[i],
                        "contribution": float(contrib[i]),
                        "value": float(X[0, i]),
                    }
                    for i in idx
                ]
            except Exception as e:
                top_features = [{"error": str(e)}]

        if prob >= 0.7:
            band = "CRITICAL"
        elif prob >= 0.5:
            band = "HIGH"
        elif prob >= 0.3:
            band = "MEDIUM"
        else:
            band = "LOW"

        return {
            "probability": round(prob, 4),
            "prediction": pred,
            "threshold": 0.5,
            "risk_band": band,
            "top_features": top_features,
        }

    # -----------------------------------------------------------
    # UNIFIED entry point #1: from a feature dict
    # -----------------------------------------------------------
    def predict_from_dict(self, features: dict, top_k_shap: int = 10) -> dict:
        """
        Predict from a flat dict of model feature names -> values.

        This is THE core function. Both the manual form and the stay-ID path
        feed into this after building their own feature dict.

        Missing keys are treated as NaN (imputed + mask=1).
        """
        df = self._features_to_dataframe(features)
        X = self._build_input(df)
        return self._predict_from_matrix(X, top_k_shap=top_k_shap)

    # -----------------------------------------------------------
    # UNIFIED entry point #2: from a stay_id (extracts features, then calls #1)
    # -----------------------------------------------------------
    def predict(self, stay_id: int, top_k_shap: int = 10) -> dict:
        """Extract features for stay_id, then delegate to predict_from_dict."""
        features_df = extract_features_single(stay_id)
        # features_df is already a DataFrame with feature columns; convert to dict
        feature_dict = features_df.iloc[0].to_dict()
        result = self.predict_from_dict(feature_dict, top_k_shap=top_k_shap)
        result["stay_id"] = int(stay_id)
        return result


_predictor = None


def get_single_patient_predictor() -> SinglePatientPredictor:
    global _predictor
    if _predictor is None:
        _predictor = SinglePatientPredictor()
    return _predictor
