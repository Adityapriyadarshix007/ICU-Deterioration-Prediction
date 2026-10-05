"""
Prediction Routes
Writes consistent documents to predictions, patients, and logs.
stay_id is null for manual entry, integer for stay-ID flow.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime
import random
import logging

import pandas as pd
from app import auth, schemas
from app.database import Database
try:
    from app.ml_model import ml_model
except ImportError:
    from ml_model import ml_model

router = APIRouter()
logger = logging.getLogger(__name__)


def get_db():
    return Database.get_db()


# ============================================================
# Shared persistence helper
# Writes ONE prediction event to predictions + patients + logs
# with a consistent stay_id field (int or null).
# ============================================================
def _persist_prediction(
    db,
    *,
    patient_id: str,
    patient_name: str | None,
    stay_id: int | None,
    user_id: str,
    username: str,
    risk_score: float,
    alert_level: str,
    confidence: str,
    features,
    vitals: dict | None = None,
    patient_meta: dict | None = None,
):
    now = datetime.utcnow()

    # ---- 1. predictions ----
    pred_doc = {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "stay_id": stay_id,                       # always present
        "user_id": user_id,
        "username": username,
        "risk_score": risk_score,
        "risk_percentage": risk_score * 100,
        "alert_level": alert_level,
        "confidence": confidence,
        "features": features,
        "is_high_risk": risk_score > 0.5,
        "created_at": now,
    }
    pred_result = db.predictions.insert_one(pred_doc)
    pred_id = str(pred_result.inserted_id)
    logger.info(f"✅ Prediction saved: {pred_id}")

    # ---- 2. patients (upsert) ----
    patient_update = {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "stay_id": stay_id,                       # always present
        "risk_level": alert_level,
        "updated_at": now,
    }
    if vitals:
        patient_update["vitals"] = vitals
    if patient_meta:
        for k in ("age", "gender", "room", "diagnosis", "status"):
            if patient_meta.get(k) is not None:
                patient_update[k] = patient_meta[k]
    patient_update.setdefault("status", "Active")

    db.patients.update_one(
        {"patient_id": patient_id},
        {
            "$set": patient_update,
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )
    logger.info(f"✅ Patient upserted: {patient_id}")

    # ---- 3. logs ----
    db.logs.insert_one({
        "user_id": user_id,
        "username": username,
        "action": "PREDICTION",
        "patient_id": patient_id,
        "patient_name": patient_name,
        "stay_id": stay_id,                       # always present
        "alert_level": alert_level,
        "risk_score": risk_score,
        "prediction_id": pred_id,
        "created_at": now,
    })
    logger.info(f"✅ Log saved for: {username}")

    return pred_id


# ============================================================
# POST /api/predict/public  — manual entry (no stay_id)
# ============================================================
@router.post("/public", response_model=schemas.PredictionResponse)
async def predict_public(
    patient_data: schemas.PatientData,
    current_user: dict = Depends(auth.get_current_user),
):
    db = get_db()
    logger.info(f"📊 Prediction request for: {patient_data.patient_id}")

    try:
        risk_score = ml_model.predict(patient_data)
        risk_score = max(0.0, min(1.0, risk_score))

        if risk_score > 0.7:
            alert_level, confidence = "CRITICAL", "HIGH"
        elif risk_score > 0.5:
            alert_level, confidence = "HIGH", "MEDIUM"
        elif risk_score > 0.3:
            alert_level, confidence = "MEDIUM", "LOW"
        else:
            alert_level, confidence = "LOW", "LOW"

        features = ['heart_rate', 'sbp', 'dbp', 'gcs', 'lactate',
                    'urine_output', 'fio2', 'creatinine']

        user_id = str(current_user.get("_id", "unknown"))
        username = current_user.get("username", "unknown")

        vitals = {
            "heart_rate": patient_data.heart_rate,
            "sbp": patient_data.sbp,
            "dbp": patient_data.dbp,
            "gcs": getattr(patient_data, "gcs_total", None) or getattr(patient_data, "gcs", None),
            "lactate": patient_data.lactate,
            "urine_output": getattr(patient_data, "urine_output", None),
            "fio2": patient_data.fio2,
            "creatinine": patient_data.creatinine,
        }
        meta = {
            "age": getattr(patient_data, "age", None),
            "gender": getattr(patient_data, "gender", None),
            "room": getattr(patient_data, "room", None),
            "diagnosis": getattr(patient_data, "diagnosis", None),
            "status": getattr(patient_data, "status", None),
        }

        try:
            _persist_prediction(
                db,
                patient_id=patient_data.patient_id,
                patient_name=patient_data.patient_name,
                stay_id=getattr(patient_data, "stay_id", None),   # usually None
                user_id=user_id,
                username=username,
                risk_score=risk_score,
                alert_level=alert_level,
                confidence=confidence,
                features=patient_data.dict(),
                vitals=vitals,
                patient_meta=meta,
            )
        except Exception as e:
            logger.error(f"❌ Persist failed: {e}", exc_info=True)

        return schemas.PredictionResponse(
            patient_id=patient_data.patient_id,
            risk_score=round(risk_score, 4),
            risk_percentage=round(risk_score * 100, 2),
            alert_level=alert_level,
            confidence=confidence,
            features_used=features,
            predicted_at=datetime.utcnow(),
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# ============================================================
# GET /api/predict/patient/{patient_id}
# ============================================================
@router.get("/patient/{patient_id}")
async def get_patient_predictions(
    patient_id: str,
    limit: int = 10,
    current_user: dict = Depends(auth.get_current_user),
):
    db = get_db()
    cursor = db.predictions.find({"patient_id": patient_id}).sort("created_at", -1).limit(limit)
    result = []
    for p in cursor:
        p["_id"] = str(p["_id"])
        result.append(p)
    return result


# ============================================================
# GET /api/predict/recent
# ============================================================
@router.get("/recent")
async def get_recent_predictions(
    limit: int = 20,
    current_user: dict = Depends(auth.get_current_user),
):
    db = get_db()
    cursor = db.predictions.find().sort("created_at", -1).limit(limit)
    result = []
    for p in cursor:
        p["_id"] = str(p["_id"])
        result.append(p)
    return result


# ============================================================
# POST /api/predict/patient/{stay_id}/predict  — stay-ID flow
# ============================================================
@router.post("/patient/{stay_id}/predict")
async def predict_by_stay_id(
    stay_id: int,
    current_user: dict = Depends(auth.get_current_user),
):
    import sys
    from pathlib import Path
    _here = Path(__file__).resolve().parent.parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    from serve_single_patient import get_single_patient_predictor

    try:
        predictor = get_single_patient_predictor()
        result = predictor.predict(stay_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    db = get_db()
    user_id = str(current_user.get("_id", "unknown"))
    username = current_user.get("username", "unknown")

    try:
        _persist_prediction(
            db,
            patient_id=str(stay_id),
            patient_name=None,
            stay_id=stay_id,                      # ← set for stay-ID flow
            user_id=user_id,
            username=username,
            risk_score=result["probability"],
            alert_level=result["risk_band"],
            confidence="MODEL",
            features=result.get("top_features", []),
        )
    except Exception as e:
        logger.error(f"❌ Persist failed: {e}", exc_info=True)

    return result


# ============================================================
# POST /api/predict/unified  — unified form path
# ============================================================
@router.post("/unified")
async def predict_unified(
    patient_data: schemas.PatientData,
    current_user: dict = Depends(auth.get_current_user),
):
    import sys
    from pathlib import Path
    _here = Path(__file__).resolve().parent.parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    from serve_single_patient import get_single_patient_predictor
    from app.ml_model import MLModel

    try:
        predictor = get_single_patient_predictor()
        adapter = MLModel()
        feature_dict = adapter._build_model_input(patient_data.dict())
        result = predictor.predict_from_dict(feature_dict)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    db = get_db()
    user_id = str(current_user.get("_id", "unknown"))
    username = current_user.get("username", "unknown")

    vitals = {
        "heart_rate": patient_data.heart_rate,
        "sbp": patient_data.sbp,
        "dbp": patient_data.dbp,
        "gcs": getattr(patient_data, "gcs_total", None) or getattr(patient_data, "gcs", None),
        "lactate": patient_data.lactate,
        "urine_output": getattr(patient_data, "urine_output", None),
        "fio2": patient_data.fio2,
        "creatinine": patient_data.creatinine,
    }
    meta = {
        "age": getattr(patient_data, "age", None),
        "gender": getattr(patient_data, "gender", None),
        "room": getattr(patient_data, "room", None),
        "diagnosis": getattr(patient_data, "diagnosis", None),
        "status": getattr(patient_data, "status", None),
    }

    try:
        _persist_prediction(
            db,
            patient_id=patient_data.patient_id,
            patient_name=patient_data.patient_name,
            stay_id=getattr(patient_data, "stay_id", None),
            user_id=user_id,
            username=username,
            risk_score=result["probability"],
            alert_level=result["risk_band"],
            confidence="MODEL",
            features=result.get("top_features", []),
            vitals=vitals,
            patient_meta=meta,
        )
    except Exception as e:
        logger.error(f"❌ Persist failed: {e}", exc_info=True)

    return result


# ============================================================
# GET /api/predict/patient/{stay_id}/features
# ============================================================
@router.get("/patient/{stay_id}/features")
async def get_patient_features(
    stay_id: int,
    current_user: dict = Depends(auth.get_current_user),
):
    import sys
    from pathlib import Path
    _here = Path(__file__).resolve().parent.parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    from serve_single_patient import extract_features_single
    from app.ml_model import FIELD_MAP

    try:
        features_df = extract_features_single(stay_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")

    reverse = {v[0]: k for k, v in FIELD_MAP.items()}
    result = {}
    for model_col, form_field in reverse.items():
        if model_col in features_df.columns:
            val = features_df[model_col].iloc[0]
            if pd.notna(val):
                result[form_field] = float(val)

    return {"stay_id": stay_id, "features": result}