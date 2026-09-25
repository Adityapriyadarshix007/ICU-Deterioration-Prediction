"""
Prediction Routes
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
    # Running from inside backend/app/
    from ml_model import ml_model

router = APIRouter()
logger = logging.getLogger(__name__)

def get_db():
    return Database.get_db()


@router.post("/public", response_model=schemas.PredictionResponse)
async def predict_public(
    patient_data: schemas.PatientData,
    current_user: dict = Depends(auth.get_current_user)
):
    """Public prediction endpoint - Now requires authentication"""
    db = get_db()
    
    logger.info(f"📊 Prediction request for: {patient_data.patient_id}")
    logger.info(f"📊 Patient data: {patient_data.dict()}")
    logger.info(f"👤 User: {current_user.get('username', 'unknown')}")
    
    try:
        # Use the ML model to make prediction
        risk_score = ml_model.predict(patient_data)
        
        # Ensure risk_score is within bounds
        risk_score = max(0.0, min(1.0, risk_score))
        
        if risk_score > 0.7:
            alert_level = "CRITICAL"
            confidence = "HIGH"
        elif risk_score > 0.5:
            alert_level = "HIGH"
            confidence = "MEDIUM"
        elif risk_score > 0.3:
            alert_level = "MEDIUM"
            confidence = "LOW"
        else:
            alert_level = "LOW"
            confidence = "LOW"
        
        features = ['heart_rate', 'sbp', 'dbp', 'gcs', 'lactate', 'urine_output', 'fio2', 'creatinine']
        
        # Get user info
        user_id = current_user.get("_id")
        username = current_user.get("username", "unknown")
        
        # Save prediction to database
        try:
            prediction_doc = {
                "patient_id": patient_data.patient_id,
                "patient_name": patient_data.patient_name,
                "user_id": str(user_id) if user_id else "unknown",
                "username": username,
                "risk_score": risk_score,
                "risk_percentage": risk_score * 100,
                "alert_level": alert_level,
                "confidence": confidence,
                "features": patient_data.dict(),
                "is_high_risk": risk_score > 0.5,
                "created_at": datetime.utcnow()
            }
            result = db.predictions.insert_one(prediction_doc)
            prediction_id = str(result.inserted_id)
            logger.info(f"✅ Prediction saved to database with ID: {prediction_id}")
            
            # ============================================================
            # CRITICAL FIX: Also save to logs collection for activity tracking
            # ============================================================
            log_doc = {
                "user_id": str(user_id) if user_id else "unknown",
                "username": username,
                "action": "PREDICTION",
                "patient_id": patient_data.patient_id,
                "patient_name": patient_data.patient_name,
                "alert_level": alert_level,
                "risk_score": risk_score,
                "prediction_id": prediction_id,
                "created_at": datetime.utcnow()
            }
            db.logs.insert_one(log_doc)
            logger.info(f"✅ Activity log saved for user: {username}")
            
        except Exception as e:
            logger.warning(f"Could not save prediction: {e}")
        
        # ============================================================
        # Create or update patient record
        # ============================================================
        try:
            patient_update = {
                "patient_id": patient_data.patient_id,
                "patient_name": patient_data.patient_name,
                "risk_level": alert_level,
                "updated_at": datetime.utcnow(),
                "vitals": {
                    "heart_rate": patient_data.heart_rate,
                    "sbp": patient_data.sbp,
                    "dbp": patient_data.dbp,
                    "gcs": patient_data.gcs,
                    "lactate": patient_data.lactate,
                    "urine_output": patient_data.urine_output,
                    "fio2": patient_data.fio2,
                    "creatinine": patient_data.creatinine
                }
            }
            
            if hasattr(patient_data, 'age') and patient_data.age:
                patient_update["age"] = patient_data.age
            if hasattr(patient_data, 'gender') and patient_data.gender:
                patient_update["gender"] = patient_data.gender
            if hasattr(patient_data, 'room') and patient_data.room:
                patient_update["room"] = patient_data.room
            if hasattr(patient_data, 'diagnosis') and patient_data.diagnosis:
                patient_update["diagnosis"] = patient_data.diagnosis
            if hasattr(patient_data, 'status') and patient_data.status:
                patient_update["status"] = patient_data.status
            else:
                patient_update["status"] = "Active"
            
            db.patients.update_one(
                {"patient_id": patient_data.patient_id},
                {
                    "$set": patient_update,
                    "$setOnInsert": {
                        "created_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
            logger.info(f"✅ Patient updated: {patient_data.patient_id}")
                
        except Exception as e:
            logger.warning(f"Could not create/update patient: {e}")
        
        return schemas.PredictionResponse(
            patient_id=patient_data.patient_id,
            risk_score=round(risk_score, 4),
            risk_percentage=round(risk_score * 100, 2),
            alert_level=alert_level,
            confidence=confidence,
            features_used=features,
            predicted_at=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/patient/{patient_id}")
async def get_patient_predictions(
    patient_id: str,
    limit: int = 10,
    current_user: dict = Depends(auth.get_current_user)
):
    """Get prediction history for a patient"""
    db = get_db()
    cursor = db.predictions.find({"patient_id": patient_id}).sort("created_at", -1).limit(limit)
    result = []
    for p in cursor:
        p["_id"] = str(p["_id"])
        result.append(p)
    return result


@router.get("/recent")
async def get_recent_predictions(
    limit: int = 20,
    current_user: dict = Depends(auth.get_current_user)
):
    """Get recent predictions"""
    db = get_db()
    cursor = db.predictions.find().sort("created_at", -1).limit(limit)
    result = []
    for p in cursor:
        p["_id"] = str(p["_id"])
        result.append(p)
    return result

# ============================================================
# Single-patient prediction by stay_id (research demo path)
# ============================================================

@router.post("/patient/{stay_id}/predict")
async def predict_by_stay_id(
    stay_id: int,
    current_user: dict = Depends(auth.get_current_user),
):
    """
    Pull the patient's ICU stay, run Phase-3 feature extraction + model,
    return probability + top SHAP contributions.

    This is the "one-click" clinical path: caller only supplies stay_id.
    """
    import sys
    from pathlib import Path
    _here = Path(__file__).resolve().parent.parent   # backend/app/
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

    # Persist to MongoDB for history/recent views
    db = get_db()
    try:
        db.predictions.insert_one({
            "patient_id": str(stay_id),
            "stay_id": stay_id,
            "user_id": str(current_user.get("_id", "unknown")),
            "username": current_user.get("username", "unknown"),
            "risk_score": result["probability"],
            "risk_percentage": result["probability"] * 100,
            "alert_level": result["risk_band"],
            "features": result.get("top_features", []),
            "created_at": datetime.utcnow(),
        })
    except Exception as e:
        logger.warning(f"Could not save prediction: {e}")

    return result


# ============================================================
# Unified predict endpoint — routes BOTH form and stay-id
# sources through the same SinglePatientPredictor.predict_from_dict
# ============================================================

@router.post("/unified")
async def predict_unified(
    patient_data: schemas.PatientData,
    current_user: dict = Depends(auth.get_current_user),
):
    """
    Unified prediction endpoint.

    Takes the same PatientData schema the manual form already sends,
    converts it to model features via MLModel._build_model_input, then
    runs it through the same SinglePatientPredictor.predict_from_dict
    used by the stay-id flow. Result: manual form and stay-id flow
    produce consistent probabilities for identical inputs.
    """
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

    # Persist for history
    db = get_db()
    try:
        db.predictions.insert_one({
            "patient_id": patient_data.patient_id,
            "patient_name": patient_data.patient_name,
            "user_id": str(current_user.get("_id", "unknown")),
            "username": current_user.get("username", "unknown"),
            "risk_score": result["probability"],
            "risk_percentage": result["probability"] * 100,
            "alert_level": result["risk_band"],
            "features": result.get("top_features", []),
            "created_at": datetime.utcnow(),
        })
    except Exception as e:
        logger.warning(f"Could not save prediction: {e}")

    return result


@router.get("/patient/{stay_id}/features")
async def get_patient_features(
    stay_id: int,
    current_user: dict = Depends(auth.get_current_user),
):
    """
    Return the ~40 form-relevant features for a stay, keyed by form field name.
    Used by the frontend "Load from chart" button: clicking it populates the
    manual form with real values from the ICU chart, which the clinician can
    then edit before predicting.
    """
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

    # Map model feature name -> form field name (reverse of FIELD_MAP)
    reverse = {}
    for form_field, (model_col, _tfm) in FIELD_MAP.items():
        reverse[model_col] = form_field

    result = {}
    for model_col, form_field in reverse.items():
        if model_col in features_df.columns:
            val = features_df[model_col].iloc[0]
            if pd.notna(val):
                result[form_field] = float(val)

    return {"stay_id": stay_id, "features": result}
