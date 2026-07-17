"""
Prediction Routes
"""

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime
import random
import logging

from app import auth, schemas
from app.database import Database
from app.ml_model import ml_model

router = APIRouter()
logger = logging.getLogger(__name__)

def get_db():
    return Database.get_db()


@router.post("/public", response_model=schemas.PredictionResponse)
async def predict_public(patient_data: schemas.PatientData):
    """Public prediction endpoint"""
    db = get_db()
    
    logger.info(f"📊 Prediction request for: {patient_data.patient_id}")
    
    try:
        # Calculate risk score
        risk_score = calculate_risk_score(patient_data)
        
        if risk_score > 0.7:
            alert_level = "CRITICAL"
            confidence = "HIGH"
        elif risk_score > 0.5:
            alert_level = "WARNING"
            confidence = "MEDIUM"
        else:
            alert_level = "STABLE"
            confidence = "LOW"
        
        features = ['heart_rate', 'sbp', 'dbp', 'gcs', 'lactate', 'urine_output', 'fio2', 'creatinine']
        
        # Save to database
        try:
            prediction_doc = {
                "patient_id": patient_data.patient_id,
                "patient_name": patient_data.patient_name,
                "user_id": "public",
                "risk_score": risk_score,
                "risk_percentage": risk_score * 100,
                "alert_level": alert_level,
                "confidence": confidence,
                "features": patient_data.dict(),
                "is_high_risk": risk_score > 0.5,
                "created_at": datetime.utcnow()
            }
            db.predictions.insert_one(prediction_doc)
            logger.info("✅ Prediction saved to database")
        except Exception as e:
            logger.warning(f"Could not save prediction: {e}")
        
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
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def calculate_risk_score(data: schemas.PatientData) -> float:
    """Calculate risk score based on clinical parameters"""
    score = 0.0
    
    # Heart Rate
    if data.heart_rate > 100 or data.heart_rate < 60:
        score += 0.15
    elif data.heart_rate > 90:
        score += 0.08
    
    # Blood Pressure
    if data.sbp < 90:
        score += 0.20
    elif data.sbp < 100:
        score += 0.10
    
    # GCS
    if data.gcs < 9:
        score += 0.25
    elif data.gcs < 13:
        score += 0.15
    
    # Lactate
    if data.lactate > 4:
        score += 0.25
    elif data.lactate > 2:
        score += 0.10
    
    # Creatinine
    if data.creatinine > 2:
        score += 0.15
    elif data.creatinine > 1.5:
        score += 0.10
    
    # FiO2
    if data.fio2 > 60:
        score += 0.15
    elif data.fio2 > 40:
        score += 0.08
    
    # Urine Output
    if data.urine_output < 20:
        score += 0.10
    elif data.urine_output < 40:
        score += 0.05
    
    # Add some randomness
    score += random.uniform(-0.05, 0.05)
    
    return max(0.0, min(1.0, score))


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
