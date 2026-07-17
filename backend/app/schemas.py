"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional, List

# ============================================================
# Auth Schemas
# ============================================================

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str = Field(..., min_length=2)

class UserLogin(BaseModel):
    username: str
    password: str

class GoogleAuthRequest(BaseModel):
    id_token: str
    access_token: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    full_name: str
    role: str
    profile_picture: Optional[str] = None
    created_at: datetime
    last_login: Optional[datetime] = None

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[str] = None

# ============================================================
# Prediction Schemas
# ============================================================

class PatientData(BaseModel):
    patient_id: str
    patient_name: Optional[str] = None
    heart_rate: float
    sbp: float
    dbp: float
    gcs: float
    lactate: float
    urine_output: float
    fio2: float
    creatinine: float

class PredictionResponse(BaseModel):
    patient_id: str
    risk_score: float
    risk_percentage: float
    alert_level: str
    confidence: str
    features_used: List[str]
    predicted_at: datetime

class PredictionHistoryResponse(BaseModel):
    id: str
    patient_id: str
    risk_score: float
    alert_level: str
    created_at: datetime

# ============================================================
# Dashboard Schemas
# ============================================================

class DashboardStats(BaseModel):
    total_predictions: int
    high_risk_patients: int
    avg_risk_score: float
    critical_alerts: int
    recent_predictions: List[PredictionHistoryResponse]
    last_updated: str
