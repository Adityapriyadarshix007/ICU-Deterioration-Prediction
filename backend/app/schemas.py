"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, EmailStr, Field, validator
from datetime import datetime
from typing import Optional, List

# ============================================================
# Auth Schemas
# ============================================================

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2)
    
    @validator('password')
    def validate_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number')
        return v

class UserLogin(BaseModel):
    username: str
    password: str

class GoogleAuthRequest(BaseModel):
    id_token: str
    access_token: Optional[str] = None

# Forgot Password Request
class ForgotPasswordRequest(BaseModel):
    email: EmailStr

# Reset Password Request (with token)
class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8)
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number')
        return v

# Change Password Schema (requires current password)
class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number')
        return v

# Reset Password by Email Only (no token, no current password)
class ResetPasswordEmailRequest(BaseModel):
    email: EmailStr
    new_password: str = Field(..., min_length=8)
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number')
        return v

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
    """Patient data for prediction - includes all demographics and vitals"""
    patient_id: str
    patient_name: str = Field(..., min_length=1, description="Patient's full name")
    age: int = Field(..., ge=18, le=120, description="Patient's age in years")
    gender: str = Field(..., description="Patient's gender (Male/Female/Other)")
    room: str = Field(..., min_length=1, description="ICU room number")
    diagnosis: str = Field(..., min_length=1, description="Primary diagnosis")
    heart_rate: float = Field(..., ge=0, le=300, description="Heart rate in bpm")
    sbp: float = Field(..., ge=0, le=300, description="Systolic blood pressure in mmHg")
    dbp: float = Field(..., ge=0, le=200, description="Diastolic blood pressure in mmHg")
    gcs: float = Field(..., ge=3, le=15, description="Glasgow Coma Scale score")
    lactate: float = Field(..., ge=0, le=25, description="Lactate level in mmol/L")
    urine_output: float = Field(..., ge=0, le=500, description="Urine output in mL/hr")
    fio2: float = Field(..., ge=15, le=100, description="Fraction of inspired oxygen (%)")
    creatinine: float = Field(..., ge=0, le=20, description="Creatinine level in mg/dL")
    status: Optional[str] = Field(default="Active", description="Patient status")
    
    @validator('gender')
    def validate_gender(cls, v):
        allowed = ['Male', 'Female', 'Other']
        if v not in allowed:
            raise ValueError(f'Gender must be one of {allowed}')
        return v
    
    @validator('gcs')
    def validate_gcs(cls, v):
        if v < 3 or v > 15:
            raise ValueError('GCS must be between 3 and 15')
        return v
    
    @validator('heart_rate')
    def validate_heart_rate(cls, v):
        if v < 20 or v > 250:
            raise ValueError('Heart rate must be between 20 and 250 bpm')
        return v

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
    patient_name: Optional[str] = None
    risk_score: float
    risk_percentage: float
    alert_level: str
    confidence: Optional[str] = None
    created_at: datetime

# ============================================================
# Patient Schemas
# ============================================================

class PatientBase(BaseModel):
    patient_id: str
    patient_name: str
    age: int
    gender: str
    room: str
    diagnosis: str
    risk_level: Optional[str] = "LOW"
    status: Optional[str] = "Active"
    vitals: Optional[dict] = None

class PatientCreate(PatientBase):
    pass

class PatientUpdate(BaseModel):
    patient_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    room: Optional[str] = None
    diagnosis: Optional[str] = None
    risk_level: Optional[str] = None
    status: Optional[str] = None
    vitals: Optional[dict] = None

class PatientResponse(PatientBase):
    id: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# ============================================================
# Dashboard Schemas
# ============================================================

class DashboardStats(BaseModel):
    total_predictions: int
    high_risk_patients: int
    critical_alerts: int
    avg_risk_score: float
    recent_predictions: List[PredictionHistoryResponse]
    last_updated: str