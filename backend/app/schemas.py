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
    """Patient data for prediction - demographics + vitals + labs.

    All clinical fields are OPTIONAL. Missing fields are handled by the
    model's imputation pipeline (median fill + missingness mask).
    """

    # --- identifiers (not model features) ---
    patient_id: str
    patient_name: str = Field(..., min_length=1)

    # --- demographics ---
    age: Optional[float] = Field(default=None, ge=0, le=120)
    gender: Optional[str] = Field(default=None, description="Male / Female / Other")
    room: Optional[str] = None
    diagnosis: Optional[str] = None
    status: Optional[str] = Field(default="Active")
    admission_type: Optional[str] = Field(
        default=None,
        description=("One of: AMBULATORY OBSERVATION, DIRECT EMER., "
                     "DIRECT OBSERVATION, ELECTIVE, EU OBSERVATION, "
                     "EW EMER., OBSERVATION ADMIT, "
                     "SURGICAL SAME DAY ADMISSION, URGENT")
    )

    # --- vitals ---
    heart_rate: Optional[float] = Field(default=None, ge=0, le=300)
    respiratory_rate: Optional[float] = Field(default=None, ge=0, le=80)
    spo2: Optional[float] = Field(default=None, ge=0, le=100)
    temperature: Optional[float] = Field(default=None, ge=86, le=113,
                                          description="Fahrenheit")
    sbp: Optional[float] = Field(default=None, ge=0, le=300)
    dbp: Optional[float] = Field(default=None, ge=0, le=200)
    map: Optional[float] = Field(default=None, ge=0, le=250)

    # --- respiratory ---
    fio2: Optional[float] = Field(default=None, ge=15, le=100,
                                   description="Percent (e.g. 40 for 40%)")
    pao2: Optional[float] = Field(default=None, ge=0, le=700)

    # --- neuro (GCS components or total) ---
    gcs_eyes: Optional[float] = Field(default=None, ge=1, le=4)
    gcs_verbal: Optional[float] = Field(default=None, ge=0, le=5)
    gcs_motor: Optional[float] = Field(default=None, ge=1, le=6)
    gcs_total: Optional[float] = Field(default=None, ge=3, le=15,
                                        description="Computed if components given")

    # --- labs: chem ---
    creatinine: Optional[float] = Field(default=None, ge=0, le=30)
    bun: Optional[float] = Field(default=None, ge=0, le=300)
    sodium: Optional[float] = Field(default=None, ge=90, le=200)
    potassium: Optional[float] = Field(default=None, ge=1, le=12)
    chloride: Optional[float] = Field(default=None, ge=50, le=200)
    bicarbonate: Optional[float] = Field(default=None, ge=0, le=60)
    glucose: Optional[float] = Field(default=None, ge=0, le=1500)

    # --- labs: heme/coag ---
    hemoglobin: Optional[float] = Field(default=None, ge=0, le=25)
    hematocrit: Optional[float] = Field(default=None, ge=0, le=70)
    wbc: Optional[float] = Field(default=None, ge=0, le=200)
    platelets: Optional[float] = Field(default=None, ge=0, le=1500)
    inr: Optional[float] = Field(default=None, ge=0, le=20)
    ptt: Optional[float] = Field(default=None, ge=0, le=200)

    # --- labs: liver ---
    bilirubin: Optional[float] = Field(default=None, ge=0, le=60)
    alt: Optional[float] = Field(default=None, ge=0, le=5000)
    ast: Optional[float] = Field(default=None, ge=0, le=5000)
    lactate: Optional[float] = Field(default=None, ge=0, le=40)

    # --- vasopressors: any flag + max rate ---
    norepinephrine_any: Optional[bool] = None
    norepinephrine_max_rate: Optional[float] = Field(default=None, ge=0, le=5)
    epinephrine_any: Optional[bool] = None
    epinephrine_max_rate: Optional[float] = Field(default=None, ge=0, le=5)
    dopamine_any: Optional[bool] = None
    dopamine_max_rate: Optional[float] = Field(default=None, ge=0, le=50)
    dobutamine_any: Optional[bool] = None
    dobutamine_max_rate: Optional[float] = Field(default=None, ge=0, le=50)
    vasopressin_any: Optional[bool] = None
    vasopressin_max_rate: Optional[float] = Field(default=None, ge=0, le=5)
    phenylephrine_any: Optional[bool] = None
    phenylephrine_max_rate: Optional[float] = Field(default=None, ge=0, le=5)


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