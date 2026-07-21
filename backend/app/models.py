"""
Database models for MongoDB collections.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field

class User(BaseModel):
    username: str
    email: EmailStr
    full_name: str
    hashed_password: Optional[str] = None
    google_id: Optional[str] = None
    profile_picture: Optional[str] = None
    is_active: bool = True
    is_admin: bool = False
    role: str = "clinician"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    # Optional: Keep these for future use, but they won't be used for direct password change
    reset_password_token: Optional[str] = None
    reset_password_expires: Optional[datetime] = None

class Prediction(BaseModel):
    patient_id: str
    user_id: str
    patient_name: Optional[str] = None
    risk_score: float
    risk_percentage: float
    alert_level: str
    confidence: str
    features: dict
    is_high_risk: bool
    created_at: datetime = Field(default_factory=datetime.utcnow)