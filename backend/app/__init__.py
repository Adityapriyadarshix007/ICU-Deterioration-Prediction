"""
ICU Deterioration Prediction API
================================
FastAPI backend for early prediction of ICU patient deterioration 
using MIMIC-IV dataset and CNN-LSTM + Attention model.

This package provides:
- JWT authentication for secure API access
- ML model inference for risk prediction
- Patient data management
- Dashboard statistics
- Real-time risk scoring

Modules:
    - auth: JWT authentication and user management
    - database: MongoDB connection and management
    - models: Pydantic models for MongoDB
    - schemas: Pydantic schemas for request/response validation
    - ml_model: ML model loading and prediction logic
    - routes: API route handlers (auth, predictions, dashboard)
    - utils: Utility functions (preprocessing, validation)

Version: 1.0.0
Author: Aditya Priyadarshi
"""

__version__ = "1.0.0"
__author__ = "Aditya Priyadarshi"

# Import key components for easier access
from app.database import Database, get_database
from app.models import User, Prediction
from app.schemas import (
    # Auth schemas
    UserCreate,
    UserLogin,
    UserResponse,
    Token,
    TokenData,
    # Prediction schemas
    PatientData,
    PredictionResponse,
    PredictionHistoryResponse,
    # Dashboard schemas
    DashboardStats
)
from app.ml_model import ml_model, MLModel
from app.auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    authenticate_user,
    get_current_user,
    get_db
)

# Import utils
from app.utils import (
    preprocess_features,
    normalize_data,
    impute_missing,
    validate_patient_data,
    validate_prediction_input,
    CLINICAL_RANGES,
    FEATURE_NAMES
)

# List of exports for cleaner imports
__all__ = [
    # Database
    "Database",
    "get_database",
    
    # Models
    "User",
    "Prediction",
    
    # Schemas
    "UserCreate",
    "UserLogin",
    "UserResponse",
    "Token",
    "TokenData",
    "PatientData",
    "PredictionResponse",
    "PredictionHistoryResponse",
    "DashboardStats",
    
    # ML Model
    "ml_model",
    "MLModel",
    
    # Auth
    "get_password_hash",
    "verify_password",
    "create_access_token",
    "authenticate_user",
    "get_current_user",
    "get_db",
    
    # Utils
    "preprocess_features",
    "normalize_data",
    "impute_missing",
    "validate_patient_data",
    "validate_prediction_input",
    "CLINICAL_RANGES",
    "FEATURE_NAMES",
    
    # Metadata
    "__version__",
    "__author__"
]

# Package metadata
PACKAGE_INFO = {
    "name": "ICU Deterioration Prediction API",
    "version": __version__,
    "author": __author__,
    "description": "ML-powered early warning system for ICU patients",
    "model_used": "CNN-LSTM + Attention",
    "dataset": "MIMIC-IV v3.1",
    "database": "MongoDB Atlas"
}

def get_package_info():
    """Get package information."""
    return PACKAGE_INFO
