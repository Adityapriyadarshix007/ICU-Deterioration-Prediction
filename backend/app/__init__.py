"""
ICU Deterioration Prediction API - Backend Package
"""

import sys
import os
import logging

__version__ = "1.0.0"
__author__ = "Aditya Priyadarshi"

# Setup logging
logger = logging.getLogger(__name__)

# Try to import core modules with fallbacks
try:
    from .database import Database, get_database
except ImportError as e:
    logger.warning(f"Database import failed: {e}")
    Database = None
    get_database = None

try:
    from .models import User, Prediction
except ImportError as e:
    logger.warning(f"Models import failed: {e}")
    User = None
    Prediction = None

try:
    from .ml_model import ml_model, MLModel
except ImportError as e:
    logger.warning(f"ML Model import failed: {e}")
    ml_model = None
    MLModel = None

try:
    from .auth import (
        get_password_hash,
        verify_password,
        create_access_token,
        authenticate_user,
        get_current_user,
        get_db
    )
except ImportError as e:
    logger.warning(f"Auth import failed: {e}")
    get_password_hash = None
    verify_password = None
    create_access_token = None
    authenticate_user = None
    get_current_user = None
    get_db = None

# Import external validation modules
try:
    from .data_loaders import EICUDataLoader
except ImportError as e:
    logger.warning(f"EICUDataLoader import failed: {e}")
    EICUDataLoader = None

try:
    from .feature_engineering import EICUFeatureEngineer
except ImportError as e:
    logger.warning(f"EICUFeatureEngineer import failed: {e}")
    EICUFeatureEngineer = None

# Import ExternalValidator directly from the module file
try:
    # Import from the module file, not from the package
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "external_validation_module",
        os.path.join(os.path.dirname(__file__), "external_validation.py")
    )
    if spec and spec.loader:
        external_validation_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(external_validation_module)
        ExternalValidator = external_validation_module.ExternalValidator
    else:
        ExternalValidator = None
except Exception as e:
    logger.warning(f"ExternalValidator import failed: {e}")
    ExternalValidator = None

# Package metadata
PACKAGE_INFO = {
    "name": "ICU Deterioration Prediction API",
    "version": __version__,
    "author": __author__,
    "description": "ML-powered early warning system for ICU patients",
}

def get_package_info():
    """Get package information."""
    return PACKAGE_INFO

def get_external_validation_status():
    """Check if external validation modules are available."""
    return {
        "EICUDataLoader": EICUDataLoader is not None,
        "EICUFeatureEngineer": EICUFeatureEngineer is not None,
        "ExternalValidator": ExternalValidator is not None,
        "status": "ready" if all([EICUDataLoader, EICUFeatureEngineer, ExternalValidator]) else "partial"
    }

__all__ = [
    "Database",
    "get_database",
    "User",
    "Prediction",
    "ml_model",
    "MLModel",
    "get_password_hash",
    "verify_password",
    "create_access_token",
    "authenticate_user",
    "get_current_user",
    "get_db",
    "EICUDataLoader",
    "EICUFeatureEngineer",
    "ExternalValidator",
    "get_package_info",
    "get_external_validation_status",
    "__version__",
    "__author__"
]
