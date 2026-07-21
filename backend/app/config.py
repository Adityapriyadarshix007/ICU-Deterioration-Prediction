"""
Configuration settings for the ICU Deterioration Prediction API.
"""

import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class Settings:
    # Application
    APP_NAME: str = "ICU Deterioration Prediction API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # MongoDB
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "icu_predictor")
    
    # JWT
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Google OAuth - Use exact Client ID from Google Cloud Console
    GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "470384815602-voh3j0i6bsdnupjb4s2hvlhmi3172o0g.apps.googleusercontent.com")
    GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
    GOOGLE_REDIRECT_URI: str = os.getenv("GOOGLE_REDIRECT_URI", "https://icu-deterioration-prediction.vercel.app/oauth-callback")
    
    # CORS - Must include ALL origins your app uses
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173,https://icu-deterioration-prediction.vercel.app,https://icu-deterioration-prediction.onrender.com").split(",")
    
    # Model
    MODEL_PATH: str = os.getenv("MODEL_PATH", "./data/models/cnn_lstm_attention_model.h5")
    
    # Email Configuration
    SMTP_HOST: str = os.getenv("SMTP_HOST", "")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME: str = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    SMTP_FROM_EMAIL: str = os.getenv("SMTP_FROM_EMAIL", "noreply@icupredictor.com")
    SENDGRID_API_KEY: str = os.getenv("SENDGRID_API_KEY", "")
    
    # Frontend URL
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "https://icu-deterioration-prediction.vercel.app")

settings = Settings()

# Log the configuration
logger.info(f"Google Client ID: {settings.GOOGLE_CLIENT_ID}")
logger.info(f"Allowed Origins: {settings.ALLOWED_ORIGINS}")
logger.info(f"Frontend URL: {settings.FRONTEND_URL}")
