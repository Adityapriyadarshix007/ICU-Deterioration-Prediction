"""
ICU Deterioration Prediction API - Main Application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import logging
import os

from app.config import settings
from app.database import Database
from app.routes import auth_router, google_router, predictions_router, dashboard_router
from app.routes.patients import router as patients_router
from app.routes.admin import users_router, analytics_router, settings_router, patients_router as admin_patients_router, logs_router
from app.ml_model import ml_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="ML-powered early warning system for ICU patients",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    logger.info("🚀 Starting up...")
    try:
        Database.connect()
        logger.info("✅ MongoDB connected")
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
    logger.info("✅ ML Model initialized")

@app.on_event("shutdown")
def shutdown_event():
    logger.info("🔄 Shutting down...")
    Database.close()

# Include routers
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(google_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(predictions_router, prefix="/api/predict", tags=["Predictions"])
app.include_router(dashboard_router, prefix="/api/dashboard", tags=["Dashboard"])

# Patient routes (public - for all authenticated users)
app.include_router(patients_router, prefix="/api", tags=["Patients"])

# Admin routers
app.include_router(users_router, prefix="/api/admin", tags=["Admin - Users"])
app.include_router(analytics_router, prefix="/api/admin", tags=["Admin - Analytics"])
app.include_router(settings_router, prefix="/api/admin", tags=["Admin - Settings"])
app.include_router(admin_patients_router, prefix="/api/admin", tags=["Admin - Patients"])
app.include_router(logs_router, prefix="/api/admin", tags=["Admin - Logs"])

@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "/docs": "API Documentation",
            "/api/auth": "Authentication",
            "/api/predict": "Predictions",
            "/api/dashboard": "Dashboard",
            "/api/patients": "Patients (Public)",
            "/api/admin": "Admin Panel"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint - checks database connection"""
    mongodb_status = "disconnected"
    try:
        # Check if Database is initialized and connected
        if hasattr(Database, 'db') and Database.db is not None:
            # Try to ping the database
            Database.db.command('ping')
            mongodb_status = "connected"
        else:
            # Try to reconnect
            Database.connect()
            mongodb_status = "connected"
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        mongodb_status = "error"
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "mongodb": mongodb_status,
        "version": settings.APP_VERSION
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)