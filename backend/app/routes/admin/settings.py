"""
Admin Settings Routes
"""

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime
import logging

from app import auth
from app.database import Database

router = APIRouter()
logger = logging.getLogger(__name__)

def get_db():
    return Database.get_db()

@router.get("/settings")
async def get_settings(
    current_user: dict = Depends(auth.get_current_user)
):
    """Get system settings"""
    db = get_db()
    
    # Check if user is admin
    if not current_user.get('is_admin', False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        # Get settings from database or return defaults
        settings = db.settings.find_one({})
        
        if not settings:
            # Default settings
            settings = {
                "app_name": "ICU Predictor",
                "version": "1.0.0",
                "maintenance_mode": False,
                "model_threshold": 0.459,
                "prediction_window": 24,
                "alert_frequency": 60,
                "auto_prediction": True,
                "imputation_method": "knn",
                "email_notifications": True,
                "data_retention_days": 30,
                "updated_at": datetime.utcnow()
            }
            # Save default settings
            db.settings.insert_one(settings)
        
        settings["id"] = str(settings["_id"])
        settings.pop("_id", None)
        
        return settings
    except Exception as e:
        logger.error(f"Failed to get settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get settings: {str(e)}"
        )

@router.put("/settings")
async def update_settings(
    settings_data: dict,
    current_user: dict = Depends(auth.get_current_user)
):
    """Update system settings"""
    db = get_db()
    
    # Check if user is admin
    if not current_user.get('is_admin', False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        # Remove id from update data
        settings_data.pop("id", None)
        settings_data.pop("_id", None)
        
        # Add update timestamp
        settings_data["updated_at"] = datetime.utcnow()
        settings_data["updated_by"] = current_user.get("username")
        
        # Update settings
        result = db.settings.update_one(
            {},
            {"$set": settings_data},
            upsert=True
        )
        
        logger.info(f"Settings updated by {current_user.get('username')}")
        return {"message": "Settings updated successfully"}
    except Exception as e:
        logger.error(f"Failed to update settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update settings: {str(e)}"
        )
