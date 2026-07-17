"""
Admin - System Settings Routes
"""

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime
import logging

from app import auth
from app.database import Database
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

def get_db():
    return Database.get_db()

async def check_admin(current_user: dict):
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


@router.get("/settings")
async def get_system_settings(
    current_user: dict = Depends(auth.get_current_user)
):
    """Get system settings (admin only)"""
    await check_admin(current_user)
    db = get_db()
    
    settings_doc = db.settings.find_one({"_id": "system_settings"})
    
    if not settings_doc:
        settings_doc = {
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "maintenance_mode": False,
            "allow_registration": True,
            "max_predictions_per_day": 100,
            "created_at": datetime.utcnow()
        }
        db.settings.insert_one(settings_doc)
    
    settings_doc["_id"] = str(settings_doc["_id"])
    return settings_doc


@router.put("/settings")
async def update_system_settings(
    update_data: dict,
    current_user: dict = Depends(auth.get_current_user)
):
    """Update system settings (admin only)"""
    await check_admin(current_user)
    db = get_db()
    
    allowed_fields = [
        "app_name", "version", "maintenance_mode",
        "allow_registration", "max_predictions_per_day"
    ]
    
    update_dict = {k: v for k, v in update_data.items() if k in allowed_fields}
    
    if not update_dict:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid settings to update"
        )
    
    db.settings.update_one(
        {"_id": "system_settings"},
        {"$set": {**update_dict, "updated_at": datetime.utcnow()}},
        upsert=True
    )
    
    return {"message": "Settings updated successfully", "settings": update_dict}
