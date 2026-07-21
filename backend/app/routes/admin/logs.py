"""
Admin Logs Routes
"""

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime, timedelta
import logging

from app import auth
from app.database import Database

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

@router.get("/logs")
async def get_activity_logs(
    limit: int = 100,
    skip: int = 0,
    current_user: dict = Depends(auth.get_current_user)
):
    """Get system activity logs"""
    await check_admin(current_user)
    db = get_db()
    
    try:
        # Try to get logs from database
        logs = []
        total = 0
        
        if "logs" in db.list_collection_names():
            cursor = db.logs.find().sort("created_at", -1).skip(skip).limit(limit)
            for log in cursor:
                log["id"] = str(log["_id"])
                log.pop("_id", None)
                logs.append(log)
            total = db.logs.count_documents({})
        else:
            # Return mock data if no logs collection
            logs = [
                {
                    "id": "1",
                    "user": "admin",
                    "action": "Login",
                    "patient_name": "N/A",
                    "alert_level": "INFO",
                    "created_at": datetime.utcnow().isoformat()
                },
                {
                    "id": "2",
                    "user": "doctor",
                    "action": "Prediction",
                    "patient_name": "John Doe",
                    "alert_level": "CRITICAL",
                    "created_at": (datetime.utcnow() - timedelta(minutes=15)).isoformat()
                },
                {
                    "id": "3",
                    "user": "doctor",
                    "action": "Prediction",
                    "patient_name": "Jane Smith",
                    "alert_level": "HIGH",
                    "created_at": (datetime.utcnow() - timedelta(minutes=30)).isoformat()
                },
                {
                    "id": "4",
                    "user": "admin",
                    "action": "User Management",
                    "patient_name": "N/A",
                    "alert_level": "INFO",
                    "created_at": (datetime.utcnow() - timedelta(hours=1)).isoformat()
                },
                {
                    "id": "5",
                    "user": "doctor",
                    "action": "Patient Update",
                    "patient_name": "Robert Johnson",
                    "alert_level": "MEDIUM",
                    "created_at": (datetime.utcnow() - timedelta(minutes=90)).isoformat()
                }
            ]
            total = len(logs)
        
        return {"logs": logs, "total": total}
    except Exception as e:
        logger.error(f"Failed to get logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get logs: {str(e)}"
        )

@router.post("/logs")
async def create_log(
    log_data: dict,
    current_user: dict = Depends(auth.get_current_user)
):
    """Create a new log entry"""
    await check_admin(current_user)
    db = get_db()
    
    try:
        log_data["created_at"] = datetime.utcnow()
        log_data["user"] = current_user.get("username")
        
        result = db.logs.insert_one(log_data)
        log_data["id"] = str(result.inserted_id)
        
        return log_data
    except Exception as e:
        logger.error(f"Failed to create log: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create log: {str(e)}"
        )
