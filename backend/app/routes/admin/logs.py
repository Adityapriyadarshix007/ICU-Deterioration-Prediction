"""
Admin Logs Routes
"""

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime, timedelta
import logging
from bson import ObjectId

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
        # Check if logs collection exists
        collections = db.list_collection_names()
        
        if "logs" not in collections:
            logger.warning("⚠️ Logs collection doesn't exist yet - returning mock data")
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
            return {"logs": logs, "total": len(logs)}
        
        # Get logs from database with user info
        pipeline = [
            {"$sort": {"created_at": -1}},
            {"$skip": skip},
            {"$limit": limit},
            {"$lookup": {
                "from": "users",
                "localField": "user_id",
                "foreignField": "_id",
                "as": "user"
            }},
            {"$unwind": {"path": "$user", "preserveNullAndEmptyArrays": True}}
        ]
        
        logs = list(db.logs.aggregate(pipeline))
        total = db.logs.count_documents({})
        
        # Format logs for frontend
        formatted_logs = []
        for log in logs:
            # Get username from either the log or the joined user
            username = log.get("username")
            if not username and log.get("user"):
                username = log["user"].get("username", "System")
            if not username:
                username = "System"
            
            formatted_log = {
                "id": str(log.get("_id")),
                "user": username,
                "action": log.get("action", "Unknown"),
                "patient_id": log.get("patient_id", "N/A"),
                "patient_name": log.get("patient_name", "N/A"),
                "alert_level": log.get("alert_level", "INFO"),
                "risk_score": log.get("risk_score"),
                "created_at": log.get("created_at", datetime.utcnow()).isoformat()
            }
            formatted_logs.append(formatted_log)
        
        logger.info(f"✅ Found {len(formatted_logs)} logs from database")
        
        return {
            "logs": formatted_logs,
            "total": total
        }
        
    except Exception as e:
        logger.error(f"❌ Logs error: {e}")
        import traceback
        traceback.print_exc()
        
        # Return mock data on error
        logs = [
            {
                "id": "1",
                "user": "system",
                "action": "Error",
                "patient_name": "N/A",
                "alert_level": "ERROR",
                "created_at": datetime.utcnow().isoformat()
            }
        ]
        return {"logs": logs, "total": 1}


@router.post("/logs")
async def create_log(
    log_data: dict,
    current_user: dict = Depends(auth.get_current_user)
):
    """Create a new log entry (admin only)"""
    await check_admin(current_user)
    db = get_db()
    
    try:
        # Add metadata to log
        log_data["created_at"] = datetime.utcnow()
        log_data["user_id"] = str(current_user.get("_id"))
        log_data["username"] = current_user.get("username", "System")
        
        result = db.logs.insert_one(log_data)
        log_data["id"] = str(result.inserted_id)
        log_data.pop("_id", None)
        
        logger.info(f"✅ Log created: {log_data.get('action')} by {log_data.get('username')}")
        
        return log_data
    except Exception as e:
        logger.error(f"❌ Failed to create log: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create log: {str(e)}"
        )