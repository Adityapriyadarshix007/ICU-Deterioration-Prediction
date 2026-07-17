"""
Admin - Analytics Routes
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


@router.get("/overview")
async def get_system_overview(
    current_user: dict = Depends(auth.get_current_user)
):
    """Get system overview statistics (admin only)"""
    await check_admin(current_user)
    db = get_db()
    
    # User statistics
    total_users = db.users.count_documents({})
    active_users = db.users.count_documents({"is_active": True})
    admin_users = db.users.count_documents({"is_admin": True})
    
    # Prediction statistics
    total_predictions = db.predictions.count_documents({})
    high_risk = db.predictions.count_documents({"is_high_risk": True})
    critical_alerts = db.predictions.count_documents({"alert_level": "CRITICAL"})
    
    # Last 7 days predictions
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    weekly_predictions = db.predictions.count_documents({
        "created_at": {"$gte": seven_days_ago}
    })
    
    # Average risk score
    pipeline = [{"$group": {"_id": None, "avg_risk": {"$avg": "$risk_score"}}}]
    avg_result = list(db.predictions.aggregate(pipeline))
    avg_risk = avg_result[0]["avg_risk"] if avg_result else 0
    
    # Daily prediction trends (last 7 days)
    trends = []
    for i in range(7):
        day = datetime.utcnow() - timedelta(days=i)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        count = db.predictions.count_documents({
            "created_at": {"$gte": day_start, "$lt": day_end}
        })
        trends.append({
            "date": day_start.isoformat(),
            "count": count
        })
    
    return {
        "users": {
            "total": total_users,
            "active": active_users,
            "admins": admin_users
        },
        "predictions": {
            "total": total_predictions,
            "high_risk": high_risk,
            "critical": critical_alerts,
            "weekly": weekly_predictions,
            "avg_risk": round(float(avg_risk), 4) if avg_risk else 0
        },
        "trends": trends,
        "last_updated": datetime.utcnow().isoformat()
    }


@router.get("/logs")
async def get_activity_logs(
    limit: int = 50,
    current_user: dict = Depends(auth.get_current_user)
):
    """Get user activity logs (admin only)"""
    await check_admin(current_user)
    db = get_db()
    
    # Get recent predictions with user info
    cursor = db.predictions.find().sort("created_at", -1).limit(limit)
    
    result = []
    for log in cursor:
        # Get user info
        user = db.users.find_one({"_id": ObjectId(log.get("user_id")) if log.get("user_id") else None})
        
        result.append({
            "id": str(log["_id"]),
            "user": user.get("username", "Unknown") if user else "Unknown",
            "user_id": log.get("user_id"),
            "patient_id": log.get("patient_id"),
            "patient_name": log.get("patient_name"),
            "risk_score": log.get("risk_score"),
            "alert_level": log.get("alert_level"),
            "created_at": log.get("created_at").isoformat() if log.get("created_at") else None
        })
    
    return {"logs": result}
