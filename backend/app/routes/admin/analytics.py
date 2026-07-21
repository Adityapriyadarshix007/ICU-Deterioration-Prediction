"""
Admin Analytics Routes
"""

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime, timedelta
import logging
from bson import ObjectId  # CRITICAL: Add this import

from app import auth
from app.database import Database
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

def get_db():
    return Database.get_db()


@router.get("/overview")
async def get_overview(current_user: dict = Depends(auth.get_current_user)):
    """Get system overview analytics"""
    # Check if user is admin
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    db = get_db()
    
    try:
        # Get total users
        total_users = db.users.count_documents({})
        active_users = db.users.count_documents({"is_active": True})
        admin_users = db.users.count_documents({"is_admin": True})
        
        # Get prediction stats
        total_predictions = db.predictions.count_documents({})
        high_risk = db.predictions.count_documents({"alert_level": "HIGH"})
        critical = db.predictions.count_documents({"alert_level": "CRITICAL"})
        
        # Get weekly predictions (last 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        weekly = db.predictions.count_documents({"created_at": {"$gte": week_ago}})
        
        # Get average risk score
        pipeline = [
            {"$group": {"_id": None, "avg_risk": {"$avg": "$risk_score"}}}
        ]
        avg_result = list(db.predictions.aggregate(pipeline))
        avg_risk = avg_result[0]["avg_risk"] if avg_result else 0.0
        
        # ============================================================
        # CRITICAL FIX: Add risk_distribution
        # ============================================================
        risk_distribution = {
            "LOW": db.predictions.count_documents({"alert_level": "LOW"}),
            "MEDIUM": db.predictions.count_documents({"alert_level": "MEDIUM"}),
            "HIGH": db.predictions.count_documents({"alert_level": "HIGH"}),
            "CRITICAL": db.predictions.count_documents({"alert_level": "CRITICAL"})
        }
        
        # Get trends (last 7 days)
        trends = []
        for i in range(7):
            day = datetime.utcnow() - timedelta(days=i)
            day_start = datetime(day.year, day.month, day.day)
            day_end = day_start + timedelta(days=1)
            count = db.predictions.count_documents({
                "created_at": {"$gte": day_start, "$lt": day_end}
            })
            trends.append({
                "date": day_start.strftime("%Y-%m-%d"),
                "count": count
            })
        trends.reverse()
        
        # ============================================================
        # CRITICAL FIX: Return format that matches frontend expectations
        # ============================================================
        return {
            "users": {
                "total": total_users,
                "active": active_users,
                "admins": admin_users
            },
            "predictions": {
                "total": total_predictions,
                "high_risk": high_risk,
                "critical": critical,
                "weekly": weekly,
                "avg_risk": round(avg_risk, 4)
            },
            "risk_distribution": risk_distribution,
            "trends": trends,
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Analytics error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics: {str(e)}"
        )


@router.get("/logs")
async def get_activity_logs(
    limit: int = 100,
    skip: int = 0,
    current_user: dict = Depends(auth.get_current_user)
):
    """Get activity logs"""
    # Check if user is admin
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    db = get_db()
    
    try:
        # Check if logs collection exists
        collections = db.list_collection_names()
        
        if "logs" not in collections:
            # If logs collection doesn't exist, return empty array
            logger.warning("⚠️ Logs collection doesn't exist yet")
            return {"logs": [], "total": 0}
        
        # Get logs with user info
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
            formatted_log = {
                "id": str(log.get("_id")),
                "user": log.get("username") or log.get("user", {}).get("username", "System"),
                "action": log.get("action", "Unknown"),
                "patient_id": log.get("patient_id", "N/A"),
                "patient_name": log.get("patient_name", "N/A"),
                "alert_level": log.get("alert_level", "INFO"),
                "risk_score": log.get("risk_score"),
                "created_at": log.get("created_at", datetime.utcnow()).isoformat()
            }
            formatted_logs.append(formatted_log)
        
        return {
            "logs": formatted_logs,
            "total": total
        }
        
    except Exception as e:
        logger.error(f"❌ Logs error: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty logs instead of throwing error
        return {"logs": [], "total": 0}