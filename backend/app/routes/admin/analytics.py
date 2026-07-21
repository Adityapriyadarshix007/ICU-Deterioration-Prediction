"""
Admin Analytics Routes
"""

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime, timedelta
import logging
from bson import ObjectId

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
        # Risk distribution
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
        # CRITICAL FIX: Get recent predictions for the frontend table
        # ============================================================
        recent_predictions = []
        cursor = db.predictions.find().sort("created_at", -1).limit(10)
        
        for prediction in cursor:
            recent_predictions.append({
                "patient_name": prediction.get("patient_name", "Unknown"),
                "patient_id": prediction.get("patient_id", ""),
                "risk_score": prediction.get("risk_score", 0),
                "alert_level": prediction.get("alert_level", "LOW"),
                "username": prediction.get("username", "System"),
                "created_at": prediction.get("created_at", datetime.utcnow())
            })
        
        logger.info(f"✅ Found {len(recent_predictions)} recent predictions")
        
        # ============================================================
        # Return format that matches frontend expectations
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
            "recent_predictions": recent_predictions,  # CRITICAL: Added this
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


# ============================================================
# REMOVED: Duplicate /logs endpoint
# The logs endpoint is now only in logs.py
# ============================================================
# The get_activity_logs function has been removed from this file
# to avoid duplicate route conflicts with logs.py