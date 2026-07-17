"""
Dashboard Routes
"""

from fastapi import APIRouter, Depends
from datetime import datetime, timedelta
import logging

from app import auth
from app.database import Database

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/stats")
async def get_dashboard_stats(
    current_user: dict = Depends(auth.get_current_user)
):
    """Get dashboard statistics"""
    db = Database.get_db()
    
    try:
        total_predictions = db.predictions.count_documents({})
        
        twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
        high_risk = db.predictions.count_documents({
            "is_high_risk": True,
            "created_at": {"$gte": twenty_four_hours_ago}
        })
        
        critical_alerts = db.predictions.count_documents({
            "alert_level": "CRITICAL",
            "created_at": {"$gte": twenty_four_hours_ago}
        })
        
        pipeline = [{"$group": {"_id": None, "avg_risk": {"$avg": "$risk_score"}}}]
        avg_result = list(db.predictions.aggregate(pipeline))
        avg_risk = avg_result[0]["avg_risk"] if avg_result else 0
        
        return {
            "total_predictions": total_predictions,
            "high_risk_patients": high_risk,
            "avg_risk_score": round(float(avg_risk), 4),
            "critical_alerts": critical_alerts,
            "last_updated": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Dashboard stats error: {e}")
        return {
            "total_predictions": 0,
            "high_risk_patients": 0,
            "avg_risk_score": 0.0,
            "critical_alerts": 0,
            "last_updated": datetime.utcnow().isoformat()
        }
