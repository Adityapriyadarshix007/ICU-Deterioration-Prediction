"""
Admin - User Management Routes
"""

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime
import logging
from bson import ObjectId

from app import auth
from app.database import Database

router = APIRouter()
logger = logging.getLogger(__name__)

def get_db():
    return Database.get_db()

async def check_admin(current_user: dict):
    """Check if current user is admin"""
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


@router.get("/users")
async def get_users(
    skip: int = 0,
    limit: int = 100,
    current_user: dict = Depends(auth.get_current_user)
):
    """Get all users (admin only)"""
    await check_admin(current_user)
    db = get_db()
    
    total = db.users.count_documents({})
    users = list(db.users.find().skip(skip).limit(limit))
    
    result = []
    for user in users:
        user["id"] = str(user["_id"])
        user.pop("_id", None)
        user.pop("hashed_password", None)
        result.append(user)
    
    return {
        "total": total,
        "users": result
    }


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    current_user: dict = Depends(auth.get_current_user)
):
    """Delete user (admin only)"""
    await check_admin(current_user)
    db = get_db()
    
    if user_id == current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    try:
        result = db.users.delete_one({"_id": ObjectId(user_id)})
    except:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    db.predictions.delete_many({"user_id": user_id})
    
    return {"message": "User deleted successfully"}


@router.post("/make-admin/{user_id}")
async def make_admin(
    user_id: str,
    current_user: dict = Depends(auth.get_current_user)
):
    """Make a user admin (admin only)"""
    await check_admin(current_user)
    db = get_db()
    
    try:
        result = db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"is_admin": True, "role": "admin"}}
        )
    except:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if result.matched_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {"message": "User is now an admin"}
