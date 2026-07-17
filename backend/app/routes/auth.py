"""
Authentication Routes - Login, Register, Logout
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import datetime, timedelta
import logging

from app import auth, schemas
from app.database import Database
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

def get_db():
    return Database.get_db()


@router.post("/register", response_model=schemas.UserResponse)
async def register(user_data: schemas.UserCreate):
    """Register a new user"""
    db = get_db()
    
    logger.info(f"📝 Registering user: {user_data.username}")
    
    if db.users.find_one({"username": user_data.username}):
        raise HTTPException(status_code=400, detail="Username already taken")
    
    if db.users.find_one({"email": user_data.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = auth.get_password_hash(user_data.password)
    user_dict = {
        "username": user_data.username,
        "email": user_data.email,
        "full_name": user_data.full_name,
        "hashed_password": hashed_password,
        "role": "clinician",
        "is_admin": False,
        "is_active": True,
        "created_at": datetime.utcnow(),
        "last_login": None,
        "google_id": None,
        "profile_picture": None
    }
    
    result = db.users.insert_one(user_dict)
    user_dict["_id"] = str(result.inserted_id)
    
    logger.info(f"✅ User registered: {user_data.username}")
    
    return schemas.UserResponse(
        id=user_dict["_id"],
        username=user_dict["username"],
        email=user_dict["email"],
        full_name=user_dict["full_name"],
        role=user_dict["role"],
        created_at=user_dict["created_at"],
        last_login=user_dict["last_login"],
        profile_picture=user_dict["profile_picture"]
    )


@router.post("/login", response_model=schemas.Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login with username/password (OAuth2 form)"""
    db = get_db()
    logger.info(f"🔐 Login attempt: {form_data.username}")
    
    user = db.users.find_one({"username": form_data.username})
    if not user:
        user = db.users.find_one({"email": form_data.username})
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    if not auth.verify_password(form_data.password, user.get("hashed_password", "")):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    db.users.update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login": datetime.utcnow()}}
    )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user["username"], "user_id": str(user["_id"])},
        expires_delta=access_token_expires
    )
    
    logger.info(f"✅ User logged in: {user['username']}")
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/login/json", response_model=schemas.Token)
async def login_json(login_data: schemas.UserLogin):
    """Login with JSON body"""
    db = get_db()
    logger.info(f"🔐 JSON Login attempt: {login_data.username}")
    
    user = db.users.find_one({"username": login_data.username})
    if not user:
        user = db.users.find_one({"email": login_data.username})
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    if not auth.verify_password(login_data.password, user.get("hashed_password", "")):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    db.users.update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login": datetime.utcnow()}}
    )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user["username"], "user_id": str(user["_id"])},
        expires_delta=access_token_expires
    )
    
    logger.info(f"✅ User logged in: {user['username']}")
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=schemas.UserResponse)
async def get_current_user(current_user: dict = Depends(auth.get_current_user)):
    """Get current user info"""
    logger.info(f"📋 Getting user info for: {current_user.get('username')}")
    
    return schemas.UserResponse(
        id=current_user["id"],
        username=current_user["username"],
        email=current_user["email"],
        full_name=current_user["full_name"],
        role=current_user["role"],
        created_at=current_user["created_at"],
        last_login=current_user.get("last_login"),
        profile_picture=current_user.get("profile_picture")
    )


@router.post("/logout")
async def logout(current_user: dict = Depends(auth.get_current_user)):
    """Logout user"""
    return {"message": "Logged out successfully"}
