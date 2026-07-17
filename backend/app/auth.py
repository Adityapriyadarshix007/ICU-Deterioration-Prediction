"""
Authentication utilities
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from bson import ObjectId
import logging
import bcrypt

from app.config import settings
from app.database import Database

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not hashed_password:
        return False
    try:
        if len(plain_password) > 72:
            plain_password = plain_password[:72]
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        try:
            return bcrypt.checkpw(
                plain_password.encode('utf-8')[:72],
                hashed_password.encode('utf-8')
            )
        except:
            return False


def get_password_hash(password: str) -> str:
    if len(password) > 72:
        password = password[:72]
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=30)  # 30 minutes
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def get_db():
    return Database.get_db()


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        
        logger.info(f"🔍 Decoded token - username: {username}, user_id: {user_id}")
        
        if username is None or user_id is None:
            logger.warning("❌ Token missing username or user_id")
            raise credentials_exception
    except JWTError as e:
        logger.error(f"❌ JWT decode error: {e}")
        raise credentials_exception
    
    db = get_db()
    try:
        user = db.users.find_one({"_id": ObjectId(user_id)})
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        user = None
    
    if user is None:
        logger.warning(f"❌ User not found for id: {user_id}")
        raise credentials_exception
    
    # Convert ObjectId to string
    user["id"] = str(user["_id"])
    user.pop("_id", None)
    
    # Ensure is_admin is included
    if "is_admin" not in user:
        user["is_admin"] = False
    
    logger.info(f"✅ User authenticated: {username}")
    return user


def authenticate_user(username: str, password: str):
    db = get_db()
    user = db.users.find_one({"username": username})
    if not user:
        return False
    if not verify_password(password, user.get("hashed_password", "")):
        return False
    return user
