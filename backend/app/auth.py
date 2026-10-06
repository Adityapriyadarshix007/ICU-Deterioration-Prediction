"""
Authentication utilities
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from bson import ObjectId
from bson.errors import InvalidId
import logging
import bcrypt

from app.config_api import settings
from app.database import Database

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash using bcrypt."""
    if not hashed_password:
        return False
    try:
        if len(plain_password) > 72:
            plain_password = plain_password[:72]
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt."""
    try:
        if len(password) > 72:
            password = password[:72]
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")
    except Exception as e:
        logger.error(f"Password hashing error: {e}")
        raise


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )
    return encoded_jwt


def get_db():
    return Database.get_db()


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # 1) Decode the JWT
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        username: str = payload.get("sub")
        user_id: str = payload.get("user_id")

        logger.info(f"🔍 Decoded token - username: {username}, user_id: {user_id}")

        if username is None or user_id is None:
            logger.warning("❌ Token missing username or user_id")
            raise credentials_exception
    except JWTError as e:
        # Common reasons: expired token, wrong signature, malformed
        logger.error(f"❌ JWT decode error: {type(e).__name__}: {e}")
        raise credentials_exception

    # 2) Look up user in Mongo
    db = get_db()
    try:
        user = db.users.find_one({"_id": ObjectId(user_id)})
    except InvalidId:
        logger.error(f"❌ Invalid ObjectId in token: {user_id!r}")
        raise credentials_exception
    except Exception as e:
        logger.error(f"❌ Database error during user lookup: {type(e).__name__}: {e}")
        raise credentials_exception

    if user is None:
        logger.warning(f"❌ User not found for id: {user_id}")
        raise credentials_exception

    # 3) Normalize user dict for the rest of the app
    user["id"] = str(user["_id"])
    user.pop("_id", None)
    if "is_admin" not in user:
        user["is_admin"] = user.get("role") == "admin"

    logger.info(f"✅ User authenticated: {username}")
    return user


def authenticate_user(username: str, password: str):
    """Authenticate a user with username and password."""
    db = get_db()
    user = db.users.find_one({"username": username})
    if not user:
        return False

    # Support both field names so old records still work
    stored_hash = user.get("password_hash") or user.get("hashed_password") or ""
    if not stored_hash:
        logger.warning(f"User {username} has no password hash stored")
        return False

    if not verify_password(password, stored_hash):
        return False
    return user