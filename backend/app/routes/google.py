"""
Google OAuth Routes
"""

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime, timedelta
import random
import logging
import httpx

from app import auth, schemas
from app.database import Database
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

def get_db():
    return Database.get_db()


@router.post("/google/login", response_model=schemas.Token)
async def google_login(auth_data: schemas.GoogleAuthRequest):
    """Login with Google ID token"""
    db = get_db()
    
    logger.info("🔐 Google login attempt")
    
    if not auth_data.id_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ID token is required"
        )
    
    try:
        # IMPORTANT: Use GET, not POST for Google's tokeninfo endpoint
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://oauth2.googleapis.com/tokeninfo",
                params={"id_token": auth_data.id_token}
            )
            
            if response.status_code != 200:
                logger.error(f"Token verification failed: {response.text}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid Google token"
                )
            
            token_info = response.json()
        
        # Verify audience matches our client ID
        if token_info.get("aud") != settings.GOOGLE_CLIENT_ID:
            logger.error(f"Invalid audience: {token_info.get('aud')}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token audience"
            )
        
        google_id = token_info.get("sub")
        email = token_info.get("email")
        name = token_info.get("name", "Google User")
        picture = token_info.get("picture")
        
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email not provided by Google"
            )
        
        logger.info(f"📧 Google user email: {email}")
        
        # Check if user exists
        existing_user = db.users.find_one({
            "$or": [
                {"google_id": google_id},
                {"email": email}
            ]
        })
        
        if existing_user:
            db.users.update_one(
                {"_id": existing_user["_id"]},
                {"$set": {"last_login": datetime.utcnow()}}
            )
            user_id = str(existing_user["_id"])
            username = existing_user["username"]
            logger.info(f"✅ Existing user logged in via Google: {username}")
        else:
            # Create new user
            username = email.split("@")[0]
            existing = db.users.find_one({"username": username})
            if existing:
                username = f"{username}_{random.randint(1000, 9999)}"
            
            new_user = {
                "username": username,
                "email": email,
                "full_name": name,
                "google_id": google_id,
                "hashed_password": None,
                "profile_picture": picture,
                "role": "clinician",
                "is_admin": False,
                "is_active": True,
                "created_at": datetime.utcnow(),
                "last_login": datetime.utcnow()
            }
            
            result = db.users.insert_one(new_user)
            user_id = str(result.inserted_id)
            username = new_user["username"]
            logger.info(f"✅ New user registered via Google: {username}")
        
        # Create JWT token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = auth.create_access_token(
            data={"sub": username, "user_id": user_id},
            expires_delta=access_token_expires
        )
        
        logger.info("✅ Google login successful")
        return {"access_token": access_token, "token_type": "bearer"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Google login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Google authentication failed: {str(e)}"
        )
