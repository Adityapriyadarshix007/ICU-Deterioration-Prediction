"""
Google OAuth 2.0 authentication handler.
"""

import httpx
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class GoogleOAuth:
    """Google OAuth 2.0 client"""
    
    def __init__(self):
        self.client_id = settings.GOOGLE_CLIENT_ID
        self.client_secret = settings.GOOGLE_CLIENT_SECRET
        self.redirect_uri = settings.GOOGLE_REDIRECT_URI
    
    async def verify_id_token(self, id_token: str) -> Dict[str, Any]:
        """Verify Google ID token using GET request"""
        try:
            async with httpx.AsyncClient() as client:
                # Use GET instead of POST for tokeninfo
                response = await client.get(
                    "https://oauth2.googleapis.com/tokeninfo",
                    params={"id_token": id_token}
                )
                response.raise_for_status()
                data = response.json()
                
                # Verify audience matches our client ID
                if data.get("aud") != self.client_id:
                    logger.error(f"Invalid audience: {data.get('aud')}")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid token audience"
                    )
                
                # Verify issuer
                if data.get("iss") not in ["accounts.google.com", "https://accounts.google.com"]:
                    logger.error(f"Invalid issuer: {data.get('iss')}")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid token issuer"
                    )
                
                return data
        except httpx.HTTPStatusError as e:
            logger.error(f"Token verification HTTP error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Google token"
            )
        except Exception as e:
            logger.error(f"ID token verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid ID token: {str(e)}"
            )

# Singleton instance
google_oauth = GoogleOAuth()
