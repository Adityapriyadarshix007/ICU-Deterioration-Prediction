"""
Authentication Routes - Login, Register, Logout, Forgot Password, Reset Password
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import datetime, timedelta
import secrets
import string
import logging

from app import auth, schemas
from app.database import Database
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

def get_db():
    return Database.get_db()

# ============================================================
# Password Reset Helper Functions
# ============================================================

def generate_reset_token() -> str:
    """Generate a secure random token for password reset"""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(32))

def send_reset_email(email: str, token: str):
    """
    Send password reset email
    For development, we'll print the link.
    For production, use SendGrid, SMTP, etc.
    """
    reset_link = f"http://localhost:5173/reset-password/{token}"
    
    # For development, just print the link
    print(f"\n{'='*60}")
    print(f"🔐 PASSWORD RESET LINK")
    print(f"📧 Email: {email}")
    print(f"🔗 Link: {reset_link}")
    print(f"⏰ Expires in: 1 hour")
    print(f"{'='*60}\n")
    
    return reset_link

# ============================================================
# Password Reset Endpoints
# ============================================================

@router.post("/forgot-password", response_model=dict)
async def forgot_password(request: schemas.ForgotPasswordRequest):
    """
    Request a password reset email.
    """
    db = get_db()
    logger.info(f"🔐 Password reset requested for: {request.email}")
    
    # Find user by email
    user = db.users.find_one({"email": request.email})
    
    # For security, always return success even if email doesn't exist
    if not user:
        logger.warning(f"⚠️ Password reset requested for non-existent email: {request.email}")
        return {"message": "If an account exists with this email, a reset link has been sent"}
    
    # Generate reset token
    token = generate_reset_token()
    
    # Save token and expiry to database
    db.users.update_one(
        {"_id": user["_id"]},
        {
            "$set": {
                "reset_password_token": token,
                "reset_password_expires": datetime.utcnow() + timedelta(hours=1)
            }
        }
    )
    
    # Send email with reset link
    try:
        send_reset_email(user["email"], token)
        logger.info(f"✅ Reset email sent to: {user['email']}")
    except Exception as e:
        logger.error(f"❌ Failed to send reset email: {e}")
        pass
    
    return {"message": "If an account exists with this email, a reset link has been sent"}

@router.post("/reset-password", response_model=dict)
async def reset_password(request: schemas.ResetPasswordRequest):
    """
    Reset password using token.
    """
    try:
        db = get_db()
        logger.info(f"🔐 Password reset attempt with token: {request.token[:10]}...")
        
        # Find user with matching token
        user = db.users.find_one({"reset_password_token": request.token})
        
        if not user:
            logger.warning(f"⚠️ Invalid reset token attempted")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        logger.info(f"✅ Found user: {user.get('username')}")
        
        # Check if token has expired
        reset_expires = user.get("reset_password_expires")
        if not reset_expires or reset_expires < datetime.utcnow():
            # Clear expired token
            db.users.update_one(
                {"_id": user["_id"]},
                {
                    "$set": {
                        "reset_password_token": None,
                        "reset_password_expires": None
                    }
                }
            )
            logger.warning("⚠️ Expired reset token attempted")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Reset token has expired"
            )
        
        # Update password
        hashed_password = auth.get_password_hash(request.new_password)
        result = db.users.update_one(
            {"_id": user["_id"]},
            {
                "$set": {
                    "hashed_password": hashed_password,
                    "reset_password_token": None,
                    "reset_password_expires": None
                }
            }
        )
        
        if result.modified_count == 0:
            logger.error(f"❌ Failed to update password for user: {user['username']}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update password"
            )
        
        logger.info(f"✅ Password reset successful for user: {user['username']}")
        return {"message": "Password reset successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Reset password error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reset password failed: {str(e)}"
        )

# ============================================================
# NEW: Reset Password by Email Only (Simple Version)
# ============================================================

@router.post("/reset-password-email", response_model=dict)
async def reset_password_by_email(
    request: schemas.ResetPasswordEmailRequest
):
    """
    Reset password using only email.
    Finds user by email and updates password.
    No token or current password required.
    """
    db = get_db()
    logger.info(f"🔐 Password reset requested for email: {request.email}")
    
    # Find user by email
    user = db.users.find_one({"email": request.email})
    
    if not user:
        logger.warning(f"⚠️ User not found with email: {request.email}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found with this email address"
        )
    
    logger.info(f"✅ Found user: {user.get('username')}")
    
    # Hash new password
    hashed_password = auth.get_password_hash(request.new_password)
    
    # Update password in database
    result = db.users.update_one(
        {"_id": user["_id"]},
        {"$set": {"hashed_password": hashed_password}}
    )
    
    if result.modified_count == 0:
        logger.error(f"❌ Failed to update password for: {user['username']}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update password"
        )
    
    logger.info(f"✅ Password updated successfully for: {user['username']} ({user['email']})")
    return {"message": "Password changed successfully"}

# ============================================================
# Change Password Endpoint (Requires Current Password)
# ============================================================

@router.post("/change-password", response_model=dict)
async def change_password(
    request: schemas.ChangePasswordRequest,
    current_user: dict = Depends(auth.get_current_user)
):
    """
    Change user's password directly.
    Requires current password verification.
    """
    db = get_db()
    logger.info(f"🔐 Password change requested for: {current_user.get('username')}")
    
    # Get user from database
    user = db.users.find_one({"username": current_user.get("username")})
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Verify current password
    if not auth.verify_password(request.current_password, user.get("hashed_password", "")):
        logger.warning(f"⚠️ Incorrect current password for: {current_user.get('username')}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect"
        )
    
    # Hash new password
    hashed_password = auth.get_password_hash(request.new_password)
    
    # Update password in database
    result = db.users.update_one(
        {"_id": user["_id"]},
        {"$set": {"hashed_password": hashed_password}}
    )
    
    if result.modified_count == 0:
        logger.error(f"❌ Failed to update password for: {current_user.get('username')}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update password"
        )
    
    logger.info(f"✅ Password updated successfully for: {current_user.get('username')}")
    return {"message": "Password changed successfully"}

# ============================================================
# Existing Auth Endpoints
# ============================================================

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
        "profile_picture": None,
        "reset_password_token": None,
        "reset_password_expires": None
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
    logger.info(f"📤 User logged out: {current_user.get('username')}")
    return {"message": "Logged out successfully"}