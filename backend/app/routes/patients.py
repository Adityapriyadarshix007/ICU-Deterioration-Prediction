"""
Patient Routes - Public access for doctors
"""

from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime
import logging
from typing import Optional

from app import auth, schemas
from app.database import Database
from app.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

def get_db():
    return Database.get_db()


@router.get("/patients")
async def get_patients(
    search: str = "",
    limit: int = 100,
    skip: int = 0,
    current_user: dict = Depends(auth.get_current_user)
):
    """
    Get patients - accessible by any authenticated user (doctors, clinicians, admins)
    """
    db = get_db()
    logger.info(f"📋 Fetching patients for user: {current_user.get('username')}")
    
    # Build query
    query = {}
    if search:
        query = {
            "$or": [
                {"patient_id": {"$regex": search, "$options": "i"}},
                {"patient_name": {"$regex": search, "$options": "i"}},
                {"name": {"$regex": search, "$options": "i"}}
            ]
        }
    
    try:
        # Get total count
        total = db.patients.count_documents(query) if "patients" in db.list_collection_names() else 0
        patients = []
        
        if total > 0:
            cursor = db.patients.find(query).skip(skip).limit(limit).sort("created_at", -1)
            for p in cursor:
                p["id"] = str(p["_id"])
                p.pop("_id", None)
                patients.append(p)
        
        logger.info(f"✅ Found {len(patients)} patients")
        
        return {
            "total": total,
            "patients": patients,
            "limit": limit,
            "skip": skip
        }
    except Exception as e:
        logger.error(f"❌ Error fetching patients: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch patients: {str(e)}"
        )


@router.get("/patients/{patient_id}")
async def get_patient(
    patient_id: str,
    current_user: dict = Depends(auth.get_current_user)
):
    """
    Get a single patient by ID - accessible by any authenticated user
    """
    db = get_db()
    logger.info(f"📋 Fetching patient: {patient_id} for user: {current_user.get('username')}")
    
    try:
        patient = db.patients.find_one({
            "$or": [
                {"patient_id": patient_id},
                {"id": patient_id}
            ]
        })
        
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found"
            )
        
        patient["id"] = str(patient["_id"])
        patient.pop("_id", None)
        
        return patient
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching patient: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch patient: {str(e)}"
        )


@router.post("/patients")
async def create_patient(
    patient_data: dict,
    current_user: dict = Depends(auth.get_current_user)
):
    """
    Create a new patient - accessible by any authenticated user
    """
    db = get_db()
    logger.info(f"📝 Creating patient for user: {current_user.get('username')}")
    
    try:
        # Generate patient ID if not provided
        if "patient_id" not in patient_data or not patient_data["patient_id"]:
            patient_data["patient_id"] = f"PAT-{datetime.utcnow().strftime('%Y%m%d')}-{str(datetime.utcnow().timestamp()).replace('.', '')[-6:]}"
        
        # Add metadata
        patient_data["created_at"] = datetime.utcnow()
        patient_data["updated_at"] = datetime.utcnow()
        patient_data["created_by"] = current_user.get("username")
        
        result = db.patients.insert_one(patient_data)
        patient_data["id"] = str(result.inserted_id)
        
        logger.info(f"✅ Patient created: {patient_data['patient_id']}")
        return patient_data
    except Exception as e:
        logger.error(f"❌ Error creating patient: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create patient: {str(e)}"
        )


@router.put("/patients/{patient_id}")
async def update_patient(
    patient_id: str,
    patient_data: dict,
    current_user: dict = Depends(auth.get_current_user)
):
    """
    Update a patient - accessible by any authenticated user
    """
    db = get_db()
    logger.info(f"📝 Updating patient: {patient_id} for user: {current_user.get('username')}")
    
    try:
        # Find the patient
        existing = db.patients.find_one({
            "$or": [
                {"patient_id": patient_id},
                {"id": patient_id}
            ]
        })
        
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found"
            )
        
        # Remove id from update data
        patient_data.pop("id", None)
        patient_data.pop("_id", None)
        
        # Add update metadata
        patient_data["updated_at"] = datetime.utcnow()
        patient_data["updated_by"] = current_user.get("username")
        
        result = db.patients.update_one(
            {"_id": existing["_id"]},
            {"$set": patient_data}
        )
        
        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No changes made"
            )
        
        logger.info(f"✅ Patient updated: {patient_id}")
        return {"message": "Patient updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error updating patient: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update patient: {str(e)}"
        )


@router.delete("/patients/{patient_id}")
async def delete_patient(
    patient_id: str,
    current_user: dict = Depends(auth.get_current_user)
):
    """
    Delete a patient - accessible by any authenticated user
    """
    db = get_db()
    logger.info(f"🗑️ Deleting patient: {patient_id} for user: {current_user.get('username')}")
    
    try:
        result = db.patients.delete_one({
            "$or": [
                {"patient_id": patient_id},
                {"id": patient_id}
            ]
        })
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found"
            )
        
        logger.info(f"✅ Patient deleted: {patient_id}")
        return {"message": "Patient deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting patient: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete patient: {str(e)}"
        )
