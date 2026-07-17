"""
Admin - Patient Management Routes
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import StreamingResponse
from datetime import datetime
import logging
import csv
import io
import json
from bson import ObjectId
from typing import List

from app import auth
from app.database import Database

router = APIRouter()
logger = logging.getLogger(__name__)

def get_db():
    return Database.get_db()

async def check_admin(current_user: dict):
    if not current_user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


@router.get("/patients")
async def get_patients(
    skip: int = 0,
    limit: int = 100,
    search: str = "",
    current_user: dict = Depends(auth.get_current_user)
):
    """Get all patients with pagination and search"""
    await check_admin(current_user)
    db = get_db()
    
    query = {}
    if search:
        query = {
            "$or": [
                {"patient_id": {"$regex": search, "$options": "i"}},
                {"patient_name": {"$regex": search, "$options": "i"}}
            ]
        }
    
    total = db.patients.count_documents(query) if "patients" in db.list_collection_names() else 0
    patients = []
    
    if total > 0:
        cursor = db.patients.find(query).skip(skip).limit(limit).sort("created_at", -1)
        for p in cursor:
            p["id"] = str(p["_id"])
            p.pop("_id", None)
            patients.append(p)
    
    return {
        "total": total,
        "patients": patients
    }


@router.post("/patients")
async def create_patient(
    patient_data: dict,
    current_user: dict = Depends(auth.get_current_user)
):
    """Create a new patient record"""
    await check_admin(current_user)
    db = get_db()
    
    # Ensure patients collection exists
    if "patients" not in db.list_collection_names():
        db.create_collection("patients")
    
    # Generate patient ID if not provided
    if not patient_data.get("patient_id"):
        count = db.patients.count_documents({})
        patient_data["patient_id"] = f"PAT-{count + 1:05d}"
    
    patient_data["created_at"] = datetime.utcnow()
    patient_data["updated_at"] = datetime.utcnow()
    patient_data["created_by"] = current_user["username"]
    
    result = db.patients.insert_one(patient_data)
    patient_data["id"] = str(result.inserted_id)
    patient_data.pop("_id", None)
    
    logger.info(f"✅ Patient created: {patient_data['patient_id']} by {current_user['username']}")
    return patient_data


@router.put("/patients/{patient_id}")
async def update_patient(
    patient_id: str,
    patient_data: dict,
    current_user: dict = Depends(auth.get_current_user)
):
    """Update a patient record"""
    await check_admin(current_user)
    db = get_db()
    
    try:
        obj_id = ObjectId(patient_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid patient ID")
    
    # Remove fields that shouldn't be updated
    patient_data.pop("_id", None)
    patient_data.pop("id", None)
    patient_data.pop("created_at", None)
    patient_data["updated_at"] = datetime.utcnow()
    
    result = db.patients.update_one(
        {"_id": obj_id},
        {"$set": patient_data}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    logger.info(f"✅ Patient updated: {patient_id} by {current_user['username']}")
    return {"message": "Patient updated successfully"}


@router.delete("/patients/{patient_id}")
async def delete_patient(
    patient_id: str,
    current_user: dict = Depends(auth.get_current_user)
):
    """Delete a patient record"""
    await check_admin(current_user)
    db = get_db()
    
    try:
        obj_id = ObjectId(patient_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid patient ID")
    
    result = db.patients.delete_one({"_id": obj_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    logger.info(f"✅ Patient deleted: {patient_id} by {current_user['username']}")
    return {"message": "Patient deleted successfully"}


@router.post("/patients/import")
async def import_patients_csv(
    file: UploadFile = File(...),
    current_user: dict = Depends(auth.get_current_user)
):
    """Import patients from CSV file"""
    await check_admin(current_user)
    db = get_db()
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        content = await file.read()
        csv_data = content.decode('utf-8')
        csv_reader = csv.DictReader(io.StringIO(csv_data))
        
        imported = 0
        errors = []
        
        for row in csv_reader:
            try:
                # Validate required fields
                if not row.get('patient_name'):
                    errors.append(f"Missing patient_name in row {imported + 1}")
                    continue
                
                # Add timestamps
                row["created_at"] = datetime.utcnow()
                row["updated_at"] = datetime.utcnow()
                row["created_by"] = current_user["username"]
                
                # Generate patient ID if not provided
                if not row.get('patient_id'):
                    count = db.patients.count_documents({})
                    row["patient_id"] = f"PAT-{count + imported + 1:05d}"
                
                db.patients.insert_one(row)
                imported += 1
            except Exception as e:
                errors.append(f"Error in row {imported + 1}: {str(e)}")
        
        logger.info(f"✅ Imported {imported} patients by {current_user['username']}")
        
        return {
            "message": f"Successfully imported {imported} patients",
            "imported": imported,
            "errors": errors if errors else None
        }
        
    except Exception as e:
        logger.error(f"Import error: {e}")
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


@router.get("/patients/export")
async def export_patients_csv(
    current_user: dict = Depends(auth.get_current_user)
):
    """Export patients to CSV file"""
    await check_admin(current_user)
    db = get_db()
    
    patients = list(db.patients.find())
    
    if not patients:
        raise HTTPException(status_code=404, detail="No patients found")
    
    # Create CSV
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Get all fields except _id
    sample = patients[0]
    fields = [k for k in sample.keys() if k not in ['_id', 'created_at', 'updated_at']]
    writer.writerow(fields)
    
    for patient in patients:
        row = []
        for field in fields:
            value = patient.get(field, '')
            if isinstance(value, datetime):
                value = value.isoformat()
            row.append(value)
        writer.writerow(row)
    
    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=patients_export.csv"}
    )


@router.get("/patients/stats")
async def get_patient_stats(
    current_user: dict = Depends(auth.get_current_user)
):
    """Get patient statistics"""
    await check_admin(current_user)
    db = get_db()
    
    total = db.patients.count_documents({}) if "patients" in db.list_collection_names() else 0
    
    # Get recent patients
    recent = []
    if total > 0:
        cursor = db.patients.find().sort("created_at", -1).limit(5)
        for p in cursor:
            p["id"] = str(p["_id"])
            p.pop("_id", None)
            recent.append(p)
    
    return {
        "total_patients": total,
        "recent_patients": recent,
        "last_updated": datetime.utcnow().isoformat()
    }
