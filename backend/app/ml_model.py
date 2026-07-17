import numpy as np
import os
from datetime import datetime
import json
import random
from app.schemas import PredictionResponse

class MLModel:
    def __init__(self):
        self.features = [
            'heart_rate', 'sbp', 'dbp', 'gcs', 
            'lactate', 'urine_output', 'fio2', 'creatinine'
        ]
        self.model_loaded = False
        print("✅ ML Model initialized in mock mode")
    
    def preprocess_features(self, patient_data):
        """Convert input to model-ready format"""
        features = np.array([
            patient_data.heart_rate,
            patient_data.sbp,
            patient_data.dbp,
            patient_data.gcs,
            patient_data.lactate,
            patient_data.urine_output,
            patient_data.fio2,
            patient_data.creatinine
        ])
        return features.reshape(1, 1, 8)
    
    def predict(self, patient_data):
        """Make prediction for a single patient"""
        # Mock prediction based on some clinical rules
        risk_score = self._calculate_mock_risk(patient_data)
        
        # Determine alert level
        if risk_score > 0.7:
            alert_level = "CRITICAL"
            confidence = "HIGH"
        elif risk_score > 0.5:
            alert_level = "WARNING"
            confidence = "MEDIUM"
        else:
            alert_level = "STABLE"
            confidence = "LOW"
        
        return PredictionResponse(
            patient_id=patient_data.patient_id,
            risk_score=round(risk_score, 4),
            risk_percentage=round(risk_score * 100, 2),
            alert_level=alert_level,
            confidence=confidence,
            features_used=self.features,
            predicted_at=datetime.utcnow()
        )
    
    def _calculate_mock_risk(self, data):
        """Calculate a mock risk score based on clinical parameters"""
        # This mimics clinical reasoning without a real model
        score = 0.0
        
        # Heart Rate - high or low indicates risk
        if data.heart_rate > 100 or data.heart_rate < 60:
            score += 0.15
        
        # Blood Pressure - hypotension is risky
        if data.sbp < 90:
            score += 0.20
        elif data.sbp < 100:
            score += 0.10
        
        # GCS - low indicates neurological risk
        if data.gcs < 9:
            score += 0.25
        elif data.gcs < 13:
            score += 0.15
        
        # Lactate - high indicates tissue hypoperfusion
        if data.lactate > 4:
            score += 0.25
        elif data.lactate > 2:
            score += 0.10
        
        # Creatinine - high indicates renal risk
        if data.creatinine > 2:
            score += 0.15
        elif data.creatinine > 1.5:
            score += 0.10
        
        # FiO2 - high oxygen requirement
        if data.fio2 > 60:
            score += 0.15
        elif data.fio2 > 40:
            score += 0.08
        
        # Add some randomness to make it realistic
        score += random.uniform(-0.05, 0.05)
        
        return max(0.0, min(1.0, score))

# Singleton instance
ml_model = MLModel()
