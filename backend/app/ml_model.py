"""
ML Model - Mock implementation for ICU Deterioration Prediction
"""

import random
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MLModel:
    def __init__(self):
        self.model_loaded = True
        self.feature_names = ['heart_rate', 'sbp', 'dbp', 'gcs', 'lactate', 'urine_output', 'fio2', 'creatinine']
        logger.info("✅ ML Model initialized in mock mode")
    
    def predict(self, patient_data):
        """
        Make prediction for a single patient.
        Accepts PatientData object or dict.
        """
        try:
            # Convert to dict if it's a Pydantic model
            if hasattr(patient_data, 'dict'):
                data = patient_data.dict()
            else:
                data = patient_data
            
            # Extract features
            heart_rate = float(data.get('heart_rate', 70))
            sbp = float(data.get('sbp', 120))
            dbp = float(data.get('dbp', 80))
            gcs = float(data.get('gcs', 15))
            lactate = float(data.get('lactate', 1.0))
            urine_output = float(data.get('urine_output', 50))
            fio2 = float(data.get('fio2', 21))
            creatinine = float(data.get('creatinine', 0.9))
            
            # Calculate risk score based on clinical parameters
            risk_score = self._calculate_risk_score(
                heart_rate, sbp, dbp, gcs, 
                lactate, urine_output, fio2, creatinine
            )
            
            # Add small random variation for realism
            risk_score += random.uniform(-0.05, 0.05)
            risk_score = max(0.0, min(1.0, risk_score))
            
            logger.info(f"📊 Calculated risk score: {risk_score:.4f}")
            
            return risk_score
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            # Return a default low risk score
            return 0.3
    
    def _calculate_risk_score(self, heart_rate, sbp, dbp, gcs, lactate, urine_output, fio2, creatinine):
        """Calculate risk score using clinical rules"""
        score = 0.0
        
        # Heart Rate (normal: 60-100)
        if heart_rate > 120 or heart_rate < 50:
            score += 0.20
        elif heart_rate > 100 or heart_rate < 60:
            score += 0.10
        
        # Systolic BP (normal: 90-140)
        if sbp < 80:
            score += 0.25
        elif sbp < 90:
            score += 0.15
        elif sbp > 180:
            score += 0.10
        
        # Diastolic BP (normal: 60-90)
        if dbp < 40 or dbp > 110:
            score += 0.15
        elif dbp < 60 or dbp > 90:
            score += 0.08
        
        # GCS (normal: 13-15)
        if gcs < 9:
            score += 0.30
        elif gcs < 13:
            score += 0.20
        elif gcs < 15:
            score += 0.05
        
        # Lactate (normal: <2.0)
        if lactate > 4.0:
            score += 0.25
        elif lactate > 2.0:
            score += 0.15
        
        # Urine Output (normal: >30 ml/hr)
        if urine_output < 20:
            score += 0.15
        elif urine_output < 40:
            score += 0.08
        
        # FiO2 (normal: 21-40)
        if fio2 > 60:
            score += 0.15
        elif fio2 > 40:
            score += 0.08
        
        # Creatinine (normal: 0.6-1.3)
        if creatinine > 2.0:
            score += 0.20
        elif creatinine > 1.5:
            score += 0.10
        elif creatinine > 1.3:
            score += 0.05
        
        return min(score, 0.95)

# Singleton instance
ml_model = MLModel()

# For backward compatibility
def get_ml_model():
    return ml_model
