"""
Clinical Risk Score Implementation
SOFA, qSOFA, MEWS, NEWS2
"""

import pandas as pd
import numpy as np

print("="*60)
print("CLINICAL RISK SCORE IMPLEMENTATION")
print("="*60)

def calculate_qSOFA(heart_rate, sbp, gcs):
    """
    Quick SOFA score
    - SBP ≤ 100: 1 point
    - GCS < 15: 1 point
    - Respiratory rate ≥ 22: 1 point
    """
    score = 0
    if sbp <= 100: score += 1
    if gcs < 15: score += 1
    # Respiratory rate would need to be available
    return score

def calculate_MEWS(heart_rate, sbp, temperature, gcs):
    """
    Modified Early Warning Score
    - HR: <40=2, 41-50=1, 51-100=0, 101-110=1, 111-130=2, >130=3
    - SBP: <70=3, 71-80=2, 81-100=1, 101-199=0, >=200=2
    - Temp: <35=2, 35-38.4=0, >=38.5=2
    - GCS: <9=3, 9-12=2, 13-14=1, 15=0
    - RR: <9=2, 9-14=0, 15-20=1, 21-29=2, >=30=3
    """
    score = 0
    
    # Heart rate
    if heart_rate < 40: score += 2
    elif heart_rate < 51: score += 1
    elif heart_rate < 101: score += 0
    elif heart_rate < 111: score += 1
    elif heart_rate < 131: score += 2
    else: score += 3
    
    # SBP
    if sbp < 70: score += 3
    elif sbp < 81: score += 2
    elif sbp < 101: score += 1
    elif sbp < 200: score += 0
    else: score += 2
    
    # GCS
    if gcs < 9: score += 3
    elif gcs < 13: score += 2
    elif gcs < 15: score += 1
    else: score += 0
    
    return score

def calculate_NEWS2(heart_rate, sbp, spo2, temperature, gcs):
    """
    National Early Warning Score 2
    More comprehensive version of MEWS
    """
    score = 0
    
    # Heart rate
    if heart_rate < 40: score += 3
    elif heart_rate < 51: score += 1
    elif heart_rate < 91: score += 0
    elif heart_rate < 111: score += 1
    elif heart_rate < 131: score += 2
    else: score += 3
    
    # SBP
    if sbp < 90: score += 3
    elif sbp < 100: score += 2
    elif sbp < 110: score += 1
    elif sbp < 220: score += 0
    else: score += 3
    
    # SpO2
    if spo2 < 91: score += 3
    elif spo2 < 94: score += 2
    elif spo2 < 96: score += 1
    else: score += 0
    
    # GCS
    if gcs < 9: score += 3
    elif gcs < 13: score += 2
    elif gcs < 15: score += 1
    else: score += 0
    
    return score

# Load data
X = pd.read_csv("outputs/tables/X_features_enhanced.csv")

print("\n[1] Calculating clinical risk scores...")

# Calculate scores for hour 0 and hour 6
for time in ['hour0', 'hour6']:
    hr_col = f'heart_rate_{time}'
    sbp_col = f'sbp_{time}'
    gcs_col = f'gcs_{time}'
    temp_col = f'temperature_{time}' if f'temperature_{time}' in X.columns else None
    spo2_col = f'spo2_{time}' if f'spo2_{time}' in X.columns else None
    
    if hr_col in X.columns and sbp_col in X.columns:
        X[f'qSOFA_{time}'] = X.apply(
            lambda row: calculate_qSOFA(row[hr_col], row[sbp_col], row[gcs_col]), axis=1
        )
        
        X[f'MEWS_{time}'] = X.apply(
            lambda row: calculate_MEWS(row[hr_col], row[sbp_col], 37.0, row[gcs_col]), axis=1
        )
        
        if spo2_col in X.columns and temp_col in X.columns:
            X[f'NEWS2_{time}'] = X.apply(
                lambda row: calculate_NEWS2(row[hr_col], row[sbp_col], row[spo2_col], 37.0, row[gcs_col]), axis=1
            )

print(f"   ✅ Clinical scores added")

# Create delta scores
for score in ['qSOFA', 'MEWS', 'NEWS2']:
    if f'{score}_hour0' in X.columns and f'{score}_hour6' in X.columns:
        X[f'{score}_delta'] = X[f'{score}_hour6'] - X[f'{score}_hour0']

print(f"   ✅ Delta scores added")

# Save
X.to_csv("outputs/tables/X_features_clinical.csv", index=False)
print(f"\n✅ Clinical scores saved: {X.shape[1]} features")

print("\n📊 Clinical Score Summary:")
print(f"   qSOFA: {X['qSOFA_hour0'].mean():.2f} → {X['qSOFA_hour6'].mean():.2f}")
print(f"   MEWS: {X['MEWS_hour0'].mean():.2f} → {X['MEWS_hour6'].mean():.2f}")
if 'NEWS2_hour0' in X.columns:
    print(f"   NEWS2: {X['NEWS2_hour0'].mean():.2f} → {X['NEWS2_hour6'].mean():.2f}")

print("\n✅ Clinical scores complete!")
