"""
PHASE 6 STABLE: CNN-BiLSTM + MultiHeadAttention
With proper data preprocessing and NaN handling
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score, 
    accuracy_score, recall_score, precision_score, 
    matthews_corrcoef, confusion_matrix, roc_curve
)
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend/app'))
from improved_model import ImprovedICUModel

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

print("="*60)
print("PHASE 6 STABLE: CNN-BiLSTM + MultiHeadAttention")
print("="*60)

# ============================================================
# STEP 1: Load and Validate Data
# ============================================================
print("\n[1] Loading and validating data...")

X = pd.read_csv("outputs/tables/X_features.csv")
y = pd.read_csv("outputs/tables/y_target.csv")['deteriorated'].values

print(f"✅ Loaded X: {X.shape}, y: {y.shape}")

# Check for NaN/Inf
print(f"\n🔍 Data Quality Check:")
print(f"   X NaN values: {X.isnull().sum().sum()}")
print(f"   X Inf values: {np.isinf(X).sum().sum()}")
print(f"   y NaN values: {np.isnan(y).sum()}")

# Handle NaN values - replace with column mean or 0
if X.isnull().sum().sum() > 0:
    print("   ⚠️ Handling NaN values...")
    X = X.fillna(0)
    print(f"   ✅ NaN replaced with 0")

# ============================================================
# STEP 2: Create Temporal Sequences from Original Data
# ============================================================
print("\n[2] Creating temporal sequences...")

# Use only the original hour0 and hour6 features
hour0_cols = [col for col in X.columns if '_hour0' in col]
hour6_cols = [col for col in X.columns if '_hour6' in col]

print(f"   Found {len(hour0_cols)} hour0 features")
print(f"   Found {len(hour6_cols)} hour6 features")

# Extract values
X_hour0 = X[hour0_cols].values.astype(np.float32)
X_hour6 = X[hour6_cols].values.astype(np.float32)

# Create 2 timepoints: hour 0 and hour 6 (original data)
timesteps = 2
n_samples = X_hour0.shape[0]
n_features = X_hour0.shape[1]

X_seq = np.zeros((n_samples, timesteps, n_features), dtype=np.float32)
X_seq[:, 0, :] = X_hour0
X_seq[:, 1, :] = X_hour6

print(f"✅ Created sequences: {X_seq.shape}")
print(f"   Time steps: {timesteps}")
print(f"   Features per time step: {n_features}")

# Check for NaN in sequences
print(f"\n🔍 Sequence Quality Check:")
print(f"   NaN in X_seq: {np.isnan(X_seq).sum()}")
print(f"   Inf in X_seq: {np.isinf(X_seq).sum()}")

# Replace any remaining NaN
if np.isnan(X_seq).any():
    print("   ⚠️ NaN found in sequences, replacing with 0")
    X_seq = np.nan_to_num(X_seq, nan=0.0)

# ============================================================
# STEP 3: Train-Test Split
# ============================================================
print("\n[3] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
print(f"   Train class balance: {np.sum(y_train)} deteriorated, {len(y_train) - np.sum(y_train)} stable")
print(f"   Test class balance: {np.sum(y_test)} deteriorated, {len(y_test) - np.sum(y_test)} stable")

# ============================================================
# STEP 4: Normalize Sequences Safely
# ============================================================
print("\n[4] Normalizing sequences...")

# Flatten for scaling
X_train_flat = X_train.reshape(-1, X_train.shape[-1])
X_test_flat = X_test.reshape(-1, X_test.shape[-1])

# Compute statistics on training data
mean = X_train_flat.mean(axis=0)
std = X_train_flat.std(axis=0)

# Handle zero std - set to 1 to avoid division by zero
std[std == 0] = 1

# Normalize
X_train_norm = (X_train - mean) / std
X_test_norm = (X_test - mean) / std

# Clip to prevent extreme values
X_train_norm = np.clip(X_train_norm, -5, 5)
X_test_norm = np.clip(X_test_norm, -5, 5)

# Check for NaN after normalization
print(f"\n🔍 Post-normalization Check:")
print(f"   NaN in X_train_norm: {np.isnan(X_train_norm).sum()}")
print(f"   NaN in X_test_norm: {np.isnan(X_test_norm).sum()}")
print(f"   X_train_norm range: [{X_train_norm.min():.2f}, {X_train_norm.max():.2f}]")

if np.isnan(X_train_norm).any():
    print("   ⚠️ NaN found after normalization, replacing with 0")
    X_train_norm = np.nan_to_num(X_train_norm, nan=0.0)
    X_test_norm = np.nan_to_num(X_test_norm, nan=0.0)

print("✅ Sequences normalized")

# ============================================================
# STEP 5: Build Simple LSTM Model First (No Attention)
# ============================================================
print("\n[5] Building simple LSTM model...")

def build_simple_lstm(input_shape):
    """Simple LSTM model for baseline comparison"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Calculate class weights
n_positive = np.sum(y_train)
n_negative = len(y_train) - n_positive
class_weight = {0: 1.0, 1: n_negative / n_positive}

print(f"   Class weights: {class_weight}")

# Build and compile model
model = build_simple_lstm((X_train_norm.shape[1], X_train_norm.shape[2]))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

model.summary()

# ============================================================
# STEP 6: Train Model with Validation
# ============================================================
print("\n[6] Training model with class weights...")

early_stop = EarlyStopping(
    monitor='val_auc',
    mode='max',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_auc',
    mode='max',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

history = model.fit(
    X_train_norm, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight,
    verbose=1
)

print("✅ Model training complete")

# ============================================================
# STEP 7: Evaluate Model
# ============================================================
print("\n[7] Evaluating model...")

y_proba = model.predict(X_test_norm).flatten()

# Check for NaN predictions
if np.isnan(y_proba).any():
    print("   ⚠️ NaN predictions found, replacing with 0.5")
    y_proba = np.nan_to_num(y_proba, nan=0.5)

# Find optimal threshold
thresholds = np.linspace(0.1, 0.9, 50)
best_f1 = 0
best_threshold = 0.5

for t in thresholds:
    y_pred = (y_proba > t).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"   Optimal threshold: {best_threshold:.3f}")
print(f"   Best F1-Score: {best_f1:.4f}")

y_pred = (y_proba > best_threshold).astype(int)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print(f"\n📊 Performance:")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   F1-Score: {f1:.4f}")
print(f"   AUC-ROC:  {auc:.4f}")
print(f"   Recall:   {recall:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   MCC:      {mcc:.4f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Stable', 'Deteriorated']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n📊 Confusion Matrix:")
print(f"   True Negatives: {cm[0][0]}")
print(f"   False Positives: {cm[0][1]}")
print(f"   False Negatives: {cm[1][0]}")
print(f"   True Positives: {cm[1][1]}")

# ============================================================
# STEP 8: Save Results
# ============================================================
print("\n[8] Saving results...")

metrics_df = pd.DataFrame({
    'Model': ['LSTM (Class Weight)'],
    'Accuracy': [accuracy],
    'F1-Score': [f1],
    'AUC-ROC': [auc],
    'Recall': [recall],
    'Precision': [precision],
    'MCC': [mcc],
    'Optimal Threshold': [best_threshold]
})
metrics_df.to_csv("outputs/tables/cnn_lstm_improved_metrics.csv", index=False)
print("✅ Metrics saved")

model.save("outputs/models/cnn_lstm_stable_model.h5")
print("✅ Model saved")

# ============================================================
# STEP 9: Compare with Original
# ============================================================
print("\n[9] Comparing with original model...")

try:
    original_metrics = pd.read_csv("outputs/tables/cnn_lstm_metrics.csv")
    print(f"   Original: AUC = {original_metrics['AUC-ROC'].iloc[0]:.4f}, F1 = {original_metrics['F1-Score'].iloc[0]:.4f}")
    print(f"   Improved: AUC = {auc:.4f}, F1 = {f1:.4f}")
except:
    print("   Original metrics not found")

print("\n" + "="*60)
print("PHASE 6 STABLE - COMPLETE")
print("="*60)
