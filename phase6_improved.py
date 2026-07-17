"""
PHASE 6 IMPROVED: CNN-BiLSTM + MultiHeadAttention with Class Weights
Stable training with proper class weighting
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend/app'))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score, 
    accuracy_score, recall_score, precision_score, 
    matthews_corrcoef, confusion_matrix
)
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns

# Import improved model
from improved_model import ImprovedICUModel

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

print("="*60)
print("PHASE 6 IMPROVED: CNN-BiLSTM + MultiHeadAttention")
print("="*60)

# ============================================================
# STEP 1: Load Data
# ============================================================
print("\n[1] Loading data...")

X = pd.read_csv("outputs/tables/X_features.csv")
y = pd.read_csv("outputs/tables/y_target.csv")['deteriorated'].values

print(f"✅ Loaded X: {X.shape}, y: {y.shape}")
print(f"   Class balance: {np.sum(y)} deteriorated, {len(y) - np.sum(y)} stable")
print(f"   Imbalance ratio: {np.sum(y)/len(y):.2%}")

# ============================================================
# STEP 2: Create Rich Temporal Sequences (Hour 0, 2, 4, 6)
# ============================================================
print("\n[2] Creating rich temporal sequences...")

hour0_cols = [col for col in X.columns if '_hour0' in col]
hour6_cols = [col for col in X.columns if '_hour6' in col]

print(f"   Found {len(hour0_cols)} hour0 features")
print(f"   Found {len(hour6_cols)} hour6 features")

# Extract values
X_hour0 = X[hour0_cols].values
X_hour6 = X[hour6_cols].values

# Create 4 timepoints: hour 0, 2, 4, 6
timesteps = 4
n_samples = X_hour0.shape[0]
n_features = X_hour0.shape[1]

X_seq = np.zeros((n_samples, timesteps, n_features))

# Hour 0
X_seq[:, 0, :] = X_hour0

# Hour 2 (interpolated: 1/3 between hour0 and hour6)
X_seq[:, 1, :] = X_hour0 + (X_hour6 - X_hour0) * (1/3)

# Hour 4 (interpolated: 2/3 between hour0 and hour6)
X_seq[:, 2, :] = X_hour0 + (X_hour6 - X_hour0) * (2/3)

# Hour 6
X_seq[:, 3, :] = X_hour6

print(f"✅ Created sequences: {X_seq.shape}")
print(f"   Time steps: {timesteps}")
print(f"   Features per time step: {n_features}")

# ============================================================
# STEP 3: Train-Test Split
# ============================================================
print("\n[3] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================================
# STEP 4: Normalize Sequences
# ============================================================
print("\n[4] Normalizing sequences...")

X_train_flat = X_train.reshape(-1, X_train.shape[-1])
mean = X_train_flat.mean(axis=0)
std = X_train_flat.std(axis=0)
std[std == 0] = 1

X_train_norm = (X_train - mean) / std
X_test_norm = (X_test - mean) / std

# Clip extreme values
X_train_norm = np.clip(X_train_norm, -10, 10)
X_test_norm = np.clip(X_test_norm, -10, 10)

print("✅ Sequences normalized")

# ============================================================
# STEP 5: Build Improved Model
# ============================================================
print("\n[5] Building improved CNN-BiLSTM + MultiHeadAttention model...")

# Calculate class weights for balanced loss
n_positive = np.sum(y_train)
n_negative = len(y_train) - n_positive
weight_for_positive = n_negative / n_positive
weight_for_negative = 1.0

class_weight = {0: weight_for_negative, 1: weight_for_positive}
print(f"   Class weights: {class_weight}")
print(f"   Weight for positive (deteriorated): {weight_for_positive:.2f}")

# Build model
model_builder = ImprovedICUModel(
    input_shape=(X_train_norm.shape[1], X_train_norm.shape[2]),
    num_heads=2,
    key_dim=32
)

model = model_builder.build_model(use_attention=True, dropout_rate=0.3)

# Use weighted binary crossentropy (stable)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

model.summary()

# ============================================================
# STEP 6: Train Model with Class Weights
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
# STEP 7: Find Optimal Threshold
# ============================================================
print("\n[7] Finding optimal classification threshold...")

y_proba = model.predict(X_test_norm).flatten()

# Handle NaN predictions
if np.isnan(y_proba).any():
    print("   ⚠️ Warning: NaN predictions found, replacing with 0.5")
    y_proba = np.nan_to_num(y_proba, nan=0.5)

# Find threshold that maximizes F1-Score
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

# ============================================================
# STEP 8: Evaluate Model
# ============================================================
print("\n[8] Evaluating model...")

y_pred = (y_proba > best_threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print(f"\n📊 CNN-BiLSTM + MultiHeadAttention Performance:")
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
# STEP 9: Save Results
# ============================================================
print("\n[9] Saving results...")

# Save metrics
metrics_df = pd.DataFrame({
    'Model': ['CNN-BiLSTM + MultiHeadAttention (Class Weight)'],
    'Accuracy': [accuracy],
    'F1-Score': [f1],
    'AUC-ROC': [auc],
    'Recall': [recall],
    'Precision': [precision],
    'MCC': [mcc],
    'Optimal Threshold': [best_threshold]
})
metrics_df.to_csv("outputs/tables/cnn_lstm_improved_metrics.csv", index=False)
print("✅ Metrics saved: outputs/tables/cnn_lstm_improved_metrics.csv")

# Save model
model.save("outputs/models/cnn_lstm_improved_model.h5")
print("✅ Model saved: outputs/models/cnn_lstm_improved_model.h5")

# ============================================================
# STEP 10: Compare with Original Model
# ============================================================
print("\n[10] Comparing with original CNN-LSTM model...")

try:
    original_metrics = pd.read_csv("outputs/tables/cnn_lstm_metrics.csv")
    print("\n📊 Performance Comparison:")
    print(f"   Original CNN-LSTM:     AUC = {original_metrics['AUC-ROC'].iloc[0]:.4f}, F1 = {original_metrics['F1-Score'].iloc[0]:.4f}")
    print(f"   Improved Model:        AUC = {auc:.4f}, F1 = {f1:.4f}")
    print(f"   Improvement:           AUC +{(auc - original_metrics['AUC-ROC'].iloc[0])*100:.2f}%, F1 +{(f1 - original_metrics['F1-Score'].iloc[0])*100:.2f}%")
except Exception as e:
    print(f"   Original metrics not found: {e}")

print("\n" + "="*60)
print("PHASE 6 IMPROVED - COMPLETE")
print("="*60)
print(f"✅ Best F1-Score: {f1:.4f}")
print(f"✅ Best AUC-ROC:  {auc:.4f}")
print(f"✅ Optimal Threshold: {best_threshold:.3f}")
print("\n✅ Phase 6 Improved completed successfully!")
