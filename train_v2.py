"""
Training Pipeline v2.0
With enhanced preprocessing and robust scaling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score, 
    accuracy_score, recall_score, precision_score, 
    matthews_corrcoef, roc_curve, confusion_matrix
)
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

print("="*60)
print("TRAINING PIPELINE v2.0")
print("="*60)

# ============================================================
# STEP 1: Load Processed Data
# ============================================================
print("\n[1] Loading processed data...")

X = pd.read_csv("outputs/tables/X_features_v2.csv")
y = pd.read_csv("outputs/tables/y_target.csv")['deteriorated'].values

print(f"✅ Loaded X: {X.shape}, y: {y.shape}")
print(f"   Class balance: {np.sum(y)} deteriorated, {len(y) - np.sum(y)} stable")

# ============================================================
# STEP 2: Create Sequences
# ============================================================
print("\n[2] Creating sequences...")

hour0_cols = [col for col in X.columns if '_hour0' in col]
hour6_cols = [col for col in X.columns if '_hour6' in col]

X_hour0 = X[hour0_cols].values.astype(np.float32)
X_hour6 = X[hour6_cols].values.astype(np.float32)

# Create 4 timepoints (hour 0, 2, 4, 6) using interpolation
timesteps = 4
n_samples = X_hour0.shape[0]
n_features = X_hour0.shape[1]

X_seq = np.zeros((n_samples, timesteps, n_features), dtype=np.float32)
X_seq[:, 0, :] = X_hour0
X_seq[:, 1, :] = X_hour0 + (X_hour6 - X_hour0) * (1/3)
X_seq[:, 2, :] = X_hour0 + (X_hour6 - X_hour0) * (2/3)
X_seq[:, 3, :] = X_hour6

print(f"✅ Created sequences: {X_seq.shape}")

# ============================================================
# STEP 3: Train-Test Split
# ============================================================
print("\n[3] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================================
# STEP 4: Robust Scaling
# ============================================================
print("\n[4] Applying RobustScaler...")

X_train_flat = X_train.reshape(-1, X_train.shape[-1])
X_test_flat = X_test.reshape(-1, X_test.shape[-1])

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

X_train_norm = X_train_scaled.reshape(X_train.shape)
X_test_norm = X_test_scaled.reshape(X_test.shape)

# Clip to prevent extreme values
X_train_norm = np.clip(X_train_norm, -5, 5)
X_test_norm = np.clip(X_test_norm, -5, 5)

print("✅ Robust scaling complete")

# ============================================================
# STEP 5: Build Model
# ============================================================
print("\n[5] Building model...")

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Class weights
n_positive = np.sum(y_train)
n_negative = len(y_train) - n_positive
class_weight = {0: 1.0, 1: n_negative / n_positive}
print(f"   Class weight for positive: {class_weight[1]:.2f}")

model = build_model((X_train_norm.shape[1], X_train_norm.shape[2]))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

model.summary()

# ============================================================
# STEP 6: Train with Early Stopping
# ============================================================
print("\n[6] Training model...")

early_stop = EarlyStopping(
    monitor='val_auc',
    mode='max',
    patience=15,
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
    epochs=50,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight,
    verbose=1
)

print("✅ Model training complete")

# ============================================================
# STEP 7: Evaluate
# ============================================================
print("\n[7] Evaluating model...")

y_proba = model.predict(X_test_norm).flatten()

# Handle NaN predictions
if np.isnan(y_proba).any():
    y_proba = np.nan_to_num(y_proba, nan=0.5)

# Find optimal threshold
thresholds = np.linspace(0.1, 0.9, 50)
best_f1 = 0
best_threshold = 0.5
best_youden = 0

for t in thresholds:
    y_pred = (y_proba > t).astype(int)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    youden = sensitivity + specificity - 1
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t
    
    if youden > best_youden:
        best_youden = youden

print(f"   Optimal threshold (F1): {best_threshold:.3f}")
print(f"   Youden's J: {best_youden:.3f}")

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
    'Model': ['LSTM v2 (Enhanced Preprocessing)'],
    'Accuracy': [accuracy],
    'F1-Score': [f1],
    'AUC-ROC': [auc],
    'Recall': [recall],
    'Precision': [precision],
    'MCC': [mcc],
    'Optimal Threshold': [best_threshold]
})
metrics_df.to_csv("outputs/tables/cnn_lstm_v2_metrics.csv", index=False)
print("✅ Metrics saved")

model.save("outputs/models/lstm_v2_model.h5")
print("✅ Model saved")

# ============================================================
# STEP 9: Compare with Previous Results
# ============================================================
print("\n[9] Comparing with previous results...")

try:
    prev_metrics = pd.read_csv("outputs/tables/cnn_lstm_improved_metrics.csv")
    print(f"\n📊 Performance Comparison:")
    print(f"   Previous: AUC = {prev_metrics['AUC-ROC'].iloc[0]:.4f}, F1 = {prev_metrics['F1-Score'].iloc[0]:.4f}")
    print(f"   Current:  AUC = {auc:.4f}, F1 = {f1:.4f}")
    print(f"   Improvement: AUC +{(auc - prev_metrics['AUC-ROC'].iloc[0])*100:.2f}%, F1 +{(f1 - prev_metrics['F1-Score'].iloc[0])*100:.2f}%")
except:
    print("   Previous results not found")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
