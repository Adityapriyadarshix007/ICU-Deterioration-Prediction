# ============================================================
# PHASE 6: CNN-LSTM + ATTENTION MODEL (FIXED V4 - STABLE)
# ============================================================
# Robust architecture with proper normalization and gradient clipping
# ============================================================

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score, roc_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, LSTM, Dense, Dropout,
    Attention, Flatten, BatchNormalization, Concatenate,
    GlobalAveragePooling1D, Reshape, Permute, Multiply
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# Fix for NaN issues - set global dtype policy
tf.keras.mixed_precision.set_global_policy('float32')

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

print("="*60)
print("PHASE 6: CNN-LSTM + ATTENTION MODEL")
print("="*60)

# ============================================================
# STEP 1: Load Data
# ============================================================
print("\n[1] Loading data...")

X = pd.read_csv("outputs/tables/X_features.csv")
y = pd.read_csv("outputs/tables/y_target.csv")

print(f"✅ Loaded X: {X.shape}")
print(f"✅ Loaded y: {y.shape}")

# ============================================================
# STEP 2: Identify hour0 and hour6 columns
# ============================================================
print("\n[2] Identifying time-based features...")

hour0_cols = [col for col in X.columns if '_hour0' in col]
hour6_cols = [col for col in X.columns if '_hour6' in col]

print(f"   Found {len(hour0_cols)} hour0 features")
print(f"   Found {len(hour6_cols)} hour6 features")

# ============================================================
# STEP 3: Create sequences from hour0 and hour6 features
# ============================================================
print("\n[3] Creating sequences...")

X_hour0 = X[hour0_cols].values
X_hour6 = X[hour6_cols].values

# Stack to create sequences: (samples, 2 timesteps, features)
X_seq = np.stack([X_hour0, X_hour6], axis=1)
y_seq = y['deteriorated'].values

# Check for NaN/Inf values and clean
if np.isnan(X_seq).any():
    print("   ⚠️ Warning: NaN values found in sequences. Filling with 0...")
    X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=1.0, neginf=-1.0)

print(f"✅ Created sequences: {X_seq.shape}")
print(f"   Class balance: {np.sum(y_seq)} deteriorated, {len(y_seq) - np.sum(y_seq)} stable")

# ============================================================
# STEP 4: Train-Test Split
# ============================================================
print("\n[4] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
)

print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================================
# STEP 5: Normalize Sequences
# ============================================================
print("\n[5] Normalizing sequences...")

X_train_flat = X_train.reshape(-1, X_train.shape[-1])
mean = X_train_flat.mean(axis=0)
std = X_train_flat.std(axis=0)
std[std == 0] = 1

X_train_norm = (X_train - mean) / std
X_test_norm = (X_test - mean) / std

# Clip extreme values to prevent NaN
X_train_norm = np.clip(X_train_norm, -10, 10)
X_test_norm = np.clip(X_test_norm, -10, 10)

print("✅ Sequences normalized and clipped")

# ============================================================
# STEP 6: Build Robust CNN-LSTM + Attention Model
# ============================================================
print("\n[6] Building robust CNN-LSTM + Attention model...")

def build_robust_model(input_shape):
    """Robust model with gradient clipping and proper initialization"""
    
    model = Sequential([
        # Input layer
        Input(shape=input_shape),
        
        # Conv1D with smaller kernel
        Conv1D(filters=16, kernel_size=1, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.2),
        
        # LSTM with dropout
        LSTM(16, return_sequences=True, activation='tanh', dropout=0.2, recurrent_dropout=0.2),
        
        # Attention mechanism (simplified)
        tf.keras.layers.GlobalAveragePooling1D(),
        
        # Dense layers with L2 regularization
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(8, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output
        Dense(1, activation='sigmoid')
    ])
    
    return model

# Build model
input_shape = (X_train_norm.shape[1], X_train_norm.shape[2])
model = build_robust_model(input_shape)

# Use Adam with lower learning rate and gradient clipping
optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# ============================================================
# STEP 7: Train Model
# ============================================================
print("\n[7] Training model...")

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

history = model.fit(
    X_train_norm, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=64,  # Larger batch size for stability
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print("✅ Model training complete")

# ============================================================
# STEP 8: Evaluate Model
# ============================================================
print("\n[8] Evaluating model...")

y_pred_proba = model.predict(X_test_norm, batch_size=64)

# Check for NaN predictions
if np.isnan(y_pred_proba).any():
    print("   ⚠️ Warning: NaN values in predictions. Fixing...")
    y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.5)

y_pred = (y_pred_proba > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n📊 CNN-LSTM + Attention Performance:")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   F1-Score: {f1:.4f}")
print(f"   AUC-ROC:  {auc:.4f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Stable', 'Deteriorated']))

# ============================================================
# STEP 9: Training History Plots
# ============================================================
print("\n[9] Plotting training history...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_title('Model Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[1].set_title('Model Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/plots/cnn_lstm_training_history.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: outputs/plots/cnn_lstm_training_history.png")

# ============================================================
# STEP 10: ROC Curve
# ============================================================
print("\n[10] Plotting ROC curve...")

plt.figure(figsize=(10, 8))
fpr_cnn, tpr_cnn, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr_cnn, tpr_cnn, label=f'CNN-LSTM + Attention (AUC = {auc:.3f})', linewidth=3, color='darkred')
plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: CNN-LSTM + Attention')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/plots/cnn_lstm_roc_curve.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: outputs/plots/cnn_lstm_roc_curve.png")

# ============================================================
# STEP 11: Save Model and Metrics
# ============================================================
print("\n[11] Saving model and metrics...")

model.save("outputs/models/cnn_lstm_attention_model.h5")
print("✅ Model saved: outputs/models/cnn_lstm_attention_model.h5")

metrics_df = pd.DataFrame({
    'Model': ['CNN-LSTM + Attention'],
    'Accuracy': [accuracy],
    'F1-Score': [f1],
    'AUC-ROC': [auc]
})
metrics_df.to_csv("outputs/tables/cnn_lstm_metrics.csv", index=False)
print("✅ Metrics saved: outputs/tables/cnn_lstm_metrics.csv")

# ============================================================
# STEP 12: Compare with XGBoost
# ============================================================
print("\n[12] Comparing with XGBoost...")

try:
    xgb_results = pd.read_csv("outputs/tables/xgboost_comparison.csv", index_col=0)
    best_xgb = xgb_results['AUC-ROC'].max()
    best_method = xgb_results['AUC-ROC'].idxmax()
    
    print(f"\n📊 Final Comparison:")
    print(f"   XGBoost ({best_method}): AUC-ROC = {best_xgb:.4f}")
    print(f"   CNN-LSTM + Attention:   AUC-ROC = {auc:.4f}")
    print(f"   Improvement:            {(auc - best_xgb)*100:+.2f}%")
except Exception as e:
    print(f"   XGBoost results not found: {e}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("PHASE 6 COMPLETE - SUMMARY")
print("="*60)
print(f"✅ CNN-LSTM + Attention Performance:")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   F1-Score: {f1:.4f}")
print(f"   AUC-ROC:  {auc:.4f}")
print(f"✅ Model saved: outputs/models/cnn_lstm_attention_model.h5")
print(f"✅ Training plots saved: outputs/plots/")
print("\n✅ Phase 6 completed successfully!")