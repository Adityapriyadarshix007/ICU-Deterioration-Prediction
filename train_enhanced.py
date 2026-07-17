"""
Training Pipeline with Enhanced Features
Includes XGBoost, LightGBM (if available), LSTM comparison
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
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("⚠️ LightGBM not available, skipping")

os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)

print("="*60)
print("TRAINING WITH ENHANCED FEATURES")
print("="*60)

# ============================================================
# STEP 1: Load Enhanced Data
# ============================================================
print("\n[1] Loading enhanced data...")

X = pd.read_csv("outputs/tables/X_features_enhanced.csv")
y = pd.read_csv("outputs/tables/y_target.csv")['deteriorated'].values

print(f"✅ Loaded X: {X.shape}, y: {y.shape}")
print(f"   Class balance: {np.sum(y)} deteriorated, {len(y) - np.sum(y)} stable")

# ============================================================
# STEP 2: Prepare for Model Comparison
# ============================================================
print("\n[2] Preparing data for models...")

# Extract features for tree-based models
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")

# Scale for neural network
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Scaling complete")

# ============================================================
# STEP 3: XGBoost Baseline
# ============================================================
print("\n[3] Training XGBoost...")

scale_pos_weight = len(y_train[y_train == 0]) / (len(y_train[y_train == 1]) + 1e-6)

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

# Find optimal threshold
thresholds = np.linspace(0.1, 0.9, 50)
best_f1 = 0
best_threshold = 0.5

for t in thresholds:
    y_pred = (xgb_proba > t).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

xgb_pred = (xgb_proba > best_threshold).astype(int)
xgb_auc = roc_auc_score(y_test, xgb_proba)
xgb_f1 = f1_score(y_test, xgb_pred)
xgb_recall = recall_score(y_test, xgb_pred)
xgb_precision = precision_score(y_test, xgb_pred)

print(f"   XGBoost AUC: {xgb_auc:.4f}")
print(f"   XGBoost F1: {xgb_f1:.4f}")
print(f"   XGBoost Recall: {xgb_recall:.4f}")
print(f"   XGBoost Precision: {xgb_precision:.4f}")

# ============================================================
# STEP 4: LightGBM (if available)
# ============================================================
if LGB_AVAILABLE:
    print("\n[4] Training LightGBM...")

    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        class_weight='balanced',
        random_state=42
    )

    lgb_model.fit(X_train, y_train)
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]

    lgb_pred = (lgb_proba > best_threshold).astype(int)
    lgb_auc = roc_auc_score(y_test, lgb_proba)
    lgb_f1 = f1_score(y_test, lgb_pred)
    lgb_recall = recall_score(y_test, lgb_pred)
    lgb_precision = precision_score(y_test, lgb_pred)

    print(f"   LightGBM AUC: {lgb_auc:.4f}")
    print(f"   LightGBM F1: {lgb_f1:.4f}")
    print(f"   LightGBM Recall: {lgb_recall:.4f}")
    print(f"   LightGBM Precision: {lgb_precision:.4f}")
else:
    lgb_auc = lgb_f1 = lgb_recall = lgb_precision = 0

# ============================================================
# STEP 5: LSTM with Enhanced Features
# ============================================================
print("\n[5] Training LSTM with enhanced features...")

# Create sequences (hour0 and hour6)
hour0_cols = [c for c in X.columns if '_hour0' in c]
hour6_cols = [c for c in X.columns if '_hour6' in c]

print(f"   Found {len(hour0_cols)} hour0 features")
print(f"   Found {len(hour6_cols)} hour6 features")

# Create sequences
X_seq = np.zeros((len(X), 2, len(hour0_cols)))
X_seq[:, 0, :] = X[hour0_cols].values
X_seq[:, 1, :] = X[hour6_cols].values

# Split
X_train_seq, X_test_seq, y_train, y_test = train_test_split(
    X_seq, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train: {len(X_train_seq)}, Test: {len(X_test_seq)}")

# Scale
X_train_flat = X_train_seq.reshape(-1, X_train_seq.shape[-1])
X_test_flat = X_test_seq.reshape(-1, X_test_seq.shape[-1])
scaler_seq = RobustScaler()
X_train_scaled_seq = scaler_seq.fit_transform(X_train_flat)
X_test_scaled_seq = scaler_seq.transform(X_test_flat)

X_train_norm = X_train_scaled_seq.reshape(X_train_seq.shape)
X_test_norm = X_test_scaled_seq.reshape(X_test_seq.shape)

print("✅ Scaling complete")

# Build LSTM
n_positive = np.sum(y_train)
n_negative = len(y_train) - n_positive
class_weight = {0: 1.0, 1: n_negative / n_positive if n_positive > 0 else 1.0}
print(f"   Class weight for positive: {class_weight[1]:.2f}")

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(2, len(hour0_cols))),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

model.summary()

early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=5, min_lr=0.00001)

history = model.fit(
    X_train_norm, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight,
    verbose=1
)

lstm_proba = model.predict(X_test_norm).flatten()
lstm_auc = roc_auc_score(y_test, lstm_proba)
lstm_f1 = f1_score(y_test, (lstm_proba > best_threshold).astype(int))
lstm_recall = recall_score(y_test, (lstm_proba > best_threshold).astype(int))

print(f"\n   LSTM Enhanced AUC: {lstm_auc:.4f}")
print(f"   LSTM Enhanced F1: {lstm_f1:.4f}")
print(f"   LSTM Enhanced Recall: {lstm_recall:.4f}")

# ============================================================
# STEP 6: Comparison Summary
# ============================================================
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

results = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM', 'LSTM Enhanced'],
    'AUC-ROC': [xgb_auc, lgb_auc, lstm_auc],
    'F1-Score': [xgb_f1, lgb_f1, lstm_f1],
    'Recall': [xgb_recall, lgb_recall, lstm_recall]
})

print(results.to_string(index=False))

# ============================================================
# STEP 7: Save Results
# ============================================================
print("\n[7] Saving results...")

results.to_csv("outputs/tables/enhanced_model_comparison.csv", index=False)
print("✅ Results saved: outputs/tables/enhanced_model_comparison.csv")

# Save best model
import joblib
joblib.dump(xgb_model, "outputs/models/xgboost_enhanced.pkl")
print("✅ XGBoost model saved: outputs/models/xgboost_enhanced.pkl")

model.save("outputs/models/lstm_enhanced_model.h5")
print("✅ LSTM model saved: outputs/models/lstm_enhanced_model.h5")

# ============================================================
# STEP 8: Plot Comparison
# ============================================================
print("\n[8] Creating comparison plot...")

fig, ax = plt.subplots(figsize=(10, 6))
models = ['XGBoost', 'LightGBM', 'LSTM Enhanced']
metrics = ['AUC-ROC', 'F1-Score', 'Recall']
values = [
    [xgb_auc, xgb_f1, xgb_recall],
    [lgb_auc, lgb_f1, lgb_recall],
    [lstm_auc, lstm_f1, lstm_recall]
]

x = np.arange(len(metrics))
width = 0.25

for i, model in enumerate(models):
    ax.bar(x + i*width, values[i], width, label=model)

ax.set_xlabel('Metrics')
ax.set_ylabel('Score')
ax.set_title('Model Comparison with Enhanced Features')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/plots/enhanced_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Comparison plot saved: outputs/plots/enhanced_model_comparison.png")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
