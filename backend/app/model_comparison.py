"""
Model Comparison Framework
Trains and evaluates multiple models for fair comparison
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class ModelComparison:
    """Train and compare multiple models"""
    
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.results = {}
        
    def train_all_models(self):
        """Train all baseline models"""
        
        # 1. Logistic Regression
        self._train_logistic_regression()
        
        # 2. Random Forest
        self._train_random_forest()
        
        # 3. XGBoost
        self._train_xgboost()
        
        # 4. LightGBM
        self._train_lightgbm()
        
        # 5. LSTM
        self._train_lstm()
        
        # 6. GRU
        self._train_gru()
        
        logger.info("✅ All models trained successfully")
        
    def _train_logistic_regression(self):
        """Train Logistic Regression"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train_scaled, self.y_train)
        
        self.models['Logistic Regression'] = {
            'model': model,
            'scaler': scaler,
            'predict': lambda X: model.predict_proba(scaler.transform(X))[:, 1]
        }
    
    def _train_random_forest(self):
        """Train Random Forest"""
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        
        self.models['Random Forest'] = {
            'model': model,
            'predict': lambda X: model.predict_proba(X)[:, 1]
        }
    
    def _train_xgboost(self):
        """Train XGBoost"""
        # Handle class imbalance
        scale_pos_weight = len(self.y_train[self.y_train == 0]) / len(self.y_train[self.y_train == 1])
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(self.X_train, self.y_train)
        
        self.models['XGBoost'] = {
            'model': model,
            'predict': lambda X: model.predict_proba(X)[:, 1]
        }
    
    def _train_lightgbm(self):
        """Train LightGBM"""
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        
        self.models['LightGBM'] = {
            'model': model,
            'predict': lambda X: model.predict_proba(X)[:, 1]
        }
    
    def _train_lstm(self):
        """Train LSTM"""
        # Reshape for LSTM: (samples, timesteps, features)
        timesteps = 2  # hour 0 and hour 6
        features = self.X_train.shape[1] // 2  # 8 features
        
        X_train_lstm = self.X_train.reshape(-1, timesteps, features)
        X_test_lstm = self.X_test.reshape(-1, timesteps, features)
        
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(timesteps, features)),
            Dropout(0.2),
            LSTM(16),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        model.fit(
            X_train_lstm, self.y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        self.models['LSTM'] = {
            'model': model,
            'predict': lambda X: model.predict(X.reshape(-1, timesteps, features), verbose=0).flatten()
        }
    
    def _train_gru(self):
        """Train GRU"""
        timesteps = 2
        features = self.X_train.shape[1] // 2
        
        X_train_gru = self.X_train.reshape(-1, timesteps, features)
        X_test_gru = self.X_test.reshape(-1, timesteps, features)
        
        model = Sequential([
            GRU(32, return_sequences=True, input_shape=(timesteps, features)),
            Dropout(0.2),
            GRU(16),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        model.fit(
            X_train_gru, self.y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        self.models['GRU'] = {
            'model': model,
            'predict': lambda X: model.predict(X.reshape(-1, timesteps, features), verbose=0).flatten()
        }
    
    def compare_all(self) -> pd.DataFrame:
        """Compare all models and return results"""
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, brier_score_loss
        
        results = []
        
        for name, model_dict in self.models.items():
            y_pred = model_dict['predict'](self.X_test) if 'predict' in model_dict else model_dict['model'].predict_proba(self.X_test)[:, 1]
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            results.append({
                'Model': name,
                'Accuracy': accuracy_score(self.y_test, y_pred_binary),
                'F1-Score': f1_score(self.y_test, y_pred_binary),
                'AUC-ROC': roc_auc_score(self.y_test, y_pred),
                'Brier Score': brier_score_loss(self.y_test, y_pred)
            })
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def cross_validate(self, n_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """Perform cross-validation for all models"""
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
        
        cv_results = {}
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for name, model_dict in self.models.items():
            auc_scores = []
            acc_scores = []
            f1_scores = []
            
            for train_idx, val_idx in skf.split(self.X_train, self.y_train):
                X_cv_train = self.X_train[train_idx]
                X_cv_val = self.X_train[val_idx]
                y_cv_train = self.y_train[train_idx]
                y_cv_val = self.y_train[val_idx]
                
                # For deep learning models, we need to handle them differently
                if name in ['LSTM', 'GRU']:
                    timesteps = 2
                    features = self.X_train.shape[1] // 2
                    X_cv_train_lstm = X_cv_train.reshape(-1, timesteps, features)
                    X_cv_val_lstm = X_cv_val.reshape(-1, timesteps, features)
                    
                    model = model_dict['model']
                    model.fit(X_cv_train_lstm, y_cv_train, epochs=10, batch_size=32, verbose=0)
                    y_pred = model.predict(X_cv_val_lstm, verbose=0).flatten()
                else:
                    model = model_dict['model']
                    if 'scaler' in model_dict:
                        X_cv_train_scaled = model_dict['scaler'].fit_transform(X_cv_train)
                        X_cv_val_scaled = model_dict['scaler'].transform(X_cv_val)
                        model.fit(X_cv_train_scaled, y_cv_train)
                        y_pred = model.predict_proba(X_cv_val_scaled)[:, 1]
                    else:
                        model.fit(X_cv_train, y_cv_train)
                        y_pred = model.predict_proba(X_cv_val)[:, 1]
                
                y_pred_binary = (y_pred > 0.5).astype(int)
                
                auc_scores.append(roc_auc_score(y_cv_val, y_pred))
                acc_scores.append(accuracy_score(y_cv_val, y_pred_binary))
                f1_scores.append(f1_score(y_cv_val, y_pred_binary))
            
            cv_results[name] = {
                'AUC-ROC': {'mean': np.mean(auc_scores), 'std': np.std(auc_scores)},
                'Accuracy': {'mean': np.mean(acc_scores), 'std': np.std(acc_scores)},
                'F1-Score': {'mean': np.mean(f1_scores), 'std': np.std(f1_scores)}
            }
        
        return cv_results
