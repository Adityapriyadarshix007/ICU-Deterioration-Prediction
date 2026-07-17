"""
Multi-Output Model for Organ-Specific Deterioration Prediction
Predicts: Risk Score + Organ Failure + Time-to-Deterioration
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class MultiOutputICUModel:
    """Multi-task learning model for ICU deterioration"""
    
    def __init__(self, input_shape, num_organs=6):
        self.input_shape = input_shape
        self.num_organs = num_organs  # respiratory, cardiovascular, renal, liver, coagulation, neurological
        
    def build_model(self):
        """Build multi-output model"""
        
        # Input
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # Shared layers
        x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)
        
        # Bi-LSTM
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.Dropout(0.3)(x)
        
        # Multi-head Attention
        attention = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = layers.Add()([x, attention])
        x = layers.GlobalAveragePooling1D()(x)
        
        # Shared representation
        shared = layers.Dense(128, activation='relu')(x)
        shared = layers.BatchNormalization()(shared)
        shared = layers.Dropout(0.3)(shared)
        
        # Output 1: Overall Risk Score
        risk_output = layers.Dense(64, activation='relu')(shared)
        risk_output = layers.Dropout(0.2)(risk_output)
        risk_output = layers.Dense(1, activation='sigmoid', name='risk')(risk_output)
        
        # Output 2: Organ-Specific Risks
        organ_outputs = []
        organ_names = ['Respiratory', 'Cardiovascular', 'Renal', 'Liver', 'Coagulation', 'Neurological']
        
        for i, organ_name in enumerate(organ_names):
            organ = layers.Dense(32, activation='relu')(shared)
            organ = layers.Dropout(0.2)(organ)
            organ = layers.Dense(1, activation='sigmoid', name=f'organ_{i}')(organ)
            organ_outputs.append(organ)
        
        # Output 3: Time-to-Deterioration (hours)
        time_output = layers.Dense(32, activation='relu')(shared)
        time_output = layers.Dropout(0.2)(time_output)
        time_output = layers.Dense(1, activation='relu', name='time_to_event')(time_output)
        
        # Output 4: Prediction Confidence
        confidence_output = layers.Dense(32, activation='relu')(shared)
        confidence_output = layers.Dropout(0.2)(confidence_output)
        confidence_output = layers.Dense(1, activation='sigmoid', name='confidence')(confidence_output)
        
        # Create model
        model = Model(
            inputs=inputs,
            outputs=[risk_output] + organ_outputs + [time_output, confidence_output],
            name='multi_output_icu_model'
        )
        
        return model
    
    def compile_model(self, model):
        """Compile with appropriate losses for each output"""
        
        # Define losses for each output
        losses = {
            'risk': 'binary_crossentropy',
            **{f'organ_{i}': 'binary_crossentropy' for i in range(self.num_organs)},
            'time_to_event': 'mse',
            'confidence': 'binary_crossentropy'
        }
        
        # Define loss weights
        loss_weights = {
            'risk': 1.0,
            **{f'organ_{i}': 0.5 for i in range(self.num_organs)},
            'time_to_event': 0.3,
            'confidence': 0.2
        }
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=losses,
            loss_weights=loss_weights,
            metrics={
                'risk': ['accuracy'],
                **{f'organ_{i}': ['accuracy'] for i in range(self.num_organs)}
            }
        )
        
        return model
