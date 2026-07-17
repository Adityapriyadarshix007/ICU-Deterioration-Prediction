"""
Improved CNN-BiLSTM + MultiHeadAttention Model
With Focal Loss and Class Weighting Support
Optimized for short sequences (4 timepoints)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Bidirectional, LSTM,
    MultiHeadAttention, LayerNormalization, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling1D, Add
)
from tensorflow.keras.regularizers import l2
import numpy as np

class ImprovedICUModel:
    """
    CNN-BiLSTM with MultiHeadAttention for ICU Deterioration Prediction
    Optimized for short sequences (4 timepoints)
    """
    
    def __init__(self, input_shape, num_heads=4, key_dim=64):
        self.input_shape = input_shape
        self.num_heads = num_heads
        self.key_dim = key_dim
        
    def build_model(self, use_attention=True, dropout_rate=0.3):
        """Build the improved model - optimized for short sequences"""
        
        inputs = Input(shape=self.input_shape, name='input')
        
        # ============================================================
        # CNN Layers - One layer with padding to preserve length
        # ============================================================
        x = Conv1D(filters=32, kernel_size=2, activation='relu', 
                   padding='same', kernel_regularizer=l2(0.001))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Only one MaxPooling to avoid negative dimensions
        x = MaxPooling1D(pool_size=2)(x)
        
        # Second CNN layer
        x = Conv1D(filters=64, kernel_size=2, activation='relu', 
                   padding='same', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # ============================================================
        # BiLSTM - Capture temporal dependencies both directions
        # ============================================================
        x = Bidirectional(LSTM(32, return_sequences=True, 
                               dropout=dropout_rate, 
                               recurrent_dropout=dropout_rate))(x)
        x = LayerNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # ============================================================
        # REAL MultiHeadAttention
        # ============================================================
        if use_attention:
            attention_output = MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.key_dim,
                dropout=dropout_rate
            )(x, x)
            x = Add()([x, attention_output])
            x = LayerNormalization()(x)
        
        # ============================================================
        # Global Pooling
        # ============================================================
        x = GlobalAveragePooling1D()(x)
        
        # ============================================================
        # Dense Layers
        # ============================================================
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Output
        outputs = Dense(1, activation='sigmoid', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def get_class_weights(self, y_train):
        """Calculate class weights for imbalance"""
        total = len(y_train)
        n_positive = np.sum(y_train)
        n_negative = total - n_positive
        
        if n_positive == 0:
            return {0: 1.0, 1: 1.0}
        
        weight_positive = total / (2 * n_positive)
        weight_negative = total / (2 * n_negative)
        
        return {0: weight_negative, 1: weight_positive}
