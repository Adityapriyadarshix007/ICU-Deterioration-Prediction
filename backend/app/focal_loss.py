"""
Focal Loss for Imbalanced Classification
Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
"""

import tensorflow as tf
from tensorflow.keras import backend as K

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for binary classification
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Balancing parameter (default: 0.25)
    """
    def focal_loss_fixed(y_true, y_pred):
        # Clip predictions to prevent log(0)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        loss = -alpha_t * K.pow(1 - p_t, gamma) * K.log(p_t)
        return K.mean(loss)
    
    return focal_loss_fixed


def weighted_binary_crossentropy(class_weight):
    """
    Weighted Binary Crossentropy for imbalanced data
    
    Args:
        class_weight: Weight for positive class (deteriorated)
    """
    def loss(y_true, y_pred):
        # Clip predictions
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        
        # Weighted loss
        loss = - (class_weight * y_true * K.log(y_pred) + 
                  (1 - y_true) * K.log(1 - y_pred))
        return K.mean(loss)
    
    return loss
