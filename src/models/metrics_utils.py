import numpy as np
from sklearn.metrics import accuracy_score
import joblib
import os
import mlflow
from sklearn.ensemble import RandomForestClassifier


def calculate_business_metrics(y_true, y_pred, is_holiday_col):
    """
    Calculates business-specific metrics for Walmart sales.
    
    Metrics:
    1. Holiday Accuracy: Accuracy specifically during holiday weeks.
    2. Weighted Classification Error (WCE): Penalizes holiday misses 5x more than regular weeks.
       Formula: (5 * holiday_error + 1 * regular_error) / 6
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    is_holiday = np.array(is_holiday_col)
    
    holiday_mask = (is_holiday == 1)
    if np.sum(holiday_mask) > 0:
        holiday_acc = accuracy_score(y_true[holiday_mask], y_pred[holiday_mask])
    else:
        holiday_acc = 0.0
        
    regular_mask = (is_holiday == 0)
    if np.sum(regular_mask) > 0:
        regular_acc = accuracy_score(y_true[regular_mask], y_pred[regular_mask])
    else:
        regular_acc = 0.0
        
    # WCE
    holiday_error = 1.0 - holiday_acc
    regular_error = 1.0 - regular_acc
    
    # We use a weighted average to keep the metric between 0 and 1
    wce = (5 * holiday_error + 1 * regular_error) / 6
    
    return {
        "holiday_accuracy": holiday_acc,
        "weighted_classification_error": wce,
        "regular_accuracy": regular_acc
    }

def load_or_train_model(X, y, params, MODEL_PATH, model):
    """Checks for existing model, otherwise trains a new one."""
    if os.path.exists(MODEL_PATH):
        print(f"--- Found existing model at {MODEL_PATH}. Loading... ---")
        model = joblib.load(MODEL_PATH)
    else:
        print("--- No local model found. Training from scratch... ---")
        model = model(**params)
        model.fit(X, y)
        
        # Save to local file
        joblib.dump(model, MODEL_PATH)
        print(f"--- Model saved to {MODEL_PATH} ---")
        
    return model