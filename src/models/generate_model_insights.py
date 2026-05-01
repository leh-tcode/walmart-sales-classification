import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from metrics_utils import calculate_business_metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Paths (adjust as needed)
MODEL_DIR = Path("src/models/")
DATA_DIR = Path("data/model_ready/")
RESULTS_PATH = DATA_DIR / "model_results.json"

# List your model files and names here
MODEL_FILES = {
    "Random Forest": MODEL_DIR / "rf_model.pkl",
    "XGBoost": MODEL_DIR / "xgboost_model.pkl",
    "Logistic Regression": MODEL_DIR / "logreg_model.pkl",
    # "KNN": MODEL_DIR / "knn_model.pkl",
    "Dummy": MODEL_DIR / "dummy_model.pkl",
    # Add more models as needed
}

# Load test data
test_df = pd.read_csv(DATA_DIR / "test.csv")


features_selected = [
    "Size",
    "Store",
    "Dept",
    "CPI",
    "DeptFrequency",
    "Week_cos",
    "IsPreHoliday",
    "Week_sin",
    "Fuel_Price",
    "ConsumerConfRatio",
    "AvgMarkDownAmount",
]
target = "Sales_Class"
holiday_col = "IsHoliday"


X_test = test_df[features_selected]
y_test = test_df[target]
# If you have a holiday column for business metrics
# Derive holiday flags from the original test dataframe so we detect holiday rows
# even when `IsHoliday` isn't included in `features_selected`.
is_holiday = test_df[holiday_col] if holiday_col in test_df.columns else np.zeros(len(test_df))

results = {"models": {}, "feature_importance": {}}
best_f1 = -1
best_model = None

for name, model_path in MODEL_FILES.items():
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        continue
    with open(model_path, "rb") as f:
        try:
            model = pickle.load(f)
        except ModuleNotFoundError as e:
            missing = getattr(e, "name", None) or str(e)
            print(f"Cannot load {model_path}: missing module {missing}.")
            print(f"Install it with: pip install {missing}")
            continue
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")
            continue

    # Pick prediction input type to avoid sklearn "feature names" warnings and keep column order stable.
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        missing = [c for c in expected if c not in X_test.columns]
        if missing:
            raise ValueError(f"{name}: X_test is missing expected columns: {missing}")
        X_pred = X_test.reindex(columns=expected)
    else:
        # Model was likely fit on an array; pass array to avoid "X has feature names" warnings.
        X_pred = X_test.to_numpy()

    y_pred = model.predict(X_pred)
    y_proba = model.predict_proba(X_pred)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan,
        # "train_time_seconds": ... # Add if available
    }
    # Add business metrics
    business_metrics = calculate_business_metrics(y_test, y_pred, is_holiday)
    metrics.update(business_metrics)

    results["models"][name] = metrics

    # Feature importance per model
    fi = {}
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fi = dict(zip(X_test.columns, importances))
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
        fi = dict(zip(X_test.columns, importances))

    results["feature_importance"][name] = fi

    # Track best model
    if metrics["f1"] > best_f1:
        best_f1 = metrics["f1"]
        best_model = name

results["best_model"] = best_model

# Save to JSON
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2, default=_json_default)

print(f"Model insights saved to {RESULTS_PATH}")
