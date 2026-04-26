import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

from src.features.feature_engineering import FEATURES_PATH
from src.features.feature_engineering import FEATURES_PATH
from src.utils.logger import logger

# Paths
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("data/model_ready")
EDA_DIR = Path("data/eda_ready")
ARTIFACTS_DIR = Path("data/artifacts")

EDA_PATH = EDA_DIR / "eda_dataset.csv"
TRAIN_PATH = MODEL_DIR / "train.csv"
TEST_PATH = MODEL_DIR / "test.csv"
SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
FEATURE_META_PATH = ARTIFACTS_DIR / "feature_metadata.json"
REPORT_JSON_PATH = PROCESSED_DIR / "preprocessing_report.json"
REPORT_TEXT_PATH = PROCESSED_DIR / "preprocessing_report.txt"

# Constants
TARGET = "Sales_Class"

DROP_FOR_MODEL = [
    "Date",
    "Weekly_Sales",
]

EDA_ONLY_COLS = [
    "Date",
    "Weekly_Sales",
]

CATEGORICAL_COLS = ["Type"]

BOOLEAN_COLS = ["IsHoliday"]

ORDINAL_MAPS = {
    "Type": {"A": 2, "B": 1, "C": 0},
}

NO_SCALE_COLS = [
    TARGET,
    "Store", "Dept",
    "IsHoliday",
    "TypeEncoded",
    "SizeQuartile",
    "HolidayType",
    "IsPreHoliday", "IsPostHoliday",
    "IsPeakSeason", "IsBackToSchool",
    "IsMonthStart", "IsMonthEnd",
    "IsYearStart", "IsYearEnd",
    "HasAnyMarkDown",
    "has_MarkDown1", "has_MarkDown2", "has_MarkDown3",
    "has_MarkDown4", "has_MarkDown5",
    "is_return",
    "Promo_Holiday",
    "Holiday_Type",
    "ActiveMarkDownCount",
    "Year",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42
STRATIFY_COL = TARGET

SCALER_TYPE = "robust"


# Helpers
def _shape_str(df: pd.DataFrame) -> str:
    return f"{len(df):,} rows × {len(df.columns)} cols"


def _pct(count: int, total: int) -> float:
    return round(count / total * 100, 3) if total else 0.0


# STEP 1: PREPARE EDA DATASET
def prepare_eda_dataset(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    logger.info("Step 1: Preparing EDA dataset …")
    eda = df.copy()

    eda = eda.sort_values("Date").reset_index(drop=True)

    if "Date" in eda.columns:
        eda["YearMonth"] = eda["Date"].dt.to_period("M").astype(str)
        eda["YearWeek"] = (
            eda["Date"].dt.isocalendar().year.astype(str)
            + "-W"
            + eda["Date"].dt.isocalendar().week.astype(str).str.zfill(2)
        )

    eda["Sales_Label"] = eda[TARGET].map({0: "Low", 1: "High"})

    if "Type" in eda.columns:
        eda["StoreTypeLabel"] = eda["Type"].map({
            "A": "Type A (Large)",
            "B": "Type B (Medium)",
            "C": "Type C (Small)",
        })

    if "Weekly_Sales" in eda.columns:
        eda["SalesBucket"] = pd.cut(
            eda["Weekly_Sales"],
            bins=[-np.inf, 0, 5000, 15000, 30000, 50000, np.inf],
            labels=["Negative/Return", "Very Low", "Low", "Medium", "High", "Very High"],
        )

    if "HolidayType" in eda.columns:
        eda["HolidayName"] = eda["HolidayType"].map({
            0: "None",
            1: "Super Bowl",
            2: "Labor Day",
            3: "Thanksgiving",
            4: "Christmas",
        })

    step_report = {
        "step": "Prepare EDA Dataset",
        "shape": _shape_str(eda),
        "columns": len(eda.columns),
        "helper_columns_added": [
            "YearMonth", "YearWeek", "Sales_Label",
            "StoreTypeLabel", "SalesBucket", "HolidayName",
        ],
        "date_range": {
            "min": str(eda["Date"].min().date()) if "Date" in eda.columns else "N/A",
            "max": str(eda["Date"].max().date()) if "Date" in eda.columns else "N/A",
        },
        "note": "Unscaled, unencoded, all columns retained for visualization",
    }

    report["steps"].append(step_report)
    logger.info("  EDA dataset: {} | +6 helper columns", _shape_str(eda))
    return eda


# STEP 2: ENCODE CATEGORICALS
def encode_categoricals(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    logger.info("Step 2: Encoding categoricals …")

    encoded_cols = []

    for col in BOOLEAN_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)
            encoded_cols.append({"column": col, "method": "bool_to_int"})

    cats_to_drop = []
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            if f"{col}Encoded" in df.columns or "TypeEncoded" in df.columns:
                cats_to_drop.append(col)
                encoded_cols.append({
                    "column": col,
                    "method": "dropped (ordinal version exists)",
                })

    df = df.drop(columns=cats_to_drop, errors="ignore")

    step_report = {
        "step": "Encode Categoricals",
        "encoded": encoded_cols,
        "dropped_originals": cats_to_drop,
        "remaining_object_cols": list(df.select_dtypes(include=["object"]).columns),
    }

    report["steps"].append(step_report)
    logger.info("  Encoded: {} columns | Dropped originals: {}", len(encoded_cols), cats_to_drop)
    return df


# STEP 3: DROP LEAKAGE / REDUNDANT COLUMNS
def drop_leakage_columns(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    logger.info("Step 3: Dropping leakage/redundant columns …")

    cols_before = list(df.columns)
    to_drop = [c for c in DROP_FOR_MODEL if c in df.columns]

    remaining_objects = list(df.select_dtypes(include=["object", "category"]).columns)
    to_drop.extend([c for c in remaining_objects if c not in to_drop])

    to_drop = sorted(set(to_drop))
    df = df.drop(columns=to_drop, errors="ignore")

    step_report = {
        "step": "Drop Leakage & Redundant Columns",
        "dropped": to_drop,
        "reason": {
            "Weekly_Sales": "Direct basis of target — leakage",
            "Date": "Decomposed into 12+ temporal features",
        },
        "columns_before": len(cols_before),
        "columns_after": len(df.columns),
    }

    report["steps"].append(step_report)
    logger.info("  Dropped {} columns: {}", len(to_drop), to_drop)
    return df


# STEP 4: TRAIN/TEST SPLIT
def split_data(
    df: pd.DataFrame,
    report: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    logger.info("Step 4: Splitting into train/test …")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_dist = y_train.value_counts(normalize=True).round(4).to_dict()
    test_dist = y_test.value_counts(normalize=True).round(4).to_dict()

    step_report = {
        "step": "Train/Test Split",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratified_by": STRATIFY_COL,
        "train_shape": _shape_str(X_train),
        "test_shape": _shape_str(X_test),
        "train_target_distribution": train_dist,
        "test_target_distribution": test_dist,
        "split_method": "Stratified random (sklearn train_test_split)",
    }

    report["steps"].append(step_report)
    logger.info(
        "  Train: {} | Test: {} | Stratified on {}",
        _shape_str(X_train), _shape_str(X_test), STRATIFY_COL,
    )
    return X_train, X_test, y_train, y_test


# STEP 5: SCALE NUMERIC FEATURES
def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    report: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    logger.info("Step 5: Scaling numeric features …")

    all_cols = list(X_train.columns)
    no_scale = set(NO_SCALE_COLS) & set(all_cols)
    scale_cols = [
        c for c in all_cols
        if c not in no_scale
        and pd.api.types.is_numeric_dtype(X_train[c])
    ]

    logger.info("  Scaling {} of {} columns", len(scale_cols), len(all_cols))
    logger.info("  Not scaling {} columns (binary/ordinal/ID)", len(no_scale))

    if SCALER_TYPE == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    if scale_cols:
        X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
        X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

    train_nulls = int(X_train_scaled.isna().sum().sum())
    test_nulls = int(X_test_scaled.isna().sum().sum())
    train_inf = int(np.isinf(X_train_scaled.select_dtypes(include=[np.number])).sum().sum())
    test_inf = int(np.isinf(X_test_scaled.select_dtypes(include=[np.number])).sum().sum())

    step_report = {
        "step": "Scale Numeric Features",
        "scaler_type": SCALER_TYPE,
        "scaler_class": type(scaler).__name__,
        "scaled_columns": scale_cols,
        "scaled_column_count": len(scale_cols),
        "unscaled_columns": sorted(no_scale & set(all_cols)),
        "unscaled_column_count": len(no_scale & set(all_cols)),
        "fit_on": "train only",
        "post_scale_nulls_train": train_nulls,
        "post_scale_nulls_test": test_nulls,
        "post_scale_inf_train": train_inf,
        "post_scale_inf_test": test_inf,
    }

    report["steps"].append(step_report)
    logger.info(
        "  {} scaler fit on train ({} cols) | Post-scale nulls: train={}, test={}",
        type(scaler).__name__, len(scale_cols), train_nulls, test_nulls,
    )
    return X_train_scaled, X_test_scaled, scaler


# STEP 6: SAVE FEATURE METADATA
def save_feature_metadata(
    X_train: pd.DataFrame,
    scaler: Any,
    scale_cols: list[str],
    report: dict,
) -> dict:
    logger.info("Step 6: Saving feature metadata …")

    all_features = list(X_train.columns)
    numeric_features = list(X_train.select_dtypes(include=[np.number]).columns)
    binary_features = [
        c for c in numeric_features
        if set(X_train[c].unique()) <= {0, 1, 0.0, 1.0}
    ]
    continuous_features = [c for c in numeric_features if c not in binary_features]

    metadata = {
        "target": TARGET,
        "feature_count": len(all_features),
        "feature_names": all_features,
        "feature_dtypes": {col: str(dtype) for col, dtype in X_train.dtypes.items()},
        "feature_groups": {
            "numeric": numeric_features,
            "binary": binary_features,
            "continuous": continuous_features,
            "scaled": scale_cols,
            "unscaled": [c for c in all_features if c not in scale_cols],
        },
        "preprocessing_params": {
            "scaler_type": SCALER_TYPE,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratify_col": STRATIFY_COL,
            "drop_for_model": DROP_FOR_MODEL,
        },
        "shapes": {
            "train_rows": len(X_train),
            "test_rows": int(len(X_train) * TEST_SIZE / (1 - TEST_SIZE)),
            "features": len(all_features),
        },
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_META_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    report["steps"].append({
        "step": "Save Feature Metadata",
        "path": str(FEATURE_META_PATH),
        "feature_count": len(all_features),
        "binary_features": len(binary_features),
        "continuous_features": len(continuous_features),
    })

    logger.info(
        "  Metadata saved: {} features ({} continuous, {} binary)",
        len(all_features), len(continuous_features), len(binary_features),
    )
    return metadata

# REPORT GENERATION
def _generate_summary(
    df_input: pd.DataFrame,
    eda_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    report: dict,
) -> dict:

    report["summary"] = {
        "input": {
            "shape": _shape_str(df_input),
            "columns": len(df_input.columns),
        },
        "eda_output": {
            "shape": _shape_str(eda_df),
            "columns": len(eda_df.columns),
            "path": str(EDA_PATH),
        },
        "model_output": {
            "train_shape": _shape_str(X_train),
            "test_shape": _shape_str(X_test),
            "features": len(X_train.columns),
            "train_path": str(TRAIN_PATH),
            "test_path": str(TEST_PATH),
        },
        "artifacts": {
            "scaler": str(SCALER_PATH),
            "feature_metadata": str(FEATURE_META_PATH),
        },
    }
    return report

def _save_json_report(report: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Preprocessing JSON report saved to: {}", REPORT_JSON_PATH)

def _save_text_report(report: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 70,
        "  WALMART SALES CLASSIFICATION — PREPROCESSING REPORT",
        "=" * 70,
        "",
    ]

    s = report.get("summary", {})
    lines.extend([
        "SUMMARY",
        "-" * 70,
        f"  Input:       {s.get('input', {}).get('shape', '?')}",
        f"  EDA Output:  {s.get('eda_output', {}).get('shape', '?')}",
        f"  Train:       {s.get('model_output', {}).get('train_shape', '?')}",
        f"  Test:        {s.get('model_output', {}).get('test_shape', '?')}",
        f"  Features:    {s.get('model_output', {}).get('features', '?')}",
        "",
    ])

    for step in report.get("steps", []):
        lines.append(f"{'─' * 70}")
        lines.append(f"  {step.get('step', 'Unknown')}")
        lines.append(f"{'─' * 70}")
        for k, v in step.items():
            if k != "step":
                lines.append(f"    {k}: {v}")
        lines.append("")

    lines.append("=" * 70)

    with open(REPORT_TEXT_PATH, "w") as f:
        f.write("\n".join(lines))
    logger.info("Preprocessing text report saved to: {}", REPORT_TEXT_PATH)


# ORCHESTRATOR
def run_preprocessing(df: pd.DataFrame) -> dict[str, Any]:
    logger.info("=" * 60)
    logger.info("Starting Preprocessing Pipeline")
    logger.info("=" * 60)
    logger.info("Input shape: {}", _shape_str(df))

    report: dict[str, Any] = {"steps": []}

    eda_df = prepare_eda_dataset(df, report)

    model_df = df.copy()
    model_df = encode_categoricals(model_df, report)
    model_df = drop_leakage_columns(model_df, report)

    X_train, X_test, y_train, y_test = split_data(model_df, report)

    all_cols = list(X_train.columns)
    no_scale = set(NO_SCALE_COLS) & set(all_cols)
    scale_cols = [
        c for c in all_cols
        if c not in no_scale
        and pd.api.types.is_numeric_dtype(X_train[c])
    ]

    X_train, X_test, scaler = scale_features(X_train, X_test, report)

    feature_meta = save_feature_metadata(X_train, scaler, scale_cols, report)


    report = _generate_summary(df, eda_df, X_train, X_test, report)

    for d in [EDA_DIR, MODEL_DIR, ARTIFACTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    eda_df.to_csv(EDA_PATH, index=False)
    logger.info("EDA dataset saved to: {}", EDA_PATH)

    train_out = X_train.copy()
    train_out[TARGET] = y_train.values
    test_out = X_test.copy()
    test_out[TARGET] = y_test.values
    train_out.to_csv(TRAIN_PATH, index=False)
    test_out.to_csv(TEST_PATH, index=False)
    logger.info("Train saved to: {} ({})", TRAIN_PATH, _shape_str(train_out))
    logger.info("Test saved to: {} ({})", TEST_PATH, _shape_str(test_out))

    joblib.dump(scaler, SCALER_PATH)
    logger.info("Scaler saved to: {}", SCALER_PATH)

    _save_json_report(report)
    _save_text_report(report)

    logger.info("")
    logger.info("=" * 60)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info("  Input:       {}", _shape_str(df))
    logger.info("  EDA Output:  {} → {}", _shape_str(eda_df), EDA_PATH)
    logger.info("  Train:       {} → {}", _shape_str(X_train), TRAIN_PATH)
    logger.info("  Test:        {} → {}", _shape_str(X_test), TEST_PATH)
    logger.info("  Features:    {}", len(X_train.columns))
    logger.info("  Scaler:      {} → {}", type(scaler).__name__, SCALER_PATH)
    logger.info("=" * 60)

    return {
        "eda_df": eda_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_meta": feature_meta,
        "report": report,
    }

if __name__ == "__main__":
    
    featured_df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    outputs = run_preprocessing(featured_df)
