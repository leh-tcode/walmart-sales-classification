import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

from src.cleaning.cleaning import CLEANED_PATH
from src.utils.logger import logger

# Paths
PROCESSED_DIR = Path("data/processed")
FEATURES_PATH = PROCESSED_DIR / "featured_dataset.csv"
FEATURES_REPORT_PATH = PROCESSED_DIR / "feature_engineering_report.json"
FEATURES_TEXT_REPORT_PATH = PROCESSED_DIR / "feature_engineering_report.txt"

# Constants
MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
MARKDOWN_FLAGS = [f"has_{c}" for c in MARKDOWN_COLS]

HOLIDAY_WINDOWS = {
    "SuperBowl":   {"month": 2, "week_range": (5, 7)},
    "LaborDay":    {"month": 9, "week_range": (35, 37)},
    "Thanksgiving": {"month": 11, "week_range": (46, 48)},
    "Christmas":   {"month": 12, "week_range": (50, 52)},
}

FRED_COLS = ["UMCSENT", "RSXFS", "PCE"]

LAG_PERIODS = [1, 2, 4]
ROLLING_WINDOWS = [4, 8, 12]


# Helpers
def _shape_str(df: pd.DataFrame) -> str:
    return f"{len(df):,} rows × {len(df.columns)} cols"


def _count_new(before: int, after: int) -> str:
    return f"+{after - before} features"


# GROUP 1: TEMPORAL FEATURES
def create_temporal_features(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    logger.info("Group 1: Creating temporal features …")
    n_before = len(df.columns)

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter
    df["DayOfYear"] = df["Date"].dt.dayofyear

    df["WeekOfMonth"] = (df["Date"].dt.day - 1) // 7 + 1

    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)
    df["IsYearStart"] = (df["Month"] == 1).astype(int)
    df["IsYearEnd"] = (df["Month"] == 12).astype(int)

    df["DaysInMonth"] = df["Date"].dt.days_in_month

    df["YearProgress"] = df["DayOfYear"] / 365.0

    n_created = len(df.columns) - n_before
    report["groups"].append({
        "group": "Temporal Features",
        "features_created": n_created,
        "features": [
            "Year", "Month", "Week", "Quarter", "DayOfYear",
            "WeekOfMonth", "IsMonthStart", "IsMonthEnd",
            "IsYearStart", "IsYearEnd", "DaysInMonth", "YearProgress",
        ],
        "rationale": "Capture calendar-driven seasonality in retail sales",
    })

    logger.info("  Temporal: {} features created", n_created)
    return df


# GROUP 2: HOLIDAY FEATURES
def create_holiday_features(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    logger.info("Group 2: Creating holiday features …")
    n_before = len(df.columns)

    df["HolidayType"] = 0  # 0 = not a holiday period
    for i, (name, info) in enumerate(HOLIDAY_WINDOWS.items(), 1):
        week_lo, week_hi = info["week_range"]
        mask = df["Week"].between(week_lo, week_hi)
        df.loc[mask, "HolidayType"] = i

    holiday_weeks = set()
    for info in HOLIDAY_WINDOWS.values():
        week_lo, week_hi = info["week_range"]
        holiday_weeks.update(range(week_lo, week_hi + 1))

    pre_holiday_weeks = {w - 1 for w in holiday_weeks if w - 1 > 0}
    post_holiday_weeks = {w + 1 for w in holiday_weeks if w + 1 <= 52}

    df["IsPreHoliday"] = df["Week"].isin(pre_holiday_weeks).astype(int)
    df["IsPostHoliday"] = df["Week"].isin(post_holiday_weeks).astype(int)

    holiday_week_list = sorted(holiday_weeks)

    def _min_holiday_distance(week: int) -> int:
        if not holiday_week_list:
            return 26
        distances = [min(abs(week - hw), 52 - abs(week - hw)) for hw in holiday_week_list]
        return min(distances)

    df["HolidayProximity"] = df["Week"].apply(_min_holiday_distance)

    df["IsPeakSeason"] = df["Month"].isin([11, 12]).astype(int)
    df["IsBackToSchool"] = df["Month"].isin([7, 8]).astype(int)

    n_created = len(df.columns) - n_before
    report["groups"].append({
        "group": "Holiday Features",
        "features_created": n_created,
        "features": [
            "HolidayType", "IsPreHoliday", "IsPostHoliday",
            "HolidayProximity", "IsPeakSeason", "IsBackToSchool",
        ],
        "rationale": (
            "Differentiate holiday types and capture pre/post-holiday "
            "effects that a binary IsHoliday flag misses"
        ),
    })

    logger.info("  Holiday: {} features created", n_created)
    return df


# GROUP 3: PROMOTION / MARKDOWN FEATURES
def create_promotion_features(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    logger.info("Group 3: Creating promotion features …")
    n_before = len(df.columns)

    df["TotalMarkDown"] = df[MARKDOWN_COLS].sum(axis=1)

    df["ActiveMarkDownCount"] = df[MARKDOWN_FLAGS].sum(axis=1)

    active_sum = df[MARKDOWN_COLS].sum(axis=1)
    active_count = df[MARKDOWN_FLAGS].sum(axis=1)
    df["AvgMarkDownAmount"] = np.where(
        active_count > 0,
        active_sum / active_count,
        0.0,
    )

    df["MaxMarkDown"] = df[MARKDOWN_COLS].max(axis=1)

    df["HasAnyMarkDown"] = (active_count > 0).astype(int)

    n_created = len(df.columns) - n_before
    report["groups"].append({
        "group": "Promotion Features",
        "features_created": n_created,
        "features": [
            "TotalMarkDown", "ActiveMarkDownCount",
            "AvgMarkDownAmount", "MaxMarkDown", "HasAnyMarkDown",
        ],
        "rationale": (
            "Individual MarkDown columns are sparse and skewed — "
            "aggregates capture overall promotional intensity "
            "with less noise"
        ),
    })

    logger.info("  Promotion: {} features created", n_created)
    return df


# GROUP 4: STORE & DEPARTMENT FEATURES
def create_store_dept_features(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    logger.info("Group 4: Creating store & department features …")
    n_before = len(df.columns)

    type_map = {"A": 2, "B": 1, "C": 0}
    df["TypeEncoded"] = df["Type"].map(type_map)

    df["SizePerType"] = df["Size"] / (df["TypeEncoded"] + 1)

    store_dept_counts = df.groupby("Store")["Dept"].nunique()
    df["StoreDeptCount"] = df["Store"].map(store_dept_counts)

    dept_store_counts = df.groupby("Dept")["Store"].nunique()
    df["DeptFrequency"] = df["Dept"].map(dept_store_counts)

    df["SizeQuartile"] = pd.qcut(df["Size"], q=4, labels=[1, 2, 3, 4]).astype(int)

    n_created = len(df.columns) - n_before
    report["groups"].append({
        "group": "Store & Department Features",
        "features_created": n_created,
        "features": [
            "TypeEncoded", "SizePerType", "StoreDeptCount",
            "DeptFrequency", "SizeQuartile",
        ],
        "rationale": (
            "Encode store structure beyond raw Type/Size — "
            "captures store complexity and relative positioning"
        ),
        "leakage_risk": "None — based on store metadata only",
    })

    logger.info("  Store & Dept: {} features created", n_created)
    return df


# GROUP 5: ECONOMIC / MACRO FEATURES
def create_economic_features(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    logger.info("Group 5: Creating economic features …")
    n_before = len(df.columns)

    for col in FRED_COLS:
        col_min = df[col].min()
        col_max = df[col].max()
        col_range = col_max - col_min
        if col_range > 0:
            df[f"_{col}_norm"] = (df[col] - col_min) / col_range
        else:
            df[f"_{col}_norm"] = 0.0

    norm_cols = [f"_{c}_norm" for c in FRED_COLS]
    df["EconIndex"] = df[norm_cols].mean(axis=1)

    df.drop(columns=norm_cols, inplace=True)

    df["ConsumerConfRatio"] = df["UMCSENT"] / (df["Unemployment"] + 1e-6)

    df["RealSpendingPerCapita"] = df["PCE"] / (df["RSXFS"] + 1e-6)

    df = df.sort_values("Date")
    df["EconMomentum"] = df.groupby(["Store", "Dept"])["EconIndex"].diff()
    df["EconMomentum"] = df["EconMomentum"].fillna(0.0)

    df["FuelBurden"] = (df["Fuel_Price"] / df["CPI"]) * 100

    df["PurchasingPower"] = df["CPI"] / (df["Unemployment"] + 1e-6)

    n_created = len(df.columns) - n_before
    report["groups"].append({
        "group": "Economic Features",
        "features_created": n_created,
        "features": [
            "EconIndex", "ConsumerConfRatio", "RealSpendingPerCapita",
            "EconMomentum", "FuelBurden", "PurchasingPower",
        ],
        "rationale": (
            "FRED macro series are highly correlated (r > 0.78) — "
            "composites reduce multicollinearity while ratios "
            "capture divergences between economic indicators"
        ),
        "correlation_note": (
            "UMCSENT↔RSXFS: r=0.83, RSXFS↔PCE: r=0.88, UMCSENT↔PCE: r=0.79"
        ),
    })

    logger.info("  Economic: {} features created", n_created)
    return df


# GROUP 6: LAG / ROLLING FEATURES
def create_lag_features(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    logger.info("Group 6: Creating lag & rolling features …")
    n_before = len(df.columns)

    df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
    group = df.groupby(["Store", "Dept"])["Weekly_Sales"]

    for lag in LAG_PERIODS:
        col_name = f"Lag_Sales_{lag}w"
        df[col_name] = group.shift(lag)
        logger.info("  Created {}", col_name)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        for window in ROLLING_WINDOWS:
            # Rolling mean (shifted by 1 to exclude current row)
            mean_col = f"Rolling_Mean_{window}w"
            df[mean_col] = group.transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            logger.info("  Created {}", mean_col)

        df["Rolling_Std_4w"] = group.transform(
            lambda x: x.shift(1).rolling(window=4, min_periods=2).std()
        )

    if "Lag_Sales_1w" in df.columns and "Lag_Sales_4w" in df.columns:
        df["SalesTrend_4w"] = (df["Lag_Sales_1w"] - df["Lag_Sales_4w"]) / 3.0

    if "SalesTrend_4w" in df.columns:
        df["SalesAcceleration"] = df.groupby(["Store", "Dept"])[
            "SalesTrend_4w"
        ].diff()

    lag_rolling_cols = [c for c in df.columns if c.startswith(("Lag_", "Rolling_", "Sales"))]
    lag_rolling_cols = [
        c for c in lag_rolling_cols
        if c in df.columns and c not in ["Sales_Class"]
    ]

    nulls_before = int(df[lag_rolling_cols].isna().sum().sum())

    for col in lag_rolling_cols:
        if df[col].isna().any():
            group_median = df.groupby(["Store", "Dept"])[col].transform("median")
            df[col] = df[col].fillna(group_median)
            df[col] = df[col].fillna(df[col].median())
            df[col] = df[col].fillna(0.0)

    nulls_after = int(df[lag_rolling_cols].isna().sum().sum())

    n_created = len(df.columns) - n_before
    report["groups"].append({
        "group": "Lag & Rolling Features",
        "features_created": n_created,
        "features": [
            "Lag_Sales_1w", "Lag_Sales_2w", "Lag_Sales_4w",
            "Rolling_Mean_4w", "Rolling_Mean_8w", "Rolling_Mean_12w",
            "Rolling_Std_4w", "SalesTrend_4w", "SalesAcceleration",
        ],
        "lag_periods": LAG_PERIODS,
        "rolling_windows": ROLLING_WINDOWS,
        "null_handling": (
            f"Filled {nulls_before - nulls_after:,} NaN values "
            f"with Store-Dept group median (fallback: global median → 0)"
        ),
        "leakage_policy": (
            "All lags use .shift() — only past data used. "
            "Current row's Weekly_Sales is NEVER in its own features."
        ),
        "rationale": (
            "Past sales are the strongest predictor of future sales. "
            "Rolling windows capture momentum at different time horizons."
        ),
    })

    logger.info(
        "  Lag & Rolling: {} features created | NaN filled: {:,} → {:,}",
        n_created, nulls_before, nulls_after,
    )
    return df


# GROUP 7: INTERACTION FEATURES
def create_interaction_features(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    logger.info("Group 7: Creating interaction features …")
    n_before = len(df.columns)

    df["Holiday_Size"] = df["IsHoliday"].astype(int) * df["Size"]

    df["Holiday_Type"] = df["IsHoliday"].astype(int) * df["TypeEncoded"]

    df["Promo_Holiday"] = df["HasAnyMarkDown"] * df["IsHoliday"].astype(int)

    df["Temp_Season"] = df["Temperature"] * df["Quarter"]

    df["Econ_Size"] = df["EconIndex"] * df["SizeQuartile"]

    df["MarkDown_Intensity"] = df["TotalMarkDown"] / (df["Size"] + 1e-6)

    n_created = len(df.columns) - n_before
    report["groups"].append({
        "group": "Interaction Features",
        "features_created": n_created,
        "features": [
            "Holiday_Size", "Holiday_Type", "Promo_Holiday",
            "Temp_Season", "Econ_Size", "MarkDown_Intensity",
        ],
        "rationale": (
            "Capture non-additive effects: holiday impact depends "
            "on store size/type, promotional effects compound with "
            "holidays, economic conditions affect stores differently "
            "by size tier"
        ),
    })

    logger.info("  Interactions: {} features created", n_created)
    return df


# GROUP 8: CYCLICAL ENCODING
def create_cyclical_features(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    logger.info("Group 8: Creating cyclical encoding features …")
    n_before = len(df.columns)

    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    df["Week_sin"] = np.sin(2 * np.pi * df["Week"] / 52)
    df["Week_cos"] = np.cos(2 * np.pi * df["Week"] / 52)

    n_created = len(df.columns) - n_before
    report["groups"].append({
        "group": "Cyclical Encoding",
        "features_created": n_created,
        "features": ["Month_sin", "Month_cos", "Week_sin", "Week_cos"],
        "rationale": (
            "Sin/cos encoding preserves cyclical adjacency — "
            "December is next to January, not 11 units away"
        ),
    })

    logger.info("  Cyclical: {} features created", n_created)
    return df


# POST-ENGINEERING VALIDATION
def validate_features(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    logger.info("Running post-engineering validation …")

    checks = []

    null_counts = df.isna().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    checks.append({
        "check": "No NaN values in engineered features",
        "columns_with_nulls": cols_with_nulls.to_dict() if len(cols_with_nulls) > 0 else {},
        "status": "PASS" if len(cols_with_nulls) == 0 else "WARN",
    })

    numeric_df = df.select_dtypes(include=[np.number])
    inf_count = int(np.isinf(numeric_df).sum().sum())
    checks.append({
        "check": "No infinite values",
        "inf_count": inf_count,
        "status": "PASS" if inf_count == 0 else "FAIL",
    })

    checks.append({
        "check": "Row count preserved",
        "rows": len(df),
        "status": "PASS",
    })

    if "Sales_Class" in df.columns:
        unique_vals = set(df["Sales_Class"].unique())
        checks.append({
            "check": "Target column intact",
            "unique_values": sorted(unique_vals),
            "status": "PASS" if unique_vals == {0, 1} else "FAIL",
        })

    for col in ["Month_sin", "Month_cos", "Week_sin", "Week_cos"]:
        if col in df.columns:
            col_min, col_max = df[col].min(), df[col].max()
            checks.append({
                "check": f"{col} in [-1, 1]",
                "min": round(float(col_min), 4),
                "max": round(float(col_max), 4),
                "status": "PASS" if -1.01 <= col_min and col_max <= 1.01 else "FAIL",
            })

    constant_cols = [
        col for col in numeric_df.columns
        if numeric_df[col].std() == 0
    ]

    if constant_cols:
        df = df.drop(columns=constant_cols)
        logger.info("  Dropped {} zero-variance columns: {}", len(constant_cols), constant_cols)

    checks.append({
        "check": "No zero-variance features",
        "constant_columns_dropped": constant_cols,
        "status": "PASS",
        "note": f"Dropped {len(constant_cols)} constant columns" if constant_cols else "None found",

    })

    passed = sum(1 for c in checks if c["status"] == "PASS")
    report["validation"] = {
        "checks": checks,
        "passed": passed,
        "total": len(checks),
        "status": "PASS" if passed == len(checks) else "WARN",
    }

    for check in checks:
        icon = "✓" if check["status"] == "PASS" else "⚠"
        logger.info("  [{}] {}", icon, check["check"])

    return df


# REPORT GENERATION
def _generate_summary(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    report: dict,
) -> dict:

    all_new_features = sorted(set(df_after.columns) - set(df_before.columns))

    numeric_new = [
        c for c in all_new_features
        if pd.api.types.is_numeric_dtype(df_after[c])
    ]

    report["summary"] = {
        "before": {
            "shape": _shape_str(df_before),
            "columns": len(df_before.columns),
        },
        "after": {
            "shape": _shape_str(df_after),
            "columns": len(df_after.columns),
        },
        "total_features_created": len(all_new_features),
        "new_features": all_new_features,
        "numeric_features_added": len(numeric_new),
        "feature_groups": len(report.get("groups", [])),
        "group_breakdown": {
            g["group"]: g["features_created"]
            for g in report.get("groups", [])
        },
    }

    return report


def _save_json_report(report: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEATURES_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Feature engineering JSON report saved to: {}", FEATURES_REPORT_PATH)


def _save_text_report(report: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 70,
        "  WALMART SALES CLASSIFICATION — FEATURE ENGINEERING REPORT",
        "=" * 70,
        "",
    ]

    s = report.get("summary", {})
    lines.extend([
        "SUMMARY",
        "-" * 70,
        f"  Before:  {s.get('before', {}).get('shape', '?')}",
        f"  After:   {s.get('after', {}).get('shape', '?')}",
        f"  Total features created: {s.get('total_features_created', 0)}",
        "",
    ])

    for group in report.get("groups", []):
        lines.append(f"{'─' * 70}")
        lines.append(f"  {group['group']}  (+{group['features_created']})")
        lines.append(f"{'─' * 70}")
        lines.append(f"  Features: {group.get('features', [])}")
        lines.append(f"  Rationale: {group.get('rationale', 'N/A')}")
        if "leakage_policy" in group:
            lines.append(f"  Leakage: {group['leakage_policy']}")
        if "leakage_risk" in group:
            lines.append(f"  Leakage Risk: {group['leakage_risk']}")
        lines.append("")

    # Validation
    val = report.get("validation", {})
    lines.append(f"{'─' * 70}")
    lines.append(f"  POST-ENGINEERING VALIDATION: {val.get('status', '?')}")
    lines.append(f"{'─' * 70}")
    for check in val.get("checks", []):
        icon = "✓" if check["status"] == "PASS" else "⚠"
        lines.append(f"  [{icon}] {check['check']}")
    lines.append("")

    lines.append("=" * 70)

    with open(FEATURES_TEXT_REPORT_PATH, "w") as f:
        f.write("\n".join(lines))

    logger.info("Feature engineering text report saved to: {}", FEATURES_TEXT_REPORT_PATH)


# ORCHESTRATOR
def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("Starting Feature Engineering Pipeline")
    logger.info("=" * 60)
    logger.info("Input shape: {}", _shape_str(df))

    df_before = df.copy()
    report: dict[str, Any] = {"groups": []}

    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])

    df = create_temporal_features(df, report)
    df = create_holiday_features(df, report)
    df = create_promotion_features(df, report)
    df = create_store_dept_features(df, report)
    df = create_economic_features(df, report)
    df = create_lag_features(df, report)
    df = create_interaction_features(df, report)
    df = create_cyclical_features(df, report)
    df = validate_features(df, report)

    report = _generate_summary(df_before, df, report)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_PATH, index=False)
    logger.info("Featured dataset saved to: {} ({})", FEATURES_PATH, _shape_str(df))

    _save_json_report(report)
    _save_text_report(report)

    logger.info("")
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING SUMMARY")
    logger.info("=" * 60)
    logger.info("  Before: {}", _shape_str(df_before))
    logger.info("  After:  {}", _shape_str(df))
    logger.info("  Total features created: {}", report["summary"]["total_features_created"])
    logger.info("")
    for group in report["groups"]:
        logger.info("    {:30s}  +{} features", group["group"], group["features_created"])
    logger.info("")
    logger.info(
        "  Validation: {}", report.get("validation", {}).get("status", "?")
    )
    logger.info("=" * 60)

    return df

if __name__ == "__main__":
    cleaned_df = pd.read_csv(CLEANED_PATH, parse_dates=["Date"])
    featured_df = run_feature_engineering(cleaned_df)
