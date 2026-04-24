import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import logger

PROCESSED_DIR = Path("data/processed")
REPORT_PATH = PROCESSED_DIR / "validation_report.txt"
JSON_SUMMARY_PATH = PROCESSED_DIR / "validation_summary.json"
CSV_SUMMARY_PATH = PROCESSED_DIR / "validation_summary.csv"

REQUIRED_COLUMNS = [
    "Store",
    "Dept",
    "Date",
    "Weekly_Sales",
    "IsHoliday",
    "Type",
    "Size",
    "Temperature",
    "Fuel_Price",
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
    "CPI",
    "Unemployment",
    "UMCSENT",
    "RSXFS",
    "PCE",
    "Sales_Class",
]

STRICT_DTYPE_EXPECTATIONS = {
    "Store": "integer",
    "Dept": "integer",
    "Date": "datetime",
    "Weekly_Sales": "numeric",
    "IsHoliday": "bool",
    "Type": "string",
    "Size": "integer",
    "Temperature": "numeric",
    "Fuel_Price": "numeric",
    "MarkDown1": "numeric",
    "MarkDown2": "numeric",
    "MarkDown3": "numeric",
    "MarkDown4": "numeric",
    "MarkDown5": "numeric",
    "CPI": "numeric",
    "Unemployment": "numeric",
    "UMCSENT": "numeric",
    "RSXFS": "numeric",
    "PCE": "numeric",
    "Sales_Class": "integer",
}

SEVERE_COLUMN_MISSINGNESS_PCT = 40.0
SEVERE_ROW_MISSINGNESS_PCT = 30.0


def _dtype_matches(series: pd.Series, expected: str) -> bool:
    if expected == "integer":
        return pd.api.types.is_integer_dtype(series)
    if expected == "numeric":
        return pd.api.types.is_numeric_dtype(series)
    if expected == "datetime":
        return pd.api.types.is_datetime64_any_dtype(series)
    if expected == "bool":
        return pd.api.types.is_bool_dtype(series)
    if expected == "string":
        return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)
    return False

def check_shape(df: pd.DataFrame) -> dict[str, Any]:
    result = {
        "check": "Shape / Size",
        "rows": len(df),
        "columns": len(df.columns),
        "meets_min_rows": len(df) >= 5000,
        "meets_min_cols": len(df.columns) >= 10,
        "status": "PASS" if (len(df) >= 5000 and len(df.columns) >= 10) else "FAIL",
    }
    logger.info("Shape check: {} rows, {} columns — {}", result["rows"], result["columns"], result["status"])
    return result


def check_missing_values(df: pd.DataFrame) -> dict[str, Any]:
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)

    missing_summary = pd.DataFrame({
        "missing_count": missing_counts,
        "missing_pct": missing_pct,
    }).query("missing_count > 0").sort_values("missing_pct", ascending=False)

    total_missing_cells = int(missing_counts.sum())
    total_cells = df.size
    overall_completeness = round(100 * (1 - total_missing_cells / total_cells), 2)

    result = {
        "check": "Missing Values",
        "total_missing_cells": total_missing_cells,
        "overall_completeness_pct": overall_completeness,
        "columns_with_missing": missing_summary.to_dict(orient="index"),
        "status": "WARN" if total_missing_cells > 0 else "PASS",
    }

    logger.info(
        "Missing values: {:,} cells ({:.2f}% complete) — {}",
        total_missing_cells, overall_completeness, result["status"],
    )
    return result


def check_required_schema(df: pd.DataFrame) -> dict[str, Any]:
    actual = set(df.columns)
    required = set(REQUIRED_COLUMNS)
    missing = sorted(required - actual)
    extra = sorted(actual - required)

    status = "PASS"
    if missing:
        status = "FAIL"
    elif extra:
        status = "WARN"

    result = {
        "check": "Required Schema",
        "required_columns": REQUIRED_COLUMNS,
        "missing_required_columns": missing,
        "extra_columns": extra,
        "status": status,
    }
    logger.info(
        "Schema check — missing: {}, extra: {} — {}",
        len(missing),
        len(extra),
        status,
    )
    return result


def check_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    full_dups = df.duplicated().sum()
    key_dups = df.duplicated(subset=["Store", "Dept", "Date"]).sum() if all(
        c in df.columns for c in ["Store", "Dept", "Date"]
    ) else None

    result = {
        "check": "Duplicates",
        "full_row_duplicates": int(full_dups),
        "key_duplicates_store_dept_date": int(key_dups) if key_dups is not None else "N/A",
        "status": "PASS" if full_dups == 0 else "WARN",
    }

    logger.info(
        "Duplicates — full rows: {}, key (Store+Dept+Date): {} — {}",
        full_dups, key_dups, result["status"],
    )
    return result


def check_data_types(df: pd.DataFrame) -> dict[str, Any]:
    dtype_map = {col: str(dtype) for col, dtype in df.dtypes.items()}
    object_cols = [col for col, dt in df.dtypes.items() if dt == "object"]

    result = {
        "check": "Data Types",
        "dtype_map": dtype_map,
        "unexpected_object_columns": object_cols,
        "status": "WARN" if len(object_cols) > 2 else "PASS",
    }

    logger.info(
        "Data types — {} columns, {} object dtype cols — {}",
        len(dtype_map), len(object_cols), result["status"],
    )
    return result


def check_strict_dtypes(df: pd.DataFrame) -> dict[str, Any]:
    missing_columns = [c for c in STRICT_DTYPE_EXPECTATIONS if c not in df.columns]
    mismatches = {}

    for col, expected in STRICT_DTYPE_EXPECTATIONS.items():
        if col not in df.columns:
            continue
        if not _dtype_matches(df[col], expected):
            mismatches[col] = {
                "expected": expected,
                "actual": str(df[col].dtype),
            }

    status = "PASS"
    if mismatches:
        status = "FAIL"
    elif missing_columns:
        status = "WARN"

    result = {
        "check": "Strict Dtype Expectations",
        "missing_columns": missing_columns,
        "mismatches": mismatches,
        "status": status,
    }
    logger.info(
        "Strict dtypes — missing: {}, mismatches: {} — {}",
        len(missing_columns),
        len(mismatches),
        status,
    )
    return result


def check_date_range(df: pd.DataFrame) -> dict[str, Any]:
    if "Date" not in df.columns:
        return {"check": "Date Range", "status": "SKIP", "reason": "No Date column found"}

    min_date = df["Date"].min()
    max_date = df["Date"].max()
    expected_start = pd.Timestamp("2010-01-01")
    expected_end = pd.Timestamp("2012-12-31")

    out_of_range = df[(df["Date"] < expected_start) | (df["Date"] > expected_end)]
    future_dates = df[df["Date"] > pd.Timestamp.today()]

    result = {
        "check": "Date Range",
        "min_date": str(min_date.date()),
        "max_date": str(max_date.date()),
        "out_of_range_rows": len(out_of_range),
        "future_date_rows": len(future_dates),
        "status": "PASS" if len(out_of_range) == 0 else "WARN",
    }

    logger.info(
        "Date range: {} to {} | out-of-range: {} | future: {} — {}",
        result["min_date"], result["max_date"],
        result["out_of_range_rows"], result["future_date_rows"],
        result["status"],
    )
    return result


def check_negative_sales(df: pd.DataFrame) -> dict[str, Any]:
    if "Weekly_Sales" not in df.columns:
        return {"check": "Negative Sales", "status": "SKIP", "reason": "No Weekly_Sales column"}

    negative_rows = df[df["Weekly_Sales"] < 0]
    pct_negative = round(100 * len(negative_rows) / len(df), 3)

    result = {
        "check": "Negative Weekly_Sales",
        "negative_rows": len(negative_rows),
        "pct_negative": pct_negative,
        "min_sales_value": float(df["Weekly_Sales"].min()),
        "status": "WARN" if len(negative_rows) > 0 else "PASS",
        "note": "Negative values may reflect product returns — retain but flag.",
    }

    logger.info(
        "Negative sales: {:,} rows ({:.3f}%) — {}",
        len(negative_rows), pct_negative, result["status"],
    )
    return result


def check_row_level_missingness(df: pd.DataFrame) -> dict[str, Any]:
    row_missing_counts = df.isna().sum(axis=1)
    rows_with_missing = int((row_missing_counts > 0).sum())
    rows_with_missing_pct = round(100 * rows_with_missing / len(df), 2)

    result = {
        "check": "Row-level Missingness",
        "rows_with_missing": rows_with_missing,
        "rows_with_missing_pct": rows_with_missing_pct,
        "max_missing_cells_in_row": int(row_missing_counts.max()),
        "p95_missing_cells_in_row": float(row_missing_counts.quantile(0.95)),
        "status": "WARN" if rows_with_missing > 0 else "PASS",
    }
    logger.info(
        "Row missingness — rows with missing: {} ({:.2f}%) — {}",
        rows_with_missing,
        rows_with_missing_pct,
        result["status"],
    )
    return result


def check_severe_missingness_thresholds(df: pd.DataFrame) -> dict[str, Any]:
    col_missing_pct = (df.isna().mean() * 100).round(2)
    severe_columns = {
        col: float(pct)
        for col, pct in col_missing_pct.items()
        if pct >= SEVERE_COLUMN_MISSINGNESS_PCT
    }

    row_missing_pct = df.isna().mean(axis=1) * 100
    severe_rows = int((row_missing_pct >= SEVERE_ROW_MISSINGNESS_PCT).sum())
    severe_rows_pct = round(100 * severe_rows / len(df), 2)

    status = "WARN" if severe_columns or severe_rows > 0 else "PASS"
    result = {
        "check": "Severe Missingness Thresholds",
        "column_threshold_pct": SEVERE_COLUMN_MISSINGNESS_PCT,
        "row_threshold_pct": SEVERE_ROW_MISSINGNESS_PCT,
        "severe_columns": severe_columns,
        "severe_row_count": severe_rows,
        "severe_row_pct": severe_rows_pct,
        "status": status,
    }
    logger.info(
        "Severe missingness — severe cols: {}, severe rows: {} — {}",
        len(severe_columns),
        severe_rows,
        status,
    )
    return result


def check_value_ranges(df: pd.DataFrame) -> dict[str, Any]:
    numeric_df = df.select_dtypes(include=[np.number])
    stats = numeric_df.describe().round(2).to_dict()

    outlier_flags = []
    for col in numeric_df.columns:
        q75 = numeric_df[col].quantile(0.75)
        col_max = numeric_df[col].max()
        if q75 > 0 and col_max > 10 * q75:
            outlier_flags.append(col)

    result = {
        "check": "Value Ranges",
        "numeric_column_count": len(numeric_df.columns),
        "descriptive_stats": stats,
        "potential_outlier_columns": outlier_flags,
        "status": "WARN" if outlier_flags else "PASS",
    }

    logger.info(
        "Value ranges — {} numeric cols | potential outlier cols: {} — {}",
        len(numeric_df.columns), outlier_flags, result["status"],
    )
    return result


def check_class_distribution(df: pd.DataFrame) -> dict[str, Any]:
    if "Sales_Class" not in df.columns:
        return {"check": "Class Distribution", "status": "SKIP", "reason": "Target not yet created"}

    dist = df["Sales_Class"].value_counts()
    pct = (df["Sales_Class"].value_counts(normalize=True) * 100).round(2)
    imbalance_ratio = round(dist.max() / dist.min(), 3)

    result = {
        "check": "Class Distribution",
        "class_counts": dist.to_dict(),
        "class_percentages": pct.to_dict(),
        "imbalance_ratio": imbalance_ratio,
        "is_balanced": imbalance_ratio < 1.5,
        "status": "PASS" if imbalance_ratio < 2.0 else "WARN",
    }

    logger.info(
        "Class distribution — High: {} ({:.1f}%), Low: {} ({:.1f}%) | ratio: {:.3f} — {}",
        dist.get(1, 0), pct.get(1, 0),
        dist.get(0, 0), pct.get(0, 0),
        imbalance_ratio, result["status"],
    )
    return result


def check_target_validity(df: pd.DataFrame) -> dict[str, Any]:
    if "Sales_Class" not in df.columns:
        return {
            "check": "Target Validity",
            "status": "FAIL",
            "reason": "Sales_Class column is missing",
        }

    null_count = int(df["Sales_Class"].isna().sum())
    invalid_values = sorted(set(df["Sales_Class"].dropna().unique()) - {0, 1})

    status = "PASS" if null_count == 0 and not invalid_values else "FAIL"
    result = {
        "check": "Target Validity",
        "null_count": null_count,
        "invalid_values": invalid_values,
        "status": status,
    }
    logger.info(
        "Target validity — nulls: {}, invalid values: {} — {}",
        null_count,
        len(invalid_values),
        status,
    )
    return result


def check_categorical_domains(df: pd.DataFrame) -> dict[str, Any]:
    issues = []
    details: dict[str, Any] = {}

    if "Type" in df.columns:
        allowed = {"A", "B", "C"}
        invalid = sorted(set(df["Type"].dropna().astype(str).unique()) - allowed)
        details["Type_allowed"] = sorted(allowed)
        details["Type_invalid_values"] = invalid
        if invalid:
            issues.append(f"Invalid Type values: {invalid}")

    if "IsHoliday" in df.columns:
        allowed_bool = {True, False}
        unique_vals = set(df["IsHoliday"].dropna().unique())
        invalid_bool = sorted([v for v in unique_vals if v not in allowed_bool], key=str)
        details["IsHoliday_invalid_values"] = invalid_bool
        if invalid_bool:
            issues.append(f"Invalid IsHoliday values: {invalid_bool}")

    result = {
        "check": "Categorical Domain Checks",
        "issues": issues,
        "details": details,
        "status": "PASS" if not issues else "WARN",
    }
    logger.info("Categorical domain checks — issues: {} — {}", len(issues), result["status"])
    return result


def check_fred_coverage(df: pd.DataFrame) -> dict[str, Any]:
    fred_cols = ["UMCSENT", "RSXFS", "PCE"]
    coverage = {}
    for col in fred_cols:
        if col in df.columns:
            null_pct = round(100 * df[col].isna().mean(), 2)
            coverage[col] = {
                "present": True,
                "null_pct": null_pct,
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
        else:
            coverage[col] = {"present": False, "null_pct": 100.0}

    all_present = all(v["present"] for v in coverage.values())
    all_populated = all(v["null_pct"] < 5 for v in coverage.values() if v["present"])

    result = {
        "check": "FRED Coverage",
        "fred_column_coverage": coverage,
        "all_series_present": all_present,
        "status": "PASS" if (all_present and all_populated) else "WARN",
    }

    logger.info("FRED coverage — all present: {} — {}", all_present, result["status"])
    return result


def check_referential_integrity(df: pd.DataFrame) -> dict[str, Any]:
    issues = []

    if "Store" in df.columns:
        store_ids = df["Store"].unique()
        invalid_stores = [s for s in store_ids if not (1 <= s <= 45)]
        if invalid_stores:
            issues.append(f"Invalid Store IDs: {invalid_stores}")

    if "Dept" in df.columns:
        dept_ids = df["Dept"].unique()
        invalid_depts = [d for d in dept_ids if not (1 <= d <= 99)]
        if invalid_depts:
            issues.append(f"Invalid Dept IDs (out of 1–99): {len(invalid_depts)} values")

    result = {
        "check": "Referential Integrity",
        "unique_stores": int(df["Store"].nunique()) if "Store" in df.columns else "N/A",
        "unique_depts": int(df["Dept"].nunique()) if "Dept" in df.columns else "N/A",
        "issues_found": issues,
        "status": "PASS" if not issues else "WARN",
    }

    logger.info(
        "Referential integrity — stores: {}, depts: {}, issues: {} — {}",
        result["unique_stores"], result["unique_depts"],
        len(issues), result["status"],
    )
    return result

def run_validation(df: pd.DataFrame) -> dict[str, Any]:
    logger.info("=" * 60)
    logger.info("Starting Data Validation Pipeline")
    logger.info("=" * 60)

    all_checks = [
        check_shape(df),
        check_required_schema(df),
        check_missing_values(df),
        check_row_level_missingness(df),
        check_severe_missingness_thresholds(df),
        check_duplicates(df),
        check_data_types(df),
        check_strict_dtypes(df),
        check_date_range(df),
        check_negative_sales(df),
        check_value_ranges(df),
        check_target_validity(df),
        check_class_distribution(df),
        check_categorical_domains(df),
        check_fred_coverage(df),
        check_referential_integrity(df),
    ]

    missing_snapshot = (
        pd.DataFrame({
            "missing_count": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
        })
        .query("missing_count > 0")
        .sort_values("missing_pct", ascending=False)
        .head(10)
        .to_dict(orient="index")
    )

    schema_snapshot = {col: str(dtype) for col, dtype in df.dtypes.items()}
    head_snapshot = df.head(5).copy()
    if "Date" in head_snapshot.columns:
        head_snapshot["Date"] = head_snapshot["Date"].astype(str)

    pass_count = sum(1 for c in all_checks if c.get("status") == "PASS")
    warn_count = sum(1 for c in all_checks if c.get("status") == "WARN")
    fail_count = sum(1 for c in all_checks if c.get("status") == "FAIL")

    report = {
        "dataset_shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": list(df.columns),
        "summary": {
            "total_checks": len(all_checks),
            "passed": pass_count,
            "warnings": warn_count,
            "failed": fail_count,
        },
        "checks": {c["check"]: c for c in all_checks},
        "snapshots": {
            "schema": schema_snapshot,
            "top_missing_columns": missing_snapshot,
            "sample_rows_head": head_snapshot.to_dict(orient="records"),
        },
    }

    logger.info(
        "Validation complete — PASS: {}, WARN: {}, FAIL: {}",
        pass_count, warn_count, fail_count,
    )

    _save_text_report(report)
    _save_json_summary(report)
    _save_csv_summary(report)

    return report


def _save_text_report(report: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 70,
        "  WALMART SALES CLASSIFICATION — DATA VALIDATION REPORT",
        "=" * 70,
        f"  Dataset shape : {report['dataset_shape']['rows']:,} rows x {report['dataset_shape']['columns']} columns",
        f"  Total checks  : {report['summary']['total_checks']}",
        f"  PASS          : {report['summary']['passed']}",
        f"  WARN          : {report['summary']['warnings']}",
        f"  FAIL          : {report['summary']['failed']}",
        "=" * 70,
        "",
    ]

    for check_name, check_result in report["checks"].items():
        status = check_result.get("status", "?")
        lines.append(f"[{status}] {check_name}")
        for k, v in check_result.items():
            if k not in ("check", "status", "descriptive_stats", "dtype_map"):
                lines.append(f"      {k}: {v}")
        lines.append("")

    lines.extend([
        "-" * 70,
        "VALIDATION SNAPSHOTS",
        "-" * 70,
        "Top missing columns:",
    ])

    top_missing = report.get("snapshots", {}).get("top_missing_columns", {})
    if top_missing:
        for col, stats in top_missing.items():
            lines.append(f"  - {col}: {stats}")
    else:
        lines.append("  - None")

    lines.extend([
        "",
        "Schema snapshot (column -> dtype):",
    ])
    schema_snapshot = report.get("snapshots", {}).get("schema", {})
    for col, dtype in schema_snapshot.items():
        lines.append(f"  - {col}: {dtype}")

    lines.extend([
        "",
        "Sample rows (head):",
    ])
    for idx, row in enumerate(report.get("snapshots", {}).get("sample_rows_head", []), start=1):
        lines.append(f"  [{idx}] {row}")

    lines.append("=" * 70)

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))

    logger.info("Validation report saved to: {}", REPORT_PATH)


def _save_json_summary(report: dict[str, Any]) -> None:
    with open(JSON_SUMMARY_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Validation JSON summary saved to: {}", JSON_SUMMARY_PATH)


def _save_csv_summary(report: dict[str, Any]) -> None:
    rows = []
    for check_name, check_result in report["checks"].items():
        details = {
            k: v
            for k, v in check_result.items()
            if k not in {"check", "status", "descriptive_stats", "dtype_map"}
        }
        rows.append(
            {
                "check": check_name,
                "status": check_result.get("status", "?"),
                "details": json.dumps(details, default=str),
            }
        )

    pd.DataFrame(rows).to_csv(CSV_SUMMARY_PATH, index=False)
    logger.info("Validation CSV summary saved to: {}", CSV_SUMMARY_PATH)


if __name__ == "__main__":
    from src.data.acquisition import run_acquisition_pipeline

    merged_df = run_acquisition_pipeline()
    run_validation(merged_df)
