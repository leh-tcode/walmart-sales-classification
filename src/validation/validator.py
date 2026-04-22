"""
Data Validation Module
======================
Performs comprehensive data quality checks on the merged Walmart + FRED dataset.
Covers all validation dimensions discussed in the course:
  - Structural integrity (shape, dtypes, column presence)
  - Completeness (missing values per column and row)
  - Uniqueness (duplicate records)
  - Consistency (date ranges, value ranges, type consistency)
  - Class distribution (target balance)
  - Domain validity (negative sales, future dates, etc.)
  - Referential integrity (store IDs, dept IDs cross-check)
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import logger

PROCESSED_DIR = Path("data/processed")
REPORT_PATH = PROCESSED_DIR / "validation_report.txt"


# ── Individual Validation Checks ──────────────────────────────────────────────

def check_shape(df: pd.DataFrame) -> dict[str, Any]:
    """Check row and column counts meet minimum project requirements."""
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
    """Compute missing value counts and percentages for each column."""
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


def check_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    """
    Check for duplicate records.
    A duplicate in this dataset means same Store + Dept + Date.
    """
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
    """Verify column data types and flag any unexpected object columns."""
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


def check_date_range(df: pd.DataFrame) -> dict[str, Any]:
    """Validate that Date column is within expected Walmart range (2010–2012)."""
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
    """
    Flag rows with negative Weekly_Sales.
    Negative sales can legitimately occur due to returns but should be documented.
    """
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


def check_value_ranges(df: pd.DataFrame) -> dict[str, Any]:
    """Compute descriptive statistics for numeric columns and flag anomalies."""
    numeric_df = df.select_dtypes(include=[np.number])
    stats = numeric_df.describe().round(2).to_dict()

    # Flag columns where max is suspiciously large (> 10x the 75th percentile)
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
    """Check the distribution of the binary classification target Sales_Class."""
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


def check_fred_coverage(df: pd.DataFrame) -> dict[str, Any]:
    """Verify FRED columns are present and not entirely null after the merge."""
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
    """
    Check store ID and department ID ranges are within expected Walmart bounds.
    Walmart competition has 45 stores and dept IDs typically 1–99.
    """
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


# ── Full Validation Runner ─────────────────────────────────────────────────────

def run_validation(df: pd.DataFrame) -> dict[str, Any]:
    """
    Execute all validation checks and compile a comprehensive report.

    Parameters
    ----------
    df : pd.DataFrame
        The merged dataset to validate.

    Returns
    -------
    dict
        Full validation report with all check results.
    """
    logger.info("=" * 60)
    logger.info("Starting Data Validation Pipeline")
    logger.info("=" * 60)

    all_checks = [
        check_shape(df),
        check_missing_values(df),
        check_duplicates(df),
        check_data_types(df),
        check_date_range(df),
        check_negative_sales(df),
        check_value_ranges(df),
        check_class_distribution(df),
        check_fred_coverage(df),
        check_referential_integrity(df),
    ]

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
    }

    logger.info(
        "Validation complete — PASS: {}, WARN: {}, FAIL: {}",
        pass_count, warn_count, fail_count,
    )

    # Save text report
    _save_text_report(report)

    return report


def _save_text_report(report: dict) -> None:
    """Save a human-readable validation report to data/processed/."""
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

    lines.append("=" * 70)

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))

    logger.info("Validation report saved to: {}", REPORT_PATH)


if __name__ == "__main__":
    from src.data.acquisition import run_acquisition_pipeline

    merged_df = run_acquisition_pipeline()
    run_validation(merged_df)
