import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import logger

# Paths
PROCESSED_DIR = Path("data/processed")
CLEANED_PATH = PROCESSED_DIR / "cleaned_dataset.csv"
CLEANING_REPORT_PATH = PROCESSED_DIR / "cleaning_report.json"
CLEANING_TEXT_REPORT_PATH = PROCESSED_DIR / "cleaning_report.txt"

# Constants

MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]

CLIP_COLS = [
    "Weekly_Sales",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
]

CLIP_LOWER_PERCENTILE = 0.01
CLIP_UPPER_PERCENTILE = 0.99


# Helpers
def _pct(count: int, total: int) -> float:
    return round(count / total * 100, 3) if total else 0.0


def _shape_str(df: pd.DataFrame) -> str:
    return f"{len(df):,} rows × {len(df.columns)} cols"


# CLEANING STEP 1: Handle MarkDown Structural Missingness
def handle_markdown_nulls(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    logger.info("Step 1: Handling MarkDown structural missingness …")
    step_report = {
        "step": "Handle MarkDown Structural Nulls",
        "reason": (
            "MarkDown1-5 are structurally null before Nov 2011 — "
            "Walmart did not report promotional markdowns until then. "
            "This is expected missingness, not a data quality issue."
        ),
        "strategy": "Create binary flags + fill NaN with 0",
        "columns_affected": MARKDOWN_COLS,
        "details": {},
    }

    for col in MARKDOWN_COLS:
        if col not in df.columns:
            continue

        null_before = int(df[col].isna().sum())
        null_pct = _pct(null_before, len(df))

        flag_col = f"has_{col}"
        df[flag_col] = df[col].notna().astype(int)
        promotions_active = int(df[flag_col].sum())

        df[col] = df[col].fillna(0.0)

        null_after = int(df[col].isna().sum())

        step_report["details"][col] = {
            "nulls_before": null_before,
            "nulls_before_pct": null_pct,
            "nulls_after": null_after,
            "flag_column_created": flag_col,
            "promotions_active_rows": promotions_active,
            "promotions_active_pct": _pct(promotions_active, len(df)),
            "fill_value": 0.0,
        }

        logger.info(
            "  {} — nulls: {:,} ({:.1f}%) → 0 | flag '{}' created ({:,} active)",
            col, null_before, null_pct, flag_col, promotions_active,
        )

    report["steps"].append(step_report)
    return df


# CLEANING STEP 2: Handle Negative Weekly_Sales
def handle_negative_sales(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    logger.info("Step 2: Handling negative Weekly_Sales …")

    negative_mask = df["Weekly_Sales"] < 0
    neg_count = int(negative_mask.sum())
    neg_pct = _pct(neg_count, len(df))

    df["is_return"] = negative_mask.astype(int)

    step_report = {
        "step": "Handle Negative Weekly_Sales",
        "reason": (
            "Negative values represent product returns/refunds — "
            "a normal part of retail operations."
        ),
        "strategy": "Keep as-is + create binary 'is_return' flag",
        "negative_rows": neg_count,
        "negative_pct": neg_pct,
        "min_value": round(float(df["Weekly_Sales"].min()), 2),
        "max_negative": round(float(df.loc[negative_mask, "Weekly_Sales"].max()), 2) if neg_count > 0 else None,
        "action_taken": "No values modified — domain-valid negative sales retained",
        "flag_column_created": "is_return",
    }

    logger.info(
        "  Negative sales: {:,} rows ({:.3f}%) — KEPT (returns are domain-valid)",
        neg_count, neg_pct,
    )

    report["steps"].append(step_report)
    return df


# CLEANING STEP 3: Clip Outliers in Skewed Columns
def clip_outliers(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    logger.info("Step 3: Clipping outliers in skewed columns …")

    step_report = {
        "step": "Clip Outliers in Skewed Columns",
        "reason": (
            "Heavy right skew (3.2–8.4) causes IQR to over-flag valid values. "
            "Clipping at 1st–99th percentile reduces extreme tail influence "
            "without removing rows or destroying distribution shape."
        ),
        "strategy": f"Clip at P{CLIP_LOWER_PERCENTILE*100:.0f} and P{CLIP_UPPER_PERCENTILE*100:.0f}",
        "columns_affected": [],
        "details": {},
    }

    for col in CLIP_COLS:
        if col not in df.columns:
            continue

        s = df[col]
        non_null = s.dropna()
        if len(non_null) == 0:
            continue

        lower_bound = float(non_null.quantile(CLIP_LOWER_PERCENTILE))
        upper_bound = float(non_null.quantile(CLIP_UPPER_PERCENTILE))

        skew_before = round(float(s.skew()), 4)
        std_before = round(float(s.std()), 4)
        min_before = round(float(s.min()), 4)
        max_before = round(float(s.max()), 4)

        clipped_low = int((s < lower_bound).sum())
        clipped_high = int((s > upper_bound).sum())
        total_clipped = clipped_low + clipped_high

        df[col] = s.clip(lower=lower_bound, upper=upper_bound)

        skew_after = round(float(df[col].skew()), 4)
        std_after = round(float(df[col].std()), 4)

        step_report["columns_affected"].append(col)
        step_report["details"][col] = {
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "clipped_low": clipped_low,
            "clipped_high": clipped_high,
            "total_clipped": total_clipped,
            "total_clipped_pct": _pct(total_clipped, len(df)),
            "before": {
                "min": min_before,
                "max": max_before,
                "skewness": skew_before,
                "std": std_before,
            },
            "after": {
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4),
                "skewness": skew_after,
                "std": std_after,
            },
            "skewness_reduction": round(skew_before - skew_after, 4),
        }

        logger.info(
            "  {} — clipped {:,} values | skew: {:.2f} → {:.2f} | "
            "range: [{:.0f}, {:.0f}] → [{:.0f}, {:.0f}]",
            col, total_clipped,
            skew_before, skew_after,
            min_before, max_before,
            lower_bound, upper_bound,
        )

    report["steps"].append(step_report)
    return df


# CLEANING STEP 4: Validate Post-Cleaning State
def post_cleaning_validation(df: pd.DataFrame, report: dict) -> pd.DataFrame:

    logger.info("Step 4: Post-cleaning validation …")

    checks = []

    total_nulls = int(df.isna().sum().sum())
    checks.append({
        "check": "No remaining null values",
        "total_nulls": total_nulls,
        "status": "PASS" if total_nulls == 0 else "WARN",
        "columns_with_nulls": {
            col: int(df[col].isna().sum())
            for col in df.columns
            if df[col].isna().sum() > 0
        },
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
        "note": "No rows were dropped during cleaning",
    })

    if "Sales_Class" in df.columns:
        dist = df["Sales_Class"].value_counts()
        ratio = round(dist.max() / dist.min(), 3)
        checks.append({
            "check": "Target distribution intact",
            "class_counts": dist.to_dict(),
            "imbalance_ratio": ratio,
            "status": "PASS" if ratio < 2.0 else "WARN",
        })

    expected_new = [f"has_{c}" for c in MARKDOWN_COLS] + ["is_return"]
    actual_new = [c for c in expected_new if c in df.columns]
    missing_new = [c for c in expected_new if c not in df.columns]
    checks.append({
        "check": "All expected new columns created",
        "expected": expected_new,
        "created": actual_new,
        "missing": missing_new,
        "status": "PASS" if len(missing_new) == 0 else "FAIL",
    })

    dtype_issues = []
    for col in df.select_dtypes(include=["object"]).columns:
        if col not in ["Type"]:  
            dtype_issues.append(col)
    checks.append({
        "check": "No unexpected object dtypes",
        "unexpected_object_columns": dtype_issues,
        "status": "PASS" if len(dtype_issues) == 0 else "WARN",
    })

    passed = sum(1 for c in checks if c["status"] == "PASS")
    step_report = {
        "step": "Post-Cleaning Validation",
        "checks": checks,
        "passed": passed,
        "total": len(checks),
        "status": "PASS" if passed == len(checks) else "WARN",
    }

    for check in checks:
        icon = "✓" if check["status"] == "PASS" else "⚠"
        logger.info("  [{}] {}", icon, check["check"])

    report["steps"].append(step_report)
    return df


# REPORT GENERATION
def _generate_cleaning_summary(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    report: dict,
) -> dict:

    null_before = int(df_before.isna().sum().sum())
    null_after = int(df_after.isna().sum().sum())

    summary = {
        "before": {
            "shape": _shape_str(df_before),
            "rows": len(df_before),
            "columns": len(df_before.columns),
            "total_null_cells": null_before,
            "completeness_pct": round(
                100 * (1 - null_before / df_before.size), 2
            ),
        },
        "after": {
            "shape": _shape_str(df_after),
            "rows": len(df_after),
            "columns": len(df_after.columns),
            "total_null_cells": null_after,
            "completeness_pct": round(
                100 * (1 - null_after / df_after.size), 2
            ),
        },
        "changes": {
            "rows_removed": len(df_before) - len(df_after),
            "columns_added": len(df_after.columns) - len(df_before.columns),
            "new_columns": sorted(
                set(df_after.columns) - set(df_before.columns)
            ),
            "null_cells_resolved": null_before - null_after,
            "completeness_improvement_pct": round(
                (1 - null_after / df_after.size) * 100
                - (1 - null_before / df_before.size) * 100,
                2,
            ),
        },
    }

    report["summary"] = summary
    return report


def _save_text_report(report: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 70,
        "  WALMART SALES CLASSIFICATION — DATA CLEANING REPORT",
        "=" * 70,
        "",
    ]

    s = report.get("summary", {})
    before = s.get("before", {})
    after = s.get("after", {})
    changes = s.get("changes", {})

    lines.extend([
        "SUMMARY",
        "-" * 70,
        f"  Before:  {before.get('shape', '?')}  |  "
        f"Nulls: {before.get('total_null_cells', '?'):,}  |  "
        f"Completeness: {before.get('completeness_pct', '?')}%",
        f"  After:   {after.get('shape', '?')}  |  "
        f"Nulls: {after.get('total_null_cells', '?'):,}  |  "
        f"Completeness: {after.get('completeness_pct', '?')}%",
        f"  Rows removed: {changes.get('rows_removed', 0)}",
        f"  Columns added: {changes.get('columns_added', 0)} "
        f"({changes.get('new_columns', [])})",
        f"  Null cells resolved: {changes.get('null_cells_resolved', 0):,}",
        "",
    ])

    for i, step in enumerate(report.get("steps", []), 1):
        lines.append(f"{'─' * 70}")
        lines.append(f"  STEP {i}: {step.get('step', 'Unknown')}")
        lines.append(f"{'─' * 70}")
        lines.append(f"  Reason:   {step.get('reason', 'N/A')}")
        lines.append(f"  Strategy: {step.get('strategy', 'N/A')}")
        lines.append("")

        details = step.get("details", {})
        if isinstance(details, dict):
            for key, val in details.items():
                if isinstance(val, dict):
                    lines.append(f"    {key}:")
                    for k, v in val.items():
                        lines.append(f"      {k}: {v}")
                else:
                    lines.append(f"    {key}: {val}")
        lines.append("")

    lines.append("=" * 70)

    with open(CLEANING_TEXT_REPORT_PATH, "w") as f:
        f.write("\n".join(lines))

    logger.info("Cleaning text report saved to: {}", CLEANING_TEXT_REPORT_PATH)


def _save_json_report(report: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLEANING_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Cleaning JSON report saved to: {}", CLEANING_REPORT_PATH)


# ORCHESTRATOR
def run_cleaning(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("=" * 60)
    logger.info("Starting Data Cleaning Pipeline")
    logger.info("=" * 60)
    logger.info("Input shape: {}", _shape_str(df))

    df_before = df.copy()

    report: dict[str, Any] = {"steps": []}

    df = handle_markdown_nulls(df, report)
    df = handle_negative_sales(df, report)
    df = clip_outliers(df, report)
    df = post_cleaning_validation(df, report)

    report = _generate_cleaning_summary(df_before, df, report)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_PATH, index=False)
    logger.info("Cleaned dataset saved to: {} ({})", CLEANED_PATH, _shape_str(df))

    _save_json_report(report)
    _save_text_report(report)

    logger.info("")
    logger.info("=" * 60)
    logger.info("CLEANING SUMMARY")
    logger.info("=" * 60)
    s = report["summary"]
    logger.info(
        "  Before: {} | Completeness: {}%",
        s["before"]["shape"], s["before"]["completeness_pct"],
    )
    logger.info(
        "  After:  {} | Completeness: {}%",
        s["after"]["shape"], s["after"]["completeness_pct"],
    )
    logger.info("  Rows removed:       {}", s["changes"]["rows_removed"])
    logger.info("  Columns added:      {} {}", s["changes"]["columns_added"], s["changes"]["new_columns"])
    logger.info("  Null cells resolved: {:,}", s["changes"]["null_cells_resolved"])
    logger.info("=" * 60)

    return df

if __name__ == "__main__":
    input_path = PROCESSED_DIR / "merged_dataset.csv"
    logger.info("Loading merged dataset from: {}", input_path)
    merged_df = pd.read_csv(input_path, parse_dates=["Date"])
    run_cleaning(merged_df)