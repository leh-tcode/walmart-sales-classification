import json
from typing import Any

import pandas as pd

from src.utils.logger import logger
from src.validation.checks import (
    check_accuracy,
    check_completeness,
    check_consistency,
    check_distribution_profile,
    check_outliers,
    check_relationships,
    check_uniqueness,
)
from src.validation.constants import (
    CSV_SUMMARY_PATH,
    JSON_SUMMARY_PATH,
    PROCESSED_DIR,
    REPORT_PATH,
)


# ORCHESTRATOR
def run_validation(df: pd.DataFrame) -> dict[str, Any]:
    logger.info("=" * 60)
    logger.info("Starting Data Validation Pipeline")
    logger.info("=" * 60)

    dimensions = {
        "1_accuracy": check_accuracy(df),
        "2_completeness": check_completeness(df),
        "3_consistency": check_consistency(df),
        "4_uniqueness": check_uniqueness(df),
        "5_outlier_detection": check_outliers(df),
        "6_distribution_profile": check_distribution_profile(df),
        "7_relationships": check_relationships(df),
    }

    dim_statuses = {k: v.get("overall_status", "N/A") for k, v in dimensions.items()}
    has_fail = any(s == "FAIL" for s in dim_statuses.values())
    has_warn = any(s == "WARN" for s in dim_statuses.values())
    overall = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")

    total_checks = sum(d.get("total_checks", 0) for d in dimensions.values())
    total_passed = sum(d.get("passed", 0) for d in dimensions.values())
    total_warned = sum(d.get("warnings", 0) for d in dimensions.values())
    total_failed = sum(d.get("failed", 0) for d in dimensions.values())

    missing_snapshot = (
        pd.DataFrame(
            {
                "missing_count": df.isna().sum(),
                "missing_pct": (df.isna().mean() * 100).round(2),
            }
        )
        .query("missing_count > 0")
        .sort_values("missing_pct", ascending=False)
        .head(10)
        .to_dict(orient="index")
    )
    schema_snapshot = {col: str(dtype) for col, dtype in df.dtypes.items()}
    head = df.head(5).copy()
    if "Date" in head.columns:
        head["Date"] = head["Date"].astype(str)

    report = {
        "dataset_shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": list(df.columns),
        "summary": {
            "total_checks": total_checks,
            "passed": total_passed,
            "warnings": total_warned,
            "failed": total_failed,
            "dimensions_evaluated": len(dimensions),
            "dimensions_passed": sum(1 for s in dim_statuses.values() if s == "PASS"),
            "dimensions_warned": sum(1 for s in dim_statuses.values() if s == "WARN"),
            "dimensions_failed": sum(1 for s in dim_statuses.values() if s == "FAIL"),
            "dimension_statuses": dim_statuses,
            "overall_status": overall,
        },
        "dimensions": dimensions,
        "snapshots": {
            "schema": schema_snapshot,
            "top_missing_columns": missing_snapshot,
            "sample_rows_head": head.to_dict(orient="records"),
        },
    }

    logger.info("")
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    for dim, status in dim_statuses.items():
        icon = "✓" if status == "PASS" else ("⚠" if status == "WARN" else "✗")
        logger.info("  [{}]  {:40s}  {}", icon, dim, status)
    logger.info("-" * 60)
    logger.info(
        "  Overall: {}  ({}/{} dimensions passed)",
        report["summary"]["overall_status"],
        report["summary"]["dimensions_passed"],
        len(dimensions),
    )
    logger.info("=" * 60)

    _save_text_report(report)
    _save_json_summary(report)
    _save_csv_summary(report)

    return report


# Report writers
def _save_text_report(report: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 70,
        "  WALMART SALES CLASSIFICATION — DATA VALIDATION REPORT",
        "=" * 70,
        f"  Dataset shape : {report['dataset_shape']['rows']:,} rows x "
        f"{report['dataset_shape']['columns']} columns",
        f"  Total checks  : {report['summary']['total_checks']}",
        f"  PASS          : {report['summary']['passed']}",
        f"  WARN          : {report['summary']['warnings']}",
        f"  FAIL          : {report['summary']['failed']}",
        "=" * 70,
        "",
    ]

    for dim_key, dim_data in report["dimensions"].items():
        dim_name = dim_data.get("dimension", dim_key)
        dim_status = dim_data.get("overall_status", "?")
        lines.append(f"{'─' * 70}")
        lines.append(f"  DIMENSION: {dim_name}   [{dim_status}]")
        lines.append(f"{'─' * 70}")

        for check in dim_data.get("checks", []):
            status = check.get("status", "?")
            name = check.get("check", "unnamed")
            lines.append(f"  [{status:4s}] {name}")
            for k, v in check.items():
                if k not in ("check", "status"):
                    lines.append(f"          {k}: {v}")
            lines.append("")

        if "columns" in dim_data and isinstance(dim_data["columns"], dict):
            for col_name, col_info in dim_data["columns"].items():
                col_status = col_info.get("overall_status", "?")
                lines.append(f"  [{col_status:4s}] {col_name}")
                for method in ["iqr_method", "zscore_method"]:
                    if method in col_info:
                        lines.append(f"          {method}: {col_info[method]}")
                lines.append("")

    lines.extend(
        [
            "-" * 70,
            "SNAPSHOTS",
            "-" * 70,
            "Top missing columns:",
        ]
    )
    for col, st in report.get("snapshots", {}).get("top_missing_columns", {}).items():
        lines.append(f"  - {col}: {st}")

    lines.append("")
    lines.append("Schema snapshot:")
    for col, dtype in report.get("snapshots", {}).get("schema", {}).items():
        lines.append(f"  - {col}: {dtype}")

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
    for dim_key, dim_data in report["dimensions"].items():
        dim_name = dim_data.get("dimension", dim_key)
        for check in dim_data.get("checks", []):
            details = {k: v for k, v in check.items() if k not in {"check", "status"}}
            rows.append(
                {
                    "dimension": dim_name,
                    "check": check.get("check", "unnamed"),
                    "status": check.get("status", "?"),
                    "details": json.dumps(details, default=str),
                }
            )
        if "columns" in dim_data and isinstance(dim_data["columns"], dict):
            for col_name, col_info in dim_data["columns"].items():
                rows.append(
                    {
                        "dimension": dim_name,
                        "check": f"Outlier — {col_name}",
                        "status": col_info.get("overall_status", "?"),
                        "details": json.dumps(col_info, default=str),
                    }
                )

    pd.DataFrame(rows).to_csv(CSV_SUMMARY_PATH, index=False)
    logger.info("Validation CSV summary saved to: {}", CSV_SUMMARY_PATH)


if __name__ == "__main__":
    from src.data.acquisition import run_acquisition_pipeline

    merged_df = run_acquisition_pipeline()
    run_validation(merged_df)
