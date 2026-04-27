from typing import Any

import pandas as pd

from src.utils.logger import logger
from src.validation.common import _dimension_summary, _pct, _pf
from src.validation.constants import (
    COMPLETENESS_THRESHOLDS,
    REQUIRED_COLUMNS,
    SEVERE_COLUMN_MISSINGNESS_PCT,
    SEVERE_ROW_MISSINGNESS_PCT,
)


# 2 ─ COMPLETENESS
def check_completeness(df: pd.DataFrame) -> dict[str, Any]:
    logger.info("Running [2] COMPLETENESS checks …")
    checks: list[dict] = []
    n = len(df)

    # 2-a  Row count minimum
    meets_rows = n >= 5_000
    checks.append(
        {
            "check": "Minimum row count >= 5 000",
            "type": "row_count",
            "actual_rows": n,
            "minimum_expected": 5_000,
            "status": _pf(meets_rows),
        }
    )

    # 2-b  Required schema presence
    actual = set(df.columns)
    required = set(REQUIRED_COLUMNS)
    missing_cols = sorted(required - actual)
    extra_cols = sorted(actual - required)
    checks.append(
        {
            "check": "All required columns present",
            "type": "schema",
            "expected": len(REQUIRED_COLUMNS),
            "actual": len(df.columns),
            "missing_columns": missing_cols,
            "extra_columns": extra_cols,
            "status": _pf(len(missing_cols) == 0),
        }
    )

    # 2-c  Per-column null thresholds
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            checks.append(
                {
                    "check": f"{col} null threshold",
                    "column": col,
                    "type": "column_missingness",
                    "status": "FAIL",
                    "note": "Column missing from dataset",
                }
            )
            continue
        null_count = int(df[col].isna().sum())
        null_pct = _pct(null_count, n)
        max_null = COMPLETENESS_THRESHOLDS.get(col, 20.0)
        criticality = "critical" if max_null == 0.0 else "high" if max_null <= 5.0 else "medium"
        checks.append(
            {
                "check": f"{col} null ≤ {max_null}%",
                "column": col,
                "type": "column_missingness",
                "null_count": null_count,
                "null_pct": null_pct,
                "max_allowed_null_pct": max_null,
                "criticality": criticality,
                "status": _pf(null_pct <= max_null),
            }
        )

    # 2-d  Row-level missingness
    row_miss = df.isna().sum(axis=1)
    rows_with_miss = int((row_miss > 0).sum())
    checks.append(
        {
            "check": "Row-level missingness profile",
            "type": "row_missingness",
            "rows_with_missing": rows_with_miss,
            "rows_with_missing_pct": _pct(rows_with_miss, n),
            "max_missing_cells_in_row": int(row_miss.max()),
            "p95_missing_cells_in_row": float(row_miss.quantile(0.95)),
            "status": "WARN" if rows_with_miss > 0 else "PASS",
        }
    )

    # 2-e  Severe missingness thresholds
    miss_pct = (df.isna().mean() * 100).round(2)
    severe_cols = {col: float(pct) for col, pct in miss_pct.items() if pct >= SEVERE_COLUMN_MISSINGNESS_PCT}
    row_miss_pct = df.isna().mean(axis=1) * 100
    severe_rows = int((row_miss_pct >= SEVERE_ROW_MISSINGNESS_PCT).sum())
    checks.append(
        {
            "check": f"No columns ≥ {SEVERE_COLUMN_MISSINGNESS_PCT}% missing & no rows ≥ {SEVERE_ROW_MISSINGNESS_PCT}% missing (all columns)",
            "type": "severe_missingness",
            "severe_columns": severe_cols,
            "severe_row_count": severe_rows,
            "severe_row_pct": _pct(severe_rows, n),
            "status": "PASS" if (not severe_cols and severe_rows == 0) else "WARN",
        }
    )

    # 2-f  FRED macro series coverage
    fred_cols = ["UMCSENT", "RSXFS", "PCE"]
    fred_coverage: dict[str, Any] = {}
    for col in fred_cols:
        if col in df.columns:
            null_p = round(100 * df[col].isna().mean(), 2)
            fred_coverage[col] = {
                "present": True,
                "null_pct": null_p,
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
        else:
            fred_coverage[col] = {"present": False, "null_pct": 100.0}
    all_fred_ok = all(v["present"] and v["null_pct"] < 5 for v in fred_coverage.values())
    checks.append(
        {
            "check": "FRED macro series fully populated",
            "type": "external_coverage",
            "fred_column_coverage": fred_coverage,
            "status": _pf(all_fred_ok),
        }
    )

    # Overall completeness metrics
    total_missing = int(df.isna().sum().sum())
    overall_completeness = round(100 * (1 - total_missing / df.size), 2)

    report = _dimension_summary("Completeness", checks)
    report["overall_missing_cells"] = total_missing
    report["overall_completeness_pct"] = overall_completeness
    logger.info("  Completeness: {}/{} checks passed", report["passed"], report["total_checks"])
    return report
