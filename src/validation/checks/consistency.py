from typing import Any

import pandas as pd

from src.utils.logger import logger
from src.validation.common import _dimension_summary, _dtype_matches, _pct, _pf
from src.validation.constants import STRICT_DTYPE_EXPECTATIONS


# 3 ─ CONSISTENCY
def check_consistency(df: pd.DataFrame) -> dict[str, Any]:
    logger.info("Running [3] CONSISTENCY checks …")
    checks: list[dict] = []

    # 3-a  Strict dtype expectations
    mismatches = {}
    for col, expected in STRICT_DTYPE_EXPECTATIONS.items():
        if col not in df.columns:
            continue
        if not _dtype_matches(df[col], expected):
            mismatches[col] = {"expected": expected, "actual": str(df[col].dtype)}
            checks.append({
                "check": f"{col} dtype is {expected}",
                "column": col, "type": "dtype_consistency",
                "expected_dtype": expected, "actual_dtype": str(df[col].dtype),
                "status": "FAIL",
            })
        else:
            checks.append({
                "check": f"{col} dtype is {expected}",
                "column": col, "type": "dtype_consistency",
                "expected_dtype": expected, "actual_dtype": str(df[col].dtype),
                "status": "PASS",
            })

    # 3-b  Referential integrity — Store IDs in [1, 45], Dept in [1, 99]
    if "Store" in df.columns:
        invalid_stores = [int(s) for s in df["Store"].unique() if not (1 <= s <= 45)]
        checks.append({
            "check": "Store IDs in [1, 45]",
            "column": "Store", "type": "referential_integrity",
            "unique_stores": int(df["Store"].nunique()),
            "invalid_store_ids": invalid_stores[:20],
            "status": _pf(len(invalid_stores) == 0),
        })

    if "Dept" in df.columns:
        invalid_depts = [int(d) for d in df["Dept"].unique() if not (1 <= d <= 99)]
        checks.append({
            "check": "Dept IDs in [1, 99]",
            "column": "Dept", "type": "referential_integrity",
            "unique_depts": int(df["Dept"].nunique()),
            "invalid_dept_count": len(invalid_depts),
            "status": _pf(len(invalid_depts) == 0),
        })

    # 3-c  Cross-column logical rules
    if {"Weekly_Sales", "Size"} <= set(df.columns):
        mask = df["Weekly_Sales"].notna() & df["Size"].notna() & (df["Size"] > 0)
        sub = df.loc[mask]
        sales_per_sqft = sub["Weekly_Sales"] / sub["Size"]
        extreme = int((sales_per_sqft.abs() > 100).sum())
        checks.append({
            "check": "|Weekly_Sales / Size| ≤ 100 (sales-per-sqft sanity)",
            "type": "cross_column_logic",
            "violations": extreme,
            "violation_pct": _pct(extreme, len(sub)),
            "status": _pf(_pct(extreme, len(sub)) <= 2.0),
        })

    if {"Temperature", "Date"} <= set(df.columns):
        summer = df[df["Date"].dt.month.isin([6, 7, 8])]
        frozen_summer = int((summer["Temperature"].dropna() < 0).sum())
        checks.append({
            "check": "No sub-zero temperatures in Jun–Aug",
            "type": "cross_column_logic",
            "violations": frozen_summer,
            "note": "Fahrenheit assumed — sub-zero in summer is suspect",
            "status": _pf(frozen_summer == 0),
        })

    # 3-d  Holiday flag consistency (IsHoliday should be True only on known weeks)
    if "IsHoliday" in df.columns:
        non_bool = int((~df["IsHoliday"].dropna().isin({True, False})).sum())
        checks.append({
            "check": "IsHoliday contains only boolean values",
            "type": "domain_consistency",
            "violations": non_bool,
            "status": _pf(non_bool == 0),
        })

    report = _dimension_summary("Consistency", checks)
    logger.info("  Consistency: {}/{} checks passed", report["passed"], report["total_checks"])
    return report
