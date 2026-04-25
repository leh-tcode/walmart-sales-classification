from typing import Any

import pandas as pd

from src.utils.logger import logger
from src.validation.common import _dimension_summary, _pct, _pf
from src.validation.constants import VALUE_RANGES


# 1 ─ ACCURACY
def check_accuracy(df: pd.DataFrame) -> dict[str, Any]:
    logger.info("Running [1] ACCURACY checks …")
    checks: list[dict] = []

    # 1-a  Range validation for numeric columns
    for col, (lo, hi) in VALUE_RANGES.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        violations = int(((s < lo) | (s > hi)).sum())
        violation_pct = _pct(violations, len(s))
        checks.append(
            {
                "check": f"{col} in [{lo}, {hi}]",
                "column": col,
                "type": "range",
                "valid_min": lo,
                "valid_max": hi,
                "violations": violations,
                "violation_pct": violation_pct,
                "status": _pf(violation_pct <= 1.0),
            }
        )

    # 1-b  Categorical domain checks
    if "Type" in df.columns:
        allowed = {"A", "B", "C"}
        s = df["Type"].dropna().astype(str).str.strip()
        invalid = sorted(set(s.unique()) - allowed)
        checks.append(
            {
                "check": "Type in {A, B, C}",
                "column": "Type",
                "type": "set_membership",
                "valid_set": sorted(allowed),
                "sample_invalid_values": invalid[:10],
                "violations": int((~s.isin(allowed)).sum()),
                "status": _pf(len(invalid) == 0),
            }
        )

    if "IsHoliday" in df.columns:
        unique_vals = set(df["IsHoliday"].dropna().unique())
        invalid_bool = sorted(
            [v for v in unique_vals if v not in {True, False}], key=str
        )
        checks.append(
            {
                "check": "IsHoliday in {True, False}",
                "column": "IsHoliday",
                "type": "set_membership",
                "violations": len(invalid_bool),
                "sample_invalid_values": invalid_bool[:10],
                "status": _pf(len(invalid_bool) == 0),
            }
        )

    # 1-c  Date range validity
    if "Date" in df.columns:
        expected_start = pd.Timestamp("2010-01-01")
        expected_end = pd.Timestamp("2012-12-31")
        out = df[(df["Date"] < expected_start) | (df["Date"] > expected_end)]
        future = df[df["Date"] > pd.Timestamp.today()]
        checks.append(
            {
                "check": "Date within [2010-01-01, 2012-12-31]",
                "column": "Date",
                "type": "range",
                "min_date": str(df["Date"].min().date()),
                "max_date": str(df["Date"].max().date()),
                "out_of_range_rows": len(out),
                "future_date_rows": len(future),
                "status": _pf(len(out) == 0 and len(future) == 0),
            }
        )

    # 1-d  Target validity (Sales_Class must be 0 or 1)
    if "Sales_Class" in df.columns:
        null_count = int(df["Sales_Class"].isna().sum())
        invalid_vals = sorted(set(df["Sales_Class"].dropna().unique()) - {0, 1})
        checks.append(
            {
                "check": "Sales_Class values in {0, 1}",
                "column": "Sales_Class",
                "type": "target_validity",
                "null_count": null_count,
                "invalid_values": invalid_vals,
                "status": _pf(null_count == 0 and len(invalid_vals) == 0),
            }
        )

    # 1-e  Negative sales flagging (domain knowledge: returns are valid)
    if "Weekly_Sales" in df.columns:
        neg = df[df["Weekly_Sales"] < 0]
        pct_neg = _pct(len(neg), len(df))
        checks.append(
            {
                "check": "Negative Weekly_Sales inspection",
                "column": "Weekly_Sales",
                "type": "domain_flag",
                "negative_rows": len(neg),
                "pct_negative": pct_neg,
                "min_value": float(df["Weekly_Sales"].min()),
                "note": "Negative values may reflect returns — retained but flagged",
                "status": "WARN" if len(neg) > 0 else "PASS",
            }
        )

    report = _dimension_summary("Accuracy", checks)
    logger.info(
        "  Accuracy: {}/{} checks passed", report["passed"], report["total_checks"]
    )
    return report
