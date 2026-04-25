from typing import Any

import pandas as pd

from src.utils.logger import logger
from src.validation.common import _dimension_summary, _pct, _pf


# 4 ─ UNIQUENESS
def check_uniqueness(df: pd.DataFrame) -> dict[str, Any]:
    logger.info("Running [4] UNIQUENESS checks …")
    checks: list[dict] = []
    n = len(df)

    # 4-a  Exact full-row duplicates
    full_dups = int(df.duplicated().sum())
    checks.append({
        "check": "No exact full-row duplicates",
        "type": "exact_duplicate",
        "duplicate_count": full_dups,
        "duplicate_pct": _pct(full_dups, n),
        "action_if_failed": "df.drop_duplicates()",
        "status": _pf(full_dups == 0),
    })

    # 4-b  Business-key duplicates (Store + Dept + Date)
    key_cols = ["Store", "Dept", "Date"]
    if all(c in df.columns for c in key_cols):
        key_dups = int(df.duplicated(subset=key_cols).sum())
        checks.append({
            "check": "Unique business key (Store + Dept + Date)",
            "type": "key_duplicate",
            "columns_used": key_cols,
            "duplicate_count": key_dups,
            "duplicate_pct": _pct(key_dups, n),
            "note": "Each (Store, Dept, Date) should appear exactly once",
            "status": _pf(key_dups == 0),
        })

    # 4-c  Near-duplicate fingerprint
    fp_cols = ["Store", "Dept", "Date", "Weekly_Sales"]
    if all(c in df.columns for c in fp_cols):
        fp_dups = int(df.duplicated(subset=fp_cols).sum())
        checks.append({
            "check": "Unique listing fingerprint (Store+Dept+Date+Sales)",
            "type": "fingerprint_duplicate",
            "columns_used": fp_cols,
            "duplicate_count": fp_dups,
            "duplicate_pct": _pct(fp_dups, n),
            "status": _pf(_pct(fp_dups, n) <= 0.5),
        })

    report = _dimension_summary("Uniqueness", checks)
    logger.info("  Uniqueness: {}/{} checks passed", report["passed"], report["total_checks"])
    return report
