from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.utils.logger import logger
from src.validation.common import _iqr_bounds, _pct, _pf
from src.validation.constants import (IQR_MULTIPLIER, OUTLIER_THRESHOLD_PCT,
                                      ZSCORE_THRESHOLD)


# 5 ─ OUTLIER DETECTION
def check_outliers(df: pd.DataFrame) -> dict[str, Any]:
    logger.info("Running [5] OUTLIER DETECTION …")
    col_results: dict[str, Any] = {}

    outlier_cols = [
        "Weekly_Sales",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "Size",
        "MarkDown1",
        "MarkDown2",
        "MarkDown3",
        "MarkDown4",
        "MarkDown5",
    ]

    for col in outlier_cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue

        lb, ub = _iqr_bounds(s, IQR_MULTIPLIER)
        iqr_mask = (s < lb) | (s > ub)
        iqr_count = int(iqr_mask.sum())
        iqr_pct = _pct(iqr_count, len(s))

        z = np.abs(scipy_stats.zscore(s))
        z_mask = z > ZSCORE_THRESHOLD
        z_count = int(z_mask.sum())
        z_pct = _pct(z_count, len(s))

        iqr_pass = iqr_pct <= OUTLIER_THRESHOLD_PCT
        z_pass = z_pct <= OUTLIER_THRESHOLD_PCT

        skewness = float(s.skew())
        is_skewed = abs(skewness) > 1.0

        if is_skewed:
            overall_pass = iqr_pass or z_pass
        else:
            overall_pass = iqr_pass and z_pass

        col_results[col] = {
            "n_non_null": len(s),
            "skewness": round(skewness, 4),
            "is_skewed": is_skewed,
            "iqr_method": {
                "lower_fence": round(float(lb), 4),
                "upper_fence": round(float(ub), 4),
                "outlier_count": iqr_count,
                "outlier_pct": iqr_pct,
                "status": _pf(iqr_pass),
            },
            "zscore_method": {
                "threshold": ZSCORE_THRESHOLD,
                "outlier_count": z_count,
                "outlier_pct": z_pct,
                "status": _pf(z_pass),
            },
            "sample_outlier_values_iqr": sorted(
                [round(float(x), 2) for x in s[iqr_mask].unique()[:10]]
            ),
            "evaluation_mode": (
                "lenient (skewed)" if is_skewed else "strict (symmetric)"
            ),
            "overall_status": _pf(overall_pass),
        }

    total = len(col_results)
    passed = sum(1 for v in col_results.values() if v["overall_status"] == "PASS")

    report = {
        "dimension": "Outlier Detection",
        "methods_used": [
            f"IQR (Tukey fences, multiplier={IQR_MULTIPLIER})",
            f"Z-score (threshold={ZSCORE_THRESHOLD})",
        ],
        "evaluation_policy": (
            "Symmetric columns: FAIL if EITHER method exceeds threshold. "
            "Skewed columns (|skew| > 1): FAIL only if BOTH methods exceed threshold."
        ),
        "outlier_flag_threshold_pct": OUTLIER_THRESHOLD_PCT,
        "columns": col_results,
        "total_checks": total,
        "passed": passed,
        "failed": total - passed,
        "warnings": 0,
        "skipped": 0,
        "overall_status": _pf(passed == total),
        "checks": [],
    }

    logger.info("  Outlier Detection: {}/{} columns within threshold", passed, total)
    return report
