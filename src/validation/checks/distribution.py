from typing import Any

import pandas as pd

from src.utils.logger import logger
from src.validation.common import _iqr_bounds, _pct, _pf
from src.validation.constants import NUMERIC_COLS


# 6 ─ DISTRIBUTION PROFILE
def check_distribution_profile(df: pd.DataFrame) -> dict[str, Any]:
    logger.info("Running [6] DISTRIBUTION PROFILE …")
    sanity_checks: list[dict] = []

    # ── Numeric profiles ──
    numeric_profile: dict[str, Any] = {}
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        lb, ub = _iqr_bounds(s)
        iqr_out = int(((s < lb) | (s > ub)).sum())
        numeric_profile[col] = {
            "count": int(len(s)),
            "null_pct": _pct(int(df[col].isna().sum()), len(df)),
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "q1": round(float(s.quantile(0.25)), 4),
            "q3": round(float(s.quantile(0.75)), 4),
            "skewness": round(float(s.skew()), 4),
            "kurtosis": round(float(s.kurtosis()), 4),
            "distribution_shape": (
                "heavily_right_skewed"
                if s.skew() > 2
                else (
                    "right_skewed"
                    if s.skew() > 1
                    else "left_skewed" if s.skew() < -1 else "approximately_symmetric"
                )
            ),
            "iqr_outlier_count": iqr_out,
            "iqr_outlier_pct": _pct(iqr_out, len(s)),
        }

    # ── Categorical profiles ──
    categorical_profile: dict[str, Any] = {}
    for col in ["Type", "IsHoliday"]:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False)
        top_pct = _pct(int(vc.iloc[0]), len(df)) if len(vc) else 0
        categorical_profile[col] = {
            "unique_values": int(df[col].nunique(dropna=True)),
            "top_10_values": {str(k): int(v) for k, v in vc.head(10).items()},
            "dominant_value_pct": top_pct,
            "balance_flag": (
                "highly_imbalanced"
                if top_pct > 70
                else "imbalanced" if top_pct > 50 else "balanced"
            ),
        }

    if "Weekly_Sales" in df.columns:
        median_sales = float(df["Weekly_Sales"].dropna().median())
        sanity_checks.append(
            {
                "check": "Median Weekly_Sales in [5 000, 30 000]",
                "value": round(median_sales, 2),
                "status": _pf(5_000 <= median_sales <= 30_000),
            }
        )

    if "Sales_Class" in df.columns:
        dist = df["Sales_Class"].value_counts()
        if len(dist) >= 2:
            imbalance = round(dist.max() / dist.min(), 3)
            sanity_checks.append(
                {
                    "check": "Class imbalance ratio < 2.0",
                    "imbalance_ratio": imbalance,
                    "class_counts": dist.to_dict(),
                    "class_pct": (df["Sales_Class"].value_counts(normalize=True) * 100)
                    .round(2)
                    .to_dict(),
                    "is_balanced": imbalance < 1.5,
                    "status": _pf(imbalance < 2.0),
                }
            )

    if "Temperature" in df.columns:
        med_temp = float(df["Temperature"].dropna().median())
        sanity_checks.append(
            {
                "check": "Median Temperature in [40, 80] °F",
                "value": round(med_temp, 2),
                "status": _pf(40 <= med_temp <= 80),
            }
        )

    if "Unemployment" in df.columns:
        med_unemp = float(df["Unemployment"].dropna().median())
        sanity_checks.append(
            {
                "check": "Median Unemployment in [4, 12]%",
                "value": round(med_unemp, 2),
                "status": _pf(4 <= med_unemp <= 12),
            }
        )

    if "Type" in df.columns:
        most_common = df["Type"].mode()[0]
        sanity_checks.append(
            {
                "check": "Most common store Type is A",
                "value": most_common,
                "status": _pf(most_common == "A"),
                "note": "Type A stores dominate Walmart dataset",
            }
        )

    passed = sum(1 for c in sanity_checks if c["status"] == "PASS")
    report = {
        "dimension": "Distribution Profile",
        "numeric_profiles": numeric_profile,
        "categorical_profiles": categorical_profile,
        "sanity_checks": sanity_checks,
        "total_checks": len(sanity_checks),
        "passed": passed,
        "failed": len(sanity_checks) - passed,
        "warnings": 0,
        "skipped": 0,
        "overall_status": _pf(passed == len(sanity_checks)),
        "checks": sanity_checks,
    }

    logger.info(
        "  Distribution Profile: {}/{} sanity checks passed", passed, len(sanity_checks)
    )
    return report
