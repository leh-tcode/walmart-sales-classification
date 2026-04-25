from typing import Any

import pandas as pd
from scipy import stats as scipy_stats

from src.utils.logger import logger
from src.validation.common import _pf
from src.validation.constants import (
    EXPECTED_NEGATIVE_CORRELATIONS,
    EXPECTED_POSITIVE_CORRELATIONS,
    NUMERIC_COLS,
)


# 7 ─ RELATIONSHIPS
def check_relationships(df: pd.DataFrame) -> dict[str, Any]:
    logger.info("Running [7] RELATIONSHIPS …")

    avail_numeric = [c for c in NUMERIC_COLS if c in df.columns]
    numeric_df = df[avail_numeric].dropna()

    if len(numeric_df) == 0:
        report = {
            "dimension": "Relationships",
            "error": "No complete numeric rows for correlation analysis",
            "total_checks": 0, "passed": 0, "failed": 0,
            "warnings": 0, "skipped": 0,
            "overall_status": "SKIP", "checks": [],
        }
        return report

    pair_results: list[dict] = []

    def _check_pair(col_a, col_b, expected_direction):
        if col_a not in numeric_df or col_b not in numeric_df:
            return
        r, p = scipy_stats.pearsonr(numeric_df[col_a], numeric_df[col_b])
        sr, sp = scipy_stats.spearmanr(numeric_df[col_a], numeric_df[col_b])
        if expected_direction == "positive":
            ok = r > 0
        else:
            ok = r < 0
        pair_results.append({
            "check": f"{col_a} vs {col_b} ({expected_direction})",
            "pair": f"{col_a} vs {col_b}",
            "expected": f"{expected_direction}_correlation",
            "pearson_r": round(float(r), 4),
            "pearson_p": round(float(p), 6),
            "spearman_r": round(float(sr), 4),
            "spearman_p": round(float(sp), 6),
            "strength": (
                "strong" if abs(r) > 0.6 else
                "moderate" if abs(r) > 0.3 else "weak"
            ),
            "statistically_significant": bool(p < 0.05),
            "status": _pf(ok),
        })

    for col_a, col_b in EXPECTED_POSITIVE_CORRELATIONS:
        _check_pair(col_a, col_b, "positive")

    for col_a, col_b in EXPECTED_NEGATIVE_CORRELATIONS:
        _check_pair(col_a, col_b, "negative")

    target = "Weekly_Sales"
    target_corrs: dict[str, Any] = {}
    if target in numeric_df.columns:
        for col in avail_numeric:
            if col == target:
                continue
            r, p = scipy_stats.pearsonr(numeric_df[col], numeric_df[target])
            sr, sp = scipy_stats.spearmanr(numeric_df[col], numeric_df[target])
            target_corrs[col] = {
                "pearson_r": round(float(r), 4),
                "pearson_p": round(float(p), 6),
                "spearman_r": round(float(sr), 4),
                "spearman_p": round(float(sp), 6),
                "significant": bool(p < 0.05),
                "direction": "positive" if r > 0 else "negative",
                "strength": (
                    "strong" if abs(r) > 0.6 else
                    "moderate" if abs(r) > 0.3 else "weak"
                ),
            }

    corr_matrix = numeric_df.corr(method="pearson").round(4).to_dict()

    passed = sum(1 for p in pair_results if p["status"] == "PASS")
    report = {
        "dimension": "Relationships",
        "pairwise_checks": {
            "results": pair_results,
            "total": len(pair_results),
            "passed": passed,
            "failed": len(pair_results) - passed,
        },
        "target_feature_correlations": target_corrs,
        "pearson_correlation_matrix": corr_matrix,
        "total_checks": len(pair_results),
        "passed": passed,
        "failed": len(pair_results) - passed,
        "warnings": 0,
        "skipped": 0,
        "overall_status": _pf(passed == len(pair_results)),
        "checks": pair_results,
    }

    logger.info("  Relationships: {}/{} directional checks passed", passed, len(pair_results))
    return report
