import pandas as pd


# Helpers
def _pf(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def _pct(count: int, total: int) -> float:
    return round(count / total * 100, 3) if total else 0.0


def _iqr_bounds(series: pd.Series, multiplier: float = 1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - multiplier * iqr, q3 + multiplier * iqr


def _dtype_matches(series: pd.Series, expected: str) -> bool:
    if expected == "integer":
        return pd.api.types.is_integer_dtype(series)
    if expected == "numeric":
        return pd.api.types.is_numeric_dtype(series)
    if expected == "datetime":
        return pd.api.types.is_datetime64_any_dtype(series)
    if expected == "bool":
        return pd.api.types.is_bool_dtype(series)
    if expected == "string":
        return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(
            series
        )
    return False


def _dimension_summary(dimension: str, checks: list[dict]) -> dict:
    passed = sum(1 for c in checks if c.get("status") == "PASS")
    warned = sum(1 for c in checks if c.get("status") == "WARN")
    failed = sum(1 for c in checks if c.get("status") == "FAIL")
    skipped = sum(1 for c in checks if c.get("status") == "SKIP")
    overall = (
        "PASS" if failed == 0 and warned == 0 else ("WARN" if failed == 0 else "FAIL")
    )
    return {
        "dimension": dimension,
        "total_checks": len(checks),
        "passed": passed,
        "warnings": warned,
        "failed": failed,
        "skipped": skipped,
        "overall_status": overall,
        "checks": checks,
    }
