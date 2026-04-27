import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from src.utils.logger import logger

load_dotenv()

FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
RAW_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_DIR = PROCESSED_DIR / "intermediate"
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
INTEGRATION_REPORT_PATH = PROCESSED_DIR / "integration_report.txt"

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

FRED_SERIES: dict[str, str] = {
    "UMCSENT": "University of Michigan Consumer Sentiment Index",
    "RSXFS": "Advance Real Retail and Food Services Sales (Millions USD)",
    "PCE": "Personal Consumption Expenditures (Billions USD)",
}

WALMART_DATE_START = "2010-02-05"
WALMART_DATE_END = "2012-11-02"


def _save_intermediate(df: pd.DataFrame, filename: str) -> Path:
    path = INTERMEDIATE_DIR / filename
    df.to_csv(path, index=False)
    logger.info("Saved intermediate dataset: {} ({:,} rows, {} cols)", path, *df.shape)
    return path


def _log_standard_merge(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    *,
    step_name: str,
    keys: list[str],
    right_added_cols: list[str],
    indicator_col: str,
    strategy: str,
) -> dict[str, Any]:
    rows_before = len(left_df)
    rows_after = len(merged_df)
    unmatched_left_rows = int((merged_df[indicator_col] == "left_only").sum())

    rows_with_any_new_nulls = 0
    null_introduced_by_column: dict[str, int] = {}
    for col in right_added_cols:
        if col in merged_df.columns:
            null_count = int(merged_df[col].isna().sum())
            null_introduced_by_column[col] = null_count

    if right_added_cols:
        rows_with_any_new_nulls = int(merged_df[right_added_cols].isna().any(axis=1).sum())

    logger.info(
        "{} — rows before: {:,}, rows after: {:,}, delta: {:+,}",
        step_name,
        rows_before,
        rows_after,
        rows_after - rows_before,
    )
    logger.info(
        "{} — unmatched left rows: {:,}, rows with nulls in added cols: {:,}",
        step_name,
        unmatched_left_rows,
        rows_with_any_new_nulls,
    )

    return {
        "step": step_name,
        "left_rows": rows_before,
        "right_rows": len(right_df),
        "output_rows": rows_after,
        "keys": keys,
        "strategy": strategy,
        "unmatched_left_rows": unmatched_left_rows,
        "rows_with_any_new_nulls": rows_with_any_new_nulls,
        "null_introduced_by_column": null_introduced_by_column,
    }


def _save_integration_report(report: dict[str, Any]) -> None:
    lines = [
        "=" * 80,
        "WALMART SALES CLASSIFICATION — DATA ACQUISITION & INTEGRATION REPORT",
        "=" * 80,
        "",
        "SOURCES:",
        "1) Kaggle Walmart Store Sales Forecasting (train.csv, stores.csv, features.csv)",
        "2) FRED API series: UMCSENT, RSXFS, PCE",
        "",
        "MERGE STRATEGY (EXACT):",
        "- Step 1: LEFT JOIN train + stores on key [Store]",
        "- Step 2: LEFT JOIN (step1) + features on keys [Store, Date]",
        "- Step 3: AS-OF BACKWARD merge weekly Walmart with monthly FRED on [Date]",
        "  Each Walmart row receives the most recent previous FRED observation",
        "  to avoid look-ahead bias.",
        "",
    ]

    for merge in report.get("merge_steps", []):
        lines.extend(
            [
                f"[{merge['step']}]",
                f"  strategy                 : {merge['strategy']}",
                f"  keys                     : {merge['keys']}",
                f"  left rows                : {merge['left_rows']:,}",
                f"  right rows               : {merge['right_rows']:,}",
                f"  output rows              : {merge['output_rows']:,}",
                f"  unmatched left rows      : {merge['unmatched_left_rows']:,}",
                f"  rows with new nulls      : {merge['rows_with_any_new_nulls']:,}",
                f"  nulls by added column    : {merge['null_introduced_by_column']}",
                "",
            ]
        )

    lines.extend(
        [
            "INTERMEDIATE DATASETS SAVED:",
        ]
    )
    for artifact in report.get("artifacts", []):
        lines.append(f"- {artifact}")

    lines.extend(
        [
            "",
            f"FINAL DATASET: {report.get('final_output', 'N/A')}",
            "=" * 80,
        ]
    )

    with open(INTEGRATION_REPORT_PATH, "w") as f:
        f.write("\n".join(lines))

    logger.info("Integration report saved to: {}", INTEGRATION_REPORT_PATH)


def load_walmart_data(merge_report: dict[str, Any] | None = None) -> pd.DataFrame:
    logger.info("Loading Walmart Kaggle data from: {}", RAW_DIR)

    required_files = {
        "train": RAW_DIR / "train.csv",
        "stores": RAW_DIR / "stores.csv",
        "features": RAW_DIR / "features.csv",
    }

    for name, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}\nDownload from: https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data"
            )

    train_df = pd.read_csv(required_files["train"], parse_dates=["Date"])
    stores_df = pd.read_csv(required_files["stores"])
    features_df = pd.read_csv(required_files["features"], parse_dates=["Date"])

    logger.info(
        "Raw shapes — train: {}, stores: {}, features: {}",
        train_df.shape,
        stores_df.shape,
        features_df.shape,
    )

    merge_indicator = "_merge_stores"
    df = train_df.merge(stores_df, on="Store", how="left", indicator=merge_indicator)

    store_added_cols = [c for c in stores_df.columns if c not in {"Store"}]
    store_merge_stats = _log_standard_merge(
        train_df,
        stores_df,
        df,
        step_name="Merge 1 — train + stores",
        keys=["Store"],
        right_added_cols=store_added_cols,
        indicator_col=merge_indicator,
        strategy="left",
    )
    df.drop(columns=[merge_indicator], inplace=True)
    _save_intermediate(df, "walmart_train_stores_merged.csv")
    train_stores_df = df.copy()

    merge_indicator = "_merge_features"
    df = df.merge(
        features_df,
        on=["Store", "Date"],
        how="left",
        suffixes=("", "_feat"),
        indicator=merge_indicator,
    )

    holiday_cols = [c for c in df.columns if c.startswith("IsHoliday")]
    if len(holiday_cols) > 1:
        df.drop(columns=["IsHoliday_feat"], inplace=True, errors="ignore")

    logger.info("After merging features: {} rows, {} columns", *df.shape)

    feature_added_cols = [col for col in features_df.columns if col not in {"Store", "Date", "IsHoliday"}]
    feature_merge_stats = _log_standard_merge(
        train_stores_df,
        features_df,
        df,
        step_name="Merge 2 — (train+stores) + features",
        keys=["Store", "Date"],
        right_added_cols=feature_added_cols,
        indicator_col=merge_indicator,
        strategy="left",
    )
    df.drop(columns=[merge_indicator], inplace=True)
    _save_intermediate(df, "walmart_internal_merged.csv")

    if merge_report is not None:
        merge_report.setdefault("merge_steps", []).extend([store_merge_stats, feature_merge_stats])

    return df


def fetch_fred_series(series_id: str, retries: int = 3) -> pd.DataFrame:
    if not FRED_API_KEY or FRED_API_KEY == "your_fred_api_key_here":
        raise EnvironmentError(
            "FRED_API_KEY is not set. Please add it to your .env file.\nGet a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": WALMART_DATE_START,
        "observation_end": WALMART_DATE_END,
    }

    logger.info("Fetching FRED series: {} ({})", series_id, FRED_SERIES.get(series_id, ""))

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(FRED_BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            logger.warning("Attempt {}/{} failed for {}: {}", attempt, retries, series_id, exc)
            if attempt == retries:
                raise
            time.sleep(2**attempt)

    observations = response.json().get("observations", [])
    if not observations:
        logger.warning("No observations returned for series: {}", series_id)
        return pd.DataFrame(columns=["Date", series_id])

    df = pd.DataFrame(observations)[["date", "value"]].copy()
    df.rename(columns={"date": "Date", "value": series_id}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])

    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")

    logger.info(
        "Series {} fetched: {} observations, {} missing",
        series_id,
        len(df),
        df[series_id].isna().sum(),
    )

    return df


def fetch_all_fred_series() -> pd.DataFrame:
    logger.info("Fetching {} FRED series: {}", len(FRED_SERIES), list(FRED_SERIES.keys()))

    combined = None
    for series_id in FRED_SERIES:
        df = fetch_fred_series(series_id)
        if combined is None:
            combined = df
        else:
            combined = combined.merge(df, on="Date", how="outer")

    combined.sort_values("Date", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    logger.info("FRED combined shape: {}", combined.shape)
    _save_intermediate(combined, "fred_combined.csv")
    return combined


def merge_walmart_fred(
    walmart_df: pd.DataFrame,
    fred_df: pd.DataFrame,
    merge_report: dict[str, Any] | None = None,
) -> pd.DataFrame:
    logger.info(
        "Merging Walmart ({} rows) with FRED ({} rows) using merge_asof",
        len(walmart_df),
        len(fred_df),
    )

    walmart_sorted = walmart_df.sort_values("Date").reset_index(drop=True)
    fred_sorted = fred_df.sort_values("Date").reset_index(drop=True)

    merged = pd.merge_asof(
        walmart_sorted,
        fred_sorted,
        on="Date",
        direction="backward",
    )

    rows_before = len(walmart_df)
    rows_after = len(merged)

    logger.info(
        "Merge complete — rows before: {}, rows after: {} (delta: {})",
        rows_before,
        rows_after,
        rows_after - rows_before,
    )

    fred_cols = [c for c in fred_df.columns if c != "Date"]
    unmatched_rows = int(merged[fred_cols].isna().all(axis=1).sum()) if fred_cols else 0
    rows_with_any_fred_null = int(merged[fred_cols].isna().any(axis=1).sum()) if fred_cols else 0
    null_introduced_by_column = {col: int(merged[col].isna().sum()) for col in fred_cols if col in merged.columns}

    logger.info(
        "Merge 3 — unmatched Walmart rows (no FRED match): {:,}",
        unmatched_rows,
    )
    logger.info(
        "Merge 3 — rows with nulls in any FRED column: {:,}",
        rows_with_any_fred_null,
    )

    assert rows_after == rows_before, f"Row count mismatch after merge: expected {rows_before}, got {rows_after}"

    if merge_report is not None:
        merge_report.setdefault("merge_steps", []).append(
            {
                "step": "Merge 3 — Walmart + FRED (asof)",
                "left_rows": rows_before,
                "right_rows": len(fred_df),
                "output_rows": rows_after,
                "keys": ["Date"],
                "strategy": "merge_asof(direction='backward')",
                "unmatched_left_rows": unmatched_rows,
                "rows_with_any_new_nulls": rows_with_any_fred_null,
                "null_introduced_by_column": null_introduced_by_column,
            }
        )

    _save_intermediate(merged, "walmart_fred_merged.csv")

    return merged


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating store-specific median target variable: Sales_Class")

    store_medians = df.groupby("Store")["Weekly_Sales"].transform("median")
    df["Sales_Class"] = (df["Weekly_Sales"] > store_medians).astype(int)

    distribution = df["Sales_Class"].value_counts()
    logger.info(
        "Target distribution — High (1): {:,} ({:.1f}%), Low (0): {:,} ({:.1f}%)",
        distribution.get(1, 0),
        100 * distribution.get(1, 0) / len(df),
        distribution.get(0, 0),
        100 * distribution.get(0, 0) / len(df),
    )

    return df


def run_acquisition_pipeline() -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("Starting Data Acquisition Pipeline")
    logger.info("=" * 60)

    report: dict[str, Any] = {"merge_steps": [], "artifacts": []}

    walmart_df = load_walmart_data(merge_report=report)
    report["artifacts"].append(str(INTERMEDIATE_DIR / "walmart_train_stores_merged.csv"))
    report["artifacts"].append(str(INTERMEDIATE_DIR / "walmart_internal_merged.csv"))

    fred_df = fetch_all_fred_series()
    report["artifacts"].append(str(INTERMEDIATE_DIR / "fred_combined.csv"))

    merged_df = merge_walmart_fred(walmart_df, fred_df, merge_report=report)
    report["artifacts"].append(str(INTERMEDIATE_DIR / "walmart_fred_merged.csv"))

    merged_df = create_target_variable(merged_df)

    output_path = PROCESSED_DIR / "merged_dataset.csv"
    merged_df.to_csv(output_path, index=False)
    logger.info(
        "Merged dataset saved to: {} ({:,} rows, {} cols)",
        output_path,
        *merged_df.shape,
    )

    report["final_output"] = str(output_path)
    _save_integration_report(report)

    return merged_df


if __name__ == "__main__":
    run_acquisition_pipeline()
