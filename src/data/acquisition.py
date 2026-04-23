"""
Data Acquisition Module
=======================
Handles loading Walmart Kaggle CSVs and fetching FRED macroeconomic
indicators via the FRED REST API.

Sources
-------
1. Kaggle  : Walmart Store Sales Forecasting
             https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data
2. FRED API: St. Louis Federal Reserve Economic Data
             https://fred.stlouisfed.org/docs/api/fred/
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.utils.logger import logger

# ── Environment ────────────────────────────────────────────────────────────────
load_dotenv()

FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
RAW_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# FRED series to pull — all relevant to retail spending 2010–2012
FRED_SERIES: dict[str, str] = {
    "UMCSENT": "University of Michigan Consumer Sentiment Index",
    "RSXFS": "Advance Real Retail and Food Services Sales (Millions USD)",
    "PCE": "Personal Consumption Expenditures (Billions USD)",
}

WALMART_DATE_START = "2010-02-05"
WALMART_DATE_END = "2012-11-02"


# ── Kaggle Data Loader ─────────────────────────────────────────────────────────


def load_walmart_data() -> pd.DataFrame:
    """
    Load and merge the three Walmart Kaggle CSV files:
      - train.csv      : weekly sales per store/department
      - stores.csv     : store type and size metadata
      - features.csv   : external features (temperature, fuel price, CPI, etc.)

    Returns
    -------
    pd.DataFrame
        Merged Walmart DataFrame with parsed dates.

    Raises
    ------
    FileNotFoundError
        If any of the three required CSV files are missing from data/raw/.
    """
    logger.info("Loading Walmart Kaggle data from: {}", RAW_DIR)

    required_files = {
        "train": RAW_DIR / "train.csv",
        "stores": RAW_DIR / "stores.csv",
        "features": RAW_DIR / "features.csv",
    }

    for name, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}\n"
                f"Download from: https://www.kaggle.com/competitions/"
                f"walmart-recruiting-store-sales-forecasting/data"
            )

    # Load individual files
    train_df = pd.read_csv(required_files["train"], parse_dates=["Date"])
    stores_df = pd.read_csv(required_files["stores"])
    features_df = pd.read_csv(required_files["features"], parse_dates=["Date"])

    logger.info(
        "Raw shapes — train: {}, stores: {}, features: {}",
        train_df.shape,
        stores_df.shape,
        features_df.shape,
    )

    # Merge train + stores on Store
    df = train_df.merge(stores_df, on="Store", how="left")
    logger.info("After merging stores: {}", df.shape)

    # Merge with features on Store + Date
    df = df.merge(features_df, on=["Store", "Date"], how="left", suffixes=("", "_feat"))

    # Drop duplicate IsHoliday column introduced by features merge
    holiday_cols = [c for c in df.columns if c.startswith("IsHoliday")]
    if len(holiday_cols) > 1:
        df.drop(columns=["IsHoliday_feat"], inplace=True, errors="ignore")

    logger.info("After merging features: {} rows, {} columns", *df.shape)

    return df


# ── FRED API Client ────────────────────────────────────────────────────────────


def fetch_fred_series(series_id: str, retries: int = 3) -> pd.DataFrame:
    """
    Fetch a single FRED time series as a DataFrame.

    Parameters
    ----------
    series_id : str
        FRED series identifier (e.g., 'UMCSENT').
    retries : int
        Number of retry attempts on network failure.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Date', series_id] filtered to Walmart date range.

    Raises
    ------
    EnvironmentError
        If FRED_API_KEY is not set.
    requests.HTTPError
        If the API returns a non-200 status code.
    """
    if not FRED_API_KEY or FRED_API_KEY == "your_fred_api_key_here":
        raise EnvironmentError(
            "FRED_API_KEY is not set. Please add it to your .env file.\n"
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": WALMART_DATE_START,
        "observation_end": WALMART_DATE_END,
    }

    logger.info(
        "Fetching FRED series: {} ({})", series_id, FRED_SERIES.get(series_id, "")
    )

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(FRED_BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            logger.warning(
                "Attempt {}/{} failed for {}: {}", attempt, retries, series_id, exc
            )
            if attempt == retries:
                raise
            time.sleep(2**attempt)  # Exponential back-off

    observations = response.json().get("observations", [])
    if not observations:
        logger.warning("No observations returned for series: {}", series_id)
        return pd.DataFrame(columns=["Date", series_id])

    df = pd.DataFrame(observations)[["date", "value"]].copy()
    df.rename(columns={"date": "Date", "value": series_id}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])

    # FRED uses '.' for missing values — replace with NaN
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")

    logger.info(
        "Series {} fetched: {} observations, {} missing",
        series_id,
        len(df),
        df[series_id].isna().sum(),
    )

    return df


def fetch_all_fred_series() -> pd.DataFrame:
    """
    Fetch all configured FRED series and combine into a single monthly DataFrame.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame indexed by Date with one column per FRED series.
    """
    logger.info(
        "Fetching {} FRED series: {}", len(FRED_SERIES), list(FRED_SERIES.keys())
    )

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
    return combined


# ── Merge Strategy ─────────────────────────────────────────────────────────────


def merge_walmart_fred(walmart_df: pd.DataFrame, fred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge weekly Walmart data with monthly FRED data using pd.merge_asof().

    Strategy
    --------
    Since FRED data is monthly (first of month) and Walmart data is weekly,
    we use a backward merge: each Walmart week is assigned the economic value
    from the most recent preceding FRED observation date.

    This correctly reflects the information available to a retailer at that
    point in time (no look-ahead bias).

    Parameters
    ----------
    walmart_df : pd.DataFrame
        Merged Walmart DataFrame with 'Date' column.
    fred_df : pd.DataFrame
        FRED macroeconomic DataFrame with 'Date' column.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with FRED columns appended to Walmart rows.
    """
    logger.info(
        "Merging Walmart ({} rows) with FRED ({} rows) using merge_asof",
        len(walmart_df),
        len(fred_df),
    )

    # Both must be sorted by Date for merge_asof
    walmart_sorted = walmart_df.sort_values("Date").reset_index(drop=True)
    fred_sorted = fred_df.sort_values("Date").reset_index(drop=True)

    merged = pd.merge_asof(
        walmart_sorted,
        fred_sorted,
        on="Date",
        direction="backward",  # assign the most recent past FRED value
    )

    rows_before = len(walmart_df)
    rows_after = len(merged)

    logger.info(
        "Merge complete — rows before: {}, rows after: {} (delta: {})",
        rows_before,
        rows_after,
        rows_after - rows_before,
    )

    # Validate no Walmart rows were dropped
    assert (
        rows_after == rows_before
    ), f"Row count mismatch after merge: expected {rows_before}, got {rows_after}"

    return merged


# ── Target Variable Engineering ────────────────────────────────────────────────


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary classification target: Sales_Class.

    Definition
    ----------
    For each store, compute the store-specific median of Weekly_Sales.
    Weeks where Weekly_Sales > store median are labelled 1 (High).
    Weeks where Weekly_Sales <= store median are labelled 0 (Low).

    This store-relative threshold avoids bias where large stores always appear
    'High' and small stores always appear 'Low' in absolute terms.

    Parameters
    ----------
    df : pd.DataFrame
        Merged DataFrame containing 'Store' and 'Weekly_Sales' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with new column 'Sales_Class' (0 = Low, 1 = High).
    """
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


# ── Main Pipeline Entry Point ──────────────────────────────────────────────────


def run_acquisition_pipeline() -> pd.DataFrame:
    """
    Execute the full data acquisition and integration pipeline.

    Steps
    -----
    1. Load Walmart Kaggle CSVs (train + stores + features).
    2. Fetch FRED macroeconomic series via REST API.
    3. Merge weekly Walmart data with monthly FRED data using merge_asof.
    4. Create binary target variable (store-specific median split).
    5. Save merged dataset to data/processed/merged_dataset.csv.

    Returns
    -------
    pd.DataFrame
        Final merged DataFrame ready for validation and modelling.
    """
    logger.info("=" * 60)
    logger.info("Starting Data Acquisition Pipeline")
    logger.info("=" * 60)

    # Step 1: Walmart data
    walmart_df = load_walmart_data()

    # Step 2: FRED data
    fred_df = fetch_all_fred_series()

    # Step 3: Merge
    merged_df = merge_walmart_fred(walmart_df, fred_df)

    # Step 4: Target variable
    merged_df = create_target_variable(merged_df)

    # Step 5: Save
    output_path = PROCESSED_DIR / "merged_dataset.csv"
    merged_df.to_csv(output_path, index=False)
    logger.info(
        "Merged dataset saved to: {} ({:,} rows, {} cols)",
        output_path,
        *merged_df.shape,
    )

    return merged_df


if __name__ == "__main__":
    run_acquisition_pipeline()
