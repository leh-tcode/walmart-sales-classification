"""
Unit Tests — Data Validation Module
=====================================
Tests cover:
  - Shape check (row/col minimums)
  - Missing value detection
  - Duplicate detection
  - Date range validation
  - Negative sales detection
  - Class distribution check
  - FRED coverage check
  - Referential integrity check
"""

import numpy as np
import pandas as pd
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def clean_df():
    """A minimal clean DataFrame that passes all validation checks."""
    n = 6000
    rng = np.random.default_rng(42)
    stores = rng.integers(1, 46, size=n)
    dates = pd.date_range("2010-02-05", periods=n, freq="W")
    sales = rng.uniform(5000, 80000, size=n)
    return pd.DataFrame(
        {
            "Store": stores,
            "Dept": rng.integers(1, 72, size=n),
            "Date": dates,
            "Weekly_Sales": sales,
            "IsHoliday": rng.choice([True, False], size=n),
            "Type": rng.choice(["A", "B", "C"], size=n),
            "Size": rng.integers(30000, 200000, size=n),
            "Temperature": rng.uniform(20, 90, size=n),
            "Fuel_Price": rng.uniform(2.5, 4.5, size=n),
            "CPI": rng.uniform(200, 230, size=n),
            "Unemployment": rng.uniform(5, 12, size=n),
            "UMCSENT": rng.uniform(60, 95, size=n),
            "RSXFS": rng.uniform(300000, 400000, size=n),
            "PCE": rng.uniform(9500, 11000, size=n),
            "Sales_Class": (sales > pd.Series(sales).median()).astype(int),
        }
    )


# ── Shape Tests ───────────────────────────────────────────────────────────────


class TestCheckShape:

    def test_passes_with_sufficient_rows_and_cols(self, clean_df):
        from src.validation.validator import check_shape

        result = check_shape(clean_df)
        assert result["status"] == "PASS"

    def test_fails_with_too_few_rows(self, clean_df):
        from src.validation.validator import check_shape

        small_df = clean_df.head(100)
        result = check_shape(small_df)
        assert result["status"] == "FAIL"
        assert result["meets_min_rows"] is False

    def test_fails_with_too_few_columns(self, clean_df):
        from src.validation.validator import check_shape

        narrow_df = clean_df[["Store", "Date", "Weekly_Sales"]]
        result = check_shape(narrow_df)
        assert result["status"] == "FAIL"
        assert result["meets_min_cols"] is False


# ── Missing Values Tests ───────────────────────────────────────────────────────


class TestCheckMissingValues:

    def test_pass_on_complete_data(self, clean_df):
        from src.validation.validator import check_missing_values

        result = check_missing_values(clean_df)
        assert result["status"] == "PASS"
        assert result["total_missing_cells"] == 0

    def test_warns_on_missing_data(self, clean_df):
        from src.validation.validator import check_missing_values

        df_with_missing = clean_df.copy()
        df_with_missing.loc[0:50, "UMCSENT"] = np.nan
        result = check_missing_values(df_with_missing)
        assert result["status"] == "WARN"
        assert result["total_missing_cells"] > 0

    def test_counts_correctly(self, clean_df):
        from src.validation.validator import check_missing_values

        df = clean_df.copy()
        df.loc[0:9, "Weekly_Sales"] = np.nan  # 10 missing
        result = check_missing_values(df)
        assert result["total_missing_cells"] == 10


# ── Duplicate Tests ────────────────────────────────────────────────────────────


class TestCheckDuplicates:

    def test_pass_on_unique_data(self, clean_df):
        from src.validation.validator import check_duplicates

        result = check_duplicates(clean_df)
        assert result["status"] == "PASS"
        assert result["full_row_duplicates"] == 0

    def test_detects_full_row_duplicates(self, clean_df):
        from src.validation.validator import check_duplicates

        df_with_dups = pd.concat([clean_df, clean_df.head(5)], ignore_index=True)
        result = check_duplicates(df_with_dups)
        assert result["status"] == "WARN"
        assert result["full_row_duplicates"] == 5


# ── Date Range Tests ───────────────────────────────────────────────────────────


class TestCheckDateRange:

    def test_pass_on_valid_dates(self):
        """Use a dedicated fixture with dates strictly within 2010-2012."""
        from src.validation.validator import check_date_range
        import numpy as np

        rng = np.random.default_rng(42)
        # 143 weeks fits exactly within 2010-02-05 to 2012-12-28
        dates = pd.date_range("2010-02-05", periods=143, freq="W")
        df = pd.DataFrame(
            {
                "Date": np.resize(dates, 6000),
                "Weekly_Sales": rng.uniform(5000, 80000, size=6000),
            }
        )
        result = check_date_range(df)
        assert result["status"] == "PASS"
        assert result["out_of_range_rows"] == 0

    def test_skips_without_date_column(self, clean_df):
        from src.validation.validator import check_date_range

        df_no_date = clean_df.drop(columns=["Date"])
        result = check_date_range(df_no_date)
        assert result["status"] == "SKIP"

    def test_warns_on_out_of_range_dates(self, clean_df):
        from src.validation.validator import check_date_range

        df = clean_df.copy()
        df.loc[0, "Date"] = pd.Timestamp("2005-01-01")
        result = check_date_range(df)
        assert result["out_of_range_rows"] >= 1


# ── Negative Sales Tests ───────────────────────────────────────────────────────


class TestCheckNegativeSales:

    def test_pass_on_positive_sales(self, clean_df):
        from src.validation.validator import check_negative_sales

        result = check_negative_sales(clean_df)
        assert result["status"] == "PASS"

    def test_warns_on_negative_sales(self, clean_df):
        from src.validation.validator import check_negative_sales

        df = clean_df.copy()
        df.loc[0:4, "Weekly_Sales"] = -500.0
        result = check_negative_sales(df)
        assert result["status"] == "WARN"
        assert result["negative_rows"] == 5


# ── Class Distribution Tests ───────────────────────────────────────────────────


class TestCheckClassDistribution:

    def test_pass_on_balanced_target(self, clean_df):
        from src.validation.validator import check_class_distribution

        result = check_class_distribution(clean_df)
        assert result["status"] == "PASS"
        assert result["imbalance_ratio"] < 2.0

    def test_skip_without_target(self, clean_df):
        from src.validation.validator import check_class_distribution

        df_no_target = clean_df.drop(columns=["Sales_Class"])
        result = check_class_distribution(df_no_target)
        assert result["status"] == "SKIP"


# ── FRED Coverage Tests ────────────────────────────────────────────────────────


class TestCheckFredCoverage:

    def test_pass_with_all_fred_cols(self, clean_df):
        from src.validation.validator import check_fred_coverage

        result = check_fred_coverage(clean_df)
        assert result["all_series_present"] is True

    def test_warns_when_fred_col_missing(self, clean_df):
        from src.validation.validator import check_fred_coverage

        df_missing_fred = clean_df.drop(columns=["UMCSENT"])
        result = check_fred_coverage(df_missing_fred)
        assert result["status"] == "WARN"
        assert result["fred_column_coverage"]["UMCSENT"]["present"] is False


# ── Referential Integrity Tests ────────────────────────────────────────────────


class TestCheckReferentialIntegrity:

    def test_pass_on_valid_store_ids(self, clean_df):
        from src.validation.validator import check_referential_integrity

        result = check_referential_integrity(clean_df)
        assert result["status"] == "PASS"

    def test_warns_on_invalid_store_id(self, clean_df):
        from src.validation.validator import check_referential_integrity

        df = clean_df.copy()
        df.loc[0, "Store"] = 99  # Store 99 doesn't exist in Walmart (max 45)
        result = check_referential_integrity(df)
        assert result["status"] == "WARN"
