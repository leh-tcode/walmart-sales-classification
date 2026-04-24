import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def clean_df():
    n = 6000
    rng = np.random.default_rng(42)
    stores = rng.integers(1, 46, size=n)
    dates = pd.date_range("2010-02-05", periods=n, freq="W")
    sales = rng.uniform(5000, 80000, size=n)
    medians = pd.Series(sales).groupby(stores).transform("median")
    return pd.DataFrame({
        "Store": stores,
        "Dept": rng.integers(1, 72, size=n),
        "Date": dates,
        "Weekly_Sales": sales,
        "IsHoliday": rng.choice([True, False], size=n),
        "Type": rng.choice(["A", "B", "C"], size=n),
        "Size": rng.integers(30000, 200000, size=n),
        "Temperature": rng.uniform(20, 90, size=n),
        "Fuel_Price": rng.uniform(2.5, 4.5, size=n),
        "MarkDown1": rng.uniform(0, 20000, size=n),
        "MarkDown2": rng.uniform(0, 20000, size=n),
        "MarkDown3": rng.uniform(0, 20000, size=n),
        "MarkDown4": rng.uniform(0, 20000, size=n),
        "MarkDown5": rng.uniform(0, 20000, size=n),
        "CPI": rng.uniform(200, 230, size=n),
        "Unemployment": rng.uniform(5, 12, size=n),
        "UMCSENT": rng.uniform(60, 95, size=n),
        "RSXFS": rng.uniform(300000, 400000, size=n),
        "PCE": rng.uniform(9500, 11000, size=n),
        "Sales_Class": (sales > pd.Series(sales).median()).astype(int),
    })

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


class TestCheckRequiredSchema:

    def test_pass_with_required_columns(self, clean_df):
        from src.validation.validator import check_required_schema
        result = check_required_schema(clean_df)
        assert result["status"] == "PASS"

    def test_fail_when_required_missing(self, clean_df):
        from src.validation.validator import check_required_schema
        df = clean_df.drop(columns=["Weekly_Sales"])
        result = check_required_schema(df)
        assert result["status"] == "FAIL"
        assert "Weekly_Sales" in result["missing_required_columns"]


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
        df.loc[0:9, "Weekly_Sales"] = np.nan
        result = check_missing_values(df)
        assert result["total_missing_cells"] == 10


class TestRowLevelAndSevereMissingness:

    def test_row_level_missingness_warns(self, clean_df):
        from src.validation.validator import check_row_level_missingness
        df = clean_df.copy()
        df.loc[0:20, "UMCSENT"] = np.nan
        result = check_row_level_missingness(df)
        assert result["status"] == "WARN"
        assert result["rows_with_missing"] > 0

    def test_severe_missingness_detects_columns(self, clean_df):
        from src.validation.validator import check_severe_missingness_thresholds
        df = clean_df.copy()
        df.loc[:4000, "PCE"] = np.nan
        result = check_severe_missingness_thresholds(df)
        assert result["status"] == "WARN"
        assert "PCE" in result["severe_columns"]

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


class TestStrictDtypes:

    def test_strict_dtypes_pass(self, clean_df):
        from src.validation.validator import check_strict_dtypes
        result = check_strict_dtypes(clean_df)
        assert result["status"] == "PASS"

    def test_strict_dtypes_fail_on_mismatch(self, clean_df):
        from src.validation.validator import check_strict_dtypes
        df = clean_df.copy()
        df["Store"] = df["Store"].astype(str)
        result = check_strict_dtypes(df)
        assert result["status"] == "FAIL"
        assert "Store" in result["mismatches"]


class TestCheckDateRange:

    def test_pass_on_valid_dates(self):
        from src.validation.validator import check_date_range
        import numpy as np
        rng = np.random.default_rng(42)
        dates = pd.date_range("2010-02-05", periods=143, freq="W")
        df = pd.DataFrame({
            "Date": np.resize(dates, 6000),
            "Weekly_Sales": rng.uniform(5000, 80000, size=6000),
        })
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


class TestTargetAndCategoricalValidity:

    def test_target_validity_pass(self, clean_df):
        from src.validation.validator import check_target_validity
        result = check_target_validity(clean_df)
        assert result["status"] == "PASS"

    def test_target_validity_fail_on_invalid(self, clean_df):
        from src.validation.validator import check_target_validity
        df = clean_df.copy()
        df.loc[0, "Sales_Class"] = 2
        result = check_target_validity(df)
        assert result["status"] == "FAIL"

    def test_categorical_domain_warns(self, clean_df):
        from src.validation.validator import check_categorical_domains
        df = clean_df.copy()
        df.loc[0, "Type"] = "Z"
        result = check_categorical_domains(df)
        assert result["status"] == "WARN"


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


class TestCheckReferentialIntegrity:

    def test_pass_on_valid_store_ids(self, clean_df):
        from src.validation.validator import check_referential_integrity
        result = check_referential_integrity(clean_df)
        assert result["status"] == "PASS"

    def test_warns_on_invalid_store_id(self, clean_df):
        from src.validation.validator import check_referential_integrity
        df = clean_df.copy()
        df.loc[0, "Store"] = 99
        result = check_referential_integrity(df)
        assert result["status"] == "WARN"


class TestValidationOutputs:

    def test_run_validation_writes_json_and_csv(self, clean_df, tmp_path, monkeypatch):
        import src.validation.validator as validator

        monkeypatch.setattr(validator, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(validator, "REPORT_PATH", tmp_path / "validation_report.txt")
        monkeypatch.setattr(validator, "JSON_SUMMARY_PATH", tmp_path / "validation_summary.json")
        monkeypatch.setattr(validator, "CSV_SUMMARY_PATH", tmp_path / "validation_summary.csv")

        validator.run_validation(clean_df)

        assert (tmp_path / "validation_report.txt").exists()
        assert (tmp_path / "validation_summary.json").exists()
        assert (tmp_path / "validation_summary.csv").exists()