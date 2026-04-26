import json

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
        }
    )


# helpers to dig into dimension reports


def _find_check(report: dict, substring: str) -> dict | None:
    for c in report.get("checks", []):
        if substring.lower() in c.get("check", "").lower():
            return c
    return None


# 1 ─ ACCURACY


class TestCheckAccuracy:

    def test_range_violation_detected(self, clean_df):
        from src.validation.validator import check_accuracy

        df = clean_df.copy()
        df.loc[0:99, "Temperature"] = 200.0
        result = check_accuracy(df)
        chk = _find_check(result, "Temperature")
        assert chk is not None
        assert chk["violations"] >= 100

    def test_invalid_type_value_detected(self, clean_df):
        from src.validation.validator import check_accuracy

        df = clean_df.copy()
        df.loc[0, "Type"] = "Z"
        result = check_accuracy(df)
        chk = _find_check(result, "Type in")
        assert chk is not None
        assert chk["violations"] >= 1
        assert "Z" in chk["sample_invalid_values"]

    def test_invalid_isholiday_detected(self, clean_df):
        from src.validation.validator import check_accuracy

        df = clean_df.copy()
        df["IsHoliday"] = df["IsHoliday"].astype(object)
        df.loc[0, "IsHoliday"] = "maybe"
        result = check_accuracy(df)
        chk = _find_check(result, "IsHoliday")
        assert chk is not None
        assert chk["status"] == "FAIL"

    def test_date_out_of_range_detected(self, clean_df):
        from src.validation.validator import check_accuracy

        df = clean_df.copy()
        df.loc[0, "Date"] = pd.Timestamp("2005-01-01")
        result = check_accuracy(df)
        chk = _find_check(result, "Date within")
        assert chk is not None
        assert chk["out_of_range_rows"] >= 1

    def test_sales_class_invalid_value(self, clean_df):
        from src.validation.validator import check_accuracy

        df = clean_df.copy()
        df.loc[0, "Sales_Class"] = 5
        result = check_accuracy(df)
        chk = _find_check(result, "Sales_Class values")
        assert chk is not None
        assert chk["status"] == "FAIL"
        assert 5 in chk["invalid_values"]

    def test_negative_sales_warns(self, clean_df):
        from src.validation.validator import check_accuracy

        df = clean_df.copy()
        df.loc[0:4, "Weekly_Sales"] = -500.0
        result = check_accuracy(df)
        chk = _find_check(result, "Negative Weekly_Sales")
        assert chk is not None
        assert chk["status"] == "WARN"
        assert chk["negative_rows"] == 5


# ══════════════════════════════════════════════════════════════
# 2 ─ COMPLETENESS
# ══════════════════════════════════════════════════════════════


class TestCheckCompleteness:

    def test_passes_on_clean_data(self, clean_df):
        from src.validation.validator import check_completeness

        result = check_completeness(clean_df)
        assert result["dimension"] == "Completeness"
        assert result["overall_completeness_pct"] == 100.0

    def test_row_count_check(self, clean_df):
        from src.validation.validator import check_completeness

        small = clean_df.head(100)
        result = check_completeness(small)
        chk = _find_check(result, "Minimum row count")
        assert chk is not None
        assert chk["status"] == "FAIL"

    def test_missing_column_detected(self, clean_df):
        from src.validation.validator import check_completeness

        df = clean_df.drop(columns=["Weekly_Sales"])
        result = check_completeness(df)
        chk = _find_check(result, "required columns")
        assert chk is not None
        assert "Weekly_Sales" in chk["missing_columns"]
        assert chk["status"] == "FAIL"

    def test_column_null_threshold_breach(self, clean_df):
        from src.validation.validator import check_completeness

        df = clean_df.copy()
        # Temperature threshold is 1 %, inject ~10 % nulls
        n_nulls = int(len(df) * 0.10)
        df.loc[: n_nulls - 1, "Temperature"] = np.nan
        result = check_completeness(df)
        chk = _find_check(result, "Temperature null")
        assert chk is not None
        assert chk["status"] == "FAIL"
        assert chk["null_pct"] > 1.0

    def test_row_level_missingness_warns(self, clean_df):
        from src.validation.validator import check_completeness

        df = clean_df.copy()
        df.loc[0:20, "UMCSENT"] = np.nan
        result = check_completeness(df)
        chk = _find_check(result, "Row-level missingness")
        assert chk is not None
        assert chk["status"] == "WARN"
        assert chk["rows_with_missing"] > 0

    def test_fred_coverage_pass(self, clean_df):
        from src.validation.validator import check_completeness

        result = check_completeness(clean_df)
        chk = _find_check(result, "FRED")
        assert chk is not None
        assert chk["status"] == "PASS"

    def test_fred_coverage_fail_when_missing(self, clean_df):
        from src.validation.validator import check_completeness

        df = clean_df.drop(columns=["UMCSENT"])
        result = check_completeness(df)
        chk = _find_check(result, "FRED")
        assert chk is not None
        assert chk["status"] == "FAIL"
        assert chk["fred_column_coverage"]["UMCSENT"]["present"] is False

    def test_total_missing_cells_counted(self, clean_df):
        from src.validation.validator import check_completeness

        df = clean_df.copy()
        df.loc[0:9, "Weekly_Sales"] = np.nan
        result = check_completeness(df)
        assert result["overall_missing_cells"] == 10


# ══════════════════════════════════════════════════════════════
# 3 ─ CONSISTENCY
# ══════════════════════════════════════════════════════════════


class TestCheckConsistency:

    def test_passes_on_clean_data(self, clean_df):
        from src.validation.validator import check_consistency

        result = check_consistency(clean_df)
        assert result["dimension"] == "Consistency"
        assert result["overall_status"] == "PASS"

    def test_invalid_store_id_detected(self, clean_df):
        from src.validation.validator import check_consistency

        df = clean_df.copy()
        df.loc[0, "Store"] = 99
        result = check_consistency(df)
        chk = _find_check(result, "Store IDs")
        assert chk is not None
        assert chk["status"] == "FAIL"
        assert 99 in chk["invalid_store_ids"]

    def test_invalid_dept_id_detected(self, clean_df):
        from src.validation.validator import check_consistency

        df = clean_df.copy()
        df.loc[0, "Dept"] = 200
        result = check_consistency(df)
        chk = _find_check(result, "Dept IDs")
        assert chk is not None
        assert chk["status"] == "FAIL"

    def test_sales_per_sqft_violation(self, clean_df):
        from src.validation.validator import check_consistency

        df = clean_df.copy()
        # Force extreme sales-per-sqft: sales=500k, size=1000 → 500
        df.loc[0:199, "Weekly_Sales"] = 500_000
        df.loc[0:199, "Size"] = 1_000
        result = check_consistency(df)
        chk = _find_check(result, "sales-per-sqft")
        assert chk is not None
        assert chk["violations"] >= 200

    def test_frozen_summer_detected(self, clean_df):
        from src.validation.validator import check_consistency

        df = clean_df.copy()
        summer_mask = df["Date"].dt.month.isin([6, 7, 8])
        summer_idx = df.index[summer_mask][:5]
        df.loc[summer_idx, "Temperature"] = -10.0
        result = check_consistency(df)
        chk = _find_check(result, "sub-zero")
        assert chk is not None
        assert chk["violations"] >= 5
        assert chk["status"] == "FAIL"

    def test_isholiday_boolean_check(self, clean_df):
        from src.validation.validator import check_consistency

        result = check_consistency(clean_df)
        chk = _find_check(result, "IsHoliday contains only boolean")
        assert chk is not None
        assert chk["status"] == "PASS"


# ══════════════════════════════════════════════════════════════
# 4 ─ UNIQUENESS
# ══════════════════════════════════════════════════════════════


class TestCheckUniqueness:

    def test_passes_on_unique_data(self, clean_df):
        from src.validation.validator import check_uniqueness

        result = check_uniqueness(clean_df)
        assert result["dimension"] == "Uniqueness"
        assert result["overall_status"] == "PASS"

    def test_full_row_duplicates_detected(self, clean_df):
        from src.validation.validator import check_uniqueness

        df = pd.concat([clean_df, clean_df.head(5)], ignore_index=True)
        result = check_uniqueness(df)
        chk = _find_check(result, "full-row duplicates")
        assert chk is not None
        assert chk["duplicate_count"] == 5
        assert chk["status"] == "FAIL"

    def test_business_key_duplicate_detected(self, clean_df):
        from src.validation.validator import check_uniqueness

        df = clean_df.copy()
        dup_row = df.iloc[0].copy()
        dup_row["Weekly_Sales"] = 999999  # different sales, same key
        df = pd.concat([df, pd.DataFrame([dup_row])], ignore_index=True)
        result = check_uniqueness(df)
        chk = _find_check(result, "business key")
        assert chk is not None
        assert chk["duplicate_count"] >= 1

    def test_fingerprint_duplicate(self, clean_df):
        from src.validation.validator import check_uniqueness

        df = pd.concat([clean_df, clean_df.head(3)], ignore_index=True)
        result = check_uniqueness(df)
        chk = _find_check(result, "fingerprint")
        assert chk is not None
        assert chk["duplicate_count"] >= 3


# ══════════════════════════════════════════════════════════════
# 5 ─ OUTLIER DETECTION
# ══════════════════════════════════════════════════════════════


class TestCheckOutliers:

    def test_passes_on_clean_data(self, clean_df):
        from src.validation.validator import check_outliers

        result = check_outliers(clean_df)
        assert result["dimension"] == "Outlier Detection"
        # uniform data should have very few outliers
        assert result["overall_status"] == "PASS"

    def test_outlier_column_structure(self, clean_df):
        from src.validation.validator import check_outliers

        result = check_outliers(clean_df)
        assert "columns" in result
        assert isinstance(result["columns"], dict)
        for col_name, col_info in result["columns"].items():
            assert "iqr_method" in col_info
            assert "zscore_method" in col_info
            assert "overall_status" in col_info
            assert "skewness" in col_info
            assert "is_skewed" in col_info

    def test_iqr_and_zscore_both_present(self, clean_df):
        from src.validation.validator import check_outliers

        result = check_outliers(clean_df)
        for col_info in result["columns"].values():
            iqr = col_info["iqr_method"]
            z = col_info["zscore_method"]
            assert "outlier_count" in iqr
            assert "outlier_pct" in iqr
            assert "outlier_count" in z
            assert "outlier_pct" in z

    def test_methods_listed(self, clean_df):
        from src.validation.validator import check_outliers

        result = check_outliers(clean_df)
        assert "methods_used" in result
        assert len(result["methods_used"]) == 2

    def test_skewed_column_lenient_evaluation(self, clean_df):
        from src.validation.validator import check_outliers

        df = clean_df.copy()
        # Create a right-skewed distribution (skew > 1)
        rng = np.random.default_rng(99)
        df["MarkDown1"] = rng.exponential(scale=5000, size=len(df))
        result = check_outliers(df)
        md1 = result["columns"].get("MarkDown1")
        assert md1 is not None
        assert md1["is_skewed"] is True
        assert md1["evaluation_mode"].startswith("lenient")


# ══════════════════════════════════════════════════════════════
# 6 ─ DISTRIBUTION PROFILE
# ══════════════════════════════════════════════════════════════


class TestCheckDistributionProfile:

    def test_numeric_profiles_present(self, clean_df):
        from src.validation.validator import check_distribution_profile

        result = check_distribution_profile(clean_df)
        profiles = result["numeric_profiles"]
        assert "Weekly_Sales" in profiles
        ws = profiles["Weekly_Sales"]
        for key in ["mean", "median", "std", "min", "max", "skewness", "kurtosis"]:
            assert key in ws

    def test_categorical_profiles_present(self, clean_df):
        from src.validation.validator import check_distribution_profile

        result = check_distribution_profile(clean_df)
        cats = result["categorical_profiles"]
        assert "Type" in cats
        assert "IsHoliday" in cats
        assert "unique_values" in cats["Type"]
        assert "balance_flag" in cats["IsHoliday"]

    def test_class_imbalance_check(self, clean_df):
        from src.validation.validator import check_distribution_profile

        result = check_distribution_profile(clean_df)
        chk = _find_check(result, "Class imbalance")
        assert chk is not None
        assert chk["status"] == "PASS"
        assert chk["imbalance_ratio"] < 2.0

    def test_temperature_sanity(self, clean_df):
        from src.validation.validator import check_distribution_profile

        result = check_distribution_profile(clean_df)
        chk = _find_check(result, "Median Temperature")
        assert chk is not None
        assert chk["status"] == "PASS"

    def test_unemployment_sanity(self, clean_df):
        from src.validation.validator import check_distribution_profile

        result = check_distribution_profile(clean_df)
        chk = _find_check(result, "Median Unemployment")
        assert chk is not None
        assert chk["status"] == "PASS"

    def test_store_type_sanity(self, clean_df):
        from src.validation.validator import check_distribution_profile

        result = check_distribution_profile(clean_df)
        chk = _find_check(result, "Most common store Type")
        assert chk is not None
        # In the fixture, Type is uniformly random among A/B/C, so this
        # may or may not be "A" — we just verify the check ran.
        assert chk["status"] in ("PASS", "FAIL")

    def test_class_imbalance_skip_without_target(self, clean_df):
        from src.validation.validator import check_distribution_profile

        df = clean_df.drop(columns=["Sales_Class"])
        result = check_distribution_profile(df)
        chk = _find_check(result, "Class imbalance")
        assert chk is None  # check should not appear


# ══════════════════════════════════════════════════════════════
# 7 ─ RELATIONSHIPS
# ══════════════════════════════════════════════════════════════


class TestCheckRelationships:

    def test_structure_on_clean_data(self, clean_df):
        from src.validation.validator import check_relationships

        result = check_relationships(clean_df)
        assert result["dimension"] == "Relationships"
        assert "pairwise_checks" in result
        assert "target_feature_correlations" in result
        assert "pearson_correlation_matrix" in result

    def test_pairwise_direction_checks_run(self, clean_df):
        from src.validation.validator import check_relationships

        result = check_relationships(clean_df)
        pairs = result["pairwise_checks"]["results"]
        assert len(pairs) >= 3  # 2 positive + 1 negative expected

    def test_pair_result_structure(self, clean_df):
        from src.validation.validator import check_relationships

        result = check_relationships(clean_df)
        for pair in result["pairwise_checks"]["results"]:
            assert "pearson_r" in pair
            assert "pearson_p" in pair
            assert "spearman_r" in pair
            assert "strength" in pair
            assert "status" in pair

    def test_target_correlations_present(self, clean_df):
        from src.validation.validator import check_relationships

        result = check_relationships(clean_df)
        tc = result["target_feature_correlations"]
        assert isinstance(tc, dict)
        # Should have entries for numeric columns other than Weekly_Sales
        assert "Size" in tc or "Temperature" in tc

    def test_rsxfs_pce_positive_correlation(self, clean_df):
        from src.validation.validator import check_relationships

        # Force strong positive correlation
        df = clean_df.copy()
        df["PCE"] = df["RSXFS"] * 0.03 + np.random.default_rng(1).normal(0, 10, len(df))
        result = check_relationships(df)
        pairs = result["pairwise_checks"]["results"]
        rsxfs_pce = [p for p in pairs if "RSXFS" in p["pair"] and "PCE" in p["pair"]]
        assert len(rsxfs_pce) == 1
        assert rsxfs_pce[0]["status"] == "PASS"
        assert rsxfs_pce[0]["pearson_r"] > 0


# ══════════════════════════════════════════════════════════════
# ORCHESTRATOR — run_validation
# ══════════════════════════════════════════════════════════════


class TestRunValidation:

    def test_returns_all_dimensions(self, clean_df, tmp_path, monkeypatch):
        import src.validation.validator as validator

        monkeypatch.setattr(validator, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(
            validator, "REPORT_PATH", tmp_path / "validation_report.txt"
        )
        monkeypatch.setattr(
            validator, "JSON_SUMMARY_PATH", tmp_path / "validation_summary.json"
        )
        monkeypatch.setattr(
            validator, "CSV_SUMMARY_PATH", tmp_path / "validation_summary.csv"
        )

        report = validator.run_validation(clean_df)

        assert "dimensions" in report
        expected_dims = {
            "1_accuracy",
            "2_completeness",
            "3_consistency",
            "4_uniqueness",
            "5_outlier_detection",
            "6_distribution_profile",
            "7_relationships",
        }
        assert set(report["dimensions"].keys()) == expected_dims

    def test_summary_present(self, clean_df, tmp_path, monkeypatch):
        import src.validation.validator as validator

        monkeypatch.setattr(validator, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(
            validator, "REPORT_PATH", tmp_path / "validation_report.txt"
        )
        monkeypatch.setattr(
            validator, "JSON_SUMMARY_PATH", tmp_path / "validation_summary.json"
        )
        monkeypatch.setattr(
            validator, "CSV_SUMMARY_PATH", tmp_path / "validation_summary.csv"
        )

        report = validator.run_validation(clean_df)
        summary = report["summary"]

        assert "total_checks" in summary
        assert "passed" in summary
        assert "warnings" in summary
        assert "failed" in summary
        assert "overall_status" in summary
        assert summary["overall_status"] in ("PASS", "WARN", "FAIL")
        assert summary["dimensions_evaluated"] == 7

    def test_snapshots_present(self, clean_df, tmp_path, monkeypatch):
        import src.validation.validator as validator

        monkeypatch.setattr(validator, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(
            validator, "REPORT_PATH", tmp_path / "validation_report.txt"
        )
        monkeypatch.setattr(
            validator, "JSON_SUMMARY_PATH", tmp_path / "validation_summary.json"
        )
        monkeypatch.setattr(
            validator, "CSV_SUMMARY_PATH", tmp_path / "validation_summary.csv"
        )

        report = validator.run_validation(clean_df)

        assert "snapshots" in report
        assert "schema" in report["snapshots"]
        assert "top_missing_columns" in report["snapshots"]
        assert "sample_rows_head" in report["snapshots"]

    def test_writes_text_report(self, clean_df, tmp_path, monkeypatch):
        import src.validation.validator as validator

        monkeypatch.setattr(validator, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(
            validator, "REPORT_PATH", tmp_path / "validation_report.txt"
        )
        monkeypatch.setattr(
            validator, "JSON_SUMMARY_PATH", tmp_path / "validation_summary.json"
        )
        monkeypatch.setattr(
            validator, "CSV_SUMMARY_PATH", tmp_path / "validation_summary.csv"
        )

        validator.run_validation(clean_df)

        txt_path = tmp_path / "validation_report.txt"
        assert txt_path.exists()
        content = txt_path.read_text()
        assert "WALMART SALES CLASSIFICATION" in content
        assert "DIMENSION" in content

    def test_writes_json_summary(self, clean_df, tmp_path, monkeypatch):
        import src.validation.validator as validator

        monkeypatch.setattr(validator, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(
            validator, "REPORT_PATH", tmp_path / "validation_report.txt"
        )
        monkeypatch.setattr(
            validator, "JSON_SUMMARY_PATH", tmp_path / "validation_summary.json"
        )
        monkeypatch.setattr(
            validator, "CSV_SUMMARY_PATH", tmp_path / "validation_summary.csv"
        )

        validator.run_validation(clean_df)

        json_path = tmp_path / "validation_summary.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "dimensions" in data
        assert "summary" in data

    def test_writes_csv_summary(self, clean_df, tmp_path, monkeypatch):
        import src.validation.validator as validator

        monkeypatch.setattr(validator, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(
            validator, "REPORT_PATH", tmp_path / "validation_report.txt"
        )
        monkeypatch.setattr(
            validator, "JSON_SUMMARY_PATH", tmp_path / "validation_summary.json"
        )
        monkeypatch.setattr(
            validator, "CSV_SUMMARY_PATH", tmp_path / "validation_summary.csv"
        )

        validator.run_validation(clean_df)

        csv_path = tmp_path / "validation_summary.csv"
        assert csv_path.exists()
        csv_df = pd.read_csv(csv_path)
        assert "dimension" in csv_df.columns
        assert "check" in csv_df.columns
        assert "status" in csv_df.columns
        assert len(csv_df) > 0

    def test_overall_fail_propagates(self, clean_df, tmp_path, monkeypatch):
        import src.validation.validator as validator

        monkeypatch.setattr(validator, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(
            validator, "REPORT_PATH", tmp_path / "validation_report.txt"
        )
        monkeypatch.setattr(
            validator, "JSON_SUMMARY_PATH", tmp_path / "validation_summary.json"
        )
        monkeypatch.setattr(
            validator, "CSV_SUMMARY_PATH", tmp_path / "validation_summary.csv"
        )

        df = clean_df.copy()
        # Break consistency: invalid store id
        df.loc[0, "Store"] = 99
        # Break accuracy: invalid Sales_Class
        df.loc[0, "Sales_Class"] = 5
        report = validator.run_validation(df)

        assert report["summary"]["overall_status"] == "FAIL"
        assert report["summary"]["dimensions_failed"] >= 1
