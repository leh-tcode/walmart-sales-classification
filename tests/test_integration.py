import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def raw_walmart_df():
    rng = np.random.default_rng(0)

    n_stores = 22
    n_depts = 5
    n_weeks = 100
    n = n_stores * n_depts * n_weeks

    store_ids = np.arange(1, n_stores + 1)
    stores = np.repeat(store_ids, n_depts * n_weeks)
    depts = np.tile(np.repeat(np.arange(1, n_depts + 1), n_weeks), n_stores)
    dates = np.tile(pd.date_range("2010-02-05", periods=n_weeks, freq="W"), n_stores * n_depts)

    sizes = np.repeat(
        rng.integers(34_000, 220_000, size=n_stores),
        n_depts * n_weeks,
    )

    sales_base = sizes / 10.0 + rng.normal(0, 3_000, n)
    sales_base[:5] = [-3_000, -1_500, -500, -200, -50]

    store_types_raw = (["A"] * 12 + ["B"] * 6 + ["C"] * 4)[:n_stores]
    store_types = np.repeat(store_types_raw, n_depts * n_weeks)

    trend = np.linspace(0, 1, n)
    rsxfs = 302_000 + trend * 53_000 + rng.normal(0, 1_000, n)
    pce = 9_800 + trend * 1_400 + rng.normal(0, 50, n)
    unemp = 10.0 - trend * 6.0 + rng.normal(0, 0.3, n)

    df = pd.DataFrame(
        {
            "Store": stores,
            "Dept": depts,
            "Date": dates,
            "Weekly_Sales": sales_base,
            "IsHoliday": rng.choice([True, False], size=n, p=[0.07, 0.93]),
            "Type": store_types,
            "Size": sizes,
            "Temperature": rng.uniform(20.0, 90.0, n),
            "Fuel_Price": rng.uniform(2.5, 4.5, n),
            "MarkDown1": np.where(rng.random(n) > 0.65, rng.uniform(100, 30_000, n), np.nan),
            "MarkDown2": np.where(rng.random(n) > 0.74, rng.uniform(10, 20_000, n), np.nan),
            "MarkDown3": np.where(rng.random(n) > 0.68, rng.uniform(1, 5_000, n), np.nan),
            "MarkDown4": np.where(rng.random(n) > 0.68, rng.uniform(1, 10_000, n), np.nan),
            "MarkDown5": np.where(rng.random(n) > 0.64, rng.uniform(100, 40_000, n), np.nan),
            "CPI": rng.uniform(211, 228, n),
            "Unemployment": unemp.clip(2.0, 20.0),
            "UMCSENT": rng.uniform(55, 83, n),
            "RSXFS": rsxfs,
            "PCE": pce,
        }
    )

    df["Sales_Class"] = df.groupby("Store")["Weekly_Sales"].transform(lambda x: (x > x.median()).astype(int))

    return df


@pytest.fixture(scope="module")
def cleaned_df(raw_walmart_df):
    from src.cleaning.cleaning import run_cleaning

    return run_cleaning(raw_walmart_df.copy())


@pytest.fixture(scope="module")
def featured_df(cleaned_df):
    from src.features.feature_engineering import run_feature_engineering

    return run_feature_engineering(cleaned_df.copy())


class TestAcquisitionToValidation:
    def test_raw_dataset_passes_validation_overall(self, raw_walmart_df, tmp_path, monkeypatch):
        import src.validation.validator as validator

        monkeypatch.setattr(validator, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(validator, "REPORT_PATH", tmp_path / "r.txt")
        monkeypatch.setattr(validator, "JSON_SUMMARY_PATH", tmp_path / "r.json")
        monkeypatch.setattr(validator, "CSV_SUMMARY_PATH", tmp_path / "r.csv")

        report = validator.run_validation(raw_walmart_df)

        assert report["summary"]["overall_status"] in ("PASS", "WARN"), f"Acquisition output triggered a FAIL in validation: {report['summary']}"

    def test_all_seven_dimensions_evaluated(self, raw_walmart_df, tmp_path, monkeypatch):
        import src.validation.validator as validator

        monkeypatch.setattr(validator, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(validator, "REPORT_PATH", tmp_path / "r.txt")
        monkeypatch.setattr(validator, "JSON_SUMMARY_PATH", tmp_path / "r.json")
        monkeypatch.setattr(validator, "CSV_SUMMARY_PATH", tmp_path / "r.csv")

        report = validator.run_validation(raw_walmart_df)

        assert report["summary"]["dimensions_evaluated"] == 7

    def test_row_count_unchanged_through_validation(self, raw_walmart_df, tmp_path, monkeypatch):
        import src.validation.validator as validator

        monkeypatch.setattr(validator, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(validator, "REPORT_PATH", tmp_path / "r.txt")
        monkeypatch.setattr(validator, "JSON_SUMMARY_PATH", tmp_path / "r.json")
        monkeypatch.setattr(validator, "CSV_SUMMARY_PATH", tmp_path / "r.csv")

        original_len = len(raw_walmart_df)
        validator.run_validation(raw_walmart_df)

        assert len(raw_walmart_df) == original_len, "Validation must not modify the input dataframe in-place"

    def test_negative_sales_produce_warn_not_fail(self, raw_walmart_df, tmp_path, monkeypatch):
        import src.validation.validator as validator

        monkeypatch.setattr(validator, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(validator, "REPORT_PATH", tmp_path / "r.txt")
        monkeypatch.setattr(validator, "JSON_SUMMARY_PATH", tmp_path / "r.json")
        monkeypatch.setattr(validator, "CSV_SUMMARY_PATH", tmp_path / "r.csv")

        report = validator.run_validation(raw_walmart_df)
        accuracy = report["dimensions"]["1_accuracy"]
        neg_check = next(
            (c for c in accuracy["checks"] if "Negative" in c.get("check", "")),
            None,
        )

        assert neg_check is not None, "Negative sales check must be present in accuracy dimension"
        assert neg_check["status"] == "WARN", (
            f"Negative sales should be WARN not {neg_check['status']} — they are valid returns that cleaning preserves"
        )

    def test_fred_columns_fully_covered_after_acquisition(self, raw_walmart_df, tmp_path, monkeypatch):
        import src.validation.validator as validator

        monkeypatch.setattr(validator, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(validator, "REPORT_PATH", tmp_path / "r.txt")
        monkeypatch.setattr(validator, "JSON_SUMMARY_PATH", tmp_path / "r.json")
        monkeypatch.setattr(validator, "CSV_SUMMARY_PATH", tmp_path / "r.csv")

        report = validator.run_validation(raw_walmart_df)
        completeness = report["dimensions"]["2_completeness"]
        fred_check = next(
            (c for c in completeness["checks"] if "FRED" in c.get("check", "")),
            None,
        )

        assert fred_check is not None
        assert fred_check["status"] == "PASS", "FRED columns must be fully populated after acquisition merge"


class TestAcquisitionToCleaning:
    def test_cleaning_accepts_raw_acquisition_output(self, raw_walmart_df):
        from src.cleaning.cleaning import run_cleaning

        result = run_cleaning(raw_walmart_df.copy())
        assert result is not None

    def test_row_count_preserved_across_boundary(self, raw_walmart_df, cleaned_df):
        assert len(cleaned_df) == len(raw_walmart_df)

    def test_cleaning_resolves_all_markdown_nulls(self, raw_walmart_df, cleaned_df):
        md_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
        for col in md_cols:
            assert cleaned_df[col].isna().sum() == 0, f"{col} still has nulls after cleaning — feature engineering aggregations will propagate NaN"

    def test_cleaning_adds_expected_flag_columns(self, cleaned_df):
        expected_new = [
            "has_MarkDown1",
            "has_MarkDown2",
            "has_MarkDown3",
            "has_MarkDown4",
            "has_MarkDown5",
            "is_return",
        ]
        for col in expected_new:
            assert col in cleaned_df.columns, f"Cleaning must create '{col}' — feature engineering reads it downstream"

    def test_cleaning_output_has_no_nulls(self, cleaned_df):
        total_nulls = cleaned_df.isna().sum().sum()
        assert total_nulls == 0, f"Cleaned dataset has {total_nulls} nulls — feature engineering assumes a null-free input"

    def test_target_column_intact_after_cleaning(self, raw_walmart_df, cleaned_df):
        assert "Sales_Class" in cleaned_df.columns
        pd.testing.assert_series_equal(
            cleaned_df["Sales_Class"].reset_index(drop=True),
            raw_walmart_df["Sales_Class"].reset_index(drop=True),
            check_names=False,
        )

    def test_original_columns_still_present_after_cleaning(self, raw_walmart_df, cleaned_df):
        for col in raw_walmart_df.columns:
            assert col in cleaned_df.columns, f"Cleaning dropped '{col}' which was present in acquisition output"


class TestCleaningToFeatureEngineering:
    def test_feature_engineering_accepts_cleaning_output(self, cleaned_df):
        from src.features.feature_engineering import run_feature_engineering

        result = run_feature_engineering(cleaned_df.copy())
        assert result is not None

    def test_feature_engineering_expands_column_count(self, cleaned_df, featured_df):
        assert len(featured_df.columns) > len(cleaned_df.columns), (
            f"Feature engineering must add columns: got {len(featured_df.columns)} vs input {len(cleaned_df.columns)}"
        )

    def test_row_count_preserved_across_boundary(self, cleaned_df, featured_df):
        assert len(featured_df) == len(cleaned_df)

    def test_no_nulls_introduced_by_feature_engineering(self, featured_df):
        total_nulls = featured_df.isna().sum().sum()
        assert total_nulls == 0, (
            f"Feature engineering introduced {total_nulls} nulls — columns: {featured_df.columns[featured_df.isna().any()].tolist()}"
        )

    def test_no_infinities_introduced_by_feature_engineering(self, featured_df):
        numeric = featured_df.select_dtypes(include=[np.number])
        inf_count = np.isinf(numeric).sum().sum()
        assert inf_count == 0, (
            f"Feature engineering introduced {inf_count} infinite values — check division-based interaction features for zero denominators"
        )

    def test_target_column_intact_after_feature_engineering(self, cleaned_df, featured_df):
        assert "Sales_Class" in featured_df.columns
        pd.testing.assert_series_equal(
            featured_df["Sales_Class"].reset_index(drop=True),
            cleaned_df["Sales_Class"].reset_index(drop=True),
            check_names=False,
        )

    def test_cleaning_flag_columns_survive_feature_engineering(self, featured_df):
        required = [
            "is_return",
            "has_MarkDown1",
            "has_MarkDown2",
            "has_MarkDown3",
            "has_MarkDown4",
            "has_MarkDown5",
        ]
        for col in required:
            assert col in featured_df.columns, f"Flag column '{col}' was dropped during feature engineering — it must survive as a model feature"


class TestFeatureEngineeringToPreprocessing:
    def test_preprocessing_accepts_feature_engineering_output(self, featured_df, tmp_path, monkeypatch):
        import src.features.preprocessing as pp

        monkeypatch.setattr(pp, "EDA_DIR", tmp_path / "eda")
        monkeypatch.setattr(pp, "MODEL_DIR", tmp_path / "model")
        monkeypatch.setattr(pp, "ARTIFACTS_DIR", tmp_path / "artifacts")
        monkeypatch.setattr(pp, "EDA_PATH", tmp_path / "eda" / "eda_dataset.csv")
        monkeypatch.setattr(pp, "TRAIN_PATH", tmp_path / "model" / "train.csv")
        monkeypatch.setattr(pp, "TEST_PATH", tmp_path / "model" / "test.csv")
        monkeypatch.setattr(pp, "SCALER_PATH", tmp_path / "artifacts" / "scaler.joblib")
        monkeypatch.setattr(pp, "FEATURE_META_PATH", tmp_path / "artifacts" / "feature_metadata.json")
        monkeypatch.setattr(pp, "REPORT_JSON_PATH", tmp_path / "preprocessing_report.json")
        monkeypatch.setattr(pp, "REPORT_TEXT_PATH", tmp_path / "preprocessing_report.txt")

        result = pp.run_preprocessing(featured_df.copy())
        assert result is not None

    def test_leakage_columns_not_in_model_splits(self, featured_df, tmp_path, monkeypatch):
        import src.features.preprocessing as pp

        monkeypatch.setattr(pp, "EDA_DIR", tmp_path / "eda")
        monkeypatch.setattr(pp, "MODEL_DIR", tmp_path / "model")
        monkeypatch.setattr(pp, "ARTIFACTS_DIR", tmp_path / "artifacts")
        monkeypatch.setattr(pp, "EDA_PATH", tmp_path / "eda" / "eda_dataset.csv")
        monkeypatch.setattr(pp, "TRAIN_PATH", tmp_path / "model" / "train.csv")
        monkeypatch.setattr(pp, "TEST_PATH", tmp_path / "model" / "test.csv")
        monkeypatch.setattr(pp, "SCALER_PATH", tmp_path / "artifacts" / "scaler.joblib")
        monkeypatch.setattr(pp, "FEATURE_META_PATH", tmp_path / "artifacts" / "feature_metadata.json")
        monkeypatch.setattr(pp, "REPORT_JSON_PATH", tmp_path / "preprocessing_report.json")
        monkeypatch.setattr(pp, "REPORT_TEXT_PATH", tmp_path / "preprocessing_report.txt")

        result = pp.run_preprocessing(featured_df.copy())
        X_train = result["X_train"]
        X_test = result["X_test"]

        for leakage_col in ["Weekly_Sales", "Date"]:
            assert leakage_col not in X_train.columns, f"'{leakage_col}' found in X_train — this is target leakage"
            assert leakage_col not in X_test.columns, f"'{leakage_col}' found in X_test — this is target leakage"

    def test_split_ratio_is_approximately_80_20(self, featured_df, tmp_path, monkeypatch):
        import src.features.preprocessing as pp

        monkeypatch.setattr(pp, "EDA_DIR", tmp_path / "eda")
        monkeypatch.setattr(pp, "MODEL_DIR", tmp_path / "model")
        monkeypatch.setattr(pp, "ARTIFACTS_DIR", tmp_path / "artifacts")
        monkeypatch.setattr(pp, "EDA_PATH", tmp_path / "eda" / "eda_dataset.csv")
        monkeypatch.setattr(pp, "TRAIN_PATH", tmp_path / "model" / "train.csv")
        monkeypatch.setattr(pp, "TEST_PATH", tmp_path / "model" / "test.csv")
        monkeypatch.setattr(pp, "SCALER_PATH", tmp_path / "artifacts" / "scaler.joblib")
        monkeypatch.setattr(pp, "FEATURE_META_PATH", tmp_path / "artifacts" / "feature_metadata.json")
        monkeypatch.setattr(pp, "REPORT_JSON_PATH", tmp_path / "preprocessing_report.json")
        monkeypatch.setattr(pp, "REPORT_TEXT_PATH", tmp_path / "preprocessing_report.txt")

        result = pp.run_preprocessing(featured_df.copy())
        total = len(result["X_train"]) + len(result["X_test"])
        train_ratio = len(result["X_train"]) / total

        assert 0.78 <= train_ratio <= 0.82, f"Train ratio is {train_ratio:.2f} — expected ~0.80 (80/20 split)"

    def test_stratification_preserves_class_balance(self, featured_df, tmp_path, monkeypatch):
        import src.features.preprocessing as pp

        monkeypatch.setattr(pp, "EDA_DIR", tmp_path / "eda")
        monkeypatch.setattr(pp, "MODEL_DIR", tmp_path / "model")
        monkeypatch.setattr(pp, "ARTIFACTS_DIR", tmp_path / "artifacts")
        monkeypatch.setattr(pp, "EDA_PATH", tmp_path / "eda" / "eda_dataset.csv")
        monkeypatch.setattr(pp, "TRAIN_PATH", tmp_path / "model" / "train.csv")
        monkeypatch.setattr(pp, "TEST_PATH", tmp_path / "model" / "test.csv")
        monkeypatch.setattr(pp, "SCALER_PATH", tmp_path / "artifacts" / "scaler.joblib")
        monkeypatch.setattr(pp, "FEATURE_META_PATH", tmp_path / "artifacts" / "feature_metadata.json")
        monkeypatch.setattr(pp, "REPORT_JSON_PATH", tmp_path / "preprocessing_report.json")
        monkeypatch.setattr(pp, "REPORT_TEXT_PATH", tmp_path / "preprocessing_report.txt")

        result = pp.run_preprocessing(featured_df.copy())
        train_balance = result["y_train"].mean()
        test_balance = result["y_test"].mean()

        assert 0.40 <= train_balance <= 0.60, f"Train class balance is {train_balance:.2f} — stratification may have failed"
        assert 0.40 <= test_balance <= 0.60, f"Test class balance is {test_balance:.2f} — stratification may have failed"

    def test_scaler_fit_on_train_not_test(self, featured_df, tmp_path, monkeypatch):
        import src.features.preprocessing as pp

        monkeypatch.setattr(pp, "EDA_DIR", tmp_path / "eda")
        monkeypatch.setattr(pp, "MODEL_DIR", tmp_path / "model")
        monkeypatch.setattr(pp, "ARTIFACTS_DIR", tmp_path / "artifacts")
        monkeypatch.setattr(pp, "EDA_PATH", tmp_path / "eda" / "eda_dataset.csv")
        monkeypatch.setattr(pp, "TRAIN_PATH", tmp_path / "model" / "train.csv")
        monkeypatch.setattr(pp, "TEST_PATH", tmp_path / "model" / "test.csv")
        monkeypatch.setattr(pp, "SCALER_PATH", tmp_path / "artifacts" / "scaler.joblib")
        monkeypatch.setattr(pp, "FEATURE_META_PATH", tmp_path / "artifacts" / "feature_metadata.json")
        monkeypatch.setattr(pp, "REPORT_JSON_PATH", tmp_path / "preprocessing_report.json")
        monkeypatch.setattr(pp, "REPORT_TEXT_PATH", tmp_path / "preprocessing_report.txt")

        result = pp.run_preprocessing(featured_df.copy())

        if "Weekly_Sales" not in result["X_train"].columns and "Size" in result["X_train"].columns:
            train_median = result["X_train"]["Size"].median()
            assert abs(train_median) < 0.5, (
                f"Train 'Size' median after scaling is {train_median:.3f} — expected ~0.0 if scaler was fit on train (RobustScaler centres at median)"
            )

    def test_no_nulls_in_final_splits(self, featured_df, tmp_path, monkeypatch):
        import src.features.preprocessing as pp

        monkeypatch.setattr(pp, "EDA_DIR", tmp_path / "eda")
        monkeypatch.setattr(pp, "MODEL_DIR", tmp_path / "model")
        monkeypatch.setattr(pp, "ARTIFACTS_DIR", tmp_path / "artifacts")
        monkeypatch.setattr(pp, "EDA_PATH", tmp_path / "eda" / "eda_dataset.csv")
        monkeypatch.setattr(pp, "TRAIN_PATH", tmp_path / "model" / "train.csv")
        monkeypatch.setattr(pp, "TEST_PATH", tmp_path / "model" / "test.csv")
        monkeypatch.setattr(pp, "SCALER_PATH", tmp_path / "artifacts" / "scaler.joblib")
        monkeypatch.setattr(pp, "FEATURE_META_PATH", tmp_path / "artifacts" / "feature_metadata.json")
        monkeypatch.setattr(pp, "REPORT_JSON_PATH", tmp_path / "preprocessing_report.json")
        monkeypatch.setattr(pp, "REPORT_TEXT_PATH", tmp_path / "preprocessing_report.txt")

        result = pp.run_preprocessing(featured_df.copy())

        train_nulls = result["X_train"].isna().sum().sum()
        test_nulls = result["X_test"].isna().sum().sum()

        assert train_nulls == 0, f"X_train has {train_nulls} nulls after preprocessing"
        assert test_nulls == 0, f"X_test has {test_nulls} nulls after preprocessing"

    def test_no_object_columns_in_model_splits(self, featured_df, tmp_path, monkeypatch):
        import src.features.preprocessing as pp

        monkeypatch.setattr(pp, "EDA_DIR", tmp_path / "eda")
        monkeypatch.setattr(pp, "MODEL_DIR", tmp_path / "model")
        monkeypatch.setattr(pp, "ARTIFACTS_DIR", tmp_path / "artifacts")
        monkeypatch.setattr(pp, "EDA_PATH", tmp_path / "eda" / "eda_dataset.csv")
        monkeypatch.setattr(pp, "TRAIN_PATH", tmp_path / "model" / "train.csv")
        monkeypatch.setattr(pp, "TEST_PATH", tmp_path / "model" / "test.csv")
        monkeypatch.setattr(pp, "SCALER_PATH", tmp_path / "artifacts" / "scaler.joblib")
        monkeypatch.setattr(pp, "FEATURE_META_PATH", tmp_path / "artifacts" / "feature_metadata.json")
        monkeypatch.setattr(pp, "REPORT_JSON_PATH", tmp_path / "preprocessing_report.json")
        monkeypatch.setattr(pp, "REPORT_TEXT_PATH", tmp_path / "preprocessing_report.txt")

        result = pp.run_preprocessing(featured_df.copy())

        train_obj = result["X_train"].select_dtypes(include=["object"]).columns.tolist()
        test_obj = result["X_test"].select_dtypes(include=["object"]).columns.tolist()

        assert train_obj == [], f"X_train has object columns: {train_obj}"
        assert test_obj == [], f"X_test has object columns: {test_obj}"


class TestFullPipelineEndToEnd:
    @pytest.fixture(scope="class")
    def e2e_result(self, raw_walmart_df, tmp_path_factory):
        import src.features.preprocessing as pp
        from src.cleaning.cleaning import run_cleaning
        from src.features.feature_engineering import run_feature_engineering

        tmp_path = tmp_path_factory.mktemp("e2e")

        pp.EDA_DIR = tmp_path / "eda"
        pp.MODEL_DIR = tmp_path / "model"
        pp.ARTIFACTS_DIR = tmp_path / "artifacts"
        pp.EDA_PATH = tmp_path / "eda" / "eda_dataset.csv"
        pp.TRAIN_PATH = tmp_path / "model" / "train.csv"
        pp.TEST_PATH = tmp_path / "model" / "test.csv"
        pp.SCALER_PATH = tmp_path / "artifacts" / "scaler.joblib"
        pp.FEATURE_META_PATH = tmp_path / "artifacts" / "feature_metadata.json"
        pp.REPORT_JSON_PATH = tmp_path / "preprocessing_report.json"
        pp.REPORT_TEXT_PATH = tmp_path / "preprocessing_report.txt"

        cleaned = run_cleaning(raw_walmart_df.copy())
        featured = run_feature_engineering(cleaned.copy())
        result = pp.run_preprocessing(featured.copy())

        return {"cleaned": cleaned, "featured": featured, "preprocessed": result}

    def test_pipeline_completes_without_error(self, e2e_result):
        assert "X_train" in e2e_result["preprocessed"]
        assert "X_test" in e2e_result["preprocessed"]
        assert "y_train" in e2e_result["preprocessed"]
        assert "y_test" in e2e_result["preprocessed"]
        assert "scaler" in e2e_result["preprocessed"]

    def test_total_row_count_conserved_end_to_end(self, raw_walmart_df, e2e_result):
        total_out = len(e2e_result["preprocessed"]["X_train"]) + len(e2e_result["preprocessed"]["X_test"])
        assert total_out == len(raw_walmart_df), f"Row count mismatch: {total_out} rows out vs {len(raw_walmart_df)} rows in"

    def test_feature_count_is_substantial(self, e2e_result):
        n_features = len(e2e_result["preprocessed"]["X_train"].columns)
        assert n_features >= 50, f"Only {n_features} features produced end-to-end — expected 50+ after 8 feature engineering groups"

    def test_column_count_grows_at_each_stage(self, raw_walmart_df, e2e_result):
        raw_cols = len(raw_walmart_df.columns)
        cleaned_cols = len(e2e_result["cleaned"].columns)
        featured_cols = len(e2e_result["featured"].columns)

        assert cleaned_cols >= raw_cols, f"Cleaning reduced columns: {raw_cols} → {cleaned_cols}"
        assert featured_cols > cleaned_cols, f"Feature engineering did not add columns: {cleaned_cols} → {featured_cols}"

    def test_target_binary_and_balanced_in_final_splits(self, e2e_result):
        y_train = e2e_result["preprocessed"]["y_train"]
        y_test = e2e_result["preprocessed"]["y_test"]

        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})
        assert 0.40 <= y_train.mean() <= 0.60
        assert 0.40 <= y_test.mean() <= 0.60

    def test_scaler_artifact_produced(self, e2e_result):
        from sklearn.preprocessing import RobustScaler

        scaler = e2e_result["preprocessed"]["scaler"]
        assert isinstance(scaler, RobustScaler), f"Expected RobustScaler, got {type(scaler).__name__}"
        assert hasattr(scaler, "center_"), "Scaler was not fitted (missing center_ attribute)"

    def test_data_flow_integrity_no_column_overwritten(self, raw_walmart_df, e2e_result):
        featured = e2e_result["featured"]
        if "Sales_Class" in featured.columns:
            pd.testing.assert_series_equal(
                featured["Sales_Class"].reset_index(drop=True),
                raw_walmart_df["Sales_Class"].reset_index(drop=True),
                check_names=False,
                check_dtype=False,
            )
