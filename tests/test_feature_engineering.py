import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (create_cyclical_features,
                                              create_economic_features,
                                              create_holiday_features,
                                              create_interaction_features,
                                              create_lag_features,
                                              create_promotion_features,
                                              create_store_dept_features,
                                              create_temporal_features,
                                              run_feature_engineering)


# Fixtures
@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 500

    df = pd.DataFrame(
        {
            "Store": np.repeat(range(1, 6), 100),
            "Dept": np.tile(np.repeat(range(1, 11), 10), 5),
            "Date": pd.date_range("2010-02-05", periods=n, freq="W"),
            "Weekly_Sales": np.random.normal(15000, 8000, n).clip(-5000, 200000),
            "IsHoliday": np.random.choice([True, False], size=n, p=[0.07, 0.93]),
            "Type": np.random.choice(["A", "B", "C"], size=n),
            "Size": np.random.randint(34000, 220000, size=n),
            "Temperature": np.random.normal(60, 18, n),
            "Fuel_Price": np.random.uniform(2.5, 4.5, n),
            "MarkDown1": np.random.uniform(0, 20000, n),
            "MarkDown2": np.random.uniform(0, 10000, n),
            "MarkDown3": np.random.uniform(0, 3000, n),
            "MarkDown4": np.random.uniform(0, 8000, n),
            "MarkDown5": np.random.uniform(0, 30000, n),
            "CPI": np.random.uniform(126, 228, n),
            "Unemployment": np.random.uniform(4, 14, n),
            "UMCSENT": np.random.uniform(55, 83, n),
            "RSXFS": np.random.uniform(302000, 355000, n),
            "PCE": np.random.uniform(10000, 11200, n),
            "Sales_Class": np.random.choice([0, 1], size=n),
            "has_MarkDown1": np.random.choice([0, 1], size=n),
            "has_MarkDown2": np.random.choice([0, 1], size=n),
            "has_MarkDown3": np.random.choice([0, 1], size=n),
            "has_MarkDown4": np.random.choice([0, 1], size=n),
            "has_MarkDown5": np.random.choice([0, 1], size=n),
            "is_return": (np.random.normal(15000, 8000, n) < 0).astype(int),
        }
    )
    return df


@pytest.fixture
def empty_report():
    return {"groups": []}


# Group 1: Temporal
class TestTemporalFeatures:

    def test_columns_created(self, sample_df, empty_report):
        df = create_temporal_features(sample_df.copy(), empty_report)
        expected = {
            "Year",
            "Month",
            "Week",
            "Quarter",
            "DayOfYear",
            "WeekOfMonth",
            "IsMonthStart",
            "IsMonthEnd",
            "IsYearStart",
            "IsYearEnd",
            "DaysInMonth",
            "YearProgress",
        }
        assert expected <= set(df.columns)

    def test_month_range(self, sample_df, empty_report):
        df = create_temporal_features(sample_df.copy(), empty_report)
        assert df["Month"].between(1, 12).all()

    def test_year_progress_range(self, sample_df, empty_report):
        df = create_temporal_features(sample_df.copy(), empty_report)
        assert df["YearProgress"].between(0, 1.01).all()

    def test_row_count(self, sample_df, empty_report):
        df = create_temporal_features(sample_df.copy(), empty_report)
        assert len(df) == len(sample_df)


# Group 2: Holiday
class TestHolidayFeatures:

    def test_columns_created(self, sample_df, empty_report):
        df = create_temporal_features(sample_df.copy(), empty_report)
        df = create_holiday_features(df, empty_report)
        expected = {
            "HolidayType",
            "IsPreHoliday",
            "IsPostHoliday",
            "HolidayProximity",
            "IsPeakSeason",
            "IsBackToSchool",
        }
        assert expected <= set(df.columns)

    def test_holiday_type_range(self, sample_df, empty_report):
        df = create_temporal_features(sample_df.copy(), empty_report)
        df = create_holiday_features(df, empty_report)
        assert set(df["HolidayType"].unique()) <= {0, 1, 2, 3, 4}

    def test_peak_season_only_nov_dec(self, sample_df, empty_report):
        df = create_temporal_features(sample_df.copy(), empty_report)
        df = create_holiday_features(df, empty_report)
        peak = df[df["IsPeakSeason"] == 1]
        if len(peak) > 0:
            assert set(peak["Month"].unique()) <= {11, 12}


# Group 3: Promotion
class TestPromotionFeatures:

    def test_columns_created(self, sample_df, empty_report):
        df = create_promotion_features(sample_df.copy(), empty_report)
        expected = {
            "TotalMarkDown",
            "ActiveMarkDownCount",
            "AvgMarkDownAmount",
            "MaxMarkDown",
            "HasAnyMarkDown",
        }
        assert expected <= set(df.columns)

    def test_total_markdown_non_negative(self, sample_df, empty_report):
        df = create_promotion_features(sample_df.copy(), empty_report)
        assert (df["TotalMarkDown"] >= 0).all()

    def test_active_count_range(self, sample_df, empty_report):
        df = create_promotion_features(sample_df.copy(), empty_report)
        assert df["ActiveMarkDownCount"].between(0, 5).all()


# Group 4: Store & Dept
class TestStoreDeptFeatures:

    def test_columns_created(self, sample_df, empty_report):
        df = create_store_dept_features(sample_df.copy(), empty_report)
        expected = {
            "TypeEncoded",
            "SizePerType",
            "StoreDeptCount",
            "DeptFrequency",
            "SizeQuartile",
        }
        assert expected <= set(df.columns)

    def test_type_encoding(self, sample_df, empty_report):
        df = create_store_dept_features(sample_df.copy(), empty_report)
        assert set(df["TypeEncoded"].unique()) <= {0, 1, 2}

    def test_size_quartile_range(self, sample_df, empty_report):
        df = create_store_dept_features(sample_df.copy(), empty_report)
        assert set(df["SizeQuartile"].unique()) <= {1, 2, 3, 4}


# Group 5: Economic
class TestEconomicFeatures:

    def test_columns_created(self, sample_df, empty_report):
        df = create_economic_features(sample_df.copy(), empty_report)
        expected = {
            "EconIndex",
            "ConsumerConfRatio",
            "RealSpendingPerCapita",
            "EconMomentum",
            "FuelBurden",
            "PurchasingPower",
        }
        assert expected <= set(df.columns)

    def test_econ_index_range(self, sample_df, empty_report):
        df = create_economic_features(sample_df.copy(), empty_report)
        assert df["EconIndex"].between(-0.01, 1.01).all()

    def test_no_temp_columns_left(self, sample_df, empty_report):
        df = create_economic_features(sample_df.copy(), empty_report)
        temp_cols = [c for c in df.columns if c.startswith("_") and c.endswith("_norm")]
        assert len(temp_cols) == 0


# Group 6: Lag & Rolling
class TestLagFeatures:

    def test_columns_created(self, sample_df, empty_report):
        df = create_lag_features(sample_df.copy(), empty_report)
        expected = {
            "Lag_Sales_1w",
            "Lag_Sales_2w",
            "Lag_Sales_4w",
            "Rolling_Mean_4w",
            "Rolling_Mean_8w",
            "Rolling_Mean_12w",
            "Rolling_Std_4w",
            "SalesTrend_4w",
            "SalesAcceleration",
        }
        assert expected <= set(df.columns)

    def test_no_nulls_after_fill(self, sample_df, empty_report):
        df = create_lag_features(sample_df.copy(), empty_report)
        lag_cols = [
            c for c in df.columns if c.startswith(("Lag_", "Rolling_", "Sales"))
        ]
        lag_cols = [c for c in lag_cols if c != "Sales_Class" and c != "Weekly_Sales"]
        assert df[lag_cols].isna().sum().sum() == 0

    def test_row_count_preserved(self, sample_df, empty_report):
        df = create_lag_features(sample_df.copy(), empty_report)
        assert len(df) == len(sample_df)


# Group 7: Interactions
class TestInteractionFeatures:

    def _prepare(self, sample_df, empty_report):
        df = create_temporal_features(sample_df.copy(), empty_report)
        df = create_holiday_features(df, empty_report)
        df = create_promotion_features(df, empty_report)
        df = create_store_dept_features(df, empty_report)
        df = create_economic_features(df, empty_report)
        return df

    def test_columns_created(self, sample_df, empty_report):
        df = self._prepare(sample_df, empty_report)
        df = create_interaction_features(df, empty_report)
        expected = {
            "Holiday_Size",
            "Holiday_Type",
            "Promo_Holiday",
            "Temp_Season",
            "Econ_Size",
            "MarkDown_Intensity",
        }
        assert expected <= set(df.columns)

    def test_no_nulls(self, sample_df, empty_report):
        df = self._prepare(sample_df, empty_report)
        df = create_interaction_features(df, empty_report)
        interaction_cols = [
            "Holiday_Size",
            "Holiday_Type",
            "Promo_Holiday",
            "Temp_Season",
            "Econ_Size",
            "MarkDown_Intensity",
        ]
        assert df[interaction_cols].isna().sum().sum() == 0


# Group 8: Cyclical
class TestCyclicalFeatures:

    def test_columns_created(self, sample_df, empty_report):
        df = create_temporal_features(sample_df.copy(), empty_report)
        df = create_cyclical_features(df, empty_report)
        expected = {"Month_sin", "Month_cos", "Week_sin", "Week_cos"}
        assert expected <= set(df.columns)

    def test_sin_cos_range(self, sample_df, empty_report):
        df = create_temporal_features(sample_df.copy(), empty_report)
        df = create_cyclical_features(df, empty_report)
        for col in ["Month_sin", "Month_cos", "Week_sin", "Week_cos"]:
            assert df[col].between(-1.01, 1.01).all(), f"{col} out of [-1, 1]"


# Full Pipeline
class TestFullPipeline:

    def test_no_rows_dropped(self, sample_df):
        result = run_feature_engineering(sample_df.copy())
        assert len(result) == len(sample_df)

    def test_no_nulls(self, sample_df):
        result = run_feature_engineering(sample_df.copy())
        assert result.isna().sum().sum() == 0

    def test_no_infinities(self, sample_df):
        result = run_feature_engineering(sample_df.copy())
        numeric = result.select_dtypes(include=[np.number])
        assert np.isinf(numeric).sum().sum() == 0

    def test_target_preserved(self, sample_df):
        result = run_feature_engineering(sample_df.copy())
        pd.testing.assert_series_equal(
            result["Sales_Class"].reset_index(drop=True),
            sample_df["Sales_Class"].reset_index(drop=True),
            check_names=False,
        )

    def test_more_columns_than_before(self, sample_df):
        result = run_feature_engineering(sample_df.copy())
        assert len(result.columns) > len(sample_df.columns)

    def test_original_columns_kept(self, sample_df):
        result = run_feature_engineering(sample_df.copy())
        missing = [c for c in sample_df.columns if c not in result.columns]
        assert len(missing) <= 2, f"Too many original columns missing: {missing}"
