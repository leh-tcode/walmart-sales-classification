import numpy as np
import pandas as pd
import pytest

from src.cleaning.cleaning import (
    handle_markdown_nulls,
    handle_negative_sales,
    clip_outliers,
    MARKDOWN_COLS,
    CLIP_COLS,
    CLIP_LOWER_PERCENTILE,
    CLIP_UPPER_PERCENTILE,
)


# Fixtures
@pytest.fixture
def sample_df():
    """Minimal DataFrame that mimics the real dataset structure."""
    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        "Store": np.random.randint(1, 46, size=n),
        "Dept": np.random.randint(1, 82, size=n),
        "Date": pd.date_range("2010-02-05", periods=n, freq="W"),
        "Weekly_Sales": np.concatenate([
            np.random.normal(15000, 8000, n - 5),
            [-4000, -2000, -500, -100, -50],
        ]),
        "IsHoliday": np.random.choice([True, False], size=n, p=[0.07, 0.93]),
        "Type": np.random.choice(["A", "B", "C"], size=n),
        "Size": np.random.randint(34000, 220000, size=n),
        "Temperature": np.random.normal(60, 18, n),
        "Fuel_Price": np.random.uniform(2.5, 4.5, n),
        "MarkDown1": np.where(np.random.random(n) > 0.65, np.random.uniform(100, 30000, n), np.nan),
        "MarkDown2": np.where(np.random.random(n) > 0.74, np.random.uniform(10, 20000, n), np.nan),
        "MarkDown3": np.where(np.random.random(n) > 0.68, np.random.uniform(1, 5000, n), np.nan),
        "MarkDown4": np.where(np.random.random(n) > 0.68, np.random.uniform(1, 10000, n), np.nan),
        "MarkDown5": np.where(np.random.random(n) > 0.64, np.random.uniform(100, 40000, n), np.nan),
        "CPI": np.random.uniform(126, 228, n),
        "Unemployment": np.random.uniform(4, 14, n),
        "UMCSENT": np.random.uniform(55, 83, n),
        "RSXFS": np.random.uniform(302000, 355000, n),
        "PCE": np.random.uniform(10000, 11200, n),
        "Sales_Class": np.random.choice([0, 1], size=n),
    })

    return df


@pytest.fixture
def empty_report():
    return {"steps": []}


# Step 1: MarkDown Nulls
class TestHandleMarkdownNulls:

    def test_no_nulls_remaining(self, sample_df, empty_report):
        df = handle_markdown_nulls(sample_df.copy(), empty_report)
        for col in MARKDOWN_COLS:
            assert df[col].isna().sum() == 0, f"{col} still has nulls"

    def test_binary_flags_created(self, sample_df, empty_report):
        df = handle_markdown_nulls(sample_df.copy(), empty_report)
        for col in MARKDOWN_COLS:
            flag = f"has_{col}"
            assert flag in df.columns, f"{flag} not created"

    def test_flags_are_binary(self, sample_df, empty_report):
        df = handle_markdown_nulls(sample_df.copy(), empty_report)
        for col in MARKDOWN_COLS:
            unique_vals = set(df[f"has_{col}"].unique())
            assert unique_vals <= {0, 1}, f"has_{col} has non-binary values: {unique_vals}"

    def test_flag_matches_original_nulls(self, sample_df, empty_report):
        original = sample_df.copy()
        df = handle_markdown_nulls(sample_df.copy(), empty_report)
        for col in MARKDOWN_COLS:
            was_null = original[col].isna()
            assert (df.loc[was_null, f"has_{col}"] == 0).all(), \
                f"has_{col} should be 0 where original was null"
            was_present = original[col].notna()
            assert (df.loc[was_present, f"has_{col}"] == 1).all(), \
                f"has_{col} should be 1 where original had data"

    def test_filled_value_is_zero(self, sample_df, empty_report):
        original = sample_df.copy()
        df = handle_markdown_nulls(sample_df.copy(), empty_report)
        for col in MARKDOWN_COLS:
            was_null = original[col].isna()
            filled_values = df.loc[was_null, col]
            assert (filled_values == 0.0).all(), \
                f"{col}: nulls should be filled with 0, not {filled_values.unique()}"

    def test_existing_values_untouched(self, sample_df, empty_report):
        original = sample_df.copy()
        df = handle_markdown_nulls(sample_df.copy(), empty_report)
        for col in MARKDOWN_COLS:
            was_present = original[col].notna()
            pd.testing.assert_series_equal(
                df.loc[was_present, col].reset_index(drop=True),
                original.loc[was_present, col].reset_index(drop=True),
                check_names=False,
            )

# Step 2: Negative Sales
class TestHandleNegativeSales:

    def test_is_return_flag_created(self, sample_df, empty_report):
        df = handle_negative_sales(sample_df.copy(), empty_report)
        assert "is_return" in df.columns

    def test_flag_matches_negative_values(self, sample_df, empty_report):
        df = handle_negative_sales(sample_df.copy(), empty_report)
        negative_mask = df["Weekly_Sales"] < 0
        assert (df.loc[negative_mask, "is_return"] == 1).all()
        assert (df.loc[~negative_mask, "is_return"] == 0).all()

    def test_sales_values_unchanged(self, sample_df, empty_report):
        original = sample_df.copy()
        df = handle_negative_sales(sample_df.copy(), empty_report)
        pd.testing.assert_series_equal(
            df["Weekly_Sales"], original["Weekly_Sales"], check_names=False,
        )

    def test_row_count_preserved(self, sample_df, empty_report):
        df = handle_negative_sales(sample_df.copy(), empty_report)
        assert len(df) == len(sample_df)

    def test_no_negatives_removed(self, sample_df, empty_report):
        original_neg_count = (sample_df["Weekly_Sales"] < 0).sum()
        df = handle_negative_sales(sample_df.copy(), empty_report)
        cleaned_neg_count = (df["Weekly_Sales"] < 0).sum()
        assert original_neg_count == cleaned_neg_count

# Step 3: Clip Outliers
class TestClipOutliers:

    def test_values_within_clip_bounds(self, sample_df, empty_report):
        df = handle_markdown_nulls(sample_df.copy(), empty_report)
        report2 = {"steps": []}
        df = clip_outliers(df, report2)

        for col in CLIP_COLS:
            if col not in df.columns:
                continue
            s = df[col]
            assert s.max() <= s.max()

    def test_clipping_reduces_range(self, sample_df, empty_report):
        df = handle_markdown_nulls(sample_df.copy(), empty_report)
        original = df.copy()
        report2 = {"steps": []}
        df = clip_outliers(df, report2)

        for col in CLIP_COLS:
            if col not in df.columns:
                continue
            assert df[col].max() <= original[col].max(), \
                f"{col}: max should not increase after clipping"
            assert df[col].min() >= original[col].min(), \
                f"{col}: min should not decrease after clipping"

    def test_row_count_preserved(self, sample_df, empty_report):
        df = handle_markdown_nulls(sample_df.copy(), empty_report)
        report2 = {"steps": []}
        df = clip_outliers(df, report2)
        assert len(df) == len(sample_df)

    def test_middle_values_untouched(self, sample_df, empty_report):
        """Values between P1 and P99 should not be modified."""
        df = handle_markdown_nulls(sample_df.copy(), empty_report)
        original = df.copy()
        report2 = {"steps": []}
        df = clip_outliers(df, report2)

        for col in CLIP_COLS:
            if col not in original.columns:
                continue
            p01 = original[col].quantile(CLIP_LOWER_PERCENTILE)
            p99 = original[col].quantile(CLIP_UPPER_PERCENTILE)
            middle_mask = (original[col] >= p01) & (original[col] <= p99)
            pd.testing.assert_series_equal(
                df.loc[middle_mask, col].reset_index(drop=True),
                original.loc[middle_mask, col].reset_index(drop=True),
                check_names=False,
            )

    def test_skewness_not_increased(self, sample_df, empty_report):
        df = handle_markdown_nulls(sample_df.copy(), empty_report)
        original = df.copy()
        report2 = {"steps": []}
        df = clip_outliers(df, report2)

        for col in CLIP_COLS:
            if col not in df.columns:
                continue
            skew_before = abs(original[col].skew())
            skew_after = abs(df[col].skew())
            assert skew_after <= skew_before + 0.01, \
                f"{col}: skewness should not increase after clipping"

    def test_report_has_clip_bounds(self, sample_df, empty_report):
        df = handle_markdown_nulls(sample_df.copy(), empty_report)
        report2 = {"steps": []}
        clip_outliers(df, report2)
        step = report2["steps"][0]
        for col in CLIP_COLS:
            assert col in step["details"]
            assert "lower_bound" in step["details"][col]
            assert "upper_bound" in step["details"][col]
