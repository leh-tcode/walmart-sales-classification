# Walmart Sales Classification — Phase 1 Report Draft

**Date:** 2026-04-24  
**Project:** Walmart Weekly Sales Classification (High vs Low)  
**Stage:** Phase 1 (Acquisition, Integration, Validation)

---

## 1) Executive Summary
Phase 1 builds a validated training dataset for classification by combining Walmart internal sales data with macroeconomic indicators from FRED. The pipeline is automated, tested, and produces both merged data artifacts and structured validation outputs.

---

## 2) Project Scope and Objective
Goal: classify each weekly store-department record as:
- **1 (High):** `Weekly_Sales` above store median
- **0 (Low):** `Weekly_Sales` at or below store median

This store-relative target avoids bias from store size differences.

---

## 3) Data Sources
1. **Kaggle Walmart Store Sales Forecasting**
   - `data/raw/train.csv`
   - `data/raw/stores.csv`
   - `data/raw/features.csv`
2. **FRED API (St. Louis Fed)**
   - `UMCSENT`
   - `RSXFS`
   - `PCE`

Date scope aligned to Walmart period: **2010-02-05 to 2012-11-02**.

---

## 4) Acquisition and Integration

### Completed
- Merged Walmart internal files (`train + stores + features`)
- Merged Walmart weekly data with FRED monthly data
- Created `Sales_Class` target
- Saved final merged dataset

### Merge strategy
- `LEFT JOIN` on `Store` for `train + stores`
- `LEFT JOIN` on `Store, Date` for features
- `merge_asof(direction="backward")` on `Date` for Walmart + FRED

### Diagnostics added
- Row counts before/after every merge
- Unmatched row counts
- Null-introduced rows/columns reporting
- Intermediate datasets saved
- Integration report written with exact strategy and counts

### Output artifacts
- `data/processed/merged_dataset.csv`
- `data/processed/integration_report.txt`
- `data/processed/intermediate/walmart_train_stores_merged.csv`
- `data/processed/intermediate/walmart_internal_merged.csv`
- `data/processed/intermediate/fred_combined.csv`
- `data/processed/intermediate/walmart_fred_merged.csv`

---

## 5) Validation and Documentation (Refined)

This section clarifies **what each validation concept checks** and **where it is tested**.

| Validation concept | Implemented in | Tested in |
|---|---|---|
| Shape / minimum size | `check_shape()` | `TestCheckShape` |
| Required schema (required columns) | `REQUIRED_COLUMNS`, `check_required_schema()` | `TestCheckRequiredSchema` |
| Missing values (column-level) | `check_missing_values()` | `TestCheckMissingValues` |
| Missing values (row-level) | `check_row_level_missingness()` | `TestRowLevelAndSevereMissingness.test_row_level_missingness_warns` |
| Severe missingness thresholds | `SEVERE_COLUMN_MISSINGNESS_PCT`, `SEVERE_ROW_MISSINGNESS_PCT`, `check_severe_missingness_thresholds()` | `TestRowLevelAndSevereMissingness.test_severe_missingness_detects_columns` |
| Duplicates | `check_duplicates()` | `TestCheckDuplicates` |
| Data type profile | `check_data_types()` | Covered in end-to-end run test |
| Strict dtype expectations | `STRICT_DTYPE_EXPECTATIONS`, `_dtype_matches()`, `check_strict_dtypes()` | `TestStrictDtypes` |
| Date / temporal validity | `check_date_range()` | `TestCheckDateRange` |
| Domain rule: negative sales | `check_negative_sales()` | `TestCheckNegativeSales` |
| Value ranges and outlier flags | `check_value_ranges()` | Covered in end-to-end run test |
| Target validity (`Sales_Class` exists, only `{0,1}`) | `check_target_validity()` | `TestTargetAndCategoricalValidity.test_target_validity_*` |
| Class balance | `check_class_distribution()` | `TestCheckClassDistribution` |
| Categorical domain checks (`Type`, `IsHoliday`) | `check_categorical_domains()` | `TestTargetAndCategoricalValidity.test_categorical_domain_warns` |
| External feature coverage (FRED columns) | `check_fred_coverage()` | `TestCheckFredCoverage` |
| Referential integrity (`Store`, `Dept`) | `check_referential_integrity()` | `TestCheckReferentialIntegrity` |
| Full validation pipeline and artifacts | `run_validation()` | `TestValidationOutputs.test_run_validation_writes_json_and_csv` |

### Validation outputs
- `data/processed/validation_report.txt` (human-readable)
- `data/processed/validation_summary.json` (structured summary)
- `data/processed/validation_summary.csv` (tabular summary)

### Validation snapshots included in report
- Top missing columns
- Schema (`column -> dtype`)
- Sample rows (`head`)

---

## 6) Testing and CI
- Unit tests in `tests/test_acquisition.py` and `tests/test_validation.py`
- Validation module tests passing locally (`29 passed`)
- CI workflow runs lint + formatting checks + tests

---

## 7) Current Status Summary
- Acquisition and integration requirements: **completed**
- Validation requirements (including stricter schema/dtype/target checks): **completed**
- Documentation and output artifacts: **completed for Phase 1**

---

## 8) Next Actions (Phase 2)
1. Missing value treatment policy per feature group
2. Outlier handling strategy for model-ready features
3. Train/validation split and feature pipeline
4. Baseline classification models and metrics
5. Error analysis and feature importance reporting

---

## 9) File References
- `src/data/acquisition.py`
- `src/validation/validator.py`
- `tests/test_acquisition.py`
- `tests/test_validation.py`
- `data/processed/merged_dataset.csv`
- `data/processed/integration_report.txt`
- `data/processed/validation_report.txt`
- `data/processed/validation_summary.json`
- `data/processed/validation_summary.csv`
