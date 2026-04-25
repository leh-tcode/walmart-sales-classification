# Walmart Weekly Sales Classification

**Spring 2026 | Applied Data Science Project — Phase 1**

## Team Members
| # | Name | Role |
|---|------|------|
| 1 | Member 1 | Data Acquisition & FRED Integration |
| 2 | Member 2 | Data Validation & Quality Report |
| 3 | Member 3 | Merging Strategy & Pipeline |
| 4 | Member 4 | Report Writing & Documentation |

## Project Description
Binary classification of Walmart weekly store sales as **High** or **Low** relative to each store's own historical median. Combines Kaggle retail data (2010–2012) with FRED macroeconomic indicators (Consumer Sentiment, Retail Sales, PCE).

## Data Sources
1. **Kaggle** – Walmart Store Sales Forecasting: https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data
2. **FRED API** – St. Louis Fed Economic Data: https://fred.stlouisfed.org/

## Setup & Run

### Prerequisites
- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)

### Installation
```bash
git clone <repo-url>
cd walmart-sales-classification
cp .env .env.local   # fill in your FRED_API_KEY
poetry install
```

### Download Kaggle Data
Place the following files in `data/raw/`:
- `train.csv`
- `stores.csv`
- `features.csv`

### Run Phase 1 Pipeline
```bash
make acquire      # Download FRED data + merge with Walmart
make validate     # Run full validation report
make preprocess   # Run preprocessing + feature engineering
make full       # Run all steps sequentially
```

### Acquisition Outputs
After `make acquire`, the pipeline writes:
- `data/processed/merged_dataset.csv` (final merged dataset)
- `data/processed/integration_report.txt` (exact merge strategy + row/null diagnostics)
- `data/processed/intermediate/walmart_train_stores_merged.csv`
- `data/processed/intermediate/walmart_internal_merged.csv`
- `data/processed/intermediate/fred_combined.csv`
- `data/processed/intermediate/walmart_fred_merged.csv`

### Run Tests
```bash
make test
```

### Preprocessing Outputs
After `make preprocess`, the pipeline writes:
- `data/processed/train_preprocessed.csv`
- `data/processed/validation_preprocessed.csv`
- `data/processed/test_preprocessed.csv`
- `data/processed/preprocessing_summary.json`
- `data/processed/preprocessing_summary.txt`

### Preprocessing Notes
- Time-leakage safety: **time-based split** (train → validation → test by date)
- Leakage column dropped: `Weekly_Sales`
- Feature engineering includes date parts (`Year`, `Month`, `Week`, `Quarter`), holiday-season flags, and lag/rolling features
- Applies imputation, outlier capping, categorical encoding, scaling, and feature selection

## Project Structure
```
walmart_sales_classification/
├── src/
│   ├── data/
│   │   ├── acquisition.py      # Kaggle loader + FRED API client
│   │   └── preprocessing.py    # Preprocessing + feature engineering pipeline
│   ├── validation/
│   │   └── validator.py        # Full validation report generator
│   └── utils/
│       └── logger.py           # Loguru-based logging setup
├── tests/
│   ├── test_acquisition.py
│   └── test_validation.py
├── data/
│   ├── raw/                    # Place Kaggle CSVs here
│   └── processed/              # Merged output saved here
├── config/
├── logs/
├── .env
├── Makefile
├── pyproject.toml
└── README.md
```
