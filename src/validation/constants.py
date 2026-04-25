from pathlib import Path

PROCESSED_DIR = Path("data/processed")
REPORT_PATH = PROCESSED_DIR / "validation_report.txt"
JSON_SUMMARY_PATH = PROCESSED_DIR / "validation_summary.json"
CSV_SUMMARY_PATH = PROCESSED_DIR / "validation_summary.csv"

REQUIRED_COLUMNS = [
    "Store", "Dept", "Date", "Weekly_Sales", "IsHoliday", "Type",
    "Size", "Temperature", "Fuel_Price", "MarkDown1", "MarkDown2",
    "MarkDown3", "MarkDown4", "MarkDown5", "CPI", "Unemployment",
    "UMCSENT", "RSXFS", "PCE", "Sales_Class",
]

STRICT_DTYPE_EXPECTATIONS = {
    "Store": "integer", "Dept": "integer", "Date": "datetime",
    "Weekly_Sales": "numeric", "IsHoliday": "bool", "Type": "string",
    "Size": "integer", "Temperature": "numeric", "Fuel_Price": "numeric",
    "MarkDown1": "numeric", "MarkDown2": "numeric", "MarkDown3": "numeric",
    "MarkDown4": "numeric", "MarkDown5": "numeric", "CPI": "numeric",
    "Unemployment": "numeric", "UMCSENT": "numeric", "RSXFS": "numeric",
    "PCE": "numeric", "Sales_Class": "integer",
}

VALUE_RANGES = {
    "Store":         (1, 45),
    "Dept":          (1, 99),
    "Weekly_Sales":  (-50_000, 700_000),
    "Temperature":   (-10.0, 120.0),
    "Fuel_Price":    (1.0, 6.0),
    "CPI":           (100.0, 250.0),
    "Unemployment":  (2.0, 20.0),
    "Size":          (1_000, 300_000),
    "UMCSENT":       (40.0, 120.0),
    "RSXFS":         (200_000.0, 500_000.0),
    "PCE":           (5_000.0, 20_000.0),
}

COMPLETENESS_THRESHOLDS = {
    "Store": 0.0, "Dept": 0.0, "Date": 0.0, "Weekly_Sales": 0.0,
    "IsHoliday": 0.0, "Type": 0.0, "Size": 0.0, "Sales_Class": 0.0,
    "Temperature": 1.0, "Fuel_Price": 1.0, "CPI": 5.0,
    "Unemployment": 5.0, "UMCSENT": 5.0, "RSXFS": 5.0, "PCE": 5.0,
    "MarkDown1": 75.0, "MarkDown2": 75.0, "MarkDown3": 75.0,
    "MarkDown4": 75.0, "MarkDown5": 75.0,
}

SEVERE_COLUMN_MISSINGNESS_PCT = 40.0
SEVERE_ROW_MISSINGNESS_PCT = 30.0

IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0
OUTLIER_THRESHOLD_PCT = 5.0

NUMERIC_COLS = [
    "Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "Size", "UMCSENT", "RSXFS", "PCE",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
]

EXPECTED_POSITIVE_CORRELATIONS = [
    ("Size", "Weekly_Sales"),
    ("RSXFS", "PCE"),
]

EXPECTED_NEGATIVE_CORRELATIONS = [
    ("Unemployment", "Weekly_Sales"),
]
