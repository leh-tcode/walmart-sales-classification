#  Walmart Sales Classification — Makefile

.PHONY: acquire validate cleaning features preprocessing eda full full-no-acquire test lint format clean

## Run data acquisition + FRED merge
acquire:
	poetry run python -m src.data.acquisition

## Run full validation report
validate:
	poetry run python -m src.validation.validator

## Run cleaning
cleaning:
	poetry run python -m src.cleaning.cleaning

## Run feature engineering
features:
	poetry run python -m src.features.feature_engineering

## Run preprocessing
preprocessing:
	poetry run python -m src.features.preprocessing

## Run exploratory data analysis
eda:
	poetry run python -m src.eda.eda

## Train
train:
	poetry run jupyter nbconvert --to notebook --execute --inplace src/models/*.ipynb

## Run full Phase 1 
full: acquire validate cleaning features preprocessing eda train
	@echo "full pipeline complete."

## Run full Phase 1 pipeline without data acquisition
full-no-acquire: validate cleaning features preprocessing eda
	@echo "full pipeline complete (without acquisition)."

## Run full Phase 1 pipeline without train
full-no-train:  acquire validate cleaning features preprocessing eda 
	@echo "full pipeline complete (without train)."
## Run tests with coverage
test:
	poetry run pytest tests/ --cov=src --cov-report=term-missing -v

## Lint check
lint:
	poetry run ruff check src/ tests/ 

## Auto-format code
format:
	poetry run ruff check src/ tests/ --fix
	poetry run ruff format src/ tests/

## Clean generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f data/processed/*.csv
	rm -f logs/*.log
