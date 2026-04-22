# ─────────────────────────────────────────
#  Walmart Sales Classification — Makefile
# ─────────────────────────────────────────

.PHONY: acquire validate phase1 test lint format clean

## Run data acquisition + FRED merge
acquire:
	poetry run python -m src.data.acquisition

## Run full validation report
validate:
	poetry run python -m src.validation.validator

## Run full Phase 1 pipeline
phase1: acquire validate
	@echo "Phase 1 pipeline complete."

## Run tests with coverage
test:
	poetry run pytest tests/ --cov=src --cov-report=term-missing -v

## Lint check
lint:
	poetry run flake8 src/ tests/ --max-line-length=100

## Auto-format code
format:
	poetry run black src/ tests/
	poetry run isort src/ tests/

## Clean generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f data/processed/*.csv
	rm -f logs/*.log
