.PHONY: setup data-retail data-energy backtest-retail backtest-energy tune-retail forecast-retail anomaly-energy drift-energy test lint format typecheck check all clean

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	pre-commit install

data-retail:
	python scripts/fetch_m5.py --out data/retail/ --subset small

data-energy:
	python scripts/fetch_opsd.py --out data/energy/
	python scripts/build_weather.py --source data/energy/ --out data/energy/weather.parquet

backtest-retail:
	python -m src.cli.backtest --config configs/retail_m5.yaml --models prophet lgbm tft

backtest-energy:
	python -m src.cli.backtest --config configs/energy_opsd.yaml --models prophet lgbm

tune-retail:
	python -m src.cli.tune --config configs/retail_m5.yaml --model lgbm --trials 50

forecast-retail:
	python -m src.cli.forecast --config configs/retail_m5.yaml --horizon 28 --save artifacts/reports/retail_forecast.csv

anomaly-energy:
	python -m src.cli.anomalies --config configs/energy_opsd.yaml --method residual --k 20
	python -m src.cli.anomalies --config configs/energy_opsd.yaml --method iforest --k 20

drift-energy:
	python -m src.drift.evidently_report --dataset energy --out artifacts/reports/energy_drift.html

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	ruff check src/ tests/ scripts/
	black --check src/ tests/ scripts/

format:
	ruff check --fix src/ tests/ scripts/
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

typecheck:
	mypy src/ --ignore-missing-imports

check: lint typecheck test

mlflow-ui:
	mlflow ui --backend-store-uri artifacts/mlruns --port 5000

docker-build:
	docker build -t ts-forecast-lab:latest .

docker-run-backtest:
	docker run --rm -v $$(pwd)/artifacts:/app/artifacts ts-forecast-lab:latest \
		python -m src.cli.backtest --config configs/retail_m5.yaml --models prophet lgbm

all: setup data-retail data-energy backtest-retail backtest-energy

clean:
	rm -rf .venv __pycache__ .pytest_cache .ruff_cache .mypy_cache htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
