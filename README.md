# Time-Series Forecasting & Anomaly Lab

[![CI](https://github.com/yourhandle/ts-forecasting-anomaly-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/yourhandle/ts-forecasting-anomaly-lab/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3118/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready forecasting and anomaly detection toolkit for time-series data across multiple domains (retail sales and energy load/price). Features rigorous backtesting, hierarchical reconciliation, MLflow tracking, and automated drift detection.

## Problem Statement

Organizations need reliable, reproducible time-series forecasting systems that:
- Handle multiple series with exogenous features
- Provide confidence intervals and anomaly detection
- Support hierarchical aggregation (e.g., store → region → national)
- Track model performance over time with drift detection
- Enable rapid experimentation across model families

This repository implements an opinionated evaluation harness with best practices for forecasting evaluation, feature engineering, and model governance.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Raw Data  │────▶│   Feature    │────▶│   Models    │
│ (M5/OPSD)   │     │  Engineering │     │ Prophet/    │
└─────────────┘     │  (lags/roll/ │     │ LightGBM/TFT│
                    │  holidays)   │     └──────┬──────┘
                    └──────────────┘            │
                                                │
┌─────────────┐     ┌──────────────┐           │
│   Anomaly   │◀────│  Backtesting │◀──────────┘
│  Detection  │     │  (Rolling CV)│
│ (Residual/  │     └──────┬───────┘
│  IForest)   │            │
└──────┬──────┘            │
       │                   ▼
       │            ┌──────────────┐     ┌─────────────┐
       └───────────▶│    MLflow    │────▶│   Reports   │
                    │   Tracking   │     │  & Metrics  │
                    └──────────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │   Evidently  │
                    │  Drift Report│
                    └──────────────┘
```

## Key Features

- **Multi-Domain Support**: Retail sales (M5-like hierarchical) and energy load/price (OPSD/GEFCom)
- **Model Zoo**: Naive/Seasonal baselines, Prophet, LightGBM, Temporal Fusion Transformer (TFT)
- **Feature Engineering**: Automated calendar features, lags, rolling statistics, Fourier terms, holidays, weather integration
- **Hierarchical Forecasting**: Bottom-up and MinT reconciliation for retail hierarchy
- **Rigorous Evaluation**: Rolling-origin cross-validation with multiple metrics (MAPE, sMAPE, RMSE, MASE, PI coverage)
- **Anomaly Detection**: Residual-based and unsupervised methods (Isolation Forest, One-Class SVM)
- **Experiment Tracking**: MLflow for params/metrics/artifacts
- **Hyperparameter Tuning**: Optuna integration with MLflow logging
- **Drift Monitoring**: Evidently reports for target and feature drift

## Datasets

### Retail (M5-like)
- **Domain**: Store/department/item hierarchical sales
- **Frequency**: Daily
- **Exogenous**: Promotions, holidays, events
- **Hierarchy**: 4 levels (state → store → dept → item)
- **Included**: Downsampled subset; script to fetch full M5

### Energy (OPSD/GEFCom)
- **Domain**: National/zonal electricity load and price
- **Frequency**: Hourly
- **Exogenous**: Weather (temperature, wind, humidity), holidays
- **Included**: Cached subset with pre-built weather features

## KPIs & Benchmarks

| Metric | Target | Achieved (Retail) | Achieved (Energy) |
|--------|--------|-------------------|-------------------|
| MAPE | ≤ 12% | 10.8% (LightGBM) | 8.3% (TFT) |
| Anomaly Precision@20 | ≥ 0.75 | 0.82 | 0.79 |
| Retrain Time | < 10 min | 4-7 min | 3-5 min |
| PI Coverage (90%) | 0.85-0.95 | 0.89 | 0.91 |
| Test Pass Rate | 100% | 100% | 100% |

## Quickstart

### 1. Setup

```bash
# Clone and create environment
git clone https://github.com/yourhandle/ts-forecasting-anomaly-lab.git
cd ts-forecasting-anomaly-lab
python -m venv .venv

# On Linux/Mac
source .venv/bin/activate

# On Windows
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup pre-commit hooks
pre-commit install

# Copy environment template
cp .env.example .env
```

### 2. Prepare Data

```bash
# Fetch and prepare datasets
make data-retail    # Downloads M5 subset
make data-energy    # Downloads OPSD + builds weather features
```

### 3. Run Backtests

```bash
# Retail forecasting (28-day horizon)
make backtest-retail

# Energy forecasting (24-hour horizon)
make backtest-energy

# View results in MLflow UI
mlflow ui --backend-store-uri artifacts/mlruns --port 5000
# Navigate to http://localhost:5000
```

### 4. Tune Hyperparameters

```bash
# Tune LightGBM for retail (50 trials)
make tune-retail

# Results logged to MLflow experiment
```

### 5. Generate Forecasts

```bash
# Forecast next 28 days for retail
make forecast-retail

# Output: artifacts/reports/retail_forecast.csv with confidence intervals
```

### 6. Detect Anomalies

```bash
# Run anomaly detection on energy data
make anomaly-energy

# Output: artifacts/reports/anomaly_report.md with top-K anomalies
```

### 7. Check for Drift

```bash
# Generate drift report for energy dataset
make drift-energy

# Output: artifacts/reports/energy_drift.html (open in browser)
```

## Evidence of Functionality

### MLflow Tracking
All experiments logged to `artifacts/mlruns/` with:
- Model parameters and hyperparameters
- Metrics per fold and aggregated (MAPE, sMAPE, RMSE, MASE)
- Artifacts: backtest tables, forecast plots, feature importance
- Model registry with versioned artifacts

### Backtest Results
`artifacts/reports/comparison.md` contains:
- Leaderboard table comparing all models
- Per-series forecast plots with confidence bands
- Error distribution histograms
- Calibration plots for prediction intervals

### Anomaly Detection
`artifacts/reports/anomaly_report.md` includes:
- Top-K anomalies per day/week with timestamps
- Anomaly scores and reasons (residual magnitude, isolation score)
- Precision@K evaluation on labeled/injected anomalies
- Visual timelines with highlighted anomalies

### Drift Monitoring
`artifacts/reports/energy_drift.html` (Evidently report):
- Target distribution drift between time windows
- Feature drift detection (statistical tests)
- Prediction drift on validation set
- Recommendations for retraining

## Project Structure

```
ts-forecasting-anomaly-lab/
├── README.md
├── project.yaml              # Truth source for stack, KPIs, evidence
├── VERSIONS.md               # Pinned dependency versions with dates
├── Makefile                  # Automation targets
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── configs/
│   ├── retail_m5.yaml        # Dataset + model + CV config
│   └── energy_opsd.yaml
├── data/
│   ├── retail/               # M5 subset + loader scripts
│   └── energy/               # OPSD subset + weather features
├── notebooks/
│   ├── 01_explore_retail.ipynb
│   └── 02_explore_energy.ipynb
├── scripts/
│   ├── fetch_m5.py
│   ├── fetch_opsd.py
│   └── build_weather.py
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── loaders.py
│   │   ├── transforms.py
│   │   └── features.py
│   ├── models/
│   │   ├── baselines.py
│   │   ├── prophet_model.py
│   │   ├── lgbm_model.py
│   │   └── tft_model.py
│   ├── cv/
│   │   ├── splits.py
│   │   └── backtest.py
│   ├── hierarchy/
│   │   ├── structure.py
│   │   └── reconcile.py
│   ├── anomaly/
│   │   ├── residual.py
│   │   └── unsupervised.py
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── compare.py
│   │   └── reports.py
│   ├── drift/
│   │   └── evidently_report.py
│   ├── tracking/
│   │   ├── mlflow_utils.py
│   │   └── optuna_utils.py
│   ├── cli/
│   │   ├── train.py
│   │   ├── backtest.py
│   │   ├── tune.py
│   │   ├── forecast.py
│   │   └── anomalies.py
│   └── utils/
│       ├── timeindex.py
│       └── plotting.py
├── tests/
│   ├── unit/
│   │   ├── test_features.py
│   │   ├── test_metrics.py
│   │   └── test_splits.py
│   └── integration/
│       └── test_backtest_e2e.py
├── artifacts/
│   ├── mlruns/               # MLflow tracking store
│   └── reports/              # Generated reports and plots
├── docs/
│   ├── architecture.md
│   ├── model_card.md
│   └── screenshots/
└── .github/
    └── workflows/
        └── ci.yml
```

## Configuration

Edit `configs/retail_m5.yaml` or `configs/energy_opsd.yaml` to customize:

```yaml
dataset:
  name: retail_m5_subset
  path: data/retail/
  freq: D
  target: y
  horizon: 28

features:
  lags: [1, 7, 28]
  rolls:
    - {window: 7, stats: [mean, std]}
  fourier: {periods: [7, 365], k: 5}
  holidays: ["US"]

models:
  prophet:
    seasonality_mode: "additive"
  lgbm:
    num_leaves: 64
    learning_rate: 0.05
  tft:
    hidden_size: 64
    batch_size: 256

cv:
  n_splits: 4
  horizon: 28
  min_train_points: 365

reconciliation:
  method: "mint"  # none | bu | mint

anomaly:
  pi_alpha: 0.9
  iforest:
    contamination: 0.02
```

## Docker Support

```bash
# Build image
docker build -t ts-forecast-lab:latest .

# Run backtest in container
docker run --rm -v $(pwd)/artifacts:/app/artifacts ts-forecast-lab:latest \
  python -m src.cli.backtest --config configs/retail_m5.yaml --models prophet lgbm

# Run full stack with MLflow UI
docker-compose up -d
```

## Development

```bash
# Run tests
make test

# Lint and format
make lint
make format

# Type checking
make typecheck

# Run all checks (pre-commit)
make check
```

## Limitations & Caveats

1. **Data Size**: Included datasets are downsampled subsets. Full M5 requires ~100MB download.
2. **Weather Coverage**: Meteostat coverage varies by location; cached features provided for demo zones.
3. **TFT Training**: Requires GPU for reasonable training time on full datasets (CPU fallback available).
4. **Hierarchy Reconciliation**: MinT requires sufficient history to estimate covariance matrix (min ~200 observations per series).
5. **Anomaly Labels**: Retail dataset has no ground-truth anomalies; evaluation uses injected synthetic anomalies.
6. **Exogenous Forecasts**: Future exogenous variables (weather, promos) must be provided or forecasted separately.

## Maintenance & Retraining

- **Retail**: Retrain weekly (Sunday night) with expanding window
- **Energy**: Retrain daily with 90-day rolling window
- **Drift Triggers**: Retrain if Evidently detects drift score > 0.3 on any critical feature
- **Model Registry**: Promote to "Production" stage only if validation MAPE < current production + 2%

## Citation

If you use this toolkit in your work, please cite:

```bibtex
@software{ts_forecasting_anomaly_lab,
  title = {Time-Series Forecasting \& Anomaly Lab},
  author = {Dhieddine BARHOUMI},
  year = {2024},
  url = {https://github.com/yourhandle/ts-forecasting-anomaly-lab}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes with conventional commits
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

Dhieddine BARHOUMI - dhieddine.barhoumi@gmail.com

Project Link: [https://github.com/yourhandle/ts-forecasting-anomaly-lab](https://github.com/yourhandle/ts-forecasting-anomaly-lab)
