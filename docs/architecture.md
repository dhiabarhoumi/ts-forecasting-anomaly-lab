# Architecture

## System Overview

The Time-Series Forecasting & Anomaly Lab is designed as a modular, extensible framework for time-series analysis across multiple domains.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Layer                               │
├─────────────────────────────────────────────────────────────────┤
│  • Loaders (M5, OPSD)                                           │
│  • Transforms (alignment, scaling, splits)                      │
│  • Feature Engineering (lags, rolls, calendar, Fourier)         │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                      Model Layer                                 │
├─────────────────────────────────────────────────────────────────┤
│  • Baselines (Naive, Seasonal Naive, ETS)                       │
│  • Statistical (Prophet)                                         │
│  • ML (LightGBM)                                                │
│  • Deep Learning (TFT via pytorch-forecasting)                  │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                   Evaluation Layer                               │
├─────────────────────────────────────────────────────────────────┤
│  • Cross-Validation (Rolling Origin)                            │
│  • Metrics (MAPE, sMAPE, RMSE, MASE, Coverage)                 │
│  • Hierarchical Reconciliation (Bottom-Up, MinT)                │
│  • Anomaly Detection (Residual, IForest, OCSVM)                │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                   Tracking Layer                                 │
├─────────────────────────────────────────────────────────────────┤
│  • MLflow (experiments, runs, artifacts, registry)              │
│  • Optuna (hyperparameter tuning)                               │
│  • Evidently (drift detection)                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Component Interactions

### 1. Data Pipeline
- **Input**: Raw CSV files (M5 sales, OPSD energy)
- **Processing**:
  - Load into tidy long format: `series_id | ds | y | features`
  - Apply transformations (gap filling, alignment)
  - Engineer features (lags, rolling stats, calendar, Fourier)
- **Output**: Feature-rich DataFrame ready for modeling

### 2. Model Training
- **Configuration-driven**: YAML files specify dataset, features, models, CV
- **Abstraction**: Common interface for all models (fit, predict, predict_with_intervals)
- **Feature handling**:
  - Prophet: native regressor support
  - LightGBM: lag-based tabular features
  - TFT: encoder-decoder with static/known/observed features

### 3. Cross-Validation
- **Rolling Origin**: Expanding window with fixed horizon
- **Panel Support**: Splits each series independently
- **Configurable**: n_splits, horizon, min_train_points, step_size

### 4. Hierarchical Forecasting
- **Hierarchy Definition**: Summing matrix S from levels
- **Reconciliation Methods**:
  - Bottom-Up: Aggregate bottom-level forecasts
  - MinT: Trace minimization with shrinkage covariance
- **Application**: Retail hierarchy (item → dept → store → state)

### 5. Anomaly Detection
- **Residual-based**:
  - Quantile thresholds on forecast errors
  - Prediction interval violations
- **Unsupervised**:
  - Isolation Forest on [residuals, features]
  - One-Class SVM
- **Evaluation**: Precision@K on known/injected anomalies

### 6. Experiment Tracking
- **MLflow Integration**:
  - Log params, metrics, artifacts per run
  - Model registry for versioning
  - UI for comparison
- **Artifacts**:
  - Backtest tables (per split/series)
  - Forecast plots with confidence intervals
  - Feature importance
  - Reconciliation reports

### 7. Drift Monitoring
- **Evidently Reports**:
  - Target distribution drift
  - Feature drift (statistical tests)
  - Prediction drift
- **Triggers**: Automated retraining thresholds

## Design Decisions

### Why Rolling Origin CV?
- Respects temporal order (no data leakage)
- Evaluates on multiple forecast horizons
- Mimics production deployment (expanding training window)

### Why Multiple Model Families?
- **Baselines**: Establish performance floor
- **Prophet**: Interpretable, handles holidays/events
- **LightGBM**: Fast, handles tabular features well
- **TFT**: Captures complex interactions, attention mechanism

### Why Hierarchical Reconciliation?
- Ensures forecast coherence across aggregation levels
- Reduces error at aggregate levels
- Provides flexibility in reporting

### Configuration as Code
- Single source of truth (project.yaml, configs/*.yaml)
- Reproducible experiments
- Easy to version and share

## Scalability Considerations

### Current Implementation
- **In-memory**: Suitable for datasets up to ~10M rows
- **Single machine**: CPU for LightGBM/Prophet, GPU optional for TFT
- **Local MLflow**: File-based tracking store

### Production Extensions
- **Distributed**: Spark/Dask for large-scale feature engineering
- **Remote MLflow**: Database backend + artifact storage (S3/GCS)
- **Batch inference**: Parallel processing per series
- **Streaming**: Online feature computation for real-time forecasting

## Security & Privacy
- No PII in sample datasets
- Model artifacts contain no raw data
- Drift reports aggregate statistics only
- Easy to integrate with data anonymization pipelines
