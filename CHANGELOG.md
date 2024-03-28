# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-04-15

### Added
- Initial project structure and configuration
- Data loading for M5 and OPSD datasets
- Feature engineering pipeline (lags, rolling stats, calendar, Fourier)
- Baseline models (Naive, Seasonal Naive, Exponential Smoothing)
- Prophet model wrapper
- LightGBM model wrapper
- Rolling origin cross-validation
- Evaluation metrics (MAPE, sMAPE, RMSE, MASE, Coverage)
- MLflow experiment tracking
- Anomaly detection (residual-based, Isolation Forest, One-Class SVM)
- Model comparison and reporting utilities
- Docker and docker-compose support
- GitHub Actions CI pipeline
- Comprehensive test suite
- Documentation (README, architecture, model card)

### Fixed
- Deprecated pandas fillna method parameter
- GitIgnore pattern for models directory

### Documentation
- Architecture diagrams and design decisions
- Model card with performance metrics and limitations
- Setup and quickstart guide
