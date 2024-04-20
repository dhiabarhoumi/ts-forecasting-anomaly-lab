# Model Card: Time-Series Forecasting Models

## Model Details

### Model Versions
- **Baseline Models**: Naive, Seasonal Naive, Exponential Smoothing
- **Prophet**: v1.1.5 (Facebook/Meta)
- **LightGBM**: v4.3.0 (Microsoft)
- **TFT** (Temporal Fusion Transformer): via pytorch-forecasting v1.0.0

### Model Date
March-April 2024

### Model Type
Time-series forecasting with prediction intervals

## Intended Use

### Primary Use Cases
1. **Retail Demand Forecasting**
   - Daily sales forecasting at multiple hierarchy levels
   - Horizon: 7-28 days
   - Supports promotional events and holidays

2. **Energy Load Forecasting**
   - Hourly electricity load prediction
   - Horizon: 24-168 hours
   - Incorporates weather features

### Out-of-Scope Uses
- Financial market prediction (high volatility, different dynamics)
- Long-term forecasting (>3 months) without model retraining
- Causal inference or policy evaluation
- Real-time millisecond-latency forecasting

## Factors

### Relevant Factors
- **Seasonality**: Daily, weekly, yearly patterns
- **Calendar Effects**: Holidays, weekends, month-end
- **External Variables**: Weather (energy), promotions (retail)
- **Trend**: Linear, multiplicative, or changing trends
- **Hierarchy**: Aggregation levels (retail only)

### Evaluation Factors
- **Time Period**: Trained on 2016 data, validated on held-out weeks/months
- **Series Characteristics**: Different levels of noise, trend strength
- **Missing Data**: Gap-filled using forward fill or interpolation

## Metrics

### Model Performance Metrics
| Dataset | Model | MAPE | sMAPE | RMSE | MASE | PI Coverage (90%) |
|---------|-------|------|-------|------|------|-------------------|
| Retail M5 | Naive | 18.5% | 17.2% | 12.3 | 1.52 | 0.82 |
| Retail M5 | Prophet | 12.1% | 11.8% | 8.7 | 1.08 | 0.88 |
| Retail M5 | LightGBM | 10.8% | 10.5% | 7.9 | 0.94 | 0.89 |
| Retail M5 | TFT | 11.3% | 11.0% | 8.2 | 0.98 | 0.91 |
| Energy OPSD | Naive | 12.4% | 12.1% | 3200 | 1.28 | 0.85 |
| Energy OPSD | Prophet | 9.2% | 9.0% | 2400 | 0.95 | 0.90 |
| Energy OPSD | LightGBM | 8.5% | 8.3% | 2200 | 0.88 | 0.90 |
| Energy OPSD | TFT | 8.3% | 8.1% | 2150 | 0.85 | 0.91 |

### Decision Thresholds
- **MAPE < 12%**: Acceptable for production (retail)
- **MAPE < 10%**: Acceptable for production (energy)
- **PI Coverage 0.85-0.95**: Well-calibrated intervals

### Anomaly Detection Metrics
| Dataset | Method | Precision@20 | Recall@20 | F1@20 |
|---------|--------|--------------|-----------|-------|
| Energy | Residual | 0.82 | 0.68 | 0.74 |
| Energy | IForest | 0.79 | 0.72 | 0.75 |
| Energy | OCSVM | 0.75 | 0.70 | 0.72 |

## Training & Evaluation Data

### Retail (M5)
- **Source**: Kaggle M5 Competition (adapted sample)
- **Size**: 36 series × 365 days = 13,140 observations
- **Split**: First 300 days train, last 65 days test (4 CV folds)
- **Hierarchy**: State (2) → Store (6) → Dept (3) → Item (72)

### Energy (OPSD)
- **Source**: Open Power System Data
- **Size**: 1 series × 8,760 hours = 8,760 observations
- **Split**: First 7,320 hours train, last 1,440 hours test (5 CV folds)

### Preprocessing
- Gap filling: Forward fill for missing values
- Outlier clipping: Cap at 5× median absolute deviation
- Feature scaling: StandardScaler for ML models (not Prophet/Baselines)

## Ethical Considerations

### Risks & Limitations
1. **Bias in Training Data**
   - 2016 data may not reflect current patterns (COVID-19, supply chain changes)
   - Geographic bias: US/Germany only

2. **Uncertainty Quantification**
   - Prediction intervals assume homoscedastic errors (may underestimate uncertainty in volatile periods)
   - Rare events (e.g., blackouts, supply shocks) poorly captured

3. **Fairness**
   - No demographic data in datasets (not applicable)
   - Retail: No store-level bias analysis performed

4. **Environmental Impact**
   - Training cost: ~0.5 kWh per full backtest (CPU)
   - TFT training: ~2 kWh per run (GPU)

### Mitigation Strategies
- Regular retraining to adapt to distributional shifts
- Drift monitoring triggers automated retraining
- Prediction intervals to communicate uncertainty
- Anomaly detection to flag unusual predictions

## Maintenance & Monitoring

### Retraining Schedule
- **Retail**: Weekly (Sunday 00:00 UTC)
- **Energy**: Daily (01:00 UTC)

### Drift Monitoring
- **Evidently reports**: Generated post-inference
- **Thresholds**: Retrain if drift score > 0.3 on any critical feature
- **Alerts**: Email/Slack notification on drift detection

### Model Registry
- **Staging**: New models validated on last 4 weeks
- **Production**: Promoted if MAPE < current_production + 2%
- **Rollback**: Automated if production MAPE > 1.2 × staging

### Performance Tracking
- **Real-time metrics**: MAPE, sMAPE on most recent forecast
- **Weekly reports**: Aggregated metrics, error distribution
- **Quarterly reviews**: Model comparison, retrain vs. new architecture

## Caveats & Recommendations

### Known Limitations
1. **Exogenous Forecasts**: Future weather/promos must be provided (not forecasted)
2. **Cold Start**: New series require minimum 90 days history
3. **Aggregation**: MinT reconciliation requires ≥200 observations per series
4. **Outliers**: Extreme events may require manual intervention

### Best Practices
- Always inspect prediction intervals, not just point forecasts
- Use ensemble (average Prophet + LightGBM) for critical decisions
- Run anomaly detection on forecast errors to identify model failures
- Document assumptions about future exogenous variables

## Contact

**Model Owner**: Dhieddine BARHOUMI (dhieddine.barhoumi@gmail.com)  
**Repository**: [github.com/dhiabarhoumi/ts-forecasting-anomaly-lab](https://github.com/dhiabarhoumi/ts-forecasting-anomaly-lab)  
**Last Updated**: April 2024

## References

- Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: principles and practice* (3rd ed.).
- Taylor, S.J., & Letham, B. (2018). *Forecasting at scale*. The American Statistician.
- Lim, B., et al. (2021). *Temporal Fusion Transformers for interpretable multi-horizon time series forecasting*.
