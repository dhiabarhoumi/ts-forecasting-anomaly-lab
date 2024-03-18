"""Integration test for end-to-end backtest."""

import pytest
import pandas as pd
import numpy as np

from src.data.features import build_features
from src.cv.splits import rolling_origin_split
from src.eval.metrics import calculate_metrics


def test_backtest_e2e(sample_timeseries, sample_config):
    """Test end-to-end backtest pipeline."""
    # Build features
    df = build_features(
        sample_timeseries,
        sample_config["features"],
        ts_col="ds",
        target_col="y",
    )
    
    # Remove rows with NaN from lags
    df = df.dropna()
    
    assert len(df) > 0, "Features should produce non-empty dataframe"
    
    # Generate CV splits
    splits = list(rolling_origin_split(
        df,
        n_splits=sample_config["cv"]["n_splits"],
        horizon=sample_config["cv"]["horizon"],
        min_train_points=sample_config["cv"]["min_train_points"],
        ts_col="ds",
    ))
    
    assert len(splits) > 0, "Should generate at least one split"
    
    # Test each split
    for train, test in splits:
        assert len(train) >= sample_config["cv"]["min_train_points"]
        assert len(test) == sample_config["cv"]["horizon"]
        
        # Simple baseline forecast (naive)
        last_value = train["y"].iloc[-1]
        predictions = np.full(len(test), last_value)
        
        # Calculate metrics
        metrics = calculate_metrics(
            test["y"].values,
            predictions,
            y_train=train["y"].values,
        )
        
        assert "mape" in metrics
        assert "rmse" in metrics
        assert metrics["mape"] >= 0
        assert metrics["rmse"] >= 0
