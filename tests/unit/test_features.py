"""Unit tests for feature engineering."""

import pandas as pd
import numpy as np
import pytest

from src.data.features import create_lag_features, create_rolling_features, build_features


def test_create_lag_features():
    """Test lag feature creation."""
    df = pd.DataFrame({
        "ds": pd.date_range("2024-01-01", periods=10, freq="D"),
        "y": range(10),
    })
    
    result = create_lag_features(df, lags=[1, 7], target_col="y")
    
    assert "lag_1" in result.columns
    assert "lag_7" in result.columns
    assert result["lag_1"].iloc[1] == 0
    assert result["lag_1"].iloc[5] == 4


def test_create_rolling_features():
    """Test rolling window features."""
    df = pd.DataFrame({
        "ds": pd.date_range("2024-01-01", periods=20, freq="D"),
        "y": range(20),
    })
    
    windows = [{"window": 7, "stats": ["mean", "std"]}]
    result = create_rolling_features(df, windows, target_col="y")
    
    assert "rolling_7_mean" in result.columns
    assert "rolling_7_std" in result.columns
    
    # Check calculation
    assert result["rolling_7_mean"].iloc[6] == pytest.approx(3.0)


def test_build_features():
    """Test full feature building pipeline."""
    df = pd.DataFrame({
        "ds": pd.date_range("2024-01-01", periods=30, freq="D"),
        "y": np.random.randn(30) * 10 + 100,
    })
    
    config = {
        "lags": [1, 7],
        "rolls": [{"window": 7, "stats": ["mean"]}],
        "fourier": {"periods": [7], "k": 2},
    }
    
    result = build_features(df, config)
    
    # Check calendar features
    assert "dayofweek" in result.columns
    assert "month" in result.columns
    
    # Check lag features
    assert "lag_1" in result.columns
    assert "lag_7" in result.columns
    
    # Check rolling features
    assert "rolling_7_mean" in result.columns
    
    # Check Fourier features
    assert "fourier_sin_7_1" in result.columns
    assert "fourier_cos_7_1" in result.columns
