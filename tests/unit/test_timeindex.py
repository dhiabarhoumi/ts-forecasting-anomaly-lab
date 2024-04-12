"""Unit tests for time index utilities."""

import pandas as pd
import pytest
from src.utils.timeindex import infer_frequency, get_calendar_features, create_fourier_features


def test_infer_frequency_daily():
    """Test frequency inference for daily data."""
    ds = pd.date_range("2024-01-01", periods=30, freq="D")
    freq = infer_frequency(ds)
    assert freq == "D"


def test_infer_frequency_hourly():
    """Test frequency inference for hourly data."""
    ds = pd.date_range("2024-01-01", periods=48, freq="H")
    freq = infer_frequency(ds)
    assert freq == "H"


def test_get_calendar_features():
    """Test calendar feature extraction."""
    ds = pd.Series(pd.date_range("2024-01-01", periods=7, freq="D"))
    features = get_calendar_features(ds)
    
    assert "dayofweek" in features.columns
    assert "month" in features.columns
    assert "is_weekend" in features.columns
    
    # Monday is 0, Saturday is 5
    assert features["dayofweek"].iloc[5] == 5  # Saturday
    assert features["is_weekend"].iloc[5] == 1


def test_create_fourier_features():
    """Test Fourier feature creation."""
    ds = pd.Series(pd.date_range("2024-01-01", periods=14, freq="D"))
    features = create_fourier_features(ds, periods=[7], k=2)
    
    assert "fourier_sin_7_1" in features.columns
    assert "fourier_cos_7_1" in features.columns
    assert "fourier_sin_7_2" in features.columns
    assert "fourier_cos_7_2" in features.columns
    
    assert len(features) == 14
