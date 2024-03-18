"""Test configuration."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_timeseries():
    """Create sample time series data."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    data = {
        "ds": dates,
        "y": np.random.randn(100) * 10 + 50,
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_config():
    """Create sample configuration."""
    return {
        "dataset": {
            "name": "test_data",
            "path": "data/test/",
            "freq": "D",
            "target": "y",
            "horizon": 7,
        },
        "features": {
            "lags": [1, 7],
            "rolls": [{"window": 7, "stats": ["mean"]}],
        },
        "cv": {
            "n_splits": 3,
            "horizon": 7,
            "min_train_points": 30,
        },
    }
