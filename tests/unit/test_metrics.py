"""Unit tests for evaluation metrics."""

import numpy as np
import pytest

from src.eval.metrics import mape, smape, rmse, mae, mase, coverage, calculate_metrics


def test_mape():
    """Test MAPE calculation."""
    y_true = np.array([100, 200, 300, 400])
    y_pred = np.array([110, 190, 310, 390])
    
    result = mape(y_true, y_pred)
    expected = np.mean([10/100, 10/200, 10/300, 10/400]) * 100
    
    assert np.isclose(result, expected, rtol=1e-5)


def test_smape():
    """Test sMAPE calculation."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])
    
    result = smape(y_true, y_pred)
    
    # sMAPE should be between 0 and 200
    assert 0 <= result <= 200


def test_rmse():
    """Test RMSE calculation."""
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    
    result = rmse(y_true, y_pred)
    expected = np.sqrt(np.mean([0.25, 0.25, 0, 1]))
    
    assert np.isclose(result, expected, rtol=1e-5)


def test_mae():
    """Test MAE calculation."""
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    
    result = mae(y_true, y_pred)
    expected = np.mean([0.5, 0.5, 0, 1])
    
    assert np.isclose(result, expected, rtol=1e-5)


def test_coverage():
    """Test prediction interval coverage."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    y_upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    
    result = coverage(y_true, y_lower, y_upper)
    
    # All points should be within interval
    assert result == 1.0


def test_coverage_partial():
    """Test partial coverage."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_lower = np.array([0.5, 1.5, 2.5, 3.5, 5.5])  # Last point outside
    y_upper = np.array([1.5, 2.5, 3.5, 4.5, 6.5])
    
    result = coverage(y_true, y_lower, y_upper)
    
    # 4 out of 5 points within interval
    assert result == 0.8


def test_calculate_metrics():
    """Test calculate_metrics function."""
    y_true = np.array([100, 200, 300, 400])
    y_pred = np.array([110, 190, 310, 390])
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert "mape" in metrics
    assert "smape" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert metrics["mape"] > 0
    assert metrics["rmse"] > 0


def test_mape_with_zeros():
    """Test MAPE with zero values."""
    y_true = np.array([0, 100, 200])
    y_pred = np.array([10, 110, 190])
    
    result = mape(y_true, y_pred)
    
    # Should only calculate for non-zero actuals
    expected = np.mean([10/100, 10/200]) * 100
    assert np.isclose(result, expected, rtol=1e-5)
