"""Evaluation metrics for time series forecasting."""

import numpy as np
import pandas as pd
from typing import Optional


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    mask = y_true != 0
    if not mask.any():
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred))
    mask = denominator != 0
    if not mask.any():
        return np.inf
    return np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonal_period: int = 1,
) -> float:
    """Mean Absolute Scaled Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        y_train: Training data for scaling
        seasonal_period: Seasonal period for naive forecast
    
    Returns:
        MASE value
    """
    mae_forecast = mae(y_true, y_pred)
    
    # Seasonal naive forecast on training data
    naive_forecast = y_train[seasonal_period:]
    naive_actual = y_train[:-seasonal_period]
    mae_naive = mae(naive_actual, naive_forecast)
    
    if mae_naive == 0:
        return np.inf
    
    return mae_forecast / mae_naive


def coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """Prediction interval coverage.
    
    Args:
        y_true: Actual values
        y_lower: Lower bound of prediction interval
        y_upper: Upper bound of prediction interval
    
    Returns:
        Coverage percentage (0-1)
    """
    within_interval = (y_true >= y_lower) & (y_true <= y_upper)
    return np.mean(within_interval)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
    seasonal_period: int = 1,
) -> dict:
    """Calculate all metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        y_train: Training data for MASE
        y_lower: Lower PI bound
        y_upper: Upper PI bound
        seasonal_period: Period for MASE
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
    }
    
    if y_train is not None:
        metrics["mase"] = mase(y_true, y_pred, y_train, seasonal_period)
    
    if y_lower is not None and y_upper is not None:
        metrics["coverage"] = coverage(y_true, y_lower, y_upper)
    
    return metrics
