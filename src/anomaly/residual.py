"""Residual-based anomaly detection."""

import numpy as np
import pandas as pd
from typing import Tuple


def detect_anomalies_residual(
    actuals: np.ndarray,
    predictions: np.ndarray,
    method: str = "quantile",
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect anomalies based on forecast residuals.
    
    Args:
        actuals: Actual values
        predictions: Predicted values
        method: Detection method ('quantile' or 'std')
        lower_quantile: Lower quantile threshold
        upper_quantile: Upper quantile threshold
    
    Returns:
        Tuple of (anomaly_flags, anomaly_scores)
    """
    residuals = actuals - predictions
    
    if method == "quantile":
        lower_bound = np.quantile(residuals, lower_quantile)
        upper_bound = np.quantile(residuals, upper_quantile)
        
        anomalies = (residuals < lower_bound) | (residuals > upper_bound)
        scores = np.abs(residuals)
    
    elif method == "std":
        mean_resid = np.mean(residuals)
        std_resid = np.std(residuals)
        z_threshold = 3
        
        z_scores = np.abs((residuals - mean_resid) / std_resid)
        anomalies = z_scores > z_threshold
        scores = z_scores
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return anomalies, scores


def detect_anomalies_pi(
    actuals: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    """Detect anomalies outside prediction intervals.
    
    Args:
        actuals: Actual values
        lower_bounds: Lower PI bounds
        upper_bounds: Upper PI bounds
    
    Returns:
        Boolean array of anomaly flags
    """
    return (actuals < lower_bounds) | (actuals > upper_bounds)
