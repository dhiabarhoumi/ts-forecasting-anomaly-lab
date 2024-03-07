"""Baseline forecasting models."""

from typing import Optional
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class NaiveForecaster:
    """Naive forecasting model."""
    
    def __init__(self, seasonal_period: int = 1):
        """Initialize naive forecaster.
        
        Args:
            seasonal_period: Seasonal period for seasonal naive
        """
        self.seasonal_period = seasonal_period
        self.last_values = None
    
    def fit(self, y: np.ndarray):
        """Fit the model.
        
        Args:
            y: Training data
        """
        if self.seasonal_period == 1:
            # Simple naive - last value
            self.last_values = y[-1]
        else:
            # Seasonal naive - last seasonal period
            self.last_values = y[-self.seasonal_period:]
        
        return self
    
    def predict(self, horizon: int) -> np.ndarray:
        """Generate forecasts.
        
        Args:
            horizon: Forecast horizon
        
        Returns:
            Array of predictions
        """
        if self.seasonal_period == 1:
            return np.full(horizon, self.last_values)
        else:
            # Repeat seasonal pattern
            n_full_periods = horizon // self.seasonal_period
            remainder = horizon % self.seasonal_period
            
            forecast = np.tile(self.last_values, n_full_periods + 1)
            return forecast[:horizon]


class SeasonalNaiveForecaster(NaiveForecaster):
    """Seasonal naive forecaster."""
    
    def __init__(self, seasonal_period: int = 7):
        """Initialize seasonal naive forecaster.
        
        Args:
            seasonal_period: Seasonal period (default: 7 for weekly)
        """
        super().__init__(seasonal_period=seasonal_period)


class ExponentialSmoothingForecaster:
    """Exponential Smoothing forecaster."""
    
    def __init__(
        self,
        trend: Optional[str] = None,
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
    ):
        """Initialize exponential smoothing forecaster.
        
        Args:
            trend: Trend component ('add', 'mul', or None)
            seasonal: Seasonal component ('add', 'mul', or None)
            seasonal_periods: Number of periods in season
        """
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model = None
        self.fitted_model = None
    
    def fit(self, y: np.ndarray):
        """Fit the model.
        
        Args:
            y: Training data
        """
        try:
            self.model = ExponentialSmoothing(
                y,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
            )
            self.fitted_model = self.model.fit()
        except Exception as e:
            # Fallback to simple exponential smoothing
            self.model = ExponentialSmoothing(y)
            self.fitted_model = self.model.fit()
        
        return self
    
    def predict(self, horizon: int) -> np.ndarray:
        """Generate forecasts.
        
        Args:
            horizon: Forecast horizon
        
        Returns:
            Array of predictions
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        forecast = self.fitted_model.forecast(steps=horizon)
        return forecast.values
    
    def predict_with_intervals(
        self, horizon: int, alpha: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts with prediction intervals.
        
        Args:
            horizon: Forecast horizon
            alpha: Significance level for intervals
        
        Returns:
            Tuple of (forecast, lower_bound, upper_bound)
        """
        forecast = self.predict(horizon)
        
        # Simple approach: use training data residuals for intervals
        fitted_values = self.fitted_model.fittedvalues
        residuals = self.model.endog - fitted_values
        std_residuals = np.std(residuals)
        
        # Wider intervals for longer horizons
        horizon_factor = np.sqrt(np.arange(1, horizon + 1))
        interval_width = 1.96 * std_residuals * horizon_factor
        
        lower = forecast - interval_width
        upper = forecast + interval_width
        
        return forecast, lower, upper
