"""Prophet forecasting model wrapper."""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from prophet import Prophet


class ProphetForecaster:
    """Prophet model wrapper for time series forecasting."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Prophet forecaster.
        
        Args:
            config: Configuration dictionary with Prophet parameters
        """
        self.config = config or {}
        self.model = None
        self.feature_cols = []
    
    def fit(
        self,
        df: pd.DataFrame,
        ts_col: str = "ds",
        target_col: str = "y",
        exog_cols: Optional[list] = None,
    ):
        """Fit Prophet model.
        
        Args:
            df: Training dataframe
            ts_col: Timestamp column name
            target_col: Target column name
            exog_cols: Exogenous feature columns
        """
        # Prepare data for Prophet
        train_df = df[[ts_col, target_col]].copy()
        train_df.columns = ["ds", "y"]
        
        # Initialize model with config
        self.model = Prophet(**self.config)
        
        # Add regressors if provided
        if exog_cols:
            self.feature_cols = exog_cols
            for col in exog_cols:
                self.model.add_regressor(col)
            
            # Add exogenous features to training data
            for col in exog_cols:
                if col in df.columns:
                    train_df[col] = df[col].values
        
        # Fit model
        self.model.fit(train_df)
        
        return self
    
    def predict(
        self,
        horizon: int,
        freq: str = "D",
        exog_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate forecasts.
        
        Args:
            horizon: Forecast horizon
            freq: Frequency string
            exog_df: DataFrame with future exogenous features
        
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=horizon, freq=freq)
        
        # Add exogenous features if needed
        if self.feature_cols and exog_df is not None:
            for col in self.feature_cols:
                if col in exog_df.columns:
                    # Merge by date
                    future = future.merge(
                        exog_df[["ds", col]],
                        on="ds",
                        how="left",
                    )
        
        # Predict
        forecast = self.model.predict(future)
        
        # Return only future periods
        forecast = forecast.tail(horizon)
        
        return forecast
    
    def predict_with_intervals(
        self,
        horizon: int,
        freq: str = "D",
        exog_df: Optional[pd.DataFrame] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts with prediction intervals.
        
        Args:
            horizon: Forecast horizon
            freq: Frequency string
            exog_df: DataFrame with future exogenous features
        
        Returns:
            Tuple of (forecast, lower_bound, upper_bound)
        """
        forecast_df = self.predict(horizon, freq, exog_df)
        
        return (
            forecast_df["yhat"].values,
            forecast_df["yhat_lower"].values,
            forecast_df["yhat_upper"].values,
        )
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from fitted model.
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prophet stores regressor coefficients
        if not self.feature_cols:
            return pd.DataFrame()
        
        params = self.model.params
        
        importance_data = []
        for col in self.feature_cols:
            coef_name = f"beta_{col}"
            if coef_name in params:
                importance_data.append({
                    "feature": col,
                    "coefficient": params[coef_name].mean(),
                    "std": params[coef_name].std(),
                })
        
        return pd.DataFrame(importance_data)
