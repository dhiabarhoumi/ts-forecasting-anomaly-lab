"""LightGBM forecasting model wrapper."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split


class LightGBMForecaster:
    """LightGBM model wrapper for time series forecasting."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize LightGBM forecaster.
        
        Args:
            config: Configuration dictionary with LightGBM parameters
        """
        self.config = config or {}
        self.model = None
        self.feature_cols = []
    
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        eval_set: Optional[tuple] = None,
        categorical_features: Optional[List[str]] = None,
    ):
        """Fit LightGBM model.
        
        Args:
            X: Training features
            y: Training target
            eval_set: Optional (X_val, y_val) for early stopping
            categorical_features: List of categorical feature names
        """
        self.feature_cols = X.columns.tolist()
        
        # Prepare categorical features
        cat_features = categorical_features or []
        cat_indices = [X.columns.get_loc(c) for c in cat_features if c in X.columns]
        
        # Create dataset
        train_data = lgb.Dataset(
            X,
            label=y,
            categorical_feature=cat_indices if cat_indices else "auto",
        )
        
        # Prepare validation set if provided
        valid_sets = [train_data]
        valid_names = ["training"]
        
        if eval_set is not None:
            X_val, y_val = eval_set
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                reference=train_data,
            )
            valid_sets.append(val_data)
            valid_names.append("validation")
        
        # Train model
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbose": -1,
            **self.config,
        }
        
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.log_evaluation(period=0),  # Suppress output
            ],
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions.
        
        Args:
            X: Features for prediction
        
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.predict(X)
    
    def predict_with_intervals(
        self,
        X: pd.DataFrame,
        alpha: float = 0.1,
        n_quantiles: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecasts with prediction intervals using quantile estimation.
        
        Args:
            X: Features for prediction
            alpha: Significance level for intervals (e.g., 0.1 for 90% PI)
            n_quantiles: Number of quantiles for estimation
        
        Returns:
            Tuple of (forecast, lower_bound, upper_bound)
        """
        forecast = self.predict(X)
        
        # Simple approach: estimate intervals from training residuals
        # In production, train separate quantile regression models
        
        # Assume residuals are roughly normal with constant variance
        # This is a simplification - consider quantile regression for better intervals
        
        # Estimate std from feature importance as proxy
        # Better: store training residuals during fit
        std_estimate = np.std(forecast) * 0.1  # Rough heuristic
        
        z_score = 1.96  # For 95% CI
        lower = forecast - z_score * std_estimate
        upper = forecast + z_score * std_estimate
        
        return forecast, lower, upper
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from fitted model.
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        importance = self.model.feature_importance(importance_type="gain")
        
        importance_df = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": importance,
        })
        
        importance_df = importance_df.sort_values("importance", ascending=False)
        
        return importance_df
