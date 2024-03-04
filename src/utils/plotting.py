"""Plotting utilities for time series visualization."""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns


def plot_series(
    df: pd.DataFrame,
    ts_col: str = "ds",
    y_col: str = "y",
    series_id: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Plot time series."""
    fig, ax = plt.subplots(figsize=figsize)
    
    if series_id:
        df_plot = df[df["series_id"] == series_id].copy()
        title = title or f"Series: {series_id}"
    else:
        df_plot = df.copy()
        title = title or "Time Series"
    
    df_plot = df_plot.sort_values(ts_col)
    ax.plot(df_plot[ts_col], df_plot[y_col], linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_forecast_with_intervals(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    ts_col: str = "ds",
    y_col: str = "y",
    pred_col: str = "yhat",
    lower_col: str = "yhat_lower",
    upper_col: str = "yhat_upper",
    series_id: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Plot forecast with prediction intervals."""
    fig, ax = plt.subplots(figsize=figsize)
    
    if series_id:
        actuals = actuals[actuals["series_id"] == series_id].copy()
        forecasts = forecasts[forecasts["series_id"] == series_id].copy()
        title = title or f"Forecast: {series_id}"
    else:
        title = title or "Forecast"
    
    actuals = actuals.sort_values(ts_col)
    forecasts = forecasts.sort_values(ts_col)
    
    # Plot actuals
    ax.plot(actuals[ts_col], actuals[y_col], label="Actual", color="black", linewidth=1.5)
    
    # Plot forecast
    ax.plot(forecasts[ts_col], forecasts[pred_col], label="Forecast", color="steelblue", linewidth=2)
    
    # Plot prediction interval
    if lower_col in forecasts.columns and upper_col in forecasts.columns:
        ax.fill_between(
            forecasts[ts_col],
            forecasts[lower_col],
            forecasts[upper_col],
            alpha=0.3,
            color="steelblue",
            label="90% PI",
        )
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_residuals(
    residuals: pd.Series,
    title: str = "Residuals Distribution",
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Plot residual distribution and QQ plot."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    axes[0].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Histogram")
    axes[0].grid(True, alpha=0.3)
    
    # QQ plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot")
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def plot_feature_importance(
    importance: pd.DataFrame,
    top_n: int = 20,
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot feature importance."""
    fig, ax = plt.subplots(figsize=figsize)
    
    importance_sorted = importance.nlargest(top_n, "importance")
    
    ax.barh(importance_sorted["feature"], importance_sorted["importance"])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    
    return fig


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    metric_col: str = "mape",
    model_col: str = "model",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot comparison of metrics across models."""
    fig, ax = plt.subplots(figsize=figsize)
    
    models = metrics_df[model_col].unique()
    values = [metrics_df[metrics_df[model_col] == m][metric_col].mean() for m in models]
    
    ax.bar(models, values, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_col.upper())
    ax.set_title(title or f"{metric_col.upper()} by Model")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    return fig
