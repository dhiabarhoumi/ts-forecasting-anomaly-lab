"""Time index utilities for handling time series data."""

from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd


def infer_frequency(ds: pd.Series) -> str:
    """Infer frequency from datetime series."""
    if len(ds) < 2:
        raise ValueError("Need at least 2 timestamps to infer frequency")
    
    freq = pd.infer_freq(ds.sort_values())
    if freq is None:
        # Fallback: compute median difference
        diffs = ds.sort_values().diff().dropna()
        median_diff = diffs.median()
        
        if median_diff < pd.Timedelta(minutes=2):
            return "T"  # Minute
        elif median_diff < pd.Timedelta(hours=2):
            return "H"  # Hour
        elif median_diff < pd.Timedelta(days=2):
            return "D"  # Day
        elif median_diff < pd.Timedelta(days=8):
            return "W"  # Week
        else:
            return "M"  # Month
    
    return freq


def fill_time_gaps(
    df: pd.DataFrame,
    ts_col: str = "ds",
    id_col: Optional[str] = None,
    freq: Optional[str] = None,
    method: str = "ffill",
) -> pd.DataFrame:
    """Fill gaps in time series data.
    
    Args:
        df: DataFrame with time series
        ts_col: Name of timestamp column
        id_col: Name of ID column for panel data
        freq: Frequency string (if None, will be inferred)
        method: Fill method ('ffill', 'bfill', 'interpolate', 'zero')
    
    Returns:
        DataFrame with filled gaps
    """
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    
    if freq is None:
        freq = infer_frequency(df[ts_col])
    
    if id_col is None:
        # Single series
        df = df.set_index(ts_col).sort_index()
        idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
        df = df.reindex(idx)
        
        if method == "ffill":
            df = df.fillna(method="ffill")
        elif method == "bfill":
            df = df.fillna(method="bfill")
        elif method == "interpolate":
            df = df.interpolate(method="linear")
        elif method == "zero":
            df = df.fillna(0)
        
        df = df.reset_index().rename(columns={"index": ts_col})
    else:
        # Panel data
        filled_dfs = []
        for sid, group in df.groupby(id_col):
            group = group.set_index(ts_col).sort_index()
            idx = pd.date_range(group.index.min(), group.index.max(), freq=freq)
            group = group.reindex(idx)
            
            if method == "ffill":
                group = group.fillna(method="ffill")
            elif method == "bfill":
                group = group.fillna(method="bfill")
            elif method == "interpolate":
                group = group.interpolate(method="linear")
            elif method == "zero":
                group = group.fillna(0)
            
            group = group.reset_index().rename(columns={"index": ts_col})
            group[id_col] = sid
            filled_dfs.append(group)
        
        df = pd.concat(filled_dfs, ignore_index=True)
    
    return df


def get_calendar_features(ds: pd.Series) -> pd.DataFrame:
    """Extract calendar features from datetime series.
    
    Args:
        ds: Series of timestamps
    
    Returns:
        DataFrame with calendar features
    """
    ds = pd.to_datetime(ds)
    
    features = pd.DataFrame({
        "dayofweek": ds.dt.dayofweek,
        "dayofmonth": ds.dt.day,
        "dayofyear": ds.dt.dayofyear,
        "week": ds.dt.isocalendar().week,
        "month": ds.dt.month,
        "quarter": ds.dt.quarter,
        "year": ds.dt.year,
        "is_weekend": ds.dt.dayofweek.isin([5, 6]).astype(int),
        "is_month_start": ds.dt.is_month_start.astype(int),
        "is_month_end": ds.dt.is_month_end.astype(int),
        "is_quarter_start": ds.dt.is_quarter_start.astype(int),
        "is_quarter_end": ds.dt.is_quarter_end.astype(int),
    })
    
    return features


def create_fourier_features(
    ds: pd.Series,
    periods: List[float],
    k: int = 5
) -> pd.DataFrame:
    """Create Fourier features for seasonality.
    
    Args:
        ds: Series of timestamps
        periods: List of seasonal periods (in days)
        k: Number of Fourier terms
    
    Returns:
        DataFrame with Fourier features
    """
    ds = pd.to_datetime(ds)
    # Convert to days since epoch
    t = (ds - ds.min()).dt.total_seconds() / 86400
    
    features = {}
    for period in periods:
        for i in range(1, k + 1):
            features[f"fourier_sin_{period}_{i}"] = np.sin(2 * np.pi * i * t / period)
            features[f"fourier_cos_{period}_{i}"] = np.cos(2 * np.pi * i * t / period)
    
    return pd.DataFrame(features)
