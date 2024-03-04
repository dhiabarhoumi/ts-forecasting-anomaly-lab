"""Feature engineering for time series forecasting."""

from typing import Dict, List, Optional

import holidays
import numpy as np
import pandas as pd

from src.utils.timeindex import create_fourier_features, get_calendar_features


def create_lag_features(
    df: pd.DataFrame,
    lags: List[int],
    target_col: str = "y",
    id_col: Optional[str] = None,
) -> pd.DataFrame:
    """Create lag features.
    
    Args:
        df: DataFrame with time series
        lags: List of lag periods
        target_col: Name of target column
        id_col: Name of ID column for panel data
    
    Returns:
        DataFrame with lag features
    """
    df = df.copy()
    
    if id_col:
        for lag in lags:
            df[f"lag_{lag}"] = df.groupby(id_col)[target_col].shift(lag)
    else:
        for lag in lags:
            df[f"lag_{lag}"] = df[target_col].shift(lag)
    
    return df


def create_rolling_features(
    df: pd.DataFrame,
    windows: List[Dict[str, any]],
    target_col: str = "y",
    id_col: Optional[str] = None,
) -> pd.DataFrame:
    """Create rolling window features.
    
    Args:
        df: DataFrame with time series
        windows: List of window configs with 'window' and 'stats' keys
        target_col: Name of target column
        id_col: Name of ID column for panel data
    
    Returns:
        DataFrame with rolling features
    """
    df = df.copy()
    
    for window_config in windows:
        window = window_config["window"]
        stats = window_config.get("stats", ["mean"])
        
        if id_col:
            grouped = df.groupby(id_col)[target_col]
        else:
            grouped = df[target_col]
        
        for stat in stats:
            if stat == "mean":
                df[f"rolling_{window}_mean"] = grouped.transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
            elif stat == "std":
                df[f"rolling_{window}_std"] = grouped.transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )
            elif stat == "min":
                df[f"rolling_{window}_min"] = grouped.transform(
                    lambda x: x.rolling(window, min_periods=1).min()
                )
            elif stat == "max":
                df[f"rolling_{window}_max"] = grouped.transform(
                    lambda x: x.rolling(window, min_periods=1).max()
                )
    
    return df


def create_holiday_features(
    df: pd.DataFrame,
    ts_col: str = "ds",
    countries: Optional[List[str]] = None,
    lookback: int = 0,
    lookahead: int = 0,
) -> pd.DataFrame:
    """Create holiday features.
    
    Args:
        df: DataFrame with time series
        ts_col: Name of timestamp column
        countries: List of country codes
        lookback: Days before holiday to mark
        lookahead: Days after holiday to mark
    
    Returns:
        DataFrame with holiday features
    """
    df = df.copy()
    countries = countries or ["US"]
    
    # Get all holidays
    years = df[ts_col].dt.year.unique()
    all_holidays = set()
    
    for country in countries:
        for year in years:
            try:
                country_holidays = holidays.country_holidays(country, years=year)
                all_holidays.update(country_holidays.keys())
            except Exception:
                pass  # Skip if country not found
    
    # Create holiday feature
    df["is_holiday"] = df[ts_col].dt.date.isin(all_holidays).astype(int)
    
    # Create lookback/lookahead features
    if lookback > 0 or lookahead > 0:
        holiday_dates = pd.DataFrame({"date": list(all_holidays)})
        holiday_dates["date"] = pd.to_datetime(holiday_dates["date"])
        
        for i in range(1, lookback + 1):
            before_dates = set(holiday_dates["date"] - pd.Timedelta(days=i))
            df[f"holiday_minus_{i}"] = df[ts_col].dt.date.isin(before_dates).astype(int)
        
        for i in range(1, lookahead + 1):
            after_dates = set(holiday_dates["date"] + pd.Timedelta(days=i))
            df[f"holiday_plus_{i}"] = df[ts_col].dt.date.isin(after_dates).astype(int)
    
    return df


def create_promo_features(
    df: pd.DataFrame,
    promos: List[Dict[str, str]],
    ts_col: str = "ds",
) -> pd.DataFrame:
    """Create promotion features.
    
    Args:
        df: DataFrame with time series
        promos: List of promo dicts with 'start', 'end', 'name' keys
        ts_col: Name of timestamp column
    
    Returns:
        DataFrame with promo features
    """
    df = df.copy()
    
    for promo in promos:
        start = pd.to_datetime(promo["start"])
        end = pd.to_datetime(promo["end"])
        name = promo.get("name", f"promo_{start.strftime('%Y%m%d')}")
        
        df[f"promo_{name}"] = (
            (df[ts_col] >= start) & (df[ts_col] <= end)
        ).astype(int)
    
    return df


def build_features(
    df: pd.DataFrame,
    config: Dict,
    ts_col: str = "ds",
    target_col: str = "y",
    id_col: Optional[str] = None,
) -> pd.DataFrame:
    """Build all features from configuration.
    
    Args:
        df: DataFrame with time series
        config: Features configuration dict
        ts_col: Name of timestamp column
        target_col: Name of target column
        id_col: Name of ID column for panel data
    
    Returns:
        DataFrame with all features
    """
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    
    # Calendar features
    calendar_feats = get_calendar_features(df[ts_col])
    df = pd.concat([df, calendar_feats], axis=1)
    
    # Fourier features
    if "fourier" in config and config["fourier"]:
        fourier_config = config["fourier"]
        periods = fourier_config.get("periods", [7, 365.25])
        k = fourier_config.get("k", 5)
        
        fourier_feats = create_fourier_features(df[ts_col], periods, k)
        df = pd.concat([df, fourier_feats], axis=1)
    
    # Lag features
    if "lags" in config and config["lags"]:
        df = create_lag_features(df, config["lags"], target_col, id_col)
    
    # Rolling features
    if "rolls" in config and config["rolls"]:
        df = create_rolling_features(df, config["rolls"], target_col, id_col)
    
    # Holiday features
    if "holidays" in config and config["holidays"]:
        holiday_config = config["holidays"]
        countries = holiday_config.get("countries", ["US"])
        lookback = holiday_config.get("lookback", 0)
        lookahead = holiday_config.get("lookahead", 0)
        
        df = create_holiday_features(df, ts_col, countries, lookback, lookahead)
    
    # Promo features
    if "promos" in config and config["promos"]:
        df = create_promo_features(df, config["promos"], ts_col)
    
    return df
