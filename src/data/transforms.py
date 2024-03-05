"""Data transformation utilities."""

from typing import Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler


def align_calendars(
    df: pd.DataFrame,
    ts_col: str = "ds",
    freq: str = "D",
) -> pd.DataFrame:
    """Ensure consistent calendar alignment.
    
    Args:
        df: DataFrame with time series
        ts_col: Name of timestamp column
        freq: Target frequency
    
    Returns:
        Aligned DataFrame
    """
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    
    # Round timestamps to frequency
    if freq == "H":
        df[ts_col] = df[ts_col].dt.floor("H")
    elif freq == "D":
        df[ts_col] = df[ts_col].dt.floor("D")
    elif freq == "W":
        df[ts_col] = df[ts_col].dt.to_period("W").dt.start_time
    elif freq == "M":
        df[ts_col] = df[ts_col].dt.to_period("M").dt.start_time
    
    return df


def scale_features(
    df: pd.DataFrame,
    feature_cols: list,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Scale numeric features.
    
    Args:
        df: DataFrame with features
        feature_cols: Columns to scale
        scaler: Optional pre-fitted scaler
        fit: Whether to fit the scaler
    
    Returns:
        Tuple of (scaled DataFrame, scaler)
    """
    df = df.copy()
    
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
    
    return df, scaler


def split_train_test(
    df: pd.DataFrame,
    test_size: int,
    ts_col: str = "ds",
    id_col: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets.
    
    Args:
        df: DataFrame with time series
        test_size: Number of periods for test set
        ts_col: Name of timestamp column
        id_col: Name of ID column for panel data
    
    Returns:
        Tuple of (train_df, test_df)
    """
    df = df.copy()
    df = df.sort_values([id_col, ts_col] if id_col else ts_col)
    
    if id_col:
        # Split each series separately
        train_dfs = []
        test_dfs = []
        
        for sid, group in df.groupby(id_col):
            n = len(group)
            train_dfs.append(group.iloc[: n - test_size])
            test_dfs.append(group.iloc[n - test_size :])
        
        train = pd.concat(train_dfs, ignore_index=True)
        test = pd.concat(test_dfs, ignore_index=True)
    else:
        n = len(df)
        train = df.iloc[: n - test_size]
        test = df.iloc[n - test_size :]
    
    return train, test
