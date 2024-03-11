"""Cross-validation splits for time series."""

from typing import Iterator, Optional, Tuple
import pandas as pd
import numpy as np


def rolling_origin_split(
    df: pd.DataFrame,
    n_splits: int,
    horizon: int,
    min_train_points: int,
    step_size: Optional[int] = None,
    ts_col: str = "ds",
    id_col: Optional[str] = None,
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate rolling origin cross-validation splits.
    
    Args:
        df: DataFrame with time series
        n_splits: Number of splits
        horizon: Forecast horizon
        min_train_points: Minimum training observations
        step_size: Step size between splits (if None, uses horizon)
        ts_col: Timestamp column name
        id_col: ID column for panel data
    
    Yields:
        Tuples of (train_df, test_df)
    """
    step_size = step_size or horizon
    df = df.sort_values([id_col, ts_col] if id_col else ts_col)
    
    if id_col:
        # Panel data - split each series
        for split_idx in range(n_splits):
            train_parts = []
            test_parts = []
            
            for series_id, group in df.groupby(id_col):
                n_total = len(group)
                test_end_idx = n_total - (n_splits - split_idx - 1) * step_size
                test_start_idx = test_end_idx - horizon
                train_end_idx = test_start_idx
                
                if train_end_idx < min_train_points:
                    continue
                
                train_parts.append(group.iloc[:train_end_idx])
                test_parts.append(group.iloc[test_start_idx:test_end_idx])
            
            if train_parts and test_parts:
                yield pd.concat(train_parts), pd.concat(test_parts)
    else:
        # Single series
        n_total = len(df)
        
        for split_idx in range(n_splits):
            test_end_idx = n_total - (n_splits - split_idx - 1) * step_size
            test_start_idx = test_end_idx - horizon
            train_end_idx = test_start_idx
            
            if train_end_idx < min_train_points:
                continue
            
            train = df.iloc[:train_end_idx]
            test = df.iloc[test_start_idx:test_end_idx]
            
            yield train, test
