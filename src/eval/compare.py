"""Comparison utilities for model evaluation."""

import pandas as pd
from typing import List, Dict


def create_leaderboard(
    results: List[Dict],
    metrics: List[str] = None,
) -> pd.DataFrame:
    """Create leaderboard from model results.
    
    Args:
        results: List of result dictionaries
        metrics: Metrics to include
    
    Returns:
        Leaderboard DataFrame sorted by primary metric
    """
    if metrics is None:
        metrics = ["mape", "smape", "rmse", "mase"]
    
    df = pd.DataFrame(results)
    
    # Aggregate by model if multiple runs
    if "model" in df.columns:
        agg_funcs = {m: "mean" for m in metrics if m in df.columns}
        df = df.groupby("model").agg(agg_funcs).reset_index()
    
    # Sort by primary metric (MAPE)
    if "mape" in df.columns:
        df = df.sort_values("mape")
    
    return df


def compare_models(
    results_dict: Dict[str, List[float]],
    metric_name: str = "MAPE",
) -> pd.DataFrame:
    """Compare models across metrics.
    
    Args:
        results_dict: Dictionary mapping model names to metric values
        metric_name: Name of metric being compared
    
    Returns:
        Comparison DataFrame
    """
    comparison = pd.DataFrame(results_dict).T
    comparison.columns = [f"fold_{i+1}" for i in range(len(comparison.columns))]
    comparison["mean"] = comparison.mean(axis=1)
    comparison["std"] = comparison.std(axis=1)
    comparison = comparison.sort_values("mean")
    
    return comparison
