"""MLflow tracking utilities."""

import mlflow
from typing import Any, Dict, Optional
import pandas as pd


def setup_mlflow(experiment_name: str, tracking_uri: Optional[str] = None):
    """Setup MLflow tracking.
    
    Args:
        experiment_name: Name of experiment
        tracking_uri: Tracking URI (if None, uses default)
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    mlflow.set_experiment(experiment_name)


def log_params(params: Dict[str, Any]):
    """Log parameters to MLflow.
    
    Args:
        params: Dictionary of parameters
    """
    mlflow.log_params(params)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    """Log metrics to MLflow.
    
    Args:
        metrics: Dictionary of metrics
        step: Optional step number
    """
    mlflow.log_metrics(metrics, step=step)


def log_artifact(file_path: str):
    """Log artifact file to MLflow.
    
    Args:
        file_path: Path to file
    """
    mlflow.log_artifact(file_path)


def log_dataframe(df: pd.DataFrame, filename: str):
    """Log dataframe as artifact.
    
    Args:
        df: DataFrame to log
        filename: Filename for artifact
    """
    temp_path = f"/tmp/{filename}"
    df.to_csv(temp_path, index=False)
    mlflow.log_artifact(temp_path)


def start_run(run_name: Optional[str] = None) -> mlflow.ActiveRun:
    """Start MLflow run.
    
    Args:
        run_name: Optional run name
    
    Returns:
        Active run context
    """
    return mlflow.start_run(run_name=run_name)


def end_run():
    """End current MLflow run."""
    mlflow.end_run()
