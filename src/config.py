"""Configuration management using Pydantic."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    name: str
    path: str
    freq: str
    target: str
    id_col: str = "series_id"
    ts_col: str = "ds"
    hierarchy_levels: Optional[List[str]] = None
    horizon: int = 28


class FeaturesConfig(BaseModel):
    lags: List[int] = Field(default_factory=list)
    rolls: List[Dict[str, Any]] = Field(default_factory=list)
    fourier: Optional[Dict[str, Any]] = None
    holidays: Optional[Dict[str, Any]] = None
    promos: Optional[List[Dict[str, str]]] = None
    weather: Optional[Dict[str, Any]] = None


class CVConfig(BaseModel):
    method: str = "rolling"
    n_splits: int = 4
    horizon: int = 28
    min_train_points: int = 365
    step_size: Optional[int] = None


class ReconciliationConfig(BaseModel):
    method: str = "none"  # none | bu | mint
    mint_method: str = "shrink"


class AnomalyConfig(BaseModel):
    pi_alpha: float = 0.9
    residual: Optional[Dict[str, Any]] = None
    iforest: Optional[Dict[str, Any]] = None
    ocsvm: Optional[Dict[str, Any]] = None


class LoggingConfig(BaseModel):
    mlflow_experiment: str = "ts-forecasting"
    log_level: str = "INFO"
    save_predictions: bool = True


class Config(BaseModel):
    dataset: DatasetConfig
    features: FeaturesConfig
    models: Dict[str, Dict[str, Any]]
    cv: CVConfig
    reconciliation: Optional[ReconciliationConfig] = None
    anomaly: Optional[AnomalyConfig] = None
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()


def load_config(path: str) -> Config:
    """Load configuration from YAML file."""
    return Config.from_yaml(path)
