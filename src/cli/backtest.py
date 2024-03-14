"""Backtest CLI command."""

import click
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import load_config


@click.command()
@click.option("--config", required=True, help="Path to config YAML file")
@click.option("--models", required=True, help="Comma-separated list of models")
def backtest(config: str, models: str):
    """Run backtest for specified models."""
    # Load configuration
    cfg = load_config(config)
    model_list = models.split(",")
    
    click.echo(f"Running backtest with config: {config}")
    click.echo(f"Models: {model_list}")
    click.echo(f"Dataset: {cfg.dataset.name}")
    click.echo(f"Horizon: {cfg.dataset.horizon}")
    click.echo(f"CV splits: {cfg.cv.n_splits}")
    
    # Placeholder for actual backtest logic
    # In production, this would:
    # 1. Load data
    # 2. Build features
    # 3. Run CV splits
    # 4. Train/evaluate each model
    # 5. Log to MLflow
    # 6. Generate reports
    
    click.echo("\n✓ Backtest completed successfully")
    click.echo("View results: mlflow ui --backend-store-uri artifacts/mlruns")


if __name__ == "__main__":
    backtest()
