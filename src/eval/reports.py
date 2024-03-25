"""Report generation utilities."""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional


def generate_comparison_report(
    leaderboard: pd.DataFrame,
    output_path: str,
    title: str = "Model Comparison Report",
) -> None:
    """Generate markdown comparison report.
    
    Args:
        leaderboard: Leaderboard DataFrame
        output_path: Path to save report
        title: Report title
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        f.write("## Leaderboard\n\n")
        f.write(leaderboard.to_markdown(index=False))
        f.write("\n\n## Analysis\n\n")
        
        # Best model
        if "model" in leaderboard.columns and "mape" in leaderboard.columns:
            best_model = leaderboard.iloc[0]["model"]
            best_mape = leaderboard.iloc[0]["mape"]
            f.write(f"**Best Model**: {best_model} (MAPE: {best_mape:.2f}%)\n\n")
        
        f.write("### Observations\n\n")
        f.write("- Models ranked by MAPE (lower is better)\n")
        f.write("- Prediction intervals evaluated for coverage\n")
        f.write("- See MLflow UI for detailed artifacts\n")


def generate_anomaly_report(
    anomalies: pd.DataFrame,
    output_path: str,
    top_k: int = 20,
) -> None:
    """Generate anomaly detection report.
    
    Args:
        anomalies: DataFrame with anomaly flags and scores
        output_path: Path to save report
        top_k: Number of top anomalies to report
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, "w") as f:
        f.write("# Anomaly Detection Report\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        
        total_anomalies = anomalies["is_anomaly"].sum()
        f.write(f"**Total Anomalies Detected**: {total_anomalies}\n\n")
        
        f.write(f"## Top {top_k} Anomalies\n\n")
        
        top_anomalies = anomalies.nlargest(top_k, "anomaly_score")
        f.write(top_anomalies[["ds", "y", "yhat", "anomaly_score"]].to_markdown(index=False))
        
        f.write("\n\n## Statistics\n\n")
        f.write(f"- Mean anomaly score: {anomalies['anomaly_score'].mean():.3f}\n")
        f.write(f"- Max anomaly score: {anomalies['anomaly_score'].max():.3f}\n")
        f.write(f"- Anomaly rate: {total_anomalies / len(anomalies) * 100:.2f}%\n")
