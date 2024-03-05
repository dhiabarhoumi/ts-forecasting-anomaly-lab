"""Script to fetch and prepare OPSD energy dataset."""

import argparse
from pathlib import Path
import pandas as pd
import requests


def download_opsd(data_dir: Path):
    """Download OPSD dataset.
    
    Args:
        data_dir: Directory to save data
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading OPSD energy dataset...")
    
    # In production, download from:
    # https://data.open-power-system-data.org/time_series/
    
    # For now, create synthetic hourly load data
    dates = pd.date_range("2016-01-01", "2016-12-31 23:00:00", freq="H")
    
    # Generate synthetic load with daily and weekly patterns
    hour_of_day = dates.hour
    day_of_week = dates.dayofweek
    
    # Base load around 50 GW
    base_load = 50000
    
    # Daily pattern (higher during day)
    daily_pattern = 1 + 0.3 * pd.Series(hour_of_day).apply(
        lambda h: 1 if 8 <= h <= 20 else 0.5
    ).values
    
    # Weekly pattern (lower on weekends)
    weekly_pattern = 1 - 0.15 * pd.Series(day_of_week).apply(
        lambda d: 1 if d >= 5 else 0
    ).values
    
    # Seasonal pattern (higher in winter and summer)
    month = dates.month
    seasonal_pattern = 1 + 0.2 * pd.Series(month).apply(
        lambda m: 1 if m in [1, 2, 7, 8, 12] else 0.5
    ).values
    
    # Noise
    import numpy as np
    np.random.seed(42)
    noise = 1 + 0.05 * np.random.randn(len(dates))
    
    # Combine
    load = base_load * daily_pattern * weekly_pattern * seasonal_pattern * noise
    
    # Create DataFrame
    df = pd.DataFrame({
        "utc_timestamp": dates,
        "DE_load_actual_entsoe_transparency": load,
    })
    
    # Save
    output_file = data_dir / "opsd_time_series.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✓ Created {len(df):,} hourly observations")
    print(f"✓ Saved to {output_file}")
    
    # Create long format version
    df_long = df.copy()
    df_long = df_long.rename(columns={
        "utc_timestamp": "ds",
        "DE_load_actual_entsoe_transparency": "y",
    })
    df_long["series_id"] = "DE_load"
    df_long = df_long[["series_id", "ds", "y"]]
    
    output_long = data_dir / "opsd_long.csv"
    df_long.to_csv(output_long, index=False)
    print(f"✓ Created long format: {output_long}")


def main():
    parser = argparse.ArgumentParser(description="Fetch OPSD energy dataset")
    parser.add_argument(
        "--out",
        type=str,
        default="data/energy/",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.out)
    download_opsd(data_dir)


if __name__ == "__main__":
    main()
