"""Script to build weather features."""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def build_weather_features(source_dir: Path, output_file: Path):
    """Build weather features from OPSD data.
    
    Args:
        source_dir: Directory with source data
        output_file: Output parquet file
    """
    print("Building weather features...")
    
    # Load energy data to get timestamps
    opsd_file = source_dir / "opsd_long.csv"
    if not opsd_file.exists():
        raise FileNotFoundError(f"OPSD data not found at {opsd_file}")
    
    df = pd.read_csv(opsd_file)
    df["ds"] = pd.to_datetime(df["ds"])
    
    # Generate synthetic weather data
    # In production, use meteostat or similar API
    
    dates = df["ds"].unique()
    hour_of_day = pd.Series([d.hour for d in dates])
    day_of_year = pd.Series([d.dayofyear for d in dates])
    
    # Temperature (°C): seasonal + daily variation
    temp_base = 10 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    temp_daily = -3 * np.cos(2 * np.pi * hour_of_day / 24)
    np.random.seed(42)
    temp_noise = 2 * np.random.randn(len(dates))
    temperature = temp_base + temp_daily + temp_noise
    
    # Wind speed (m/s): random with some persistence
    np.random.seed(43)
    wind_speed = 5 + 3 * np.random.randn(len(dates))
    wind_speed = wind_speed.clip(min=0, max=25)
    
    # Humidity (%): inverse of temperature roughly
    np.random.seed(44)
    humidity = 70 - 0.5 * temperature + 10 * np.random.randn(len(dates))
    humidity = humidity.clip(min=30, max=100)
    
    # Create DataFrame
    weather_df = pd.DataFrame({
        "ds": dates,
        "temperature": temperature,
        "wind_speed": wind_speed,
        "humidity": humidity,
    })
    
    # Save as parquet
    output_file.parent.mkdir(parents=True, exist_ok=True)
    weather_df.to_parquet(output_file, index=False)
    
    print(f"✓ Created weather features for {len(weather_df):,} timestamps")
    print(f"✓ Saved to {output_file}")
    print(f"\nFeature statistics:")
    print(weather_df.describe())


def main():
    parser = argparse.ArgumentParser(description="Build weather features")
    parser.add_argument(
        "--source",
        type=str,
        default="data/energy/",
        help="Source data directory",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/energy/weather.parquet",
        help="Output parquet file",
    )
    
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    output_file = Path(args.out)
    
    build_weather_features(source_dir, output_file)


if __name__ == "__main__":
    main()
