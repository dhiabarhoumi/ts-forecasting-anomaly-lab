"""Script to fetch and prepare M5 retail dataset."""

import argparse
from pathlib import Path
import pandas as pd
import requests
from io import BytesIO
from zipfile import ZipFile


def download_m5(data_dir: Path, subset: str = "small"):
    """Download M5 dataset from Kaggle or create sample.
    
    Args:
        data_dir: Directory to save data
        subset: Size of subset (small, medium, large)
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating M5 {subset} subset...")
    
    # Create synthetic M5-like data for demo
    # In production, this would download from Kaggle API
    
    # Generate sample hierarchy
    states = ["CA", "TX"]
    stores = ["_1", "_2", "_3"]
    depts = ["FOODS", "HOBBIES", "HOUSEHOLD"]
    items_per_dept = 2 if subset == "small" else 5
    
    series_data = []
    
    for state in states:
        for store in stores:
            store_id = f"{state}{store}"
            for dept in depts:
                for item_num in range(1, items_per_dept + 1):
                    item_id = f"{dept}_{item_num:03d}"
                    series_id = f"{state}_{store_id}_{dept}_{item_id}"
                    series_data.append({
                        "series_id": series_id,
                        "state": state,
                        "store": store_id,
                        "dept": dept,
                        "item": item_id,
                    })
    
    # Generate time series data
    dates = pd.date_range("2016-01-01", "2016-12-31", freq="D")
    
    all_data = []
    
    for series_info in series_data:
        # Generate synthetic sales with seasonality
        n_days = len(dates)
        base_level = 10 + (hash(series_info["series_id"]) % 20)
        
        # Weekly seasonality
        weekly_pattern = 1 + 0.3 * pd.Series(dates.dayofweek).apply(
            lambda x: 1 if x >= 5 else 0
        ).values
        
        # Trend
        trend = 1 + 0.0005 * pd.Series(range(n_days)).values
        
        # Noise
        noise = 1 + 0.1 * pd.Series(range(n_days)).apply(
            lambda x: (hash(str(x) + series_info["series_id"]) % 100 - 50) / 100
        ).values
        
        # Combine
        sales = base_level * weekly_pattern * trend * noise
        sales = sales.clip(min=0)
        
        for date, sale in zip(dates, sales):
            all_data.append({
                **series_info,
                "ds": date,
                "y": max(0, int(sale)),
            })
    
    # Save to CSV
    df = pd.DataFrame(all_data)
    output_file = data_dir / f"m5_{subset}_long.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✓ Created {len(series_data)} series with {len(dates)} days each")
    print(f"✓ Saved to {output_file}")
    print(f"✓ Total rows: {len(df):,}")


def main():
    parser = argparse.ArgumentParser(description="Fetch M5 retail dataset")
    parser.add_argument(
        "--out",
        type=str,
        default="data/retail/",
        help="Output directory",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="Dataset size",
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.out)
    download_m5(data_dir, args.subset)


if __name__ == "__main__":
    main()
