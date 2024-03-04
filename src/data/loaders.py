"""Data loaders for different time series datasets."""

from pathlib import Path
from typing import Optional

import pandas as pd


def load_m5_data(
    data_dir: str,
    subset: Optional[str] = None,
) -> pd.DataFrame:
    """Load M5 retail dataset.
    
    Args:
        data_dir: Path to data directory
        subset: Optional subset name (small, medium, large)
    
    Returns:
        DataFrame in long format with columns: series_id, ds, y
    """
    data_path = Path(data_dir)
    
    # Check for preprocessed file
    if subset:
        preprocessed = data_path / f"m5_{subset}_long.csv"
    else:
        preprocessed = data_path / "m5_long.csv"
    
    if preprocessed.exists():
        df = pd.read_csv(preprocessed)
        df["ds"] = pd.to_datetime(df["ds"])
        return df
    
    # If raw M5 files exist, load and convert
    # This is a simplified loader - real M5 requires more processing
    sales_file = data_path / "sales_train_validation.csv"
    calendar_file = data_path / "calendar.csv"
    
    if not sales_file.exists():
        raise FileNotFoundError(
            f"M5 data not found at {data_path}. "
            "Run 'python scripts/fetch_m5.py' to download."
        )
    
    sales = pd.read_csv(sales_file)
    calendar = pd.read_csv(calendar_file)
    
    # Extract hierarchy from item_id
    sales["state"] = sales["store_id"].str[:2]
    sales["store"] = sales["store_id"]
    sales["dept"] = sales["dept_id"]
    sales["item"] = sales["item_id"]
    
    # Create series_id
    sales["series_id"] = (
        sales["state"] + "_" +
        sales["store"] + "_" +
        sales["dept"] + "_" +
        sales["item"]
    )
    
    # Melt to long format
    id_cols = ["series_id", "state", "store", "dept", "item"]
    value_cols = [c for c in sales.columns if c.startswith("d_")]
    
    df_long = sales.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="d",
        value_name="y",
    )
    
    # Join with calendar to get dates
    calendar_map = calendar.set_index("d")["date"].to_dict()
    df_long["ds"] = df_long["d"].map(calendar_map)
    df_long["ds"] = pd.to_datetime(df_long["ds"])
    
    # Sort and clean
    df_long = df_long.sort_values(["series_id", "ds"]).reset_index(drop=True)
    df_long = df_long[["series_id", "ds", "y", "state", "store", "dept", "item"]]
    
    # Save preprocessed
    df_long.to_csv(preprocessed, index=False)
    
    return df_long


def load_opsd_data(
    data_dir: str,
) -> pd.DataFrame:
    """Load OPSD energy dataset.
    
    Args:
        data_dir: Path to data directory
    
    Returns:
        DataFrame in long format with columns: series_id, ds, y
    """
    data_path = Path(data_dir)
    
    # Check for preprocessed file
    preprocessed = data_path / "opsd_long.csv"
    
    if preprocessed.exists():
        df = pd.read_csv(preprocessed)
        df["ds"] = pd.to_datetime(df["ds"])
        return df
    
    # Load raw OPSD file
    opsd_file = data_path / "opsd_time_series.csv"
    
    if not opsd_file.exists():
        raise FileNotFoundError(
            f"OPSD data not found at {data_path}. "
            "Run 'python scripts/fetch_opsd.py' to download."
        )
    
    df = pd.read_csv(opsd_file)
    df["ds"] = pd.to_datetime(df["utc_timestamp"])
    
    # Focus on load columns
    load_cols = [c for c in df.columns if "load_actual" in c.lower()]
    
    if not load_cols:
        raise ValueError("No load columns found in OPSD data")
    
    # Convert to long format
    df_long_parts = []
    
    for col in load_cols:
        df_sub = df[["ds", col]].copy()
        df_sub = df_sub.dropna(subset=[col])
        df_sub["series_id"] = col
        df_sub = df_sub.rename(columns={col: "y"})
        df_long_parts.append(df_sub)
    
    df_long = pd.concat(df_long_parts, ignore_index=True)
    df_long = df_long.sort_values(["series_id", "ds"]).reset_index(drop=True)
    
    # Save preprocessed
    df_long.to_csv(preprocessed, index=False)
    
    return df_long


def load_dataset(config_path: str, config: dict) -> pd.DataFrame:
    """Load dataset based on configuration.
    
    Args:
        config_path: Path to config file (for resolving relative paths)
        config: Dataset configuration dict
    
    Returns:
        DataFrame in long format
    """
    dataset_name = config["name"]
    data_dir = config["path"]
    
    # Resolve relative paths
    if not Path(data_dir).is_absolute():
        config_dir = Path(config_path).parent
        data_dir = config_dir / data_dir
    
    if "m5" in dataset_name.lower() or "retail" in dataset_name.lower():
        return load_m5_data(str(data_dir))
    elif "opsd" in dataset_name.lower() or "energy" in dataset_name.lower():
        return load_opsd_data(str(data_dir))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
