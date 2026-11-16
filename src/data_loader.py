"""
Data loader module for time-series forecasting.
Handles loading raw CSV datasets, dataset validation, and metadata extraction.
"""

from pathlib import Path
from typing import Dict, List, Union
import numpy as np
import pandas as pd

# BANKEX index constituents represented by the 10 rows
BANK_NAMES: List[str] = [
    "Axis Bank (AXISBANK.BO)",
    "Bank of Baroda (BANKBARODA.BO)",
    "Federal Bank (FEDERALBNK.BO)",
    "HDFC Bank (HDFCBANK.BO)",
    "ICICI Bank (ICICIBANK.BO)",
    "IndusInd Bank (INDUSINDBK.BO)",
    "Kotak Mahindra Bank (KOTAKBANK.BO)",
    "Punjab National Bank (PNB.BO)",
    "State Bank of India (SBIN.BO)",
    "Yes Bank (YESBANK.BO)",
]


def load_raw_data(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load raw time-series data from CSV file.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        np.ndarray: 2D array of shape (num_series, time_steps).
        
    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is empty or malformed.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    # Load CSV using pandas for fast and robust parsing
    df = pd.read_csv(path, header=None)
    data = df.to_numpy(dtype=np.float32)

    if data.size == 0:
        raise ValueError(f"Loaded empty dataset from {path}")

    return data


def get_series(data: np.ndarray, series_idx: int = 0) -> np.ndarray:
    """
    Extract a single univariate time series by row index.
    
    Args:
        data: 2D array of time series.
        series_idx: Row index (0 to num_series - 1).
        
    Returns:
        np.ndarray: 1D array of shape (time_steps,).
    """
    if series_idx < 0 or series_idx >= data.shape[0]:
        raise IndexError(
            f"Series index {series_idx} out of bounds for data with {data.shape[0]} series."
        )
    return data[series_idx, :].copy()


def get_dataset_info(data: np.ndarray, dataset_name: str = "Dataset") -> Dict[str, Union[str, int, float]]:
    """
    Compute basic summary statistics and validation checks for the dataset.
    
    Args:
        data: 2D array of time series data.
        dataset_name: Label for the dataset.
        
    Returns:
        dict: Summary statistics including shape, min, max, mean, and missing value counts.
    """
    num_series, num_timesteps = data.shape
    nan_count = int(np.isnan(data).sum())

    return {
        "dataset_name": dataset_name,
        "num_series": num_series,
        "num_timesteps": num_timesteps,
        "nan_count": nan_count,
        "min_value": float(np.nanmin(data)),
        "max_value": float(np.nanmax(data)),
        "mean_value": float(np.nanmean(data)),
        "std_value": float(np.nanstd(data)),
    }

