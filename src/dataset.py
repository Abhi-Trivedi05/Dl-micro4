"""
PyTorch Dataset and DataLoader wrappers for time-series forecasting.
"""

from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    """
    Custom PyTorch Dataset for sequential time-series windows.
    Wraps input sequences X and target sequences Y.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        """
        Args:
            X: Input array of shape (N, sequence_length, num_features).
            Y: Target array of shape (N, forecast_horizon).
        """
        if len(X) != len(Y):
            raise ValueError(f"Length mismatch: X has {len(X)} samples, Y has {len(Y)} samples.")

        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


def create_dataloaders(
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoader instances for train, validation, and test splits.
    
    Args:
        train_data: (train_X, train_Y)
        val_data: (val_X, val_Y)
        test_data: (test_X, test_Y)
        batch_size: Batch size for mini-batch stochastic optimization.
        num_workers: DataLoader worker processes.
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    train_dataset = TimeSeriesDataset(*train_data)
    val_dataset = TimeSeriesDataset(*val_data)
    test_dataset = TimeSeriesDataset(*test_data)

    # Note: Shuffling is enabled on training samples to break correlation between consecutive windows
    # Validation and testing loaders must remain sequential (shuffle=False) for reproducible evaluation.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader

