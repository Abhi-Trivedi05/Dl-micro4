"""
Evaluation metrics module for time-series forecasting.
Computes MAE, MSE, RMSE, and Directional Accuracy on original and scaled data.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from src.preprocessing import TimeSeriesScaler


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error (MSE)."""
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Directional Accuracy (MDA) percentage.
    Measures the frequency with which the model accurately predicts the directional move (up/down).
    
    Formula:
        MDA = mean( sign(y_{t} - y_{t-1}) == sign(\\hat{y}_{t} - \\hat{y}_{t-1}) )
    """
    actual = np.asarray(y_true, dtype=np.float64).ravel()
    predicted = np.asarray(y_pred, dtype=np.float64).ravel()

    if len(actual) < 2:
        return 0.0

    actual_diff = np.sign(actual[1:] - actual[:-1])
    pred_diff = np.sign(predicted[1:] - predicted[:-1])

    # Direction matches when signs are equal
    matches = (actual_diff == pred_diff).astype(float)
    return float(np.mean(matches) * 100.0)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute all standard forecasting metrics for a pair of actual and predicted arrays.
    
    Args:
        y_true: Ground truth target values.
        y_pred: Predicted values.
        prefix: Optional prefix for metric dictionary keys (e.g., 'unscaled_' or 'scaled_').
        
    Returns:
        dict: Keys for MAE, MSE, RMSE, and Directional_Accuracy_Pct.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mda = mean_directional_accuracy(y_true, y_pred)

    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}MAE": mae,
        f"{p}MSE": mse,
        f"{p}RMSE": rmse,
        f"{p}MDA_Pct": mda,
    }


def predict_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run model inference over a DataLoader.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (y_true, y_pred) arrays on the scaled domain.
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(dev)
            out, _ = model(batch_x)
            preds.append(out.cpu().numpy())
            targets.append(batch_y.numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets, axis=0)
    return y_true, y_pred


def evaluate_forecast(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    scaler: Optional[TimeSeriesScaler] = None,
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Evaluate forecast quality both on the scaled domain [0, 1] and on the original unscaled domain.
    """
    scaled_metrics = compute_all_metrics(y_true_scaled, y_pred_scaled, prefix="scaled")

    if scaler is not None and scaler.is_fitted:
        y_true_orig = scaler.inverse_transform(y_true_scaled)
        y_pred_orig = scaler.inverse_transform(y_pred_scaled)
        orig_metrics = compute_all_metrics(y_true_orig, y_pred_orig, prefix="unscaled")
    else:
        orig_metrics = {}

    combined = {}
    combined.update(scaled_metrics)
    combined.update(orig_metrics)
    return combined

