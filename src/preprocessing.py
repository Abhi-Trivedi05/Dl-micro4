"""
Time-series preprocessing pipeline: leakage-free scaling and sliding-window generation.
Converts continuous sequential observations into supervised learning sample pairs (X, Y).
"""

from typing import Optional, Tuple, Union
import numpy as np


class TimeSeriesScaler:
    """
    Leakage-free scaler for time series.
    Fits scale parameters strictly on the training partition and transforms
    training, validation, and test arrays consistently.
    """

    def __init__(self, scaler_type: str = "minmax"):
        """
        Args:
            scaler_type: 'minmax' for [0, 1] scaling or 'standard' for zero mean, unit variance.
        """
        self.scaler_type = scaler_type.lower()
        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None
        self.mean_val: Optional[float] = None
        self.std_val: Optional[float] = None
        self.is_fitted: bool = False

    def fit(self, data: np.ndarray) -> "TimeSeriesScaler":
        """
        Fit scaler parameters exclusively on training data.
        
        Args:
            data: 1D or 2D numpy array of training observations.
        """
        flat_data = np.asarray(data, dtype=np.float32).ravel()
        if flat_data.size == 0:
            raise ValueError("Cannot fit scaler on empty array.")

        if self.scaler_type == "minmax":
            self.min_val = float(np.min(flat_data))
            self.max_val = float(np.max(flat_data))
            # Handle constant series edge case
            if self.max_val == self.min_val:
                self.max_val += 1e-7
        elif self.scaler_type == "standard":
            self.mean_val = float(np.mean(flat_data))
            self.std_val = float(np.std(flat_data))
            if self.std_val == 0.0:
                self.std_val = 1e-7
        else:
            raise ValueError(f"Unsupported scaler_type '{self.scaler_type}'. Use 'minmax' or 'standard'.")

        self.is_fitted = True
        return self

    def transform(self, data: Union[np.ndarray, float]) -> np.ndarray:
        """
        Transform data using fitted parameters.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted on training data before calling transform.")

        arr = np.asarray(data, dtype=np.float32)
        if self.scaler_type == "minmax":
            return (arr - self.min_val) / (self.max_val - self.min_val)
        else:
            return (arr - self.mean_val) / self.std_val

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.
        """
        return self.fit(data).transform(data)

    def inverse_transform(self, data: Union[np.ndarray, float]) -> np.ndarray:
        """
        Invert the transformation back to original physical units (e.g. stock price in INR).
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before calling inverse_transform.")

        arr = np.asarray(data, dtype=np.float32)
        if self.scaler_type == "minmax":
            return (arr * (self.max_val - self.min_val)) + self.min_val
        else:
            return (arr * self.std_val) + self.mean_val


def create_sliding_windows(
    data: np.ndarray,
    sequence_length: int,
    forecast_horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1D time series into supervised learning input-target pairs (X, Y)
    using a sliding window approach.
    
    Explanation:
        Time-series data is inherently sequential. Supervised machine learning algorithms
        require explicit feature vectors (X) and ground-truth target outputs (Y).
        A lookback window of length `w` (sequence_length) slides across the series.
        At each step t:
            X[t] = [x_t, x_{t+1}, ..., x_{t+w-1}]   (shape: w, 1)
            Y[t] = [x_{t+w}, ..., x_{t+w+f-1}]      (shape: f)
            
    Args:
        data: 1D array of time-series observations.
        sequence_length: Number of past time steps to use as input features (w).
        forecast_horizon: Number of future time steps to predict ahead (f).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            X: Input sequences of shape (num_samples, sequence_length, 1).
            Y: Future target sequences of shape (num_samples, forecast_horizon).
    """
    flat = np.asarray(data, dtype=np.float32).ravel()
    window_total = sequence_length + forecast_horizon
    total_steps = len(flat)

    if total_steps < window_total:
        raise ValueError(
            f"Series length ({total_steps}) is shorter than window_total ({window_total})."
        )

    num_samples = total_steps - window_total + 1
    X = np.empty((num_samples, sequence_length, 1), dtype=np.float32)
    Y = np.empty((num_samples, forecast_horizon), dtype=np.float32)

    for i in range(num_samples):
        X[i, :, 0] = flat[i : i + sequence_length]
        Y[i, :] = flat[i + sequence_length : i + window_total]

    return X, Y


def split_time_series_chronological(
    series: np.ndarray,
    test_samples: int = 251,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronologically split a single 1D time series into Train, Validation, and Test partitions.
    
    This preserves the natural time order and strictly prevents future lookahead leakage.
    
    Args:
        series: 1D array of time series values.
        test_samples: Fixed number of trailing samples reserved for testing (matching reference study).
        val_ratio: Fraction of the remaining training portion to allocate for validation.
        
    Returns:
        Tuple of (train_series, val_series, test_series).
    """
    series = np.asarray(series, dtype=np.float32).ravel()
    total_len = len(series)

    if total_len <= test_samples:
        raise ValueError(f"Series length {total_len} must be greater than test_samples {test_samples}.")

    # Split off the test partition from the end of the timeline
    train_val_series = series[:-test_samples]
    test_series = series[-test_samples:]

    # Split train_val into train and validation partitions
    num_val = int(len(train_val_series) * val_ratio)
    if num_val < 1:
        raise ValueError("val_ratio is too small; validation set has 0 samples.")

    train_series = train_val_series[:-num_val]
    val_series = train_val_series[-num_val:]

    return train_series, val_series, test_series


def prepare_datasets_pipeline(
    series: np.ndarray,
    sequence_length: int = 30,
    forecast_horizon: int = 1,
    test_samples: int = 251,
    val_ratio: float = 0.15,
    scaler_type: str = "minmax",
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], TimeSeriesScaler]:
    """
    Complete end-to-end preprocessing pipeline:
    1. Chronological splitting (Train, Val, Test).
    2. Fitting scaler strictly on Train data only (Leakage-free).
    3. Normalizing Train, Val, and Test series.
    4. Attaching lookback context to Val and Test so full target lengths are predicted.
    5. Generating sliding-window (X, Y) pairs for all three partitions.
    
    Returns:
        ((train_X, train_Y), (val_X, val_Y), (test_X, test_Y), scaler)
    """
    train_raw, val_raw, test_raw = split_time_series_chronological(
        series, test_samples=test_samples, val_ratio=val_ratio
    )

    # Fit scaler strictly on the training partition
    scaler = TimeSeriesScaler(scaler_type=scaler_type)
    scaler.fit(train_raw)

    train_scaled = scaler.transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    # To ensure validation and test sets can predict starting from their very first element,
    # prepend the lookback window context from the preceding partition.
    val_with_context = np.concatenate([train_scaled[-sequence_length:], val_scaled])
    test_with_context = np.concatenate([val_scaled[-sequence_length:], test_scaled])

    train_X, train_Y = create_sliding_windows(train_scaled, sequence_length, forecast_horizon)
    val_X, val_Y = create_sliding_windows(val_with_context, sequence_length, forecast_horizon)
    test_X, test_Y = create_sliding_windows(test_with_context, sequence_length, forecast_horizon)

    return (train_X, train_Y), (val_X, val_Y), (test_X, test_Y), scaler


def persistence_baseline(X: np.ndarray, forecast_horizon: int = 1) -> np.ndarray:
    """
    Naive persistence baseline: repeats the last observed price/value x_t for all f forecast steps.
    
    Args:
        X: Input tensor of shape (batch_size, sequence_length, 1).
        forecast_horizon: f steps ahead.
        
    Returns:
        np.ndarray: Predicted tensor of shape (batch_size, forecast_horizon).
    """
    last_known_value = X[:, -1, 0]  # shape: (batch_size,)
    return np.repeat(last_known_value[:, np.newaxis], forecast_horizon, axis=1)

