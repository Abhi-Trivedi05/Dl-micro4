"""
Centralized Configuration for Time-Series Forecasting Experiments.
Defines paths, random seeds, model hyperparameter presets, and dataset settings.
"""

from dataclasses import dataclass
from pathlib import Path

# Base project directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"

# Ensure runtime directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, PLOTS_DIR, METRICS_DIR, CHECKPOINTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetConfig:
    """Configuration for dataset selection and preprocessing."""
    dataset_name: str = "BANKEX"  # 'BANKEX' or 'activities'
    file_path: Path = RAW_DATA_DIR / "BANKEX.csv"
    selected_series_idx: int = 0   # Default series index to train on (0 to 9)
    test_samples: int = 251        # Number of samples reserved chronologically for testing
    val_ratio: float = 0.15        # Fraction of training data reserved for validation
    sequence_length: int = 30      # Lookback window w (configurable: 7, 14, 30, 60)
    forecast_horizon: int = 1      # Horizon f (e.g., 1 or 20)
    scaler_type: str = "minmax"    # 'minmax' or 'standard'


@dataclass
class TrainingConfig:
    """Hyperparameters for training PyTorch recurrent neural networks."""
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 150
    patience: int = 15             # Early stopping patience
    min_delta: float = 1e-5        # Minimum improvement for early stopping
    grad_clip: float = 1.0         # Gradient clipping norm
    seed: int = 42
    device: str = "cpu"            # 'cpu', 'cuda', or 'mps'


@dataclass
class ModelConfig:
    """Architecture configuration for RNN, LSTM, and GRU."""
    input_size: int = 1
    hidden_size: int = 64          # Configurable: 32, 64, 128
    num_layers: int = 1            # Number of recurrent layers
    output_size: int = 1           # Matches forecast_horizon f
    dropout: float = 0.0           # Configurable: 0.0, 0.2, 0.4

