"""
Models package for time-series forecasting.
Exports VanillaRNN, LSTMModel, GRUModel, and model factory helper.
"""

import torch.nn as nn
from models.vanilla_rnn import VanillaRNN
from models.lstm import LSTMModel
from models.gru import GRUModel


def build_model(
    model_name: str,
    input_size: int = 1,
    hidden_size: int = 64,
    num_layers: int = 1,
    output_size: int = 1,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Factory function to instantiate models by string identifier.
    
    Args:
        model_name: 'rnn', 'lstm', or 'gru' (case-insensitive).
        input_size: Feature dimensions per time step.
        hidden_size: Number of hidden units.
        num_layers: Number of stacked recurrent layers.
        output_size: Forecast horizon f.
        dropout: Dropout rate.
        
    Returns:
        nn.Module: Instantiated PyTorch recurrent model.
    """
    name = model_name.lower().strip()
    if name in ["rnn", "vanilla_rnn", "vanillarnn"]:
        return VanillaRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
        )
    elif name == "lstm":
        return LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
        )
    elif name == "gru":
        return GRUModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Choose from 'rnn', 'lstm', 'gru'."
        )


__all__ = ["VanillaRNN", "LSTMModel", "GRUModel", "build_model"]

