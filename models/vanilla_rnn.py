"""
Vanilla (Elman) Recurrent Neural Network (RNN) for Time-Series Forecasting.
Uses PyTorch nn.RNN with configurable hidden dimensions, layers, and dropout.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    """
    Standard Elman RNN model for multi-step or single-step time-series forecasting.
    
    Mathematical Formulation:
        h_t = tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})
        \hat{y} = W_{ho} h_T + b_{ho}
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        output_size: int = 1,
        dropout: float = 0.0,
    ):
        """
        Args:
            input_size: Number of features per time step (1 for univariate series).
            hidden_size: Number of recurrent units in hidden state.
            num_layers: Number of recurrent layers.
            output_size: Dimension of forecasted output (f steps ahead).
            dropout: Dropout probability between recurrent layers (if num_layers > 1).
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_rate = dropout

        # Apply dropout between RNN layers if num_layers > 1
        rnn_dropout = dropout if num_layers > 1 else 0.0

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=rnn_dropout,
        )

        # Fully connected head mapping final hidden state to output forecast horizon
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(
        self, x: torch.Tensor, h_0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).
            h_0: Optional initial hidden state of shape (num_layers, batch_size, hidden_size).
            
        Returns:
            out: Forecast predictions of shape (batch_size, output_size).
            h_n: Final hidden state of shape (num_layers, batch_size, hidden_size).
        """
        # rnn_out: (batch_size, seq_len, hidden_size)
        # h_n: (num_layers, batch_size, hidden_size)
        rnn_out, h_n = self.rnn(x, h_0)

        # Extract last time step's hidden state
        last_hidden = rnn_out[:, -1, :]  # shape: (batch_size, hidden_size)

        # Apply dropout and linear projection
        dropped = self.dropout(last_hidden)
        out = self.fc(dropped)  # shape: (batch_size, output_size)

        return out, h_n

