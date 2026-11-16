"""
Gated Recurrent Unit (GRU) Network for Time-Series Forecasting.
Combines forget and input gates into an update gate, streamlining recurrent memory without a separate cell state.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class GRUModel(nn.Module):
    """
    GRU model with reset gate and update gate.
    
    Mathematical Formulation:
        r_t = \\sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr})
        z_t = \\sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz})
        n_t = \\tanh(W_{in} x_t + b_{in} + r_t \\odot (W_{hn} h_{t-1} + b_{hn}))
        h_t = (1 - z_t) \\odot n_t + z_t \\odot h_{t-1}
        \\hat{y} = W_{fc} h_T + b_{fc}
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

        gru_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

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
        # gru_out: (batch_size, seq_len, hidden_size)
        gru_out, h_n = self.gru(x, h_0)

        # Extract last time step's hidden state
        last_hidden = gru_out[:, -1, :]  # shape: (batch_size, hidden_size)

        dropped = self.dropout(last_hidden)
        out = self.fc(dropped)  # shape: (batch_size, output_size)

        return out, h_n

