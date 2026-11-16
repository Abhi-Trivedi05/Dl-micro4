"""
Long Short-Term Memory (LSTM) Network for Time-Series Forecasting.
Employs input, forget, and output gating mechanisms with internal memory cell state.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM model with input gate, forget gate, cell candidate, output gate, and cell state.
    
    Mathematical Formulation:
        f_t = \\sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf})
        i_t = \\sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi})
        g_t = \\tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg})
        o_t = \\sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho})
        c_t = f_t \\odot c_{t-1} + i_t \\odot g_t
        h_t = o_t \\odot \\tanh(c_t)
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

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).
            state: Optional tuple (h_0, c_0) each of shape (num_layers, batch_size, hidden_size).
            
        Returns:
            out: Forecast predictions of shape (batch_size, output_size).
            (h_n, c_n): Final hidden state and cell state.
        """
        # lstm_out: (batch_size, seq_len, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x, state)

        # Extract last time step's hidden state
        last_hidden = lstm_out[:, -1, :]  # shape: (batch_size, hidden_size)

        dropped = self.dropout(last_hidden)
        out = self.fc(dropped)  # shape: (batch_size, output_size)

        return out, (h_n, c_n)

