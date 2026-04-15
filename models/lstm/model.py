# Univariate model
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    Encoder-decoder LSTM simples para forecasting multi-step.

    Entrada:
        x: [B, L, C]

    Saída:
        y_hat: [B, H, C]
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        horizon: int = 96,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.decoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.projection = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C]
        return: [B, H, C]
        """
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype

        _, (h_n, c_n) = self.encoder(x)

        dec_in = torch.zeros(batch_size, self.horizon, self.input_size, device=device, dtype=dtype)
        dec_out, _ = self.decoder(dec_in, (h_n, c_n))

        y_hat = self.projection(dec_out)
        return y_hat