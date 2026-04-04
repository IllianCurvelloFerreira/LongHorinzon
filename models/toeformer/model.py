from __future__ import annotations

import torch
import torch.nn as nn

from models.toeformer.blocks import (
    GlobalLocalConvEncoder,
    MovingAvgDecomp,
    SeasonalDecoderCrossAttn,
)


class TOEformer(nn.Module):
    """
    TOEformer-like:
    - decompose x into seasonal/trend
    - trend forecast by linear map
    - seasonal forecast by conv encoder + cross-attention decoder
    """
    def __init__(
        self,
        c_in: int,
        lookback: int,
        horizon: int,
        d_model: int = 128,
        n_heads: int = 4,
        decomp_kernel: int = 25,
        k_global: int = 25,
        k_local: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.c_in = c_in
        self.lookback = lookback
        self.horizon = horizon

        self.decomp = MovingAvgDecomp(kernel_size=decomp_kernel)

        self.trend_linear = nn.Linear(lookback, horizon)

        self.season_encoder = GlobalLocalConvEncoder(
            c_in=c_in,
            d_model=d_model,
            k_global=k_global,
            k_local=k_local,
            dropout=dropout,
        )
        self.season_decoder = SeasonalDecoderCrossAttn(
            c_out=c_in,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B, L, C]
        returns:
          y_hat:    [B, H, C]
          y_season: [B, H, C]
          y_trend:  [B, H, C]
        """
        seasonal, trend = self.decomp(x)

        trend_t = trend.transpose(1, 2)      # [B, C, L]
        y_trend = self.trend_linear(trend_t) # [B, C, H]
        y_trend = y_trend.transpose(1, 2)    # [B, H, C]

        tail_len = min(self.horizon, seasonal.shape[1])
        seasonal_tail = seasonal[:, -tail_len:, :]
        enc_out = self.season_encoder(seasonal)
        y_season = self.season_decoder(seasonal_tail, enc_out)

        if tail_len < self.horizon:
            pad = torch.zeros(
                (x.size(0), self.horizon - tail_len, self.c_in),
                device=x.device,
                dtype=x.dtype,
            )
            y_season = torch.cat([y_season, pad], dim=1)

        y_hat = y_trend + y_season
        return y_hat, y_season, y_trend