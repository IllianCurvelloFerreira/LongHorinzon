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
    Entrada:
        x: [B, L, C_in]

    Saída:
        y_hat: [B, H, 1]
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        target_idx: int,
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
        self.c_out = c_out
        self.target_idx = target_idx
        self.lookback = lookback
        self.horizon = horizon

        self.decomp = MovingAvgDecomp(kernel_size=decomp_kernel)

        # tendência só da variável alvo
        self.trend_linear = nn.Linear(lookback, horizon)

        # encoder enxerga todas as variáveis
        self.season_encoder = GlobalLocalConvEncoder(
            c_in=c_in,
            d_model=d_model,
            k_global=k_global,
            k_local=k_local,
            dropout=dropout,
        )

        # decoder prevê só OT
        self.season_decoder = SeasonalDecoderCrossAttn(
            c_out=c_out,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B, L, C_in]
        returns:
          y_hat:    [B, H, 1]
          y_season: [B, H, 1]
          y_trend:  [B, H, 1]
        """
        seasonal, trend = self.decomp(x)

        # trend: só target
        trend_target = trend[:, :, self.target_idx:self.target_idx + 1]  # [B, L, 1]
        trend_t = trend_target.transpose(1, 2)                           # [B, 1, L]
        y_trend = self.trend_linear(trend_t)                             # [B, 1, H]
        y_trend = y_trend.transpose(1, 2)                                # [B, H, 1]

        # seasonal: query só target, contexto de todas as variáveis
        seasonal_target = seasonal[:, :, self.target_idx:self.target_idx + 1]  # [B, L, 1]
        tail_len = min(self.horizon, seasonal_target.shape[1])
        seasonal_tail = seasonal_target[:, -tail_len:, :]  # [B, tail_len, 1]

        enc_out = self.season_encoder(seasonal)  # [B, L, D]
        y_season = self.season_decoder(seasonal_tail, enc_out)  # [B, tail_len, 1]

        if tail_len < self.horizon:
            pad = torch.zeros(
                (x.size(0), self.horizon - tail_len, self.c_out),
                device=x.device,
                dtype=x.dtype,
            )
            y_season = torch.cat([y_season, pad], dim=1)

        y_hat = y_trend + y_season
        return y_hat, y_season, y_trend