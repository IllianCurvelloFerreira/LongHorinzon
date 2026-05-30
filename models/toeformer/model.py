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
    TOEformer com dois modos de saída:

    1) target_idx = int
       - comportamento antigo
       - entrada:  [B, L, C_in]
       - saída:   [B, H, 1]
       - prevê apenas a variável alvo, por exemplo OT

    2) target_idx = None
       - novo comportamento
       - entrada:  [B, L, C_in]
       - saída:   [B, H, C_in]
       - prevê todos os canais/variáveis
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        target_idx: int | None,
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

        # Trend: Linear L -> H aplicado por canal de saída.
        # Se target_idx=int, usa só 1 canal.
        # Se target_idx=None, usa todos os canais.
        self.trend_linear = nn.Linear(lookback, horizon)

        # Encoder continua enxergando todas as variáveis de entrada.
        self.season_encoder = GlobalLocalConvEncoder(
            c_in=c_in,
            d_model=d_model,
            k_global=k_global,
            k_local=k_local,
            dropout=dropout,
        )

        # Decoder prevê c_out canais:
        # - c_out=1 no modo antigo
        # - c_out=C no modo multivariado
        self.season_decoder = SeasonalDecoderCrossAttn(
            c_out=c_out,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

    def _select_output_channels(self, x: torch.Tensor) -> torch.Tensor:
        """
        Seleciona os canais que serão previstos.

        target_idx=int:
            retorna apenas o canal alvo [B, L, 1]

        target_idx=None:
            retorna todos os canais [B, L, C]
        """
        if self.target_idx is None:
            return x

        return x[:, :, self.target_idx:self.target_idx + 1]

    def forward(self, x: torch.Tensor):
        """
        x: [B, L, C_in]

        returns:
            y_hat:    [B, H, c_out]
            y_season: [B, H, c_out]
            y_trend:  [B, H, c_out]
        """
        seasonal, trend = self.decomp(x)

        # No modo antigo, seleciona apenas OT.
        # No modo multivariado, mantém todos os canais.
        trend_out = self._select_output_channels(trend)
        seasonal_out = self._select_output_channels(seasonal)

        # Trend forecast
        trend_t = trend_out.transpose(1, 2)      # [B, c_out, L]
        y_trend = self.trend_linear(trend_t)     # [B, c_out, H]
        y_trend = y_trend.transpose(1, 2)        # [B, H, c_out]

        # Seasonal forecast
        tail_len = min(self.horizon, seasonal_out.shape[1])
        seasonal_tail = seasonal_out[:, -tail_len:, :]  # [B, tail_len, c_out]

        # Encoder usa todas as variáveis sazonais como contexto.
        enc_out = self.season_encoder(seasonal)  # [B, L, D]

        y_season = self.season_decoder(
            seasonal_tail,
            enc_out,
        )  # [B, tail_len, c_out]

        if tail_len < self.horizon:
            pad = torch.zeros(
                (x.size(0), self.horizon - tail_len, self.c_out),
                device=x.device,
                dtype=x.dtype,
            )
            y_season = torch.cat([y_season, pad], dim=1)

        y_hat = y_trend + y_season

        return y_hat, y_season, y_trend