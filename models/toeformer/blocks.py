from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAvgDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        x_t = x.transpose(1, 2)  # [B, C, L]
        pad = self.kernel_size // 2
        x_pad = F.pad(x_t, (pad, pad), mode="replicate")
        trend = self.avg(x_pad).transpose(1, 2)  # [B, L, C]
        seasonal = x - trend
        return seasonal, trend


class GlobalLocalConvEncoder(nn.Module):
    def __init__(
        self,
        c_in: int,
        d_model: int = 128,
        k_global: int = 25,
        k_local: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(c_in, d_model)
        self.global_conv = nn.Conv1d(d_model, d_model, kernel_size=k_global, padding=k_global // 2)
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size=k_local, padding=k_local // 2)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fuse = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)  # [B, L, D]
        h_t = h.transpose(1, 2)

        g = self.drop(self.act(self.global_conv(h_t))).transpose(1, 2)
        l = self.drop(self.act(self.local_conv(h_t))).transpose(1, 2)

        out = self.fuse(torch.cat([g, l], dim=-1))
        return out


class SeasonalDecoderCrossAttn(nn.Module):
    def __init__(
        self,
        c_out: int,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.q_proj = nn.Linear(c_out, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.out = nn.Linear(d_model, c_out)

    def forward(self, seasonal_tail: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(seasonal_tail)
        attn_out, _ = self.attn(q, enc_out, enc_out)
        h = attn_out + self.ffn(attn_out)
        y_season = self.out(h)
        return y_season