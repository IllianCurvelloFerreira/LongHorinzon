from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.token_conv = nn.Conv1d(c_in, d_model, kernel_size=1)
        nn.init.kaiming_normal_(self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        w = torch.zeros(c_in, d_model)
        position = torch.arange(0, c_in, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        emb = nn.Embedding(c_in, d_model)
        emb.weight = nn.Parameter(w, requires_grad=False)
        self.emb = emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model: int, with_minute: bool):
        super().__init__()
        self.with_minute = with_minute
        self.month_embed = FixedEmbedding(13, d_model)
        self.day_embed = FixedEmbedding(32, d_model)
        self.weekday_embed = FixedEmbedding(7, d_model)
        self.hour_embed = FixedEmbedding(24, d_model)
        if with_minute:
            self.minute_embed = FixedEmbedding(4, d_model)

    def forward(self, x_mark: torch.Tensor) -> torch.Tensor:
        if x_mark.shape[-1] == 0:
            return 0.0

        out = (
            self.month_embed(x_mark[:, :, 0])
            + self.day_embed(x_mark[:, :, 1])
            + self.weekday_embed(x_mark[:, :, 2])
            + self.hour_embed(x_mark[:, :, 3])
        )

        if self.with_minute and x_mark.shape[-1] >= 5:
            out = out + self.minute_embed(x_mark[:, :, 4])

        return out


class DataEmbedding(nn.Module):
    def __init__(
        self,
        c_in: int,
        d_model: int,
        dropout: float,
        use_position: bool,
        use_time: bool,
        with_minute: bool,
    ):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model) if use_position else None
        self.temporal_embedding = TemporalEmbedding(d_model, with_minute) if use_time else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor) -> torch.Tensor:
        out = self.value_embedding(x)

        if self.position_embedding is not None:
            out = out + self.position_embedding(x)

        if self.temporal_embedding is not None:
            temp = self.temporal_embedding(x_mark)
            if isinstance(temp, torch.Tensor):
                out = out + temp

        return self.dropout(out)