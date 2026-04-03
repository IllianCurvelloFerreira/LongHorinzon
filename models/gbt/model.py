from __future__ import annotations

import torch
import torch.nn as nn

from src.models.gbt.blocks import CausalSelfAttentionBlock, FirstStage
from src.models.gbt.embeddings import DataEmbedding


class GBTVanillaStandalone(nn.Module):
    def __init__(
        self,
        enc_in: int,
        dec_in: int,
        c_out: int,
        label_len: int,
        pred_len: int,
        fd_model: int,
        d_model: int,
        d_layers: int,
        n_heads: int,
        s_layers: list[int],
        dropout: float,
        use_time: bool,
        with_minute: bool,
    ):
        super().__init__()
        self.label_len = label_len
        self.pred_len = pred_len

        self.first_stage = FirstStage(
            enc_in=enc_in,
            c_out=c_out,
            label_len=label_len,
            pred_len=pred_len,
            fd_model=fd_model,
            pyramid_blocks=s_layers,
            dropout=dropout,
            use_time=use_time,
            with_minute=with_minute,
        )

        self.second_embed = DataEmbedding(
            c_in=dec_in,
            d_model=d_model,
            dropout=dropout,
            use_position=True,
            use_time=use_time,
            with_minute=with_minute,
        )

        self.second_stage = nn.ModuleList(
            [CausalSelfAttentionBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(d_layers)]
        )
        self.projection = nn.Conv1d(d_model, c_out, kernel_size=1)

    def set_stage_trainable(self, stage: str) -> None:
        if stage == "first stage":
            for p in self.first_stage.parameters():
                p.requires_grad = True
            for p in self.second_embed.parameters():
                p.requires_grad = False
            for layer in self.second_stage:
                for p in layer.parameters():
                    p.requires_grad = False
            for p in self.projection.parameters():
                p.requires_grad = False

        elif stage == "second stage":
            for p in self.first_stage.parameters():
                p.requires_grad = False
            for p in self.second_embed.parameters():
                p.requires_grad = True
            for layer in self.second_stage:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self.projection.parameters():
                p.requires_grad = True
        else:
            raise ValueError("stage deve ser 'first stage' ou 'second stage'.")

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        label_x = x_dec[:, :self.label_len, :]
        label_mark = x_mark_dec[:, :self.label_len, :] if x_mark_dec.shape[-1] > 0 else x_mark_dec

        base = self.first_stage(label_x, label_mark)

        if stage == "first stage":
            return base

        base_detached = base.detach()
        future_mark = x_mark_dec[:, -self.pred_len:, :] if x_mark_dec.shape[-1] > 0 else x_mark_dec[:, -self.pred_len:, :]
        h = self.second_embed(base_detached, future_mark)

        for block in self.second_stage:
            h = block(h)

        residual = self.projection(h.permute(0, 2, 1)).transpose(1, 2)
        return base_detached + residual