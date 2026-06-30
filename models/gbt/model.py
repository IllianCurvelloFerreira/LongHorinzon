from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gbt.blocks import CausalSelfAttentionBlock, FirstStage
from models.gbt.embeddings import DataEmbedding


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
        target_idx: int = 0,
        input_mode: str = "univariate",
    ):
        super().__init__()

        self.label_len = label_len
        self.pred_len = pred_len
        self.target_idx = target_idx
        self.input_mode = input_mode

        # Branch principal: sempre focada no alvo.
        # Mesmo no multivariado, a primeira previsão é feita sobre a variável target.
        self.first_stage = FirstStage(
            enc_in=1,
            c_out=c_out,
            label_len=label_len,
            pred_len=pred_len,
            fd_model=fd_model,
            pyramid_blocks=s_layers,
            dropout=dropout,
            use_time=use_time,
            with_minute=with_minute,
        )

        # Branch auxiliar para contexto multivariado.
        # Essa branch usa todas as variáveis como apoio para ajustar a previsão do alvo.
        if input_mode == "multivariate":
            self.context_embed = DataEmbedding(
                c_in=enc_in,
                d_model=d_model,
                dropout=dropout,
                use_position=False,
                use_time=use_time,
                with_minute=with_minute,
            )

            self.context_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, c_out),
            )
        else:
            self.context_embed = None
            self.context_proj = None

        # Segunda etapa: recebe a previsão base e aprende um resíduo autorregressivo.
        self.second_embed = DataEmbedding(
            c_in=dec_in,
            d_model=d_model,
            dropout=dropout,
            use_position=True,
            use_time=use_time,
            with_minute=with_minute,
        )

        self.second_stage = nn.ModuleList(
            [
                CausalSelfAttentionBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(d_layers)
            ]
        )

        self.projection = nn.Conv1d(
            d_model,
            c_out,
            kernel_size=1,
        )

    def set_stage_trainable(self, stage: str) -> None:
        if stage == "first stage":
            for p in self.first_stage.parameters():
                p.requires_grad = True

            if self.context_embed is not None:
                for p in self.context_embed.parameters():
                    p.requires_grad = True

            if self.context_proj is not None:
                for p in self.context_proj.parameters():
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

            if self.context_embed is not None:
                for p in self.context_embed.parameters():
                    p.requires_grad = False

            if self.context_proj is not None:
                for p in self.context_proj.parameters():
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

    def _resize_context_to_pred_len(
        self,
        ctx: torch.Tensor,
        target_len: int,
    ) -> torch.Tensor:
        """
        Redimensiona o contexto multivariado para o mesmo horizonte da previsão base.

        Problema anterior:
            Para pred_len=96:
                base = [B, 96, C]
                ctx  = [B, 96, C]
                funciona.

            Para pred_len=192:
                base = [B, 192, C]
                ctx  = [B, 96, C]
                erro na soma.

        Correção:
            Interpolamos ctx na dimensão temporal para que ele tenha o mesmo tamanho
            de base, ou seja, pred_len.
        """
        if ctx.size(1) == target_len:
            return ctx

        ctx = F.interpolate(
            ctx.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

        return ctx

    def build_base_forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
    ) -> torch.Tensor:
        # Histórico da variável alvo.
        target_x = x_enc[
            :,
            -self.label_len:,
            self.target_idx:self.target_idx + 1,
        ]

        if x_mark_enc.shape[-1] > 0:
            target_mark = x_mark_enc[:, -self.label_len:, :]
        else:
            target_mark = x_mark_enc

        # Primeira previsão baseada apenas no alvo.
        # Shape esperado:
        #   base = [B, pred_len, c_out]
        base = self.first_stage(target_x, target_mark)

        # Contexto multivariado opcional.
        if self.input_mode == "multivariate":
            context_x = x_enc[:, -self.label_len:, :]

            if x_mark_enc.shape[-1] > 0:
                context_mark = x_mark_enc[:, -self.label_len:, :]
            else:
                context_mark = x_mark_enc

            # ctx começa com comprimento label_len.
            # Exemplo:
            #   label_len = 96
            #   pred_len = 192, 336 ou 720
            ctx = self.context_embed(context_x, context_mark)  # [B, label_len, D]
            ctx = self.context_proj(ctx)                       # [B, label_len, c_out]

            # Correção principal:
            # garante que ctx tenha o mesmo comprimento temporal de base.
            ctx = self._resize_context_to_pred_len(
                ctx=ctx,
                target_len=base.size(1),
            )

            base = base + ctx

        return base

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        base = self.build_base_forecast(
            x_enc=x_enc,
            x_mark_enc=x_mark_enc,
        )

        if stage == "first stage":
            return base

        base_detached = base.detach()

        if x_mark_dec.shape[-1] > 0:
            future_mark = x_mark_dec[:, -self.pred_len:, :]
        else:
            future_mark = x_mark_dec[:, -self.pred_len:, :]

        h = self.second_embed(
            base_detached,
            future_mark,
        )

        for block in self.second_stage:
            h = block(h)

        residual = self.projection(
            h.permute(0, 2, 1)
        ).transpose(1, 2)

        return base_detached + residual