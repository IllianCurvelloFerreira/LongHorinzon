from __future__ import annotations

import torch
import torch.nn as nn

from src.models.gbt.embeddings import DataEmbedding


class ResidualDownBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, dropout: float = 0.05, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_in, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv1d(c_in, c_out, kernel_size=3, padding=1)
        self.skip = nn.Conv1d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.permute(0, 2, 1)
        y = self.pool(self.skip(z)).transpose(1, 2)

        z = self.dropout(self.conv1(z))
        z = self.act(z)
        z = self.dropout(self.conv2(z))
        z = self.act(z)
        z = self.pool(z).transpose(1, 2)

        return z + y


class PyramidEncoder(nn.Module):
    def __init__(self, d_model: int, block_nums: int, pred_len: int, c_out: int, dropout: float = 0.05):
        super().__init__()
        layers = []
        c_in = d_model

        for i in range(block_nums):
            c_out_block = d_model * (2 ** (i + 1))
            layers.append(ResidualDownBlock(c_in, c_out_block, dropout=dropout))
            c_in = c_out_block

        self.blocks = nn.ModuleList(layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(pred_len * c_out),
        )
        self.pred_len = pred_len
        self.c_out = c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        out = self.head(x)
        return out.view(-1, self.pred_len, self.c_out)


class FirstStage(nn.Module):
    def __init__(
        self,
        enc_in: int,
        c_out: int,
        label_len: int,
        pred_len: int,
        fd_model: int,
        pyramid_blocks: list[int],
        dropout: float,
        use_time: bool,
        with_minute: bool,
    ):
        super().__init__()
        self.label_len = label_len

        self.embeds = nn.ModuleList(
            [
                DataEmbedding(
                    c_in=enc_in,
                    d_model=fd_model,
                    dropout=dropout,
                    use_position=False,
                    use_time=use_time,
                    with_minute=with_minute,
                )
                for _ in pyramid_blocks
            ]
        )

        self.encoders = nn.ModuleList(
            [
                PyramidEncoder(
                    d_model=fd_model,
                    block_nums=block_nums,
                    pred_len=pred_len,
                    c_out=c_out,
                    dropout=dropout,
                )
                for block_nums in pyramid_blocks
            ]
        )

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor) -> torch.Tensor:
        outputs = []

        for i, (embed, encoder) in enumerate(zip(self.embeds, self.encoders)):
            cur_len = self.label_len // (2 ** i)
            x_i = x[:, -cur_len:, :]
            x_mark_i = x_mark[:, -cur_len:, :] if x_mark.shape[-1] > 0 else x_mark
            h = embed(x_i, x_mark_i)
            outputs.append(encoder(h))

        return torch.stack(outputs, dim=0).mean(dim=0)


class CausalSelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        mask = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x