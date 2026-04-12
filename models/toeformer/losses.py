from __future__ import annotations

import torch
import torch.nn.functional as F


def lrve_loss(
    y_season_true: torch.Tensor,
    y_season_pred: torch.Tensor,
    y_trend_true: torch.Tensor,
    y_trend_pred: torch.Tensor,
    alpha: float = 0.6,
    eps: float = 1e-12,
):
    se = y_season_true - y_season_pred
    te = y_trend_true - y_trend_pred

    a = torch.tensor(alpha, device=se.device, dtype=se.dtype)
    a = torch.clamp(a, eps, 1.0 - eps)

    exp_term = torch.pow(a, (se + te).pow(2))
    term = 1.0 - exp_term + se.pow(2) + te.pow(2)
    return term.mean()


def toeformer_total_loss(
    y_true: torch.Tensor,
    y_hat: torch.Tensor,
    y_season_true: torch.Tensor,
    y_season_pred: torch.Tensor,
    y_trend_true: torch.Tensor,
    y_trend_pred: torch.Tensor,
    lam1: float = 1.0,
    lam2: float = 0.1,
    alpha: float = 0.6,
):
    mse = F.mse_loss(y_hat, y_true)
    lrve = lrve_loss(
        y_season_true=y_season_true,
        y_season_pred=y_season_pred,
        y_trend_true=y_trend_true,
        y_trend_pred=y_trend_pred,
        alpha=alpha,
    )
    return lam1 * mse + lam2 * lrve, mse.detach(), lrve.detach()