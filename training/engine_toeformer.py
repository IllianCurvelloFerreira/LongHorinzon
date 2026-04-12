from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from datasets.ett_sliding_window import (
    SlidingWindowDataset,
    SlidingWindowTargetDataset,
    load_multivariate_series,
    load_univariate_series,
    train_val_test_split_time,
)
from models.toeformer.blocks import MovingAvgDecomp
from models.toeformer.losses import toeformer_total_loss
from models.toeformer.model import TOEformer


@dataclass
class Metrics:
    mse: float
    mae: float


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    mse_sum, mae_sum, n = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_hat, _, _ = model(x)

        mse_sum += torch.nn.functional.mse_loss(y_hat, y, reduction="sum").item()
        mae_sum += torch.nn.functional.l1_loss(y_hat, y, reduction="sum").item()
        n += y.numel()

    return mse_sum / n, mae_sum / n


def build_loaders(args):
    if args.input_mode == "univariate":
        series = load_univariate_series(
            root_path=args.root_path,
            data_name=args.data,
            target_col=args.target,
            data_dir=args.data_dir,
        )
        target_idx = 0
        feature_dim = 1

        train_series, val_series, test_series = train_val_test_split_time(
            series,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )

        scaler = StandardScaler()
        train_series = scaler.fit_transform(train_series)
        val_series = scaler.transform(val_series)
        test_series = scaler.transform(test_series)

        train_ds = SlidingWindowDataset(
            train_series,
            lookback=args.lookback,
            horizon=args.horizon,
            stride=args.stride,
        )
        val_ds = SlidingWindowDataset(
            val_series,
            lookback=args.lookback,
            horizon=args.horizon,
            stride=args.stride,
        )
        test_ds = SlidingWindowDataset(
            test_series,
            lookback=args.lookback,
            horizon=args.horizon,
            stride=args.stride,
        )

    elif args.input_mode == "multivariate":
        series, target_idx, feature_cols = load_multivariate_series(
            root_path=args.root_path,
            data_name=args.data,
            target_col=args.target,
            data_dir=args.data_dir,
        )
        feature_dim = len(feature_cols)

        train_series, val_series, test_series = train_val_test_split_time(
            series,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )

        scaler = StandardScaler()
        train_series = scaler.fit_transform(train_series)
        val_series = scaler.transform(val_series)
        test_series = scaler.transform(test_series)

        train_ds = SlidingWindowTargetDataset(
            train_series,
            lookback=args.lookback,
            horizon=args.horizon,
            target_idx=target_idx,
            stride=args.stride,
        )
        val_ds = SlidingWindowTargetDataset(
            val_series,
            lookback=args.lookback,
            horizon=args.horizon,
            target_idx=target_idx,
            stride=args.stride,
        )
        test_ds = SlidingWindowTargetDataset(
            test_series,
            lookback=args.lookback,
            horizon=args.horizon,
            target_idx=target_idx,
            stride=args.stride,
        )

    else:
        raise ValueError("input_mode deve ser 'univariate' ou 'multivariate'.")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader, test_loader, target_idx, feature_dim


def run_experiment(args, run_seed: int, device: str, set_seed_fn):
    set_seed_fn(run_seed)

    train_loader, val_loader, test_loader, target_idx, feature_dim = build_loaders(args)

    model = TOEformer(
        c_in=feature_dim,
        c_out=1,
        target_idx=target_idx,
        lookback=args.lookback,
        horizon=args.horizon,
        d_model=args.d_model,
        n_heads=args.n_heads,
        decomp_kernel=args.decomp_kernel,
        k_global=args.k_global,
        k_local=args.k_local,
        dropout=args.dropout,
    ).to(device)

    model.decomp = MovingAvgDecomp(kernel_size=model.decomp.kernel_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val = float("inf")
    best_state = None

    print(
        f"\n===== Run seed={run_seed} | data={args.data} | horizon={args.horizon} | "
        f"input_mode={args.input_mode} | device={device} ====="
    )

    for epoch in range(1, args.train_epochs + 1):
        model.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            y_hat, y_season_pred, y_trend_pred = model(x)

            if args.input_mode == "univariate":
                xy_target = torch.cat([x, y], dim=1)  # [B, L+H, 1]
            else:
                x_target = x[:, :, target_idx:target_idx + 1]
                xy_target = torch.cat([x_target, y], dim=1)  # [B, L+H, 1]

            season_xy, trend_xy = model.decomp(xy_target)

            y_season_true = season_xy[:, -args.horizon:, :]
            y_trend_true = trend_xy[:, -args.horizon:, :]

            loss, mse, lrve = toeformer_total_loss(
                y_true=y,
                y_hat=y_hat,
                y_season_true=y_season_true,
                y_season_pred=y_season_pred,
                y_trend_true=y_trend_true,
                y_trend_pred=y_trend_pred,
                lam1=args.lam1,
                lam2=args.lam2,
                alpha=args.alpha,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_mse, val_mae = evaluate(model, val_loader, device)

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d} | "
            f"Val MSE={val_mse:.6f} | "
            f"Val MAE={val_mae:.6f} | "
            f"Best={best_val:.6f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_mse, test_mae = evaluate(model, test_loader, device)
    print(f"TEST | MSE={test_mse:.6f} | MAE={test_mae:.6f}")

    return Metrics(mse=test_mse, mae=test_mae)