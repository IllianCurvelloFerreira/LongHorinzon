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
from models.lstm.model import LSTMForecaster


@dataclass
class Metrics:
    mse: float
    mae: float


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Metrics:
    model.eval()

    mse_sum = 0.0
    mae_sum = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        mse_sum += nn.functional.mse_loss(pred, y, reduction="sum").item()
        mae_sum += nn.functional.l1_loss(pred, y, reduction="sum").item()
        n += y.numel()

    return Metrics(
        mse=mse_sum / n,
        mae=mae_sum / n,
    )


def build_loaders(args):
    if args.input_mode == "univariate":
        series = load_univariate_series(
            root_path=args.root_path,
            data_name=args.data,
            target_col=args.target,
            data_dir=args.data_dir,
        )

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

        input_size = 1

    elif args.input_mode == "multivariate":
        series, target_idx, feature_cols = load_multivariate_series(
            root_path=args.root_path,
            data_name=args.data,
            target_col=args.target,
            data_dir=args.data_dir,
        )

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

        input_size = len(feature_cols)

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

    return train_loader, val_loader, test_loader, input_size


def run_experiment(args, run_seed: int, device: str, set_seed_fn):
    set_seed_fn(run_seed)

    train_loader, val_loader, test_loader, input_size = build_loaders(args)

    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        horizon=args.horizon,
        output_size=1,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    print(
        f"\n===== Run seed={run_seed} | data={args.data} | horizon={args.horizon} | "
        f"input_mode={args.input_mode} | device={device} ====="
    )

    for epoch in range(1, args.train_epochs + 1):
        model.train()
        train_losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"Train={train_loss:.6f} | "
            f"Val MSE={val_metrics.mse:.6f} | "
            f"Val MAE={val_metrics.mae:.6f}"
        )

        if val_metrics.mse < best_val:
            best_val = val_metrics.mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device)
    print(f"TEST | MSE={test_metrics.mse:.6f} | MAE={test_metrics.mae:.6f}")

    return test_metrics