from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
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
    smape: float


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    target_mean: float,
    target_scale: float,
) -> Metrics:
    """MSE/MAE padronizados; sMAPE em unidades originais do alvo OT."""
    model.eval()

    mse_sum = 0.0
    mae_sum = 0.0
    smape_sum = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        mse_sum += nn.functional.mse_loss(pred, y, reduction="sum").item()
        mae_sum += nn.functional.l1_loss(pred, y, reduction="sum").item()

        # OT padronizada -> OT na escala original.
        pred_original = pred * target_scale + target_mean
        true_original = y * target_scale + target_mean
        denominator = torch.abs(true_original) + torch.abs(pred_original)
        point_smape = (
            2.0 * torch.abs(pred_original - true_original)
            / torch.clamp(denominator, min=1e-8)
        )
        smape_sum += point_smape.sum().item()
        n += y.numel()

    if n == 0:
        raise ValueError("DataLoader de avaliacao sem amostras.")

    return Metrics(
        mse=mse_sum / n,
        mae=mae_sum / n,
        smape=100.0 * smape_sum / n,
    )


def apply_multivariate_pca(
    train_series: np.ndarray,
    val_series: np.ndarray,
    test_series: np.ndarray,
    target_idx: int,
    pca_components: int,
):
    """
    Mantém o alvo OT explícito como primeiro canal e aplica PCA
    apenas nas demais variáveis.

    Entrada:
        train/val/test: [T, C]

    Saída:
        train/val/test: [T, 1 + n_components]
        new_target_idx = 0
        input_size = 1 + n_components
    """
    if pca_components <= 0:
        raise ValueError("Quando use_pca=True, pca_components deve ser > 0.")

    train_target = train_series[:, target_idx:target_idx + 1]
    val_target = val_series[:, target_idx:target_idx + 1]
    test_target = test_series[:, target_idx:target_idx + 1]

    other_idx = [i for i in range(train_series.shape[1]) if i != target_idx]

    if len(other_idx) == 0:
        return train_target, val_target, test_target, 0, 1

    train_other = train_series[:, other_idx]
    val_other = val_series[:, other_idx]
    test_other = test_series[:, other_idx]

    n_components = min(pca_components, train_other.shape[1])

    pca = PCA(n_components=n_components)

    train_pca = pca.fit_transform(train_other).astype(np.float32)
    val_pca = pca.transform(val_other).astype(np.float32)
    test_pca = pca.transform(test_other).astype(np.float32)

    train_new = np.concatenate([train_target, train_pca], axis=1).astype(np.float32)
    val_new = np.concatenate([val_target, val_pca], axis=1).astype(np.float32)
    test_new = np.concatenate([test_target, test_pca], axis=1).astype(np.float32)

    new_target_idx = 0
    input_size = train_new.shape[1]

    return train_new, val_new, test_new, new_target_idx, input_size


def apply_multivariate_pls(
    train_series: np.ndarray,
    val_series: np.ndarray,
    test_series: np.ndarray,
    target_idx: int,
    pls_components: int,
):
    """
    Mantém o alvo OT explícito como primeiro canal e aplica PLS
    apenas nas demais variáveis, usando OT do treino como supervisão.

    O PLS é ajustado somente no treino.
    Validação e teste usam apenas transform.

    Entrada:
        train/val/test: [T, C]

    Saída:
        train/val/test: [T, 1 + n_components]
        new_target_idx = 0
        input_size = 1 + n_components
    """
    if pls_components <= 0:
        raise ValueError("Quando use_pls=True, pls_components deve ser > 0.")

    train_target = train_series[:, target_idx:target_idx + 1]
    val_target = val_series[:, target_idx:target_idx + 1]
    test_target = test_series[:, target_idx:target_idx + 1]

    other_idx = [i for i in range(train_series.shape[1]) if i != target_idx]

    if len(other_idx) == 0:
        return train_target, val_target, test_target, 0, 1

    train_other = train_series[:, other_idx]
    val_other = val_series[:, other_idx]
    test_other = test_series[:, other_idx]

    n_components = min(pls_components, train_other.shape[1])

    pls = PLSRegression(n_components=n_components)

    # Fit supervisionado apenas no treino.
    pls.fit(train_other, train_target)

    train_pls = pls.transform(train_other).astype(np.float32)
    val_pls = pls.transform(val_other).astype(np.float32)
    test_pls = pls.transform(test_other).astype(np.float32)

    train_new = np.concatenate([train_target, train_pls], axis=1).astype(np.float32)
    val_new = np.concatenate([val_target, val_pls], axis=1).astype(np.float32)
    test_new = np.concatenate([test_target, test_pls], axis=1).astype(np.float32)

    new_target_idx = 0
    input_size = train_new.shape[1]

    return train_new, val_new, test_new, new_target_idx, input_size


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

        target_mean = float(scaler.mean_[0])
        target_scale = float(scaler.scale_[0])

        # No univariado, PCA/PLS não são aplicados.
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
        train_series = scaler.fit_transform(train_series).astype(np.float32)
        val_series = scaler.transform(val_series).astype(np.float32)
        test_series = scaler.transform(test_series).astype(np.float32)

        # Registrar scaler de OT antes de PCA/PLS mudar os indices.
        target_mean = float(scaler.mean_[target_idx])
        target_scale = float(scaler.scale_[target_idx])

        input_size = len(feature_cols)

        # PCA opcional: só no multivariado.
        # Se --use_pca não for passado, nada muda no fluxo atual.
        if args.use_pca:
            train_series, val_series, test_series, target_idx, input_size = apply_multivariate_pca(
                train_series=train_series,
                val_series=val_series,
                test_series=test_series,
                target_idx=target_idx,
                pca_components=args.pca_components,
            )

        # PLS opcional: só no multivariado.
        # Se --use_pls não for passado, nada muda no fluxo atual.
        if args.use_pls:
            train_series, val_series, test_series, target_idx, input_size = apply_multivariate_pls(
                train_series=train_series,
                val_series=val_series,
                test_series=test_series,
                target_idx=target_idx,
                pls_components=args.pls_components,
            )

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

    return train_loader, val_loader, test_loader, input_size, target_mean, target_scale


def run_experiment(args, run_seed: int, device: str, set_seed_fn):
    set_seed_fn(run_seed)

    train_loader, val_loader, test_loader, input_size, target_mean, target_scale = build_loaders(args)

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
        f"input_mode={args.input_mode} | use_pca={args.use_pca} | use_pls={args.use_pls} | "
        f"device={device} ====="
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
        val_metrics = evaluate(
            model, val_loader, device,
            target_mean=target_mean, target_scale=target_scale,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train={train_loss:.6f} | "
            f"Val MSE={val_metrics.mse:.6f} | "
            f"Val MAE={val_metrics.mae:.6f} | "
            f"Val sMAPE={val_metrics.smape:.2f}%"
        )

        if val_metrics.mse < best_val:
            best_val = val_metrics.mse
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(
        model, test_loader, device,
        target_mean=target_mean, target_scale=target_scale,
    )
    print(
        f"TEST | MSE={test_metrics.mse:.6f} | "
        f"MAE={test_metrics.mae:.6f} | sMAPE={test_metrics.smape:.2f}%"
    )

    return test_metrics
