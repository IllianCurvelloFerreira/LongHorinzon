from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.ett_gbt import ETTGBTDataset
from models.gbt.model import GBTVanillaStandalone


@dataclass
class Metrics:
    mse: float
    mae: float


def parse_s_layers(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def make_decoder_input(batch_y: torch.Tensor, label_len: int, pred_len: int) -> torch.Tensor:
    zeros = torch.zeros(batch_y.size(0), pred_len, batch_y.size(-1), device=batch_y.device)
    return torch.cat([batch_y[:, :label_len, :], zeros], dim=1)


def process_batch(model: nn.Module, batch, label_len: int, pred_len: int, device: str, stage: str):
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch

    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    batch_x_mark = batch_x_mark.to(device)
    batch_y_mark = batch_y_mark.to(device)

    dec_inp = make_decoder_input(batch_y, label_len, pred_len)
    pred = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, stage=stage)
    true = batch_y[:, -pred_len:, :]

    return pred, true


@torch.no_grad()
def evaluate_loss(model, loader, criterion, args, device, stage: str):
    model.eval()
    mse_losses = []
    mae_losses = []

    for batch in loader:
        pred, true = process_batch(model, batch, args.label_len, args.pred_len, device, stage)

        mse = criterion(pred, true)
        mae = torch.mean(torch.abs(pred - true))

        mse_losses.append(mse.item())
        mae_losses.append(mae.item())

    mean_mse = float(np.mean(mse_losses)) if mse_losses else np.nan
    mean_mae = float(np.mean(mae_losses)) if mae_losses else np.nan
    return mean_mse, mean_mae


def train_one_stage(model, train_loader, val_loader, args, device: str, stage: str):
    model.set_stage_trainable(stage)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(1, args.train_epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            optimizer.zero_grad()
            pred, true = process_batch(model, batch, args.label_len, args.pred_len, device, stage)
            loss = criterion(pred, true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        val_mse, val_mae = evaluate_loss(model, val_loader, criterion, args, device, stage)

        print(
            f"[{stage}] Epoch {epoch:02d} | "
            f"Train: {train_loss:.6f} | "
            f"Val MSE: {val_mse:.6f} | "
            f"Val MAE: {val_mae:.6f}"
        )

        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    print(f"Melhor época em {stage}: {best_epoch} | Best Val: {best_val:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_state


@torch.no_grad()
def test_model(model, test_loader, test_ds, args, device: str, stage: str) -> Metrics:
    model.eval()
    preds, trues = [], []

    for batch in test_loader:
        pred, true = process_batch(model, batch, args.label_len, args.pred_len, device, stage)
        preds.append(pred.detach().cpu().numpy())
        trues.append(true.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    if args.test_inverse:
        preds = test_ds.inverse_transform_y(preds)
        trues = test_ds.inverse_transform_y(trues)

    mse = float(np.mean((preds - trues) ** 2))
    mae = float(np.mean(np.abs(preds - trues)))
    return Metrics(mse=mse, mae=mae)


def build_loaders(args):
    train_ds = ETTGBTDataset(
        root_path=args.root_path,
        data_name=args.data,
        flag="train",
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        target=args.target,
        criterion=args.criterion,
        use_time=args.time,
        input_mode=args.input_mode,
        data_dir=args.data_dir,
    )
    val_ds = ETTGBTDataset(
        root_path=args.root_path,
        data_name=args.data,
        flag="val",
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        target=args.target,
        criterion=args.criterion,
        use_time=args.time,
        input_mode=args.input_mode,
        data_dir=args.data_dir,
    )
    test_ds = ETTGBTDataset(
        root_path=args.root_path,
        data_name=args.data,
        flag="test",
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        target=args.target,
        criterion=args.criterion,
        use_time=args.time,
        input_mode=args.input_mode,
        data_dir=args.data_dir,
    )

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

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def run_experiment(args, run_seed: int, device: str, set_seed_fn) -> Metrics:
    set_seed_fn(run_seed)

    _, _, test_ds, train_loader, val_loader, test_loader = build_loaders(args)

    s_layers = parse_s_layers(args.s_layers)
    with_minute = args.data in {"ETTm1", "ETTm2"}

    enc_in = train_loader.dataset.enc_in
    c_out = 1

    model = GBTVanillaStandalone(
        enc_in=enc_in,
        dec_in=1,
        c_out=c_out,
        label_len=args.label_len,
        pred_len=args.pred_len,
        fd_model=args.fd_model,
        d_model=args.d_model,
        d_layers=args.d_layers,
        n_heads=args.n_heads,
        s_layers=s_layers,
        dropout=args.dropout,
        use_time=args.time,
        with_minute=with_minute,
    ).to(device)

    print(
        f"\n===== Run seed={run_seed} | data={args.data} | pred_len={args.pred_len} | "
        f"input_mode={args.input_mode} | device={device} ====="
    )

    best_first = train_one_stage(model, train_loader, val_loader, args, device, stage="first stage")
    model.load_state_dict(best_first)

    best_second = train_one_stage(model, train_loader, val_loader, args, device, stage="second stage")
    model.load_state_dict(best_second)

    metrics = test_model(model, test_loader, test_ds, args, device, stage="second stage")
    print(f"Test | MSE: {metrics.mse:.6f} | MAE: {metrics.mae:.6f}")
    return metrics