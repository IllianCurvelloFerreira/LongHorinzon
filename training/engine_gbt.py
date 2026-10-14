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
class StageMetrics:
    mse: float
    mae: float
    smape: float


@dataclass
class Metrics:
    # Compatibilidade: mse/mae continuam representando a Stage 2, como antes.
    mse: float
    mae: float
    smape: float
    stage1_mse: float
    stage1_mae: float
    stage1_smape: float


def parse_s_layers(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def make_decoder_input(batch_y: torch.Tensor, label_len: int, pred_len: int) -> torch.Tensor:
    zeros = torch.zeros(
        batch_y.size(0),
        pred_len,
        batch_y.size(-1),
        device=batch_y.device,
    )
    return torch.cat([batch_y[:, :label_len, :], zeros], dim=1)


def process_batch(
    model: nn.Module,
    batch,
    label_len: int,
    pred_len: int,
    device: str,
    stage: str,
):
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch

    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    batch_x_mark = batch_x_mark.to(device)
    batch_y_mark = batch_y_mark.to(device)

    dec_inp = make_decoder_input(batch_y, label_len, pred_len)

    pred = model(
        x_enc=batch_x,
        x_mark_enc=batch_x_mark,
        x_dec=dec_inp,
        x_mark_dec=batch_y_mark,
        stage=stage,
    )

    true = batch_y[:, -pred_len:, :]
    return pred, true


def calculate_smape_original(
    preds_original: np.ndarray,
    trues_original: np.ndarray,
    eps: float = 1e-8,
) -> float:
    denominator = np.abs(trues_original) + np.abs(preds_original)
    values = (
        2.0
        * np.abs(preds_original - trues_original)
        / np.maximum(denominator, eps)
    )
    return float(100.0 * np.mean(values))


@torch.no_grad()
def evaluate_loss(
    model,
    loader,
    dataset,
    criterion,
    args,
    device,
    stage: str,
):
    """
    MSE e MAE mantêm exatamente a lógica anterior de validação.
    O sMAPE é apenas acrescentado e calculado na escala original.
    O melhor checkpoint continua sendo escolhido exclusivamente por Val MSE.
    """
    model.eval()

    mse_losses = []
    mae_losses = []
    smape_sum = 0.0
    smape_n = 0

    for batch in loader:
        pred, true = process_batch(
            model=model,
            batch=batch,
            label_len=args.label_len,
            pred_len=args.pred_len,
            device=device,
            stage=stage,
        )

        mse = criterion(pred, true)
        mae = torch.mean(torch.abs(pred - true))

        mse_losses.append(mse.item())
        mae_losses.append(mae.item())

        pred_np = pred.detach().cpu().numpy()
        true_np = true.detach().cpu().numpy()

        pred_original = dataset.inverse_transform_y(pred_np)
        true_original = dataset.inverse_transform_y(true_np)

        denominator = np.abs(true_original) + np.abs(pred_original)
        values = (
            2.0
            * np.abs(pred_original - true_original)
            / np.maximum(denominator, 1e-8)
        )

        smape_sum += float(values.sum())
        smape_n += int(values.size)

    mean_mse = float(np.mean(mse_losses)) if mse_losses else np.nan
    mean_mae = float(np.mean(mae_losses)) if mae_losses else np.nan
    mean_smape = 100.0 * smape_sum / smape_n if smape_n > 0 else np.nan

    return mean_mse, mean_mae, mean_smape


def train_one_stage(
    model,
    train_loader,
    val_loader,
    val_ds,
    args,
    device: str,
    stage: str,
):
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

            pred, true = process_batch(
                model=model,
                batch=batch,
                label_len=args.label_len,
                pred_len=args.pred_len,
                device=device,
                stage=stage,
            )

            loss = criterion(pred, true)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        val_mse, val_mae, val_smape = evaluate_loss(
            model=model,
            loader=val_loader,
            dataset=val_ds,
            criterion=criterion,
            args=args,
            device=device,
            stage=stage,
        )

        print(
            f"[{stage}] Epoch {epoch:02d} | "
            f"Train: {train_loss:.6f} | "
            f"Val MSE: {val_mse:.6f} | "
            f"Val MAE: {val_mae:.6f} | "
            f"Val sMAPE: {val_smape:.2f}%"
        )

        # Importante: seleção continua EXCLUSIVAMENTE por Val MSE.
        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    print(f"Melhor época em {stage}: {best_epoch} | Best Val: {best_val:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_state


def print_smape_diagnostics(
    preds_original: np.ndarray,
    trues_original: np.ndarray,
) -> None:
    y_pred = preds_original.reshape(-1)
    y_true = trues_original.reshape(-1)

    abs_y = np.abs(y_true)
    denominator = np.abs(y_true) + np.abs(y_pred)
    smape_point = (
        200.0
        * np.abs(y_pred - y_true)
        / np.maximum(denominator, 1e-8)
    )
    abs_error = np.abs(y_pred - y_true)

    buckets = [
        ("|y| < 1", abs_y < 1.0),
        ("1 <= |y| < 2", (abs_y >= 1.0) & (abs_y < 2.0)),
        ("2 <= |y| < 5", (abs_y >= 2.0) & (abs_y < 5.0)),
        ("5 <= |y| < 10", (abs_y >= 5.0) & (abs_y < 10.0)),
        ("|y| >= 10", abs_y >= 10.0),
    ]

    total = len(y_true)

    print("\n===== DIAGNÓSTICO sMAPE NO TESTE — GBT STAGE 2 =====")
    print("Faixa baseada em |y_true| na escala original.")
    print(
        f"{'Faixa':<18} "
        f"{'N':>10} "
        f"{'% amostra':>11} "
        f"{'sMAPE':>10} "
        f"{'MAE orig.':>12}"
    )
    print("-" * 66)

    for label, mask in buckets:
        count = int(mask.sum())

        if count == 0:
            print(
                f"{label:<18} "
                f"{0:>10d} "
                f"{0.0:>10.2f}% "
                f"{'-':>10} "
                f"{'-':>12}"
            )
            continue

        share = 100.0 * count / total
        bucket_smape = float(np.mean(smape_point[mask]))
        bucket_mae = float(np.mean(abs_error[mask]))

        print(
            f"{label:<18} "
            f"{count:>10d} "
            f"{share:>10.2f}% "
            f"{bucket_smape:>9.2f}% "
            f"{bucket_mae:>12.4f}"
        )

    print("-" * 66)
    print(
        f"{'TOTAL':<18} "
        f"{total:>10d} "
        f"{100.0:>10.2f}% "
        f"{float(np.mean(smape_point)):>9.2f}% "
        f"{float(np.mean(abs_error)):>12.4f}"
    )

    print(f"\nValores com y_true <= 0: {100.0 * np.mean(y_true <= 0.0):.2f}%")
    print(f"Valores com |y_true| < 2: {100.0 * np.mean(abs_y < 2.0):.2f}%")


@torch.no_grad()
def test_model(
    model,
    test_loader,
    test_ds,
    args,
    device: str,
    stage: str,
    diagnostics: bool = False,
) -> StageMetrics:
    """
    Preserva MSE/MAE exatamente como no código anterior.

    - Sem --test_inverse: MSE/MAE permanecem na escala normalizada.
    - Com --test_inverse: MSE/MAE permanecem na escala original, como antes.
    - sMAPE é sempre calculado na escala original.
    """
    model.eval()

    preds = []
    trues = []

    for batch in test_loader:
        pred, true = process_batch(
            model=model,
            batch=batch,
            label_len=args.label_len,
            pred_len=args.pred_len,
            device=device,
            stage=stage,
        )

        preds.append(pred.detach().cpu().numpy())
        trues.append(true.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # Sempre guardamos uma cópia na escala original para o sMAPE.
    preds_original = test_ds.inverse_transform_y(preds)
    trues_original = test_ds.inverse_transform_y(trues)

    # MSE/MAE: comportamento antigo preservado.
    if args.test_inverse:
        preds_metric = preds_original
        trues_metric = trues_original
    else:
        preds_metric = preds
        trues_metric = trues

    mse = float(np.mean((preds_metric - trues_metric) ** 2))
    mae = float(np.mean(np.abs(preds_metric - trues_metric)))
    smape = calculate_smape_original(preds_original, trues_original)

    if diagnostics:
        print_smape_diagnostics(
            preds_original=preds_original,
            trues_original=trues_original,
        )

    return StageMetrics(mse=mse, mae=mae, smape=smape)


def build_loaders(args):
    common = dict(
        root_path=args.root_path,
        data_name=args.data,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        target=args.target,
        criterion=args.criterion,
        use_time=args.time,
        input_mode=args.input_mode,
        data_dir=args.data_dir,
        split_mode=args.split_mode,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    train_ds = ETTGBTDataset(flag="train", **common)
    val_ds = ETTGBTDataset(flag="val", **common)
    test_ds = ETTGBTDataset(flag="test", **common)

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


def resolve_target_idx(train_ds, args) -> int:
    if args.input_mode == "univariate":
        return 0

    if not hasattr(train_ds, "x_cols"):
        raise AttributeError(
            "ETTGBTDataset precisa expor 'self.x_cols' para o modo multivariado."
        )

    if args.target not in train_ds.x_cols:
        raise ValueError(
            f"Target '{args.target}' não encontrado em train_ds.x_cols={train_ds.x_cols}"
        )

    return train_ds.x_cols.index(args.target)


def run_experiment(args, run_seed: int, device: str, set_seed_fn) -> Metrics:
    set_seed_fn(run_seed)

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_loaders(args)

    s_layers = parse_s_layers(args.s_layers)
    with_minute = args.data in {"ETTm1", "ETTm2"}

    enc_in = train_ds.enc_in
    c_out = 1
    target_idx = resolve_target_idx(train_ds, args)

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
        target_idx=target_idx,
        input_mode=args.input_mode,
    ).to(device)

    print(
        f"\n===== Run seed={run_seed} | data={args.data} | "
        f"seq_len={args.seq_len} | label_len={args.label_len} | "
        f"pred_len={args.pred_len} | split_mode={args.split_mode} | "
        f"input_mode={args.input_mode} | device={device} ====="
    )

    # ============================================================
    # TREINAMENTO — mesma ordem e mesma seleção de checkpoints.
    # ============================================================
    best_first = train_one_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_ds=val_ds,
        args=args,
        device=device,
        stage="first stage",
    )
    model.load_state_dict(best_first)

    best_second = train_one_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_ds=val_ds,
        args=args,
        device=device,
        stage="second stage",
    )
    model.load_state_dict(best_second)

    # Stage 2 é avaliada primeiro para preservar o fluxo histórico do código.
    stage2_metrics = test_model(
        model=model,
        test_loader=test_loader,
        test_ds=test_ds,
        args=args,
        device=device,
        stage="second stage",
        diagnostics=getattr(args, "smape_diagnostics", False),
    )

    print(
        f"Test | MSE: {stage2_metrics.mse:.6f} | "
        f"MAE: {stage2_metrics.mae:.6f} | "
        f"sMAPE: {stage2_metrics.smape:.2f}%"
    )

    # A Stage 1 é avaliada somente após toda a execução antiga da Stage 2.
    # Isso evita alterar treinamento, checkpoint ou métricas anteriores.
    stage1_metrics = test_model(
        model=model,
        test_loader=test_loader,
        test_ds=test_ds,
        args=args,
        device=device,
        stage="first stage",
        diagnostics=False,
    )

    print(
        f"Stage 1 Test | MSE: {stage1_metrics.mse:.6f} | "
        f"MAE: {stage1_metrics.mae:.6f} | "
        f"sMAPE: {stage1_metrics.smape:.2f}%"
    )

    return Metrics(
        mse=stage2_metrics.mse,
        mae=stage2_metrics.mae,
        smape=stage2_metrics.smape,
        stage1_mse=stage1_metrics.mse,
        stage1_mae=stage1_metrics.mae,
        stage1_smape=stage1_metrics.smape,
    )
