"""GBT por serie da M3 Competition: holdout oficial, Stage 1/2 e categorias.

Colocar em scripts/run_m3_gbt.py e executar da raiz do LongHorinzon:

    python -m scripts.run_m3_gbt \
        --data-root ./data/M3 --group monthly --limit 5 \
        --output ./results/m3_monthly_gbt_test.csv \
        --lookback 12 --train-epochs 2 --itr 1

Reutiliza modelos.gbt.model.GBTVanillaStandalone do repositorio existente.
Protocolo:
  1. O historico oficial (Exchange.csv) nao inclui o teste.
  2. Ultimos H meses desse historico sao usados apenas na validacao interna.
  3. Seleciona separadamente epocas da Stage 1 e Stage 2 por MSE normalizado
     de validacao (uma origem temporal por serie e semente).
  4. Reinicializa o GBT e ajusta Stage 1 e Stage 2 com TODO o historico
     oficial, pelo numero de epocas selecionado, sem observar o teste.
  5. Preve os H meses de official_test.csv somente no final.
  6. MSE/MAE/sMAPE/MASE sao medidos na escala ORIGINAL da serie.

O CSV principal tem model='GBT' (Stage 2), com metricas da Stage 1
em colunas stage1_*. Um CSV independente GBT_STAGE1 tambem e gerado.
Datas artificiais da M3 nao sao usadas como recursos temporais.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.gbt.model import GBTVanillaStandalone
from scripts.run_m3_statistical import metrics, read_categories
from utils.seed import set_seed

METRICS = ("MSE_original", "MAE_original", "sMAPE_pct", "MASE")


class ForecastWindows(Dataset):
    """Janelas nao sobrepostas ao holdout: x=[L,1], y=[H,1]."""

    def __init__(self, values: np.ndarray, lookback: int, horizon: int):
        data = np.asarray(values, dtype=np.float32).reshape(-1)
        n = len(data) - lookback - horizon + 1
        if n <= 0:
            raise ValueError(
                f"Historico insuficiente para janelas: n={len(data)}, "
                f"L={lookback}, H={horizon}"
            )
        if not np.isfinite(data).all():
            raise ValueError("Serie normalizada tem valores nao finitos")
        self.data = torch.from_numpy(data)
        self.lookback = lookback
        self.horizon = horizon
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        l, h = self.lookback, self.horizon
        x = self.data[i:i + l].unsqueeze(-1)
        y = self.data[i + l:i + l + h].unsqueeze(-1)
        return x, y


def standardize(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values, dtype=np.float64))
    std = float(np.std(values, dtype=np.float64, ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError("Media ou desvio padrao nao finito")
    if std <= 1e-12:
        std = 1.0
    return mean, std


def make_model(args, horizon: int, device: torch.device) -> GBTVanillaStandalone:
    # A instancia preserva a arquitetura original: piramides na Stage 1
    # e atencao causal/residuo na Stage 2.
    return GBTVanillaStandalone(
        enc_in=1,
        dec_in=1,
        c_out=1,
        label_len=args.label_len,
        pred_len=horizon,
        fd_model=args.fd_model,
        d_model=args.d_model,
        d_layers=args.d_layers,
        n_heads=args.n_heads,
        s_layers=args.s_layers_list,
        dropout=args.dropout,
        use_time=False,        # as datas da M3 foram sintetizadas na exportacao
        with_minute=False,
        target_idx=0,
        input_mode="univariate",
    ).to(device)


def forward_stage(model: nn.Module, x: torch.Tensor, horizon: int, stage: str):
    # x: [B,L,1]. Os argumentos de decoder sao exigidos pela assinatura do
    # GBT; a implementacao atual usa a previsao base e nao consulta x_dec.
    b, l, _ = x.shape
    mark_enc = torch.empty((b, l, 0), dtype=torch.long, device=x.device)
    mark_dec = torch.empty((b, horizon, 0), dtype=torch.long, device=x.device)
    dec_in = torch.zeros((b, horizon, 1), dtype=x.dtype, device=x.device)
    return model(
        x_enc=x,
        x_mark_enc=mark_enc,
        x_dec=dec_in,
        x_mark_dec=mark_dec,
        stage=stage,
    )


def make_loader(values, args, horizon: int):
    ds = ForecastWindows(values, args.lookback, horizon)
    loader = DataLoader(
        ds,
        batch_size=min(args.batch_size, len(ds)),
        shuffle=True,
        drop_last=False,   # fundamental para series curtas da M3
        num_workers=0,
    )
    return loader, len(ds)


@torch.no_grad()
def predict_z(model, context_z, horizon: int, stage: str, device):
    model.eval()
    x = torch.as_tensor(
        np.asarray(context_z, dtype=np.float32).copy(),
        dtype=torch.float32,
        device=device,
    ).view(1, -1, 1)
    pred = forward_stage(model, x, horizon, stage)
    out = pred.detach().cpu().numpy().reshape(-1).astype(np.float64)
    if out.shape != (horizon,) or not np.isfinite(out).all():
        raise FloatingPointError(f"Previsao invalida ({stage})")
    return out


def run_epoch(model, loader, optimizer, horizon: int, stage: str, device):
    model.train()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = forward_stage(model, x, horizon, stage)
        loss = nn.functional.mse_loss(pred, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Loss nao finita em {stage}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        losses.append(float(loss.item()))
    if not losses:
        raise RuntimeError("Nenhum batch de treinamento")
    return float(np.mean(losses))


def select_stage_epochs(
    model, loader, val_context_z, val_true_z, horizon, stage, args, device
):
    model.set_stage_trainable(stage)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError(f"Sem parametros treinaveis em {stage}")
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)

    best_mse = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0

    for epoch in range(1, args.train_epochs + 1):
        run_epoch(model, loader, optimizer, horizon, stage, device)
        pred = predict_z(model, val_context_z, horizon, stage, device)
        val_mse = float(np.mean((val_true_z - pred) ** 2))
        if not np.isfinite(val_mse):
            raise FloatingPointError(f"MSE validacao nao finito ({stage})")
        if val_mse < best_mse - args.min_delta:
            best_mse = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError(f"Nao foi possivel selecionar epoca ({stage})")
    # Stage 2 deve comecar do melhor checkpoint selecionado da Stage 1.
    model.load_state_dict(best_state)
    return best_epoch, best_mse


def train_fixed_epochs(model, loader, horizon, stage, epochs, args, device):
    model.set_stage_trainable(stage)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    for _ in range(epochs):
        run_epoch(model, loader, optimizer, horizon, stage, device)


def single_run(history, truth, horizon, seasonality, args, device, seed: int):
    """Seleciona epocas nas observacoes de treino; refaz treinamento completo."""
    set_seed(seed)
    core = history[:-horizon]
    val_truth = history[-horizon:]
    if len(core) < args.lookback + horizon:
        raise ValueError(
            f"Historico curto: n={len(history)}, minimo={args.lookback + 2*horizon} "
            "para validacao temporal. Reduza --lookback."
        )

    mu, sd = standardize(core)
    core_z = (core - mu) / sd
    val_z = (val_truth - mu) / sd
    val_context_z = core_z[-args.lookback:]
    val_loader, n_val_windows = make_loader(core_z, args, horizon)

    selection_model = make_model(args, horizon, device)
    e1, v1 = select_stage_epochs(
        selection_model, val_loader, val_context_z, val_z,
        horizon, "first stage", args, device,
    )
    e2, v2 = select_stage_epochs(
        selection_model, val_loader, val_context_z, val_z,
        horizon, "second stage", args, device,
    )
    del selection_model, val_loader

    # Refit usando o historico oficial completo, que inclui a validacao.
    set_seed(seed)
    full_mu, full_sd = standardize(history)
    full_z = (history - full_mu) / full_sd
    full_loader, n_full_windows = make_loader(full_z, args, horizon)
    model = make_model(args, horizon, device)
    train_fixed_epochs(
        model, full_loader, horizon, "first stage", e1, args, device
    )
    train_fixed_epochs(
        model, full_loader, horizon, "second stage", e2, args, device
    )

    context_z = full_z[-args.lookback:]
    stage1_pred = predict_z(model, context_z, horizon, "first stage", device)
    stage2_pred = predict_z(model, context_z, horizon, "second stage", device)
    stage1_original = stage1_pred * full_sd + full_mu
    stage2_original = stage2_pred * full_sd + full_mu

    # A verdade do teste oficial somente e utilizada aqui, apos o treinamento.
    s1 = metrics(truth, stage1_original, history, seasonality)
    s2 = metrics(truth, stage2_original, history, seasonality)
    del model, full_loader

    return {
        "stage1": s1,
        "stage2": s2,
        "stage1_selected_epochs": e1,
        "stage2_selected_epochs": e2,
        "stage1_validation_MSE_normalized": v1,
        "stage2_validation_MSE_normalized": v2,
        "train_windows_validation": n_val_windows,
        "train_windows_refit": n_full_windows,
    }


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    df.to_csv(temp, index=False)
    os.replace(temp, path)


def build_stage1_df(main_df: pd.DataFrame) -> pd.DataFrame:
    df = main_df.copy()
    df["model"] = "GBT_STAGE1"
    for name in METRICS:
        df[name] = df["stage1_" + name]
        df[name + "_std"] = df["stage1_" + name + "_std"]
    return df


def summarize(df: pd.DataFrame, output: Path) -> None:
    stage1 = build_stage1_df(df)
    stage1_path = output.with_name(output.stem + "_stage1.csv")
    atomic_csv(stage1, stage1_path)
    both = pd.concat([df, stage1], ignore_index=True)

    print("\n===== METRICAS GLOBAIS GBT (STAGE 1 E STAGE 2) =====")
    print(both.groupby("model")[list(METRICS)].mean().round(6).to_string())
    summary = (
        both.groupby(["category", "model"], as_index=False)
        .agg(
            n_series=("series_id", "nunique"),
            MSE_mean=("MSE_original", "mean"),
            MAE_mean=("MAE_original", "mean"),
            sMAPE_mean=("sMAPE_pct", "mean"),
            sMAPE_median=("sMAPE_pct", "median"),
            MASE_mean=("MASE", "mean"),
        )
        .sort_values(["category", "model"])
    )
    out_summary = output.with_name(output.stem + "_summary_by_category.csv")
    atomic_csv(summary, out_summary)
    print("\n===== METRICAS POR CATEGORIA =====")
    print(summary.round(4).to_string(index=False))
    print(f"\nResultados GBT Stage 2: {output}")
    print(f"Resultados GBT Stage 1: {stage1_path}")
    print(f"Resumo por categoria: {out_summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data/M3")
    parser.add_argument(
        "--group", choices=["yearly", "quarterly", "monthly", "other"],
        default="monthly",
    )
    parser.add_argument("--limit", type=int, default=5, help="0 = todas as series")
    parser.add_argument("--output", default="./results/m3_monthly_gbt.csv")
    parser.add_argument("--categories-file", default=None)
    parser.add_argument("--lookback", "--seq-len", type=int, default=12)
    parser.add_argument(
        "--label-len", type=int, default=None,
        help="Contexto da Stage 1; default = lookback",
    )
    parser.add_argument("--train-epochs", type=int, default=10, help="Maximo por stage")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--fd-model", "--fd_model", type=int, default=32)
    parser.add_argument("--d-model", "--d_model", type=int, default=128)
    parser.add_argument("--n-heads", "--n_heads", type=int, default=4)
    parser.add_argument("--s-layers", "--s_layers", default="3,2,1")
    parser.add_argument("--d-layers", "--d_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=3, help="0 = desativa")
    parser.add_argument("--min-delta", type=float, default=1e-8)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.label_len = args.lookback if args.label_len is None else args.label_len
    try:
        args.s_layers_list = [int(x.strip()) for x in args.s_layers.split(",") if x.strip()]
    except ValueError:
        parser.error("--s-layers deve conter inteiros separados por virgula")

    if (
        args.limit < 0 or args.lookback < 1 or not (1 <= args.label_len <= args.lookback)
        or args.train_epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0
        or args.fd_model < 2 or args.d_model < 2
        or args.fd_model % 2 != 0 or args.d_model % 2 != 0
        or args.n_heads < 1 or args.d_model % args.n_heads != 0
        or args.d_layers < 1 or not args.s_layers_list
        or any(n < 1 for n in args.s_layers_list)
        or args.label_len // (2 ** (len(args.s_layers_list) - 1)) < 1
        or not 0 <= args.dropout < 1 or args.patience < 0
        or args.itr < 1 or args.threads < 1 or args.min_delta < 0
    ):
        parser.error("Parametros invalidos; confira dimensoes, label_len e s_layers")

    torch.set_num_threads(args.threads)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA nao disponivel")

    root = Path(args.data_root) / args.group
    folders = (
        sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)
        if root.is_dir() else []
    )
    if args.limit:
        folders = folders[:args.limit]
    if not folders:
        raise SystemExit("Nenhuma serie encontrada; rode scripts.prepare_m3 primeiro")

    cat_path = (
        Path(args.categories_file) if args.categories_file
        else Path(args.data_root) / f"m3_{args.group}_categories.csv"
    )
    categories = read_categories(
        cat_path, args.group, [p.name.upper() for p in folders]
    )

    output = Path(args.output)
    cfg_file = output.with_name(output.name + ".config.json")
    config = {
        "model": "GBT", "group": args.group,
        "data_root": str(Path(args.data_root).resolve()),
        "categories_file": str(cat_path.resolve()),
        "lookback": args.lookback, "label_len": args.label_len,
        "train_epochs": args.train_epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "fd_model": args.fd_model, "d_model": args.d_model,
        "n_heads": args.n_heads, "s_layers": args.s_layers_list,
        "d_layers": args.d_layers, "dropout": args.dropout,
        "patience": args.patience, "min_delta": args.min_delta,
        "itr": args.itr, "seed": args.seed,
        "use_time": False, "input_mode": "univariate",
        "refit_full_history": True, "selection_metric": "val_MSE_normalized",
    }

    if args.resume:
        if not output.is_file() or not cfg_file.is_file():
            raise SystemExit("--resume requer CSV e arquivo .config.json existentes")
        previous_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        if previous_cfg != config:
            raise SystemExit("--resume: parametros diferem da execucao anterior")
        previous = pd.read_csv(output)
        if (
            previous.duplicated(["series_id", "model"]).any()
            or not previous["model"].eq("GBT").all()
        ):
            raise SystemExit("--resume: arquivo de resultados invalido")
        rows = previous.to_dict(orient="records")
        done = set(previous["series_id"].astype(str).str.upper())
        print(f"[RESUME] {len(done)} series ja concluidas")
    else:
        rows, done = [], set()
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        cfg_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    errors = []
    print(
        f"[INFO] device={device} | total={len(folders)} | "
        f"L={args.lookback} | label_len={args.label_len} | "
        f"H=metadados | itr={args.itr} | stages=2", flush=True
    )

    for i, folder in enumerate(folders, start=1):
        if folder.name.upper() in done:
            continue
        try:
            meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            history = pd.read_csv(folder / "Exchange.csv")["OT"].to_numpy(dtype=np.float64)
            truth = pd.read_csv(folder / "official_test.csv")["OT"].to_numpy(dtype=np.float64)
            h = int(meta["horizon"])
            m = int(meta["seasonality"])
            if (
                h != len(truth) or m < 1 or h < 1
                or len(history) < args.lookback + 2 * h
                or not np.isfinite(history).all() or not np.isfinite(truth).all()
            ):
                raise ValueError("Serie invalida, historico insuficiente ou horizonte inconsistente")

            runs = [
                single_run(history, truth, h, m, args, device, args.seed + j)
                for j in range(args.itr)
            ]
            row = {
                "series_id": folder.name,
                "group": args.group,
                "model": "GBT",     # representa a STAGE 2
                "horizon": h,
                "train_length": len(history),
                **categories[folder.name.upper()],
                "lookback": args.lookback,
                "label_len": args.label_len,
                "train_epochs_max": args.train_epochs,
                "itr": args.itr,
                "seed": args.seed,
                "refit_full_history": True,
                "stage1_selected_epochs_mean": float(np.mean([
                    r["stage1_selected_epochs"] for r in runs
                ])),
                "stage2_selected_epochs_mean": float(np.mean([
                    r["stage2_selected_epochs"] for r in runs
                ])),
                "stage1_validation_MSE_normalized_mean": float(np.mean([
                    r["stage1_validation_MSE_normalized"] for r in runs
                ])),
                "stage2_validation_MSE_normalized_mean": float(np.mean([
                    r["stage2_validation_MSE_normalized"] for r in runs
                ])),
                "train_windows_validation": runs[0]["train_windows_validation"],
                "train_windows_refit": runs[0]["train_windows_refit"],
            }
            for metric in METRICS:
                for stage_key, prefix in [("stage2", ""), ("stage1", "stage1_")]:
                    values = [r[stage_key][metric] for r in runs]
                    row[prefix + metric] = float(np.mean(values))
                    row[prefix + metric + "_std"] = (
                        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                    )

            rows.append(row)
            atomic_csv(pd.DataFrame(rows), output)
            done.add(folder.name.upper())
            print(
                f"[{i}/{len(folders)}] [OK] {folder.name} | {row['category']} "
                f"| GBT | H={h} | Stage1 sMAPE={row['stage1_sMAPE_pct']:.2f}% "
                f"| Stage2 sMAPE={row['sMAPE_pct']:.2f}% "
                f"| epochs={row['stage1_selected_epochs_mean']:.1f}/"
                f"{row['stage2_selected_epochs_mean']:.1f}", flush=True
            )
        except Exception as exc:
            errors.append({"series_id": folder.name,
                           "error": f"{type(exc).__name__}: {exc}"})
            warnings.warn(f"[ERRO] {folder.name}: {exc}")
            atomic_csv(pd.DataFrame(errors), output.with_name(output.stem + "_errors.csv"))

    if not rows:
        raise RuntimeError("Nenhum resultado; confira dados e arquivo de erros")
    df = pd.DataFrame(rows)
    if df.duplicated(["series_id", "model"]).any():
        raise RuntimeError("Resultados duplicados por serie/modelo")
    summarize(df, output)
    print(f"Series concluidas: {len(df)} | Erros nesta execucao: {len(errors)}")


if __name__ == "__main__":
    main()
