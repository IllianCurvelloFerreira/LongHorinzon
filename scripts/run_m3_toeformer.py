"""TOEFormer univariado na M3, com holdout oficial e categorias economicas.

Executar da raiz do repositorio LongHorinzon:
python -m scripts.run_m3_toeformer --data-root ./data/M3 --group monthly \
    --limit 1428 --lookback 12 --train-epochs 10 --itr 3 \
    --output ./results/m3_monthly_toeformer_1428.csv

Reutiliza models.toeformer.model.TOEformer e a loss LRVE do repositorio.
Validacao temporal interna (ultimos H meses do historico), escolha do numero
adequado de epocas por semente; refit em TODO o historico oficial; avaliacao
unica no official_test.csv em unidades originais. Nao usa o teste para treino.

ATENCAO: no TOEFormer atual, quando lookback < horizon, os ultimos
(horizon-lookback) passos da componente sazonal sao preenchidos com zero
pelo proprio modelo. Com lookback=12 e horizonte M3=18, os ultimos 6 passos
sao previstos apenas pelo ramo de tendencia.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.toeformer.model import TOEformer
from models.toeformer.losses import toeformer_total_loss
class ForecastWindows(Dataset):
    """Janelas supervisionadas univariadas [L,1] -> [H,1]."""

    def __init__(self, values: np.ndarray, lookback: int, horizon: int):
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        self.n_windows = len(arr) - lookback - horizon + 1
        if self.n_windows < 1:
            raise ValueError(f"Nao ha janelas: n={len(arr)}, L={lookback}, H={horizon}")
        self.values = torch.from_numpy(arr)
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        l, h = self.lookback, self.horizon
        return (self.values[idx:idx+l].unsqueeze(-1),
                self.values[idx+l:idx+l+h].unsqueeze(-1))


def scaler_from_train(values: np.ndarray) -> tuple[float, float]:
    """Media e desvio populacional do treino (mesmo criterio do StandardScaler)."""
    mean = float(np.mean(values, dtype=np.float64))
    sd = float(np.std(values, dtype=np.float64, ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        sd = 1.0
    return mean, sd

from scripts.run_m3_statistical import metrics, read_categories
from utils.seed import set_seed

METRICS = ("MSE_original", "MAE_original", "sMAPE_pct", "MASE")


def atomic_csv(data: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    data.to_csv(tmp, index=False)
    os.replace(tmp, dest)


def make_model(args, horizon: int, device: torch.device) -> TOEformer:
    return TOEformer(
        c_in=1,
        c_out=1,
        target_idx=0,
        lookback=args.lookback,
        horizon=horizon,
        d_model=args.d_model,
        n_heads=args.n_heads,
        decomp_kernel=args.decomp_kernel,
        k_global=args.k_global,
        k_local=args.k_local,
        dropout=args.dropout,
    ).to(device)


def train_one_epoch(model, loader, optimizer, args, device):
    model.train()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        forecast, season_pred, trend_pred = model(x)
        # Mesma construcao de alvos da training.engine_toeformer do repositorio.
        season_xy, trend_xy = model.decomp(torch.cat([x, y], dim=1))
        h = y.size(1)
        total, _, _ = toeformer_total_loss(
            y_true=y,
            y_hat=forecast,
            y_season_true=season_xy[:, -h:, :],
            y_season_pred=season_pred,
            y_trend_true=trend_xy[:, -h:, :],
            y_trend_pred=trend_pred,
            lam1=args.lam1,
            lam2=args.lam2,
            alpha=args.alpha,
        )
        if not torch.isfinite(total):
            raise FloatingPointError("Loss nao finita no TOEFormer")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(total.item()))
    return float(np.mean(losses))


def windows_loader(series_z, lookback, horizon, batch_size):
    ds = ForecastWindows(series_z, lookback, horizon)
    loader = DataLoader(
        ds,
        batch_size=min(batch_size, len(ds)),
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    return loader, len(ds)


@torch.no_grad()
def predict_z(model: nn.Module, context, device) -> np.ndarray:
    model.eval()
    x = torch.as_tensor(context, dtype=torch.float32, device=device).reshape(1, -1, 1)
    yhat, _, _ = model(x)
    return yhat.reshape(-1).cpu().numpy().astype(np.float64)


def single_run(history, truth, horizon, period, args, device, seed):
    set_seed(seed)
    history = np.asarray(history, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    l = args.lookback
    core = history[:-horizon]
    val = history[-horizon:]
    if len(core) < l + horizon:
        raise ValueError(
            f"Historico curto: n={len(history)}; minimo={l + 2*horizon} "
            f"para validacao H={horizon}, lookback={l}"
        )
    core_mu, core_sd = scaler_from_train(core)
    core_z = (core - core_mu) / core_sd
    val_z = (val - core_mu) / core_sd
    val_context = core_z[-l:]

    model = make_model(args, horizon, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loader, n_val_windows = windows_loader(core_z, l, horizon, args.batch_size)
    best_mse = float("inf")
    best_epoch = 0
    patience_counter = 0

    for ep in range(1, args.train_epochs + 1):
        train_one_epoch(model, loader, optimizer, args, device)
        val_prediction = predict_z(model, val_context, device)
        if len(val_prediction) != horizon or not np.isfinite(val_prediction).all():
            raise FloatingPointError("Previsao de validacao invalida")
        val_mse = float(np.mean((val_z - val_prediction) ** 2))
        if val_mse < best_mse - args.min_delta:
            best_epoch, best_mse = ep, val_mse
            patience_counter = 0
        else:
            patience_counter += 1
            if args.patience and patience_counter >= args.patience:
                break

    if best_epoch < 1:
        raise RuntimeError("Nenhuma epoca valida selecionada")

    # Mesmo protocolo do LSTM M3: refit com todo o historico, sem olhar o teste.
    set_seed(seed)
    mu, sd = scaler_from_train(history)
    history_z = (history - mu) / sd
    final_model = make_model(args, horizon, device)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=args.learning_rate)
    full_loader, n_full_windows = windows_loader(
        history_z, l, horizon, args.batch_size
    )
    for _ in range(best_epoch):
        train_one_epoch(final_model, full_loader, final_optimizer, args, device)

    forecast_z = predict_z(final_model, history_z[-l:], device)
    prediction = forecast_z * sd + mu
    if len(prediction) != len(truth) or not np.isfinite(prediction).all():
        raise FloatingPointError("Previsao final nao finita ou horizonte errado")
    result = metrics(truth, prediction, history, period)
    result.update(
        selected_epochs=best_epoch,
        validation_MSE_normalized=best_mse,
        train_windows_validation=n_val_windows,
        train_windows_refit=n_full_windows,
    )
    return result


def summarize(df: pd.DataFrame, path: Path):
    print("\n===== METRICAS GLOBAIS TOEFORMER =====")
    print(df.groupby("model")[list(METRICS)].mean().round(6).to_string())
    summary = (
        df.groupby(["category", "model"], as_index=False)
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
    target = path.with_name(path.stem + "_summary_by_category.csv")
    atomic_csv(summary, target)
    print("\n===== METRICAS POR CATEGORIA =====")
    print(summary.round(4).to_string(index=False))
    print(f"Resumo categorizado: {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data/M3")
    parser.add_argument("--group", choices=["monthly", "quarterly", "yearly", "other"], default="monthly")
    parser.add_argument("--limit", type=int, default=5, help="0=todas as series")
    parser.add_argument("--output", default="./results/m3_monthly_toeformer.csv")
    parser.add_argument("--categories-file", default=None)
    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--decomp-kernel", type=int, default=25)
    parser.add_argument("--k-global", type=int, default=25)
    parser.add_argument("--k-local", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lam1", type=float, default=1.0)
    parser.add_argument("--lam2", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--patience", type=int, default=4, help="0 desativa early stopping")
    parser.add_argument("--min-delta", type=float, default=1e-8)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if (
        args.limit < 0 or args.lookback < 1 or args.train_epochs < 1
        or args.batch_size < 1 or args.learning_rate <= 0
        or args.d_model < 1 or args.n_heads < 1 or args.d_model % args.n_heads != 0
        or any(k < 1 or k % 2 == 0 for k in [args.decomp_kernel, args.k_global, args.k_local])
        or not 0 <= args.dropout < 1 or args.lam1 < 0 or args.lam2 < 0
        or not 0 < args.alpha < 1 or args.patience < 0
        or args.itr < 1 or args.threads < 1 or args.min_delta < 0
    ):
        parser.error("Parametros invalidos; kernels devem ser positivos e impares")

    torch.set_num_threads(args.threads)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA nao disponivel")

    base = Path(args.data_root) / args.group
    dirs = sorted((p for p in base.iterdir() if p.is_dir()), key=lambda x: x.name) if base.is_dir() else []
    if args.limit:
        dirs = dirs[:args.limit]
    if not dirs:
        raise SystemExit("Nenhuma serie: rode prepare_m3.py primeiro")

    cat_path = (Path(args.categories_file) if args.categories_file
                else Path(args.data_root) / f"m3_{args.group}_categories.csv")
    categories = read_categories(cat_path, args.group, [x.name.upper() for x in dirs])
    out = Path(args.output)
    cfg_file = out.with_name(out.name + ".config.json")
    config = {
        "model": "TOEFormer", "group": args.group,
        "data_root": str(Path(args.data_root).resolve()),
        "categories_file": str(cat_path.resolve()),
        "lookback": args.lookback,
        "train_epochs": args.train_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "d_model": args.d_model, "n_heads": args.n_heads,
        "decomp_kernel": args.decomp_kernel,
        "k_global": args.k_global, "k_local": args.k_local,
        "dropout": args.dropout, "lam1": args.lam1,
        "lam2": args.lam2, "alpha": args.alpha,
        "patience": args.patience, "min_delta": args.min_delta,
        "itr": args.itr, "seed": args.seed,
        "refit_full_history": True,
    }
    if args.resume and out.exists():
        if not cfg_file.is_file():
            raise SystemExit("--resume exige CSV e arquivo .config.json")
        previous_cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        if config != previous_cfg:
            raise SystemExit("--resume: parametros diferem da execucao anterior")
        previous = pd.read_csv(out)
        if previous.duplicated(["series_id", "model"]).any() or (previous["model"] != "TOEFormer").any():
            raise SystemExit("--resume: CSV invalido ou de modelo diferente")
        rows = previous.to_dict(orient="records")
        done = set(previous["series_id"].astype(str).str.upper())
        print(f"[RESUME] {len(done)} series concluidas")
    else:
        rows = []
        done = set()
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        cfg_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    errors = []
    print(f"[INFO] device={device} | total={len(dirs)} | L={args.lookback} | itr={args.itr}")
    for i, folder in enumerate(dirs, start=1):
        if folder.name.upper() in done:
            continue
        try:
            meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            history = pd.read_csv(folder / "Exchange.csv")["OT"].to_numpy(dtype=np.float64)
            truth = pd.read_csv(folder / "official_test.csv")["OT"].to_numpy(dtype=np.float64)
            h = int(meta["horizon"])
            period = int(meta["seasonality"])
            if h != len(truth) or not np.isfinite(history).all() or not np.isfinite(truth).all():
                raise ValueError("Horizonte, historico ou teste invalidos")
            runs = [single_run(history, truth, h, period, args, device, args.seed + j)
                    for j in range(args.itr)]
            row = {
                "series_id": folder.name, "group": args.group,
                "model": "TOEFormer", "horizon": h, "train_length": len(history),
                **categories[folder.name.upper()],
                "lookback": args.lookback, "train_epochs_max": args.train_epochs,
                "itr": args.itr, "seed": args.seed,
                "refit_full_history": True,
                "selected_epochs_mean": float(np.mean([r["selected_epochs"] for r in runs])),
                "validation_MSE_normalized_mean": float(np.mean([r["validation_MSE_normalized"] for r in runs])),
                "train_windows_validation": runs[0]["train_windows_validation"],
                "train_windows_refit": runs[0]["train_windows_refit"],
                "seasonal_tail_shortfall": max(0, h - args.lookback),
            }
            for metric in METRICS:
                nums = [r[metric] for r in runs]
                row[metric] = float(np.mean(nums))
                row[metric + "_std"] = float(np.std(nums, ddof=1)) if len(nums) > 1 else 0.0
            rows.append(row)
            atomic_csv(pd.DataFrame(rows), out)
            print(f"[{i}/{len(dirs)}] [OK] {folder.name} | {row['category']} "
                  f"| TOEFormer | H={h} | sMAPE={row['sMAPE_pct']:.2f}% "
                  f"| MAE={row['MAE_original']:.4g} "
                  f"| best_epochs={row['selected_epochs_mean']:.1f}", flush=True)
        except Exception as exc:
            errors.append({"series_id": folder.name, "error": f"{type(exc).__name__}: {exc}"})
            warnings.warn(f"[ERRO] {folder.name}: {exc}")
            atomic_csv(pd.DataFrame(errors), out.with_name(out.stem + "_errors.csv"))

    if not rows:
        raise RuntimeError("Nenhum resultado; verifique diretorios e logs")
    df = pd.DataFrame(rows)
    if df.duplicated(["series_id", "model"]).any():
        raise RuntimeError("Pares series/modelo duplicados")
    summarize(df, out)
    print(f"\nResultados categorizados: {out}")
    print(f"Series concluidas: {len(df)} | Erros nesta execucao: {len(errors)}")


if __name__ == "__main__":
    main()
