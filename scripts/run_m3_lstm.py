"""LSTM por serie da M3 Competition com holdout oficial e categorias.

Uso (na raiz do LongHorinzon):
    python -m scripts.run_m3_lstm \
      --data-root ./data/M3 --group monthly --limit 1428 \
      --output ./results/m3_monthly_lstm_1428.csv

Reutiliza a arquitetura do repositorio (models.lstm.model.LSTMForecaster)
e as metricas/metadados de scripts.run_m3_statistical.

Protocolo:
 - para cada serie, o arquivo Exchange.csv contem SOMENTE o historico oficial;
 - a ultima janela de H meses desse historico e reservada para validacao;
 - melhor numero de epocas escolhido pelo MSE normalizado nessa validacao;
 - o modelo e reinicializado e treinado com TODO o historico oficial;
 - official_test.csv e utilizado apenas ao final, para calcular as metricas.

Nota: modelo treinado separadamente em cada serie. Em series curtas a validacao
pode ter apenas uma janela de treino, e isso deve ser interpretado com cuidado.
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

from models.lstm.model import LSTMForecaster
from scripts.run_m3_statistical import metrics, read_categories
from utils.seed import set_seed

METRICS = ("MSE_original", "MAE_original", "sMAPE_pct", "MASE")


class ForecastWindows(Dataset):
    """Janelas supervisionadas de [L,1] para [H,1] sem vazamento temporal."""

    def __init__(self, values: np.ndarray, lookback: int, horizon: int):
        data = np.asarray(values, dtype=np.float32).reshape(-1)
        self.length = len(data) - lookback - horizon + 1
        if self.length <= 0:
            raise ValueError(
                f"Historico insuficiente: n={len(data)}, L={lookback}, H={horizon}"
            )
        self.values = torch.from_numpy(data)
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        l, h = self.lookback, self.horizon
        x = self.values[idx : idx + l].unsqueeze(-1)
        y = self.values[idx + l : idx + l + h].unsqueeze(-1)
        return x, y


def scaler_from_train(values: np.ndarray) -> tuple[float, float]:
    """Transformacao identica ao StandardScaler para dados univariados."""
    mean = float(np.mean(values, dtype=np.float64))
    sd = float(np.std(values, dtype=np.float64, ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        sd = 1.0
    return mean, sd


def make_model(args, horizon: int, device: torch.device) -> nn.Module:
    return LSTMForecaster(
        input_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        horizon=horizon,
        output_size=1,
        dropout=args.dropout,
    ).to(device)


def fit_epochs(
    model: nn.Module,
    normalized: np.ndarray,
    lookback: int,
    horizon: int,
    n_epochs: int,
    args,
    device: torch.device,
) -> None:
    dataset = ForecastWindows(normalized, lookback, horizon)
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        drop_last=False,  # series M3 curtas podem ter menos que um batch
        num_workers=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    for _ in range(n_epochs):
        model.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            loss = criterion(prediction, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Perda nao finita durante o treinamento")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


@torch.no_grad()
def predict_normalized(
    model: nn.Module, context: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    x = torch.as_tensor(context, dtype=torch.float32, device=device)
    output = model(x.reshape(1, -1, 1))
    return output.reshape(-1).cpu().numpy().astype(np.float64)


def single_run(
    history: np.ndarray,
    truth: np.ndarray,
    horizon: int,
    seasonality: int,
    args,
    device: torch.device,
    seed: int,
) -> dict:
    """Seleciona epocas em validacao temporal e refaz treino no historico integral."""
    set_seed(seed)

    l = args.lookback
    # Validacao e ultimo H do historico, sem jamais acessar official_test.csv.
    train_core = history[:-horizon]
    val_truth = history[-horizon:]
    if len(train_core) < l + horizon:
        raise ValueError(
            f"Historico insuficiente para validacao de H={horizon}, L={l}: "
            f"n={len(history)}; minimo={l + 2 * horizon}. "
            "Use --lookback menor ou interprete como serie muito curta."
        )

    mu, sd = scaler_from_train(train_core)
    core_z = (train_core - mu) / sd
    val_z = (val_truth - mu) / sd
    val_context_z = (train_core[-l:] - mu) / sd

    model = make_model(args, horizon, device)
    dataset = ForecastWindows(core_z, l, horizon)
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    best_epoch, best_mse = 0, float("inf")
    no_improvement = 0

    for epoch in range(1, args.train_epochs + 1):
        model.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Perda nao finita na selecao de epocas")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_prediction = predict_normalized(model, val_context_z, device)
        if len(val_prediction) != horizon or not np.isfinite(val_prediction).all():
            raise FloatingPointError("Previsao de validacao invalida")
        val_mse = float(np.mean((val_z - val_prediction) ** 2))
        if val_mse < best_mse - args.min_delta:
            best_mse = val_mse
            best_epoch = epoch
            no_improvement = 0
        else:
            no_improvement += 1
            if args.patience and no_improvement >= args.patience:
                break

    if best_epoch == 0:
        raise RuntimeError("Nenhuma epoca selecionada")

    # Refit em todo o historico oficial, incluindo o trecho de validacao.
    # A decisao de hiperparametros ja foi tomada sem usar o teste oficial.
    set_seed(seed)
    final_mu, final_sd = scaler_from_train(history)
    full_z = (history - final_mu) / final_sd
    final_model = make_model(args, horizon, device)
    fit_epochs(final_model, full_z, l, horizon, best_epoch, args, device)

    prediction_z = predict_normalized(final_model, full_z[-l:], device)
    prediction = prediction_z * final_sd + final_mu
    if len(prediction) != len(truth) or not np.isfinite(prediction).all():
        raise FloatingPointError("Previsao final invalida")

    values = metrics(truth, prediction, history, seasonality)
    values.update(
        selected_epochs=best_epoch,
        validation_MSE_normalized=best_mse,
        train_windows_validation=len(dataset),
        train_windows_refit=len(history) - l - horizon + 1,
        mean_train=final_mu,
        std_train=final_sd,
    )
    return values


def summarize(df: pd.DataFrame, output: Path) -> None:
    cols = list(METRICS)
    print("\n===== METRICAS GLOBAIS LSTM =====")
    print(df.groupby("model")[cols].mean().round(6).to_string())
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
    destination = output.with_name(output.stem + "_summary_by_category.csv")
    summary.to_csv(destination, index=False)
    print("\n===== METRICAS POR CATEGORIA =====")
    print(summary.round(4).to_string(index=False))
    print(f"Resumo: {destination}")


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    df.to_csv(temp, index=False)
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data/M3")
    parser.add_argument("--group", choices=["yearly", "quarterly", "monthly", "other"], default="monthly")
    parser.add_argument("--limit", type=int, default=5, help="0 = todas as series")
    parser.add_argument("--output", default="./results/m3_monthly_lstm.csv")
    parser.add_argument("--categories-file", default=None)
    parser.add_argument("--lookback", type=int, default=12, help="Meses de contexto; M3 mensal: 12")
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=4, help="0 = desativa parada antecipada")
    parser.add_argument("--min-delta", type=float, default=1e-8)
    parser.add_argument("--itr", type=int, default=1, help="Quantidade de sementes por serie")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Continuar a partir do CSV e config existentes")
    args = parser.parse_args()

    if (
        args.limit < 0 or args.lookback < 1 or args.train_epochs < 1
        or args.batch_size < 1 or args.learning_rate <= 0
        or args.hidden_size < 1 or args.num_layers < 1
        or not 0 <= args.dropout < 1 or args.patience < 0
        or args.itr < 1 or args.threads < 1 or args.min_delta < 0
    ):
        parser.error("Parametros de treinamento invalidos")

    torch.set_num_threads(args.threads)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("GPU CUDA nao disponivel; use --device cpu ou auto")

    base = Path(args.data_root) / args.group
    dirs = sorted((d for d in base.iterdir() if d.is_dir()), key=lambda d: d.name) if base.is_dir() else []
    if args.limit:
        dirs = dirs[: args.limit]
    if not dirs:
        raise SystemExit("Nenhuma serie encontrada; execute scripts.prepare_m3 primeiro")

    cat_path = (
        Path(args.categories_file) if args.categories_file
        else Path(args.data_root) / f"m3_{args.group}_categories.csv"
    )
    categories = read_categories(cat_path, args.group, [d.name.upper() for d in dirs])

    output = Path(args.output)
    cfg_file = output.with_name(output.name + ".config.json")
    config = {
        "group": args.group, "data_root": str(Path(args.data_root).resolve()),
        "categories_file": str(cat_path.resolve()),
        "lookback": args.lookback, "train_epochs": args.train_epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "hidden_size": args.hidden_size, "num_layers": args.num_layers,
        "dropout": args.dropout, "patience": args.patience,
        "min_delta": args.min_delta, "itr": args.itr, "seed": args.seed,
        "model": "LSTM", "refit_full_history": True,
    }
    if args.resume and output.exists():
        if not cfg_file.exists():
            raise SystemExit("--resume requer arquivo .config.json associado ao CSV")
        old_config = json.loads(cfg_file.read_text(encoding="utf-8"))
        if old_config != config:
            raise SystemExit("--resume: configuracao diferente da execucao anterior")
        previous = pd.read_csv(output)
        if previous.duplicated(["series_id", "model"]).any():
            raise SystemExit("CSV existente possui pares serie/modelo duplicados")
        rows = previous.to_dict(orient="records")
        done = set(previous["series_id"].astype(str).str.upper())
        print(f"[RESUME] {len(done)} series ja concluidas")
    else:
        rows, done = [], set()
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        cfg_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    errors = []
    print(f"[INFO] device={device} | total={len(dirs)} | L={args.lookback} | itr={args.itr}")

    for i, folder in enumerate(dirs, 1):
        if folder.name.upper() in done:
            continue
        try:
            meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            history = pd.read_csv(folder / "Exchange.csv")["OT"].to_numpy(dtype=np.float64)
            truth = pd.read_csv(folder / "official_test.csv")["OT"].to_numpy(dtype=np.float64)
            h = int(meta["horizon"])
            m = int(meta["seasonality"])
            if len(truth) != h or not np.isfinite(history).all() or not np.isfinite(truth).all():
                raise ValueError("Dados ou horizonte inconsistentes")

            runs = [
                single_run(history, truth, h, m, args, device, args.seed + repeat)
                for repeat in range(args.itr)
            ]
            row = {
                "series_id": folder.name,
                "group": args.group,
                "model": "LSTM",
                "horizon": h,
                "train_length": len(history),
                **categories[folder.name.upper()],
                "lookback": args.lookback,
                "train_epochs_max": args.train_epochs,
                "itr": args.itr,
                "seed": args.seed,
                "refit_full_history": True,
                "selected_epochs_mean": float(np.mean([r["selected_epochs"] for r in runs])),
                "validation_MSE_normalized_mean": float(np.mean([r["validation_MSE_normalized"] for r in runs])),
                "train_windows_validation": runs[0]["train_windows_validation"],
                "train_windows_refit": runs[0]["train_windows_refit"],
            }
            for name in METRICS:
                vals = [r[name] for r in runs]
                row[name] = float(np.mean(vals))
                row[name + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            rows.append(row)
            atomic_csv(pd.DataFrame(rows), output)
            print(
                f"[{i}/{len(dirs)}] [OK] {folder.name} | {row['category']} "
                f"| LSTM | H={h} | sMAPE={row['sMAPE_pct']:.2f}% "
                f"| MAE={row['MAE_original']:.4g} "
                f"| best_epochs={row['selected_epochs_mean']:.1f}"
            )
        except Exception as exc:
            errors.append({"series_id": folder.name, "error": f"{type(exc).__name__}: {exc}"})
            warnings.warn(f"[ERRO] {folder.name}: {exc}")
            atomic_csv(pd.DataFrame(errors), output.with_name(output.stem + "_errors.csv"))

    if not rows:
        raise RuntimeError("Nenhum resultado; confira caminhos e logs de erro")
    df = pd.DataFrame(rows)
    if df.duplicated(["series_id", "model"]).any():
        raise RuntimeError("Pares serie/modelo duplicados")
    summarize(df, output)
    print(f"\nResultados categorizados: {output}")
    print(f"Series concluidas: {len(df)} | Erros nesta execucao: {len(errors)}")


if __name__ == "__main__":
    main()
