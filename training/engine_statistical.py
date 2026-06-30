from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from data_loader.preprocess_ett import preprocess_ett_dataset

from models.statistical.arima import (
    ARIMAConfig,
    fit_forecast_arima_auto,
    fit_forecast_arima_statsmodels,
)
from models.statistical.sarima import (
    SARIMAConfig,
    fit_forecast_sarima_auto,
    fit_forecast_sarima_statsmodels,
)


ALLOWED_DATASETS = {"ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange"}


@dataclass
class Metrics:
    mse: float
    mae: float


def ensure_univariate_csv(
    root_path: str | Path,
    data_name: str,
    target_col: str = "OT",
    data_dir: str | Path = "./nixtla_cache",
) -> Path:
    """
    Garante que exista um CSV univariado no formato:
        date, OT

    Exemplos:
        data/ETT/ETTh1.csv
        data/ETT/Weather.csv
        data/ETT/Exchange.csv
    """
    if data_name not in ALLOWED_DATASETS:
        raise ValueError(f"data_name deve ser um destes: {sorted(ALLOWED_DATASETS)}")

    root_path = Path(root_path)
    csv_path = root_path / f"{data_name}.csv"

    if not csv_path.exists():
        print(f"[INFO] {csv_path} não encontrado. Gerando automaticamente...")
        root_path.mkdir(parents=True, exist_ok=True)

        try:
            preprocess_ett_dataset(
                group=data_name,
                data_dir=data_dir,
                out_dir=root_path,
                target_col=target_col,
                multivariate=False,
                normalize=False,
            )
        except TypeError:
            preprocess_ett_dataset(
                group=data_name,
                data_dir=data_dir,
                out_dir=root_path,
            )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado mesmo após preprocessamento: {csv_path}"
        )

    return csv_path


def load_univariate_series(
    root_path: str | Path,
    data_name: str,
    target_col: str = "OT",
    data_dir: str | Path = "./nixtla_cache",
) -> np.ndarray:
    """
    Carrega uma série univariada a partir de um CSV.

    Para os experimentos deste projeto, a coluna alvo esperada é OT.
    Caso o CSV tenha apenas uma coluna numérica além de date, ela é usada.
    """
    csv_path = ensure_univariate_csv(
        root_path=root_path,
        data_name=data_name,
        target_col=target_col,
        data_dir=data_dir,
    )

    df = pd.read_csv(csv_path)

    if "date" in df.columns:
        value_cols = [c for c in df.columns if c != "date"]
    else:
        value_cols = list(df.columns)

    if target_col not in df.columns:
        if len(value_cols) == 1:
            print(
                f"[WARN] Coluna alvo '{target_col}' não encontrada em {csv_path.name}. "
                f"Usando '{value_cols[0]}' como alvo."
            )
            target_col = value_cols[0]
        else:
            raise ValueError(
                f"{csv_path.name} precisa ter a coluna alvo '{target_col}'. "
                f"Colunas disponíveis: {list(df.columns)}"
            )

    y = pd.to_numeric(df[target_col], errors="coerce")
    y = y.replace([np.inf, -np.inf, -9999, -9999.0], np.nan)
    y = y.interpolate(limit_direction="both").dropna()

    if len(y) == 0:
        raise ValueError(f"A série '{target_col}' em {csv_path.name} ficou vazia.")

    return y.to_numpy(dtype=np.float64)


def seasonal_period_for_group(data_name: str) -> int:
    """
    Define o período sazonal usado no SARIMA.

    ETTh:
        frequência horária -> ciclo diário = 24

    ETTm:
        frequência de 15 minutos -> ciclo diário = 96

    Weather:
        frequência de 10 minutos -> ciclo diário = 144

    Exchange:
        frequência diária -> ciclo semanal = 7
    """
    if data_name in {"ETTh1", "ETTh2"}:
        return 24

    if data_name in {"ETTm1", "ETTm2"}:
        return 96

    if data_name == "Weather":
        return 144

    if data_name == "Exchange":
        return 7

    raise ValueError(f"Dataset não suportado: {data_name}")


def train_val_test_split_time_1d(
    y: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide a série temporal em treino, validação e teste sem embaralhar.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio deve estar entre 0 e 1.")

    if not (0 <= val_ratio < 1):
        raise ValueError("val_ratio deve estar entre 0 e 1.")

    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio deve ser menor que 1.")

    n = len(y)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = y[:train_end]
    val = y[train_end:val_end]
    test = y[val_end:]

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(
            f"Split inválido: train={len(train)}, val={len(val)}, test={len(test)}"
        )

    return train, val, test


def mse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))

    return float(mse), float(mae)


def rolling_origin_eval(
    y: np.ndarray,
    horizon: int,
    model_kind: str,
    m_season: int,
    stride: Optional[int] = None,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    max_origins: Optional[int] = None,
    progress_every: int = 2,
    sarima_light: bool = True,
    use_auto_arima: bool = False,
    progress_prefix: str = "",
) -> Metrics:
    """
    Avaliação rolling-origin apenas no conjunto de teste.

    A janela histórica é expansiva:
        treino + validação + parte já observada do teste
    """
    train, val, test = train_val_test_split_time_1d(
        y,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    hist_base = np.concatenate([train, val])

    if stride is None:
        stride = horizon

    n_test = len(test)
    last_origin = n_test - horizon

    if last_origin < 0:
        raise ValueError(
            f"Test set muito curto para horizon={horizon}. test_len={n_test}"
        )

    origins = list(range(0, last_origin + 1, stride))

    if max_origins is not None:
        origins = origins[:max_origins]

    if len(origins) == 0:
        raise ValueError(
            f"Nenhuma origem válida para horizon={horizon}, test_len={n_test}."
        )

    preds_all: List[np.ndarray] = []
    trues_all: List[np.ndarray] = []

    if model_kind == "ARIMA":
        cfg = ARIMAConfig(
            order=(1, 1, 1),
            seasonal_order=None,
            use_auto_arima=use_auto_arima,
        )

    elif model_kind == "SARIMA":
        if sarima_light:
            seasonal_order = (1, 0, 1, m_season)
        else:
            seasonal_order = (1, 1, 1, m_season)

        cfg = SARIMAConfig(
            order=(1, 1, 1),
            seasonal_order=seasonal_order,
            use_auto_arima=use_auto_arima,
        )

    else:
        raise ValueError(f"Unknown model_kind={model_kind}")

    for idx, origin in enumerate(origins, 1):
        y_hist = np.concatenate([hist_base, test[:origin]])
        y_true = test[origin:origin + horizon].astype(np.float64)

        if model_kind == "ARIMA":
            if cfg.use_auto_arima:
                y_pred = fit_forecast_arima_auto(y_hist, horizon)
            else:
                y_pred = fit_forecast_arima_statsmodels(y_hist, horizon, cfg)

        else:
            if cfg.use_auto_arima:
                y_pred = fit_forecast_sarima_auto(y_hist, horizon, m=m_season)
            else:
                y_pred = fit_forecast_sarima_statsmodels(y_hist, horizon, cfg)

        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

        if y_pred.shape[0] < horizon:
            pad_value = y_pred[-1] if len(y_pred) > 0 else y_hist[-1]
            y_pred = np.pad(
                y_pred,
                pad_width=(0, horizon - y_pred.shape[0]),
                mode="constant",
                constant_values=pad_value,
            )

        if y_pred.shape[0] > horizon:
            y_pred = y_pred[:horizon]

        preds_all.append(y_pred)
        trues_all.append(y_true)

        if progress_every and (
            idx == 1 or idx % progress_every == 0 or idx == len(origins)
        ):
            print(f"{progress_prefix}{model_kind} origin {idx}/{len(origins)} done")

    y_pred_all = np.concatenate(preds_all, axis=0)
    y_true_all = np.concatenate(trues_all, axis=0)

    mse, mae = mse_mae(y_true_all, y_pred_all)
    return Metrics(mse=mse, mae=mae)


def run_single_experiment(args, model_kind: str) -> Metrics:
    y = load_univariate_series(
        root_path=args.root_path,
        data_name=args.data,
        target_col=args.target,
        data_dir=getattr(args, "data_dir", "./nixtla_cache"),
    )

    m = seasonal_period_for_group(args.data)

    stride = None if args.stride_mode == "H" else 1

    print(
        f"\n--- {model_kind} | data={args.data} | "
        f"horizon={args.horizon} | seasonal_m={m} | "
        f"stride={('H' if stride is None else stride)} | "
        f"max_origins={args.max_origins} ---"
    )

    metrics = rolling_origin_eval(
        y=y,
        horizon=args.horizon,
        model_kind=model_kind,
        m_season=m,
        stride=stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_origins=args.max_origins,
        progress_every=args.progress_every,
        sarima_light=args.sarima_light,
        use_auto_arima=args.use_auto_arima,
        progress_prefix="  ",
    )

    print(f"{model_kind}: MSE={metrics.mse:.6f} | MAE={metrics.mae:.6f}")
    return metrics


def run_benchmark(args) -> pd.DataFrame:
    datasets = (
        [args.data]
        if not args.run_all
        else ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange"]
    )

    horizons = [args.horizon] if not args.run_all_horizons else [96, 192, 336, 720]

    rows = []

    for ds in datasets:
        args.data = ds
        m = seasonal_period_for_group(ds)

        for horizon in horizons:
            args.horizon = horizon
            print(f"\n==================== DATASET: {ds} | H={horizon} ====================")

            for model_kind in args.models:
                metrics = run_single_experiment(args, model_kind=model_kind)

                rows.append(
                    {
                        "Dataset": ds,
                        "Horizon": horizon,
                        "Model": model_kind,
                        "MSE": metrics.mse,
                        "MAE": metrics.mae,
                        "seasonal_m": m,
                        "stride": args.stride_mode,
                        "max_origins": args.max_origins,
                        "auto_arima": args.use_auto_arima,
                        "sarima_light": args.sarima_light,
                    }
                )

    df_res = (
        pd.DataFrame(rows)
        .sort_values(["Dataset", "Horizon", "Model"])
        .reset_index(drop=True)
    )

    return df_res
