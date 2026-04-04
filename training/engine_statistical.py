from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from datasets.ett_univariate import (
    load_univariate_series,
    seasonal_period_for_group,
    train_val_test_split_time_1d,
)
from models.statistical.arima import (
    ARIMAConfig,
    HAS_PMDARIMA as HAS_PMDARIMA_ARIMA,
    fit_forecast_arima_auto,
    fit_forecast_arima_statsmodels,
)
from models.statistical.sarima import (
    SARIMAConfig,
    HAS_PMDARIMA as HAS_PMDARIMA_SARIMA,
    fit_forecast_sarima_auto,
    fit_forecast_sarima_statsmodels,
)


@dataclass
class Metrics:
    mse: float
    mae: float


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
    Rolling origin evaluation on TEST only (expanding window).
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
        raise ValueError(f"Test set too short for horizon={horizon}. test_len={n_test}")

    origins = list(range(0, last_origin + 1, stride))
    if max_origins is not None:
        origins = origins[:max_origins]

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

        else:  # SARIMA
            if cfg.use_auto_arima:
                y_pred = fit_forecast_sarima_auto(y_hist, horizon, m=m_season)
            else:
                y_pred = fit_forecast_sarima_statsmodels(y_hist, horizon, cfg)

        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        if y_pred.shape[0] != horizon:
            y_pred = y_pred[:horizon]

        preds_all.append(y_pred)
        trues_all.append(y_true)

        if progress_every and (idx == 1 or idx % progress_every == 0 or idx == len(origins)):
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
    datasets = [args.data] if not args.run_all else ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
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

    df_res = pd.DataFrame(rows).sort_values(["Dataset", "Horizon", "Model"]).reset_index(drop=True)
    return df_res