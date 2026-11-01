
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from models.statistical.arima import (
    ARIMAConfig,
    fit_forecast_arima_statsmodels,
)

from models.statistical.sarima import (
    SARIMAConfig,
    fit_forecast_sarima_statsmodels,
)


def metrics(
    y: np.ndarray,
    y_hat: np.ndarray,
    y_train: np.ndarray,
    period: int,
) -> dict:

    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)

    if y.shape != y_hat.shape:
        raise ValueError("Dimensoes incompativeis")

    if not (
        np.isfinite(y).all()
        and np.isfinite(y_hat).all()
    ):
        raise ValueError("Previsao com valores invalidos")

    mae = float(np.mean(np.abs(y - y_hat)))

    m = max(1, period)

    if len(y_train) > m:
        scale = float(
            np.mean(
                np.abs(y_train[m:] - y_train[:-m])
            )
        )
    else:
        scale = float("nan")

    mse = float(np.mean((y - y_hat) ** 2))

    smape = float(
        np.mean(
            200
            * np.abs(y - y_hat)
            / np.maximum(
                np.abs(y) + np.abs(y_hat),
                1e-8,
            )
        )
    )

    mase = (
        mae / scale
        if np.isfinite(scale) and scale > 1e-12
        else float("nan")
    )

    return {
        "MSE_original": mse,
        "MAE_original": mae,
        "sMAPE_pct": smape,
        "MASE": mase,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        default="./data/M3",
    )

    parser.add_argument(
        "--group",
        choices=[
            "yearly",
            "quarterly",
            "monthly",
            "other",
        ],
        default="monthly",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["ARIMA", "SARIMA"],
        default=["ARIMA"],
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output",
        default="./results/m3_statistical_results.csv",
    )

    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit nao pode ser negativo")

    base = Path(args.data_root) / args.group

    subdirs = (
        sorted(d for d in base.iterdir() if d.is_dir())
        if base.exists()
        else []
    )

    if args.limit:
        subdirs = subdirs[:args.limit]

    if not subdirs:
        raise SystemExit(
            "Nenhum dataset encontrado. "
            "Execute prepare_m3.py primeiro."
        )

    rows = []

    for directory in subdirs:

        meta = json.loads(
            (directory / "metadata.json").read_text(
                encoding="utf-8"
            )
        )

        train = pd.read_csv(
            directory / "Exchange.csv"
        )["OT"].to_numpy(dtype=float)

        truth = pd.read_csv(
            directory / "official_test.csv"
        )["OT"].to_numpy(dtype=float)

        h = int(meta["horizon"])
        m = int(meta["seasonality"])

        assert h == len(truth)

        for model in args.models:

            try:
                if model == "ARIMA":

                    pred = fit_forecast_arima_statsmodels(
                        train,
                        h,
                        ARIMAConfig(
                            order=(1, 1, 1)
                        ),
                    )

                else:
                    if m <= 1:
                        print(
                            f"[SKIP] {directory.name}: "
                            "sem sazonalidade definida"
                        )
                        continue

                    pred = fit_forecast_sarima_statsmodels(
                        train,
                        h,
                        SARIMAConfig(
                            order=(1, 1, 1),
                            seasonal_order=(1, 0, 1, m),
                            maxiter=25,
                        ),
                    )

                result = metrics(
                    truth,
                    np.asarray(pred),
                    train,
                    m,
                )

                rows.append({
                    "series_id": directory.name,
                    "group": args.group,
                    "model": model,
                    "horizon": h,
                    "train_length": len(train),
                    **result,
                })

                print(
                    f"[OK] {directory.name} | "
                    f"{model} | "
                    f"sMAPE={result['sMAPE_pct']:.2f}%"
                )

            except Exception as exc:
                warnings.warn(
                    f"[ERRO] {directory.name} "
                    f"{model}: {exc}"
                )

    if not rows:
        raise RuntimeError(
            "Nenhuma previsao calculada"
        )

    df = pd.DataFrame(rows)

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(output, index=False)

    print("\n===== RESULTADOS FINAIS =====")

    print(
        df.groupby("model")[
            ["MAE_original", "sMAPE_pct", "MASE"]
        ].mean().to_string()
    )

    print(f"\nResultados salvos em: {output}")


if __name__ == "__main__":
    main()
