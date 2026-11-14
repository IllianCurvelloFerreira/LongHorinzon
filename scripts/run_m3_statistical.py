
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

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


METRICS = [
    "MSE_original",
    "MAE_original",
    "sMAPE_pct",
    "MASE",
]

CATEGORIES = {
    "DEMOGRAPHIC",
    "FINANCE",
    "INDUSTRY",
    "MACRO",
    "MICRO",
    "OTHER",
}


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

    m = max(1, int(period))

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
            200 * np.abs(y - y_hat)
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


def read_categories(
    path: Path,
    group: str,
    selected_series_ids: list[str],
) -> dict:

    if not path.is_file():
        raise FileNotFoundError(
            f"Metadados nao encontrados: {path}. "
            "Execute antes: "
            "Rscript scripts/export_m3_categories.R"
        )

    meta = pd.read_csv(
        path,
        dtype={"series_id": "string"},
    )

    required = {
        "series_id",
        "category",
    }

    if not required.issubset(meta.columns):
        raise ValueError(
            f"CSV de categorias precisa conter "
            f"{sorted(required)}"
        )

    meta["series_id"] = (
        meta["series_id"]
        .str.strip()
        .str.upper()
    )

    meta["category"] = (
        meta["category"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if (
        meta["series_id"].isna().any()
        or meta["series_id"].duplicated().any()
    ):
        raise ValueError(
            "Metadados com series_id vazio ou duplicado"
        )

    if (
        meta["category"].isna().any()
        or not set(meta["category"]).issubset(CATEGORIES)
    ):
        raise ValueError(
            "Metadados com categorias invalidas"
        )

    if "frequency" in meta.columns:
        frequency = (
            meta["frequency"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        if not frequency.eq(group).all():
            raise ValueError(
                f"Metadados nao correspondem "
                f"a frequencia {group}"
            )

    missing = (
        set(selected_series_ids)
        - set(meta["series_id"])
    )

    if missing:
        raise ValueError(
            f"{len(missing)} series sem categoria: "
            f"{sorted(missing)[:12]}"
        )

    if "description" not in meta.columns:
        meta["description"] = ""

    return (
        meta.set_index("series_id")[
            ["category", "description"]
        ]
        .fillna("")
        .to_dict(orient="index")
    )


def summarize(
    df: pd.DataFrame,
    output: Path,
) -> None:

    print("\n===== RESULTADOS FINAIS =====")

    print(
        df.groupby("model")[METRICS]
        .mean()
        .round(6)
        .to_string()
    )

    if "category" not in df.columns:
        return

    summary = (
        df.groupby(
            ["category", "model"],
            as_index=False,
        )
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

    summary_path = output.with_name(
        output.stem + "_summary_by_category.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n===== METRICAS POR CATEGORIA =====")
    print(summary.round(4).to_string(index=False))
    print(f"Resumo por categoria: {summary_path}")

    if {"ARIMA", "SARIMA"}.issubset(
        set(df["model"])
    ):

        pivot = df.pivot(
            index=["series_id", "category"],
            columns="model",
            values="sMAPE_pct",
        )

        pairs = pivot[
            ["ARIMA", "SARIMA"]
        ].dropna()

        if len(pairs):

            wins = (
                pairs.assign(
                    arima_win=(
                        pairs["ARIMA"]
                        < pairs["SARIMA"]
                    ).astype(int),

                    sarima_win=(
                        pairs["SARIMA"]
                        < pairs["ARIMA"]
                    ).astype(int),

                    draw=(
                        pairs["ARIMA"]
                        == pairs["SARIMA"]
                    ).astype(int),
                )
                .groupby(level="category")[
                    [
                        "arima_win",
                        "sarima_win",
                        "draw",
                    ]
                ]
                .sum()
                .assign(
                    n_pairs=lambda x: x.sum(axis=1),

                    sarima_win_pct=lambda x: (
                        100
                        * x["sarima_win"]
                        / x["n_pairs"]
                    ),
                )
                .reset_index()
            )

            wins_path = output.with_name(
                output.stem
                + "_wins_by_category.csv"
            )

            wins.to_csv(
                wins_path,
                index=False,
            )

            print(
                "\n===== VITORIAS POR CATEGORIA "
                "(sMAPE) ====="
            )

            print(
                wins.round(2).to_string(index=False)
            )

            print(
                f"Vitorias por categoria: {wins_path}"
            )


def main() -> None:

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
        help="0 = todas as series",
    )

    parser.add_argument(
        "--output",
        default="./results/m3_statistical_results.csv",
    )

    parser.add_argument(
        "--categories-file",
        default=None,
        help=(
            "CSV oficial; mensal: "
            "<data-root>/m3_monthly_categories.csv"
        ),
    )

    args = parser.parse_args()

    if args.limit < 0:
        parser.error(
            "--limit nao pode ser negativo"
        )

    base = (
        Path(args.data_root)
        / args.group
    )

    subdirs = (
        sorted(
            d for d in base.iterdir()
            if d.is_dir()
        )
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

    if args.categories_file:
        categories_path = Path(
            args.categories_file
        )
    elif args.group == "monthly":
        categories_path = (
            Path(args.data_root)
            / "m3_monthly_categories.csv"
        )
    else:
        categories_path = None

    if categories_path is not None:
        categories = read_categories(
            categories_path,
            args.group,
            [
                d.name.upper()
                for d in subdirs
            ],
        )
    else:
        categories = None

    if categories is None:
        print(
            "[INFO] Execucao sem categorias "
            "para esse grupo"
        )

    rows = []

    for directory in subdirs:

        meta = json.loads(
            (
                directory / "metadata.json"
            ).read_text(encoding="utf-8")
        )

        train = pd.read_csv(
            directory / "Exchange.csv"
        )["OT"].to_numpy(dtype=float)

        truth = pd.read_csv(
            directory / "official_test.csv"
        )["OT"].to_numpy(dtype=float)

        h = int(meta["horizon"])
        m = int(meta["seasonality"])

        if h != len(truth):
            raise ValueError(
                f"Horizonte inconsistente: "
                f"{directory.name}"
            )

        for model in args.models:

            try:

                if model == "ARIMA":

                    pred = (
                        fit_forecast_arima_statsmodels(
                            train,
                            h,
                            ARIMAConfig(
                                order=(1, 1, 1)
                            ),
                        )
                    )

                else:

                    if m <= 1:
                        print(
                            f"[SKIP] {directory.name}: "
                            "sem sazonalidade definida"
                        )
                        continue

                    pred = (
                        fit_forecast_sarima_statsmodels(
                            train,
                            h,
                            SARIMAConfig(
                                order=(1, 1, 1),
                                seasonal_order=(
                                    1, 0, 1, m
                                ),
                                maxiter=200,
                            ),
                        )
                    )

                result = metrics(
                    truth,
                    np.asarray(pred),
                    train,
                    m,
                )

                row = {
                    "series_id": directory.name,
                    "group": args.group,
                    "model": model,
                    "horizon": h,
                    "train_length": len(train),
                }

                if categories is not None:
                    row.update(
                        categories[
                            directory.name.upper()
                        ]
                    )

                rows.append({
                    **row,
                    **result,
                })

                cat_text = (
                    f" | categoria={row['category']}"
                    if "category" in row
                    else ""
                )

                print(
                    f"[OK] {directory.name} | "
                    f"{model}{cat_text} | "
                    f"sMAPE="
                    f"{result['sMAPE_pct']:.2f}%"
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

    if df.duplicated(
        ["series_id", "model"]
    ).any():
        raise ValueError(
            "Resultados duplicados "
            "por serie e modelo"
        )

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output,
        index=False,
    )

    summarize(df, output)

    print(
        "\nResultados detalhados "
        f"(categorizados): {output}"
    )


if __name__ == "__main__":
    main()
