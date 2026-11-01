"""Audita previsoes SARIMA gravadas pelo benchmark M3 sem alterar modelos existentes.

Na raiz do projeto:
    python -m scripts.audit_m3_sarima \
        --data-root ./data/M3 \
        --group monthly \
        --limit 5 \
        --original-results ./results/m3_monthly.csv

O fit independente reproduz as configuracoes usadas no run_m3_statistical.py:
ARIMA=(1,1,1), sazonal=(1,0,1,m), maxiter=25, sem tendencia.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def calculate_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    if y.shape != pred.shape or not np.isfinite(pred).all():
        raise ValueError("Previsao invalida, diferente do horizonte ou nao finita")
    return {
        "MSE_original": float(np.mean((y - pred) ** 2)),
        "MAE_original": float(np.mean(np.abs(y - pred))),
        "sMAPE_pct": float(
            np.mean(200.0 * np.abs(y - pred) / np.maximum(np.abs(y) + np.abs(pred), 1e-8))
        ),
    }


def same_metrics(saved: pd.Series, calculated: dict[str, float]) -> bool:
    # Os arquivos CSV guardam precisao integral. Usamos margem para diferencias numericas.
    return all(
        col in saved.index
        and pd.notna(saved[col])
        and bool(np.isclose(float(saved[col]), val, rtol=1e-5, atol=1e-4))
        for col, val in calculated.items()
    )


def audit_directory(directory: Path, original: pd.DataFrame | None, maxiter: int) -> dict:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    y_train = pd.read_csv(directory / "Exchange.csv")["OT"].to_numpy(dtype=np.float64)
    y_test = pd.read_csv(directory / "official_test.csv")["OT"].to_numpy(dtype=np.float64)
    h = int(metadata["horizon"])
    m = int(metadata["seasonality"])
    if len(y_test) != h or m <= 1:
        raise ValueError("Horizonte inconsistente ou frequencia sem sazonalidade (m<=1)")
    if len(y_train) < 2 * m:
        raise ValueError("Historico muito curto para verificacao sazonal")

    old = None
    if original is not None:
        matched = original.loc[
            (original["series_id"].astype(str) == directory.name)
            & (original["model"].astype(str).str.upper() == "SARIMA")
        ]
        if len(matched) != 1:
            raise ValueError(f"Esperada exatamente 1 linha antiga de SARIMA; encontradas {len(matched)}")
        old = matched.iloc[0]

    # Exatamente a mesma parametrizacao do SARIMA existente, mas SEM fallback.
    model = SARIMAX(
        y_train,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, m),
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
        simple_differencing=False,
        concentrate_scale=True,
    )

    fit_error = None
    result = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            try:
                result = model.fit(disp=False, method="lbfgs", maxiter=maxiter, low_memory=True)
            except TypeError:
                result = model.fit(disp=False, method="lbfgs", maxiter=maxiter)
        except Exception as exc:
            fit_error = f"{type(exc).__name__}: {exc}"

    predicted = None
    direct_metrics = None
    converged = False
    names = []
    params = {}
    if result is not None:
        converged = bool(result.mle_retvals.get("converged", False))
        names = [str(n) for n in result.param_names]
        params = {str(n): float(v) for n, v in zip(result.param_names, result.params)}
        try:
            predicted = np.asarray(result.forecast(steps=h), dtype=np.float64)
            direct_metrics = calculate_metrics(y_test, predicted)
        except Exception as exc:
            fit_error = f"{type(exc).__name__}: {exc}"

    seasonal_terms = any(".S.L" in name for name in names)
    seasonality = np.tile(y_train[-m:], int(np.ceil(h / m)))[:h]
    naive_metrics = calculate_metrics(y_test, seasonality)
    old_equals_sarima = old is not None and direct_metrics is not None and same_metrics(old, direct_metrics)
    old_equals_naive = old is not None and same_metrics(old, naive_metrics)

    if old is None:
        status = "SEM_CSV_ANTIGO" if fit_error is None else "ERRO_FIT"
    elif old_equals_sarima and converged and seasonal_terms:
        status = "CONFIRMADO_SARIMA_CONVERGIU"
    elif old_equals_sarima:
        status = "SARIMA_COM_ALERTA_CONVERGENCIA_OU_SAZONALIDADE"
    elif old_equals_naive:
        status = "PROVAVEL_FALLBACK_SEASONAL_NAIVE"
    else:
        status = "DIVERGENCIA_REVISAR_CONFIGURACAO"

    return {
        "series_id": directory.name,
        "status": status,
        "converged": converged,
        "seasonal_terms": seasonal_terms,
        "params": json.dumps(params, ensure_ascii=False),
        "fit_error": fit_error or "",
        "warning_types": ",".join(sorted({type(w.message).__name__ for w in caught})),
        "sMAPE_original_csv": float(old["sMAPE_pct"]) if old is not None else np.nan,
        "sMAPE_SARIMA_recalculado": direct_metrics["sMAPE_pct"] if direct_metrics else np.nan,
        "sMAPE_seasonal_naive": naive_metrics["sMAPE_pct"],
        "MSE_SARIMA_recalculado": direct_metrics["MSE_original"] if direct_metrics else np.nan,
        "MAE_SARIMA_recalculado": direct_metrics["MAE_original"] if direct_metrics else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data/M3")
    parser.add_argument("--group", choices=["monthly", "quarterly", "yearly", "other"], default="monthly")
    parser.add_argument("--limit", type=int, default=5, help="0 = todas as series exportadas")
    parser.add_argument("--original-results", default="./results/m3_monthly.csv")
    parser.add_argument("--output", default="./results/m3_sarima_audit.csv")
    parser.add_argument("--maxiter", type=int, default=25, help="Deve igualar o valor do experimento antigo")
    args = parser.parse_args()
    if args.limit < 0 or args.maxiter <= 0:
        parser.error("limit >= 0 e maxiter > 0 sao obrigatorios")

    source = Path(args.original_results)
    if not source.is_file():
        raise SystemExit(f"Arquivo antigo nao encontrado: {source}")
    original = pd.read_csv(source)
    base = Path(args.data_root) / args.group
    if not base.is_dir():
        raise SystemExit(f"Pasta de dados nao encontrada: {base}")
    subdirs = sorted(path for path in base.iterdir() if path.is_dir())
    if args.limit:
        subdirs = subdirs[: args.limit]
    if not subdirs:
        raise SystemExit("Nenhuma serie encontrada")

    records = []
    for directory in subdirs:
        try:
            row = audit_directory(directory, original, args.maxiter)
            records.append(row)
            print(
                f"{row['series_id']}: {row['status']} | "
                f"convergiu={row['converged']} | "
                f"sMAPE_csv={row['sMAPE_original_csv']:.4f}% | "
                f"sMAPE_SARIMA={row['sMAPE_SARIMA_recalculado']:.4f}% | "
                f"sMAPE_naive={row['sMAPE_seasonal_naive']:.4f}%"
            )
        except Exception as exc:
            records.append({"series_id": directory.name, "status": "ERRO_AUDITORIA", "fit_error": str(exc)})
            print(f"{directory.name}: ERRO_AUDITORIA: {exc}")

    df = pd.DataFrame(records)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    print("\nContagem dos estados:")
    print(df["status"].value_counts().to_string())
    print(f"\nAuditoria salva em: {target}")


if __name__ == "__main__":
    main()
