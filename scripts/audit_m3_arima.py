"""Auditoria independente do ARIMA (1,1,1) utilizado no benchmark M3.

Executar a partir da raiz do repositorio LongHorinzon:

    python -m scripts.audit_m3_arima \
      --data-root ./data/M3 \
      --group monthly \
      --limit 50 \
      --original-results ./results/m3_monthly_50.csv \
      --output ./results/m3_arima_audit_50.csv

Nao altera os modelos existentes. Reproduz exatamente o ajuste de
models/statistical/arima.py: SARIMAX com order=(1,1,1),
seasonal_order=None, trend='n', enforce_stationarity=False,
enforce_invertibility=False, seguido de fit(disp=False).

Importante: o ARIMA existente nao tem fallback. Por isso uma previsao
reproduzida com converged=False deve ser tratada como ajuste nao convergido,
nao como previsao naive.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

METRICS = ("MSE_original", "MAE_original", "sMAPE_pct", "MASE")


def calculate_metrics(y_true: np.ndarray, forecast: np.ndarray,
                      history: np.ndarray, period: int) -> dict[str, float]:
    """Mesmas formulas de metrics() em scripts/run_m3_statistical.py."""
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(forecast, dtype=np.float64).reshape(-1)
    hist = np.asarray(history, dtype=np.float64).reshape(-1)

    if y.shape != pred.shape or not (np.isfinite(y).all() and np.isfinite(pred).all()):
        raise ValueError("Previsao invalida: horizonte ou valores nao finitos")

    mae = float(np.mean(np.abs(y - pred)))
    m = max(1, int(period))
    scale = (float(np.mean(np.abs(hist[m:] - hist[:-m])))
             if len(hist) > m else float("nan"))

    return {
        "MSE_original": float(np.mean((y - pred) ** 2)),
        "MAE_original": mae,
        "sMAPE_pct": float(np.mean(
            200.0 * np.abs(y - pred)
            / np.maximum(np.abs(y) + np.abs(pred), 1e-8)
        )),
        "MASE": mae / scale if np.isfinite(scale) and scale > 1e-12 else float("nan"),
    }


def matching_metrics(saved: pd.Series, fresh: dict[str, float],
                     rtol: float, atol: float) -> tuple[bool, list[str]]:
    """Compara MSE, MAE, sMAPE e MASE; informa eventuais divergencias."""
    mismatches = []
    for col in METRICS:
        if col not in saved.index:
            mismatches.append(f"coluna_ausente:{col}")
            continue
        old = float(saved[col]) if pd.notna(saved[col]) else np.nan
        new = fresh[col]
        if not np.isclose(old, new, rtol=rtol, atol=atol, equal_nan=True):
            mismatches.append(col)
    return len(mismatches) == 0, mismatches


def audit_directory(directory: Path, original: pd.DataFrame,
                    maxiter: int | None, rtol: float, atol: float) -> dict:
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    y_train = pd.read_csv(directory / "Exchange.csv")["OT"].to_numpy(dtype=np.float64)
    y_test = pd.read_csv(directory / "official_test.csv")["OT"].to_numpy(dtype=np.float64)
    h = int(meta["horizon"])
    m = int(meta["seasonality"])

    if len(y_test) != h:
        raise ValueError(f"Horizonte diferente do teste: {h} != {len(y_test)}")
    if len(y_train) < 5 or not np.isfinite(y_train).all():
        raise ValueError("Historico insuficiente ou contem valores nao finitos")

    matching_rows = original.loc[
        (original["series_id"].astype(str) == directory.name)
        & (original["model"].astype(str).str.upper() == "ARIMA")
    ]
    if len(matching_rows) != 1:
        raise ValueError(
            f"Esperada uma linha ARIMA para {directory.name}; "
            f"encontradas {len(matching_rows)}"
        )
    saved = matching_rows.iloc[0]

    # Especificacao identica a models/statistical/arima.py.
    model = SARIMAX(
        y_train,
        order=(1, 1, 1),
        seasonal_order=None,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fit_result = None
    fit_error = ""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            # Sem --maxiter: reproduz fit(disp=False) da implementacao original.
            kwargs = {"disp": False}
            if maxiter is not None:
                kwargs["maxiter"] = maxiter
            fit_result = model.fit(**kwargs)
        except Exception as exc:
            fit_error = f"{type(exc).__name__}: {exc}"

    converged = False
    iterations = np.nan
    params: dict[str, float] = {}
    recalc = None

    if fit_result is not None:
        retvals = getattr(fit_result, "mle_retvals", {}) or {}
        converged = bool(retvals.get("converged", False))
        iterations = retvals.get("iterations", np.nan)
        params = {
            str(name): float(value)
            for name, value in zip(fit_result.param_names, fit_result.params)
        }
        try:
            predicted = np.asarray(fit_result.forecast(steps=h), dtype=np.float64)
            recalc = calculate_metrics(y_test, predicted, y_train, m)
        except Exception as exc:
            fit_error = f"{type(exc).__name__}: {exc}"

    # Apenas comparacao diagnostica; o ARIMA original NAO tem fallback naive.
    naive_forecast = np.full(h, y_train[-1], dtype=np.float64)
    naive_smape = calculate_metrics(y_test, naive_forecast, y_train, m)["sMAPE_pct"]

    same, different = (
        matching_metrics(saved, recalc, rtol, atol)
        if recalc is not None else (False, ["sem_previsao"])
    )

    if recalc is None:
        status = "ERRO_AJUSTE_OU_PREVISAO"
    elif not same:
        status = "DIVERGENCIA_REVISAR_CONFIGURACAO"
    elif not converged:
        status = "ARIMA_REPRODUZIDO_MAS_NAO_CONVERGIU"
    else:
        status = "CONFIRMADO_ARIMA_CONVERGIU"

    return {
        "series_id": directory.name,
        "status": status,
        "converged": converged,
        "iterations": iterations,
        "params": json.dumps(params, ensure_ascii=False),
        "fit_error": fit_error,
        "warning_types": ",".join(sorted({type(w.message).__name__ for w in captured})),
        "metrics_match": same,
        "metrics_different": ",".join(different),
        "MSE_csv": float(saved["MSE_original"]) if "MSE_original" in saved else np.nan,
        "MSE_ARIMA_recalculado": recalc["MSE_original"] if recalc else np.nan,
        "MAE_csv": float(saved["MAE_original"]) if "MAE_original" in saved else np.nan,
        "MAE_ARIMA_recalculado": recalc["MAE_original"] if recalc else np.nan,
        "sMAPE_csv": float(saved["sMAPE_pct"]) if "sMAPE_pct" in saved else np.nan,
        "sMAPE_ARIMA_recalculado": recalc["sMAPE_pct"] if recalc else np.nan,
        "MASE_csv": float(saved["MASE"]) if "MASE" in saved else np.nan,
        "MASE_ARIMA_recalculado": recalc["MASE"] if recalc else np.nan,
        "sMAPE_naive_persistencia": naive_smape,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="./data/M3")
    p.add_argument("--group", choices=["monthly", "quarterly", "yearly", "other"], default="monthly")
    p.add_argument("--limit", type=int, default=50, help="0 = todas as series da pasta")
    p.add_argument("--original-results", default="./results/m3_monthly_50.csv")
    p.add_argument("--output", default="./results/m3_arima_audit_50.csv")
    p.add_argument("--maxiter", type=int, default=None,
                   help="Omitir para reproduzir exatamente o fit(disp=False) original")
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--atol", type=float, default=1e-4)
    args = p.parse_args()

    if args.limit < 0 or (args.maxiter is not None and args.maxiter <= 0):
        p.error("--limit deve ser >= 0 e --maxiter (se informado) deve ser > 0")

    csv_path = Path(args.original_results)
    if not csv_path.is_file():
        raise SystemExit(f"CSV original nao encontrado: {csv_path}")
    original = pd.read_csv(csv_path)
    required = {"series_id", "model"}
    if not required.issubset(original.columns):
        raise SystemExit(f"CSV sem as colunas obrigatorias: {sorted(required)}")

    folder = Path(args.data_root) / args.group
    if not folder.is_dir():
        raise SystemExit(f"Pasta de dados nao encontrada: {folder}")
    directories = sorted(p for p in folder.iterdir() if p.is_dir())
    if args.limit:
        directories = directories[:args.limit]
    if not directories:
        raise SystemExit("Nenhuma serie encontrada para auditar")

    rows = []
    for directory in directories:
        try:
            row = audit_directory(directory, original, args.maxiter, args.rtol, args.atol)
            rows.append(row)
            print(
                f"{row['series_id']}: {row['status']} | "
                f"convergiu={row['converged']} | "
                f"MSE_csv={row['MSE_csv']:.6g} | "
                f"MSE_ARIMA={row['MSE_ARIMA_recalculado']:.6g} | "
                f"MAE_csv={row['MAE_csv']:.6g} | "
                f"MAE_ARIMA={row['MAE_ARIMA_recalculado']:.6g} | "
                f"sMAPE_csv={row['sMAPE_csv']:.4f}% | "
                f"sMAPE_ARIMA={row['sMAPE_ARIMA_recalculado']:.4f}%"
            )
        except Exception as exc:
            rows.append({
                "series_id": directory.name,
                "status": "ERRO_AUDITORIA",
                "fit_error": f"{type(exc).__name__}: {exc}",
            })
            print(f"{directory.name}: ERRO_AUDITORIA: {exc}")

    audited = pd.DataFrame(rows)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    audited.to_csv(target, index=False)

    print("\nContagem dos estados:")
    print(audited["status"].value_counts().to_string())
    print(f"\nAuditoria salva em: {target}")

if __name__ == "__main__":
    main()
