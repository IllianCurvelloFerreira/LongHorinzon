"""Cruza categorias oficiais M3 (exportadas com Mcomp/R) com os resultados.

Uso (na raiz LongHorinzon):
python -m scripts.analyze_m3_categories \
  --results results/m3_monthly_1428.csv \
  --metadata data/M3/m3_monthly_categories.csv \
  --out-dir results

Não altera os modelos nem os dados, somente cria relatórios CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

METRICS = ["MSE_original", "MAE_original", "sMAPE_pct", "MASE"]
CATEGORIES = ["DEMOGRAPHIC", "FINANCE", "INDUSTRY", "MACRO", "MICRO", "OTHER"]
EXPECTED = dict(zip(CATEGORIES, [111,145,334,312,474,52]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results/m3_monthly_1428.csv")
    p.add_argument("--metadata", default="data/M3/m3_monthly_categories.csv")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(args.results, dtype={"series_id": "string"})
    meta = pd.read_csv(args.metadata, dtype={"series_id": "string"})
    required_result = {"series_id", "model"} | set(METRICS)
    required_meta = {"series_id", "category"}
    for label, df, cols in [("results",results,required_result),("metadata",meta,required_meta)]:
        missing = cols - set(df.columns)
        if missing:
            raise ValueError(f"Colunas ausentes em {label}: {sorted(missing)}")
    results["series_id"] = results["series_id"].str.strip().str.upper()
    meta["series_id"] = meta["series_id"].str.strip().str.upper()
    meta["category"] = meta["category"].astype("string").str.strip().str.upper()

    if meta["series_id"].duplicated().any():
        raise ValueError("Há IDs duplicados nos metadados")
    if results.duplicated(subset=["series_id", "model"]).any():
        raise ValueError("Há resultados duplicados para a mesma série/modelo")
    if not set(meta["category"]).issubset(set(CATEGORIES)):
        raise ValueError(f"Categorias inesperadas: {set(meta['category'])-set(CATEGORIES)}")
    counts = meta["category"].value_counts().to_dict()
    if len(meta) != 1428 or counts != EXPECTED:
        raise ValueError(f"A base de metadados deve ter 1.428 séries com as contagens oficiais; observado: {counts}")

    merged = results.merge(meta, on="series_id", how="left", validate="many_to_one", indicator=True)
    missing_ids = merged.loc[merged["_merge"]!="both", "series_id"].unique().tolist()
    if missing_ids:
        raise ValueError(f"Séries sem categoria oficial ({len(missing_ids)}): {missing_ids[:15]}")
    merged = merged.drop(columns="_merge")
    merged.to_csv(out_dir / "m3_monthly_categorized.csv", index=False)

    summary = (merged.groupby(["category", "model"], observed=True)
                .agg(n_series=("series_id", "nunique"),
                     MSE_mean=("MSE_original", "mean"),
                     MAE_mean=("MAE_original", "mean"),
                     sMAPE_mean=("sMAPE_pct", "mean"),
                     sMAPE_median=("sMAPE_pct", "median"),
                     MASE_mean=("MASE", "mean"))
                .reset_index().sort_values(["category","model"]))
    summary.to_csv(out_dir / "m3_category_summary.csv", index=False)

    # Apenas pares completos, para não premiar o único modelo presente
    piv = merged.pivot(index=["series_id", "category"], columns="model", values="sMAPE_pct")
    if not {"ARIMA", "SARIMA"}.issubset(piv.columns):
        raise ValueError("Faltam resultados ARIMA ou SARIMA")
    pairs = piv[["ARIMA", "SARIMA"]].dropna()
    pairs = pairs.assign(
        arima_win=(pairs["ARIMA"] < pairs["SARIMA"]).astype(int),
        sarima_win=(pairs["SARIMA"] < pairs["ARIMA"]).astype(int),
        draw=(pairs["ARIMA"] == pairs["SARIMA"]).astype(int),
    )
    wins = (pairs.groupby(level="category")[["arima_win", "sarima_win", "draw"]]
            .sum().assign(n_pairs=lambda x:x.sum(axis=1),
                          sarima_win_pct=lambda x:100*x["sarima_win"]/x["n_pairs"])
            .reset_index())
    wins.to_csv(out_dir / "m3_category_wins_smape.csv", index=False)

    print("\nCONTAGEM ORIGINAL POR CATEGORIA:")
    print(meta["category"].value_counts().sort_index().to_string())
    print("\nMÉTRICAS MÉDIAS POR CATEGORIA/MODELO:")
    print(summary.round(4).to_string(index=False))
    print("\nVITÓRIAS POR CATEGORIA (sMAPE):")
    print(wins.round(2).to_string(index=False))
    print("\nArquivos gerados em:", out_dir.resolve())

if __name__ == "__main__":
    main()
