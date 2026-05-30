from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

try:
    from datasetsforecast.long_horizon2 import LongHorizon2
except ImportError:
    from datasetsforecast.long_horizon import LongHorizon as LongHorizon2


ALLOWED_GROUPS = {"ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange"}


def load_ett_long(data_dir: str | Path, group: str, normalize: bool = False) -> pd.DataFrame:
    if group not in ALLOWED_GROUPS:
        raise ValueError(f"group inválido: {group}")

    loaded = LongHorizon2.load(directory=str(data_dir), group=group)

    df = loaded[0] if isinstance(loaded, tuple) else loaded
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])

    return df


def pivot_long_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    wide = (
        df_long.pivot(index="ds", columns="unique_id", values="y")
        .sort_index()
        .dropna()
    )

    return wide


def extract_univariate(df_long: pd.DataFrame, target_col: str = "OT") -> pd.DataFrame:
    wide = pivot_long_to_wide(df_long)

    if target_col in wide.columns:
        col = target_col
    else:
        col = wide.columns[-1]
        print(f"[WARN] {target_col} não encontrado → usando {col}")

    return (
        wide[[col]]
        .reset_index()
        .rename(columns={"ds": "date", col: "OT"})
    )


def extract_multivariate(df_long: pd.DataFrame, target_col: str = "OT") -> pd.DataFrame:
    wide = pivot_long_to_wide(df_long)

    if target_col not in wide.columns:
        fallback_col = wide.columns[-1]
        print(f"[WARN] {target_col} não encontrado → usando {fallback_col} como OT")
        wide = wide.rename(columns={fallback_col: "OT"})
    elif target_col != "OT":
        wide = wide.rename(columns={target_col: "OT"})

    df = wide.reset_index().rename(columns={"ds": "date"})

    # Garante que OT fique como última coluna
    cols = [c for c in df.columns if c not in ["date", "OT"]]
    df = df[["date"] + cols + ["OT"]]

    return df


def preprocess_ett_dataset(
    group: str,
    data_dir: str | Path = "./nixtla_cache",
    out_dir: str | Path = "./data/ETT",
    target_col: str = "OT",
    multivariate: bool = False,
    normalize: bool = False,
) -> Path:
    df_long = load_ett_long(data_dir, group, normalize)

    if multivariate:
        df_out = extract_multivariate(df_long, target_col=target_col)
        suffix = "_multivariate"
    else:
        df_out = extract_univariate(df_long, target_col=target_col)
        suffix = ""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{group}{suffix}.csv"
    df_out.to_csv(out_path, index=False)

    print(f"[OK] {group} salvo em {out_path}")
    return out_path


def preprocess_all_ett(
    data_dir: str | Path = "./nixtla_cache",
    out_dir: str | Path = "./data/ETT",
    multivariate: bool = False,
) -> Dict[str, Path]:
    results = {}

    for g in sorted(ALLOWED_GROUPS):
        results[g] = preprocess_ett_dataset(
            group=g,
            data_dir=data_dir,
            out_dir=out_dir,
            multivariate=multivariate,
        )

    return results


def load_univariate_numpy(
    group: str,
    data_dir: str | Path = "./nixtla_cache",
) -> np.ndarray:
    df_long = load_ett_long(data_dir, group)
    df_uni = extract_univariate(df_long)

    return df_uni["OT"].to_numpy(dtype=np.float32)


if __name__ == "__main__":
    preprocess_all_ett()