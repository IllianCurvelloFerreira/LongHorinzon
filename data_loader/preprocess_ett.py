from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

try:
    from datasetsforecast.long_horizon2 import LongHorizon2
except ImportError:
    LongHorizon2 = None

try:
    from datasetsforecast.long_horizon import LongHorizon
except ImportError:
    LongHorizon = None


ALLOWED_GROUPS = {"ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange"}


def _load_dataset_with_optional_normalize(
    loader,
    data_dir: str | Path,
    group: str,
    normalize: bool = False,
) -> pd.DataFrame:
    """
    Alguns loaders aceitam normalize=..., outros não.
    Esta função tenta primeiro com normalize e, se der TypeError,
    tenta novamente sem esse argumento.
    """
    try:
        loaded = loader.load(
            directory=str(data_dir),
            group=group,
            normalize=normalize,
        )
    except TypeError:
        loaded = loader.load(
            directory=str(data_dir),
            group=group,
        )

    df = loaded[0] if isinstance(loaded, tuple) else loaded
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])

    return df


def load_ett_long(
    data_dir: str | Path,
    group: str,
    normalize: bool = False,
) -> pd.DataFrame:
    if group not in ALLOWED_GROUPS:
        raise ValueError(f"group inválido: {group}")

    # Caso especial:
    # Exchange não está disponível no LongHorizon2.
    # Para não alterar o comportamento dos datasets que já funcionaram,
    # apenas Exchange usa LongHorizon.
    if group == "Exchange":
        if LongHorizon is None:
            raise ImportError(
                "LongHorizon não está disponível. "
                "Verifique a instalação do pacote datasetsforecast."
            )

        print("[INFO] Carregando Exchange via LongHorizon")
        return _load_dataset_with_optional_normalize(
            loader=LongHorizon,
            data_dir=data_dir,
            group=group,
            normalize=normalize,
        )

    # Para ETTh1, ETTh2, ETTm1, ETTm2 e Weather:
    # mantém o comportamento anterior usando LongHorizon2.
    if LongHorizon2 is not None:
        print(f"[INFO] Carregando {group} via LongHorizon2")
        return _load_dataset_with_optional_normalize(
            loader=LongHorizon2,
            data_dir=data_dir,
            group=group,
            normalize=normalize,
        )

    # Fallback apenas se LongHorizon2 não existir no ambiente.
    # Isso preserva a lógica antiga, que usava LongHorizon como alternativa.
    if LongHorizon is not None:
        print(f"[WARN] LongHorizon2 indisponível. Carregando {group} via LongHorizon")
        return _load_dataset_with_optional_normalize(
            loader=LongHorizon,
            data_dir=data_dir,
            group=group,
            normalize=normalize,
        )

    raise ImportError(
        "Nenhum loader disponível. "
        "Verifique a instalação de datasetsforecast."
    )


def pivot_long_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Converte o formato longo:
        ds, unique_id, y

    para formato wide:
        date, var1, var2, ..., OT

    Também corrige valores sentinela como -9999, comuns no Weather.
    """
    wide = (
        df_long.pivot(index="ds", columns="unique_id", values="y")
        .sort_index()
    )

    # Corrige valores sentinela de dados ausentes/erro de medição.
    # No Weather, por exemplo, OT pode vir com -9999.
    wide = wide.replace(-9999, np.nan)

    # Interpola respeitando a ordem temporal.
    # limit_direction="both" também preenche NaNs no começo/fim, se existirem.
    wide = wide.interpolate(method="linear", limit_direction="both")

    # Remove qualquer linha que ainda tenha NaN após interpolação.
    wide = wide.dropna()

    return wide


def extract_univariate(
    df_long: pd.DataFrame,
    target_col: str = "OT",
) -> pd.DataFrame:
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


def extract_multivariate(
    df_long: pd.DataFrame,
    target_col: str = "OT",
) -> pd.DataFrame:
    wide = pivot_long_to_wide(df_long)

    if target_col not in wide.columns:
        fallback_col = wide.columns[-1]
        print(f"[WARN] {target_col} não encontrado → usando {fallback_col} como OT")
        wide = wide.rename(columns={fallback_col: "OT"})
    elif target_col != "OT":
        wide = wide.rename(columns={target_col: "OT"})

    df = wide.reset_index().rename(columns={"ds": "date"})

    # Garante que OT fique como última coluna.
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
    df_long = load_ett_long(
        data_dir=data_dir,
        group=group,
        normalize=normalize,
    )

    if multivariate:
        df_out = extract_multivariate(
            df_long,
            target_col=target_col,
        )
        suffix = "_multivariate"
    else:
        df_out = extract_univariate(
            df_long,
            target_col=target_col,
        )
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
    df_long = load_ett_long(
        data_dir=data_dir,
        group=group,
    )
    df_uni = extract_univariate(df_long)

    return df_uni["OT"].to_numpy(dtype=np.float32)


if __name__ == "__main__":
    preprocess_all_ett()