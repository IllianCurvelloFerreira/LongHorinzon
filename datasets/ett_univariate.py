from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_loader.preprocess_ett import preprocess_ett_dataset


def ensure_ett_csv(
    root_path: str | Path,
    data_name: str,
    target_col: str = "OT",
    data_dir: str | Path = "./nixtla_cache",
) -> Path:
    """
    Garante que o CSV univariado exista.
    Se não existir, gera automaticamente.
    """
    root_path = Path(root_path)
    csv_path = root_path / f"{data_name}.csv"

    if not csv_path.exists():
        print(f"[INFO] {csv_path} não encontrado. Gerando automaticamente...")
        root_path.mkdir(parents=True, exist_ok=True)

        preprocess_ett_dataset(
            group=data_name,
            data_dir=data_dir,
            out_dir=root_path,
        )

    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado mesmo após preprocessamento: {csv_path}")

    return csv_path


def load_univariate_series(
    root_path: str | Path,
    data_name: str,
    target_col: str = "OT",
    data_dir: str | Path = "./nixtla_cache",
) -> np.ndarray:
    """
    Carrega a série OT como vetor 1D [T,].
    """
    path = ensure_ett_csv(
        root_path=root_path,
        data_name=data_name,
        target_col=target_col,
        data_dir=data_dir,
    )

    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError(f"{data_name}.csv precisa ter a coluna 'date'.")
    if target_col not in df.columns:
        raise ValueError(f"{data_name}.csv precisa ter a coluna alvo '{target_col}'.")

    return df[target_col].to_numpy(dtype=np.float32)


def train_val_test_split_time_1d(
    series: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
):
    """
    series shape: [T,]
    """
    assert series.ndim == 1, "series must be 1D [T,]"

    total = series.shape[0]
    t_train = int(total * train_ratio)
    t_val = int(total * (train_ratio + val_ratio))

    train = series[:t_train]
    val = series[t_train:t_val]
    test = series[t_val:]

    return train, val, test


def seasonal_period_for_group(group: str) -> int:
    """
    ETTh*: hourly -> m=24
    ETTm*: 15-min -> m=96
    """
    if group.startswith("ETTh"):
        return 24
    if group.startswith("ETTm"):
        return 96
    return 1