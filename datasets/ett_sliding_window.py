from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Import do seu preprocess
from data_loader.preprocess_ett import preprocess_ett_dataset


# =========================================================
# GARANTIR CSV (AUTO-GENERATE)
# =========================================================
def ensure_ett_csv(
    root_path: str | Path,
    data_name: str,
    target_col: str = "OT",
    data_dir: str | Path = "./nixtla_cache",
) -> Path:
    """
    Garante que o CSV do dataset exista.
    Se não existir, gera automaticamente.
    """
    root_path = Path(root_path)
    csv_path = root_path / f"{data_name}.csv"

    if not csv_path.exists():
        print(f"[INFO] {csv_path} não encontrado. Gerando automaticamente...")

        root_path.mkdir(parents=True, exist_ok=True)

        # Chama seu preprocess
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


# =========================================================
# LOAD SERIES
# =========================================================
def load_univariate_series(
    root_path: str | Path,
    data_name: str,
    target_col: str = "OT",
    data_dir: str | Path = "./nixtla_cache",
) -> np.ndarray:
    """
    Carrega série univariada [T, 1]
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
        raise ValueError(
            f"{data_name}.csv precisa ter a coluna alvo '{target_col}'."
        )

    series = df[[target_col]].to_numpy(dtype=np.float32)  # [T, 1]
    return series


# =========================================================
# SPLIT
# =========================================================
def train_val_test_split_time(
    series: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
):
    """
    Split temporal da série
    """
    assert series.ndim == 2, "series must be [T, C]"

    total = series.shape[0]
    t_train = int(total * train_ratio)
    t_val = int(total * (train_ratio + val_ratio))

    train = series[:t_train]
    val = series[t_train:t_val]
    test = series[t_val:]

    return train, val, test


# =========================================================
# DATASET
# =========================================================
class SlidingWindowDataset(Dataset):
    """
    x: [lookback, C]
    y: [horizon, C]
    """

    def __init__(
        self,
        series: np.ndarray,
        lookback: int,
        horizon: int,
        stride: int = 1,
    ):
        assert series.ndim == 2, "series must be [T, C]"

        self.series = series.astype(np.float32)
        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride

        total = self.series.shape[0]
        end = total - (lookback + horizon) + 1

        self.idxs = list(range(0, max(0, end), stride))

        if len(self.idxs) == 0:
            raise ValueError(
                f"Série muito curta para lookback={lookback} e horizon={horizon}"
            )

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx: int):
        i = self.idxs[idx]

        x = self.series[i:i + self.lookback]
        y = self.series[i + self.lookback:i + self.lookback + self.horizon]

        return torch.from_numpy(x), torch.from_numpy(y)