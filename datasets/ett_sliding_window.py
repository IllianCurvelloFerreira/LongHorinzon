from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_univariate_series(root_path: str | Path, data_name: str, target_col: str = "OT") -> np.ndarray:
    path = Path(root_path) / f"{data_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError(f"{data_name}.csv precisa ter a coluna 'date'.")
    if target_col not in df.columns:
        raise ValueError(f"{data_name}.csv precisa ter a coluna alvo '{target_col}'.")

    series = df[[target_col]].to_numpy(dtype=np.float32)  # [T, 1]
    return series


def train_val_test_split_time(
    series: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
):
    """
    series shape: [T, C]
    """
    assert series.ndim == 2, "series must be [T, C]"
    total = series.shape[0]

    t_train = int(total * train_ratio)
    t_val = int(total * (train_ratio + val_ratio))

    train = series[:t_train]
    val = series[t_train:t_val]
    test = series[t_val:]

    return train, val, test


class SlidingWindowDataset(Dataset):
    """
    x: [lookback, C]
    y: [horizon, C]
    """
    def __init__(self, series: np.ndarray, lookback: int, horizon: int, stride: int = 1):
        assert series.ndim == 2, "series must be [T, C]"

        self.series = series.astype(np.float32)
        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride

        total = self.series.shape[0]
        end = total - (lookback + horizon) + 1
        self.idxs = list(range(0, max(0, end), stride))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx: int):
        i = self.idxs[idx]
        x = self.series[i:i + self.lookback]
        y = self.series[i + self.lookback:i + self.lookback + self.horizon]

        return torch.from_numpy(x), torch.from_numpy(y)