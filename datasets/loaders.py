from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# =========================================================
# CSV LOADER
# =========================================================
def load_csv_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError("CSV precisa ter coluna 'date'")

    if "OT" not in df.columns:
        raise ValueError("CSV precisa ter coluna 'OT'")

    df["date"] = pd.to_datetime(df["date"])
    return df


# =========================================================
# NUMPY LOADER (GLOBAL)
# =========================================================
def load_series_numpy(path: str | Path) -> np.ndarray:
    df = load_csv_dataset(path)
    return df["OT"].to_numpy(dtype=np.float32)


# =========================================================
# TRAIN / VAL / TEST SPLIT
# =========================================================
def train_val_test_split(
    series: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
):
    T = len(series)

    t_train = int(T * train_ratio)
    t_val = int(T * (train_ratio + val_ratio))

    train = series[:t_train]
    val = series[t_train:t_val]
    test = series[t_val:]

    return train, val, test


# =========================================================
# SLIDING WINDOW (UNIVERSAL)
# =========================================================
def create_sliding_windows(
    series: np.ndarray,
    lookback: int,
    horizon: int,
    stride: int = 1,
):
    X, Y = [], []

    end = len(series) - (lookback + horizon) + 1

    for i in range(0, max(0, end), stride):
        x = series[i:i + lookback]
        y = series[i + lookback:i + lookback + horizon]

        X.append(x)
        Y.append(y)

    return np.array(X), np.array(Y)


# =========================================================
# TORCH DATASET (GENÉRICO)
# =========================================================
class SlidingWindowDataset:
    def __init__(self, series, lookback, horizon, stride=1):
        import torch

        self.series = series.astype(np.float32)
        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride

        self.X, self.Y = create_sliding_windows(series, lookback, horizon, stride)

        self.X = torch.tensor(self.X)
        self.Y = torch.tensor(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]