from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from datasetsforecast.long_horizon2 import LongHorizon2
except ImportError:
    from datasetsforecast.long_horizon import LongHorizon as LongHorizon2

from data_loader.preprocess_ett import preprocess_ett_dataset


ALLOWED_GROUPS = {"ETTh1", "ETTh2", "ETTm1", "ETTm2"}


# =========================================================
# INTERNAL LONG LOADER
# =========================================================
def _load_ett_long(
    data_dir: str | Path,
    group: str,
) -> pd.DataFrame:
    if group not in ALLOWED_GROUPS:
        raise ValueError(f"group deve ser um destes: {sorted(ALLOWED_GROUPS)}")

    try:
        loaded = LongHorizon2.load(
            directory=str(data_dir),
            group=group,
            normalize=False,
        )
    except TypeError:
        loaded = LongHorizon2.load(
            directory=str(data_dir),
            group=group,
        )

    df = loaded[0] if isinstance(loaded, tuple) else loaded
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    return df


# =========================================================
# UNIVARIATE CSV (CURRENT BEHAVIOR)
# =========================================================
def ensure_ett_csv(
    root_path: str | Path,
    data_name: str,
    target_col: str = "OT",
    data_dir: str | Path = "./nixtla_cache",
) -> Path:
    """
    Garante que o CSV univariado exista.
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
        raise FileNotFoundError(
            f"Arquivo não encontrado mesmo após preprocessamento: {csv_path}"
        )

    return csv_path


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

    series = df[[target_col]].to_numpy(dtype=np.float32)
    return series


# =========================================================
# MULTIVARIATE CSV (NEW BEHAVIOR)
# =========================================================
def ensure_ett_multivariate_csv(
    root_path: str | Path,
    data_name: str,
    data_dir: str | Path = "./nixtla_cache",
) -> Path:
    """
    Garante que o CSV multivariado exista.
    Salva todas as variáveis: date + features.
    """
    root_path = Path(root_path)
    csv_path = root_path / f"{data_name}_multivariate.csv"

    if not csv_path.exists():
        print(f"[INFO] {csv_path} não encontrado. Gerando automaticamente...")
        root_path.mkdir(parents=True, exist_ok=True)

        df_long = _load_ett_long(
            data_dir=data_dir,
            group=data_name,
        )

        wide = (
            df_long.pivot(index="ds", columns="unique_id", values="y")
            .sort_index()
            .dropna()
            .reset_index()
            .rename(columns={"ds": "date"})
        )

        wide.to_csv(csv_path, index=False)
        print(f"[OK] {data_name} salvo em {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado mesmo após geração multivariada: {csv_path}"
        )

    return csv_path


def load_multivariate_series(
    root_path: str | Path,
    data_name: str,
    target_col: str = "OT",
    data_dir: str | Path = "./nixtla_cache",
):
    """
    Carrega série multivariada [T, C] e retorna:
    - series: np.ndarray [T, C]
    - target_idx: índice da coluna alvo
    - feature_cols: nomes das features
    """
    path = ensure_ett_multivariate_csv(
        root_path=root_path,
        data_name=data_name,
        data_dir=data_dir,
    )

    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError(f"{path.name} precisa ter a coluna 'date'.")

    feature_cols = [c for c in df.columns if c != "date"]
    if target_col not in feature_cols:
        raise ValueError(f"{path.name} precisa ter a coluna alvo '{target_col}'.")

    series = df[feature_cols].to_numpy(dtype=np.float32)
    target_idx = feature_cols.index(target_col)

    return series, target_idx, feature_cols


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
# CURRENT DATASET (UNCHANGED)
# x: [lookback, C]
# y: [horizon, C]
# =========================================================
class SlidingWindowDataset(Dataset):
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


# =========================================================
# NEW DATASET FOR MULTIVARIATE INPUT / UNIVARIATE TARGET
# x: [lookback, C]
# y: [horizon, 1]
# =========================================================
class SlidingWindowTargetDataset(Dataset):
    def __init__(
        self,
        series: np.ndarray,
        lookback: int,
        horizon: int,
        target_idx: int,
        stride: int = 1,
    ):
        assert series.ndim == 2, "series must be [T, C]"

        self.series = series.astype(np.float32)
        self.lookback = lookback
        self.horizon = horizon
        self.target_idx = target_idx
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

        x = self.series[i:i + self.lookback]  # [L, C]
        y_full = self.series[i + self.lookback:i + self.lookback + self.horizon]  # [H, C]
        y = y_full[:, self.target_idx:self.target_idx + 1]  # [H, 1]

        return torch.from_numpy(x), torch.from_numpy(y)