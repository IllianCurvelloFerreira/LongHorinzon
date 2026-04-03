from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from torch.utils.data import Dataset


# =========================================================
# AUTO DOWNLOAD + PREPROCESS
# =========================================================
def ensure_ett_csv(root_path: str | Path, data_name: str, cache_dir="./nixtla_cache") -> Path:
    root_path = Path(root_path)
    root_path.mkdir(parents=True, exist_ok=True)

    csv_path = root_path / f"{data_name}.csv"

    if csv_path.exists():
        return csv_path

    print(f"[INFO] {csv_path} não encontrado. Gerando automaticamente...")

    # tentativa de import flexível
    preprocess_fn = None

    try:
        from data_loader.preprocess_ett import preprocess_ett_dataset
        preprocess_fn = preprocess_ett_dataset
    except ImportError:
        try:
            from data_loader.preprocess_ett import preprocess_ett_dataset
            preprocess_fn = preprocess_ett_dataset
        except ImportError:
            try:
                preprocess_fn = preprocess_ett_dataset
            except ImportError:
                try:
                    from data_loader.preprocess_ett import preprocess_ett_dataset
                    preprocess_fn = preprocess_ett_dataset
                except ImportError:
                    pass

    if preprocess_fn is None:
        raise ImportError(
            "Verifique o caminho do arquivo preprocess_ett.py."
        )

    preprocess_fn(
        group=data_name,
        data_dir=cache_dir,
        out_dir=root_path,
    )

    if not csv_path.exists():
        raise FileNotFoundError(f"Falha ao gerar dataset: {csv_path}")

    print(f"[OK] Dataset gerado em {csv_path}")
    return csv_path


# =========================================================
# TIME FEATURES
# =========================================================
def time_features(df_stamp: pd.DataFrame, data_name: str) -> np.ndarray:
    df_stamp = df_stamp.copy()
    df_stamp["date"] = pd.to_datetime(df_stamp["date"])

    df_stamp["month"] = df_stamp["date"].dt.month
    df_stamp["day"] = df_stamp["date"].dt.day
    df_stamp["weekday"] = df_stamp["date"].dt.weekday
    df_stamp["hour"] = df_stamp["date"].dt.hour

    if data_name in {"ETTm1", "ETTm2"}:
        df_stamp["minute"] = (df_stamp["date"].dt.minute // 15).astype(int)
        cols = ["month", "day", "weekday", "hour", "minute"]
    else:
        cols = ["month", "day", "weekday", "hour"]

    return df_stamp[cols].to_numpy(dtype=np.int64)


# =========================================================
# BORDERS
# =========================================================
def get_borders(data_name: str, seq_len: int) -> tuple[list[int], list[int]]:
    if data_name in {"ETTh1", "ETTh2"}:
        train = 12 * 30 * 24
        val = 4 * 30 * 24
        test = 4 * 30 * 24
    elif data_name in {"ETTm1", "ETTm2"}:
        train = 12 * 30 * 24 * 4
        val = 4 * 30 * 24 * 4
        test = 4 * 30 * 24 * 4
    else:
        raise ValueError(f"Dataset não suportado: {data_name}")

    border1s = [0, train - seq_len, train + val - seq_len]
    border2s = [train, train + val, train + val + test]
    return border1s, border2s


# =========================================================
# DATASET
# =========================================================
class ETTGBTDataset(Dataset):
    def __init__(
        self,
        root_path: str | Path,
        data_name: str,
        flag: str,
        seq_len: int,
        label_len: int,
        pred_len: int,
        target: str = "OT",
        criterion: str = "Standard",
        use_time: bool = False,
        cache_dir: str = "./nixtla_cache", 
    ):
        assert flag in {"train", "val", "test"}

        self.root_path = root_path
        self.data_name = data_name
        self.flag = flag
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.target = target
        self.criterion = criterion
        self.use_time = use_time
        self.cache_dir = cache_dir

        self.scaler = None
        self.data_x = None
        self.data_y = None
        self.data_stamp = None

        self._read_data()

    def _read_data(self) -> None:
        file_path = ensure_ett_csv(
            self.root_path,
            self.data_name,
            cache_dir=self.cache_dir,
        )

        df_raw = pd.read_csv(file_path)

        if "date" not in df_raw.columns:
            raise ValueError(f"{self.data_name}.csv precisa ter a coluna 'date'.")
        if self.target not in df_raw.columns:
            raise ValueError(f"{self.data_name}.csv precisa ter a coluna alvo '{self.target}'.")

        df_data = df_raw[[self.target]].copy()

        border1s, border2s = get_borders(self.data_name, self.seq_len)
        set_type = {"train": 0, "val": 1, "test": 2}[self.flag]
        border1 = border1s[set_type]
        border2 = border2s[set_type]

        train_data = df_data.iloc[border1s[0]:border2s[0]].values.astype(np.float32)
        full_data = df_data.values.astype(np.float32)

        if self.criterion == "Standard":
            self.scaler = StandardScaler()
        elif self.criterion == "MaxAbs":
            self.scaler = MaxAbsScaler()
        else:
            raise ValueError("criterion deve ser 'Standard' ou 'MaxAbs'.")

        self.scaler.fit(train_data)
        scaled = self.scaler.transform(full_data).astype(np.float32)

        self.data_x = scaled[border1:border2]
        self.data_y = scaled[border1:border2]

        if self.use_time:
            df_stamp = df_raw[["date"]].iloc[border1:border2].copy()
            self.data_stamp = time_features(df_stamp, self.data_name)
        else:
            length = border2 - border1
            self.data_stamp = np.zeros((length, 0), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int):
        s_begin = index
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
            torch.tensor(seq_x_mark, dtype=torch.long),
            torch.tensor(seq_y_mark, dtype=torch.long),
        )

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        shape = arr.shape
        flat = arr.reshape(-1, shape[-1])
        inv = self.scaler.inverse_transform(flat)
        return inv.reshape(shape)