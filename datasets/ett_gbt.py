from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from torch.utils.data import Dataset

try:
    from datasetsforecast.long_horizon2 import LongHorizon2
except ImportError:
    from datasetsforecast.long_horizon import LongHorizon as LongHorizon2

from data_loader.preprocess_ett import preprocess_ett_dataset


ALLOWED_GROUPS = {"ETTh1", "ETTh2", "ETTm1", "ETTm2"}


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


def ensure_ett_csv(
    root_path: str | Path,
    data_name: str,
    target_col: str = "OT",
    data_dir: str | Path = "./nixtla_cache",
) -> Path:
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


def ensure_ett_multivariate_csv(
    root_path: str | Path,
    data_name: str,
    data_dir: str | Path = "./nixtla_cache",
) -> Path:
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
        raise FileNotFoundError(f"Arquivo não encontrado mesmo após geração multivariada: {csv_path}")

    return csv_path


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


def get_file_path(
    root_path: str | Path,
    data_name: str,
    input_mode: str = "univariate",
    data_dir: str | Path = "./nixtla_cache",
) -> str:
    if input_mode == "univariate":
        path = ensure_ett_csv(
            root_path=root_path,
            data_name=data_name,
            data_dir=data_dir,
        )
    elif input_mode == "multivariate":
        path = ensure_ett_multivariate_csv(
            root_path=root_path,
            data_name=data_name,
            data_dir=data_dir,
        )
    else:
        raise ValueError("input_mode deve ser 'univariate' ou 'multivariate'.")

    return str(path)


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
        input_mode: str = "univariate",
        data_dir: str | Path = "./nixtla_cache",
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
        self.input_mode = input_mode
        self.data_dir = data_dir

        self.scaler_x = None
        self.scaler_y = None
        self.data_x = None
        self.data_y = None
        self.data_stamp = None
        self.enc_in = None
        self.c_out = 1

        self._read_data()

    def _read_data(self) -> None:
        file_path = get_file_path(
            self.root_path,
            self.data_name,
            input_mode=self.input_mode,
            data_dir=self.data_dir,
        )
        df_raw = pd.read_csv(file_path)

        if "date" not in df_raw.columns:
            raise ValueError(f"{Path(file_path).name} precisa ter a coluna 'date'.")
        if self.target not in df_raw.columns:
            raise ValueError(f"{Path(file_path).name} precisa ter a coluna alvo '{self.target}'.")

        if self.input_mode == "univariate":
            x_cols = [self.target]
        else:
            x_cols = [c for c in df_raw.columns if c != "date"]

        y_cols = [self.target]

        df_x = df_raw[x_cols].copy()
        df_y = df_raw[y_cols].copy()

        self.enc_in = df_x.shape[1]

        border1s, border2s = get_borders(self.data_name, self.seq_len)
        set_type = {"train": 0, "val": 1, "test": 2}[self.flag]
        border1 = border1s[set_type]
        border2 = border2s[set_type]

        train_x = df_x.iloc[border1s[0]:border2s[0]].values.astype(np.float32)
        full_x = df_x.values.astype(np.float32)

        train_y = df_y.iloc[border1s[0]:border2s[0]].values.astype(np.float32)
        full_y = df_y.values.astype(np.float32)

        if self.criterion == "Standard":
            self.scaler_x = StandardScaler()
            self.scaler_y = StandardScaler()
        elif self.criterion == "MaxAbs":
            self.scaler_x = MaxAbsScaler()
            self.scaler_y = MaxAbsScaler()
        else:
            raise ValueError("criterion deve ser 'Standard' ou 'MaxAbs'.")

        self.scaler_x.fit(train_x)
        self.scaler_y.fit(train_y)

        scaled_x = self.scaler_x.transform(full_x).astype(np.float32)
        scaled_y = self.scaler_y.transform(full_y).astype(np.float32)

        self.data_x = scaled_x[border1:border2]
        self.data_y = scaled_y[border1:border2]

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

        seq_x = self.data_x[s_begin:s_end]             # [seq_len, enc_in]
        seq_y = self.data_y[r_begin:r_end]             # [label_len+pred_len, 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
            torch.tensor(seq_x_mark, dtype=torch.long),
            torch.tensor(seq_y_mark, dtype=torch.long),
        )

    def inverse_transform_y(self, arr: np.ndarray) -> np.ndarray:
        shape = arr.shape
        flat = arr.reshape(-1, shape[-1])
        inv = self.scaler_y.inverse_transform(flat)
        return inv.reshape(shape)