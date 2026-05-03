from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from datasets.ett_sliding_window import (
    load_multivariate_series,
    load_univariate_series,
    train_val_test_split_time,
)
from models.tree.model import build_tree_model


@dataclass
class Metrics:
    mse: float
    mae: float


def make_supervised_windows(
    series_x: np.ndarray,
    series_y: np.ndarray,
    lookback: int,
    horizon: int,
    stride: int = 1,
):
    """
    Transforma série temporal em problema tabular supervisionado.

    series_x: [T, C]
    series_y: [T, 1]

    Retorna:
        X: [N, lookback * C]
        y: [N, horizon]
    """
    assert series_x.ndim == 2, "series_x deve ser [T, C]"
    assert series_y.ndim == 2, "series_y deve ser [T, 1]"

    total = len(series_x)
    end = total - lookback - horizon + 1

    X_list = []
    y_list = []

    for i in range(0, max(0, end), stride):
        x_window = series_x[i:i + lookback]
        y_window = series_y[i + lookback:i + lookback + horizon]

        X_list.append(x_window.reshape(-1))
        y_list.append(y_window.reshape(-1))

    if not X_list:
        raise ValueError(
            "Nenhuma janela foi criada. Verifique lookback, horizon e tamanho da série."
        )

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)

    return X, y


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return Metrics(mse=mse, mae=mae)


def apply_pca_if_needed(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    args,
):
    """
    PCA opcional após flatten das janelas.

    Aqui o PCA é aplicado no vetor tabular completo:
        [lookback * n_features]

    Fit apenas no treino.
    """
    if not args.use_pca:
        return train_x, val_x, test_x

    if args.pca_components <= 0:
        raise ValueError("Quando --use_pca for usado, --pca_components deve ser > 0.")

    n_components = min(args.pca_components, train_x.shape[1])

    pca = PCA(n_components=n_components)
    train_x = pca.fit_transform(train_x).astype(np.float32)
    val_x = pca.transform(val_x).astype(np.float32)
    test_x = pca.transform(test_x).astype(np.float32)

    print(f"[INFO] PCA aplicado nos inputs tabulares.")
    print(f"[INFO] PCA components: {n_components}")
    print(f"[INFO] Variância explicada total: {pca.explained_variance_ratio_.sum():.4f}")

    return train_x, val_x, test_x


def build_data(args):
    if args.input_mode == "univariate":
        series = load_univariate_series(
            root_path=args.root_path,
            data_name=args.data,
            target_col=args.target,
            data_dir=args.data_dir,
        )

        train_series, val_series, test_series = train_val_test_split_time(
            series,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        train_x_series = scaler_x.fit_transform(train_series).astype(np.float32)
        val_x_series = scaler_x.transform(val_series).astype(np.float32)
        test_x_series = scaler_x.transform(test_series).astype(np.float32)

        train_y_series = scaler_y.fit_transform(train_series).astype(np.float32)
        val_y_series = scaler_y.transform(val_series).astype(np.float32)
        test_y_series = scaler_y.transform(test_series).astype(np.float32)

    elif args.input_mode == "multivariate":
        series, target_idx, feature_cols = load_multivariate_series(
            root_path=args.root_path,
            data_name=args.data,
            target_col=args.target,
            data_dir=args.data_dir,
        )

        train_series, val_series, test_series = train_val_test_split_time(
            series,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )

        scaler_x = StandardScaler()
        train_x_series = scaler_x.fit_transform(train_series).astype(np.float32)
        val_x_series = scaler_x.transform(val_series).astype(np.float32)
        test_x_series = scaler_x.transform(test_series).astype(np.float32)

        scaler_y = StandardScaler()
        train_y_raw = train_series[:, target_idx:target_idx + 1]
        val_y_raw = val_series[:, target_idx:target_idx + 1]
        test_y_raw = test_series[:, target_idx:target_idx + 1]

        train_y_series = scaler_y.fit_transform(train_y_raw).astype(np.float32)
        val_y_series = scaler_y.transform(val_y_raw).astype(np.float32)
        test_y_series = scaler_y.transform(test_y_raw).astype(np.float32)

    else:
        raise ValueError("input_mode deve ser 'univariate' ou 'multivariate'.")

    train_x, train_y = make_supervised_windows(
        train_x_series,
        train_y_series,
        lookback=args.lookback,
        horizon=args.horizon,
        stride=args.stride,
    )

    val_x, val_y = make_supervised_windows(
        val_x_series,
        val_y_series,
        lookback=args.lookback,
        horizon=args.horizon,
        stride=args.stride,
    )

    test_x, test_y = make_supervised_windows(
        test_x_series,
        test_y_series,
        lookback=args.lookback,
        horizon=args.horizon,
        stride=args.stride,
    )

    train_x, val_x, test_x = apply_pca_if_needed(
        train_x,
        val_x,
        test_x,
        args,
    )

    return train_x, train_y, val_x, val_y, test_x, test_y


def run_experiment(args, run_seed: int):
    args.seed = run_seed

    train_x, train_y, val_x, val_y, test_x, test_y = build_data(args)

    model = build_tree_model(args)

    print(
        f"\n===== Run seed={run_seed} | model={args.model} | "
        f"data={args.data} | horizon={args.horizon} | "
        f"input_mode={args.input_mode} | use_pca={args.use_pca} ====="
    )

    print(f"[INFO] train_x shape: {train_x.shape}")
    print(f"[INFO] train_y shape: {train_y.shape}")

    model.fit(train_x, train_y)

    val_pred = model.predict(val_x)
    test_pred = model.predict(test_x)

    val_metrics = evaluate(val_y, val_pred)
    test_metrics = evaluate(test_y, test_pred)

    print(f"VAL  | MSE={val_metrics.mse:.6f} | MAE={val_metrics.mae:.6f}")
    print(f"TEST | MSE={test_metrics.mse:.6f} | MAE={test_metrics.mae:.6f}")

    return test_metrics