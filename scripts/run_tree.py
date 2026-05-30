# Adding changes
from __future__ import annotations

import argparse

import numpy as np

from training.engine_tree import run_experiment


ALL_DATASETS = ALL_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange"]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["random_forest", "xgboost"],
    )

    parser.add_argument("--root_path", type=str, default="./data/ETT")
    parser.add_argument("--data_dir", type=str, default="./nixtla_cache")
    parser.add_argument("--data", type=str, default="ETTh1", choices=ALL_DATASETS)
    parser.add_argument("--target", type=str, default="OT")

    parser.add_argument(
        "--input_mode",
        type=str,
        default="univariate",
        choices=["univariate", "multivariate"],
    )

    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    # Hiperparâmetros comuns
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--n_jobs", type=int, default=-1)

    # Random Forest
    parser.add_argument("--min_samples_leaf", type=int, default=1)

    # XGBoost
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)

    # PCA opcional no vetor tabular
    parser.add_argument("--use_pca", action="store_true")
    parser.add_argument("--pca_components", type=int, default=0)

    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.use_pca and args.pca_components <= 0:
        raise ValueError("Quando --use_pca for usado, --pca_components deve ser > 0.")

    return args


def main():
    args = parse_args()

    metrics_runs = []

    for i in range(args.itr):
        run_seed = args.seed + i
        metrics_runs.append(run_experiment(args, run_seed=run_seed))

    mse_values = [m.mse for m in metrics_runs]
    mae_values = [m.mae for m in metrics_runs]

    mse_mean = float(np.mean(mse_values))
    mse_std = float(np.std(mse_values, ddof=1)) if len(mse_values) > 1 else 0.0

    mae_mean = float(np.mean(mae_values))
    mae_std = float(np.std(mae_values, ddof=1)) if len(mae_values) > 1 else 0.0

    print("\n===== MÉDIA FINAL =====")
    print(f"MSE: {mse_mean:.6f} ± {mse_std:.6f}")
    print(f"MAE: {mae_mean:.6f} ± {mae_std:.6f}")


if __name__ == "__main__":
    main()