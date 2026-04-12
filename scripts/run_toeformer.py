from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from training.engine_toeformer import run_experiment
from utils.device import get_device
from utils.seed import set_seed


ALL_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]


def parse_args():
    parser = argparse.ArgumentParser()

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

    parser.add_argument("--lookback", type=int, default=336)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--decomp_kernel", type=int, default=25)
    parser.add_argument("--k_global", type=int, default=25)
    parser.add_argument("--k_local", type=int, default=3)

    parser.add_argument("--lam1", type=float, default=1.0)
    parser.add_argument("--lam2", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.6)

    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--results_csv", type=str, default="")

    return parser.parse_args()


def summarize_metrics(metrics_runs):
    mse_mean = float(np.mean([m.mse for m in metrics_runs]))
    mse_std = float(np.std([m.mse for m in metrics_runs], ddof=1)) if len(metrics_runs) > 1 else 0.0
    mae_mean = float(np.mean([m.mae for m in metrics_runs]))
    mae_std = float(np.std([m.mae for m in metrics_runs], ddof=1)) if len(metrics_runs) > 1 else 0.0

    return {
        "mse_mean": mse_mean,
        "mse_std": mse_std,
        "mae_mean": mae_mean,
        "mae_std": mae_std,
    }


def run_single_dataset(args, device: str):
    metrics_runs = []

    for i in range(args.itr):
        run_seed = args.seed + i
        metrics_runs.append(run_experiment(args, run_seed, device=device, set_seed_fn=set_seed))

    summary = summarize_metrics(metrics_runs)

    print("\n===== MÉDIA FINAL =====")
    print(f"MSE: {summary['mse_mean']:.6f} ± {summary['mse_std']:.6f}")
    print(f"MAE: {summary['mae_mean']:.6f} ± {summary['mae_std']:.6f}")

    return summary


def run_all_datasets(args, device: str):
    rows = []

    for dataset in ALL_DATASETS:
        print("\n" + "=" * 70)
        print(f"DATASET: {dataset}")
        print("=" * 70)

        args.data = dataset
        summary = run_single_dataset(args, device)

        rows.append(
            {
                "data": dataset,
                "input_mode": args.input_mode,
                "lookback": args.lookback,
                "horizon": args.horizon,
                "itr": args.itr,
                "lam1": args.lam1,
                "lam2": args.lam2,
                "alpha": args.alpha,
                "learning_rate": args.learning_rate,
                "train_epochs": args.train_epochs,
                "batch_size": args.batch_size,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "decomp_kernel": args.decomp_kernel,
                "k_global": args.k_global,
                "k_local": args.k_local,
                "dropout": args.dropout,
                "mse_mean": summary["mse_mean"],
                "mse_std": summary["mse_std"],
                "mae_mean": summary["mae_mean"],
                "mae_std": summary["mae_std"],
            }
        )

    df = pd.DataFrame(rows)

    print("\n===== RESULTADOS FINAIS =====")
    print(df.to_string(index=False))

    if args.results_csv:
        df.to_csv(args.results_csv, index=False)
        print(f"\nResultados salvos em: {args.results_csv}")


def main():
    args = parse_args()
    device = get_device()

    if args.run_all:
        run_all_datasets(args, device)
    else:
        run_single_dataset(args, device)


if __name__ == "__main__":
    main()