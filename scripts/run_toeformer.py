from __future__ import annotations

import argparse

import numpy as np

from training.engine_toeformer import run_experiment
from utils.device import get_device
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", type=str, default="./data/ETT")
    parser.add_argument("--data", type=str, default="ETTh1", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"])
    parser.add_argument("--target", type=str, default="OT")

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

    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    metrics_runs = []
    for i in range(args.itr):
        run_seed = args.seed + i
        metrics_runs.append(run_experiment(args, run_seed, device=device, set_seed_fn=set_seed))

    mse_mean = float(np.mean([m.mse for m in metrics_runs]))
    mse_std = float(np.std([m.mse for m in metrics_runs], ddof=1)) if len(metrics_runs) > 1 else 0.0
    mae_mean = float(np.mean([m.mae for m in metrics_runs]))
    mae_std = float(np.std([m.mae for m in metrics_runs], ddof=1)) if len(metrics_runs) > 1 else 0.0

    print("\n===== MÉDIA FINAL =====")
    print(f"MSE: {mse_mean:.6f} ± {mse_std:.6f}")
    print(f"MAE: {mae_mean:.6f} ± {mae_std:.6f}")


if __name__ == "__main__":
    main()