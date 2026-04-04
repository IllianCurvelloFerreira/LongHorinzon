from __future__ import annotations

import argparse

from training.engine_statistical import run_benchmark


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path", type=str, default="./data/ETT")
    parser.add_argument("--data", type=str, default="ETTh1", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"])
    parser.add_argument("--target", type=str, default="OT")

    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--stride_mode", type=str, default="H", choices=["H", "1"])
    parser.add_argument("--max_origins", type=int, default=5)

    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    parser.add_argument("--models", nargs="+", default=["ARIMA", "SARIMA"], choices=["ARIMA", "SARIMA"])
    parser.add_argument("--use_auto_arima", action="store_true")
    parser.add_argument("--sarima_light", action="store_true")
    parser.add_argument("--progress_every", type=int, default=2)

    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--run_all_horizons", action="store_true")
    parser.add_argument("--results_csv", type=str, default="")

    return parser.parse_args()


def main():
    args = parse_args()

    df_res = run_benchmark(args)

    print("\n===== RESULTADOS FINAIS =====")
    print(df_res.to_string(index=False))

    if args.results_csv:
        df_res.to_csv(args.results_csv, index=False)
        print(f"\nResultados salvos em: {args.results_csv}")

    print("\n===== LATEX =====")
    print(df_res.to_latex(index=False, float_format='%.4f'))


if __name__ == "__main__":
    main()