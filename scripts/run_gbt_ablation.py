"""GBT — ablação A/B/C: alvo OT, multivariado completo e controle sem contexto.

Coloque em scripts/run_gbt_ablation.py, na raiz do projeto LongHorinzon:
    python -m scripts.run_gbt_ablation --data Weather --horizon 96 --train_epochs 1

Não altera nenhum arquivo original do projeto nem regenera os CSVs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets.ett_gbt import ETTGBTDataset
from models.gbt.model import GBTVanillaStandalone
from training.engine_gbt import (
    build_loaders,
    evaluate_loss,
    parse_s_layers,
    test_model,
    train_one_stage,
)
from utils.device import get_device
from utils.seed import set_seed


CONFIGS = {
    "A": ("univariate", "univariate", "OT apenas"),
    "B": ("multivariate", "multivariate", "OT + contexto multivariado"),
    "C": ("multivariate", "univariate", "21 entradas carregadas; contexto desativado"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(args) -> dict:
    root = Path(args.root_path)
    uni_path = root / f"{args.data}.csv"
    multi_path = root / f"{args.data}_multivariate.csv"
    for p in [uni_path, multi_path]:
        if not p.is_file():
            raise FileNotFoundError(
                f"Arquivo ausente: {p}. Gere e congele os CSVs antes da ablação."
            )

    uni = pd.read_csv(uni_path)
    multi = pd.read_csv(multi_path)
    if "OT" not in uni or "OT" not in multi:
        raise ValueError("Os dois CSVs precisam conter a coluna OT.")
    if "date" not in uni or "date" not in multi:
        raise ValueError("Os dois CSVs precisam conter a coluna date.")
    if uni["date"].duplicated().any() or multi["date"].duplicated().any():
        raise ValueError("Há datas duplicadas em pelo menos um dos CSVs.")
    if len(uni) != len(multi) or not uni["date"].equals(multi["date"]):
        raise ValueError("As datas do univariado e multivariado não coincidem.")
    if not np.array_equal(uni["OT"].to_numpy(), multi["OT"].to_numpy()):
        raise ValueError("Os valores de OT diferem entre os arquivos.")
    n_vars = sum(c != "date" for c in multi.columns)
    if n_vars < 2:
        raise ValueError(f"Multivariado tem apenas {n_vars} variável(is).")

    target = uni["OT"].to_numpy(dtype=np.float64)
    print("\n[OK] Mesmas datas e mesma OT nos dois modos.")
    print(f"[INFO] {args.data}: {len(uni)} datas, {n_vars} variáveis multivariadas")
    print(f"[INFO] OT: min={target.min():.6f} max={target.max():.6f} "
          f"mean={target.mean():.6f}")
    file_info = {
        "univariate": {"path": str(uni_path.resolve()), "sha256": sha256(uni_path)},
        "multivariate": {"path": str(multi_path.resolve()), "sha256": sha256(multi_path)},
        "rows": len(uni), "multivariate_features": n_vars,
        "OT_min": float(target.min()), "OT_max": float(target.max()),
        "OT_mean": float(target.mean()),
    }
    for name in ("univariate", "multivariate"):
        print(f"[SHA256] {name}: {file_info[name]['sha256']}")
    return file_info


def make_engine_args(args, loader_mode: str):
    return SimpleNamespace(
        root_path=args.root_path,
        data_dir=args.data_dir,
        data=args.data,
        target="OT",
        input_mode=loader_mode,
        seq_len=args.horizon,
        label_len=args.horizon,
        pred_len=args.horizon,
        criterion="Standard",
        time=args.time,
        test_inverse=False,  # MSE/MAE normalizados; sMAPE na escala original.
        split_mode="ratio",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_epochs=args.train_epochs,
        learning_rate=args.learning_rate,
    )


def run_one(args, config: str, seed: int, device: str) -> list[dict]:
    loader_mode, model_mode, description = CONFIGS[config]
    print("\n" + "=" * 80)
    print(f"{args.data} H={args.horizon} | Config {config} ({description}) | seed={seed}")
    print("=" * 80)

    # Fundamental para emparelhar seeds de A e C.
    set_seed(seed)
    eargs = make_engine_args(args, loader_mode)
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_loaders(eargs)
    target_idx = train_ds.x_cols.index("OT")

    model = GBTVanillaStandalone(
        enc_in=train_ds.enc_in,
        dec_in=1,
        c_out=1,
        label_len=args.horizon,
        pred_len=args.horizon,
        fd_model=args.fd_model,
        d_model=args.d_model,
        d_layers=args.d_layers,
        n_heads=args.n_heads,
        s_layers=parse_s_layers(args.s_layers),
        dropout=args.dropout,
        use_time=args.time,
        with_minute=args.data in {"ETTm1", "ETTm2"},
        target_idx=target_idx,
        input_mode=model_mode,
    ).to(device)

    # C usa o dataset multivariado, mas o modelo SEM a branch de contexto.
    if config == "C":
        assert train_ds.enc_in > 1
        assert model.context_embed is None and model.context_proj is None
    if config == "B":
        assert model.context_embed is not None and model.context_proj is not None

    best_stage1 = train_one_stage(
        model, train_loader, val_loader, val_ds, eargs, device, "first stage"
    )
    if best_stage1 is None:
        raise RuntimeError("Treinamento Stage 1 não retornou checkpoint.")
    model.load_state_dict(best_stage1)

    best_stage2 = train_one_stage(
        model, train_loader, val_loader, val_ds, eargs, device, "second stage"
    )
    if best_stage2 is None:
        raise RuntimeError("Treinamento Stage 2 não retornou checkpoint.")
    model.load_state_dict(best_stage2)

    rows = []
    for stage_name, stage in [("Stage 1", "first stage"), ("Stage 2", "second stage")]:
        val_mse, val_mae, val_smape = evaluate_loss(
            model, val_loader, val_ds, nn.MSELoss(), eargs, device, stage
        )
        tm = test_model(
            model, test_loader, test_ds, eargs, device, stage, diagnostics=False
        )
        print(f"[RESULTADO] {config} seed={seed} {stage_name}: "
              f"val_MSE={val_mse:.6f}, test_MSE={tm.mse:.6f}, "
              f"test_MAE={tm.mae:.6f}, test_sMAPE={tm.smape:.2f}%")
        rows.append({
            "data": args.data,
            "horizon": args.horizon,
            "seed": seed,
            "config": config,
            "description": description,
            "loader_mode": loader_mode,
            "model_mode": model_mode,
            "n_input_features": train_ds.enc_in,
            "stage": stage_name,
            "val_mse_norm": val_mse,
            "val_mae_norm": val_mae,
            "val_smape_original_pct": val_smape,
            "test_mse_norm": tm.mse,
            "test_mae_norm": tm.mae,
            "test_smape_original_pct": tm.smape,
        })
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return rows


def create_summary(df: pd.DataFrame) -> pd.DataFrame:
    metric_names = ["val_mse_norm", "test_mse_norm", "test_mae_norm", "test_smape_original_pct"]
    rows = []
    for (config, stage), group in df.groupby(["config", "stage"], sort=True):
        result = {"config": config, "stage": stage, "n_seeds": len(group)}
        for metric in metric_names:
            result[f"{metric}_mean"] = float(group[metric].mean())
            result[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(result)
    return pd.DataFrame(rows).sort_values(["stage", "config"])


def create_paired_comparison(df: pd.DataFrame) -> pd.DataFrame:
    tables = []
    for metric in ["test_mse_norm", "test_mae_norm", "test_smape_original_pct"]:
        w = df.pivot(index=["seed", "stage"], columns="config", values=metric)
        w = w.reset_index()
        w.insert(2, "metric", metric)
        w["B_minus_A"] = w["B"] - w["A"]
        w["C_minus_A"] = w["C"] - w["A"]
        w["B_minus_C"] = w["B"] - w["C"]
        tables.append(w)
    return pd.concat(tables, ignore_index=True)


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser(description="GBT A/B/C ablation, no modification to original models")
    p.add_argument("--data", default="Weather", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange"])
    p.add_argument("--root_path", default="./data/ETT")
    p.add_argument("--data_dir", default="./nixtla_cache")
    p.add_argument("--out_dir", default="./results/ablation_gbt")
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--seed_start", type=int, default=42)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--train_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--fd_model", type=int, default=32)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_layers", type=int, default=2)
    p.add_argument("--s_layers", default="3,2,1")
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--time", action="store_true", help="Usa features temporais, se necessário")
    return p.parse_args()


def main():
    args = parse_args()
    if args.horizon <= 0 or args.n_seeds <= 0 or args.train_epochs <= 0:
        raise ValueError("Horizon, número de seeds e número de épocas devem ser positivos.")
    if not 0 < args.train_ratio < 1 or not 0 < args.val_ratio < 1 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Proporções de treino/validação inválidas.")
    print("GBT — A/B/C | MSE/MAE padronizados | sMAPE na escala original")
    files_info = verify_files(args)
    device = get_device()
    print(f"Dispositivo: {device}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    last_seed = args.seed_start + args.n_seeds - 1
    prefix = f"{args.data}_H{args.horizon}_E{args.train_epochs}_S{args.seed_start}-{last_seed}"
    run_file = out / f"{prefix}_runs.csv"
    summary_file = out / f"{prefix}_summary.csv"
    paired_file = out / f"{prefix}_paired.csv"
    metadata_file = out / f"{prefix}_metadata.json"

    metadata = {
        "args": vars(args),
        "files": files_info,
        "git_commit": git_sha(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "configs": {k: {"loader_mode": x[0], "model_mode": x[1], "description": x[2]} for k, x in CONFIGS.items()},
        "notes": "Same target OT/date required; checkpoints per stage by minimum validation MSE. "
                 "Stage 1 and 2 test scores from the same seed; no seed selected using test data. "
                 "A and C are expected to be numerically close; if not, investigate."
    }
    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

    all_rows = []
    for seed in range(args.seed_start, last_seed + 1):
        for config in CONFIGS:
            all_rows.extend(run_one(args, config, seed, device))
            pd.DataFrame(all_rows).to_csv(run_file, index=False)

    runs = pd.DataFrame(all_rows)
    summary = create_summary(runs)
    paired = create_paired_comparison(runs)
    summary.to_csv(summary_file, index=False)
    paired.to_csv(paired_file, index=False)

    print("\n" + "=" * 80)
    print("RESULTADOS — MÉDIAS POR CONFIGURAÇÃO E STAGE")
    print("=" * 80)
    selected = ["config", "stage", "n_seeds", "test_mse_norm_mean", "test_mse_norm_std", "test_mae_norm_mean", "test_smape_original_pct_mean"]
    print(summary[selected].round(6).to_string(index=False))

    print("\n" + "=" * 80)
    print("CONTROLE A × C — DIFERENÇA DE MSE POR SEED/STAGE")
    print("=" * 80)
    mse_cmp = paired[paired["metric"] == "test_mse_norm"]
    print(mse_cmp[["seed", "stage", "A", "B", "C", "B_minus_C", "C_minus_A"]].round(6).to_string(index=False))
    if mse_cmp["C_minus_A"].abs().max() > 1e-4:
        print("[AVISO] A e C diferem além de 1e-4 no MSE; investigar inicialização, dados ou não-determinismo CUDA.")
    else:
        print("[OK] A e C equivalentes dentro da tolerância 1e-4 de MSE.")

    print("\nArquivos salvos:")
    for p in [run_file, summary_file, paired_file, metadata_file]:
        print(" -", p)


if __name__ == "__main__":
    main()
