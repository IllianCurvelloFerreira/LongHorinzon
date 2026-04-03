from __future__ import annotations

from pathlib import Path
from typing import List

try:
    from datasetsforecast.long_horizon2 import LongHorizon2
except ImportError:
    from datasetsforecast.long_horizon import LongHorizon as LongHorizon2


ALLOWED_GROUPS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]


def download_ett_dataset(
    group: str,
    data_dir: str | Path = "./nixtla_cache",
) -> None:
    """
    Faz download (ou garante cache) de um dataset ETT.
    """
    if group not in ALLOWED_GROUPS:
        raise ValueError(f"group inválido: {group}")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Baixando {group}...")

    # Apenas chamar já baixa/cacheia
    LongHorizon2.load(directory=str(data_dir), group=group)

    print(f"[OK] {group} disponível em {data_dir}")


def download_all_ett(
    data_dir: str | Path = "./nixtla_cache",
    groups: List[str] = ALLOWED_GROUPS,
) -> None:
    """
    Baixa todos os datasets ETT.
    """
    for g in groups:
        download_ett_dataset(g, data_dir)


if __name__ == "__main__":
    download_all_ett()