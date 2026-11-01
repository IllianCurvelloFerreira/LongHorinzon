
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FREQ = {
    "yearly": "YS",
    "quarterly": "QS",
    "monthly": "MS",
    "other": "D",
}

HORIZON = {
    "yearly": 6,
    "quarterly": 8,
    "monthly": 18,
    "other": 8,
}

PERIOD = {
    "yearly": 1,
    "quarterly": 4,
    "monthly": 12,
    "other": 1,
}


def export_series(series, group: str, output_root: Path) -> dict:
    name = str(series.sn)

    valid_chars = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789_-"
    )

    if not name or any(ch not in valid_chars for ch in name):
        raise ValueError(f"ID invalido: {name!r}")

    train = np.asarray(
        series.x, dtype=np.float64
    ).reshape(-1)

    test = np.asarray(
        series.xx, dtype=np.float64
    ).reshape(-1)

    h = int(series.h)

    if h != HORIZON[group] or len(test) != h:
        raise ValueError(
            f"Horizonte inesperado em {name}: "
            f"h={h}, test={len(test)}"
        )

    if (
        len(train) == 0
        or not np.isfinite(train).all()
        or not np.isfinite(test).all()
    ):
        raise ValueError(f"Serie invalida: {name}")

    base = output_root / group / name
    base.mkdir(parents=True, exist_ok=True)

    # Datas sinteticas para compatibilidade com o projeto.
    dates = pd.date_range(
        "2000-01-01",
        periods=len(train) + h,
        freq=FREQ[group],
    )

    pd.DataFrame({
        "date": dates[:len(train)],
        "OT": train,
    }).to_csv(
        base / "Exchange.csv",
        index=False,
    )

    pd.DataFrame({
        "date": dates[len(train):],
        "OT": test,
    }).to_csv(
        base / "official_test.csv",
        index=False,
    )

    meta = {
        "series_id": name,
        "group": group,
        "train_length": len(train),
        "test_length": len(test),
        "horizon": h,
        "seasonality": PERIOD[group],
        "data_alias_for_legacy_scripts": "Exchange",
        "dates_are_synthetic": True,
        "warning": (
            "Exchange.csv contem apenas o historico. "
            "Os executores antigos usam splits internos."
        ),
    }

    (base / "metadata.json").write_text(
        json.dumps(
            meta,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return meta


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--group",
        choices=list(FREQ),
        default="monthly",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="0 = todas as series",
    )

    parser.add_argument(
        "--output-root",
        default="./data/M3",
    )

    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit nao pode ser negativo")

    try:
        from fcompdata import M3
    except ImportError as exc:
        raise SystemExit(
            "Instale: pip install fcompdata"
        ) from exc

    subset = M3.subset(args.group)
    results = []

    for i, series in enumerate(subset):
        if args.limit and i >= args.limit:
            break

        meta = export_series(
            series,
            args.group,
            Path(args.output_root),
        )

        results.append(meta)

        print(
            f"[OK] {meta['series_id']} | "
            f"Treino={meta['train_length']} | "
            f"Teste={meta['horizon']}"
        )

    if not results:
        raise RuntimeError("Nenhuma serie exportada")

    pd.DataFrame(results).to_csv(
        Path(args.output_root) / f"index_{args.group}.csv",
        index=False,
    )

    print(
        f"Exportadas {len(results)} series "
        f"da frequencia {args.group}."
    )


if __name__ == "__main__":
    main()
