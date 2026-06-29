"""Roda o ETL: grava os snapshots processados em data/processed/.

Run:  python run_etl.py
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pricing.etl import build_processed  # noqa: E402


def main() -> None:
    out = ROOT / "data" / "processed"
    paths = build_processed(out)
    print("ETL concluido. Snapshots em data/processed/:")
    for name, p in paths.items():
        print(f"  {name:<16} {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
