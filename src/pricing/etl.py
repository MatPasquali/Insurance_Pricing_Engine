"""ETL: raw -> processed (a camada silver/gold do projeto).

Materializa um snapshot estavel e pronto-para-analise em data/processed/, para
que notebooks e modelos carreguem uma tabela fixa em vez de recalcular tudo a
partir do download cru do OpenML.

Proveniencia: freMTPL2 (French Motor Third-Party Liability), OpenML 41214
(freq, uma linha por apolice) + 41215 (sev, uma linha por sinistro), unidos por
IDpol. Limpeza herdada de data.py (ClaimNb<=4, Exposure<=1, exposicao>0).

Run:  python run_etl.py
"""
from __future__ import annotations

import pathlib

import pandas as pd

from .data import load_frequency, load_modeling_frame


def _save(df: pd.DataFrame, path_noext: pathlib.Path) -> pathlib.Path:
    """Salva em parquet (preserva dtypes); cai para CSV se faltar pyarrow."""
    try:
        import pyarrow  # noqa: F401

        out = path_noext.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    except Exception:
        out = path_noext.with_suffix(".csv")
        df.to_csv(out, index=False)
    return out


def build_processed(out_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    """Constroi e grava os snapshots processados; retorna os caminhos."""
    out_dir.mkdir(parents=True, exist_ok=True)

    freq = load_frequency()

    model = load_modeling_frame()
    model["Frequency"] = model["ClaimNb"] / model["Exposure"]
    model["PurePremium"] = model["ClaimAmount"] / model["Exposure"]

    return {
        "frequency": _save(freq, out_dir / "freMTPL2_frequency"),
        "modeling_frame": _save(model, out_dir / "freMTPL2_modeling"),
    }
