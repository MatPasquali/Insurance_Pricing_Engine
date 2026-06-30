"""Monitoramento de drift via PSI (Population Stability Index).

O PSI mede o quanto a distribuição de uma variável mudou entre uma base de
**referência** (ex.: treino) e uma base **nova** (ex.: produção/novo período).
É o KPI de MLOps que dispara o alerta de "o mundo mudou, revise o modelo".

Regra de bolso (indústria):
  - PSI < 0,10  -> estável
  - 0,10–0,25   -> mudança moderada (observar)
  - PSI > 0,25  -> drift relevante (agir)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ESTAVEL, MODERADO = 0.10, 0.25


def psi_numeric(expected, actual, bins: int = 10) -> float:
    """PSI de uma variável numérica, usando faixas por quantis da referência."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(expected, actual) -> float:
    """PSI de uma variável categórica (sobre as proporções de cada categoria)."""
    e = pd.Series(expected).value_counts(normalize=True)
    a = pd.Series(actual).value_counts(normalize=True)
    cats = e.index.union(a.index)
    e = np.clip(e.reindex(cats, fill_value=0.0).to_numpy(), 1e-6, None)
    a = np.clip(a.reindex(cats, fill_value=0.0).to_numpy(), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def classify(psi_value: float) -> str:
    if psi_value < ESTAVEL:
        return "estavel"
    if psi_value < MODERADO:
        return "moderado"
    return "drift"


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    bins: int = 10,
) -> pd.DataFrame:
    """Relatório de PSI por feature, ordenado do maior drift para o menor."""
    rows = []
    for c in numeric:
        rows.append({"feature": c, "tipo": "num",
                     "psi": psi_numeric(reference[c].dropna(), current[c].dropna(), bins)})
    for c in categorical:
        rows.append({"feature": c, "tipo": "cat",
                     "psi": psi_categorical(reference[c], current[c])})
    rep = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    rep["status"] = rep["psi"].map(classify)
    rep["psi"] = rep["psi"].round(4)
    return rep
