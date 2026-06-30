"""Testa a decomposição do GLM: base × fatores deve reconstruir a previsão."""
import numpy as np
import pandas as pd

from src.pricing.data import CATEGORICAL, NUMERIC
from src.pricing.explain import glm_contributions
from src.pricing.frequency import build_frequency_pipeline


def _synthetic(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {c: rng.normal(size=n) for c in NUMERIC}
    for c in CATEGORICAL:
        data[c] = rng.integers(0, 3, n).astype(float)
    return pd.DataFrame(data)[CATEGORICAL + NUMERIC]


def test_decomposicao_reconstroi_a_previsao():
    rng = np.random.default_rng(1)
    X = _synthetic()
    y = rng.poisson(0.1, len(X)).astype(float)
    glm = build_frequency_pipeline().fit(X, y, glm__sample_weight=np.ones(len(X)))

    row = X.iloc[[0]]
    d = glm_contributions(glm, row)
    pred = float(glm.predict(row)[0])

    assert abs(d["previsto"] - pred) <= 1e-6 * pred + 1e-9   # base x fatores == predict
    assert d["base"] > 0
    assert all(v > 0 for v in d["fatores"].values())
    assert set(d["fatores"]) == set(CATEGORICAL + NUMERIC)   # um fator por variável
