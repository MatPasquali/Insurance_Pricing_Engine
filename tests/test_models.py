"""Testes dos modelos v2 em dados sintéticos (rápidos, sem dependência de dados)."""
import numpy as np
import pandas as pd

from src.pricing.es_data import ES_CATEGORICAL, ES_NUMERIC
from src.pricing.es_models import (
    RETENTION_NUMERIC,
    build_cost_model,
    build_retention_model,
    optimize_premium,
)


def _synthetic(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {c: rng.normal(size=n) for c in ES_NUMERIC}
    data["Premium"] = rng.uniform(100, 500, n)
    for c in ES_CATEGORICAL:
        data[c] = rng.integers(0, 3, n)
    return pd.DataFrame(data)


def test_retencao_retorna_probabilidades_validas():
    df = _synthetic()
    y = (df["Premium"] < 300).astype(int)
    feats = RETENTION_NUMERIC + ES_CATEGORICAL
    model = build_retention_model().fit(df[feats], y)
    p = model.predict_proba(df[feats])[:, 1]
    assert p.shape == (len(df),)
    assert ((p >= 0) & (p <= 1)).all()


def test_custo_preve_nao_negativo():
    df = _synthetic()
    y = np.abs(np.random.default_rng(1).normal(150, 50, len(df)))
    feats = ES_NUMERIC + ES_CATEGORICAL
    model = build_cost_model().fit(df[feats], y)
    assert (model.predict(df[feats]) >= 0).all()


def test_otimizacao_respeita_a_grade():
    df = _synthetic()
    y = (df["Premium"] < 300).astype(int)
    feats = RETENTION_NUMERIC + ES_CATEGORICAL
    model = build_retention_model().fit(df[feats], y)
    cost = np.full(len(df), 100.0)
    res = optimize_premium(model, df[feats].copy(), df["Premium"], cost,
                           np.array([0.9, 1.0, 1.1]))
    assert {"optimal_price", "exp_profit", "p_renew"}.issubset(res.columns)
    assert (res["optimal_price"] >= df["Premium"] * 0.9 - 1e-6).all()
    assert (res["optimal_price"] <= df["Premium"] * 1.1 + 1e-6).all()
