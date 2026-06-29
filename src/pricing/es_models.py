"""Modelos do v2 (base ES): custo, retenção e otimização de prêmio.

- Custo: GLM Tweedie sobre o custo de sinistros do ano (o "piso" de cada apólice).
- Retenção: classificador de P(renovar | preço, perfil) — o `Premium` é uma feature,
  então o modelo aprende a sensibilidade da renovação ao preço (elasticidade).
- Otimização: para cada apólice, escolhe o prêmio que maximiza o lucro esperado
  E[lucro] = P(renovar | preço) × (preço − custo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .es_data import ES_CATEGORICAL, ES_NUMERIC

# A retenção inclui o preço (Premium) como alavanca.
RETENTION_NUMERIC = ES_NUMERIC + ["Premium"]


def _preprocessor(numeric, categorical) -> ColumnTransformer:
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    return ColumnTransformer(
        [("num", num, numeric), ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)]
    )


def build_cost_model() -> Pipeline:
    """Custo esperado por GBM (loss Poisson).

    Trocamos o GLM Tweedie log-link por um GBM Poisson porque, nesta base, os
    outliers de sinistro (até ~260k EUR) faziam o log-link **explodir** (previsões
    de 10^x). O GBM é robusto a outliers. Treine sobre o custo **capado/winsorizado**
    para conter a cauda — prática atuarial padrão.
    """
    gbr = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
        min_samples_leaf=200, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, random_state=42,
    )
    return Pipeline([("pre", _preprocessor(ES_NUMERIC, ES_CATEGORICAL)), ("gbr", gbr)])


def build_retention_model(C: float = 1.0) -> Pipeline:
    """Regressão logística de P(renovar | preço, perfil) — interpretável."""
    clf = LogisticRegression(C=C, max_iter=2000)
    return Pipeline([("pre", _preprocessor(RETENTION_NUMERIC, ES_CATEGORICAL)), ("clf", clf)])


def build_retention_gbm() -> Pipeline:
    """Desafiante GBM para a retenção (capta não-linearidades preço×perfil)."""
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
        min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, random_state=42,
    )
    return Pipeline([("pre", _preprocessor(RETENTION_NUMERIC, ES_CATEGORICAL)), ("clf", clf)])


def optimize_premium(
    retention_model: Pipeline,
    X: pd.DataFrame,
    current_premium: pd.Series,
    cost: np.ndarray,
    multipliers: np.ndarray,
    min_retention: float | None = None,
):
    """Escolhe, por apólice, o preço que maximiza E[lucro] = P(renovar)·(preço − custo).

    `X` deve conter as features de retenção (inclusive `Premium`, que é sobrescrito por
    cada multiplicador). Se `min_retention` for dado, só considera preços cujo P(renovar)
    fique acima do piso (restrição de retenção).
    """
    cur = current_premium.to_numpy(dtype=float)
    best_profit = np.full(len(X), -np.inf)
    best_price = cur.copy()
    best_pret = np.zeros(len(X))

    for m in multipliers:
        Xm = X.copy()
        price = cur * m
        Xm["Premium"] = price
        p_ret = retention_model.predict_proba(Xm)[:, 1]
        profit = p_ret * (price - cost)
        feasible = profit > best_profit
        if min_retention is not None:
            feasible &= p_ret >= min_retention
        best_profit = np.where(feasible, profit, best_profit)
        best_price = np.where(feasible, price, best_price)
        best_pret = np.where(feasible, p_ret, best_pret)

    return pd.DataFrame(
        {"optimal_price": best_price, "exp_profit": best_profit, "p_renew": best_pret},
        index=X.index,
    )
