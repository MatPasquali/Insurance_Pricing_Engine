"""Explicabilidade do GLM: decomposição da previsão em relatividades.

Um GLM Poisson com link log prevê:  freq = exp(intercepto + Σ coef·x).
Logo a previsão é um **produto de fatores**:

    freq = base × Π fator_i ,  com  base = exp(intercepto)  e  fator_i = exp(contribuição_i)

`base` é a frequência média da carteira (o "ponto de partida"); cada `fator_i` é o
quanto a variável i multiplica esse risco (>1 encarece, <1 barateia). É isso que
torna o GLM **auditável** — diferente de uma caixa-preta, cada variável vira um
multiplicador explícito. Esta função expõe essa decomposição para uma apólice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _feature_de(nome: str) -> str:
    """'cat__VehBrand_3.0' -> 'VehBrand' ; 'num__DrivAge' -> 'DrivAge'."""
    corpo = nome.split("__", 1)[1] if "__" in nome else nome
    if nome.startswith("num__"):
        return corpo
    return corpo.rsplit("_", 1)[0]  # remove o sufixo do nível (one-hot)


def glm_contributions(pipeline, X_row: pd.DataFrame) -> dict:
    """Decompõe a previsão de UMA apólice nos fatores multiplicativos por variável.

    Espera um Pipeline ``[('pre', ColumnTransformer), ('glm', PoissonRegressor)]``.
    Retorna ``{"base", "fatores": {var: fator}, "previsto"}`` com a garantia
    base × Π fatores == previsto (== pipeline.predict).
    """
    pre = pipeline.named_steps["pre"]
    reg = pipeline.named_steps["glm"]

    xt = pre.transform(X_row)
    if hasattr(xt, "toarray"):  # ColumnTransformer pode devolver matriz esparsa
        xt = xt.toarray()
    xt = np.asarray(xt).ravel()
    nomes = pre.get_feature_names_out()
    coef = np.asarray(reg.coef_).ravel()
    intercepto = float(np.ravel(reg.intercept_)[0])

    contrib: dict[str, float] = {}
    for nome, valor, c in zip(nomes, xt, coef):
        feat = _feature_de(nome)
        contrib[feat] = contrib.get(feat, 0.0) + float(c) * float(valor)

    base = float(np.exp(intercepto))
    fatores = {f: float(np.exp(v)) for f, v in contrib.items()}
    previsto = base
    for v in fatores.values():
        previsto *= v
    return {"base": base, "fatores": fatores, "previsto": previsto}
