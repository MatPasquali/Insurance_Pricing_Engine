"""v3 (MLOps): relatório de drift (PSI) entre períodos.

Compara a distribuição das features (e do score do modelo) entre uma base de
**referência** (renovações de 2016) e uma base **nova** (2017) da carteira ES.
Em produção, é esse sinal que dispara "o mundo mudou, revise o modelo".

Run:  python run_drift_report.py
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pricing.es_data import ES_CATEGORICAL, ES_NUMERIC, build_modeling_frame  # noqa: E402
from src.pricing.es_models import RETENTION_NUMERIC, build_retention_model  # noqa: E402
from src.pricing.monitoring import classify, drift_report, psi_numeric  # noqa: E402


def main() -> None:
    df = build_modeling_frame()
    ref = df[df["renewal_year"] == 2016]
    cur = df[df["renewal_year"] == 2017]

    print("=" * 64)
    print("v3 - Relatorio de drift (PSI): referencia 2016 -> atual 2017")
    print("=" * 64)
    print(f"referencia: {len(ref):,} apolices | atual: {len(cur):,}")

    numeric = ES_NUMERIC + ["Premium"]
    rep = drift_report(ref, cur, numeric, ES_CATEGORICAL)
    print("-" * 64)
    print(rep.to_string(index=False))

    # Drift do SCORE do modelo (retencao treinada ate 2016).
    obs = df[df["observed_outcome"]]
    tr = obs[obs["renewal_year"] <= 2016]
    feats = RETENTION_NUMERIC + ES_CATEGORICAL
    ret = build_retention_model().fit(tr[feats], tr["renewed"])
    s_ref = ret.predict_proba(ref[feats])[:, 1]
    s_cur = ret.predict_proba(cur[feats])[:, 1]
    score_psi = psi_numeric(s_ref, s_cur)

    n_drift = int((rep["status"] == "drift").sum())
    n_mod = int((rep["status"] == "moderado").sum())
    print("-" * 64)
    print(f"PSI do score de retencao: {score_psi:.4f} -> {classify(score_psi)}")
    print(f"Resumo features: {n_drift} em drift, {n_mod} moderado, "
          f"{len(rep) - n_drift - n_mod} estavel")
    print("=" * 64)


if __name__ == "__main__":
    main()
