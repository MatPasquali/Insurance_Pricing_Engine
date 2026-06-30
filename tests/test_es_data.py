"""Testes da camada de dados ES (pulados se o CSV bruto não estiver presente)."""
import pytest

from src.pricing.es_data import RAW, build_modeling_frame

pytestmark = pytest.mark.skipif(not RAW.exists(), reason="base ES bruta ausente (data/raw)")


def test_alvo_de_renovacao_e_taxa_plausivel():
    df = build_modeling_frame()
    assert set(df["renewed"].unique()) <= {0, 1}
    obs = df[df["observed_outcome"]]
    assert 0.6 < obs["renewed"].mean() < 0.95          # retenção realista
    assert df["DrivAge"].between(16, 100).mean() > 0.99  # idades sãs
