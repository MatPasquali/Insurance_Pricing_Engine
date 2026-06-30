"""Testes do monitoramento de drift (PSI) — puros, sem dados externos."""
import numpy as np

from src.pricing.monitoring import classify, psi_categorical, psi_numeric


def test_psi_zero_para_mesma_distribuicao():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert psi_numeric(x, x) < 0.01


def test_psi_detecta_shift_numerico():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(2, 1, 5000)  # média deslocada -> drift
    assert psi_numeric(a, b) > 0.25


def test_psi_detecta_shift_categorico():
    a = ["x"] * 900 + ["y"] * 100
    b = ["x"] * 500 + ["y"] * 500
    assert psi_categorical(a, b) > 0.1


def test_classify_faixas():
    assert classify(0.05) == "estavel"
    assert classify(0.15) == "moderado"
    assert classify(0.30) == "drift"
