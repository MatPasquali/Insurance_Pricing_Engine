"""Frequency model: Poisson GLM on claim counts with exposure as weight.

We model the claim *rate* (ClaimNb / Exposure) and pass Exposure as the sample
weight -- the standard scikit-learn formulation of an exposure offset for a
Poisson GLM. The GLM is interpretable by construction (log-linear coefficients),
which is exactly what a regulated pricing context wants; a gradient-boosted
challenger comes later in the XAI comparison.
"""
from __future__ import annotations

from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline

from .features import build_preprocessor


def build_frequency_pipeline(alpha: float = 1e-3, max_iter: int = 300) -> Pipeline:
    """Shared preprocessor -> Poisson GLM."""
    glm = PoissonRegressor(alpha=alpha, max_iter=max_iter)
    return Pipeline([("pre", build_preprocessor()), ("glm", glm)])
