"""Severity model: Gamma GLM on average claim cost.

Fitted only on policies that actually had a claim, with target
(ClaimAmount / ClaimNb) = average cost per claim, weighted by ClaimNb. The Gamma
distribution captures the right-skewed, strictly-positive nature of claim costs.
"""
from __future__ import annotations

from sklearn.linear_model import GammaRegressor
from sklearn.pipeline import Pipeline

from .features import build_preprocessor


def build_severity_pipeline(alpha: float = 1e-2, max_iter: int = 300) -> Pipeline:
    """Shared preprocessor -> Gamma GLM."""
    glm = GammaRegressor(alpha=alpha, max_iter=max_iter)
    return Pipeline([("pre", build_preprocessor()), ("glm", glm)])
