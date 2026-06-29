"""Direct pure-premium model: a single Tweedie GLM.

The Tweedie distribution with a power between 1 and 2 is a compound
Poisson-Gamma: it handles the mass of zero-claim policies and the positive,
right-skewed losses in one model -- the modern alternative to multiplying a
separate frequency and severity model. We keep both approaches so the README
can compare them head to head.
"""
from __future__ import annotations

from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import Pipeline

from .features import build_preprocessor


def build_tweedie_pipeline(
    power: float = 1.9, alpha: float = 1e-2, max_iter: int = 300
) -> Pipeline:
    """Shared preprocessor -> Tweedie GLM (log link)."""
    glm = TweedieRegressor(power=power, alpha=alpha, max_iter=max_iter, link="log")
    return Pipeline([("pre", build_preprocessor()), ("glm", glm)])
