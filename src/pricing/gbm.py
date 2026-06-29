"""GBM challenger for frequency: gradient boosting with a Poisson loss.

This is the "more accurate but opaque" counterpart to the GLM. We use
scikit-learn's HistGradientBoostingRegressor with native categorical support,
so it captures non-linearities and interactions the linear GLM cannot -- the
exact gap the SHAP layer then makes visible.

The feature matrix is expected with columns ordered CATEGORICAL + NUMERIC, the
categoricals ordinal-encoded to non-negative integer codes (see the runner).
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor

from .data import CATEGORICAL, NUMERIC

# Boolean mask matching the CATEGORICAL + NUMERIC column order.
CATEGORICAL_MASK = [True] * len(CATEGORICAL) + [False] * len(NUMERIC)


def build_frequency_gbm(**kwargs) -> HistGradientBoostingRegressor:
    """Poisson-loss histogram gradient boosting with sensible anti-overfit defaults."""
    params = dict(
        loss="poisson",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        categorical_features=CATEGORICAL_MASK,
    )
    params.update(kwargs)
    return HistGradientBoostingRegressor(**params)
