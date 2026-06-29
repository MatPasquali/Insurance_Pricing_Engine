"""Shared feature preprocessing for every GLM in the engine.

Keeping a single preprocessor guarantees the frequency, severity and Tweedie
models all see identically encoded features -- important when their outputs are
multiplied together into a pure premium.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode categoricals, standardize numerics."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
