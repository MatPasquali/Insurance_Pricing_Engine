"""v1 baseline: fit the Poisson frequency GLM and compare it to a naive mean.

Run:  python run_frequency_baseline.py

Reports mean Poisson deviance on a held-out test set for (a) the GLM and
(b) a constant model that predicts the portfolio's average frequency. A lower
deviance is better; the GLM should beat the naive baseline, and the gap is the
first concrete signal that the risk features carry pricing signal.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from src.pricing.data import CATEGORICAL, NUMERIC, load_frequency  # noqa: E402
from src.pricing.frequency import build_frequency_pipeline  # noqa: E402


def main() -> None:
    df = load_frequency()
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]

    X = df[CATEGORICAL + NUMERIC]
    y = df["Frequency"]
    w = df["Exposure"]

    X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
        X, y, w, test_size=0.2, random_state=42
    )

    pipe = build_frequency_pipeline()
    pipe.fit(X_tr, y_tr, glm__sample_weight=w_tr)

    pred = pipe.predict(X_te)
    base_rate = np.average(y_tr, weights=w_tr)
    base_pred = np.full(shape=y_te.shape, fill_value=base_rate)

    dev_model = mean_poisson_deviance(y_te, pred, sample_weight=w_te)
    dev_base = mean_poisson_deviance(y_te, base_pred, sample_weight=w_te)
    lift = (dev_base - dev_model) / dev_base * 100

    print("=" * 60)
    print("freMTPL2 - Poisson frequency GLM (v1 baseline)")
    print("=" * 60)
    print(f"Policies (after cleaning): {len(df):,}")
    print(f"Exposure-weighted mean frequency: {base_rate:.4f} claims/year")
    print("-" * 60)
    print(f"Mean Poisson deviance | naive mean : {dev_base:.5f}")
    print(f"Mean Poisson deviance | GLM        : {dev_model:.5f}")
    print(f"Deviance reduction vs naive        : {lift:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
