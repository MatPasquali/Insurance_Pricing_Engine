"""v1: build the pure premium two ways and compare them.

  Pure premium = E[frequency] x E[severity]   (two GLMs multiplied)
               = E[loss / exposure]            (one Tweedie GLM)

Run:  python run_pure_premium.py

Both are scored on the same held-out test set with mean Tweedie deviance
(power=1.9) against a naive constant baseline, plus a portfolio-level
calibration check (total predicted loss / total actual loss -> should be ~1).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
from sklearn.metrics import mean_tweedie_deviance
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from src.pricing.data import CATEGORICAL, NUMERIC, load_modeling_frame  # noqa: E402
from src.pricing.frequency import build_frequency_pipeline  # noqa: E402
from src.pricing.pure_premium import build_tweedie_pipeline  # noqa: E402
from src.pricing.severity import build_severity_pipeline  # noqa: E402

POWER = 1.9
FEATURES = CATEGORICAL + NUMERIC


def main() -> None:
    df = load_modeling_frame()
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    df["PurePremium"] = df["ClaimAmount"] / df["Exposure"]

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # --- Frequency model (all training policies) ---
    freq = build_frequency_pipeline()
    freq.fit(train[FEATURES], train["Frequency"],
             glm__sample_weight=train["Exposure"])

    # --- Severity model (training policies that had a claim) ---
    has_claim = (train["ClaimNb"] > 0) & (train["ClaimAmount"] > 0)
    sev_train = train[has_claim].copy()
    sev_train["Severity"] = sev_train["ClaimAmount"] / sev_train["ClaimNb"]
    sev = build_severity_pipeline()
    sev.fit(sev_train[FEATURES], sev_train["Severity"],
            glm__sample_weight=sev_train["ClaimNb"])

    # --- Direct Tweedie model ---
    tweedie = build_tweedie_pipeline(power=POWER)
    tweedie.fit(train[FEATURES], train["PurePremium"],
                glm__sample_weight=train["Exposure"])

    # --- Predict pure premium on the test set ---
    pp_product = freq.predict(test[FEATURES]) * sev.predict(test[FEATURES])
    pp_tweedie = tweedie.predict(test[FEATURES])
    base = np.average(train["PurePremium"], weights=train["Exposure"])
    pp_base = np.full(len(test), base)

    y_true = test["PurePremium"].to_numpy()
    w = test["Exposure"].to_numpy()
    total_actual = float(test["ClaimAmount"].sum())

    def report(name: str, pred: np.ndarray) -> None:
        dev = mean_tweedie_deviance(y_true, pred, sample_weight=w, power=POWER)
        calib = float(np.sum(pred * w)) / total_actual
        print(f"{name:<28} | deviance {dev:>12.2f} | total pred/actual {calib:5.2f}")

    print("=" * 78)
    print("freMTPL2 - pure premium comparison (v1)   [mean Tweedie deviance, p=1.9]")
    print("=" * 78)
    print(f"Test policies: {len(test):,} | actual loss on test: {total_actual:,.0f} EUR")
    print("-" * 78)
    report("Naive mean", pp_base)
    report("Frequency x Severity", pp_product)
    report("Direct Tweedie", pp_tweedie)
    print("=" * 78)


if __name__ == "__main__":
    main()
