"""Data loaders for the insurance pricing engine.

Primary source: **freMTPL2** (French Motor Third-Party Liability), the canonical
actuarial pricing dataset. We load it via OpenML so it works with **zero Kaggle
authentication**; the Kaggle mirror
(`karansarpal/fremtpl2-french-motor-tpl-insurance-claims`) is byte-for-byte the
same data if you prefer a local CSV.

Two tables:
- freMTPL2freq (OpenML 41214): one row per policy -> Exposure, ClaimNb, risk features.
- freMTPL2sev  (OpenML 41215): one row per claim  -> ClaimAmount, joined by IDpol.
"""
from __future__ import annotations

import pandas as pd
from sklearn.datasets import fetch_openml

FREQ_OPENML_ID = 41214  # freMTPL2freq
SEV_OPENML_ID = 41215   # freMTPL2sev

CATEGORICAL = ["VehBrand", "VehGas", "Area", "Region"]
NUMERIC = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]


def load_frequency() -> pd.DataFrame:
    """Policy-level frequency table, lightly cleaned.

    Cleaning mirrors the scikit-learn Tweedie example: claim counts are capped
    at 4 and exposure at 1 year to tame a handful of outliers, and policies with
    zero exposure are dropped (they carry no information for a rate model).
    """
    df = fetch_openml(data_id=FREQ_OPENML_ID, as_frame=True).frame.copy()
    df["IDpol"] = df["IDpol"].astype("int64")
    for col in ["ClaimNb", "Exposure", "VehPower", "VehAge", "DrivAge",
                "BonusMalus", "Density"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(upper=1)
    df = df[df["Exposure"] > 0].reset_index(drop=True)
    return df


def load_severity() -> pd.DataFrame:
    """Claim-level severity table (one row per claim)."""
    df = fetch_openml(data_id=SEV_OPENML_ID, as_frame=True).frame.copy()
    df["IDpol"] = df["IDpol"].astype("int64")
    df["ClaimAmount"] = pd.to_numeric(df["ClaimAmount"], errors="coerce")
    return df


def load_modeling_frame() -> pd.DataFrame:
    """Frequency frame enriched with total claim amount per policy.

    Used downstream for severity (ClaimAmount / ClaimNb on policies with claims)
    and pure-premium / Tweedie modelling.
    """
    freq = load_frequency()
    sev = load_severity()
    sev_by_pol = sev.groupby("IDpol")["ClaimAmount"].sum().rename("ClaimAmount")
    df = freq.merge(sev_by_pol, on="IDpol", how="left")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0.0)
    return df
