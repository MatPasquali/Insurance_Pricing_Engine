"""ES motor portfolio (v2): painel 2015-2018, uma linha por apólice-ano.

Diferente do freMTPL2, esta base tem o **preço cobrado** (`Premium`), renovações e
lapsos — habilitando elasticidade-preço da retenção e otimização de prêmio.

Derivações-chave:
  - `renewed`: a apólice-ano "renova" se o mesmo `ID` reaparece no ciclo seguinte
    (o `Date_next_renewal` casa com um `Date_last_renewal` posterior do mesmo ID).
  - `observed_outcome`: desfecho é censurado para a última coorte (renovação cairia
    fora da janela de dados) -> excluímos da modelagem de retenção.
"""
from __future__ import annotations

import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "es_portfolio_v2" / "Motor vehicle insurance data.csv"

DATE_COLS = [
    "Date_start_contract", "Date_last_renewal", "Date_next_renewal",
    "Date_birth", "Date_driving_licence", "Date_lapse",
]

# Features de risco/apólice/veículo usadas nos modelos.
ES_NUMERIC = [
    "DrivAge", "LicenseSeniority", "VehicleAge", "Power", "Cylinder_capacity",
    "Value_vehicle", "N_doors", "Weight", "Seniority", "N_claims_history",
    "R_Claims_history", "Policies_in_force",
]
ES_CATEGORICAL = [
    "Type_risk", "Area", "Type_fuel", "Payment", "Distribution_channel", "Second_driver",
]


def load_raw() -> pd.DataFrame:
    """Carrega o CSV (sep ';' ou ',') e parseia as datas (DD/MM/YYYY)."""
    df = pd.read_csv(RAW, sep=";", encoding="latin-1")
    if df.shape[1] == 1:
        df = pd.read_csv(RAW, sep=",", encoding="latin-1")
    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c], format="%d/%m/%Y", errors="coerce")
    return df


def build_modeling_frame() -> pd.DataFrame:
    """Frame de modelagem: features derivadas + alvo de renovação + censura."""
    df = load_raw()

    df["renewal_year"] = df["Date_last_renewal"].dt.year
    df["DrivAge"] = (df["Date_last_renewal"] - df["Date_birth"]).dt.days / 365.25
    df["LicenseSeniority"] = (df["Date_last_renewal"] - df["Date_driving_licence"]).dt.days / 365.25
    df["VehicleAge"] = (df["renewal_year"] - df["Year_matriculation"]).clip(lower=0)

    # Alvo de retenção via continuação do painel.
    keys = set(zip(df["ID"], df["Date_last_renewal"]))
    df["renewed"] = [int((i, d) in keys) for i, d in zip(df["ID"], df["Date_next_renewal"])]
    max_obs = df["Date_last_renewal"].max()
    df["observed_outcome"] = df["Date_next_renewal"] <= max_obs

    # Custo de sinistros do ano (alvo do modelo de custo; muitos zeros + cauda -> Tweedie).
    df["LossCost"] = df["Cost_claims_year"].clip(lower=0)
    return df
