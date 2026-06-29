"""Demo Streamlit do motor de pricing: perfil -> prêmio puro + explicação SHAP.

Rodar:  streamlit run streamlit_app.py

Treina (em cache) os modelos sobre o snapshot de data/processed e permite simular
o prêmio puro de um perfil, comparando GLM vs GBM e explicando a previsão do GBM
com SHAP. Pré-requisito: `python run_etl.py` (gera data/processed/).
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.preprocessing import OrdinalEncoder

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pricing.data import CATEGORICAL, NUMERIC  # noqa: E402
from src.pricing.frequency import build_frequency_pipeline  # noqa: E402
from src.pricing.gbm import build_frequency_gbm  # noqa: E402
from src.pricing.severity import build_severity_pipeline  # noqa: E402

FEATURES = CATEGORICAL + NUMERIC
INT_FEATURES = {"DrivAge", "VehAge", "VehPower", "BonusMalus"}


@st.cache_resource(show_spinner="Treinando os modelos (uma vez)...")
def load_and_train():
    proc = ROOT / "data" / "processed"
    pq, cs = proc / "freMTPL2_modeling.parquet", proc / "freMTPL2_modeling.csv"
    if not pq.exists() and not cs.exists():
        return None
    df = pd.read_parquet(pq) if pq.exists() else pd.read_csv(cs)

    enc = OrdinalEncoder().fit(df[CATEGORICAL])
    X = df[FEATURES].copy()
    X[CATEGORICAL] = enc.transform(df[CATEGORICAL])

    glm_f = build_frequency_pipeline().fit(X, df["Frequency"], glm__sample_weight=df["Exposure"])
    gbm = build_frequency_gbm().fit(X, df["Frequency"], sample_weight=df["Exposure"])

    m = (df["ClaimNb"] > 0) & (df["ClaimAmount"] > 0)
    sev_y = df.loc[m, "ClaimAmount"] / df.loc[m, "ClaimNb"]
    glm_s = build_severity_pipeline().fit(X.loc[m], sev_y, glm__sample_weight=df.loc[m, "ClaimNb"])
    return df, enc, glm_f, gbm, glm_s


@st.cache_resource(show_spinner=False)
def get_explainer(_gbm):
    import shap

    return shap.TreeExplainer(_gbm)


def main() -> None:
    st.set_page_config(page_title="Insurance Pricing Engine", page_icon="🚗", layout="wide")
    st.title("🚗 Motor de Pricing de Seguros — demo")
    st.caption("GLM (frequência × severidade) + GBM, com explicação SHAP. Base: freMTPL2.")

    trained = load_and_train()
    if trained is None:
        st.error("data/processed/ não encontrado. Rode antes: `python run_etl.py`.")
        st.stop()
    df, enc, glm_f, gbm, glm_s = trained

    st.sidebar.header("Perfil do segurado")
    inp: dict = {}
    for c in NUMERIC:
        lo, hi, med = float(df[c].min()), float(df[c].max()), float(df[c].median())
        if c in INT_FEATURES:
            inp[c] = st.sidebar.slider(c, int(lo), int(hi), int(med))
        else:
            inp[c] = st.sidebar.slider(c, lo, hi, med)
    for c in CATEGORICAL:
        opts = sorted(df[c].dropna().unique().tolist())
        inp[c] = st.sidebar.selectbox(c, opts)

    Xrow = pd.DataFrame([inp])[FEATURES]
    Xrow[CATEGORICAL] = enc.transform(Xrow[CATEGORICAL])

    freq_glm = float(glm_f.predict(Xrow)[0])
    freq_gbm = float(gbm.predict(Xrow)[0])
    sev = float(glm_s.predict(Xrow)[0])

    st.subheader("Prêmio puro estimado")
    c1, c2, c3 = st.columns(3)
    c1.metric("Frequência (GLM)", f"{freq_glm:.3f}/ano")
    c2.metric("Frequência (GBM)", f"{freq_gbm:.3f}/ano")
    c3.metric("Severidade média", f"{sev:,.0f} EUR")
    c1.metric("Prêmio puro — GLM", f"{freq_glm * sev:,.2f} EUR")
    c2.metric("Prêmio puro — GBM", f"{freq_gbm * sev:,.2f} EUR")
    c3.caption("Prêmio puro = frequência × severidade (custo esperado/ano).")

    st.subheader("Por que esse prêmio? — SHAP (frequência GBM)")
    sv = get_explainer(gbm)(Xrow)
    import shap

    shap.plots.waterfall(sv[0], show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    st.caption(
        "Cada barra mostra quanto a feature empurra a frequência prevista para cima/baixo. "
        "Categóricas aparecem como código ordinal."
    )


if __name__ == "__main__":
    main()
