"""Software local de pricing — Flask API + frontend próprio (sem Streamlit).

Reaproveita o motor `src/pricing`: treina os modelos uma vez, serve a página e
responde a /predict com:
  - prêmio (GLM/GBM) + severidade;
  - **decomposição do GLM** (base × fatores) — como o GLM chegou ao número;
  - explicação SHAP do GBM (PNG).

Rodar:  python webapp/app.py   ->  http://127.0.0.1:5000
Seguro para uso local: escuta só em 127.0.0.1, debug desligado.
"""
from __future__ import annotations

import base64
import io
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from flask import Flask, jsonify, render_template, request  # noqa: E402
from sklearn.preprocessing import OrdinalEncoder  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "webapp"))

from content import CONCEITOS, VARIABLES  # noqa: E402
from src.pricing.data import CATEGORICAL, NUMERIC, load_modeling_frame  # noqa: E402
from src.pricing.explain import glm_contributions  # noqa: E402
from src.pricing.frequency import build_frequency_pipeline  # noqa: E402
from src.pricing.gbm import build_frequency_gbm  # noqa: E402
from src.pricing.severity import build_severity_pipeline  # noqa: E402

APP_VERSION = "0.3.0"
REPO_URL = "https://github.com/MatPasquali/Insurance_Pricing_Engine"
FEATURES = CATEGORICAL + NUMERIC

app = Flask(__name__)
_STATE: dict = {}


def get_state() -> dict:
    """Treina (uma vez) os modelos e prepara as opções do formulário."""
    if _STATE:
        return _STATE

    df = load_modeling_frame()
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]

    enc = OrdinalEncoder().fit(df[CATEGORICAL])
    X = df[FEATURES].copy()
    X[CATEGORICAL] = enc.transform(df[CATEGORICAL])

    glm_f = build_frequency_pipeline().fit(X, df["Frequency"], glm__sample_weight=df["Exposure"])
    gbm = build_frequency_gbm().fit(X, df["Frequency"], sample_weight=df["Exposure"])
    mask = (df["ClaimNb"] > 0) & (df["ClaimAmount"] > 0)
    sev_y = df.loc[mask, "ClaimAmount"] / df.loc[mask, "ClaimNb"]
    glm_s = build_severity_pipeline().fit(X.loc[mask], sev_y, glm__sample_weight=df.loc[mask, "ClaimNb"])

    # Opções do form: rótulo original (limpo) -> código ordinal (o frontend manda o código).
    cats = {c: [{"label": str(v).strip("'\""), "code": float(j)} for j, v in enumerate(enc.categories_[i])]
            for i, c in enumerate(CATEGORICAL)}
    ranges = {c: {"min": float(df[c].min()), "max": float(df[c].max()), "med": float(df[c].median())}
              for c in NUMERIC}

    _STATE.update(glm_f=glm_f, gbm=gbm, glm_s=glm_s, explainer=shap.TreeExplainer(gbm),
                  cats=cats, ranges=ranges)
    return _STATE


@app.route("/")
def index():
    s = get_state()
    return render_template(
        "index.html", categorical=CATEGORICAL, numeric=NUMERIC,
        cats=s["cats"], ranges=s["ranges"], variables=VARIABLES, conceitos=CONCEITOS,
        repo_url=REPO_URL, version=APP_VERSION,
    )


@app.route("/health")
def health():
    return jsonify(status="ok", version=APP_VERSION)


@app.route("/predict", methods=["POST"])
def predict():
    s = get_state()
    payload = request.get_json(force=True)
    X = pd.DataFrame([{c: float(payload[c]) for c in FEATURES}])[FEATURES]

    freq_glm = float(s["glm_f"].predict(X)[0])
    freq_gbm = float(s["gbm"].predict(X)[0])
    sev = float(s["glm_s"].predict(X)[0])
    decomp = glm_contributions(s["glm_f"], X)

    sv = s["explainer"](X)
    plt.figure()
    shap.plots.waterfall(sv[0], show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close("all")
    shap_png = base64.b64encode(buf.getvalue()).decode()

    return jsonify(
        freq_glm=freq_glm, freq_gbm=freq_gbm, severity=sev,
        premium_glm=freq_glm * sev, premium_gbm=freq_gbm * sev,
        glm_base=decomp["base"], glm_fatores=decomp["fatores"], glm_previsto=decomp["previsto"],
        shap_png=shap_png,
    )


if __name__ == "__main__":
    print("Treinando os modelos (uma vez, ~1 min)...")
    get_state()
    print("Pronto -> http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)
