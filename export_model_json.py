"""Exporta os GLMs (frequência e severidade) para JSON.

Alimenta a calculadora client-side do site (docs/). Um GLM é linear:
freq = exp(intercepto + sum coef*x). Logo dá para reproduzir a previsão **exata**
em JavaScript, sem backend, perfeito para o GitHub Pages (estático).

Run:  python export_model_json.py  ->  docs/assets/model.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from sklearn.preprocessing import OrdinalEncoder

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pricing.data import CATEGORICAL, NUMERIC, load_modeling_frame  # noqa: E402
from src.pricing.frequency import build_frequency_pipeline  # noqa: E402
from src.pricing.severity import build_severity_pipeline  # noqa: E402


def _export_glm(pipe) -> dict:
    pre = pipe.named_steps["pre"]
    reg = pipe.named_steps["glm"]
    names = pre.get_feature_names_out()
    coef = np.ravel(reg.coef_)
    scaler = pre.named_transformers_["num"]
    numeric = {f: {"coef": 0.0, "mean": float(scaler.mean_[i]), "scale": float(scaler.scale_[i])}
               for i, f in enumerate(NUMERIC)}
    categorical: dict[str, dict] = {c: {} for c in CATEGORICAL}
    for name, c in zip(names, coef):
        if name.startswith("num__"):
            numeric[name[5:]]["coef"] = float(c)
        else:
            feat, level = name[5:].rsplit("_", 1)
            categorical[feat][str(int(float(level)))] = float(c)
    return {"intercept": float(np.ravel(reg.intercept_)[0]), "numeric": numeric, "categorical": categorical}


def _predict_json(glm: dict, row: dict) -> float:
    lp = glm["intercept"]
    for f, p in glm["numeric"].items():
        lp += p["coef"] * ((row[f] - p["mean"]) / p["scale"])
    for f, byc in glm["categorical"].items():
        lp += byc.get(str(int(row[f])), 0.0)
    return float(np.exp(lp))


def main() -> None:
    df = load_modeling_frame()
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    enc = OrdinalEncoder().fit(df[CATEGORICAL])
    X = df[CATEGORICAL + NUMERIC].copy()
    X[CATEGORICAL] = enc.transform(df[CATEGORICAL])

    glm_f = build_frequency_pipeline().fit(X, df["Frequency"], glm__sample_weight=df["Exposure"])
    mask = (df["ClaimNb"] > 0) & (df["ClaimAmount"] > 0)
    sev_y = df.loc[mask, "ClaimAmount"] / df.loc[mask, "ClaimNb"]
    glm_s = build_severity_pipeline().fit(X.loc[mask], sev_y, glm__sample_weight=df.loc[mask, "ClaimNb"])

    options = {c: [{"label": str(v).strip("'\""), "code": int(j)} for j, v in enumerate(enc.categories_[i])]
               for i, c in enumerate(CATEGORICAL)}
    ranges = {c: {"min": float(df[c].min()), "max": float(df[c].max()), "med": float(df[c].median())}
              for c in NUMERIC}
    model = {"categorical": CATEGORICAL, "numeric": NUMERIC, "options": options, "ranges": ranges,
             "freq": _export_glm(glm_f), "sev": _export_glm(glm_s)}

    out = ROOT / "docs" / "assets" / "model.json"
    out.write_text(json.dumps(model), encoding="utf-8")

    row = X[CATEGORICAL + NUMERIC].iloc[0].to_dict()
    f_json, f_sk = _predict_json(model["freq"], row), float(glm_f.predict(X.iloc[[0]])[0])
    s_json, s_sk = _predict_json(model["sev"], row), float(glm_s.predict(X.iloc[[0]])[0])
    print(f"VALIDACAO freq: json {f_json:.6f} vs sklearn {f_sk:.6f} (diff {abs(f_json-f_sk):.2e})")
    print(f"VALIDACAO sev : json {s_json:.2f} vs sklearn {s_sk:.2f} (diff {abs(s_json-s_sk):.2e})")
    print(f"escrito: {out.relative_to(ROOT)} ({out.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
