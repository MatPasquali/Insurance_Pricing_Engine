"""v1 XAI: GLM (interpretable) vs GBM (accurate), explained with SHAP.

Run:  python run_gbm_vs_glm.py

1. Fits the Poisson frequency GLM and a Poisson GBM challenger on the same split.
2. Compares mean Poisson deviance (lower = better) against a naive baseline.
3. Shows the GLM coefficients (explainable by construction) and a SHAP ranking
   of the GBM (post-hoc explanation), saving a SHAP importance bar to
   reports/figures/shap_gbm_frequency.png.

The story: does the extra accuracy of the GBM justify losing the GLM's
transparency? SHAP is how we answer that in a regulated pricing context.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import mean_poisson_deviance  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import OrdinalEncoder  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pricing.data import CATEGORICAL, NUMERIC, load_frequency  # noqa: E402
from src.pricing.frequency import build_frequency_pipeline  # noqa: E402
from src.pricing.gbm import build_frequency_gbm  # noqa: E402

FIGDIR = ROOT / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
FEATURES = CATEGORICAL + NUMERIC


def main() -> None:
    df = load_frequency()
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]

    # Ordinal-encode categoricals to integer codes (fit on full vocabulary so
    # no category is unseen at predict time). One matrix serves both models:
    # the GLM one-hots the codes, the GBM treats them as native categoricals.
    enc = OrdinalEncoder(dtype=np.float64)
    X = df[FEATURES].copy()
    X[CATEGORICAL] = enc.fit_transform(df[CATEGORICAL])
    y = df["Frequency"]
    w = df["Exposure"]

    X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
        X, y, w, test_size=0.2, random_state=42
    )

    glm = build_frequency_pipeline()
    glm.fit(X_tr, y_tr, glm__sample_weight=w_tr)
    glm_pred = glm.predict(X_te)

    gbm = build_frequency_gbm()
    gbm.fit(X_tr, y_tr, sample_weight=w_tr)
    gbm_pred = gbm.predict(X_te)

    base = np.average(y_tr, weights=w_tr)
    base_pred = np.full(len(y_te), base)

    def dev(p: np.ndarray) -> float:
        return mean_poisson_deviance(y_te, p, sample_weight=w_te)

    d_base, d_glm, d_gbm = dev(base_pred), dev(glm_pred), dev(gbm_pred)

    print("=" * 70)
    print("freMTPL2 - frequency: GLM vs GBM   [mean Poisson deviance, test]")
    print("=" * 70)
    print(f"{'Naive mean':<26} {d_base:.5f}")
    print(f"{'GLM (Poisson)':<26} {d_glm:.5f}   ({(d_base-d_glm)/d_base*100:+.2f}% vs naive)")
    print(f"{'GBM (Poisson)':<26} {d_gbm:.5f}   ({(d_base-d_gbm)/d_base*100:+.2f}% vs naive)")
    print(f"GBM beats GLM by {(d_glm-d_gbm)/d_glm*100:+.2f}% deviance")

    # --- GLM: explainable by construction (coefficients) ---
    pre = glm.named_steps["pre"]
    reg = glm.named_steps["glm"]
    names = pre.get_feature_names_out()
    coef = reg.coef_
    order = np.argsort(np.abs(coef))[::-1][:8]
    print("-" * 70)
    print("GLM - top coefficients (exp(coef) = multiplicative effect on frequency):")
    for i in order:
        print(f"  {names[i]:<22} coef {coef[i]:+.3f}   x{np.exp(coef[i]):.3f}")

    # --- GBM: post-hoc explanation with SHAP ---
    print("-" * 70)
    import shap

    bg = X_tr.sample(n=80, random_state=42)
    sample = X_te.sample(n=300, random_state=0)
    try:
        explainer = shap.TreeExplainer(gbm)
        shap_arr = np.asarray(explainer(sample, check_additivity=False).values)
        method = "TreeExplainer"
    except Exception:
        explainer = shap.Explainer(gbm.predict, bg)
        shap_arr = np.asarray(explainer(sample).values)
        method = "model-agnostic (Permutation)"

    mean_abs = np.abs(shap_arr).mean(axis=0)
    rank = np.argsort(mean_abs)[::-1]
    print(f"GBM - SHAP feature importance ({method}, mean|SHAP| over {len(sample)} apolices):")
    for i in rank:
        print(f"  {FEATURES[i]:<14} {mean_abs[i]:.5f}")

    # --- Figure: SHAP importance bar ---
    ordered = rank[::-1]
    plt.figure(figsize=(7, 4))
    plt.barh([FEATURES[i] for i in ordered], mean_abs[ordered], color="#1f4e79")
    plt.xlabel("mean(|SHAP value|)")
    plt.title("GBM frequency - importancia das features (SHAP)")
    plt.tight_layout()
    fig_path = FIGDIR / "shap_gbm_frequency.png"
    plt.savefig(fig_path, dpi=130)
    print("-" * 70)
    print(f"Figura salva: {fig_path.relative_to(ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
