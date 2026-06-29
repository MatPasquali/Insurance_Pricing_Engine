"""v2: elasticidade de preço + otimização de prêmio, com validação temporal.

Treina nos anos <= 2016 e valida em 2017 (out-of-time). Estima:
  - custo (Tweedie), retenção (logística vs GBM), elasticidade (efeito do preço),
  - prêmio ótimo por apólice (maximiza E[lucro] = P(renovar)·(preço − custo)),
  - e salva a curva lucro × preço da carteira em reports/figures/.

Run:  python run_pricing_optimization.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import mean_tweedie_deviance, roc_auc_score  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pricing.es_data import ES_CATEGORICAL, ES_NUMERIC, build_modeling_frame  # noqa: E402
from src.pricing.es_models import (  # noqa: E402
    RETENTION_NUMERIC,
    build_cost_model,
    build_retention_gbm,
    build_retention_model,
    optimize_premium,
)

FIGDIR = ROOT / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
POWER = 1.5


def main() -> None:
    df = build_modeling_frame()
    obs = df[df["observed_outcome"]].copy()
    tr = obs[obs["renewal_year"] <= 2016]
    te = obs[obs["renewal_year"] == 2017]

    cost_feats = ES_NUMERIC + ES_CATEGORICAL
    ret_feats = RETENTION_NUMERIC + ES_CATEGORICAL

    print("=" * 72)
    print("v2 - Elasticidade & otimizacao de premio (treino <=2016, teste 2017)")
    print("=" * 72)
    print(f"Treino: {len(tr):,} apolices-ano | Teste: {len(te):,}")

    # --- Custo (GBM Poisson sobre custo capado/winsorizado) ---
    cap = tr["LossCost"].quantile(0.99)
    y_cost = tr["LossCost"].clip(upper=cap)
    cost = build_cost_model().fit(tr[cost_feats], y_cost)
    cost_te = np.clip(cost.predict(te[cost_feats]), 1e-6, None)
    dev_m = mean_tweedie_deviance(te["LossCost"], cost_te, power=POWER)
    dev_b = mean_tweedie_deviance(
        te["LossCost"], np.full(len(te), y_cost.mean()), power=POWER
    )
    print("-" * 72)
    print(f"Custo Tweedie deviance: modelo {dev_m:.1f} | naive {dev_b:.1f}  "
          f"({(dev_b - dev_m) / dev_b * 100:+.1f}%)")
    print(f"Custo medio previsto {cost_te.mean():.1f} | real {te['LossCost'].mean():.1f} EUR")

    # --- Retencao (logistica vs GBM), out-of-time ---
    ret = build_retention_model().fit(tr[ret_feats], tr["renewed"])
    gbm = build_retention_gbm().fit(tr[ret_feats], tr["renewed"])
    auc_glm = roc_auc_score(te["renewed"], ret.predict_proba(te[ret_feats])[:, 1])
    auc_gbm = roc_auc_score(te["renewed"], gbm.predict_proba(te[ret_feats])[:, 1])
    print("-" * 72)
    print(f"Retencao AUC (out-of-time 2017): logistica {auc_glm:.3f} | GBM {auc_gbm:.3f}")

    # --- Elasticidade: efeito medio de +10% no preco ---
    base_ret = ret.predict_proba(te[ret_feats])[:, 1]
    te_hi = te.copy()
    te_hi["Premium"] = te["Premium"] * 1.10
    hi_ret = ret.predict_proba(te_hi[ret_feats])[:, 1]
    print(f"Elasticidade: renovacao media {base_ret.mean():.3f} (atual) -> "
          f"{hi_ret.mean():.3f} (+10% preco)  [delta {hi_ret.mean()-base_ret.mean():+.3f}]")

    # --- Otimizacao de premio na carteira de teste ---
    Xret = te[ret_feats].copy()
    prem = te["Premium"]
    profit_cur = (base_ret * (prem.to_numpy() - cost_te)).sum()

    def resumo(res):
        return res["exp_profit"].sum(), res["p_renew"].mean(), res["optimal_price"].mean()

    res_naive = optimize_premium(ret, Xret, prem, cost_te, np.round(np.arange(0.70, 1.41, 0.05), 2))
    res_band = optimize_premium(ret, Xret, prem, cost_te, np.round(np.arange(0.85, 1.16, 0.05), 2))
    p_n, r_n, pr_n = resumo(res_naive)
    p_b, r_b, pr_b = resumo(res_band)
    print("-" * 72)
    print("Otimizacao de premio (lucro esperado na carteira de teste):")
    print(f"  ATUAL:            lucro {profit_cur/1e3:8,.0f}k | retencao {base_ret.mean():.3f} | preco medio {prem.mean():.0f}")
    print(f"  OTIMO irrestrito: lucro {p_n/1e3:8,.0f}k ({(p_n-profit_cur)/profit_cur*100:+.0f}%) | retencao {r_n:.3f} | preco medio {pr_n:.0f}  <- RED FLAG (bate no teto)")
    print(f"  OTIMO restrito +-15%: lucro {p_b/1e3:6,.0f}k ({(p_b-profit_cur)/profit_cur*100:+.0f}%) | retencao {r_b:.3f} | preco medio {pr_b:.0f}")
    print("  CAVEAT: elasticidade subestimada (endogeneidade - preco nao foi aleatorizado).")
    print("          O otimo irrestrito sobe ao teto -> NAO confiavel; producao exige experimento de preco.")

    # --- Figura: curva lucro x multiplicador global (+ retencao) ---
    mults = np.round(np.arange(0.70, 1.41, 0.05), 2)
    tot_profit, mean_ret = [], []
    for m in mults:
        Xm = te[ret_feats].copy()
        Xm["Premium"] = prem * m
        p = ret.predict_proba(Xm)[:, 1]
        tot_profit.append(float((p * (prem.to_numpy() * m - cost_te)).sum()))
        mean_ret.append(float(p.mean()))
    best_m = float(mults[int(np.argmax(tot_profit))])

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(mults, np.array(tot_profit) / 1e3, "o-", color="#1f4e79")
    ax1.set_xlabel("multiplicador de preço (carteira inteira)")
    ax1.set_ylabel("lucro esperado (k EUR)", color="#1f4e79")
    ax1.axvline(best_m, color="gray", ls=":")
    ax2 = ax1.twinx()
    ax2.plot(mults, mean_ret, "s--", color="#c0504d")
    ax2.set_ylabel("retenção média", color="#c0504d")
    ax1.set_title(f"Lucro × preço — ótimo global ~ {best_m:.2f}× (trade-off com retenção)")
    plt.tight_layout()
    out = FIGDIR / "v2_profit_vs_price.png"
    plt.savefig(out, dpi=130)
    print("-" * 72)
    print(f"Figura salva: {out.relative_to(ROOT)}  (otimo global ~{best_m:.2f}x)")
    print("=" * 72)


if __name__ == "__main__":
    main()
