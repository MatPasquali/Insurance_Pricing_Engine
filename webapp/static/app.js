const form = document.getElementById("profile-form");

// Mostra o valor atual de cada slider.
form.querySelectorAll("input[type=range]").forEach((r) => {
  const out = document.getElementById("val-" + r.dataset.name);
  const update = () => { out.textContent = Number(r.value).toFixed(0); };
  r.addEventListener("input", update);
  update();
});

function show(id, text) { document.getElementById(id).textContent = text; }

// Decomposição do GLM: base x fatores = frequência prevista.
function renderDecomp(base, fatores, previsto) {
  const el = document.getElementById("glm-decomp");
  let html = `<div class="d-row d-base"><span>Base da carteira</span><b>${base.toFixed(3)}/ano</b></div>`;
  const ordenado = Object.entries(fatores)
    .sort((a, b) => Math.abs(Math.log(b[1])) - Math.abs(Math.log(a[1])));
  for (const [feat, fator] of ordenado) {
    const cls = fator > 1 ? "up" : "down";
    html += `<div class="d-row"><span>${feat}</span><b class="${cls}">×${fator.toFixed(3)}</b></div>`;
  }
  html += `<div class="d-row d-total"><span>Frequência prevista (GLM)</span><b>${previsto.toFixed(3)}/ano</b></div>`;
  el.innerHTML = html;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = {};
  new FormData(form).forEach((v, k) => { data[k] = v; });

  const btn = form.querySelector("button");
  btn.disabled = true; btn.textContent = "Calculando...";
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const j = await res.json();
    show("freq_glm", j.freq_glm.toFixed(3) + "/ano");
    show("freq_gbm", j.freq_gbm.toFixed(3) + "/ano");
    show("severity", Math.round(j.severity).toLocaleString("pt-BR") + " EUR");
    show("premium_glm", j.premium_glm.toFixed(2) + " EUR");
    show("premium_gbm", j.premium_gbm.toFixed(2) + " EUR");
    renderDecomp(j.glm_base, j.glm_fatores, j.glm_previsto);
    document.getElementById("shap").src = "data:image/png;base64," + j.shap_png;
  } catch (err) {
    alert("Erro ao calcular: " + err);
  } finally {
    btn.disabled = false; btn.textContent = "Calcular prêmio";
  }
});
