// Calculadora client-side: reproduz os GLMs (freq e sev) no navegador.
// Um GLM é linear -> freq = exp(intercepto + sum coef*x). Previsão exata, sem servidor.
let M = null;
const el = (id) => document.getElementById(id);

fetch("assets/model.json")
  .then((r) => r.json())
  .then((m) => { M = m; buildForm(); compute(); })
  .catch(() => { const d = el("c-decomp"); if (d) d.textContent = "Não foi possível carregar o modelo."; });

function buildForm() {
  let html = "";
  for (const f of M.numeric) {
    const r = M.ranges[f];
    html += `<label class="cfield"><span>${f} <b id="cv-${f}"></b></span>
      <input type="range" data-f="${f}" min="${r.min}" max="${r.max}" value="${r.med}" step="any"></label>`;
  }
  for (const f of M.categorical) {
    const opts = M.options[f].map((o) => `<option value="${o.code}">${o.label}</option>`).join("");
    html += `<label class="cfield"><span>${f}</span><select data-f="${f}">${opts}</select></label>`;
  }
  const form = el("calc-form");
  form.innerHTML = html;
  form.querySelectorAll("[data-f]").forEach((e) => e.addEventListener("input", compute));
  form.querySelectorAll("input[type=range]").forEach((e) => {
    const out = el("cv-" + e.dataset.f);
    const upd = () => { out.textContent = Number(e.value).toFixed(0); };
    e.addEventListener("input", upd); upd();
  });
}

function readProfile() {
  const p = {};
  document.querySelectorAll("#calc-form [data-f]").forEach((e) => { p[e.dataset.f] = Number(e.value); });
  return p;
}

function glmPredict(glm, p) {
  let lp = glm.intercept;
  const factors = {};
  for (const f of M.numeric) {
    const q = glm.numeric[f];
    const c = q.coef * ((p[f] - q.mean) / q.scale);
    lp += c; factors[f] = Math.exp(c);
  }
  for (const f of M.categorical) {
    const c = (glm.categorical[f] || {})[String(p[f])] || 0;
    lp += c; factors[f] = Math.exp(c);
  }
  return { pred: Math.exp(lp), base: Math.exp(glm.intercept), factors };
}

function compute() {
  if (!M) return;
  const p = readProfile();
  const f = glmPredict(M.freq, p);
  const s = glmPredict(M.sev, p);
  el("c-freq").textContent = f.pred.toFixed(3) + "/ano";
  el("c-sev").textContent = Math.round(s.pred).toLocaleString("pt-BR") + " EUR";
  el("c-premio").textContent = (f.pred * s.pred).toFixed(2) + " EUR";

  let html = `<div class="d-row d-base"><span>Base da carteira</span><b>${f.base.toFixed(3)}/ano</b></div>`;
  Object.entries(f.factors)
    .sort((a, b) => Math.abs(Math.log(b[1])) - Math.abs(Math.log(a[1])))
    .forEach(([k, v]) => {
      html += `<div class="d-row"><span>${k}</span><b class="${v > 1 ? "up" : "down"}">×${v.toFixed(3)}</b></div>`;
    });
  html += `<div class="d-row d-total"><span>Frequência (GLM)</span><b>${f.pred.toFixed(3)}/ano</b></div>`;
  el("c-decomp").innerHTML = html;
}
