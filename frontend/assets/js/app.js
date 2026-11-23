import { api, ApiError } from "./api.js";
import * as charts from "./charts.js";
import { initRealHouses } from "./realhouses.js";
import { initSimulator, onThemeChange, openSampleById, whenReady } from "./simulator.js";
import { $, $$, algoName, esc, nf, pct, toast, usd } from "./util.js";

const loaded = {};
const failState = (sel, e) => {
  const msg = e instanceof ApiError && e.status === 503 ? `Dados indisponíveis: ${e.message}` : e.message || "Falha ao carregar.";
  $(sel).innerHTML = `<div class="state-error"><span>${esc(msg)}</span><button class="btn btn--sm" type="button" onclick="location.reload()">Tentar de novo</button></div>`;
};

// ------------------------------------------------------------------ abas
const TABS = ["predict", "real", "how"];
function selectTab(name, { focus = false } = {}) {
  if (!TABS.includes(name)) name = "predict";
  for (const t of TABS) {
    const on = t === name;
    const b = $(`#tab-${t}`);
    b.setAttribute("aria-selected", String(on));
    b.tabIndex = on ? 0 : -1;
    $(`#${t}`).hidden = !on;
  }
  if (focus) $(`#tab-${name}`).focus();
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  $("#content").scrollTop = 0;
  load(name);
  requestAnimationFrame(() => charts.resizeIn($(`#${name}`)));
}
function initTabs() {
  $$('[role="tab"]').forEach((b) => b.addEventListener("click", () => selectTab(b.getAttribute("aria-controls"))));
  $(".tabs").addEventListener("keydown", (e) => {
    const i = TABS.findIndex((t) => $(`#tab-${t}`).getAttribute("aria-selected") === "true");
    const map = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: TABS.length - 1 };
    if (e.key in map) { e.preventDefault(); selectTab(TABS[(map[e.key] + TABS.length) % TABS.length], { focus: true }); }
  });
  window.addEventListener("hashchange", () => selectTab(location.hash.slice(1)));
  $$("details.more").forEach((d) => d.addEventListener("toggle", () => { if (d.open) requestAnimationFrame(() => charts.resizeIn(d)); }));
}
function load(name) {
  if (loaded[name]) return;
  loaded[name] = true;
  ({ predict: () => initSimulator({ latency }), real: () => initRealHouses({ openInSimulator }), how: loadHow })[name]();
}
async function openInSimulator(id) {
  selectTab("predict");
  await whenReady();
  const ok = await openSampleById(id);
  toast(ok ? `Casa Id ${id} aberta no simulador.` : "Casa não encontrada.");
}

// ------------------------------------------------------------------ tema e barra de status
function effectiveDark() {
  const t = document.documentElement.dataset.theme;
  return t ? t === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
}
function initTheme() {
  $("#theme-toggle").addEventListener("click", () => {
    const next = effectiveDark() ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("hv.theme", next); } catch { /* preferência só desta sessão */ }
    charts.refreshAll();
    onThemeChange();
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { charts.refreshAll(); onThemeChange(); });
}
function latency(ms) { $("#sb-latency").textContent = `Última estimativa em ${ms} ms`; }
async function initStatus() {
  const el = $("#sb-api");
  try {
    const h = await api.health();
    el.classList.toggle("is-ok", h.model_loaded);
    el.classList.toggle("is-err", !h.model_loaded);
    el.lastChild.textContent = h.model_loaded ? "Modelo carregado" : `Modelo não carregado: ${h.model_error || ""}`;
    if (h.model_loaded) {
      const m = await api.modelInfo();
      $("#sb-model").textContent = `${algoName(m.model_config)} · versão ${m.model_version}${m.holdout_metrics ? ` · erro médio nas casas de teste ${usd(m.holdout_metrics.mae)}` : ""}`;
    }
  } catch (e) {
    el.classList.add("is-err");
    el.lastChild.textContent = "Sem conexão com o servidor";
  }
}

// ------------------------------------------------------------------ Como funciona
let comparison = null;
async function loadHow() {
  try {
    const [ov, comp, ev, imp, ana, form, eda, real] = await Promise.all([
      api.overview(), api.comparison(), api.evaluation(), api.importance(), api.modelAnalysis().catch(() => null),
      api.formOptions(), api.eda(), api.realHouses(),
    ]);
    comparison = comp;
    const labels = Object.fromEntries(form.fields.map((f) => [f.name, f.label]));
    const sm = real.summary;
    const h = ov.model.holdout_metrics, b = ov.model.baseline_holdout_metrics;
    const mono = ov.model.config.includes("monot");
    const q = ana?.segments?.price_quartile;

    $("#story").innerHTML = [
      ["Os dados", `<b>${nf(ov.dataset.labelled_rows)}</b> casas vendidas em Ames, Iowa (EUA), entre 2006 e 2010, com o preço da venda. O modelo usa <b>14 características</b> de cada casa: áreas (em m²), qualidade, conservação, ano, quartos, banheiros, garagem, bairro e outras.`],
      ["Como ele aprende", `O algoritmo escolhido, <b>Gradient Boosting</b>, monta centenas de pequenas árvores de decisão, cada uma corrigindo os erros das anteriores.${mono ? " Ele segue regras de bom senso: com o resto igual, mais área, mais qualidade ou mais garagem nunca diminuem a estimativa." : ""}`],
      ["Como foi testado", `Antes do treino, <b>${nf(ov.model.holdout_n)}</b> casas (20%) foram separadas e o modelo nunca as viu. É como uma prova com questões inéditas: o erro é medido só nelas.`],
      ["O resultado", `Nessas casas, o erro típico foi de <b>${nf(sm.median_ape_pct, 1)}%</b> e <b>${pct(sm.within_10pct)}</b> delas ficaram a até 10% do preço real. Erro médio: <b>${usd(h.mae)}</b>, contra ${usd(b.mae)} de quem sempre chutasse o preço mediano.`],
      ["Os limites", q ? `Ele puxa as estimativas para o meio: nas casas mais baratas fica, em média, ${usd(Math.abs(q[0].bias))} acima do real; nas mais caras, ${usd(Math.abs(q[3].bias))} abaixo. Vale só para essa cidade e época e não substitui uma avaliação profissional.` : "Vale só para essa cidade e época e não substitui uma avaliação profissional."],
    ].map(([t, p]) => `<li><h3>${t}</h3><p>${p}</p></li>`).join("");
    $("#how-state").hidden = true;
    $("#how-body").hidden = false;

    const perm = [...imp.app.permutation_holdout.rows].reverse();
    charts.hbar("chart-perm", perm.map((r) => labels[r.feature] || r.feature), perm.map((r) => r.importance_mean), { xTitle: "Quanto o erro aumenta", format: ".3f", errors: perm.map((r) => r.importance_std), hoverFmt: ".4f" });

    if (q) {
      $("#bias-desc").textContent = `Diferença média entre a estimativa e o preço real em cada faixa de preço (positivo: o modelo estimou acima do real). Nas faixas do meio, o erro típico fica perto de ${nf((q[1].mape_pct + q[2].mape_pct) / 2, 1)}%; nos extremos ele erra mais, porque há poucos exemplos (regressão à média).`;
      charts.biasBars("chart-bias", q);
    }

    const cand = comp.candidates_holdout?.rows || [];
    if (cand.length) {
      const t = ana?.tie_test;
      $("#algo-desc").textContent = `Testamos ${cand.length} algoritmos com os mesmos dados. A escolha foi feita antes de olhar as casas de teste, por validação cruzada.${t ? ` O escolhido e o segundo colocado ficaram praticamente empatados: a diferença entre eles é menor que a variação de uma divisão dos dados para outra.` : ""} O gráfico mostra o erro médio de cada um nas casas de teste, só para comparação.`;
      charts.algoMae("chart-algo-mae", cand);
    }
    renderTechnical(comp, ana, eda);
  } catch (e) { failState("#how-state", e); }
}

function renderTechnical(comp, ana, eda) {
  const p = comp.protocol;
  $("#protocol").innerHTML = [["Partição", p.split], ["Validação", p.cv], ["Hiperparâmetros", p.tuning], ["Seleção", p.primary_metric], ["Escala", p.metrics_scale]]
    .map(([t, d]) => `<li><b>${esc(t)}.</b> ${esc(d)}</li>`).join("");
  const h = comp.holdout;
  const rows = [["app", "Modelo da aplicação (14 características)"], ["full", "Referência (79 características)"], ["baseline", "Chute da mediana"]];
  $("#holdout-table").innerHTML = `<thead><tr><th>Modelo</th><th class="txt">Algoritmo</th><th>RMSLE</th><th>Erro médio (MAE)</th><th>RMSE</th><th>R²</th></tr></thead><tbody>${
    rows.map(([k, label]) => { const m = h.metrics[k]; return `<tr class="${k === "app" ? "is-selected" : k === "baseline" ? "is-baseline" : ""}"><td>${label}</td><td class="txt">${esc(algoName(h.artifacts[k]))}</td><td class="num">${nf(m.rmsle, 4)}</td><td class="num">${usd(m.mae)}</td><td class="num">${usd(m.rmse)}</td><td class="num">${nf(m.r2, 3)}</td></tr>`; }).join("")}</tbody>`;

  const v1 = comp.history_v1;
  if (v1) {
    const app = comp.feature_sets.app;
    const sel = app.results.find((r) => r.config === app.selected);
    const row = (label, cfg, cv, m) => `<tr><td>${label}</td><td class="txt">${esc(algoName(cfg))}</td><td class="num">${nf(cv, 4)}</td><td class="num">${nf(m.rmsle, 4)}</td><td class="num">${usd(m.mae)}</td><td class="num">${nf(m.r2, 3)}</td></tr>`;
    $("#v1v2-table").innerHTML = `<thead><tr><th>Versão</th><th class="txt">Algoritmo</th><th>RMSLE validação</th><th>RMSLE teste</th><th>Erro médio teste</th><th>R² teste</th></tr></thead><tbody>${row(`v${esc(v1.model_version)}`, v1.model_config, v1.cv_metrics.rmsle.mean, v1.holdout_metrics)}${row("v2 (atual)", app.selected, sel.cv_rmsle_mean, h.metrics.app)}</tbody>`;
    $("#v1v2-note").textContent = "A v2 acrescentou Gradient Boosting com o bairro como categoria e com restrições de bom senso. A escolha continuou sendo só pela validação cruzada. Ressalva: as casas de teste já tinham sido consultadas uma vez na v1.";
  } else $("#panel-v1v2").hidden = true;

  renderSet("app");
  $$('input[name="feature-set"]').forEach((r) => r.addEventListener("change", () => renderSet(r.value)));

  if (ana) {
    const t = ana.tie_test;
    $("#tie-verdict").textContent = t.verdict;
    $("#tie-desc").textContent = `${algoName(t.models[0])} × ${algoName(t.models[1])} em ${t.n_splits} divisões diferentes dos dados. O primeiro venceu ${t.first_wins}; diferença média ${nf(t.diff_mean, 4)} ± ${nf(t.diff_std, 4)}. ${t.rule}`;
    charts.tieDots("chart-tie", t);
    const lc = ana.learning_curve;
    $("#lc-desc").textContent = `O erro de validação cai de ${nf(lc.val_rmsle_mean[0], 4)} com ${lc.train_sizes[0]} casas para ${nf(lc.val_rmsle_mean.at(-1), 4)} com ${lc.train_sizes.at(-1)}. A distância até a linha de treino mostra o sobreajuste.`;
    charts.learningCurve("chart-lc", lc);
    const segTable = (rs, title) => `<thead><tr><th>${title}</th><th>Casas</th><th>Erro típico</th><th>Modelo − real (média)</th></tr></thead><tbody>${
      rs.map((r) => `<tr><td>${esc(r.segment)}</td><td class="num">${r.n}</td><td class="num">${nf(r.mape_pct, 1)}%</td><td class="num">${-r.bias >= 0 ? "+" : "−"}${usd(Math.abs(r.bias))}</td></tr>`).join("")}</tbody>`;
    $("#seg-sale").innerHTML = segTable(ana.segments.sale_condition, "Tipo de venda");
    $("#seg-nbhd").innerHTML = segTable(ana.segments.neighborhood.slice(0, 10), "Bairro (8+ casas)");
  }
  charts.histogram("chart-hist", eda.target_histogram, { xTitle: "Preço de venda (USD)" });
  charts.scatterArea("chart-scatter", eda.scatter_grlivarea);
}

function renderSet(key) {
  const fs = comparison.feature_sets[key];
  $("#set-desc").textContent = key === "app" ? `Características do simulador: ${fs.features.join(", ")}.` : `Todas as ${fs.features.length} características da base (referência).`;
  $("#cv-table").innerHTML = `<thead><tr><th>Algoritmo</th><th>RMSLE</th><th>± desvio</th><th>Erro médio</th><th>R²</th><th>RMSLE no treino</th><th class="txt">Hiperparâmetros</th></tr></thead><tbody>${
    [...fs.results].sort((a, b) => a.cv_rmsle_mean - b.cv_rmsle_mean).map((r) => `<tr class="${r.config === fs.selected ? "is-selected" : r.algorithm === "DummyRegressor" ? "is-baseline" : ""}"><td>${esc(algoName(r.config))}${r.config === fs.selected ? ' <span class="tag">escolhido</span>' : ""}</td><td class="num">${nf(r.cv_rmsle_mean, 4)}</td><td class="num">${nf(r.cv_rmsle_std, 4)}</td><td class="num">${usd(r.cv_mae_mean)}</td><td class="num">${nf(r.cv_r2_mean, 3)}</td><td class="num">${nf(r.train_rmsle_mean, 4)}</td><td class="txt">${esc(Object.entries(r.best_params).map(([k, v]) => `${k}=${v}`).join(", ") || "–")}</td></tr>`).join("")}</tbody>`;
}

// ------------------------------------------------------------------ início
initTabs();
initTheme();
initStatus();
selectTab(location.hash.slice(1) || "predict");
