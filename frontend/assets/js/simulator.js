import { api, ApiError } from "./api.js";
import * as charts from "./charts.js";
import { controlFor } from "./components.js";
import { $, algoName, errorStatus, esc, fmtField, nf, reduceMotion, saleText, signedPct, signedUsd, toast, usd } from "./util.js";

const GROUP_ORDER = ["Localização", "Áreas", "Qualidade", "Idade", "Cômodos"];
const GROUP_TITLE = { Localização: "Localização", Áreas: "Tamanho", Qualidade: "Qualidade e acabamento", Idade: "Idade", Cômodos: "Cômodos e itens" };
const COMPARE_ROWS = ["neighborhood", "gr_liv_area_m2", "total_bsmt_m2", "lot_area_m2", "overall_qual", "overall_cond", "year_built", "year_remod_add", "bedroom_abv_gr", "full_bath", "garage_cars", "fireplaces", "kitchen_qual", "central_air"];

const S = {
  form: null, fields: [], byApi: {}, controls: {}, values: {},
  origin: { type: "typical" }, reference: {}, mine: null, pinned: null, live: true,
  last: null, shown: 0, similar: [], hooks: {},
  predictCtl: null, sensCtl: null, cmpCtl: null, decCtl: null,
  predictTimer: 0, sensTimer: 0, cmpTimer: 0,
  samples: null, sampleIdx: -1, priceHist: null, sensField: "gr_liv_area_m2", lastCompare: null,
};
let readyResolve;
const ready = new Promise((res) => (readyResolve = res));
export const whenReady = () => ready;

export async function initSimulator(hooks) {
  S.hooks = hooks;
  try {
    const [form, eda] = await Promise.all([api.formOptions(), api.eda().catch(() => null)]);
    S.form = form;
    S.fields = form.fields;
    S.fields.forEach((f) => (S.byApi[f.api_name] = f));
    S.priceHist = eda?.target_histogram || null;
    buildForm();
    buildSensSelect();
    applyValues(typicalValues(), { type: "typical" });
    $("#form-state").hidden = true;
    $("#sim").hidden = false;
    bindToolbar();
    window.addEventListener("resize", () => drawStrip());
    refresh(0);
    readyResolve(true);
  } catch (e) {
    $("#form-state").innerHTML = `<div class="state-error"><span>${esc(e instanceof ApiError && e.status === 503 ? "Modelo indisponível: " + e.message : e.message)}</span><button class="btn btn--sm" type="button" onclick="location.reload()">Tentar de novo</button></div>`;
  }
}

// ------------------------------------------------------------------ formulário
function buildForm() {
  const groups = {};
  for (const f of S.fields) (groups[f.group] ||= []).push(f);
  const root = $("#form-groups");
  root.innerHTML = "";
  for (const g of GROUP_ORDER.filter((x) => groups[x])) {
    const sec = document.createElement("section");
    sec.className = "group";
    sec.innerHTML = `<h3 class="group__title">${GROUP_TITLE[g] || g}</h3><div class="fields"></div>`;
    const box = sec.querySelector(".fields");
    for (const f of [...groups[g]].sort((a, b) => wideRank(a) - wideRank(b))) {
      const c = controlFor(f, onFieldChange, { neighborhoodStats: S.form.neighborhoods?.stats });
      S.controls[f.api_name] = c;
      box.appendChild(c.el);
    }
    root.appendChild(sec);
  }
  // "Nunca reformado": reforma = construção (semântica do dicionário de dados)
  const remod = S.controls.year_remod_add;
  if (remod) {
    const lab = document.createElement("label");
    lab.className = "check";
    lab.innerHTML = `<input type="checkbox" id="never-remod"> Nunca reformado (usa o ano de construção)`;
    remod.el.querySelector(".field__help").after(lab);
    lab.querySelector("input").addEventListener("change", (e) => {
      remod.disable(e.target.checked);
      if (e.target.checked) onFieldChange("year_remod_add", S.values.year_built);
    });
  }
  $("#predict-form").addEventListener("submit", (e) => { e.preventDefault(); refresh(0); });
}
const wideRank = (f) => (f.api_name === "neighborhood" ? 0 : f.unit === "nota 1–10" ? 1 : f.type !== "category" && f.max - f.min > 12 ? 2 : 3);

function typicalValues() {
  const v = {};
  for (const f of S.fields) v[f.api_name] = f.type === "category" ? f.options.reduce((a, b) => (b.n_train > a.n_train ? b : a)).value : Math.round(f.median);
  if (v.year_remod_add < v.year_built) v.year_remod_add = v.year_built;
  return v;
}

function applyValues(values, origin, reference = values) {
  S.values = { ...values };
  S.reference = { ...reference };
  S.origin = origin;
  for (const [k, c] of Object.entries(S.controls)) { c.set(values[k]); c.setChanged(S.values[k] !== S.reference[k]); }
  const never = $("#never-remod");
  if (never) { never.checked = values.year_remod_add === values.year_built; S.controls.year_remod_add.disable(never.checked); }
  renderOrigin();
}

function onFieldChange(apiName, value) {
  S.values[apiName] = value;
  if (apiName === "year_built") {
    const never = $("#never-remod");
    if (never?.checked || S.values.year_remod_add < value) {
      if (!never?.checked) toast("Ano da reforma ajustado para o ano de construção.");
      S.values.year_remod_add = value;
      S.controls.year_remod_add.set(value);
    }
  }
  if (apiName === "year_remod_add" && value < S.values.year_built) {
    S.controls.year_remod_add.setError("A reforma não pode ser anterior à construção.");
    return;
  }
  for (const [k, c] of Object.entries(S.controls)) c.setChanged(S.values[k] !== S.reference[k]);
  renderOrigin();
  if (S.live) refresh(170);
  else $("#result").classList.add("is-stale");
}
const changedFields = () => Object.keys(S.values).filter((k) => S.values[k] !== S.reference[k]);
const isPureSample = () => S.origin.type === "sample" && !changedFields().length;

function renderOrigin() {
  const o = S.origin, n = changedFields().length;
  const el = $("#origin");
  const plural = (k, s) => `${k} ${s}${k > 1 ? "s" : ""}`;
  let html;
  if (o.type === "sample") {
    html = n
      ? `<span class="tag">Casa vendida · Id ${o.sample.id}</span> alterada em ${plural(n, "característica")}; o preço real dela não serve mais de comparação.`
      : `<span class="tag tag--ok">Casa vendida · Id ${o.sample.id}</span> vendida em ${o.sample.year_sold} (${esc(saleText(o.sample.sale_condition))}); o modelo nunca a viu no treino.`;
  } else {
    html = n
      ? `<span class="tag">Sua casa</span> ${plural(n, "característica")} diferente${n > 1 ? "s" : ""} do imóvel típico.`
      : `<span class="tag">Imóvel típico</span> medianas e categorias mais frequentes da base. Ajuste à vontade.`;
  }
  if (S.mine && o.type === "sample") html += ` <button type="button" class="btn btn--sm" id="btn-back-mine">Voltar para a casa que você montou</button>`;
  el.innerHTML = html;
  $("#btn-back-mine")?.addEventListener("click", restoreMine);
}

// ------------------------------------------------------------------ previsão
/** Recalcula a estimativa, a curva de sensibilidade e a comparação de algoritmos. */
function refresh(delay) {
  schedulePredict(delay);
  scheduleSens(delay ? delay + 250 : 0);
  scheduleCompare(delay ? delay + 120 : 0);
}
function schedulePredict(delay) {
  clearTimeout(S.predictTimer);
  S.predictTimer = setTimeout(predict, delay);
}

async function predict() {
  S.predictCtl?.abort();
  const ctl = (S.predictCtl = new AbortController());
  const result = $("#result");
  const btn = $("#submit-btn");
  result.classList.add("is-loading");
  btn.disabled = true; btn.classList.add("is-loading");
  const t0 = performance.now();
  try {
    const r = await api.predict({ ...S.values }, ctl.signal);
    if (ctl.signal.aborted) return;
    S.last = r;
    for (const c of Object.values(S.controls)) c.setError("");
    $("#result-error").hidden = true;
    result.classList.remove("is-stale");
    tweenTo(r.predicted_sale_price);
    renderRange(r);
    renderCompare();
    drawStrip();
    renderSimilar(r.similar_real_houses || []);
    renderScenario();
    renderAlgos();
    const dot = $("#live-dot");
    dot.classList.remove("is-pulse"); void dot.offsetWidth; dot.classList.add("is-pulse");
    S.hooks.latency?.(Math.round(performance.now() - t0));
  } catch (e) {
    if (e.name === "AbortError") return;
    result.classList.add("is-stale");
    const box = $("#result-error");
    box.hidden = false;
    box.innerHTML = `<strong>${esc(e.message)}</strong>` + (e.errors?.length ? `<ul>${e.errors.map((x) => `<li>${esc(x.message)}</li>`).join("")}</ul>` : "");
    for (const x of e.errors || []) S.controls[x.field]?.setError(x.message);
  } finally {
    if (S.predictCtl === ctl) {
      result.classList.remove("is-loading");
      btn.disabled = false; btn.classList.remove("is-loading");
    }
  }
}

function tweenTo(target) {
  const el = $("#result-value");
  const from = S.shown || target;
  S.shown = target;
  if (reduceMotion() || from === target) { el.textContent = nf(target); return; }
  const t0 = performance.now(), dur = 380;
  cancelAnimationFrame(tweenTo.raf);
  const step = (now) => {
    const k = Math.min(1, (now - t0) / dur);
    el.textContent = nf(from + (target - from) * (1 - (1 - k) ** 4)); // ease-out-quart
    if (k < 1) tweenTo.raf = requestAnimationFrame(step);
  };
  tweenTo.raf = requestAnimationFrame(step);
}

function renderRange(r) {
  const rg = r.range;
  $("#result-range").innerHTML = rg
    ? `Faixa típica: <span class="num-txt">${usd(rg.low)}</span> a <span class="num-txt">${usd(rg.high)}</span>. Nas casas de teste, o preço real caiu numa faixa calculada assim em ${nf(100 * rg.holdout_coverage)}% dos casos.`
    : "";
  const unusual = r.notes.find((n) => n.startsWith("Valores pouco comuns"));
  const w = $("#result-warn");
  w.hidden = !unusual;
  w.textContent = unusual || "";
}

/** Casa vendida carregada sem alterações: Real | diferença | Modelo. Cenário A fixado: diferença. */
function renderCompare() {
  const p = S.last?.predicted_sale_price;
  let html = "";
  if (isPureSample() && p) {
    const s = S.origin.sample, diff = p - s.actual, pctv = (100 * diff) / s.actual, st = errorStatus(pctv);
    html += `<div class="pair pair--compact">
      <div class="pair__tile pair__tile--real"><span class="pair__label"><span class="key key--real"></span>Real: preço da venda</span><span class="pair__value">${usd(s.actual)}</span><span class="pair__note">vendida em ${s.year_sold}</span></div>
      <div class="pair__vs pair__vs--${st.cls}"><span class="pair__diff">${signedPct(pctv)}</span><span class="pair__diffusd">${signedUsd(diff)}</span><span class="pair__arrow">${st.label}</span></div>
      <div class="pair__tile pair__tile--model"><span class="pair__label"><span class="key key--model"></span>Modelo</span><span class="pair__value">${usd(p)}</span><span class="pair__note">estimativa</span></div>
    </div>`;
  }
  if (S.pinned && p) {
    const d = p - S.pinned.price;
    html += `<p><span class="delta">${Math.abs(d) < 1 ? "igual" : signedUsd(d)} (${signedPct((100 * d) / S.pinned.price)})</span> <span class="fine">em relação ao cenário A</span></p>`;
  }
  $("#result-compare").innerHTML = html;
}

// ------------------------------------------------------------------ casas vendidas parecidas
/** "Parecido o bastante" para não destacar: áreas até 10% de diferença, anos até 5. */
function sameField(api, a, b) {
  const f = S.byApi[api];
  if (f?.unit === "m²") return Math.abs(a - b) <= Math.max(0.10 * Math.max(Math.abs(b), 1), 1);
  if (f?.unit === "ano") return Math.abs(a - b) <= 5;
  return a === b;
}

/** Destaca a coluna da casa na comparação completa (não altera a casa simulada). */
function focusColumn(i) {
  const sec = $("#cmp-similar");
  sec.scrollIntoView({ behavior: reduceMotion() ? "auto" : "smooth", block: "start" });
  sec.querySelectorAll(".is-focus").forEach((c) => c.classList.remove("is-focus"));
  sec.querySelectorAll(`[data-col="${i}"]`).forEach((c) => c.classList.add("is-focus"));
  clearTimeout(focusColumn.t);
  focusColumn.t = setTimeout(() => sec.querySelectorAll(".is-focus").forEach((c) => c.classList.remove("is-focus")), 2600);
}

function renderSimilar(list) {
  S.similar = list;
  const box = $("#similar-list");
  const full = $("#cmp-similar");
  if (!list.length) {
    box.innerHTML = `<p class="fine">Sem casas vendidas disponíveis para comparar.</p>`;
    full.hidden = true;
    return;
  }
  // tabela compacta (ao lado do formulário): Real | Modelo | Erro
  box.innerHTML = `<table class="rvm">
    <thead><tr><th>Casa vendida</th><th><span class="key key--real"></span>Real</th><th><span class="key key--model"></span>Modelo</th><th>Erro</th></tr></thead>
    <tbody>${list.map((h, i) => {
      const st = errorStatus(h.error_pct);
      return `<tr data-i="${i}" tabindex="0" aria-label="Ver casa Id ${h.id} na comparação completa">
        <td><b>${esc(h.neighborhood_label)}</b><small>${nf(h.area_m2)} m² · qualidade ${h.overall_qual} · ${h.year_built}</small></td>
        <td class="col-real">${usd(h.actual)}</td>
        <td class="col-model">${usd(h.predicted)}</td>
        <td><span class="chip chip--${st.cls}" title="${st.long}">${signedPct(h.error_pct)}</span></td></tr>`;
    }).join("")}</tbody></table>`;
  box.querySelectorAll("tr[data-i]").forEach((tr) => {
    const go = () => focusColumn(Number(tr.dataset.i));
    tr.addEventListener("click", go);
    tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
  });

  // comparação completa: sua casa × casas vendidas
  full.hidden = false;
  const me = S.values;
  const est = S.last?.predicted_sale_price;
  const cell = (i, inner, cls = "") => `<td data-col="${i}"${cls ? ` class="${cls}"` : ""}>${inner}</td>`;
  const head = `<thead><tr><th class="cmpx__rowhead">Característica</th><th class="cmpx__me">${S.origin.type === "sample" ? "Casa no simulador" : "Sua casa"}<small>simulada</small></th>${
    list.map((h, i) => `<th data-col="${i}">Casa vendida<small>Id ${h.id} · ${esc(h.neighborhood_label)}${h.same_neighborhood ? "" : " (outro bairro)"}</small></th>`).join("")}</tr></thead>`;
  const feat = COMPARE_ROWS.filter((k) => k in me).map((k) => `<tr><th class="cmpx__rowhead">${esc(S.byApi[k]?.label ?? k)}</th><td class="cmpx__me">${esc(fmtField(S.byApi[k], me[k]))}</td>${
    list.map((h, i) => cell(i, esc(fmtField(S.byApi[k], h.inputs[k])), sameField(k, h.inputs[k], me[k]) ? "" : "is-diff")).join("")}</tr>`).join("");
  const priceRows = `
    <tr class="cmpx__sep"><th colspan="${list.length + 2}">Preço: dado real × modelo</th></tr>
    <tr class="cmpx__real"><th class="cmpx__rowhead"><span class="key key--real"></span>Real: preço da venda</th><td class="cmpx__me cmpx__na">não foi vendida</td>${list.map((h, i) => cell(i, `${usd(h.actual)}<small>vendida em ${h.year_sold}</small>`)).join("")}</tr>
    <tr class="cmpx__model"><th class="cmpx__rowhead"><span class="key key--model"></span>Modelo: estimativa</th><td class="cmpx__me">${est ? usd(est) : "–"}</td>${list.map((h, i) => cell(i, usd(h.predicted))).join("")}</tr>
    <tr><th class="cmpx__rowhead">Diferença (modelo − real)</th><td class="cmpx__me cmpx__na">–</td>${list.map((h, i) => cell(i, `${signedUsd(h.predicted - h.actual)}<small>${signedPct(h.error_pct)}</small>`)).join("")}</tr>
    <tr><th class="cmpx__rowhead">Resultado do modelo</th><td class="cmpx__me cmpx__na">–</td>${list.map((h, i) => { const st = errorStatus(h.error_pct); return cell(i, `<span class="chip chip--${st.cls}">${st.label}</span>`); }).join("")}</tr>
    <tr><th class="cmpx__rowhead"><span class="visually-hidden">Ações</span></th><td class="cmpx__me"></td>${list.map((h, i) => cell(i, `<button type="button" class="btn btn--sm" data-load="${i}">Abrir no simulador</button>`)).join("")}</tr>`;
  $("#cmpx-table").innerHTML = head + `<tbody>${feat}${priceRows}</tbody>`;
  $("#cmpx-table").querySelectorAll("[data-load]").forEach((b) => b.addEventListener("click", () => {
    const h = list[Number(b.dataset.load)];
    loadSample({ id: h.id, inputs: h.inputs, actual: h.actual, predicted: h.predicted, sale_condition: h.sale_condition, year_sold: h.year_sold });
    $("#content").scrollTo({ top: 0, behavior: reduceMotion() ? "auto" : "smooth" });
  }));

  const errs = list.map((h) => Math.abs(h.error_pct));
  const meanErr = errs.reduce((a, b) => a + b, 0) / errs.length;
  const meanReal = list.reduce((a, h) => a + h.actual, 0) / list.length;
  $("#cmpx-intro").textContent = "Casas vendidas de verdade com características próximas (de preferência no mesmo bairro). Nenhuma delas foi usada no treino do modelo. Valores sublinhados são bem diferentes da casa simulada (área com mais de 10% de diferença, ano com mais de 5, ou outro valor).";
  $("#cmpx-summary").innerHTML = `Nessas ${list.length} casas parecidas, o modelo errou em média <span class="num-txt">${nf(meanErr, 1)}%</span> (de ${nf(Math.min(...errs), 1)}% a ${nf(Math.max(...errs), 1)}%). `
    + `O preço médio de venda delas foi <span class="num-txt">${usd(meanReal)}</span>; a estimativa para a casa simulada é <span class="num-txt">${est ? usd(est) : "–"}</span>.`;
}

// ------------------------------------------------------------------ posição entre os preços da base
function percentileOf(v) {
  const h = S.priceHist;
  let below = 0, total = 0;
  h.counts.forEach((c, i) => {
    total += c;
    const a = h.edges[i], b = h.edges[i + 1];
    if (v >= b) below += c;
    else if (v > a) below += (c * (v - a)) / (b - a);
  });
  return (100 * below) / total;
}
function medianFromHist(h) {
  const total = h.counts.reduce((a, b) => a + b, 0);
  let acc = 0;
  for (let i = 0; i < h.counts.length; i++) {
    if (acc + h.counts[i] >= total / 2) return h.edges[i] + ((total / 2 - acc) / h.counts[i]) * (h.edges[i + 1] - h.edges[i]);
    acc += h.counts[i];
  }
  return h.edges[h.edges.length - 1];
}
function drawStrip() {
  const svg = $("#strip-svg");
  const h = S.priceHist;
  if (!svg || !h || !S.last) return;
  const W = Math.max(200, svg.clientWidth || 360), H = 64, base = 44;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const lo = h.edges[0], hi = h.edges[h.edges.length - 1];
  const x = (v) => ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * W;
  const maxC = Math.max(...h.counts);
  const v = S.last.predicted_sale_price;
  const bw = W / h.counts.length;
  const median = medianFromHist(h);
  const bars = h.counts.map((c, i) => {
    const bh = Math.max(1, (c / maxC) * 34);
    return `<rect class="bar${h.edges[i + 1] <= v ? " is-below" : ""}" x="${(i * bw + 0.5).toFixed(1)}" y="${(base - bh).toFixed(1)}" width="${Math.max(1, bw - 1).toFixed(1)}" height="${bh.toFixed(1)}" rx="1"/>`;
  }).join("");
  const ticks = [100000, 200000, 300000, 400000, 500000, 600000, 700000].filter((t) => t > lo && t < hi)
    .map((t) => `<text x="${x(t).toFixed(1)}" y="${H - 2}" text-anchor="middle">${t / 1000} mil</text>`).join("");
  const real = isPureSample() ? `<g class="real-mk" transform="translate(${x(S.origin.sample.actual).toFixed(1)},0)"><line y1="6" y2="${base}"/></g>` : "";
  const prev = svg.querySelector(".marker")?.getAttribute("transform");
  svg.innerHTML = `${bars}<line class="median" x1="${x(median)}" x2="${x(median)}" y1="4" y2="${base}"/>${ticks}${real}
    <g class="marker" transform="${prev || `translate(${x(v).toFixed(1)},0)`}"><line y1="4" y2="${base + 2}"/><circle cy="4" r="3.5"/></g>`;
  requestAnimationFrame(() => svg.querySelector(".marker").setAttribute("transform", `translate(${x(v).toFixed(1)},0)`));
  const pct = percentileOf(v);
  $("#strip-pct").textContent = `≈ ${nf(pct)}%`;
  $("#strip-desc").innerHTML = `A estimativa é maior que cerca de ${nf(pct)}% dos preços de venda da base (mediana: ${usd(median)}).`
    + `<span class="strip__legend"><span><i></i>estimativa do modelo</span><span><i class="med"></i>mediana</span>${real ? '<span><i class="rl"></i>preço real</span>' : ""}</span>`;
}

// ------------------------------------------------------------------ sensibilidade
function buildSensSelect() {
  const sel = $("#sens-field");
  sel.innerHTML = S.fields.map((f) => `<option value="${f.api_name}"${f.api_name === S.sensField ? " selected" : ""}>${esc(f.label)}</option>`).join("");
  sel.addEventListener("change", () => { S.sensField = sel.value; scheduleSens(0); });
}
function scheduleSens(delay) {
  clearTimeout(S.sensTimer);
  S.sensTimer = setTimeout(sens, delay);
}
async function sens() {
  S.sensCtl?.abort();
  const ctl = (S.sensCtl = new AbortController());
  try {
    const r = await api.sensitivity({ ...S.values }, S.sensField, ctl.signal);
    if (ctl.signal.aborted) return;
    $("#chart-sens").style.height = r.type === "category" ? `${Math.max(170, r.points.length * 19 + 46)}px` : "";
    charts.sensitivity("chart-sens", r, (v) => {
      const f = S.byApi[S.sensField];
      const val = f.type === "integer" ? Math.round(Number(v)) : f.type === "number" ? Number(v) : String(v);
      if (f.type === "category" && !f.options.some((o) => o.value === val)) return;
      if (f.type !== "category" && (val < f.min - 0.1 || val > f.max + 0.1)) return;
      if (val === S.values[f.api_name]) return;
      S.controls[f.api_name].set(val);
      onFieldChange(f.api_name, val);
      if (!S.live) refresh(0);
      toast(`${f.label}: ${fmtField(f, val)}`);
    });
    const lo = Math.min(...r.points.map((p) => p.predicted)), hi = Math.max(...r.points.map((p) => p.predicted));
    $("#sens-note").textContent = `Mantendo o resto da casa, a estimativa vai de ${usd(lo)} a ${usd(hi)} ao variar "${r.label}" dentro dos valores vistos no treino. O ponto destacado é a casa atual; clique em outro ponto para usar aquele valor. Mostra como o modelo reage, não uma relação de causa.`;
  } catch (e) {
    if (e.name !== "AbortError") $("#sens-note").textContent = `Não foi possível calcular a curva: ${e.message}`;
  }
}

// ------------------------------------------------------------------ cenários
function renderScenario() {
  const body = $("#scen-body");
  const btn = $("#btn-pin");
  if (!S.pinned) {
    btn.querySelector("span").textContent = "Fixar como A";
    btn.classList.remove("is-on");
    body.innerHTML = `<p class="fine">Fixe a casa atual como cenário A, mude o que quiser e veja quanto cada mudança altera a estimativa.</p>`;
    return;
  }
  btn.querySelector("span").textContent = "Desafixar A";
  btn.classList.add("is-on");
  const diff = S.fields.filter((f) => S.pinned.values[f.api_name] !== S.values[f.api_name]);
  const d = (S.last?.predicted_sale_price ?? 0) - S.pinned.price;
  if (!diff.length) {
    body.innerHTML = `<p class="fine">Cenário A fixado em <span class="num-txt">${usd(S.pinned.price)}</span>. Altere algum campo para ver o efeito de cada mudança.</p>
      <button class="btn btn--sm btn--ghost" type="button" id="btn-restore">Voltar para A</button>`;
  } else {
    body.innerHTML = `<div class="chart chart--wf" id="chart-waterfall" role="img" aria-label="Efeito de cada mudança na estimativa"></div>
      <table id="decomp-table">
        <thead><tr><th>Mudança</th><th>Efeito sozinho</th></tr></thead>
        <tbody>${diff.map((f) => `<tr><td>${esc(f.label)}: ${esc(fmtField(f, S.pinned.values[f.api_name]))} → ${esc(fmtField(f, S.values[f.api_name]))}</td><td data-col="${f.name}">…</td></tr>`).join("")}
        <tr><td>Interação entre as mudanças</td><td id="decomp-inter">…</td></tr>
        <tr class="total"><td>Total (A → atual)</td><td>${signedUsd(d)}</td></tr></tbody>
      </table>
      <p class="fine scen__note" id="decomp-method">Cada efeito é a estimativa de A trocando só aquela característica. A interação é o que sobra, porque o modelo combina as características.</p>
      <div class="detail__actions"><button class="btn btn--sm btn--ghost" type="button" id="btn-restore">Voltar para A</button></div>`;
    decompose();
  }
  $("#btn-restore").addEventListener("click", () => {
    applyValues(S.pinned.values, S.pinned.origin, S.pinned.reference);
    refresh(0);
  });
}

async function decompose() {
  S.decCtl?.abort();
  const ctl = (S.decCtl = new AbortController());
  try {
    const r = await api.decompose(S.pinned.values, { ...S.values }, ctl.signal);
    if (ctl.signal.aborted || !$("#decomp-table")) return;
    for (const e of r.effects) {
      const td = $(`#decomp-table [data-col="${e.column}"]`);
      if (td) td.textContent = e.effect === null ? "n/d" : signedUsd(e.effect);
    }
    $("#decomp-inter").textContent = signedUsd(r.interaction);
    charts.waterfall("chart-waterfall", r);
  } catch (e) {
    if (e.name !== "AbortError" && $("#decomp-method")) $("#decomp-method").textContent = `Não foi possível separar os efeitos: ${e.message}`;
  }
}

// ------------------------------------------------------------------ algoritmos lado a lado
function scheduleCompare(delay) {
  clearTimeout(S.cmpTimer);
  S.cmpTimer = setTimeout(compareAlgos, delay);
}
async function compareAlgos() {
  S.cmpCtl?.abort();
  const ctl = (S.cmpCtl = new AbortController());
  try {
    const r = await api.compare({ ...S.values }, ctl.signal);
    if (ctl.signal.aborted) return;
    S.lastCompare = r;
    renderAlgos();
  } catch (e) {
    if (e.name !== "AbortError") $("#algos-note").textContent = `Comparação indisponível: ${e.message}`;
  }
}
function renderAlgos() {
  const r = S.lastCompare;
  if (!r) return;
  const real = isPureSample() ? S.origin.sample.actual : null;
  charts.algorithms("chart-algos", r.rows, real);
  const preds = r.rows.filter((x) => !x.is_baseline).map((x) => x.predicted);
  let txt = `Os algoritmos treinados chegam a estimativas que variam ${usd(Math.max(...preds) - Math.min(...preds))} para esta casa.`;
  if (real) {
    const best = [...r.rows].sort((a, b) => Math.abs(a.predicted - real) - Math.abs(b.predicted - real))[0];
    txt += ` Mais perto do preço real (${usd(real)}): ${algoName(best.config)}, a ${usd(Math.abs(best.predicted - real))}.`;
  }
  $("#algos-note").textContent = txt + " Entre parênteses, o erro médio de cada um nas casas de teste.";
}

// ------------------------------------------------------------------ ações
function bindToolbar() {
  const more = $("#more");
  more?.addEventListener("toggle", () => { if (more.open) requestAnimationFrame(() => charts.resizeIn(more)); });
  $("#btn-typical").addEventListener("click", () => { S.mine = null; applyValues(typicalValues(), { type: "typical" }); refresh(0); });
  $("#btn-sample").addEventListener("click", () => loadRandomSample());
  $("#btn-pin").addEventListener("click", () => {
    if (S.pinned) S.pinned = null;
    else if (S.last) S.pinned = { values: { ...S.values }, price: S.last.predicted_sale_price, origin: { ...S.origin }, reference: { ...S.reference } };
    renderScenario();
    renderCompare();
  });
  $("#live-toggle").addEventListener("change", (e) => {
    S.live = e.target.checked;
    $("#manual-submit").hidden = S.live;
    $("#live-dot").classList.toggle("is-off", !S.live);
    $("#live-dot").lastChild.textContent = S.live ? "ao vivo" : "manual";
    if (S.live) refresh(0);
  });
}

async function ensureSamples() {
  if (!S.samples) S.samples = (await api.samples()).samples;
  return S.samples;
}
async function loadRandomSample() {
  const btn = $("#btn-sample");
  btn.disabled = true;
  try {
    const list = await ensureSamples();
    let i;
    do { i = Math.floor(Math.random() * list.length); } while (list.length > 1 && i === S.sampleIdx);
    S.sampleIdx = i;
    loadSample(list[i]);
  } catch (e) {
    toast(e.message);
  } finally {
    btn.disabled = false;
  }
}

/** Carrega uma casa vendida. Se a casa atual foi montada pela pessoa, guarda para poder voltar. */
function loadSample(s) {
  if (!isPureSample()) S.mine = { values: { ...S.values }, origin: { ...S.origin }, reference: { ...S.reference } };
  applyValues({ ...s.inputs }, { type: "sample", sample: s });
  refresh(0);
}
function restoreMine() {
  if (!S.mine) return;
  const m = S.mine;
  S.mine = null;
  applyValues(m.values, m.origin, m.reference);
  refresh(0);
  toast("Voltamos para a casa que você montou.");
}

/** Abre uma casa do holdout pelo Id (usado pela aba Imóveis reais). */
export async function openSampleById(id) {
  const list = await ensureSamples();
  const s = list.find((x) => x.id === id);
  if (s) loadSample(s);
  return Boolean(s);
}

export function onThemeChange() { drawStrip(); }
