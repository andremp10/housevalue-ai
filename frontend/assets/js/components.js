import { esc, nf } from "./util.js";
// Componentes de entrada do simulador. Cada um recebe o metadado do campo vindo de
// /api/form-options (limites, opções e distribuição de treino) e expõe:
//   { el, set(value), setError(msg), setChanged(bool) }
// e chama onChange(apiName, value) quando a pessoa altera o valor.

const fmtVal = (f, v) => (f.unit === "ano" ? String(Math.round(v)) : nf(v));
// m² vindos de casas reais têm muitas casas decimais (conversão exata de pés²): exibe 1 casa,
// mas o valor exato continua sendo o enviado ao modelo enquanto a pessoa não editar o campo.
const shownNum = (f, v) => (f.type === "number" ? Math.round(v * 10) / 10 : v);
const unitShort = (f) => (f.unit === "m²" ? "m²" : "");
let uid = 0;

function shell(f, { wide = false, labelFor = null, extraTop = "" } = {}) {
  const id = `fld-${f.api_name}`;
  const el = document.createElement("div");
  el.className = "field" + (wide ? " field--wide" : "");
  el.dataset.field = f.api_name;
  const labelTag = labelFor ? `label for="${labelFor}"` : `span id="${id}-label"`;
  const closeTag = labelFor ? "label" : "span";
  el.innerHTML = `
    <div class="field__top">
      <${labelTag} class="field__label" title="Nome no dataset original: ${esc(f.name)}">${esc(f.label)}</${closeTag}>
      ${extraTop}
    </div>
    <div class="field__body"></div>
    <p class="field__help" id="${id}-help">${esc(f.help)}</p>
    <p class="field__err" id="${id}-err" hidden></p>`;
  const api = {
    el,
    body: el.querySelector(".field__body"),
    setError(msg) {
      const e = el.querySelector(".field__err");
      e.hidden = !msg;
      e.textContent = msg || "";
      el.classList.toggle("has-error", Boolean(msg));
    },
    setChanged(on) { el.classList.toggle("is-changed", Boolean(on)); },
  };
  return api;
}

// --------------------------------------------------------------- número + slider
function niceStep(span) {
  const raw = span / 220;
  if (raw <= 1) return 1;
  const p = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw) || p * 10;
}

export function rangeField(f, onChange) {
  const inputId = `in-${f.api_name}`;
  const unit = unitShort(f);
  const s = shell(f, {
    wide: true,
    extraTop: `<span class="numbox"><input id="${inputId}" type="number" inputmode="${f.type === "number" ? "decimal" : "numeric"}" min="${f.min}" max="${f.max}" step="${f.type === "number" ? "any" : "1"}" aria-label="${esc(f.label)} (valor exato)" aria-describedby="fld-${f.api_name}-help fld-${f.api_name}-err">${unit ? `<span class="num__unit">${unit}</span>` : ""}</span>`,
  });
  const d = f.distribution;
  const [lo, hi] = d ? d.common_range : [f.min, f.max];
  const step = f.unit === "ano" ? 1 : niceStep(hi - lo);
  const counts = d ? d.histogram.counts : [];
  const edges = d ? d.histogram.edges : [];
  const maxC = Math.max(1, ...counts);
  s.body.innerHTML = `
    <div class="range">
      <div class="range__hist" aria-hidden="true">${counts.map((c) => `<i style="height:${Math.max(4, (100 * c) / maxC)}%"></i>`).join("")}</div>
      <input type="range" min="${lo}" max="${hi}" step="${step}" aria-label="${esc(f.label)}" aria-describedby="fld-${f.api_name}-help">
    </div>
    <div class="range__scale"><span>${fmtVal(f, lo)}</span><span class="range__mid"></span><span class="range__max">${fmtVal(f, hi)}${hi < f.max ? "+" : ""}</span></div>`;
  const num = s.el.querySelector(`#${inputId}`);
  const slider = s.body.querySelector("input[type=range]");
  const bars = [...s.body.querySelectorAll(".range__hist i")];
  const mid = s.body.querySelector(".range__mid");
  let value = null;

  function paint(v) {
    const clamped = Math.min(Math.max(v, lo), hi);
    slider.value = clamped;
    slider.style.setProperty("--p", `${(100 * (clamped - lo)) / (hi - lo || 1)}%`);
    bars.forEach((b, i) => b.classList.toggle("is-in", edges[i] <= v));
    slider.setAttribute("aria-valuetext", `${fmtVal(f, v)}${unit ? " " + unit : ""}`);
    if (v > hi) { mid.innerHTML = `<span class="over">acima da faixa comum (97,5% dos imóveis)</span>`; }
    else if (d) {
      const p = d.percentiles;
      const pos = v <= p["25"] ? "entre os 25% menores" : v <= p["50"] ? "abaixo da mediana" : v <= p["75"] ? "acima da mediana" : v <= p["95"] ? "entre os 25% maiores" : "entre os 5% maiores";
      mid.textContent = pos;
    }
  }
  function commit(v, source) {
    value = v;
    paint(v);
    if (source !== "num") num.value = shownNum(f, v);
    onChange(f.api_name, v);
  }
  slider.addEventListener("input", () => { s.setError(""); commit(Number(slider.value), "slider"); });
  const fromNum = () => {
    const raw = num.value.trim();
    const v = Number(raw);
    if (raw === "" || !Number.isFinite(v)) return s.setError("Informe um número.");
    if (f.type !== "number" && !Number.isInteger(v)) return s.setError("Informe um número inteiro.");
    if (v < f.min || v > f.max) return s.setError(`Aceito de ${fmtVal(f, f.min)} a ${fmtVal(f, f.max)}${unit ? " " + unit : ""} (faixa dos dados de treino).`);
    s.setError("");
    if (v !== value) commit(v, "num");
  };
  num.addEventListener("change", fromNum);
  num.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); fromNum(); } });
  return Object.assign(s, {
    set(v) { value = v; num.value = shownNum(f, v); paint(v); s.setError(""); },
    disable(on) { slider.disabled = on; num.disabled = on; s.el.style.opacity = on ? 0.6 : 1; },
  });
}

// --------------------------------------------------------------- nota 1..10
const QUALITY = { 1: "Muito ruim", 2: "Ruim", 3: "Razoável", 4: "Abaixo da média", 5: "Média", 6: "Acima da média", 7: "Boa", 8: "Muito boa", 9: "Excelente", 10: "Excepcional" };

export function ratingField(f, onChange) {
  const s = shell(f, { wide: true, extraTop: `<span class="rating__now" aria-hidden="true"></span>` });
  const name = `r-${f.api_name}-${++uid}`;
  const d = f.distribution;
  const countAt = (k) => {
    if (!d) return 0;
    const e = d.histogram.edges;
    const i = e.findIndex((x, j) => j < e.length - 1 && k > x && k < e[j + 1]);
    return i >= 0 ? d.histogram.counts[i] : 0;
  };
  const maxC = Math.max(1, ...Array.from({ length: 10 }, (_, i) => countAt(i + 1)));
  s.body.innerHTML = `
    <div class="rating" role="radiogroup" aria-labelledby="fld-${f.api_name}-label" aria-describedby="fld-${f.api_name}-help">
      ${Array.from({ length: 10 }, (_, i) => {
        const k = i + 1;
        const off = k < f.min || k > f.max;
        const c = countAt(k);
        return `<label title="${k}: ${QUALITY[k]}${off ? " (sem imóveis no treino)" : ` · ${c} imóveis no treino`}">
          <input type="radio" name="${name}" value="${k}" ${off ? "disabled" : ""} aria-label="${k}, ${QUALITY[k]}${off ? ", indisponível: sem imóveis no treino" : ""}">
          <span><b style="height:${(100 * c) / maxC}%"></b>${k}</span></label>`;
      }).join("")}
    </div>
    <div class="rating__legend"><span>1 · muito ruim</span><span>10 · excepcional</span></div>`;
  const radios = [...s.body.querySelectorAll("input")];
  const now = s.el.querySelector(".rating__now");
  function paint(v) {
    radios.forEach((r) => { r.checked = Number(r.value) === v; r.classList.toggle("is-filled", Number(r.value) < v); });
    now.textContent = `${v} · ${QUALITY[v]}`;
  }
  radios.forEach((r) => r.addEventListener("change", () => { const v = Number(r.value); paint(v); onChange(f.api_name, v); }));
  s.help = s.el.querySelector(".field__help");
  return Object.assign(s, { set(v) { paint(v); s.setError(""); } });
}

// --------------------------------------------------------------- stepper
export function stepperField(f, onChange) {
  const s = shell(f);
  const d = f.distribution;
  const vals = Array.from({ length: f.max - f.min + 1 }, (_, i) => f.min + i);
  const countAt = (k) => {
    if (!d) return 0;
    const e = d.histogram.edges;
    const i = e.findIndex((x, j) => j < e.length - 1 && k > x && k < e[j + 1]);
    return i >= 0 ? d.histogram.counts[i] : 0;
  };
  const maxC = Math.max(1, ...vals.map(countAt));
  s.body.innerHTML = `
    <div class="stepper-row">
      <div class="stepper" role="group" aria-labelledby="fld-${f.api_name}-label">
        <button type="button" data-d="-1" aria-label="Diminuir ${esc(f.label)}">−</button>
        <output aria-live="polite" aria-labelledby="fld-${f.api_name}-label"></output>
        <button type="button" data-d="1" aria-label="Aumentar ${esc(f.label)}">+</button>
      </div>
      <div class="mini-dist" aria-hidden="true" title="Quantos imóveis do treino têm cada valor">${vals.map((k) => `<i data-k="${k}" style="height:${Math.max(6, (100 * countAt(k)) / maxC)}%" title="${k}: ${countAt(k)} imóveis"></i>`).join("")}</div>
    </div>`;
  const out = s.body.querySelector("output");
  const [minus, plus] = s.body.querySelectorAll("button");
  const bars = [...s.body.querySelectorAll(".mini-dist i")];
  let value = f.min;
  function paint(v, animate) {
    out.textContent = v;
    minus.disabled = v <= f.min;
    plus.disabled = v >= f.max;
    bars.forEach((b) => b.classList.toggle("is-on", Number(b.dataset.k) === v));
    if (animate) { out.classList.remove("bump"); void out.offsetWidth; out.classList.add("bump"); }
  }
  s.body.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    const v = Math.min(f.max, Math.max(f.min, value + Number(b.dataset.d)));
    if (v === value) return;
    value = v; paint(v, true); onChange(f.api_name, v);
  }));
  s.el.querySelector(".field__help").textContent = `${f.help} De ${f.min} a ${f.max}.`;
  return Object.assign(s, { set(v) { value = v; paint(v, false); s.setError(""); } });
}

// --------------------------------------------------------------- segmentado
const ORDER = { kitchen_qual: ["Fa", "TA", "Gd", "Ex"], central_air: ["N", "Y"] };

export function segmentField(f, onChange) {
  const s = shell(f);
  const name = `s-${f.api_name}-${++uid}`;
  const order = ORDER[f.api_name];
  const opts = order ? order.map((v) => f.options.find((o) => o.value === v)).filter(Boolean) : f.options;
  s.body.innerHTML = `<div class="seg seg--fill" role="radiogroup" aria-labelledby="fld-${f.api_name}-label">
    ${opts.map((o) => `<label title="${esc(o.label)} · código ${esc(o.value)} · ${o.n_train} imóveis no treino"><input type="radio" name="${name}" value="${esc(o.value)}"><span>${esc(o.label.replace("Típico/médio", "Típica"))}</span></label>`).join("")}
  </div>`;
  const radios = [...s.body.querySelectorAll("input")];
  radios.forEach((r) => r.addEventListener("change", () => onChange(f.api_name, r.value)));
  return Object.assign(s, { set(v) { radios.forEach((r) => (r.checked = r.value === v)); s.setError(""); } });
}

// --------------------------------------------------------------- combobox de bairros
export function comboField(f, onChange, stats) {
  const inputId = `cb-${f.api_name}`;
  const listId = `lb-${f.api_name}`;
  const s = shell(f, { wide: true, labelFor: inputId });
  const items = f.options.map((o) => ({ ...o, median: stats?.[o.value]?.median ?? null, n: stats?.[o.value]?.n ?? o.n_train }));
  items.sort((a, b) => (b.median ?? 0) - (a.median ?? 0));
  const maxMed = Math.max(1, ...items.map((i) => i.median || 0));
  s.body.innerHTML = `
    <div class="combo">
      <div class="combo__control">
        <svg class="ic"><use href="#i-search"/></svg>
        <input id="${inputId}" type="text" role="combobox" aria-autocomplete="list" aria-expanded="false" aria-controls="${listId}" autocomplete="off" spellcheck="false" aria-describedby="fld-${f.api_name}-help">
        <span class="combo__code" aria-hidden="true"></span>
        <svg class="ic"><use href="#i-chev"/></svg>
      </div>
      <ul class="combo__list" id="${listId}" role="listbox" aria-label="${esc(f.label)}" hidden></ul>
    </div>`;
  s.el.querySelector(".field__help").textContent = "Digite para filtrar. A barra mostra a mediana de preço das vendas do bairro na base, para comparação.";
  const input = s.body.querySelector("input");
  const list = s.body.querySelector("ul");
  const code = s.body.querySelector(".combo__code");
  let value = null, active = -1, shown = [];

  const label = (v) => items.find((i) => i.value === v)?.label ?? v;
  function render(filter = "") {
    const q = filter.trim().toLowerCase();
    shown = items.filter((i) => !q || i.label.toLowerCase().includes(q) || i.value.toLowerCase().includes(q));
    list.innerHTML = `<li class="combo__head" role="presentation" aria-hidden="true"><span>Bairro</span><span>Mediana na base</span><span>USD</span></li>` +
      (shown.length ? shown.map((i, k) => `
        <li class="combo__opt${k === active ? " is-active" : ""}" role="option" id="${listId}-${k}" data-v="${esc(i.value)}" aria-selected="${i.value === value}">
          <span class="combo__name">${esc(i.label)}<small>${esc(i.value)} · ${i.n} vendas</small></span>
          <span class="combo__bar"><i style="width:${i.median ? (100 * i.median) / maxMed : 0}%"></i></span>
          <span class="combo__val">${i.median ? nf(i.median / 1000) + " mil" : "–"}</span>
        </li>`).join("") : `<li class="combo__empty" role="presentation">Nenhum bairro encontrado.</li>`);
    input.setAttribute("aria-activedescendant", active >= 0 && shown[active] ? `${listId}-${active}` : "");
    list.querySelector(".is-active")?.scrollIntoView({ block: "nearest" });
  }
  function open() {
    if (!list.hidden) return;
    list.hidden = false; input.setAttribute("aria-expanded", "true");
    active = Math.max(0, items.findIndex((i) => i.value === value));
    input.select();
    render("");
  }
  function close(restore = true) {
    list.hidden = true; input.setAttribute("aria-expanded", "false"); active = -1;
    if (restore) input.value = label(value);
  }
  function choose(v) {
    value = v; input.value = label(v); code.textContent = v; close(false); s.setError(""); onChange(f.api_name, v);
  }
  input.addEventListener("click", open);
  input.addEventListener("input", () => { if (list.hidden) open(); active = 0; render(input.value); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (list.hidden) return open();
      active = Math.max(0, Math.min(shown.length - 1, active + (e.key === "ArrowDown" ? 1 : -1)));
      render(input.value === label(value) ? "" : input.value);
    } else if (e.key === "Enter") {
      if (!list.hidden && shown[active]) { e.preventDefault(); choose(shown[active].value); }
    } else if (e.key === "Escape") {
      if (!list.hidden) { e.preventDefault(); close(); }
    } else if (e.key === "Tab") {
      close();
    }
  });
  list.addEventListener("mousedown", (e) => e.preventDefault());
  list.addEventListener("click", (e) => { const li = e.target.closest("[data-v]"); if (li) choose(li.dataset.v); });
  input.addEventListener("blur", () => setTimeout(() => close(), 0));
  return Object.assign(s, { set(v) { value = v; input.value = label(v); code.textContent = v; s.setError(""); } });
}

export function controlFor(f, onChange, ctx) {
  if (f.api_name === "neighborhood") return comboField(f, onChange, ctx.neighborhoodStats);
  if (f.type === "category") return segmentField(f, onChange);
  if (f.unit === "nota 1–10") return ratingField(f, onChange);
  if (f.type === "integer" && f.max - f.min <= 12) return stepperField(f, onChange);
  return rangeField(f, onChange);
}

export { QUALITY };
