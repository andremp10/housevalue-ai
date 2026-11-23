// Gráficos Plotly.js. Cores lidas dos tokens CSS e convertidas para rgb (o Plotly não lê OKLCH).
// Convenção: laranja (--real) só para preço real de venda; azul (--model) só para estimativas do
// modelo; verde/âmbar/vermelho só para a situação do erro; o resto em tons neutros.
import { algoName } from "./util.js";

const registry = new Map();
const stale = new Set();
const clicks = new Map();
const probe = document.createElement("canvas").getContext("2d", { willReadFrequently: true });

function rgba(color) {
  probe.clearRect(0, 0, 1, 1);
  probe.fillStyle = "#808080";
  probe.fillStyle = color;
  probe.fillRect(0, 0, 1, 1);
  return [...probe.getImageData(0, 0, 1, 1).data];
}
function token(name, alpha = 1) {
  const [r, g, b] = rgba(getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "gray");
  return alpha === 1 ? `rgb(${r},${g},${b})` : `rgba(${r},${g},${b},${alpha})`;
}
function theme() {
  return {
    real: token("--real"), model: token("--model"), modelBand: token("--model", 0.1), modelFill: token("--model", 0.14),
    accent: token("--accent"), ok: token("--ok"), warn: token("--warn"), err: token("--err"),
    ink: token("--text"), ink2: token("--text-2"), ink3: token("--text-3"), neutral: token("--text-3", 0.55),
    grid: token("--line-soft"), line: token("--line"), panel: token("--panel"), none: "rgba(0,0,0,0)",
  };
}

function layout(t, extra = {}) {
  return {
    paper_bgcolor: t.none,
    plot_bgcolor: t.none,
    font: { family: '"Segoe UI Variable Text","Segoe UI",system-ui,sans-serif', color: t.ink2, size: 12 },
    margin: { l: 60, r: 14, t: 8, b: 44 },
    hoverlabel: { bgcolor: t.panel, bordercolor: t.line, font: { color: t.ink, size: 12 } },
    xaxis: { gridcolor: t.grid, zerolinecolor: t.line, linecolor: t.line, tickfont: { color: t.ink3 }, automargin: true },
    yaxis: { gridcolor: t.grid, zerolinecolor: t.line, linecolor: t.line, tickfont: { color: t.ink3 }, automargin: true },
    showlegend: false,
    separators: ",.",
    ...extra,
  };
}
const merge = (base, over) => ({ ...base, ...over, xaxis: { ...base.xaxis, ...(over.xaxis || {}) }, yaxis: { ...base.yaxis, ...(over.yaxis || {}) } });
const config = { displaylogo: false, responsive: true, displayModeBar: false, locale: "pt-BR" };
const usdTxt = (v) => "US$ " + Math.round(v).toLocaleString("pt-BR");
const milTxt = (v) => `${Math.round(v / 1000).toLocaleString("pt-BR")} mil`;

export const plotlyReady = () => typeof window.Plotly !== "undefined";

// Gráficos em áreas ocultas (aba não ativa, painel recolhido) não são desenhados: o Plotly
// falha em alguns tipos (ex.: scattergl/WebGL) com tamanho zero. Ficam pendentes e são
// desenhados quando aparecem (resizeIn). Handlers de clique são revinculados a cada desenho.
const visible = (node) => node.offsetParent !== null && node.clientWidth > 0;

function redraw(id) {
  const node = document.getElementById(id);
  const build = registry.get(id);
  if (!node || !build) return;
  const { data, layout: lay } = build(theme());
  window.Plotly.react(node, data, lay, config);
  const onClick = clicks.get(id);
  if (onClick) {
    node.removeAllListeners?.("plotly_click");
    node.on("plotly_click", onClick);
  }
}
function draw(id, build, onClick = null) {
  const node = document.getElementById(id);
  if (!node) return;
  registry.set(id, build);
  if (onClick) clicks.set(id, onClick);
  if (!plotlyReady()) {
    node.innerHTML = '<div class="chart-fallback">A biblioteca de gráficos não carregou. Os números continuam nas tabelas.</div>';
    return;
  }
  if (!visible(node)) { stale.add(id); return; }
  stale.delete(id);
  redraw(id);
}
export function refreshAll() {
  if (!plotlyReady()) return;
  for (const id of registry.keys()) {
    const node = document.getElementById(id);
    if (!node) { registry.delete(id); continue; }
    if (visible(node)) { stale.delete(id); redraw(id); } else stale.add(id);
  }
}
export function resizeIn(root) {
  if (!plotlyReady()) return;
  for (const id of [...stale]) {
    const node = document.getElementById(id);
    if (node && root.contains(node) && visible(node)) { stale.delete(id); redraw(id); }
  }
  root.querySelectorAll(".js-plotly-plot").forEach((n) => { if (visible(n)) window.Plotly.Plots.resize(n); });
}

/** Até 4 marcas "redondas" no eixo, rotuladas em mil (ex.: "160 mil"). */
function milTicks(lo, hi) {
  const raw = (hi - lo) / 3;
  const step = [5000, 10000, 20000, 25000, 50000, 100000].find((s) => s >= raw) || 200000;
  const vals = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) vals.push(v);
  return { tickvals: vals, ticktext: vals.map(milTxt) };
}

// ------------------------------------------------------------------ Imóveis reais
export function realVsPredicted(id, rows, selectedId, onPick) {
  draw(id, (t) => {
    const lo = Math.min(...rows.map((r) => Math.min(r.actual, r.predicted))) * 0.92;
    const hi = Math.max(...rows.map((r) => Math.max(r.actual, r.predicted))) * 1.04;
    const sel = rows.find((r) => r.id === selectedId);
    const grp = (label, color, cond) => {
      const rs = rows.filter(cond);
      return { type: "scatter", mode: "markers", name: label, x: rs.map((r) => r.actual), y: rs.map((r) => r.predicted), customdata: rs.map((r) => r.id),
        text: rs.map((r) => `Id ${r.id} · ${r.neighborhood_label}<br>${Math.round(r.area_m2)} m² · qualidade ${r.overall_qual}`),
        marker: { color, size: 8, opacity: 0.8, line: { color: t.panel, width: 1 } },
        hovertemplate: "%{text}<br>Real (venda) <b>%{x:$,.0f}</b><br>Modelo <b>%{y:$,.0f}</b><extra>clique para ver</extra>" };
    };
    return {
      data: [
        { type: "scatter", mode: "lines", x: [lo, hi, hi, lo], y: [lo * 0.9, hi * 0.9, hi * 1.1, lo * 1.1], fill: "toself", fillcolor: t.modelBand, line: { width: 0 }, hoverinfo: "skip", showlegend: false },
        { type: "scatter", mode: "lines", x: [lo, hi], y: [lo, hi], line: { color: t.ink3, dash: "dash", width: 1.4 }, hoverinfo: "skip", showlegend: false },
        grp("Perto do real (até 10%)", t.ok, (r) => Math.abs(r.error_pct) <= 10),
        grp("Erro moderado (10% a 20%)", t.warn, (r) => Math.abs(r.error_pct) > 10 && Math.abs(r.error_pct) <= 20),
        grp("Erro grande (mais de 20%)", t.err, (r) => Math.abs(r.error_pct) > 20),
        ...(sel ? [{ type: "scatter", mode: "markers", x: [sel.actual], y: [sel.predicted], marker: { size: 18, color: t.none, line: { color: t.ink, width: 2.5 } }, hoverinfo: "skip", showlegend: false }] : []),
      ],
      layout: merge(layout(t, { margin: { l: 70, r: 12, t: 30, b: 48 }, showlegend: true, legend: { orientation: "h", x: 0, y: 1.08 } }),
        { xaxis: { title: { text: "Real: preço da venda (USD)" }, tickformat: "$,.0f", range: [lo, hi] }, yaxis: { title: { text: "Modelo: estimativa (USD)" }, tickformat: "$,.0f", range: [lo, hi] } }),
    };
  }, onPick ? (ev) => { const pt = ev.points?.find((p) => p.customdata !== undefined); if (pt) onPick(pt.customdata); } : null);
}

// ------------------------------------------------------------------ Simulador
/** Curva de sensibilidade: estimativas do modelo (azul); o ponto da casa atual em destaque. */
export function sensitivity(id, s, onPick) {
  const pick = onPick ? (ev) => { const pt = ev.points?.[0]; if (pt) onPick(pt.customdata !== undefined ? pt.customdata : pt.x); } : null;
  draw(id, (t) => {
    const cur = s.current.value;
    if (s.type !== "category") {
      const isYear = s.unit === "ano";
      return {
        data: [
          { type: "scatter", mode: "lines+markers", x: s.points.map((p) => p.value), y: s.points.map((p) => p.predicted), line: { color: t.model, width: 2 }, marker: { size: 6, color: t.model },
            hovertemplate: `${isYear ? "%{x}" : "%{x:,.0f}"}${s.unit === "m²" ? " m²" : ""}<br>estimativa <b>%{y:$,.0f}</b><extra>clique para usar</extra>` },
          { type: "scatter", mode: "markers", x: [cur], y: [s.current.predicted], marker: { size: 14, color: t.model, line: { color: t.ink, width: 2.5 } }, hovertemplate: "Casa atual<br><b>%{y:$,.0f}</b><extra></extra>" },
        ],
        layout: merge(layout(t, { margin: { l: 62, r: 10, t: 6, b: 38 } }), { xaxis: { title: { text: s.unit === "m²" ? `${s.label} (m²)` : s.label, font: { size: 12 } }, tickformat: isYear ? "d" : ",.0f" }, yaxis: { tickformat: "$,.0f", nticks: 5 } }),
      };
    }
    const pts = [...s.points].sort((a, b) => a.predicted - b.predicted);
    const lo = Math.min(...pts.map((p) => p.predicted)), hi = Math.max(...pts.map((p) => p.predicted));
    const pad = Math.max((hi - lo) * 0.08, 2000);
    return {
      data: [
        { type: "scatter", mode: "lines", x: pts.flatMap((p) => [lo - pad, p.predicted, null]), y: pts.flatMap((p) => [p.label, p.label, null]), line: { color: t.grid, width: 1 }, hoverinfo: "skip" },
        { type: "scatter", mode: "markers", y: pts.map((p) => p.label), x: pts.map((p) => p.predicted), customdata: pts.map((p) => p.value),
          marker: { size: pts.map((p) => (p.value === cur ? 13 : 8)), color: t.model, line: { color: pts.map((p) => (p.value === cur ? t.ink : t.panel)), width: pts.map((p) => (p.value === cur ? 2.5 : 1)) } },
          hovertemplate: "%{y}<br>estimativa <b>%{x:$,.0f}</b><extra>clique para usar</extra>" },
      ],
      layout: merge(layout(t, { margin: { l: 8, r: 14, t: 6, b: 30 }, height: Math.max(170, pts.length * 19 + 46) }), { xaxis: { tickformat: "$,.0f", range: [lo - pad, hi + pad], nticks: 4 }, yaxis: { gridcolor: t.none, tickfont: { size: 12 }, type: "category" } }),
    };
  }, pick);
}

/** Algoritmos lado a lado: estimativa de cada um (azul); linha laranja = preço real, se houver. */
export function algorithms(id, rows, real) {
  draw(id, (t) => {
    const labels = rows.map((r) => `${algoName(r.config)}${r.holdout_mae ? ` (${milTxt(r.holdout_mae)})` : ""}`);
    const xs = rows.map((r) => r.predicted).concat(real ? [real] : []);
    const lo = Math.min(...xs), hi = Math.max(...xs), pad = Math.max((hi - lo) * 0.1, 3000);
    return {
      data: [
        { type: "scatter", mode: "lines", x: rows.flatMap((r) => [lo - pad, r.predicted, null]), y: labels.flatMap((l) => [l, l, null]), line: { color: t.grid, width: 1 }, hoverinfo: "skip" },
        { type: "scatter", mode: "markers", y: labels, x: rows.map((r) => r.predicted),
          marker: { size: rows.map((r) => (r.is_selected ? 13 : 9)), color: rows.map((r) => (r.is_baseline ? t.ink3 : t.model)), line: { color: rows.map((r) => (r.is_selected ? t.ink : t.panel)), width: rows.map((r) => (r.is_selected ? 2.5 : 1)) } },
          customdata: rows.map((r) => (r.is_selected ? "modelo em uso" : r.is_baseline ? "referência mínima" : "outro algoritmo")),
          hovertemplate: "%{y}<br>estimativa <b>%{x:$,.0f}</b><br>%{customdata}<extra></extra>" },
      ],
      layout: merge(layout(t, { margin: { l: 8, r: 14, t: real ? 22 : 6, b: 30 }, height: Math.max(190, rows.length * 28 + 50),
        shapes: real ? [{ type: "line", x0: real, x1: real, yref: "paper", y0: 0, y1: 1, line: { color: t.real, width: 2.5, dash: "dash" } }] : [],
        annotations: real ? [{ x: real, yref: "paper", y: 1.02, text: "preço real", showarrow: false, font: { size: 12, color: t.ink2 }, yanchor: "bottom" }] : [] }),
        { xaxis: { range: [lo - pad, hi + pad], tickangle: 0, ...milTicks(lo - pad, hi + pad) }, yaxis: { type: "category", gridcolor: t.none, tickfont: { size: 12 } } }),
    };
  });
}

function wfRange(d) {
  let level = d.base_price;
  const levels = [level];
  for (const e of d.effects) if (e.effect !== null) { level += e.effect; levels.push(level); }
  levels.push(d.target_price);
  const lo = Math.min(...levels), hi = Math.max(...levels), pad = Math.max((hi - lo) * 0.35, 4000);
  return [Math.max(0, lo - pad), hi + pad];
}
/** Cascata: estimativa de A (azul) → efeito de cada mudança (neutro) → interação → estimativa atual (azul). */
export function waterfall(id, d) {
  draw(id, (t) => {
    const eff = d.effects.filter((e) => e.effect !== null);
    const inter = Math.abs(d.interaction) >= 1;
    const x = ["Cenário A", ...eff.map((e) => e.label), ...(inter ? ["Interação"] : []), "Atual"];
    const y = [d.base_price, ...eff.map((e) => e.effect), ...(inter ? [d.interaction] : []), d.target_price];
    const measure = ["absolute", ...eff.map(() => "relative"), ...(inter ? ["relative"] : []), "total"];
    return {
      data: [{
        type: "waterfall", x, y, measure, connector: { line: { color: t.line, width: 1 } },
        increasing: { marker: { color: t.ink3 } }, decreasing: { marker: { color: t.neutral } }, totals: { marker: { color: t.model } },
        text: y.map((v, i) => (measure[i] === "relative" ? `${v >= 0 ? "+" : "−"}${milTxt(Math.abs(v))}` : milTxt(v))),
        textposition: "outside", textfont: { size: 12, color: t.ink2 },
        hovertemplate: "%{x}<br><b>%{y:$,.0f}</b><extra></extra>",
      }],
      layout: merge(layout(t, { margin: { l: 62, r: 10, t: 20, b: 74 } }), { xaxis: { tickangle: -25, tickfont: { size: 12 } }, yaxis: { tickformat: "$,.0f", nticks: 5, range: wfRange(d) } }),
    };
  });
}

// ------------------------------------------------------------------ Como funciona
export function histogram(id, hist, { xTitle } = {}) {
  draw(id, (t) => {
    const centers = hist.edges.slice(0, -1).map((e, i) => (e + hist.edges[i + 1]) / 2);
    const widths = hist.edges.slice(0, -1).map((e, i) => (hist.edges[i + 1] - e) * 0.9);
    const labels = hist.edges.slice(0, -1).map((e, i) => `${usdTxt(e)} a ${usdTxt(hist.edges[i + 1])}`);
    return {
      data: [{ type: "bar", x: centers, y: hist.counts, width: widths, marker: { color: t.real }, customdata: labels, hovertemplate: "%{customdata}<br><b>%{y}</b> casas<extra></extra>" }],
      layout: merge(layout(t), { xaxis: { title: { text: xTitle }, tickformat: "$,.0f" }, yaxis: { title: { text: "Casas" } }, bargap: 0 }),
    };
  });
}

export function scatterArea(id, s) {
  const K = 0.09290304; // dados originais em pés²; exibidos em m²
  draw(id, (t) => {
    const xm = s.x.map((v) => v * K);
    const big = s.x.map((v) => v > 4000);
    const pick = (arr, cond) => arr.filter((_, i) => big[i] === cond);
    const cd = s.id.map((v, i) => [v, s.overall_qual[i]]);
    const tmpl = "Id %{customdata[0]}<br>%{x:,.0f} m² · qualidade %{customdata[1]}<br>vendida por <b>%{y:$,.0f}</b><extra></extra>";
    return {
      data: [
        { type: "scattergl", mode: "markers", name: "Casas vendidas", x: pick(xm, false), y: pick(s.y, false), customdata: pick(cd, false), marker: { color: t.real, size: 6, opacity: 0.45 }, hovertemplate: tmpl },
        { type: "scatter", mode: "markers", name: "Mais de 370 m²", x: pick(xm, true), y: pick(s.y, true), customdata: pick(cd, true), marker: { color: t.none, size: 12, line: { color: t.ink, width: 2 } }, hovertemplate: tmpl },
      ],
      layout: merge(layout(t, { showlegend: true, legend: { orientation: "h", y: 1.1, x: 0 } }), { xaxis: { title: { text: "Área construída acima do solo (m²)" } }, yaxis: { title: { text: "Preço de venda (USD)" }, tickformat: "$,.0f" } }),
    };
  });
}

/** Barras horizontais neutras (ex.: importância das características). */
export function hbar(id, labels, values, { xTitle = "", format = ".3f", errors = null, hoverFmt = null } = {}) {
  draw(id, (t) => {
    const trace = { type: "bar", orientation: "h", y: labels, x: values, marker: { color: t.accent }, textposition: "none", hovertemplate: `%{y}<br><b>%{x:${hoverFmt || format}}</b><extra></extra>` };
    if (errors) trace.error_x = { type: "data", array: errors, color: t.ink3, thickness: 1.2, width: 3 };
    return { data: [trace], layout: merge(layout(t, { margin: { l: 10, r: 16, t: 6, b: 44 } }), { xaxis: { title: { text: xTitle }, tickformat: format }, yaxis: { gridcolor: t.none, tickfont: { color: t.ink2, size: 12.5 } } }) };
  });
}

/** Viés por faixa de preço, no mesmo sinal do resto do sistema: modelo − real (positivo = superestima). */
export function biasBars(id, rows) {
  draw(id, (t) => {
    const y = rows.map((r) => -r.bias);
    const lim = Math.max(...y.map(Math.abs)) * 1.4;
    return {
      data: [{
        type: "bar", x: rows.map((r) => r.segment.split(":")[0]), y, marker: { color: t.neutral },
        customdata: rows.map((r) => [r.segment, r.n, r.mape_pct.toFixed(1)]),
        text: y.map((v) => `${v >= 0 ? "superestima" : "subestima"} ${usdTxt(Math.abs(v))}`), textposition: "outside", cliponaxis: false, textfont: { size: 12, color: t.ink2 },
        hovertemplate: "%{customdata[0]}<br>modelo − real, em média: <b>%{y:$,.0f}</b><br>erro típico %{customdata[2]}% · %{customdata[1]} casas<extra></extra>",
      }],
      layout: merge(layout(t, { margin: { l: 64, r: 12, t: 22, b: 40 }, shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: t.ink3, width: 1 } }] }),
        { xaxis: { title: { text: "Faixa de preço real (quartis das casas de teste)" } }, yaxis: { title: { text: "Modelo − real (USD)" }, tickformat: "$,.0f", zeroline: false, range: [-lim, lim] } }),
    };
  });
}

/** Erro médio de cada algoritmo nas casas de teste, do menor para o maior. */
export function algoMae(id, rows) {
  draw(id, (t) => {
    const r = [...rows].sort((a, b) => b.holdout.mae - a.holdout.mae);
    return {
      data: [{ type: "bar", orientation: "h", y: r.map((x) => algoName(x.config)), x: r.map((x) => x.holdout.mae),
        marker: { color: r.map((x) => (x.is_selected ? t.accent : x.is_baseline ? t.ink3 : t.neutral)) },
        text: r.map((x) => `US$ ${(x.holdout.mae / 1000).toLocaleString("pt-BR", { maximumFractionDigits: 1 })} mil${x.is_selected ? " · em uso" : ""}`), textposition: "outside", cliponaxis: false, textfont: { size: 12, color: t.ink2 },
        hovertemplate: "%{y}<br>erro médio <b>%{x:$,.0f}</b><extra></extra>" }],
      layout: merge(layout(t, { margin: { l: 10, r: 110, t: 6, b: 40 } }), { xaxis: { title: { text: "Erro médio nas casas de teste (USD); menor é melhor" }, tickformat: "$,.0f" }, yaxis: { gridcolor: t.none } }),
    };
  });
}

/** Vencedor ou empate: erro de cada partição para os dois melhores, ligados por partição. */
export function tieDots(id, tie) {
  draw(id, (t) => {
    const [a, b] = tie.rmsle;
    const names = tie.models.map(algoName);
    return {
      data: [
        { type: "scatter", mode: "lines", x: a.flatMap((v, i) => [v, b[i], null]), y: a.flatMap((_, i) => [i + 1, i + 1, null]), line: { color: t.grid, width: 1.5 }, hoverinfo: "skip", showlegend: false },
        { type: "scatter", mode: "markers", name: names[0], x: a, y: a.map((_, i) => i + 1), marker: { color: t.accent, size: 9 }, hovertemplate: `${names[0]}<br>partição %{y}: <b>%{x:.4f}</b><extra></extra>` },
        { type: "scatter", mode: "markers", name: names[1], x: b, y: b.map((_, i) => i + 1), marker: { color: t.ink3, size: 9, symbol: "diamond" }, hovertemplate: `${names[1]}<br>partição %{y}: <b>%{x:.4f}</b><extra></extra>` },
      ],
      layout: merge(layout(t, { showlegend: true, legend: { orientation: "h", y: 1.1, x: 0 }, margin: { l: 46, r: 12, t: 34, b: 40 } }),
        { xaxis: { title: { text: "Erro na partição (RMSLE; menor é melhor)" }, tickformat: ".3f" }, yaxis: { title: { text: "Partição" }, dtick: a.length > 10 ? 4 : 1, autorange: "reversed" } }),
    };
  });
}

/** Curva de aprendizado: um tom só; treino em linha contínua, validação tracejada. */
export function learningCurve(id, lc) {
  draw(id, (t) => {
    const xs = lc.train_sizes;
    const band = lc.val_rmsle_mean.map((v, i) => v + lc.val_rmsle_std[i]).concat(lc.val_rmsle_mean.map((v, i) => v - lc.val_rmsle_std[i]).reverse());
    return {
      data: [
        { type: "scatter", mode: "lines", x: xs.concat([...xs].reverse()), y: band, fill: "toself", fillcolor: t.modelFill, line: { width: 0 }, hoverinfo: "skip", showlegend: false },
        { type: "scatter", mode: "lines+markers", name: "Treino", x: xs, y: lc.train_rmsle_mean, line: { color: t.model, width: 2 }, marker: { size: 6 }, hovertemplate: "%{x} casas<br>treino <b>%{y:.4f}</b><extra></extra>" },
        { type: "scatter", mode: "lines+markers", name: "Validação", x: xs, y: lc.val_rmsle_mean, line: { color: t.model, width: 2, dash: "dash" }, marker: { size: 6, symbol: "diamond" }, hovertemplate: "%{x} casas<br>validação <b>%{y:.4f}</b><extra></extra>" },
      ],
      layout: merge(layout(t, { showlegend: true, legend: { orientation: "h", y: 1.1, x: 0 }, margin: { l: 56, r: 12, t: 34, b: 44 } }),
        { xaxis: { title: { text: "Casas usadas no treino" } }, yaxis: { title: { text: "Erro (RMSLE)" }, tickformat: ".3f", rangemode: "tozero" } }),
    };
  });
}
