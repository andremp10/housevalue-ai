// Utilidades compartilhadas de formatação e vocabulário (uma fonte só para todas as telas).

export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];
export const nf = (v, d = 0) => Number(v).toLocaleString("pt-BR", { minimumFractionDigits: d, maximumFractionDigits: d });
export const usd = (v) => "US$ " + nf(v);
export const pct = (v, d = 0) => `${nf(100 * v, d)}%`;
export const signedPct = (v, d = 1) => `${v >= 0 ? "+" : "−"}${nf(Math.abs(v), d)}%`;
export const signedUsd = (v) => `${v >= 0 ? "+" : "−"}${usd(Math.abs(v))}`;
export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Nome do algoritmo em linguagem simples, igual em todas as telas. */
export function algoName(config) {
  const log = config.includes("log1p");
  let n = config
    .replace(" + log1p(y)", "")
    .replace("HistGradientBoostingRegressor (nativo, monotônico)", "Gradient Boosting com regras de bom senso")
    .replace("HistGradientBoostingRegressor (categorias nativas)", "Gradient Boosting (bairro como categoria)")
    .replace("HistGradientBoostingRegressor", "Gradient Boosting")
    .replace("RandomForestRegressor", "Random Forest")
    .replace("DummyRegressor", "Chute da mediana")
    .replace("Ridge", "Regressão linear (Ridge)");
  if (log && !n.startsWith("Gradient Boosting com") && !n.includes("(bairro")) n += " · preço em log";
  return n;
}

/** Situação do erro de uma estimativa em relação ao preço real. */
export function errorStatus(errPct) {
  const a = Math.abs(errPct);
  if (a <= 10) return { cls: "good", label: "Perto do real", long: "Perto do real (erro até 10%)" };
  if (a <= 20) return { cls: "ok", label: "Erro moderado", long: "Erro moderado (10% a 20%)" };
  return { cls: "bad", label: "Erro grande", long: "Erro grande (mais de 20%)" };
}

/** Tipos de venda do dataset, em português. */
export const SALE = {
  Normal: "venda normal",
  Abnorml: "venda anormal (ex.: execução de hipoteca)",
  Partial: "casa nova vendida antes de ficar pronta",
  Family: "venda entre parentes",
  Alloca: "venda de unidades vinculadas",
  AdjLand: "compra de terreno vizinho",
};
export const saleText = (code) => SALE[code] || code;

/** Valor de um campo do formulário (nomes/unidades da API) formatado para leitura. */
export function fmtField(field, v) {
  if (!field) return String(v);
  if (field.type === "category") return field.options.find((o) => o.value === v)?.label ?? String(v);
  if (field.unit === "m²") return `${nf(v)} m²`;
  if (field.unit === "nota 1–10") return `${v}/10`;
  return String(v);
}

export function toast(msg) {
  const t = $("#toast");
  if (!t) return;
  t.textContent = msg;
  t.classList.add("is-on");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.remove("is-on"), Math.min(6000, Math.max(2600, msg.length * 55)));
}
