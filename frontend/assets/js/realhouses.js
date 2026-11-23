// Aba "Imóveis reais": roda o modelo nas casas vendidas do holdout (nunca vistas no treino)
// e compara cada estimativa com o preço real da venda.
import { api, ApiError } from "./api.js";
import * as charts from "./charts.js";
import { $, algoName, errorStatus, esc, fmtField, nf, pct, saleText, signedPct, signedUsd, toast, usd } from "./util.js";

const S = { data: null, fields: [], selected: null, rows: [], hooks: {}, running: false };
const DETAIL_ORDER = ["neighborhood", "gr_liv_area_m2", "total_bsmt_m2", "lot_area_m2", "bedroom_abv_gr", "full_bath", "overall_qual", "overall_cond", "kitchen_qual", "year_built", "year_remod_add", "garage_cars", "fireplaces", "central_air"];

export async function initRealHouses(hooks) {
  S.hooks = hooks;
  $("#btn-run").addEventListener("click", () => run(true));
  ["#f-search", "#f-result", "#f-sort"].forEach((sel) => $(sel).addEventListener("input", () => { renderTable(); drawChart(); }));
  const table = $("#real-table");
  table.addEventListener("click", (e) => { const tr = e.target.closest("tr[data-id]"); if (tr) select(Number(tr.dataset.id), { fromTable: true }); });
  table.addEventListener("keydown", onTableKey);
  try { S.fields = (await api.formOptions()).fields; } catch { S.fields = []; }
  await run(false);
}

async function run(announce) {
  if (S.running) return;
  S.running = true;
  const btn = $("#btn-run");
  btn.disabled = true; btn.classList.add("is-loading");
  $(".btn__label", btn).textContent = "Rodando o modelo…";
  const t0 = performance.now();
  try {
    S.data = await api.realHouses();
    const total = Math.round(performance.now() - t0);
    $("#real-state").hidden = true;
    $("#real-body").hidden = false;
    renderStats();
    const sm = S.data.summary;
    $("#run-info").textContent = `${algoName(S.data.model_config)} calculou agora o preço de ${nf(sm.n)} casas vendidas em ${nf(sm.elapsed_ms, 1)} ms (${nf(total)} ms com a rede), às ${new Date().toLocaleTimeString("pt-BR")}. Erro médio: ${usd(sm.mae)}. São as casas de teste (20% da base), separadas antes do treino.`;
    renderTable();
    const keep = S.selected && S.data.rows.some((r) => r.id === S.selected);
    select(keep ? S.selected : typicalRow().id, {});
    if (announce) toast(`Modelo executado de novo em ${nf(sm.n)} casas.`);
  } catch (e) {
    $("#real-state").innerHTML = `<div class="state-error"><span>${esc(e instanceof ApiError && e.status === 503 ? "Modelo indisponível: " + e.message : e.message)}</span><button class="btn btn--sm" type="button" id="real-retry">Tentar de novo</button></div>`;
    $("#real-retry")?.addEventListener("click", () => run(false));
  } finally {
    S.running = false;
    btn.disabled = false; btn.classList.remove("is-loading");
    $(".btn__label", btn).textContent = "Rodar o modelo de novo";
  }
}

/** Casa com erro percentual mais próximo da mediana: um exemplo típico para começar. */
function typicalRow() {
  const med = S.data.summary.median_ape_pct;
  return [...S.data.rows].sort((a, b) => Math.abs(Math.abs(a.error_pct) - med) - Math.abs(Math.abs(b.error_pct) - med))[0];
}

function renderStats() {
  const sm = S.data.summary;
  const stat = (value, label, note) => `<div class="stat"><span class="stat__value">${value}</span><span class="stat__label">${label}</span><span class="stat__note">${note}</span></div>`;
  const el = $("#real-stats");
  el.innerHTML =
    stat(pct(sm.within_10pct), "perto do preço real", `erro de até 10% em ${nf(Math.round(sm.within_10pct * sm.n))} de ${nf(sm.n)} casas`) +
    stat(`${nf(sm.median_ape_pct, 1)}%`, "erro típico", "metade das casas erra menos que isso") +
    (sm.inside_range !== null ? stat(pct(sm.inside_range), "dentro da faixa típica", "o preço real caiu na faixa estimada") : "") +
    stat(nf(sm.n), "casas vendidas testadas", "nenhuma usada no treino");
  el.classList.remove("is-fresh"); void el.offsetWidth; el.classList.add("is-fresh");
}

function computeRows() {
  const q = $("#f-search").value.trim().toLowerCase();
  const res = $("#f-result").value;
  const by = {
    "err-desc": (a, b) => Math.abs(b.error_pct) - Math.abs(a.error_pct),
    "err-asc": (a, b) => Math.abs(a.error_pct) - Math.abs(b.error_pct),
    "price-desc": (a, b) => b.actual - a.actual,
    "price-asc": (a, b) => a.actual - b.actual,
  }[$("#f-sort").value];
  S.rows = S.data.rows
    .filter((r) => (!q || r.neighborhood_label.toLowerCase().includes(q) || r.neighborhood.toLowerCase().includes(q) || String(r.id) === q)
      && (res === "all" || errorStatus(r.error_pct).cls === res))
    .sort(by);
}

function renderTable() {
  computeRows();
  const rows = S.rows;
  $("#real-count").textContent = `${nf(rows.length)} de ${nf(S.data.rows.length)}`;
  const focusId = rows.some((r) => r.id === S.selected) ? S.selected : rows[0]?.id;
  $("#real-table").innerHTML = `<thead><tr><th>Id</th><th class="txt">Bairro</th><th title="Área construída acima do solo">Área</th>
    <th><span class="key key--real"></span>Real (venda)</th><th><span class="key key--model"></span>Modelo</th><th>Diferença</th><th class="txt">Resultado</th></tr></thead><tbody>${
    rows.map((r) => {
      const st = errorStatus(r.error_pct);
      return `<tr data-id="${r.id}" tabindex="${r.id === focusId ? 0 : -1}" class="${r.id === S.selected ? "is-active" : ""}">
        <td class="num">${r.id}</td><td class="txt">${esc(r.neighborhood_label)}</td><td class="num">${nf(r.area_m2)} m²</td>
        <td class="num col-real">${usd(r.actual)}</td><td class="num col-model">${usd(r.predicted)}</td>
        <td class="num">${signedPct(r.error_pct)}</td>
        <td class="txt"><span class="chip chip--${st.cls}">${st.label}</span></td></tr>`;
    }).join("") || `<tr><td colspan="7" class="txt fine">Nenhuma casa com esses filtros.</td></tr>`}</tbody>`;
}

function drawChart() {
  charts.realVsPredicted("chart-real", S.rows.length ? S.rows : S.data.rows, S.selected, (id) => select(Number(id), { fromChart: true }));
}

/** Teclado na tabela: um único ponto de Tab; setas movem, Enter/Espaço seleciona. */
function onTableKey(e) {
  const tr = e.target.closest("tr[data-id]");
  if (!tr) return;
  const move = (next) => { if (!next) return; tr.tabIndex = -1; next.tabIndex = 0; next.focus(); };
  if (e.key === "ArrowDown") { e.preventDefault(); move(tr.nextElementSibling); }
  else if (e.key === "ArrowUp") { e.preventDefault(); move(tr.previousElementSibling); }
  else if (e.key === "Home") { e.preventDefault(); move(tr.parentElement.firstElementChild); }
  else if (e.key === "End") { e.preventDefault(); move(tr.parentElement.lastElementChild); }
  else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(Number(tr.dataset.id), { fromTable: true }); }
}

const fieldOf = (api) => S.fields.find((f) => f.api_name === api);

function select(id, { fromTable = false, fromChart = false } = {}) {
  const r = S.data.rows.find((x) => x.id === id);
  if (!r) return;
  S.selected = id;
  document.querySelectorAll("#real-table tr.is-active").forEach((tr) => tr.classList.remove("is-active"));
  const tr = document.querySelector(`#real-table tr[data-id="${id}"]`);
  if (tr) {
    tr.classList.add("is-active");
    document.querySelectorAll('#real-table tr[tabindex="0"]').forEach((x) => (x.tabIndex = -1));
    tr.tabIndex = 0;
    if (fromChart) tr.scrollIntoView({ block: "nearest" });
  }
  drawChart();
  renderDetail(r);
  const st = errorStatus(r.error_pct);
  $("#real-live").textContent = `Casa ${r.id}, ${r.neighborhood_label}: real ${usd(r.actual)}, modelo ${usd(r.predicted)}, ${st.label.toLowerCase()}.`;
  // Em telas estreitas o detalhe fica abaixo da tabela: leva a pessoa até ele.
  if ((fromTable || fromChart) && window.matchMedia("(max-width: 1100px)").matches) $("#detail").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderDetail(r) {
  const st = errorStatus(r.error_pct);
  const diff = r.predicted - r.actual;
  const max = Math.max(r.actual, r.predicted) * 1.05;
  const w = (v) => `${((100 * v) / max).toFixed(1)}%`;
  $("#detail").innerHTML = `
    <div class="detail__head"><h2>Casa Id ${r.id} · ${esc(r.neighborhood_label)}</h2><span class="chip chip--${st.cls}">${st.long}</span></div>
    <div class="pair">
      <div class="pair__tile pair__tile--real">
        <span class="pair__label"><span class="key key--real"></span>Real: preço da venda</span>
        <span class="pair__value">${usd(r.actual)}</span>
        <span class="pair__note">vendida em ${r.year_sold} · ${esc(saleText(r.sale_condition))}</span>
      </div>
      <div class="pair__vs pair__vs--${st.cls}">
        <span class="pair__diff">${signedPct(r.error_pct)}</span>
        <span class="pair__diffusd">${signedUsd(diff)}</span>
        <span class="pair__arrow">${diff >= 0 ? "modelo acima" : "modelo abaixo"}</span>
      </div>
      <div class="pair__tile pair__tile--model">
        <span class="pair__label"><span class="key key--model"></span>Modelo: estimativa</span>
        <span class="pair__value">${usd(r.predicted)}</span>
        <span class="pair__note">${r.range_low ? `faixa típica ${usd(r.range_low)} a ${usd(r.range_high)}` : "&nbsp;"}</span>
      </div>
    </div>
    <div class="bars" aria-hidden="true">
      <div class="bars__row"><span>Real</span><span class="bars__track"><span class="bars__fill bars__fill--real" style="width:${w(r.actual)}"></span></span></div>
      <div class="bars__row"><span>Modelo</span><span class="bars__track"><span class="bars__fill bars__fill--model" style="width:${w(r.predicted)}"></span></span></div>
    </div>
    <p class="detail__err">O modelo estimou <span class="num-txt">${usd(Math.abs(diff))}</span> ${diff >= 0 ? "a mais" : "a menos"} do que o preço pelo qual a casa foi vendida.${r.range_low ? ` O preço real ficou ${r.inside_range ? "dentro" : "fora"} da faixa típica de erro.` : ""}</p>
    ${r.sale_condition !== "Normal" ? `<p class="fine">Atenção: o tipo de venda não é uma informação que o modelo recebe. Vendas fora do normal costumam ter preços que as características da casa não explicam.</p>` : ""}
    <h3 class="detail__sub">Dados da casa (os mesmos que o modelo recebeu)</h3>
    <dl class="facts">${DETAIL_ORDER.filter((k) => k in r.inputs).map((k) => `<div><dt>${esc(fieldOf(k)?.label ?? k)}</dt><dd>${esc(fmtField(fieldOf(k), r.inputs[k]))}</dd></div>`).join("")}</dl>
    <div class="detail__actions"><button class="btn btn--primary" type="button" id="btn-open-sim">Abrir esta casa no simulador</button></div>`;
  $("#btn-open-sim").addEventListener("click", () => S.hooks.openInSimulator?.(r.id));
}
