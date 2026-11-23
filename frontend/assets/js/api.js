// Cliente da API HouseValue AI. Mesmo host por padrão; ?api=http://host:porta para outro servidor.
const params = new URLSearchParams(location.search);
const API_BASE = (params.get("api") || (location.protocol === "file:" ? "http://localhost:8000" : "")).replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message, status, errors = null) {
    super(message);
    this.status = status;
    this.errors = errors;
  }
}

async function request(path, { signal, ...options } = {}) {
  let res;
  try {
    res = await fetch(API_BASE + path, { headers: { "Content-Type": "application/json" }, signal, ...options });
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new ApiError("Sem conexão com a API. Verifique se o servidor está rodando (uvicorn backend.main:app).", 0);
  }
  let data = null;
  try { data = await res.json(); } catch { /* corpo vazio */ }
  if (!res.ok) throw new ApiError(data?.detail || `Erro ${res.status}`, res.status, data?.errors || null);
  return data;
}

const post = (path, body, signal) => request(path, { method: "POST", body: JSON.stringify(body), signal });

export const api = {
  health: () => request("/api/health"),
  modelInfo: () => request("/api/model/info"),
  formOptions: () => request("/api/form-options"),
  predict: (body, signal) => post("/api/predict", body, signal),
  sensitivity: (inputs, field, signal) => post("/api/predict/sensitivity", { inputs, field }, signal),
  samples: () => request("/api/samples"),
  compare: (inputs, signal) => post("/api/predict/compare", inputs, signal),
  decompose: (base, target, signal) => post("/api/predict/decompose", { base, target }, signal),
  modelAnalysis: () => request("/api/analytics/model-analysis"),
  realHouses: () => request("/api/real-houses"),
  errorBand: () => request("/api/error-band"),
  overview: () => request("/api/analytics/overview"),
  comparison: () => request("/api/analytics/model-comparison"),
  importance: () => request("/api/analytics/feature-importance"),
  evaluation: () => request("/api/analytics/evaluation"),
  eda: () => request("/api/analytics/eda"),
};
