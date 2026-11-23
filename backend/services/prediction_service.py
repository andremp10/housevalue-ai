"""Carregamento do pipeline salvo e geração de previsões (PRD §12.1, §17).

A API nunca treina: apenas carrega o artefato gerado por `python -m ml.train`.
Antes de carregar o .joblib, o SHA-256 do arquivo é comparado ao registrado em
model_metadata.json (gerado pelo próprio pipeline), evitando carregar um artefato
de origem desconhecida ou adulterado.

Unidades: o modelo foi treinado em pés² (dados originais de Ames). A API recebe e
devolve áreas em m²; a conversão é feita aqui, na fronteira (ml/units.py).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from backend.core.config import Settings

log = logging.getLogger("housevalue.prediction")

APP_MODEL_FILE = "house_price_pipeline.joblib"
_TOL = 1e-6


class ModelUnavailableError(RuntimeError):
    pass


class DomainValidationError(ValueError):
    def __init__(self, errors: list[dict]):
        super().__init__("Entrada inválida")
        self.errors = errors


def fmt_int(v: float) -> str:
    """Formata número no padrão brasileiro de milhar (1.234)."""
    return f"{v:,.0f}".replace(",", ".")


def fmt_num(v: float, decimals: int = 1) -> str:
    s = f"{v:,.{decimals}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def format_value(v, unit: str | None) -> str:
    """Valor já na unidade de exibição."""
    if unit == "ano":
        return str(int(v))
    if unit == "nota 1–10":
        return f"{int(v)}/10"
    if unit == "carros":
        return f"{int(v)} carro(s)"
    if unit == "m²":
        return f"{fmt_int(v)} m²"
    return str(int(v)) if float(v).is_integer() else fmt_num(v)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class PredictionService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.metadata: dict | None = None
        self.form_schema: dict | None = None   # na unidade do modelo (sq ft)
        self.public_fields: list[dict] = []    # na unidade de exibição (m²)
        self.distributions: dict = {}
        self.candidates: list[dict] = []
        self.samples: list[dict] = []          # imóveis reais do holdout
        self.error_band: dict | None = None
        self.load_error: str | None = None
        # O FastAPI executa rotas síncronas em várias threads. Previsões simultâneas do
        # HistGradientBoosting disputam os threads do OpenMP (medido: ~8 s por chamada
        # concorrente contra ~70 ms isolada). Serializar e usar 1 thread resolve.
        self._predict_lock = threading.Lock()

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        models = self.settings.models_dir
        try:
            meta_path, form_path, model_path = models / "model_metadata.json", models / "form_schema.json", models / APP_MODEL_FILE
            for p in (meta_path, form_path, model_path):
                if not p.is_file():
                    raise ModelUnavailableError(f"Artefato ausente: {p.name}. Execute `python -m ml.train` e `python -m ml.evaluate`.")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if _sha256(model_path) != meta.get("artifact_sha256"):
                raise ModelUnavailableError("O SHA-256 do modelo não confere com model_metadata.json; artefato recusado.")
            self.metadata = meta
            self.form_schema = json.loads(form_path.read_text(encoding="utf-8"))
            self.model = joblib.load(model_path)
            expected = list(getattr(self.model, "feature_names_in_", meta["input_features"]))
            if expected != meta["input_features"]:
                raise ModelUnavailableError("As variáveis do pipeline não conferem com os metadados.")
            self.candidates = self._load_candidates(meta)
            dist_path = models / "field_distributions.json"
            self.distributions = json.loads(dist_path.read_text(encoding="utf-8")).get("fields", {}) if dist_path.is_file() else {}
            band_path = models / "error_band.json"
            self.error_band = json.loads(band_path.read_text(encoding="utf-8")) if band_path.is_file() else None
            samples_path = self.settings.reports_dir / "holdout_samples.json"
            self.samples = json.loads(samples_path.read_text(encoding="utf-8"))["samples"] if samples_path.is_file() else []
            self.public_fields = [self._public_field(f) for f in self.form_schema["fields"]]
            self.load_error = None
            log.info("Modelo carregado: %s", meta.get("model_config"))
        except ModelUnavailableError as exc:
            self._fail(str(exc))
        except Exception:  # noqa: BLE001 — falha de desserialização/versão
            log.exception("Falha ao carregar o modelo")
            self._fail("Modelo incompatível ou corrompido. Treine novamente com `python -m ml.train`.")

    def _fail(self, msg: str) -> None:
        self.model, self.metadata, self.form_schema, self.load_error = None, None, None, msg
        self.candidates, self.public_fields, self.samples, self.error_band = [], [], [], None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def require(self) -> None:
        if not self.ready:
            raise ModelUnavailableError(self.load_error or "Modelo não carregado.")

    def _model_predict(self, X: pd.DataFrame, model=None):
        with self._predict_lock, threadpool_limits(limits=1):
            return (model or self.model).predict(X)

    def _load_candidates(self, meta: dict) -> list[dict]:
        """Candidatos da comparação lado a lado, conferidos pelo SHA-256 do treino."""
        holdout = {}
        ev = self.settings.metrics_dir / "holdout_evaluation.json"
        if ev.is_file():
            rows = json.loads(ev.read_text(encoding="utf-8")).get("candidates", {}).get("rows", [])
            holdout = {r["config"]: r["holdout"] for r in rows}
        out = []
        for c in meta.get("candidates", []):
            path = self.settings.models_dir / c["file"]
            if not path.is_file() or _sha256(path) != c["sha256"]:
                log.warning("Candidato ignorado (ausente ou hash divergente): %s", c["config"])
                continue
            model = joblib.load(path)
            # Random Forest foi treinado com n_jobs=-1; para prever 1 linha, abrir threads
            # custa mais que a previsão. n_jobs=1 não muda os valores previstos.
            inner = getattr(model, "regressor_", model)
            final = getattr(inner, "named_steps", {}).get("model")
            if final is not None and hasattr(final, "n_jobs"):
                final.n_jobs = 1
            out.append({**{k: c[k] for k in ("config", "algorithm", "is_selected", "is_baseline", "cv_rmsle_mean")},
                        "model": model, "holdout": holdout.get(c["config"])})
        return out

    # ------------------------------------------------------------ unidades
    def _fields(self) -> list[dict]:
        return self.form_schema["fields"]

    @staticmethod
    def _factor(f: dict) -> float | None:
        return f.get("display_factor")

    def to_display(self, f: dict, v):
        k = self._factor(f)
        return float(v) * k if k else v

    def to_model(self, f: dict, v):
        k = self._factor(f)
        return float(v) / k if k else v

    def _public_field(self, f: dict) -> dict:
        """Campo do formulário na unidade de exibição (áreas em m²)."""
        k = self._factor(f)
        if not k:
            d = self.distributions.get(f["name"])
            return {**f, "distribution": ({kk: d[kk] for kk in ("histogram", "common_range", "above_common", "percentiles")} if d else None)}
        g = {kk: vv for kk, vv in f.items() if kk not in ("display_factor", "model_unit")}
        # limites exibidos arredondados para dentro: digitar o limite mostrado é sempre aceito
        g.update({
            "type": "number",
            "min": math.ceil(f["min"] * k * 10) / 10, "max": math.floor(f["max"] * k * 10) / 10,
            "median": f["median"] * k, "p05": f["p05"] * k, "p95": f["p95"] * k,
            "range_source": f["range_source"] + " (convertida de pés² para m²)",
        })
        d = self.distributions.get(f["name"])
        if d:
            g["distribution"] = {
                "histogram": {"counts": d["histogram"]["counts"], "edges": [e * k for e in d["histogram"]["edges"]]},
                "common_range": [round(d["common_range"][0] * k, 1), round(d["common_range"][1] * k, 1)],
                "above_common": d["above_common"],
                "percentiles": {q: v * k for q, v in d["percentiles"].items()},
            }
        else:
            g["distribution"] = None
        return g

    def public_by_name(self) -> dict:
        return {f["name"]: f for f in self.public_fields}

    def display_inputs(self, row_model_units: dict) -> dict:
        """Linha nas unidades do modelo → dicionário com nomes e unidades da API."""
        out = {}
        for f in self._fields():
            v = row_model_units[f["name"]]
            out[f["api_name"]] = self.to_display(f, v) if self._factor(f) else v
        return out

    def _show(self, f_model: dict, v_model) -> str:
        pub = self.public_by_name()[f_model["name"]]
        if f_model["type"] == "category":
            return next((o["label"] for o in f_model["options"] if o["value"] == v_model), str(v_model))
        return format_value(self.to_display(f_model, v_model), pub.get("unit"))

    # ------------------------------------------------------------ validation
    def validate_domain(self, payload: dict) -> dict:
        """Recebe nomes/unidades da API; devolve dict com colunas originais em unidades do modelo."""
        errors, row = [], {}
        pub = self.public_by_name()
        for f in self._fields():
            api, col = f["api_name"], f["name"]
            v = payload[api]
            if f["type"] == "integer":
                mv = self.to_model(f, v)
                if not (f["min"] - _TOL <= mv <= f["max"] + _TOL):
                    p = pub[col]
                    unit = f" {p['unit']}" if p.get("unit") == "m²" else ""
                    lo = p["min"] if p.get("unit") != "ano" else int(p["min"])
                    hi = p["max"] if p.get("unit") != "ano" else int(p["max"])
                    errors.append({"field": api, "message": f"{f['label']}: informe um valor entre {fmt_num(lo, 1) if unit else lo} e {fmt_num(hi, 1) if unit else hi}{unit}, faixa observada nos dados de treino."})
                row[col] = mv
            else:
                allowed = [o["value"] for o in f["options"]]
                if v not in allowed:
                    errors.append({"field": api, "message": f"{f['label']}: categoria '{v}' não reconhecida. Opções aceitas: {', '.join(allowed)}."})
                row[col] = v
        if not errors and row["YearRemodAdd"] < row["YearBuilt"]:
            errors.append({"field": "year_remod_add", "message": "O ano da reforma não pode ser anterior ao ano de construção."})
        if errors:
            raise DomainValidationError(errors)
        return row

    # ------------------------------------------------------------ faixa de erro
    def band(self, predicted: float) -> dict | None:
        b = self.error_band
        if not b:
            return None
        return {
            "low": round(predicted * b["factor_low"], 2), "high": round(predicted * b["factor_high"], 2),
            "level": b["level"], "holdout_coverage": b["holdout_coverage"],
            "explanation": b["explanation"],
        }

    # ------------------------------------------------------------ predict
    def predict(self, payload: dict) -> dict:
        self.require()
        row = self.validate_domain(payload)
        X = pd.DataFrame([row], columns=self.metadata["input_features"])
        value = float(self._model_predict(X)[0])
        if not np.isfinite(value):
            raise RuntimeError("Previsão inválida")
        inputs, unusual = [], []
        for f in self._fields():
            v = row[f["name"]]
            if f["type"] == "integer" and (v < f["p05"] or v > f["p95"]):
                unusual.append(f["label"])
            inputs.append({"field": f["api_name"], "column": f["name"], "label": f["label"],
                           "value": self.to_display(f, v) if self._factor(f) else v, "display": self._show(f, v)})
        b = self.band(value)
        notes = [
            "Estimativa experimental gerada por um modelo de Machine Learning treinado com vendas históricas de Ames, Iowa (EUA), de 2006 a 2010, em dólares da época.",
            "O resultado reflete a distribuição da base de treinamento e não é uma avaliação imobiliária profissional nem vale para o mercado brasileiro ou preços atuais.",
            f"Erro absoluto médio (MAE) do modelo no holdout: {fmt_int(self.metadata['holdout_metrics']['mae'])} USD." if self.metadata.get("holdout_metrics") else "Métricas de holdout ainda não calculadas.",
        ]
        if unusual:
            notes.append("Valores pouco comuns na base (fora da faixa de 5% a 95% dos imóveis de treino): " + ", ".join(unusual) + ". A estimativa tende a ser menos confiável nesses casos.")
        m = self.metadata
        return {
            "predicted_sale_price": round(value, 2),
            "range": b,
            "currency": m["currency"],
            "target": m["target"],
            "model_name": m["model_name"],
            "model_config_name": m["model_config"],
            "model_version": m["model_version"],
            "trained_at": m["trained_at"],
            "is_estimate": True,
            "dataset_context": m["dataset_context"],
            "inputs": inputs,
            "similar_real_houses": self.similar(row, value),
            "notes": notes,
        }

    # ------------------------------------------------------------ imóveis reais
    def _sample_row(self, s: dict, pred: float) -> dict:
        f = s["features"]
        pub = self.public_by_name()
        nb = next((o["label"] for o in pub["Neighborhood"]["options"] if o["value"] == f["Neighborhood"]), f["Neighborhood"])
        err = pred - s["actual"]
        b = self.band(pred)
        return {
            "id": s["id"], "neighborhood": f["Neighborhood"], "neighborhood_label": nb,
            "area_m2": f["GrLivArea"] * self._factor(next(x for x in self._fields() if x["name"] == "GrLivArea")),
            "overall_qual": f["OverallQual"], "bedrooms": f["BedroomAbvGr"], "year_built": f["YearBuilt"],
            "actual": s["actual"], "predicted": round(pred, 2), "error": round(err, 2), "error_pct": round(100 * err / s["actual"], 2),
            "range_low": b["low"] if b else None, "range_high": b["high"] if b else None,
            "inside_range": (b["low"] <= s["actual"] <= b["high"]) if b else None,
            "sale_condition": s["sale_condition"], "year_sold": s["year_sold"],
            "inputs": self.display_inputs(f),
        }

    def run_real_houses(self) -> dict:
        """Roda o modelo AGORA em todos os imóveis reais do holdout (nunca usados no treino)."""
        self.require()
        if not self.samples:
            raise ModelUnavailableError("Imóveis reais indisponíveis. Execute `python -m ml.app_assets`.")
        cols = self.metadata["input_features"]
        X = pd.DataFrame([s["features"] for s in self.samples], columns=cols)
        t0 = time.perf_counter()
        preds = self._model_predict(X)
        elapsed = (time.perf_counter() - t0) * 1000
        rows = [self._sample_row(s, float(p)) for s, p in zip(self.samples, preds)]
        actual = np.array([r["actual"] for r in rows])
        pred = np.array([r["predicted"] for r in rows])
        ape = np.abs(pred - actual) / actual
        return {
            "rows": rows,
            "summary": {
                "n": len(rows), "elapsed_ms": round(elapsed, 1),
                "mae": float(np.mean(np.abs(pred - actual))), "median_abs_error": float(np.median(np.abs(pred - actual))),
                "median_ape_pct": float(100 * np.median(ape)),
                "within_10pct": float(np.mean(ape <= 0.10)), "within_20pct": float(np.mean(ape <= 0.20)),
                "inside_range": float(np.mean([r["inside_range"] for r in rows])) if self.error_band else None,
                "range_level": self.error_band["level"] if self.error_band else None,
            },
            "source": "Casas de teste: 20% dos imóveis vendidos, separados antes do treino. O modelo nunca as viu.",
            "model_config": self.metadata["model_config"],
            "currency": self.metadata["currency"],
        }

    def similar(self, row: dict, predicted: float, k: int = 4) -> list[dict]:
        """Casas reais do holdout mais parecidas com o imóvel informado.

        Distância simples em escala padronizada (área, qualidade, ano, porão, garagem),
        com preferência pelo mesmo bairro. Serve para comparar a estimativa com
        preços reais de vendas parecidas; não entra no cálculo do modelo.
        """
        if not self.samples:
            return []
        keys = ["GrLivArea", "OverallQual", "YearBuilt", "TotalBsmtSF", "GarageCars"]
        M = np.array([[s["features"][c] for c in keys] for s in self.samples], dtype=float)
        sd = M.std(axis=0)
        sd[sd == 0] = 1
        q = np.array([row[c] for c in keys], dtype=float)
        dist = np.sqrt((((M - q) / sd) ** 2).sum(axis=1))
        same_nb = np.array([s["features"]["Neighborhood"] == row["Neighborhood"] for s in self.samples])
        dist = dist + np.where(same_nb, 0.0, 1.0)  # outro bairro pesa como 1 desvio-padrão
        out = []
        for i in np.argsort(dist)[:k]:
            s = self.samples[int(i)]
            r = self._sample_row(s, s["predicted"])
            r["same_neighborhood"] = bool(same_nb[i])
            out.append({kk: r[kk] for kk in ("id", "neighborhood_label", "area_m2", "overall_qual", "year_built", "actual", "predicted", "error_pct", "same_neighborhood", "inputs", "sale_condition", "year_sold")})
        return out

    # ------------------------------------------------------------ comparação de algoritmos
    def compare(self, payload: dict) -> dict:
        """Previsão de cada algoritmo candidato para o mesmo imóvel."""
        self.require()
        row = self.validate_domain(payload)
        X = pd.DataFrame([row], columns=self.metadata["input_features"])
        if not self.candidates:
            raise ModelUnavailableError("Candidatos não disponíveis. Execute python -m ml.train e python -m ml.evaluate.")
        rows = []
        for c in self.candidates:
            pred = float(self._model_predict(X, c["model"])[0])
            h = c["holdout"]
            rows.append({"config": c["config"], "algorithm": c["algorithm"], "predicted": round(pred, 2), "is_selected": c["is_selected"],
                         "is_baseline": c["is_baseline"], "cv_rmsle": c["cv_rmsle_mean"],
                         "holdout_mae": h["mae"] if h else None, "holdout_rmsle": h["rmsle"] if h else None})
        rows.sort(key=lambda r: r["predicted"])
        return {"rows": rows, "currency": self.metadata["currency"],
                "note": "Todos os candidatos usaram as mesmas 14 variáveis, os mesmos dados de treino e os mesmos folds. As métricas de holdout são informativas (calculadas após a seleção)."}

    def decompose(self, base_payload: dict, target_payload: dict) -> dict:
        """Efeito isolado de cada característica alterada de A para o cenário atual."""
        self.require()
        base, target = self.validate_domain(base_payload), self.validate_domain(target_payload)
        cols = self.metadata["input_features"]
        def same(a, b):  # áreas chegam como float (convertidas de m²): compara com tolerância
            return abs(a - b) < _TOL if isinstance(a, float) or isinstance(b, float) else a == b
        changed = [c for c in cols if not same(base[c], target[c])]
        rows, skipped = [dict(base), dict(target)], []
        for c in changed:
            r = dict(base, **{c: target[c]})
            if r["YearRemodAdd"] < r["YearBuilt"]:
                skipped.append(c)
                continue
            rows.append(r)
        preds = self._model_predict(pd.DataFrame(rows, columns=cols))
        p_base, p_target = float(preds[0]), float(preds[1])
        fields = {f["name"]: f for f in self._fields()}
        effects, k = [], 2
        for c in changed:
            f = fields[c]
            item = {"column": c, "field": f["api_name"], "label": f["label"], "from": self._show(f, base[c]), "to": self._show(f, target[c])}
            if c in skipped:
                item["effect"] = None
                item["note"] = "Não calculável isoladamente: violaria reforma >= construção."
            else:
                item["effect"] = round(float(preds[k]) - p_base, 2)
                k += 1
            effects.append(item)
        total = p_target - p_base
        isolated = sum(e["effect"] for e in effects if e["effect"] is not None)
        return {
            "base_price": round(p_base, 2), "target_price": round(p_target, 2), "total": round(total, 2),
            "effects": effects, "interaction": round(total - isolated, 2), "currency": self.metadata["currency"],
            "method": "Cada efeito é a previsão de A trocando só aquela característica, menos a previsão de A. A interação é o restante (total menos a soma dos efeitos). Mostra o comportamento do modelo, não causa.",
        }

    # ------------------------------------------------------------ sensibilidade
    def sensitivity(self, payload: dict, field_api: str) -> dict:
        """Curva de sensibilidade individual (ICE) com previsões reais do modelo.

        Varia só `field_api` dentro da faixa de treino (categorias aceitas ou grade
        de quantis da partição de desenvolvimento); não é efeito causal.
        """
        self.require()
        base = self.validate_domain(payload)
        f = next((x for x in self._fields() if x["api_name"] == field_api), None)
        if f is None:
            raise DomainValidationError([{"field": "field", "message": f"Variável '{field_api}' não faz parte do modelo."}])
        col = f["name"]
        pub = self.public_by_name()[col]
        if f["type"] == "integer":
            grid = self.distributions.get(col, {}).get("quantile_grid")
            if not grid:
                step = max(1, (f["max"] - f["min"]) // 30)
                grid = list(range(f["min"], f["max"] + 1, step))
            values = sorted({float(g) for g in grid} | {float(base[col])}) if self._factor(f) else sorted(set(grid) | {base[col]})
        else:
            values = [o["value"] for o in f["options"]]
        rows = []
        for v in values:
            r = dict(base, **{col: v})
            if r["YearRemodAdd"] < r["YearBuilt"]:
                continue
            rows.append(r)
        preds = self._model_predict(pd.DataFrame(rows, columns=self.metadata["input_features"]))
        points = []
        for r, p in zip(rows, preds):
            v = r[col]
            shown = round(self.to_display(f, v), 2) if self._factor(f) else v
            points.append({"value": shown, "label": self._show(f, v), "predicted": round(float(p), 2)})
        cur_shown = round(self.to_display(f, base[col]), 2) if self._factor(f) else base[col]
        current = next(pt for pt in points if pt["value"] == cur_shown)
        return {
            "field": field_api, "column": col, "label": f["label"], "type": f["type"], "unit": pub.get("unit"),
            "current": current, "points": points,
            "method": "Previsões do modelo variando só esta variável e mantendo as demais do imóvel informado (curva ICE). Mostra o comportamento do modelo, não um efeito causal.",
            "grid_source": "categorias aceitas pelo modelo" if f["type"] == "category" else ("quantis da partição de treino" if self.distributions.get(col) else "faixa de treino em passos regulares"),
            "currency": self.metadata["currency"],
        }
