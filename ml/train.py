"""Fase 3 — Treinamento, validação cruzada e seleção (PRD §9).

Uso: python -m ml.train

Protocolo (definido antes da execução):
- partição única 80% desenvolvimento / 20% holdout (seed 42); o holdout NÃO é
  tocado aqui;
- KFold(5, shuffle=True, random_state=42) idêntico para todos os modelos e
  conjuntos de atributos;
- hiperparâmetros escolhidos por GridSearchCV nesses mesmos folds, dentro do
  desenvolvimento;
- métrica primária de seleção: RMSLE (desempate: MAE);
- cada algoritmo é testado com o alvo original e com log1p(SalePrice) via
  TransformedTargetRegressor; as métricas são sempre calculadas em USD.
- o pipeline selecionado é reajustado no desenvolvimento (80%) e salvo; o
  artefato salvo é exatamente o avaliado no holdout por ml.evaluate.
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from ml import config
from ml.data import load_train, parse_data_dictionary, sha256_file
from ml.features import APP_FEATURES, FIELD_META, QUALITY_PT, YES_NO_PT
from ml.units import AREA_FIELDS, SQFT_TO_M2
from ml.metrics import METRIC_NAMES, SCORING
from ml.preprocessing import (
    MONOTONIC_INCREASING,
    assert_no_leakage_columns,
    build_hgb_native_preprocessor,
    build_preprocessor,
    dev_holdout_split,
    full_feature_list,
    split_feature_types,
)

ALGORITHMS = {
    "DummyRegressor": {"scale": False, "grid": {}, "make": lambda: DummyRegressor(strategy="median"), "log_variants": [False]},
    "Ridge": {"scale": True, "grid": {"alpha": [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]}, "make": lambda: Ridge(), "log_variants": [False, True]},
    "RandomForestRegressor": {
        "scale": False,
        "grid": {"max_features": [0.33, 0.6, 1.0], "min_samples_leaf": [1, 3]},
        "make": lambda: RandomForestRegressor(n_estimators=300, random_state=config.RANDOM_STATE, n_jobs=-1),
        "log_variants": [False, True],
    },
    "HistGradientBoostingRegressor": {
        "scale": False,
        "grid": {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [7, 15, 31]},
        "make": lambda: HistGradientBoostingRegressor(max_iter=500, early_stopping=False, random_state=config.RANDOM_STATE),
        "log_variants": [False, True],
    },
    # ---- v2: mesma grade do HGB, mas com categorias nativas (sem one-hot) e,
    # na segunda variante, restrições monotônicas definidas por domínio.
    # Só com log1p(y), porque o log venceu em todos os algoritmos da v1.
    "HistGradientBoostingRegressor (categorias nativas)": {
        "scale": False, "native": True, "monotonic": False,
        "grid": {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [7, 15, 31]},
        "log_variants": [True],
    },
    "HistGradientBoostingRegressor (nativo, monotônico)": {
        "scale": False, "native": True, "monotonic": True,
        "grid": {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [7, 15, 31]},
        "log_variants": [True],
    },
}


def make_pipeline(algo: str, numeric: list[str], categorical: list[str], log_target: bool):
    spec = ALGORITHMS[algo]
    if spec.get("native"):
        mono = {c: 1 for c in MONOTONIC_INCREASING if c in numeric} if spec.get("monotonic") else None
        model = HistGradientBoostingRegressor(
            max_iter=500, early_stopping=False, random_state=config.RANDOM_STATE,
            categorical_features=categorical or None, monotonic_cst=mono,
        )
        pipe = Pipeline([("prep", build_hgb_native_preprocessor(numeric, categorical)), ("model", model)])
    else:
        pipe = Pipeline([("prep", build_preprocessor(numeric, categorical, scale=spec["scale"])), ("model", spec["make"]())])
    prefix = "model__"
    if log_target:
        pipe = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1, check_inverse=False)
        prefix = "regressor__model__"
    grid = {prefix + k: v for k, v in spec["grid"].items()}
    return pipe, grid


def candidate_slug(config_label: str) -> str:
    import re
    import unicodedata
    ascii_ = unicodedata.normalize("NFKD", config_label).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_.lower()).strip("_")


def config_name(algo: str, log_target: bool) -> str:
    return f"{algo} + log1p(y)" if log_target else algo


def run_feature_set(name: str, features: list[str], dev: pd.DataFrame, cv: KFold) -> dict:
    assert_no_leakage_columns(features)
    numeric, categorical = split_feature_types(features, dev)
    X, y = dev[features], dev[config.TARGET]
    results, fitted = [], {}
    for algo, spec in ALGORITHMS.items():
        for log_target in spec["log_variants"]:
            pipe, grid = make_pipeline(algo, numeric, categorical, log_target)
            t0 = time.perf_counter()
            gs = GridSearchCV(pipe, grid or {}, cv=cv, scoring=SCORING, refit="rmsle", return_train_score=True, n_jobs=1, error_score="raise")
            gs.fit(X, y)
            elapsed = time.perf_counter() - t0
            i = gs.best_index_
            cvr = gs.cv_results_
            row = {
                "config": config_name(algo, log_target),
                "algorithm": algo,
                "log_target": log_target,
                "best_params": {k.split("model__")[-1]: (v.item() if hasattr(v, "item") else v) for k, v in gs.best_params_.items()},
                "n_param_combinations": len(cvr["params"]),
                "fit_seconds": round(elapsed, 2),
            }
            for m in METRIC_NAMES:
                sign = 1 if m == "r2" else -1
                row[f"cv_{m}_mean"] = float(sign * cvr[f"mean_test_{m}"][i])
                row[f"cv_{m}_std"] = float(cvr[f"std_test_{m}"][i])
                row[f"train_{m}_mean"] = float(sign * cvr[f"mean_train_{m}"][i])
                row[f"cv_{m}_folds"] = [float(sign * cvr[f"split{k}_test_{m}"][i]) for k in range(cv.get_n_splits())]
            results.append(row)
            fitted[row["config"]] = gs.best_estimator_
            print(f"[{name}] {row['config']:<42} CV RMSLE={row['cv_rmsle_mean']:.4f}±{row['cv_rmsle_std']:.4f}  MAE={row['cv_mae_mean']:,.0f}  ({elapsed:.1f}s)")

    candidates = [r for r in results if r["algorithm"] != "DummyRegressor"]
    best = min(candidates, key=lambda r: (round(r["cv_rmsle_mean"], 6), r["cv_mae_mean"]))
    baseline = next(r for r in results if r["algorithm"] == "DummyRegressor")
    return {
        "features": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "results": results,
        "selected": best["config"],
        "baseline": baseline["config"],
        "selected_beats_baseline_cv": {m: (best[f"cv_{m}_mean"] > baseline[f"cv_{m}_mean"]) if m == "r2" else (best[f"cv_{m}_mean"] < baseline[f"cv_{m}_mean"]) for m in METRIC_NAMES},
        "_fitted": fitted,
    }


def build_form_schema(dev: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict:
    """Limites e opções do formulário derivados da partição de desenvolvimento."""
    dictionary = parse_data_dictionary()
    fields = []
    for c in APP_FEATURES:
        meta = dict(FIELD_META[c])
        s = dev[c]
        field = {"name": c, **meta, "required": True, "description_dictionary": dictionary.get(c, {}).get("description")}
        if c in AREA_FIELDS:
            # números guardados na unidade do modelo (sq ft); a API converte para m²
            field.update({"model_unit": "sq ft", "display_factor": SQFT_TO_M2})
        if c in numeric:
            field.update(
                {
                    "type": "integer",
                    "min": int(s.min()),
                    "max": int(s.max()),
                    "median": float(s.median()),
                    "p05": float(s.quantile(0.05)),
                    "p95": float(s.quantile(0.95)),
                    "range_source": "mínimo e máximo observados na partição de desenvolvimento (o modelo não foi treinado fora dessa faixa)",
                }
            )
        else:
            codes = dictionary.get(c, {}).get("categories", {})
            lower = {k.lower(): v for k, v in codes.items()}
            counts = s.value_counts()
            options = []
            for v in sorted(counts.index, key=str):
                label = QUALITY_PT.get(v) if c == "KitchenQual" else YES_NO_PT.get(v) if c == "CentralAir" else None
                label = label or lower.get(str(v).lower(), str(v))
                options.append({"value": str(v), "label": label, "n_train": int(counts[v])})
            field.update({"type": "category", "options": options, "range_source": "categorias observadas na partição de desenvolvimento"})
        fields.append(field)
    rules = [{"rule": "YearRemodAdd >= YearBuilt", "message": "O ano da reforma não pode ser anterior ao ano de construção.", "violations_in_train": int((dev["YearRemodAdd"] < dev["YearBuilt"]).sum())}]
    return {"fields": fields, "cross_field_rules": rules}


def main() -> None:
    for d in (config.MODELS_DIR, config.METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    train = load_train()
    dev, hold = dev_holdout_split(train)
    assert set(dev[config.ID_COLUMN]).isdisjoint(hold[config.ID_COLUMN])
    config.SPLIT_PATH.write_text(
        json.dumps(
            {
                "method": "train_test_split(test_size=0.2, shuffle=True, random_state=42)",
                "n_total": len(train), "n_dev": len(dev), "n_holdout": len(hold),
                "dev_ids": dev[config.ID_COLUMN].tolist(), "holdout_ids": hold[config.ID_COLUMN].tolist(),
                "dev_target_median": float(dev[config.TARGET].median()), "holdout_target_median": float(hold[config.TARGET].median()),
            }
        ),
        encoding="utf-8",
    )
    print(f"Desenvolvimento: {len(dev)} | Holdout (reservado, não usado aqui): {len(hold)}")

    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    sets = {"full": full_feature_list(train), "app": APP_FEATURES}
    out = {}
    for name, feats in sets.items():
        out[name] = run_feature_set(name, feats, dev, cv)

    # Persistência dos pipelines
    app_best = out["app"]["_fitted"][out["app"]["selected"]]
    full_best = out["full"]["_fitted"][out["full"]["selected"]]
    baseline = out["app"]["_fitted"][out["app"]["baseline"]]
    joblib.dump(app_best, config.APP_MODEL_PATH)
    joblib.dump(full_best, config.FULL_MODEL_PATH)
    joblib.dump(baseline, config.MODELS_DIR / "baseline_dummy.joblib")

    # Candidatos do conjunto da aplicação (melhor hiperparâmetro de cada configuração,
    # reajustados no desenvolvimento): usados pela interface para comparar algoritmos
    # lado a lado. Não alteram a seleção, que é feita só pela CV acima.
    cand_dir = config.MODELS_DIR / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    for old in cand_dir.glob("*.joblib"):
        old.unlink()
    candidates = []
    for r in out["app"]["results"]:
        slug = candidate_slug(r["config"])
        path = cand_dir / f"{slug}.joblib"
        joblib.dump(out["app"]["_fitted"][r["config"]], path)
        candidates.append({"config": r["config"], "algorithm": r["algorithm"], "slug": slug, "file": f"candidates/{slug}.joblib",
                           "sha256": sha256_file(path), "cv_rmsle_mean": r["cv_rmsle_mean"], "is_selected": r["config"] == out["app"]["selected"],
                           "is_baseline": r["algorithm"] == "DummyRegressor"})

    # Reprodutibilidade: recarregar e comparar previsões no próprio desenvolvimento
    reloaded = joblib.load(config.APP_MODEL_PATH)
    assert np.allclose(reloaded.predict(dev[APP_FEATURES]), app_best.predict(dev[APP_FEATURES]))

    protocol = {
        "split": "80% desenvolvimento / 20% holdout, train_test_split(shuffle=True, random_state=42)",
        "cv": f"KFold(n_splits={config.CV_FOLDS}, shuffle=True, random_state={config.RANDOM_STATE}) — mesmos folds para todos",
        "tuning": "GridSearchCV nos mesmos folds, apenas no desenvolvimento (refit pela RMSLE)",
        "primary_metric": "RMSLE (desempate: MAE em USD)",
        "metrics_scale": "USD (escala original), inclusive para modelos treinados em log1p(y)",
        "note_on_cv_optimism": "As métricas de CV do melhor hiperparâmetro são levemente otimistas (seleção nos mesmos folds); o holdout é a estimativa independente.",
        "random_split_justification": "Estimativa de generalização para imóveis da mesma população (Ames, 2006–2010). Não é uma avaliação de previsão futura.",
    }
    comparison = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": protocol,
        "feature_sets": {k: {kk: vv for kk, vv in v.items() if kk != "_fitted"} for k, v in out.items()},
        "app_artifact": "artifacts/models/house_price_pipeline.joblib (conjunto 'app', usado pela API)",
        "full_artifact": "artifacts/models/house_price_pipeline_full.joblib (conjunto 'full', referência e submissão Kaggle)",
    }
    config.MODEL_COMPARISON_PATH.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for fs, v in out.items():
        for r in v["results"]:
            rows.append({"feature_set": fs, "config": r["config"], "selected": r["config"] == v["selected"],
                         **{f"cv_{m}_mean": r[f"cv_{m}_mean"] for m in METRIC_NAMES}, **{f"cv_{m}_std": r[f"cv_{m}_std"] for m in METRIC_NAMES},
                         **{f"train_{m}_mean": r[f"train_{m}_mean"] for m in METRIC_NAMES}, "best_params": json.dumps(r["best_params"])})
    df = pd.DataFrame(rows)
    df.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)

    numeric, categorical = out["app"]["numeric_features"], out["app"]["categorical_features"]
    form = build_form_schema(dev, numeric, categorical)
    config.FORM_SCHEMA_PATH.write_text(json.dumps(form, indent=2, ensure_ascii=False), encoding="utf-8")

    sel = next(r for r in out["app"]["results"] if r["config"] == out["app"]["selected"])
    manifest = json.loads(config.SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    metadata = {
        "model_name": sel["algorithm"],
        "model_config": sel["config"],
        "log_target": sel["log_target"],
        "hyperparameters": sel["best_params"],
        "model_version": config.MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact": "artifacts/models/house_price_pipeline.joblib",
        "artifact_sha256": sha256_file(config.APP_MODEL_PATH),
        "source_file": "train.csv",
        "source_sha256": next(x["sha256"] for x in manifest["working_copy"] if x["name"] == "train.csv"),
        "source_zip": manifest.get("zip"),
        "target": config.TARGET,
        "currency": "USD",
        "dataset_context": "Ames, Iowa (EUA) — vendas históricas de 2006 a 2010",
        "input_features": APP_FEATURES,
        "input_api_names": {c: FIELD_META[c]["api_name"] for c in APP_FEATURES},
        "numeric_features": numeric,
        "categorical_features": categorical,
        "preprocessing": [
            "StructuralNAFiller: NA estrutural → 'None' (regra fixa do dicionário; no conjunto app não há colunas com NA estrutural)",
        ] + ([
            "Numéricas: sem transformação (o HistGradientBoosting trata ausentes nativamente)",
            "Categóricas: OrdinalEncoder (desconhecida/ausente → NaN) + categorical_features nativo do HGB",
        ] + ([f"Restrições monotônicas crescentes: {', '.join(c for c in MONOTONIC_INCREASING if c in numeric)}"] if ALGORITHMS[sel["algorithm"]].get("monotonic") else [])
        if ALGORITHMS[sel["algorithm"]].get("native") else [
            "Numéricas: SimpleImputer(median, add_indicator=True)" + (" + StandardScaler" if ALGORITHMS[sel["algorithm"]]["scale"] else ""),
            "Categóricas: SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown='ignore')",
        ]) + (["Alvo: TransformedTargetRegressor(log1p / expm1) — previsão devolvida em USD"] if sel["log_target"] else []),
        "training_data": f"partição de desenvolvimento ({len(dev)} linhas)",
        "validation": protocol,
        "random_state": config.RANDOM_STATE,
        "cv_metrics": {m: {"mean": sel[f"cv_{m}_mean"], "std": sel[f"cv_{m}_std"]} for m in METRIC_NAMES},
        "train_metrics_in_cv": {m: sel[f"train_{m}_mean"] for m in METRIC_NAMES},
        "holdout_metrics": None,
        "full_feature_reference": {"config": out["full"]["selected"], "cv_metrics": {m: next(r for r in out["full"]["results"] if r["config"] == out["full"]["selected"])[f"cv_{m}_mean"] for m in METRIC_NAMES}},
        "candidates": candidates,
        "library_versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
    }
    config.MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Selecionado (app):  {out['app']['selected']}")
    print(f"Selecionado (full): {out['full']['selected']}")
    print("Salvo: artifacts/models/*.joblib, model_metadata.json, form_schema.json, artifacts/metrics/model_comparison.{json,csv}")


if __name__ == "__main__":
    main()
