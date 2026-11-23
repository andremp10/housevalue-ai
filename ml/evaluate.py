"""Fase 3 — Avaliação final no holdout e diagnóstico (PRD §8.5, §9.7, §9.8).

Uso: python -m ml.evaluate   (depois de python -m ml.train)

- Carrega os pipelines SALVOS (os mesmos que a API usa) e os avalia uma única
  vez no holdout de 20% reservado em ml.train. Nenhuma decisão de modelagem é
  tomada a partir destes números.
- Compara com o baseline (DummyRegressor mediana) no mesmo holdout.
- Gera diagnósticos, importância por permutação e submission.csv (opcional,
  sem rótulos → sem métricas).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from ml import config, figures
from ml.data import load_test, load_train, sha256_file
from ml.explainability import native_importance, permutation_on_holdout
from ml.features import APP_FEATURES
from ml.metrics import METRIC_NAMES, all_metrics
from ml.preprocessing import full_feature_list

def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    train = load_train()
    split = json.loads(config.SPLIT_PATH.read_text(encoding="utf-8"))
    hold = train[train[config.ID_COLUMN].isin(split["holdout_ids"])].sort_values(config.ID_COLUMN)
    assert len(hold) == split["n_holdout"]
    y = hold[config.TARGET].to_numpy()
    full_feats = full_feature_list(train)

    app = joblib.load(config.APP_MODEL_PATH)
    full = joblib.load(config.FULL_MODEL_PATH)
    base = joblib.load(config.MODELS_DIR / "baseline_dummy.joblib")
    comparison = json.loads(config.MODEL_COMPARISON_PATH.read_text(encoding="utf-8"))
    fs = comparison["feature_sets"]

    preds = {
        "app": app.predict(hold[APP_FEATURES]),
        "full": full.predict(hold[full_feats]),
        "baseline": base.predict(hold[APP_FEATURES]),
    }
    metrics = {k: all_metrics(y, p) for k, p in preds.items()}
    names = {"app": fs["app"]["selected"], "full": fs["full"]["selected"], "baseline": fs["app"]["baseline"]}

    beats = {
        k: {m: (metrics[k][m] > metrics["baseline"][m]) if m == "r2" else (metrics[k][m] < metrics["baseline"][m]) for m in METRIC_NAMES}
        for k in ("app", "full")
    }
    cv_sel = {k: next(r for r in fs[k]["results"] if r["config"] == fs[k]["selected"]) for k in ("app", "full")}

    # ---------- Previsões e maiores erros ----------
    df = pd.DataFrame({config.ID_COLUMN: hold[config.ID_COLUMN].to_numpy(), "SalePrice": y,
                       "pred_app": preds["app"], "pred_full": preds["full"], "pred_baseline": preds["baseline"]})
    df["abs_error_app"] = (df["pred_app"] - df["SalePrice"]).abs()
    df["pct_error_app"] = 100 * df["abs_error_app"] / df["SalePrice"]
    df["abs_error_full"] = (df["pred_full"] - df["SalePrice"]).abs()
    df.round(2).to_csv(config.METRICS_DIR / "holdout_predictions.csv", index=False)
    ctx_cols = [config.ID_COLUMN, "GrLivArea", "OverallQual", "Neighborhood", "SaleCondition", "YearBuilt"]
    top = df.nlargest(15, "abs_error_app").merge(hold[ctx_cols], on=config.ID_COLUMN)
    top_errors = top[[config.ID_COLUMN, "SalePrice", "pred_app", "abs_error_app", "pct_error_app", "GrLivArea", "OverallQual", "Neighborhood", "SaleCondition"]].round(2).to_dict(orient="records")

    # ---------- Figuras (desenhadas a partir dos artefatos; ver ml/figures.py) ----------
    figures.holdout_figures()
    figures.comparison_figures()
    res = y - preds["app"]  # resíduos (real − previsto) para o relatório

    # ---------- Explicabilidade ----------
    perm_app = permutation_on_holdout(app, hold[APP_FEATURES], hold[config.TARGET])
    perm_full = permutation_on_holdout(full, hold[full_feats], hold[config.TARGET], n_repeats=10)
    importance = {
        "app": {"model": names["app"], "permutation_holdout": {"measure": "aumento médio da RMSLE no holdout ao embaralhar a coluna (20 repetições)", "rows": perm_app}, "native": native_importance(app)},
        "full": {"model": names["full"], "permutation_holdout": {"measure": "aumento médio da RMSLE no holdout ao embaralhar a coluna (10 repetições)", "rows": perm_full[:25]}, "native": native_importance(full)},
        "caveat": "Importâncias medem dependência do modelo em relação à variável, não causalidade. Variáveis correlacionadas dividem/mascaram importância.",
    }
    config.FEATURE_IMPORTANCE_PATH.write_text(json.dumps(importance, indent=2, ensure_ascii=False), encoding="utf-8")
    figures.importance_figure()

    # ---------- Submissão Kaggle (opcional, sem métricas) ----------
    test = load_test()
    sub = pd.DataFrame({config.ID_COLUMN: test[config.ID_COLUMN], config.TARGET: full.predict(test[full_feats])})
    sample = pd.read_csv(config.RAW_DIR / config.SUBMISSION_SAMPLE_FILE)
    assert list(sub.columns) == list(sample.columns) and len(sub) == len(sample) and (sub[config.ID_COLUMN].values == sample[config.ID_COLUMN].values).all()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sub.to_csv(config.REPORTS_DIR / "submission.csv", index=False)

    # Métricas de holdout de cada candidato da aplicação: só informativas. A escolha do
    # modelo já foi feita pela CV em ml.train; estes números não mudam a seleção.
    meta_now = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    cand_metrics = []
    for c in meta_now.get("candidates", []):
        cm = joblib.load(config.MODELS_DIR / c["file"])
        pc = cm.predict(hold[APP_FEATURES])
        cand_metrics.append({"config": c["config"], "slug": c["slug"], "is_selected": c["is_selected"], "is_baseline": c["is_baseline"],
                             "cv_rmsle_mean": c["cv_rmsle_mean"], "holdout": all_metrics(y, pc)})

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidates": {"note": "Informativo: calculado depois da seleção feita pela validação cruzada; não foi usado para escolher o modelo.", "rows": cand_metrics},
        "holdout": {"n": int(len(hold)), "ids_source": "artifacts/metrics/split.json", "target_median": float(np.median(y))},
        "evaluated_artifacts": {
            "app": {"config": names["app"], "path": "artifacts/models/house_price_pipeline.joblib", "sha256": sha256_file(config.APP_MODEL_PATH)},
            "full": {"config": names["full"], "path": "artifacts/models/house_price_pipeline_full.joblib", "sha256": sha256_file(config.FULL_MODEL_PATH)},
            "baseline": {"config": names["baseline"], "path": "artifacts/models/baseline_dummy.joblib"},
        },
        "metrics": metrics,
        "beats_baseline": beats,
        "cv_vs_holdout": {k: {m: {"cv_mean": cv_sel[k][f"cv_{m}_mean"], "cv_std": cv_sel[k][f"cv_{m}_std"], "holdout": metrics[k][m]} for m in METRIC_NAMES} for k in ("app", "full")},
        "residuals_app": {"mean": float(res.mean()), "median": float(np.median(res)), "median_abs": float(np.median(np.abs(res))),
                          "p90_abs": float(np.quantile(np.abs(res), 0.9)), "mape_pct": float(df["pct_error_app"].mean()),
                          "median_ape_pct": float(df["pct_error_app"].median())},
        "top_errors_app": top_errors,
        "points_app": {"id": df[config.ID_COLUMN].tolist(), "actual": df["SalePrice"].tolist(), "predicted": [round(float(v), 2) for v in df["pred_app"]]},
        "submission": {"path": "artifacts/reports/submission.csv", "model": names["full"], "rows": int(len(sub)),
                       "note": "Gerada para test.csv (sem SalePrice). Não validada localmente: não há rótulos."},
    }
    config.HOLDOUT_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    meta = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    meta["holdout_metrics"] = {m: metrics["app"][m] for m in METRIC_NAMES}
    meta["holdout_n"] = int(len(hold))
    meta["baseline_holdout_metrics"] = {m: metrics["baseline"][m] for m in METRIC_NAMES}
    meta["beats_baseline_holdout"] = beats["app"]
    meta["full_feature_reference"]["holdout_metrics"] = {m: metrics["full"][m] for m in METRIC_NAMES}
    meta["evaluated_at"] = report["generated_at"]
    config.MODEL_METADATA_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    for k in ("baseline", "app", "full"):
        m = metrics[k]
        print(f"HOLDOUT {k:<8} {names[k]:<42} RMSLE={m['rmsle']:.4f} MAE={m['mae']:,.0f} RMSE={m['rmse']:,.0f} R2={m['r2']:.4f}")
    print("Salvo: holdout_evaluation.json, holdout_predictions.csv, feature_importance.json, figuras eval_*.png, submission.csv")


if __name__ == "__main__":
    main()
