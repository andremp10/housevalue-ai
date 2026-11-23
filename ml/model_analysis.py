"""Análises complementares do modelo (v2), para a apresentação e a página de modelos.

Uso: python -m ml.model_analysis   (depois de ml.train e ml.evaluate)

1. Curva de aprendizado do modelo selecionado: só na partição de desenvolvimento,
   com o mesmo KFold do treino.
2. "Vencedor ou empate?": os dois melhores candidatos do conjunto da aplicação
   comparados de forma pareada em RepeatedKFold(5 × 4) no desenvolvimento, com os
   hiperparâmetros já escolhidos. Para o conjunto completo usam-se as diferenças
   pareadas dos 5 folds já calculados no treino, sem novos ajustes.
3. Erro por segmento no holdout (faixa de preço real, condição de venda, bairro).
   Descreve onde o modelo já escolhido erra; não altera nenhuma decisão.

Saída: artifacts/metrics/model_analysis.json e figuras eval_learning_curve.png,
eval_error_segments.png.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, RepeatedKFold, learning_curve

from ml import config, figures
from ml.data import load_train
from ml.features import APP_FEATURES
from ml.metrics import SCORING, rmsle

ANALYSIS_PATH = config.METRICS_DIR / "model_analysis.json"


def segment_table(df: pd.DataFrame, by: str, order=None, min_n: int = 1) -> list[dict]:
    rows = []
    for key, g in df.groupby(by, observed=True, sort=False):
        if len(g) < min_n:
            continue
        rows.append({
            "segment": str(key),
            "n": int(len(g)),
            "mae": float(g["abs_err"].mean()),
            "mape_pct": float(g["ape"].mean()),
            "bias": float(g["res"].mean()),  # real − previsto: positivo = modelo subestima
            "median_price": float(g["SalePrice"].median()),
        })
    if order:
        rows.sort(key=lambda r: order.index(r["segment"]))
    else:
        rows.sort(key=lambda r: r["mape_pct"], reverse=True)
    return rows


def main() -> None:
    t_start = time.perf_counter()
    train = load_train()
    split = json.loads(config.SPLIT_PATH.read_text(encoding="utf-8"))
    comp = json.loads(config.MODEL_COMPARISON_PATH.read_text(encoding="utf-8"))
    meta = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    dev = train[train[config.ID_COLUMN].isin(split["dev_ids"])].sort_values(config.ID_COLUMN)
    hold = train[train[config.ID_COLUMN].isin(split["holdout_ids"])].sort_values(config.ID_COLUMN)
    X, y = dev[APP_FEATURES], dev[config.TARGET]
    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    selected = joblib.load(config.APP_MODEL_PATH)

    # ------------------------------------------------------------ 1. curva de aprendizado
    sizes, tr, va = learning_curve(clone(selected), X, y, train_sizes=np.linspace(0.1, 1.0, 8), cv=cv,
                                   scoring=SCORING["rmsle"], n_jobs=1, shuffle=False)
    lc = {
        "model": meta["model_config"],
        "train_sizes": [int(s) for s in sizes],
        "train_rmsle_mean": [float(-v) for v in tr.mean(axis=1)], "train_rmsle_std": [float(v) for v in tr.std(axis=1)],
        "val_rmsle_mean": [float(-v) for v in va.mean(axis=1)], "val_rmsle_std": [float(v) for v in va.std(axis=1)],
        "note": "Tamanhos = imóveis usados para treinar em cada fold (até 80% do desenvolvimento). Validação nos 20% restantes de cada fold.",
    }

    # ------------------------------------------------------------ 2. vencedor ou empate?
    app_rows = sorted((r for r in comp["feature_sets"]["app"]["results"] if r["algorithm"] != "DummyRegressor"), key=lambda r: r["cv_rmsle_mean"])
    top2 = app_rows[:2]
    cand_files = {c["config"]: c["file"] for c in meta.get("candidates", [])}
    models = [joblib.load(config.MODELS_DIR / cand_files[r["config"]]) for r in top2]
    rkf = RepeatedKFold(n_splits=5, n_repeats=4, random_state=config.RANDOM_STATE)
    scores = [[], []]
    for tr_idx, va_idx in rkf.split(X):
        for k, m in enumerate(models):
            est = clone(m).fit(X.iloc[tr_idx], y.iloc[tr_idx])
            scores[k].append(rmsle(y.iloc[va_idx], est.predict(X.iloc[va_idx])))
    a, b = np.array(scores[0]), np.array(scores[1])
    d = b - a  # positivo = o primeiro (melhor na CV) venceu a partição
    tie = {
        "feature_set": "app",
        "method": "RepeatedKFold(5 partições × 4 repetições = 20) no desenvolvimento, pareado, hiperparâmetros fixos",
        "models": [top2[0]["config"], top2[1]["config"]],
        "rmsle": [a.tolist(), b.tolist()],
        "mean": [float(a.mean()), float(b.mean())],
        "diff_mean": float(d.mean()), "diff_std": float(d.std(ddof=1)),
        "first_wins": int((d > 0).sum()), "n_splits": int(len(d)),
        "verdict": "empate técnico" if abs(d.mean()) < d.std(ddof=1) else f"{top2[0]['config'] if d.mean() > 0 else top2[1]['config']} melhor de forma consistente",
        "rule": "Empate técnico quando |diferença média| < desvio-padrão das diferenças entre partições.",
    }
    full_rows = sorted((r for r in comp["feature_sets"]["full"]["results"] if r["algorithm"] != "DummyRegressor"), key=lambda r: r["cv_rmsle_mean"])
    fa, fb = np.array(full_rows[0]["cv_rmsle_folds"]), np.array(full_rows[1]["cv_rmsle_folds"])
    fd = fb - fa
    tie_full = {
        "feature_set": "full", "method": "diferenças pareadas nos 5 folds já calculados no treino (sem novos ajustes)",
        "models": [full_rows[0]["config"], full_rows[1]["config"]], "rmsle": [fa.tolist(), fb.tolist()],
        "diff_mean": float(fd.mean()), "diff_std": float(fd.std(ddof=1)), "first_wins": int((fd > 0).sum()), "n_splits": 5,
        "verdict": "empate técnico" if abs(fd.mean()) < fd.std(ddof=1) else "diferença consistente nos 5 folds",
    }

    # ------------------------------------------------------------ 3. erro por segmento (holdout)
    pred = selected.predict(hold[APP_FEATURES])
    df = hold[[config.ID_COLUMN, config.TARGET, "SaleCondition", "Neighborhood"]].copy()
    df["pred"] = pred
    df["res"] = df[config.TARGET] - df["pred"]
    df["abs_err"] = df["res"].abs()
    df["ape"] = 100 * df["abs_err"] / df[config.TARGET]
    q = df[config.TARGET].quantile([0.25, 0.5, 0.75]).tolist()
    edges = [df[config.TARGET].min(), *q, df[config.TARGET].max()]
    labels = [f"Q{i + 1}: US$ {edges[i] / 1000:.0f}–{edges[i + 1] / 1000:.0f} mil" for i in range(4)]
    df["faixa"] = pd.qcut(df[config.TARGET], 4, labels=labels)
    segments = {
        "price_quartile": segment_table(df, "faixa", order=labels),
        "sale_condition": segment_table(df, "SaleCondition"),
        "neighborhood": segment_table(df, "Neighborhood", min_n=8),
        "n_total": int(len(df)),
        "bias_definition": "viés = média de (real − previsto). Positivo: o modelo subestima; negativo: superestima.",
    }

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": meta["model_config"], "model_version": meta["model_version"],
        "learning_curve": lc, "tie_test": tie, "tie_test_full": tie_full, "segments": segments,
        "runtime_seconds": round(time.perf_counter() - t_start, 1),
    }
    ANALYSIS_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    figures.analysis_figures()
    print(f"Curva de aprendizado: val RMSLE {lc['val_rmsle_mean'][0]:.4f} (n={lc['train_sizes'][0]}) → {lc['val_rmsle_mean'][-1]:.4f} (n={lc['train_sizes'][-1]})")
    print(f"Empate (app): {tie['models'][0]} vs {tie['models'][1]}: dif média {tie['diff_mean']:+.4f} ± {tie['diff_std']:.4f}, "
          f"1º vence {tie['first_wins']}/{tie['n_splits']} → {tie['verdict']}")
    print(f"Empate (full): dif {tie_full['diff_mean']:+.4f} ± {tie_full['diff_std']:.4f}, 1º vence {tie_full['first_wins']}/5 → {tie_full['verdict']}")
    for r in pq:
        print(f"  {r['segment']:<28} n={r['n']:<3} MAPE={r['mape_pct']:5.1f}%  viés={r['bias']:+,.0f}")
    print(f"Salvo: {ANALYSIS_PATH.name} ({out['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
