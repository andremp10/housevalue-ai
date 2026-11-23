"""Explicabilidade global (PRD §10.1).

- Importância por permutação no HOLDOUT, sobre as colunas originais de entrada
  (o pipeline inteiro é permutado coluna a coluna), medida pelo aumento da RMSLE.
- Complementar: coeficientes (Ridge) ou importância nativa por impureza
  (Random Forest / árvores), sobre as colunas transformadas.
Nenhuma dessas medidas é causal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer

from ml import config
from ml.metrics import rmsle


def _inner_pipeline(model):
    return model.regressor_ if isinstance(model, TransformedTargetRegressor) else model


def permutation_on_holdout(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 20) -> list[dict]:
    scorer = make_scorer(rmsle, greater_is_better=False)
    r = permutation_importance(model, X, y, scoring=scorer, n_repeats=n_repeats, random_state=config.RANDOM_STATE, n_jobs=1)
    rows = [
        {"feature": c, "importance_mean": float(m), "importance_std": float(s)}
        for c, m, s in zip(X.columns, r.importances_mean, r.importances_std)
    ]
    return sorted(rows, key=lambda d: d["importance_mean"], reverse=True)


def native_importance(model) -> dict | None:
    pipe = _inner_pipeline(model)
    est = pipe.named_steps["model"]
    names = pipe.named_steps["prep"].named_steps["columns"].get_feature_names_out()
    if hasattr(est, "coef_"):
        vals = np.ravel(est.coef_)
        kind = "coeficientes do Ridge sobre variáveis padronizadas/one-hot" + (" (escala log1p do preço)" if isinstance(model, TransformedTargetRegressor) else " (USD)")
        rows = [{"feature": n, "value": float(v)} for n, v in zip(names, vals)]
        rows.sort(key=lambda d: abs(d["value"]), reverse=True)
        return {"measure": kind, "rows": rows[:30]}
    if hasattr(est, "feature_importances_"):
        rows = [{"feature": n, "value": float(v)} for n, v in zip(names, est.feature_importances_)]
        rows.sort(key=lambda d: d["value"], reverse=True)
        return {"measure": "importância por redução de impureza (nativa do modelo de árvores; viesada a favor de variáveis contínuas/alta cardinalidade)", "rows": rows[:30]}
    return None
