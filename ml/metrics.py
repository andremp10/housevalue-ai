"""Métricas de regressão (PRD §9.6). Todas calculadas na escala original (USD)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score


def rmsle(y_true, y_pred) -> float:
    """RMSE entre log1p(real) e log1p(previsto).

    Previsões negativas (possíveis em modelos lineares sem transformação do alvo)
    são truncadas em 0 antes do log, pois preço negativo não tem significado;
    o número de truncamentos é reportado por `count_negative`.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def count_negative(y_pred) -> int:
    return int((np.asarray(y_pred) < 0).sum())


def all_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "rmsle": rmsle(y_true, y_pred),
        "negative_predictions": count_negative(y_pred),
    }


SCORING = {
    "rmsle": make_scorer(rmsle, greater_is_better=False),
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}
METRIC_NAMES = ["rmsle", "mae", "rmse", "r2"]
