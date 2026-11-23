"""Testes do modelo salvo (PRD §9, §14, §18.2)."""
import json

import joblib
import numpy as np
import pandas as pd

from ml import config
from ml.data import sha256_file
from ml.features import APP_FEATURES
from ml.metrics import all_metrics, rmsle


def _meta():
    return json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))


def test_metrics_functions():
    y = np.array([100.0, 200.0, 300.0])
    assert rmsle(y, y) == 0
    m = all_metrics(y, y + 10)
    assert np.isclose(m["mae"], 10) and np.isclose(m["rmse"], 10)
    assert rmsle([100.0], [-5.0]) == rmsle([100.0], [0.0])  # previsão negativa truncada em 0


def test_artifact_matches_metadata(trained):
    meta = _meta()
    assert sha256_file(config.APP_MODEL_PATH) == meta["artifact_sha256"]
    model = joblib.load(config.APP_MODEL_PATH)
    assert list(model.feature_names_in_) == meta["input_features"] == APP_FEATURES
    assert config.TARGET not in meta["input_features"] and config.ID_COLUMN not in meta["input_features"]
    for key in ("model_name", "model_version", "trained_at", "target", "random_state", "cv_metrics", "holdout_metrics", "library_versions", "source_sha256"):
        assert meta.get(key) is not None, key


def test_saved_pipeline_reproduces_holdout_predictions(train_df, trained):
    split = json.loads(config.SPLIT_PATH.read_text(encoding="utf-8"))
    hold = train_df[train_df[config.ID_COLUMN].isin(split["holdout_ids"])].sort_values(config.ID_COLUMN)
    model = joblib.load(config.APP_MODEL_PATH)
    preds = model.predict(hold[APP_FEATURES])
    saved = pd.read_csv(config.METRICS_DIR / "holdout_predictions.csv")
    assert np.allclose(preds, saved["pred_app"], atol=0.01)
    m = all_metrics(hold[config.TARGET], preds)
    assert np.isclose(m["rmsle"], _meta()["holdout_metrics"]["rmsle"])


def test_holdout_never_used_in_training(train_df, trained):
    split = json.loads(config.SPLIT_PATH.read_text(encoding="utf-8"))
    assert set(split["dev_ids"]).isdisjoint(split["holdout_ids"])
    assert len(split["dev_ids"]) + len(split["holdout_ids"]) == len(train_df)
    assert _meta()["training_data"].startswith(f"partição de desenvolvimento ({len(split['dev_ids'])}")


def test_all_models_compared_on_same_folds_and_features(trained):
    comp = json.loads(config.MODEL_COMPARISON_PATH.read_text(encoding="utf-8"))
    for fs in comp["feature_sets"].values():
        algos = {r["algorithm"] for r in fs["results"]}
        assert "DummyRegressor" in algos and len(algos - {"DummyRegressor"}) >= 2
        n_folds = {len(r["cv_rmsle_folds"]) for r in fs["results"]}
        assert n_folds == {config.CV_FOLDS}


def test_selected_model_beats_baseline(trained):
    """Se falhar, o resultado é real e deve ser documentado — não ajustar o teste."""
    rep = json.loads(config.HOLDOUT_REPORT_PATH.read_text(encoding="utf-8"))
    assert rep["metrics"]["app"]["rmsle"] < rep["metrics"]["baseline"]["rmsle"]
    assert rep["metrics"]["app"]["mae"] < rep["metrics"]["baseline"]["mae"]
    comp = json.loads(config.MODEL_COMPARISON_PATH.read_text(encoding="utf-8"))
    assert comp["feature_sets"]["app"]["selected_beats_baseline_cv"]["rmsle"]


def test_submission_format(trained, test_df):
    sub = pd.read_csv(config.REPORTS_DIR / "submission.csv")
    assert list(sub.columns) == [config.ID_COLUMN, config.TARGET]
    assert len(sub) == len(test_df) and sub[config.TARGET].notna().all() and (sub[config.TARGET] > 0).all()


def test_candidates_hashes_and_folds(trained):
    meta = _meta()
    assert meta["model_version"] == "2.0.0"
    for c in meta["candidates"]:
        assert sha256_file(config.MODELS_DIR / c["file"]) == c["sha256"]
    assert sum(c["is_selected"] for c in meta["candidates"]) == 1
    comp = json.loads(config.MODEL_COMPARISON_PATH.read_text(encoding="utf-8"))
    for fs in comp["feature_sets"].values():
        assert {len(r["cv_rmsle_folds"]) for r in fs["results"]} == {config.CV_FOLDS}
