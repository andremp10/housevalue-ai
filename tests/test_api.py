"""Testes da API (PRD §13, §18.3)."""
import json

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from ml import config
from ml.features import APP_FEATURES, FIELD_META
from ml.units import AREA_FIELDS, SQFT_TO_M2


@pytest.fixture(scope="module")
def client(trained):
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture(scope="module")
def holdout_row(train_df, trained):
    split = json.loads(config.SPLIT_PATH.read_text(encoding="utf-8"))
    row = train_df[train_df[config.ID_COLUMN] == split["holdout_ids"][0]].iloc[0]
    body = {}
    for c in APP_FEATURES:
        api = FIELD_META[c]["api_name"]
        if c in AREA_FIELDS:
            body[api] = float(row[c]) * SQFT_TO_M2  # API em m²
        elif FIELD_META[c].get("unit"):
            body[api] = int(row[c])
        else:
            body[api] = str(row[c])
    return row, body


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["model_loaded"] is True


def test_model_info_is_real_metadata(client):
    r = client.get("/api/model/info").json()
    meta = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    assert r["model_config"] == meta["model_config"]
    assert r["holdout_metrics"] == meta["holdout_metrics"]
    assert r["input_features"] == APP_FEATURES
    assert "artifact" not in r and str(config.PROJECT_ROOT) not in json.dumps(r)  # sem caminhos locais


def test_form_options_consistent_with_pipeline(client):
    r = client.get("/api/form-options").json()
    names = [f["name"] for f in r["fields"]]
    assert names == APP_FEATURES
    model = joblib.load(config.APP_MODEL_PATH)
    pipe = model.regressor_ if hasattr(model, "regressor_") else model
    ct = pipe.named_steps["prep"].named_steps["columns"]
    cat_tr = ct.named_transformers_["cat"]
    # v1: Pipeline(imputer + OneHotEncoder); v2: OrdinalEncoder direto (HGB nativo)
    ohe = cat_tr.named_steps["onehot"] if hasattr(cat_tr, "named_steps") else cat_tr
    cat_cols = next(cols for name, _, cols in ct.transformers_ if name == "cat")
    cats = dict(zip(cat_cols, ohe.categories_))
    for f in r["fields"]:
        if f["type"] == "category":
            assert {o["value"] for o in f["options"]} == set(map(str, cats[f["name"]]))


def test_predict_matches_saved_pipeline(client, holdout_row):
    row, body = holdout_row
    r = client.post("/api/predict", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["currency"] == "USD" and data["target"] == "SalePrice" and data["is_estimate"] is True
    model = joblib.load(config.APP_MODEL_PATH)
    expected = float(model.predict(pd.DataFrame([row[APP_FEATURES]]))[0])
    assert abs(data["predicted_sale_price"] - expected) < 0.01
    assert "Ames" in data["dataset_context"]


@pytest.mark.parametrize(
    "mutate,field",
    [
        (lambda b: b.pop("gr_liv_area_m2"), "gr_liv_area_m2"),
        (lambda b: b.update(gr_liv_area_m2="grande"), "gr_liv_area_m2"),
        (lambda b: b.update(gr_liv_area_m2=10_000_000.0), "gr_liv_area_m2"),
        (lambda b: b.update(neighborhood="Fortaleza"), "neighborhood"),
        (lambda b: b.update(extra_field=1), "extra_field"),
        (lambda b: b.update(year_built=2005, year_remod_add=1990), "year_remod_add"),
    ],
)
def test_invalid_inputs_return_422(client, holdout_row, mutate, field):
    body = dict(holdout_row[1])
    mutate(body)
    r = client.post("/api/predict", json=body)
    assert r.status_code == 422
    payload = r.json()
    assert "Traceback" not in r.text
    assert any(e["field"] == field for e in payload["errors"])


def test_body_too_large(client):
    r = client.post("/api/predict", content=b"{" + b" " * 20_000 + b"}", headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_analytics_endpoints(client):
    ov = client.get("/api/analytics/overview").json()
    profile = json.loads(config.DATA_PROFILE_PATH.read_text(encoding="utf-8"))
    assert ov["dataset"]["labelled_rows"] == profile["files"]["train.csv"]["rows"]
    assert ov["sale_price"]["median"] == profile["target"]["describe"]["50%"]
    mc = client.get("/api/analytics/model-comparison").json()
    assert mc["feature_sets"]["app"]["selected"] and mc["holdout"]["metrics"]["app"]["rmsle"] > 0
    fi = client.get("/api/analytics/feature-importance").json()
    assert "measure" in fi["app"]["permutation_holdout"]
    for path in ("/api/analytics/evaluation", "/api/analytics/eda", "/docs", "/"):
        assert client.get(path).status_code == 200, path


def test_without_artifacts_returns_503(tmp_path):
    with TestClient(create_app(artifacts_dir=tmp_path)) as c:
        h = c.get("/api/health").json()
        assert h["model_loaded"] is False
        assert c.get("/api/model/info").status_code == 503
        assert c.get("/api/form-options").status_code == 503
        assert c.get("/api/analytics/overview").status_code == 503
        r = c.post("/api/predict", json={})
        assert r.status_code in (422, 503)


def test_tampered_model_is_rejected(tmp_path):
    import shutil
    models = tmp_path / "models"
    models.mkdir()
    for name in ("model_metadata.json", "form_schema.json"):
        shutil.copy(config.MODELS_DIR / name, models / name)
    (models / "house_price_pipeline.joblib").write_bytes(b"not a model")
    with TestClient(create_app(artifacts_dir=tmp_path)) as c:
        h = c.get("/api/health").json()
        assert h["model_loaded"] is False and "SHA-256" in h["model_error"]


def test_sensitivity_uses_real_model_predictions(client, holdout_row):
    row, body = holdout_row
    r = client.post("/api/predict/sensitivity", json={"inputs": body, "field": "gr_liv_area_m2"})
    assert r.status_code == 200, r.text
    data = r.json()
    model = joblib.load(config.APP_MODEL_PATH)
    form = client.get("/api/form-options").json()
    f = next(x for x in form["fields"] if x["name"] == "GrLivArea")
    for pt in data["points"][::7]:
        X = pd.DataFrame([row[APP_FEATURES]])
        X["GrLivArea"] = round(pt["value"] / SQFT_TO_M2)  # pontos da grade são sq ft inteiros, exibidos em m²
        assert abs(float(model.predict(X)[0]) - pt["predicted"]) < 0.01
        assert f["min"] - 0.1 <= pt["value"] <= f["max"] + 0.1  # nunca extrapola a faixa de treino
    assert abs(data["current"]["value"] - row["GrLivArea"] * SQFT_TO_M2) < 0.01
    assert data["unit"] == "m²"
    pred = client.post("/api/predict", json=body).json()["predicted_sale_price"]
    assert abs(data["current"]["predicted"] - pred) < 0.01


def test_sensitivity_categorical_and_constraints(client, holdout_row):
    _, body = holdout_row
    nb = client.post("/api/predict/sensitivity", json={"inputs": body, "field": "neighborhood"}).json()
    opts = next(f for f in client.get("/api/form-options").json()["fields"] if f["api_name"] == "neighborhood")["options"]
    assert {p["value"] for p in nb["points"]} == {o["value"] for o in opts}
    yb = client.post("/api/predict/sensitivity", json={"inputs": body, "field": "year_built"}).json()
    assert all(p["value"] <= body["year_remod_add"] for p in yb["points"])  # respeita reforma >= construção
    assert client.post("/api/predict/sensitivity", json={"inputs": body, "field": "SalePrice"}).status_code == 422


def test_samples_are_real_holdout_rows(client, train_df):
    data = client.get("/api/samples").json()
    split = json.loads(config.SPLIT_PATH.read_text(encoding="utf-8"))
    assert data["n"] == len(split["holdout_ids"]) == len(data["samples"])
    s = data["samples"][0]
    assert s["id"] in split["holdout_ids"]
    real = train_df.loc[train_df[config.ID_COLUMN] == s["id"]].iloc[0]
    assert s["actual"] == real[config.TARGET]
    pred = client.post("/api/predict", json=s["inputs"]).json()["predicted_sale_price"]
    assert abs(pred - s["predicted"]) < 0.01


def test_form_options_has_training_distributions(client):
    r = client.get("/api/form-options").json()
    nums = [f for f in r["fields"] if f["type"] == "integer"]
    for f in nums:
        d = f["distribution"]
        assert len(d["histogram"]["counts"]) + 1 == len(d["histogram"]["edges"])
        assert f["min"] <= d["common_range"][0] <= d["common_range"][1] <= f["max"]
    assert r["neighborhoods"]["stats"]["CollgCr"]["n"] > 0


# ---------------------------------------------------------------- v2: comparações

def test_compare_matches_each_candidate(client, holdout_row):
    row, body = holdout_row
    data = client.post("/api/predict/compare", json=body).json()
    meta = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    assert len(data["rows"]) == len(meta["candidates"]) >= 4
    X = pd.DataFrame([row[APP_FEATURES]])
    by_cfg = {c["config"]: c for c in meta["candidates"]}
    for r in data["rows"]:
        m = joblib.load(config.MODELS_DIR / by_cfg[r["config"]]["file"])
        assert abs(float(m.predict(X)[0]) - r["predicted"]) < 0.01
    sel = [r for r in data["rows"] if r["is_selected"]]
    assert len(sel) == 1
    assert abs(sel[0]["predicted"] - client.post("/api/predict", json=body).json()["predicted_sale_price"]) < 0.01


def test_decompose_single_and_multiple_changes(client, holdout_row):
    _, body = holdout_row
    one = dict(body, overall_qual=min(body["overall_qual"] + 1, 10))
    d1 = client.post("/api/predict/decompose", json={"base": body, "target": one}).json()
    assert len(d1["effects"]) == 1
    assert abs(d1["effects"][0]["effect"] - d1["total"]) < 0.02 and abs(d1["interaction"]) < 0.02
    p_base = client.post("/api/predict", json=body).json()["predicted_sale_price"]
    p_one = client.post("/api/predict", json=one).json()["predicted_sale_price"]
    assert abs(d1["total"] - (p_one - p_base)) < 0.02

    two = dict(one, garage_cars=0 if body["garage_cars"] else 1, neighborhood="NridgHt" if body["neighborhood"] != "NridgHt" else "OldTown")
    d2 = client.post("/api/predict/decompose", json={"base": body, "target": two}).json()
    assert len(d2["effects"]) == 3
    total = sum(e["effect"] for e in d2["effects"]) + d2["interaction"]
    assert abs(total - d2["total"]) < 0.05
    for e in d2["effects"]:  # cada efeito confere com uma previsão direta
        only = dict(body, **{e["field"]: two[e["field"]]})
        p = client.post("/api/predict", json=only).json()["predicted_sale_price"]
        assert abs((p - p_base) - e["effect"]) < 0.02


def test_decompose_rejects_invalid(client, holdout_row):
    _, body = holdout_row
    bad = dict(body, neighborhood="Fortaleza")
    assert client.post("/api/predict/decompose", json={"base": body, "target": bad}).status_code == 422


def test_model_analysis_endpoint(client):
    a = client.get("/api/analytics/model-analysis").json()
    split = json.loads(config.SPLIT_PATH.read_text(encoding="utf-8"))
    assert sum(r["n"] for r in a["segments"]["price_quartile"]) == len(split["holdout_ids"])
    assert sum(r["n"] for r in a["segments"]["sale_condition"]) == len(split["holdout_ids"])
    lc = a["learning_curve"]
    assert len(lc["train_sizes"]) == len(lc["val_rmsle_mean"]) >= 5
    assert a["tie_test"]["n_splits"] == 20 and len(a["tie_test"]["rmsle"][0]) == 20


def test_model_comparison_has_v2_and_history(client):
    mc = client.get("/api/analytics/model-comparison").json()
    cfgs = [r["config"] for r in mc["feature_sets"]["app"]["results"]]
    assert any("monotônico" in c for c in cfgs) and any("categorias nativas" in c for c in cfgs)
    assert mc["history_v1"] and mc["history_v1"]["model_version"] == "1.0.0"
    assert mc["candidates_holdout"]["rows"]



# ---------------------------------------------------------------- v3: m², imóveis reais e faixa de erro

def test_area_fields_in_m2(client, holdout_row):
    row, body = holdout_row
    form = client.get("/api/form-options").json()
    area = {f["name"]: f for f in form["fields"] if f["name"] in AREA_FIELDS}
    assert set(area) == set(AREA_FIELDS)
    for c, f in area.items():
        assert f["unit"] == "m²" and f["api_name"] == AREA_FIELDS[c]
        assert "display_factor" not in f and "model_unit" not in f  # detalhe interno não vaza
    # mesmo imóvel: m² na API equivale ao sq ft usado no treino
    model = joblib.load(config.APP_MODEL_PATH)
    expected = float(model.predict(pd.DataFrame([row[APP_FEATURES]]))[0])
    got = client.post("/api/predict", json=body).json()
    assert abs(got["predicted_sale_price"] - expected) < 0.01
    shown = next(i for i in got["inputs"] if i["column"] == "GrLivArea")
    assert shown["display"].endswith("m²")


def test_real_houses_run_live(client, train_df):
    data = client.get("/api/real-houses").json()
    split = json.loads(config.SPLIT_PATH.read_text(encoding="utf-8"))
    saved = pd.read_csv(config.METRICS_DIR / "holdout_predictions.csv").set_index("Id")
    assert data["summary"]["n"] == len(split["holdout_ids"]) == len(data["rows"])
    for r in data["rows"][::29]:
        assert abs(r["predicted"] - saved.loc[r["id"], "pred_app"]) < 0.01
        assert r["actual"] == saved.loc[r["id"], "SalePrice"]
        assert abs(r["area_m2"] - train_df.set_index("Id").loc[r["id"], "GrLivArea"] * SQFT_TO_M2) < 1e-6
    ev = json.loads(config.HOLDOUT_REPORT_PATH.read_text(encoding="utf-8"))
    assert abs(data["summary"]["mae"] - ev["metrics"]["app"]["mae"]) < 0.01


def test_error_band_is_estimated_without_holdout_and_validated(client):
    b = client.get("/api/error-band").json()
    assert b["level"] == 0.8 and b["factor_low"] < 1 < b["factor_high"]
    assert "cross_val_predict" in b["method"]
    rows = client.get("/api/real-houses").json()["rows"]
    inside = sum(r["inside_range"] for r in rows) / len(rows)
    assert abs(inside - b["holdout_coverage"]) < 1e-9
    assert 0.65 <= b["holdout_coverage"] <= 0.95  # conferência: perto do nível nominal de 80%


def test_similar_real_houses(client, holdout_row):
    _, body = holdout_row
    sim = client.post("/api/predict", json=body).json()["similar_real_houses"]
    assert len(sim) == 4 and all(s["actual"] > 0 for s in sim)
    assert sim[0]["same_neighborhood"]
