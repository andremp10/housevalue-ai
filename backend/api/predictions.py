from __future__ import annotations

from fastapi import APIRouter, Request

from backend.schemas.prediction import ErrorResponse, DecomposeRequest, PredictionRequest, PredictionResponse, SensitivityRequest

router = APIRouter(prefix="/api", tags=["previsão"])

_ERRORS = {422: {"model": ErrorResponse, "description": "Entrada inválida"}, 503: {"model": ErrorResponse, "description": "Modelo indisponível"}}


@router.get("/model/info", summary="Metadados do modelo em produção", responses=_ERRORS)
def model_info(request: Request) -> dict:
    svc = request.app.state.prediction_service
    svc.require()
    m = svc.metadata
    return {
        "model_name": m["model_name"],
        "model_config": m["model_config"],
        "model_version": m["model_version"],
        "trained_at": m["trained_at"],
        "evaluated_at": m.get("evaluated_at"),
        "target": m["target"],
        "currency": m["currency"],
        "dataset_context": m["dataset_context"],
        "input_features": m["input_features"],
        "input_api_names": m["input_api_names"],
        "hyperparameters": m["hyperparameters"],
        "log_target": m["log_target"],
        "preprocessing": m["preprocessing"],
        "training_data": m["training_data"],
        "validation": m["validation"],
        "cv_metrics": m["cv_metrics"],
        "holdout_metrics": m.get("holdout_metrics"),
        "holdout_n": m.get("holdout_n"),
        "baseline_holdout_metrics": m.get("baseline_holdout_metrics"),
        "library_versions": m["library_versions"],
    }


@router.get("/form-options", summary="Campos, rótulos, limites e categorias do formulário", responses=_ERRORS)
def form_options(request: Request) -> dict:
    svc = request.app.state.prediction_service
    svc.require()
    fields = svc.public_fields
    neighborhoods = None
    try:
        eda = request.app.state.analytics_service.eda()
        neighborhoods = {
            "source": "train.csv completo (descritivo): mediana de SalePrice e número de vendas por bairro",
            "stats": {r["value"]: {"median": r["median"], "n": r["n"]} for r in eda["by_neighborhood"]},
        }
    except Exception:  # noqa: BLE001 — estatística opcional para o seletor de bairros
        neighborhoods = None
    return {"fields": fields, "cross_field_rules": svc.form_schema["cross_field_rules"], "currency": "USD", "neighborhoods": neighborhoods,
            "distribution_source": "partição de desenvolvimento (dados de treino do modelo)"}


@router.post("/predict", summary="Estimar o preço de venda (USD)", response_model=PredictionResponse, responses=_ERRORS)
def predict(body: PredictionRequest, request: Request) -> dict:
    return request.app.state.prediction_service.predict(body.model_dump())


@router.post("/predict/sensitivity", summary="Curva de sensibilidade (ICE): varia uma variável mantendo as demais", responses=_ERRORS)
def predict_sensitivity(body: SensitivityRequest, request: Request) -> dict:
    return request.app.state.prediction_service.sensitivity(body.inputs.model_dump(), body.field)


@router.get("/samples", summary="Imóveis reais do holdout (preço real e previsão do modelo salvo)", responses=_ERRORS)
def samples(request: Request) -> dict:
    svc = request.app.state.prediction_service
    svc.require()
    data = request.app.state.analytics_service.holdout_samples()
    return {
        "source": data["source"],
        "n": data["n"],
        "samples": [
            {"id": s["id"], "inputs": svc.display_inputs(s["features"]), "actual": s["actual"], "predicted": s["predicted"],
             "sale_condition": s["sale_condition"], "year_sold": s["year_sold"]}
            for s in data["samples"]
        ],
    }


@router.post("/predict/compare", summary="Previsão de cada algoritmo candidato para o mesmo imóvel", responses=_ERRORS)
def predict_compare(body: PredictionRequest, request: Request) -> dict:
    return request.app.state.prediction_service.compare(body.model_dump())


@router.post("/predict/decompose", summary="Efeito isolado de cada característica alterada entre dois cenários", responses=_ERRORS)
def predict_decompose(body: DecomposeRequest, request: Request) -> dict:
    return request.app.state.prediction_service.decompose(body.base.model_dump(), body.target.model_dump())


@router.get("/real-houses", summary="Roda o modelo agora nos imóveis reais vendidos do holdout e compara com o preço real")
def real_houses(request: Request) -> dict:
    return request.app.state.prediction_service.run_real_houses()


@router.get("/error-band", summary="Faixa típica de erro (como foi estimada e a cobertura no holdout)", responses=_ERRORS)
def error_band(request: Request) -> dict:
    svc = request.app.state.prediction_service
    svc.require()
    if not svc.error_band:
        from backend.services.prediction_service import ModelUnavailableError
        raise ModelUnavailableError("Faixa de erro indisponível. Execute `python -m ml.app_assets`.")
    return svc.error_band
