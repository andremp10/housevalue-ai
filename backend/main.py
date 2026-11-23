"""Aplicação FastAPI do HouseValue AI.

Executar: uvicorn backend.main:app --reload
Documentação: http://localhost:8000/docs  |  Interface: http://localhost:8000/
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api import analytics, health, predictions
from backend.core.config import get_settings
from backend.services.analytics_service import AnalyticsService, ArtifactUnavailableError
from backend.services.prediction_service import DomainValidationError, ModelUnavailableError, PredictionService

log = logging.getLogger("housevalue")

_TYPE_MESSAGES = {
    "missing": "Campo obrigatório ausente.",
    "int_type": "Informe um número inteiro.",
    "int_parsing": "Informe um número inteiro.",
    "float_type": "Informe um número.",
    "float_parsing": "Informe um número.",
    "greater_than": "Informe um valor maior que zero.",
    "greater_than_equal": "Informe um valor maior ou igual a zero.",
    "string_type": "Informe um texto (código da categoria).",
    "extra_forbidden": "Campo não reconhecido pelo modelo.",
    "model_attributes_type": "O corpo da requisição deve ser um objeto JSON.",
    "json_invalid": "JSON inválido.",
}


def create_app(artifacts_dir: Path | None = None) -> FastAPI:
    settings = get_settings(artifacts_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.prediction_service.load()
        yield

    app = FastAPI(
        title="HouseValue AI — API",
        version=settings.app_version,
        description=(
            "Estimativa experimental do preço de venda (USD) de imóveis de Ames, Iowa (EUA), "
            "com base em vendas históricas de 2006–2010 (Kaggle House Prices). Projeto acadêmico — não é laudo de avaliação."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.prediction_service = PredictionService(settings)
    app.state.analytics_service = AnalyticsService(settings)

    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next):
        if request.method == "POST":
            length = request.headers.get("content-length")
            if length is not None and length.isdigit() and int(length) > settings.max_body_bytes:
                return JSONResponse(status_code=413, content={"detail": "Requisição muito grande."})
            body = await request.body()
            if len(body) > settings.max_body_bytes:
                return JSONResponse(status_code=413, content={"detail": "Requisição muito grande."})
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def on_validation(_: Request, exc: RequestValidationError):
        errors = []
        for e in exc.errors():
            loc = [str(x) for x in e.get("loc", []) if x != "body"]
            errors.append({"field": ".".join(loc) or "body", "message": _TYPE_MESSAGES.get(e.get("type"), "Valor inválido.")})
        return JSONResponse(status_code=422, content={"detail": "Dados de entrada inválidos.", "errors": errors})

    @app.exception_handler(DomainValidationError)
    async def on_domain(_: Request, exc: DomainValidationError):
        return JSONResponse(status_code=422, content={"detail": "Valores fora do domínio aceito pelo modelo.", "errors": exc.errors})

    @app.exception_handler(ModelUnavailableError)
    async def on_model_missing(_: Request, exc: ModelUnavailableError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(ArtifactUnavailableError)
    async def on_artifact_missing(_: Request, exc: ArtifactUnavailableError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def on_unexpected(_: Request, exc: Exception):
        log.exception("Erro inesperado", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "Falha interna ao processar a solicitação."})

    app.include_router(health.router)
    app.include_router(predictions.router)
    app.include_router(analytics.router)

    if settings.frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    return app


app = create_app()
