from __future__ import annotations

from fastapi import APIRouter, Request

from backend.schemas.prediction import ErrorResponse

router = APIRouter(prefix="/api/analytics", tags=["indicadores"])
_ERR = {503: {"model": ErrorResponse, "description": "Artefatos de métricas indisponíveis"}}


@router.get("/overview", summary="Indicadores reais da base e do modelo", responses=_ERR)
def overview(request: Request) -> dict:
    return request.app.state.analytics_service.overview()


@router.get("/model-comparison", summary="Comparação dos modelos (CV e holdout)", responses=_ERR)
def model_comparison(request: Request) -> dict:
    return request.app.state.analytics_service.model_comparison()


@router.get("/feature-importance", summary="Importância das variáveis (com o tipo de medida)", responses=_ERR)
def feature_importance(request: Request) -> dict:
    return request.app.state.analytics_service.feature_importance()


@router.get("/evaluation", summary="Diagnóstico do holdout: real × previsto, resíduos, maiores erros", responses=_ERR)
def evaluation(request: Request) -> dict:
    return request.app.state.analytics_service.evaluation()


@router.get("/eda", summary="Agregações da análise exploratória", responses=_ERR)
def eda(request: Request) -> dict:
    return request.app.state.analytics_service.eda()


@router.get("/model-analysis", summary="Curva de aprendizado, teste de empate e erro por segmento", responses=_ERR)
def model_analysis(request: Request) -> dict:
    return request.app.state.analytics_service.model_analysis()
