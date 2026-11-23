from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api", tags=["saúde"])


@router.get("/health", summary="Estado da API e do modelo")
def health(request: Request) -> dict:
    svc = request.app.state.prediction_service
    return {
        "status": "ok",
        "model_loaded": svc.ready,
        "app_version": request.app.state.settings.app_version,
        "model_error": None if svc.ready else svc.load_error,
    }
