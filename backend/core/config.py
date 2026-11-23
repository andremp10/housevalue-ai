"""Configuração do backend. Lê variáveis de ambiente opcionais (ver .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ml import config as ml_config


@dataclass(frozen=True)
class Settings:
    app_name: str = "HouseValue AI"
    app_version: str = ml_config.APP_VERSION
    artifacts_dir: Path = field(default_factory=lambda: Path(os.environ.get("HOUSEVALUE_ARTIFACTS_DIR") or ml_config.ARTIFACTS_DIR))
    frontend_dir: Path = ml_config.PROJECT_ROOT / "frontend"
    max_body_bytes: int = int(os.environ.get("HOUSEVALUE_MAX_BODY_BYTES") or 16_384)
    cors_origins: tuple[str, ...] = tuple(
        o.strip() for o in (os.environ.get("HOUSEVALUE_CORS_ORIGINS") or "http://localhost:8000,http://127.0.0.1:8000,http://localhost:5500,http://127.0.0.1:5500").split(",") if o.strip()
    )

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def metrics_dir(self) -> Path:
        return self.artifacts_dir / "metrics"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"


def get_settings(artifacts_dir: Path | None = None) -> Settings:
    if artifacts_dir is not None:
        return Settings(artifacts_dir=Path(artifacts_dir))
    return Settings()
