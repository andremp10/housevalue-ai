"""Leitura dos artefatos de dados e métricas para o dashboard (PRD §13.5–13.7).

Todos os números vêm de arquivos gerados pelos scripts de ML; nada é calculado
a partir de dados fictícios. Se um artefato faltar, a API responde 503.
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.core.config import Settings


class ArtifactUnavailableError(RuntimeError):
    pass


class AnalyticsService:
    def __init__(self, settings: Settings):
        self.s = settings

    def _read(self, path: Path) -> dict:
        if not path.is_file():
            raise ArtifactUnavailableError(f"Artefato indisponível: {path.name}. Execute os scripts de ML (ver README).")
        return json.loads(path.read_text(encoding="utf-8"))

    def overview(self) -> dict:
        profile = self._read(self.s.artifacts_dir / "data_profile.json")
        meta = self._read(self.s.models_dir / "model_metadata.json")
        t = profile["target"]["describe"]
        return {
            "dataset": {
                "name": "House Prices: Advanced Regression Techniques (Kaggle)",
                "location": "Ames, Iowa, EUA",
                "period": "vendas de 2006 a 2010",
                "currency": "USD (valores históricos, sem correção)",
                "labelled_rows": profile["files"]["train.csv"]["rows"],
                "unlabelled_rows": profile["files"]["test.csv"]["rows"],
                "columns": profile["files"]["train.csv"]["columns"],
                "n_features": profile["feature_groups"]["n_features"],
            },
            "sale_price": {
                "min": t["min"], "max": t["max"], "median": t["50%"], "mean": t["mean"], "std": t["std"],
                "q1": t["25%"], "q3": t["75%"], "skew": profile["target"]["skew"],
            },
            "model": {
                "name": meta["model_name"],
                "config": meta["model_config"],
                "version": meta["model_version"],
                "trained_at": meta["trained_at"],
                "evaluated_at": meta.get("evaluated_at"),
                "n_input_features": len(meta["input_features"]),
                "holdout_metrics": meta.get("holdout_metrics"),
                "holdout_n": meta.get("holdout_n"),
                "baseline_holdout_metrics": meta.get("baseline_holdout_metrics"),
                "cv_metrics": meta.get("cv_metrics"),
                "full_feature_reference": meta.get("full_feature_reference"),
            },
        }

    def model_comparison(self) -> dict:
        comp = self._read(self.s.metrics_dir / "model_comparison.json")
        holdout = self._read(self.s.metrics_dir / "holdout_evaluation.json")
        return {
            "protocol": comp["protocol"],
            "feature_sets": {
                k: {
                    "features": v["features"],
                    "selected": v["selected"],
                    "baseline": v["baseline"],
                    "results": [{kk: vv for kk, vv in r.items() if not kk.endswith("_folds")} | {"cv_rmsle_folds": r["cv_rmsle_folds"]} for r in v["results"]],
                }
                for k, v in comp["feature_sets"].items()
            },
            "history_v1": self.history_v1(),
            "candidates_holdout": holdout.get("candidates"),
            "holdout": {
                "n": holdout["holdout"]["n"],
                "metrics": holdout["metrics"],
                "beats_baseline": holdout["beats_baseline"],
                "cv_vs_holdout": holdout["cv_vs_holdout"],
                "artifacts": {k: v["config"] for k, v in holdout["evaluated_artifacts"].items()},
            },
        }

    def evaluation(self) -> dict:
        h = self._read(self.s.metrics_dir / "holdout_evaluation.json")
        return {"points": h["points_app"], "residuals": h["residuals_app"], "top_errors": h["top_errors_app"], "model": h["evaluated_artifacts"]["app"]["config"], "n": h["holdout"]["n"]}

    def feature_importance(self) -> dict:
        return self._read(self.s.metrics_dir / "feature_importance.json")

    def eda(self) -> dict:
        return self._read(self.s.reports_dir / "eda_summary.json")

    def holdout_samples(self) -> dict:
        return self._read(self.s.reports_dir / "holdout_samples.json")

    def model_analysis(self) -> dict:
        return self._read(self.s.metrics_dir / "model_analysis.json")

    def history_v1(self) -> dict | None:
        base = self.s.artifacts_dir / "history" / "v1"
        try:
            meta = json.loads((base / "model_metadata.json").read_text(encoding="utf-8"))
            ev = json.loads((base / "holdout_evaluation.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return {"model_config": meta["model_config"], "model_version": meta["model_version"], "trained_at": meta["trained_at"],
                "cv_metrics": meta["cv_metrics"], "holdout_metrics": ev["metrics"]["app"], "full_holdout_metrics": ev["metrics"]["full"]}
