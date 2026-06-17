"""MLflow experiment tracking.

Disabled automatically when MLFLOW_TRACKING_URI is not set — zero overhead
on HF Spaces where no tracking server is configured.
"""
import logging
import os
from typing import TYPE_CHECKING

import mlflow

if TYPE_CHECKING:
    from src.core.schemas import CVQualityReport
    from src.services.sector_detector import SectorDetectionResult

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "ats_cv_scorer_production"


class ExperimentTracker:
    def __init__(self) -> None:
        self._enabled = bool(os.getenv("MLFLOW_TRACKING_URI"))
        if self._enabled:
            try:
                mlflow.set_experiment(EXPERIMENT_NAME)
            except Exception as exc:
                logger.warning("experiment_tracker | mlflow.set_experiment failed: %s", exc)
                self._enabled = False

    def log_cv_analysis(
        self,
        sector_result: "SectorDetectionResult",
        quality_report: "CVQualityReport",
        extraction_method: str,
        latency_ms: float,
        word_count: int,
    ) -> None:
        if not self._enabled:
            return
        try:
            with mlflow.start_run():
                mlflow.log_params({
                    "profile_id": sector_result.profile_id,
                    "sector": sector_result.sector,
                    "extraction_method": extraction_method,
                })
                mlflow.log_metrics({
                    "detection_confidence": sector_result.confidence,
                    "profile_strength_score": float(quality_report.profile_strength.score),
                    "word_count": float(word_count),
                    "latency_ms": latency_ms,
                })
        except Exception as exc:
            logger.warning("experiment_tracker | log_cv_analysis failed: %s", exc)

    def log_benchmark(
        self,
        report_path: str,
        params: dict,
        metrics: dict,
    ) -> None:
        if not self._enabled:
            return
        try:
            with mlflow.start_run(run_name="benchmark"):
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(report_path)
        except Exception as exc:
            logger.warning("experiment_tracker | log_benchmark failed: %s", exc)
