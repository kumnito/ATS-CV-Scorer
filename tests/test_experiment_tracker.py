"""Tests for ExperimentTracker — disabled when MLFLOW_TRACKING_URI is absent."""
from unittest.mock import MagicMock, patch

from src.services.experiment_tracker import ExperimentTracker


def _sector_result():
    r = MagicMock()
    r.profile_id = "ml_engineer"
    r.sector = "Informatique & Digital"
    r.confidence = 0.85
    return r


def _quality_report(score: int = 72):
    r = MagicMock()
    r.profile_strength.score = score
    return r


def test_disabled_when_no_tracking_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    tracker = ExperimentTracker()
    assert not tracker._enabled


def test_log_cv_analysis_is_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    tracker = ExperimentTracker()
    tracker.log_cv_analysis(_sector_result(), _quality_report(), "pdfplumber", 250.0, 300)


def test_log_cv_analysis_calls_mlflow(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    with patch("src.services.experiment_tracker.mlflow") as mock_mlflow:
        tracker = ExperimentTracker()
        tracker.log_cv_analysis(_sector_result(), _quality_report(), "pdfplumber", 250.0, 300)
        mock_mlflow.log_params.assert_called_once()
        mock_mlflow.log_metrics.assert_called_once()


def test_log_benchmark_calls_mlflow(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    with patch("src.services.experiment_tracker.mlflow") as mock_mlflow:
        tracker = ExperimentTracker()
        tracker.log_benchmark(
            "/tmp/report.csv",
            params={"allow_vision_llm": "False"},
            metrics={"cv_count": 15.0, "avg_time_ms": 320.5},
        )
        mock_mlflow.log_params.assert_called_once_with({"allow_vision_llm": "False"})
        mock_mlflow.log_metrics.assert_called_once()
        mock_mlflow.log_artifact.assert_called_once_with("/tmp/report.csv")
