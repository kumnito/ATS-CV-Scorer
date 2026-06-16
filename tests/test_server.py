from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.server import app
from src.core.schemas import JobListing
from src.services.sector_detector import SectorDetectionResult

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "sample_cv_ml_engineer_junior.pdf"


def _fake_orchestrator_search(
    query, location=None, distance=None, max_results=20, active_providers=None
):
    return [
        JobListing(
            title="ML Engineer",
            company="Acme",
            location="Lille",
            description="Build and deploy machine learning models.",
            url="https://example.com/job/ml-engineer",
        )
    ]


def _fake_sector_result() -> SectorDetectionResult:
    return SectorDetectionResult(
        profile_id="ml_engineer",
        sector="Informatique & Digital",
        job_title="Machine Learning Engineer",
        confidence=0.72,
        alternatives=[],
    )


def test_health_returns_budget_remaining():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "claude_budget_remaining" in data
    assert isinstance(data["claude_budget_remaining"], int)


def test_find_jobs_returns_ranked_job_match_list():
    client = TestClient(app)

    with patch("src.api.server._orchestrator.search", side_effect=_fake_orchestrator_search):
        with open(FIXTURE_PDF, "rb") as fh:
            response = client.post(
                "/find-jobs",
                files={"cv_file": ("cv.pdf", fh, "application/pdf")},
                data={"max_results": "5"},
            )

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) >= 1
    for match in payload:
        assert "job" in match
        assert "scoring_result" in match
        assert match["job"]["title"]
        assert "overall_score" in match["scoring_result"]


def test_score_returns_detected_sector():
    client = TestClient(app)

    fake_sector = _fake_sector_result()
    mock_quality = MagicMock()
    mock_quality.criteria_results = []

    with (
        patch("src.api.server._sector_detector.detect", return_value=fake_sector),
        patch("src.api.server._cv_quality_scorer.score", return_value=mock_quality),
    ):
        with open(FIXTURE_PDF, "rb") as fh:
            response = client.post(
                "/score",
                files={"cv_file": ("cv.pdf", fh, "application/pdf")},
                data={"job_description": "Machine learning engineer role", "include_feedback": "false"},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["detected_sector"] == "Informatique & Digital"
    assert data["detected_profile"] == "ml_engineer"
    assert data["detection_confidence"] == 0.72
    assert "criteria_results" in data


def test_find_jobs_uses_orchestrator_not_legacy_service():
    """Vérifie que /find-jobs passe par _orchestrator (multi-sources) et non job_search legacy."""
    client = TestClient(app)
    call_count = {"n": 0}

    def _tracking_search(query, location=None, distance=None, max_results=20, active_providers=None):
        call_count["n"] += 1
        return [
            JobListing(
                title="DevOps Engineer",
                company="Tech Corp",
                location="Paris",
                description="CI/CD pipeline management.",
                url="https://example.com/job/devops",
                source="jooble",
                source_color="#10b981",
            )
        ]

    with patch("src.api.server._orchestrator.search", side_effect=_tracking_search):
        with open(FIXTURE_PDF, "rb") as fh:
            response = client.post(
                "/find-jobs",
                files={"cv_file": ("cv.pdf", fh, "application/pdf")},
                data={"max_results": "5"},
            )

    assert response.status_code == 200
    assert call_count["n"] >= 1
    payload = response.json()
    assert len(payload) >= 1
