from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.server import app
from src.core.schemas import JobListing

FIXTURE_PDF = Path(__file__).parent / "fixtures" / "sample_cv_ml_engineer_junior.pdf"


def _fake_search(query, location=None, distance=None, max_results=20):
    return [
        JobListing(
            title="ML Engineer",
            company="Acme",
            location="Lille",
            description="Build and deploy machine learning models.",
            url="https://example.com/job/ml-engineer",
        )
    ]


def test_find_jobs_returns_ranked_job_match_list():
    client = TestClient(app)

    with patch("src.api.server._job_search_service.search", side_effect=_fake_search):
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
