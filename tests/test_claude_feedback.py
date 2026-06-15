import threading
from unittest.mock import MagicMock, patch

import pytest

from src.core.schemas import ParsedCV, ScoreBreakdown, ScoringResult
from src.services import claude_feedback
from src.services.claude_feedback import ClaudeBudgetExceeded, ClaudeFeedback


@pytest.fixture(autouse=True)
def _reset_calls_count(tmp_path, monkeypatch):
    """Isole le compteur de quota (fichier + état module) entre chaque test."""
    monkeypatch.setattr(claude_feedback, "QUOTA_FILE_PATH", tmp_path / "claude_quota.json")
    claude_feedback.CLAUDE_CALLS_COUNT = 0
    yield


def _make_parsed_cv() -> ParsedCV:
    return ParsedCV(
        raw_text="Some CV text",
        sections={"experience": "..."},
        skills=["python", "docker"],
        experience_years=3.0,
    )


def _make_scoring_result() -> ScoringResult:
    return ScoringResult(
        overall_score=72.5,
        breakdown=ScoreBreakdown(
            semantic_similarity=70.0,
            keyword_match=65.0,
            structure_completeness=80.0,
        ),
        matched_keywords=["python"],
        missing_keywords=["kubernetes"],
    )


def _mock_anthropic_response(text: str = "feedback en français") -> MagicMock:
    response = MagicMock()
    response.content = [MagicMock(text=text)]
    return response


class TestClaudeFeedbackInit:
    def test_init_raises_without_api_key(self):
        with patch("src.services.claude_feedback.settings") as mock_settings:
            mock_settings.anthropic_api_key = ""
            with pytest.raises(ValueError):
                ClaudeFeedback()


class TestClaudeFeedbackBudget:
    def test_generate_feedback_increments_counter(self):
        with patch(
            "src.services.claude_feedback.anthropic.Anthropic"
        ) as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = _mock_anthropic_response()
            mock_anthropic.return_value = mock_client

            cf = ClaudeFeedback(api_key="sk-test")
            feedback = cf.generate_feedback(
                _make_parsed_cv(), "Job description", _make_scoring_result()
            )

        assert feedback == "feedback en français"
        assert claude_feedback.CLAUDE_CALLS_COUNT == 1

    def test_generate_feedback_raises_when_budget_exceeded(self):
        claude_feedback.CLAUDE_CALLS_COUNT = claude_feedback.CLAUDE_CALLS_LIMIT

        with patch(
            "src.services.claude_feedback.anthropic.Anthropic"
        ) as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            cf = ClaudeFeedback(api_key="sk-test")
            with pytest.raises(ClaudeBudgetExceeded):
                cf.generate_feedback(
                    _make_parsed_cv(), "Job description", _make_scoring_result()
                )

        mock_client.messages.create.assert_not_called()
        assert claude_feedback.CLAUDE_CALLS_COUNT == claude_feedback.CLAUDE_CALLS_LIMIT

    def test_generate_feedback_rolls_back_counter_on_api_error(self):
        with patch(
            "src.services.claude_feedback.anthropic.Anthropic"
        ) as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = RuntimeError("boom")
            mock_anthropic.return_value = mock_client

            cf = ClaudeFeedback(api_key="sk-test")
            with pytest.raises(RuntimeError):
                cf.generate_feedback(
                    _make_parsed_cv(), "Job description", _make_scoring_result()
                )

        assert claude_feedback.CLAUDE_CALLS_COUNT == 0

    def test_concurrent_calls_do_not_exceed_quota(self):
        claude_feedback.CLAUDE_CALLS_COUNT = claude_feedback.CLAUDE_CALLS_LIMIT - 1
        claude_feedback._save_calls_count(claude_feedback.CLAUDE_CALLS_COUNT)

        results: list[bool] = []

        def _attempt():
            results.append(claude_feedback._reserve_call())

        threads = [threading.Thread(target=_attempt) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results.count(True) == 1
        assert results.count(False) == 9
        assert claude_feedback.CLAUDE_CALLS_COUNT == claude_feedback.CLAUDE_CALLS_LIMIT
