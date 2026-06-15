from unittest.mock import MagicMock, patch

import pytest

from src.core.budget_guard import BudgetGuard
from src.core.schemas import ParsedCV, ScoreBreakdown, ScoringResult
from src.services import claude_feedback
from src.services.claude_feedback import ClaudeBudgetExceeded, ClaudeFeedback


@pytest.fixture(autouse=True)
def _isolated_budget_guard(tmp_path, monkeypatch):
    """Isole le BudgetGuard partagé (fichier + état) entre chaque test."""
    guard = BudgetGuard(limit=300, path=tmp_path / "claude_quota.json")
    monkeypatch.setattr(claude_feedback, "budget_guard", guard)
    yield guard


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
    def test_generate_feedback_increments_counter(self, _isolated_budget_guard):
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
        assert _isolated_budget_guard.get_remaining() == 299

    def test_generate_feedback_raises_when_budget_exceeded(self, _isolated_budget_guard):
        for _ in range(_isolated_budget_guard.limit):
            assert _isolated_budget_guard.check_and_increment()

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
        assert _isolated_budget_guard.get_remaining() == 0

    def test_generate_feedback_rolls_back_counter_on_api_error(self, _isolated_budget_guard):
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

        assert _isolated_budget_guard.get_remaining() == _isolated_budget_guard.limit
