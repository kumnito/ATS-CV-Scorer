from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import httpx
import pytest

from src.core.budget_guard import BudgetGuard
from src.core.schemas import CriterionResult, NormalizedCV, ScoreBreakdown, ScoringResult
from src.services import claude_feedback
from src.services.claude_feedback import ClaudeBudgetExceeded, ClaudeFeedback, ClaudeServiceError


@pytest.fixture(autouse=True)
def _isolated_budget_guard(tmp_path, monkeypatch):
    """Isole le BudgetGuard partagé (fichier + état) entre chaque test."""
    guard = BudgetGuard(limit=300, path=tmp_path / "claude_quota.json")
    monkeypatch.setattr(claude_feedback, "budget_guard", guard)
    yield guard


def _make_parsed_cv() -> NormalizedCV:
    return NormalizedCV(
        raw_text="Some CV text",
        sections={"experience": "..."},
        skills_flat=["python", "docker"],
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


def _make_sector_result() -> SimpleNamespace:
    return SimpleNamespace(
        profile_id="devops",
        job_title="DevOps Engineer",
        sector="Informatique & Digital",
        confidence=0.70,
        alternatives=[],
    )


def _make_criteria_results() -> list[CriterionResult]:
    return [
        CriterionResult(
            criterion_id="eval_formation",
            label="Formation informatique",
            weight=15,
            required=True,
            score=0,
            evidence=["Aucune formation détectée"],
            weighted_score=0.0,
        ),
        CriterionResult(
            criterion_id="eval_experience",
            label="Expérience professionnelle",
            weight=20,
            required=True,
            score=100,
            evidence=["2 expérience(s) détectée(s)"],
            weighted_score=20.0,
        ),
        CriterionResult(
            criterion_id="eval_projects",
            label="Projets documentés",
            weight=15,
            required=False,
            score=0,
            evidence=["Aucun projet documenté"],
            weighted_score=0.0,
        ),
    ]


class TestClaudeFeedbackSectoriel:
    def _run_feedback(self, mock_client, sector_result=None, criteria_results=None):
        cf = ClaudeFeedback(api_key="sk-test")
        return cf.generate_feedback(
            _make_parsed_cv(),
            "Job description en anglais",
            _make_scoring_result(),
            sector_result=sector_result,
            criteria_results=criteria_results,
        )

    def test_with_sector_result_prompt_contains_job_title(self, _isolated_budget_guard):
        with patch("src.services.claude_feedback.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = _mock_anthropic_response("feedback sectoriel")
            mock_anthropic.return_value = mock_client

            self._run_feedback(mock_client, sector_result=_make_sector_result())

        call_kwargs = mock_client.messages.create.call_args.kwargs
        user_msg = call_kwargs["messages"][0]["content"]
        assert "DevOps Engineer" in user_msg

    def test_with_sector_result_prompt_lists_required_ko_criteria(self, _isolated_budget_guard):
        with patch("src.services.claude_feedback.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = _mock_anthropic_response("feedback sectoriel")
            mock_anthropic.return_value = mock_client

            self._run_feedback(
                mock_client,
                sector_result=_make_sector_result(),
                criteria_results=_make_criteria_results(),
            )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        user_msg = call_kwargs["messages"][0]["content"]
        # Critère required KO doit apparaître
        assert "Formation informatique" in user_msg
        # Critère required OK ne doit pas apparaître dans la section KO
        assert "Expérience professionnelle" not in user_msg
        # Critère optional KO doit apparaître (section recommandés)
        assert "Projets documentés" in user_msg

    def test_without_sector_result_uses_generic_prompt(self, _isolated_budget_guard):
        with patch("src.services.claude_feedback.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = _mock_anthropic_response("feedback générique")
            mock_anthropic.return_value = mock_client

            self._run_feedback(mock_client, sector_result=None)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        user_msg = call_kwargs["messages"][0]["content"]
        # Le prompt générique n'a pas la structure en 3 parties sectorielles
        assert "POINTS FORTS" not in user_msg
        # Mais il contient les éléments du prompt original
        assert "Job Description" in user_msg


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

    def test_generate_feedback_raises_service_error_on_rate_limit(self, _isolated_budget_guard):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(429, request=request)
        rate_limit_error = anthropic.RateLimitError("rate limited", response=response, body=None)

        with patch(
            "src.services.claude_feedback.anthropic.Anthropic"
        ) as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = rate_limit_error
            mock_anthropic.return_value = mock_client

            cf = ClaudeFeedback(api_key="sk-test")
            with pytest.raises(ClaudeServiceError, match="surchargé"):
                cf.generate_feedback(
                    _make_parsed_cv(), "Job description", _make_scoring_result()
                )

        assert _isolated_budget_guard.get_remaining() == _isolated_budget_guard.limit

    def test_generate_feedback_raises_service_error_on_connection_error(self, _isolated_budget_guard):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        connection_error = anthropic.APIConnectionError(request=request)

        with patch(
            "src.services.claude_feedback.anthropic.Anthropic"
        ) as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = connection_error
            mock_anthropic.return_value = mock_client

            cf = ClaudeFeedback(api_key="sk-test")
            with pytest.raises(ClaudeServiceError, match="contacter"):
                cf.generate_feedback(
                    _make_parsed_cv(), "Job description", _make_scoring_result()
                )

        assert _isolated_budget_guard.get_remaining() == _isolated_budget_guard.limit

    def test_generate_feedback_raises_service_error_on_api_status_error(self, _isolated_budget_guard):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(500, request=request)
        status_error = anthropic.APIStatusError("server error", response=response, body=None)

        with patch(
            "src.services.claude_feedback.anthropic.Anthropic"
        ) as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = status_error
            mock_anthropic.return_value = mock_client

            cf = ClaudeFeedback(api_key="sk-test")
            with pytest.raises(ClaudeServiceError, match="erreur"):
                cf.generate_feedback(
                    _make_parsed_cv(), "Job description", _make_scoring_result()
                )

        assert _isolated_budget_guard.get_remaining() == _isolated_budget_guard.limit

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
