import json
import os
import threading
from pathlib import Path

import anthropic

from src.core.config import settings
from src.core.schemas import ParsedCV, ScoringResult

_SYSTEM_PROMPT = """You are an expert ATS (Applicant Tracking System) analyst and career coach.
Evaluate CVs against job descriptions and provide concise, actionable feedback.
Focus on: skill alignment, experience relevance, missing ATS keywords, and structural improvements.
Always respond in French, regardless of the language of the CV or job description.
Be direct and specific. Format your response in markdown."""

# Budget global (process-wide) d'appels Claude — protège la clé API limitée à
# $10 sur la démo HF Spaces. ~300 appels feedback (~$0.025/appel) ≈ $7.50,
# laissant une marge de sécurité pour les appels Vision LLM (cascade niveau 3).
CLAUDE_CALLS_LIMIT = 300

# Compteur persisté hors-process (survit aux rechargements de l'app dans le
# même conteneur HF Spaces) et protégé par un verrou contre les accès
# concurrents (UI Gradio + API FastAPI multithreads).
QUOTA_FILE_PATH = Path(os.environ.get("CLAUDE_QUOTA_FILE", "/tmp/claude_quota.json"))
_QUOTA_LOCK = threading.Lock()


def _load_calls_count() -> int:
    try:
        return int(json.loads(QUOTA_FILE_PATH.read_text()).get("count", 0))
    except (OSError, ValueError, json.JSONDecodeError):
        return 0


def _save_calls_count(count: int) -> None:
    try:
        QUOTA_FILE_PATH.write_text(json.dumps({"count": count}))
    except OSError:
        pass


CLAUDE_CALLS_COUNT = _load_calls_count()


def _reserve_call() -> bool:
    """Atomically check the quota and reserve one call slot. Thread-safe."""
    global CLAUDE_CALLS_COUNT
    with _QUOTA_LOCK:
        if CLAUDE_CALLS_COUNT >= CLAUDE_CALLS_LIMIT:
            return False
        CLAUDE_CALLS_COUNT += 1
        _save_calls_count(CLAUDE_CALLS_COUNT)
        return True


def _release_call() -> None:
    """Roll back a reservation when the underlying API call failed."""
    global CLAUDE_CALLS_COUNT
    with _QUOTA_LOCK:
        CLAUDE_CALLS_COUNT = max(0, CLAUDE_CALLS_COUNT - 1)
        _save_calls_count(CLAUDE_CALLS_COUNT)


class ClaudeBudgetExceeded(Exception):
    """Levée quand CLAUDE_CALLS_LIMIT a été atteint pour ce processus."""


class ClaudeFeedback:
    def __init__(self, api_key: str = "") -> None:
        key = api_key or settings.anthropic_api_key
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is required for AI feedback.")
        self.client = anthropic.Anthropic(api_key=key)

    def generate_feedback(
        self,
        parsed_cv: ParsedCV,
        job_description: str,
        scoring_result: ScoringResult,
    ) -> str:
        if not _reserve_call():
            raise ClaudeBudgetExceeded(
                f"Claude calls limit reached ({CLAUDE_CALLS_LIMIT})."
            )

        user_content = f"""## Job Description
{job_description[:2000]}

## CV Analysis
**Detected skills:** {", ".join(parsed_cv.skills[:25]) or "none"}
**Sections found:** {", ".join(parsed_cv.sections.keys())}
**Years of experience:** {parsed_cv.experience_years or "undetermined"}

## Current ATS Scores
| Metric | Score |
|--------|-------|
| Overall | {scoring_result.overall_score}/100 |
| Keyword match | {scoring_result.breakdown.keyword_match}/100 |
| Semantic similarity | {scoring_result.breakdown.semantic_similarity}/100 |
| CV structure | {scoring_result.breakdown.structure_completeness}/100 |

## Missing Keywords
{", ".join(scoring_result.missing_keywords[:15]) or "none"}

## CV Text (excerpt)
{parsed_cv.raw_text[:3000]}

---
Provide:
1. A 2-3 sentence fit assessment for this role
2. Top 5 specific improvements ranked by ATS impact
3. Critical keywords to incorporate
4. One structural improvement suggestion"""

        try:
            response = self.client.messages.create(
                model=settings.claude_model,
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_content}],
            )
        except Exception:
            _release_call()
            raise
        return response.content[0].text
