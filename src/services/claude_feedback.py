import logging

import anthropic

from src.core.budget_guard import budget_guard
from src.core.config import settings
from src.core.schemas import ParsedCV, ScoringResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert ATS (Applicant Tracking System) analyst and career coach.
Evaluate CVs against job descriptions and provide concise, actionable feedback.
Focus on: skill alignment, experience relevance, missing ATS keywords, and structural improvements.
Always respond in French, regardless of the language of the CV or job description.
Be direct and specific. Format your response in markdown."""


class ClaudeBudgetExceeded(Exception):
    """Levée quand le quota global de budget_guard a été atteint."""


class ClaudeServiceError(Exception):
    """Levée quand l'appel à l'API Claude échoue (rate limit, erreur serveur, réseau).

    Le message porté par cette exception est destiné à l'utilisateur final
    (français, non technique) ; le détail technique est loggé séparément.
    """


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
        if not budget_guard.check_and_increment():
            raise ClaudeBudgetExceeded(
                f"Claude calls limit reached ({budget_guard.limit})."
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
        except anthropic.RateLimitError as exc:
            budget_guard.release()
            logger.warning("claude_feedback | rate limit Claude : %s", exc)
            raise ClaudeServiceError(
                "Le service IA est temporairement surchargé. Réessayez dans quelques instants."
            ) from exc
        except anthropic.APIConnectionError as exc:
            budget_guard.release()
            logger.warning("claude_feedback | erreur de connexion Claude : %s", exc)
            raise ClaudeServiceError(
                "Impossible de contacter le service IA. Vérifiez votre connexion et réessayez."
            ) from exc
        except anthropic.APIStatusError as exc:
            budget_guard.release()
            logger.warning("claude_feedback | erreur API Claude (%s) : %s", exc.status_code, exc)
            raise ClaudeServiceError(
                "Le service IA a rencontré une erreur. Réessayez plus tard."
            ) from exc
        except Exception:
            budget_guard.release()
            raise
        return response.content[0].text
