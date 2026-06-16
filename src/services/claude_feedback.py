from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import anthropic

from src.core.budget_guard import budget_guard
from src.core.config import settings
from src.core.schemas import CriterionResult, NormalizedCV, ScoringResult

if TYPE_CHECKING:
    from src.services.sector_detector import SectorDetectionResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Tu es un expert en recrutement et optimisation de CV pour le marché français.
Tu fournis des conseils précis, actionnables et bienveillants.
Réponds toujours en français, quel que soit la langue du CV ou de l'offre.
Sois direct et spécifique. Formate ta réponse en markdown."""


def _build_sector_block(
    sector_result: "SectorDetectionResult",
    criteria_results: list[CriterionResult],
) -> str:
    lines = [
        "## Profil détecté par l'ATS",
        f"- **Intitulé** : {sector_result.job_title}",
        f"- **Secteur** : {sector_result.sector}",
        f"- **Confiance** : {sector_result.confidence:.0%}",
        "",
    ]
    ko_req = [r for r in criteria_results if r.score == 0 and r.required]
    ko_opt = [r for r in criteria_results if r.score == 0 and not r.required]

    if ko_req:
        lines.append("## Critères obligatoires non satisfaits (priorité absolue)")
        for r in ko_req:
            ev = r.evidence[0] if r.evidence else "—"
            lines.append(f"- **{r.label}** ({r.weight} pts) : {ev}")
        lines.append("")

    if ko_opt:
        lines.append("## Critères recommandés non satisfaits")
        for r in ko_opt:
            lines.append(f"- {r.label} ({r.weight} pts)")
        lines.append("")

    return "\n".join(lines)


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
        parsed_cv: NormalizedCV,
        job_description: str,
        scoring_result: ScoringResult,
        sector_result: Optional["SectorDetectionResult"] = None,
        criteria_results: Optional[list[CriterionResult]] = None,
    ) -> str:
        if not budget_guard.check_and_increment():
            raise ClaudeBudgetExceeded(
                f"Claude calls limit reached ({budget_guard.limit})."
            )

        if sector_result is not None:
            sector_block = _build_sector_block(sector_result, criteria_results or [])
            user_content = f"""{sector_block}
## Offre d'emploi
{job_description[:2000]}

## Compétences détectées
{", ".join(parsed_cv.skills_flat[:25]) or "aucune"}

## Score de compatibilité : {scoring_result.overall_score:.0f}/100

## Extrait du CV
{parsed_cv.raw_text[:2000]}

---
Génère un feedback structuré en 3 parties :

**1. POINTS FORTS** (2-3 éléments positifs spécifiques au profil {sector_result.job_title})

**2. POINTS À AMÉLIORER** (basé sur les critères obligatoires non satisfaits — sois précis et actionnable, pas générique)

**3. CONSEIL PRIORITAIRE** (une action concrète à faire MAINTENANT pour ce poste spécifique)

Ton : professionnel, bienveillant, direct. Maximum 250 mots. En français."""
        else:
            user_content = f"""## Job Description
{job_description[:2000]}

## CV Analysis
**Detected skills:** {", ".join(parsed_cv.skills_flat[:25]) or "none"}
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
