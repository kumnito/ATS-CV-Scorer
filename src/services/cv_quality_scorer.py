"""CV quality scorer: produces a CVQualityReport from a NormalizedCV."""

import logging
import re
from datetime import date
from typing import TYPE_CHECKING, Optional

from src.core.lexicons import ACTION_VERBS_EN, ACTION_VERBS_FR, METRIC_PATTERNS, SKILL_CATEGORIES
from src.core.schemas import (
    ATSReadability,
    CVQualityReport,
    NormalizedCV,
    ProfileStrength,
    Recommendation,
)

if TYPE_CHECKING:
    from src.services.sector_detector import SectorDetectionResult

logger = logging.getLogger(__name__)

_REQUIRED_SECTIONS = ["experience", "education", "skills"]

_ACTION_VERBS = ACTION_VERBS_EN | {v.rstrip("é") for v in ACTION_VERBS_FR} | ACTION_VERBS_FR

# French action nouns accepted as equivalent signals (e.g. "Ingestion de données", "Contrôle qualité")
_ACTION_NOUNS_FR = frozenset({
    "ingestion", "contrôle", "automatisation", "conception", "développement",
    "déploiement", "intégration", "optimisation", "coordination", "gestion",
    "supervision", "analyse", "création", "construction", "pilotage",
    "maintenance", "accompagnement", "migration", "consolidation",
    "paramétrage", "configuration", "modélisation", "suivi",
    "traitement", "extraction", "transformation",
})


class CVQualityScorer:
    def score(
        self,
        cv: NormalizedCV,
        sector_result: Optional["SectorDetectionResult"] = None,
    ) -> CVQualityReport:
        detected = _detected_sections(cv)
        missing = [s for s in _REQUIRED_SECTIONS if s not in detected]
        career = _career_stats(cv)
        gaps = _detect_career_gaps(cv)

        total_skills = len(cv.skills.flat())
        has_metrics = _has_metrics(cv)
        has_verbs = _has_action_signals(cv)
        kd = _keyword_density(cv)
        has_dates = _has_sufficient_dates(cv)

        ats = _compute_ats_readability(cv, detected, missing)
        strength = _compute_profile_strength(cv, career, total_skills, has_metrics, has_verbs, has_dates)
        recs = _build_recommendations(cv, detected, missing, has_metrics, has_verbs, total_skills, has_dates, kd)

        logger.info("=== CVQualityScorer.score() ===")
        logger.info(
            "word_count: %d | skills: %d | experience: %d",
            cv.word_count, total_skills, len(cv.experience),
        )
        logger.info(
            "summary: %s | has_metrics: %s | has_verbs: %s | has_projects: %s | has_dates: %s",
            bool(cv.summary), has_metrics, has_verbs, bool(cv.projects), has_dates,
        )
        logger.info("profile_strength: %d/100 (%s)", strength.score, strength.level)
        logger.info("is_machine_readable: %s | kd: %.3f", ats.is_machine_readable, kd)

        criteria_results = []
        if sector_result is not None:
            from src.services.criteria_evaluator import evaluate_criteria
            criteria_results = evaluate_criteria(cv, sector_result)

        return CVQualityReport(
            ats_readability=ats,
            profile_strength=strength,
            recommendations=recs,
            total_experience_years=career["total_experience_years"],
            career_start_year=career["career_start_year"],
            most_recent_role=career["most_recent_role"],
            education_years=career["education_years"],
            career_gaps=gaps,
            extraction_method=cv.extraction_method,
            extraction_confidence=cv.extraction_confidence,
            detected_sector=sector_result.sector if sector_result else None,
            detected_profile=sector_result.profile_id if sector_result else None,
            detection_confidence=sector_result.confidence if sector_result else 0.0,
            criteria_results=criteria_results,
        )


# ---------------------------------------------------------------------------
# Axe 1 — Lisibilité ATS
# ---------------------------------------------------------------------------


def _detected_sections(cv: NormalizedCV) -> list[str]:
    detected = []
    if cv.summary:
        detected.append("summary")
    if cv.experience:
        detected.append("experience")
    if cv.education:
        detected.append("education")
    if cv.skills.flat():
        detected.append("skills")
    if cv.projects:
        detected.append("projects")
    if cv.languages:
        detected.append("languages")
    return detected


def _compute_ats_readability(
    cv: NormalizedCV,
    detected: list[str],
    missing: list[str],
) -> ATSReadability:
    layout = cv.layout_detected
    layout_label = "✅ Optimal" if layout == "single_column" else "⚠️ Risque parseur"
    return ATSReadability(
        layout=layout,
        layout_label=layout_label,
        sections_found=detected,
        sections_missing=missing,
        extraction_method=cv.extraction_method,
        is_machine_readable=cv.word_count >= 150,
    )


# ---------------------------------------------------------------------------
# Axe 2 — Solidité du profil (recalibré FR)
# ---------------------------------------------------------------------------


def _has_sufficient_dates(cv: NormalizedCV) -> bool:
    if not cv.experience:
        return False
    dated = sum(1 for e in cv.experience if e.date_start)
    return dated / len(cv.experience) >= 0.5


def _compute_profile_strength(
    cv: NormalizedCV,
    career: dict,
    total_skills: int,
    has_metrics: bool,
    has_verbs: bool,
    has_dates: bool,
) -> ProfileStrength:
    pts_wc = 15 if cv.word_count >= 300 else 0
    pts_skills = 20 if total_skills >= 10 else 0
    pts_exp = 15 if len(cv.experience) >= 1 else 0
    pts_summary = 10 if cv.summary else 0
    pts_dates = 10 if has_dates else 0
    pts_verbs = 10 if has_verbs else 0
    pts_metrics = 10 if has_metrics else 0
    pts_projects = 10 if cv.projects else 0

    score = min(100, pts_wc + pts_skills + pts_exp + pts_summary + pts_dates + pts_verbs + pts_metrics + pts_projects)
    level = "Solide" if score >= 75 else ("Correct" if score >= 50 else "À renforcer")

    strengths: list[str] = []
    if total_skills >= 10:
        strengths.append(f"Compétences techniques riches ({total_skills} skills détectés)")
    if len(cv.experience) >= 1:
        role_str = career.get("most_recent_role") or ""
        years = career.get("total_experience_years", 0)
        if years > 0 and role_str:
            strengths.append(f"Expérience réelle ({role_str}, {years:.0f} an(s))")
        elif years > 0:
            strengths.append(f"Expérience réelle ({years:.0f} an(s))")
        else:
            strengths.append("Expériences professionnelles présentes")
    if cv.summary:
        strengths.append("Accroche/profil présent")
    if has_metrics:
        strengths.append("Métriques chiffrées dans les expériences")
    if cv.projects:
        n = len(cv.projects)
        strengths.append(f"{n} projet(s) documenté(s)")
    if cv.word_count >= 300:
        strengths.append(f"CV bien développé ({cv.word_count} mots)")

    improvements: list[str] = []
    if total_skills < 10:
        improvements.append(f"Enrichir les compétences techniques ({total_skills} détectées, cible ≥ 10)")
    if len(cv.experience) < 1:
        improvements.append("Ajouter une expérience professionnelle")
    if not cv.summary:
        improvements.append("Ajouter une accroche/profil en tête de CV")
    if not has_metrics:
        improvements.append("Quantifier les résultats dans les expériences (%, chiffres, volumes)")
    if not cv.projects:
        improvements.append("Créer une section Projets avec stack et résultats")
    if cv.word_count < 300:
        improvements.append(f"Développer le contenu ({cv.word_count} mots détectés, cible ≥ 300)")

    return ProfileStrength(level=level, score=score, strengths=strengths, improvements=improvements)


# ---------------------------------------------------------------------------
# Recommandations (ordonnées par priorité décroissante)
# ---------------------------------------------------------------------------


def _build_recommendations(
    cv: NormalizedCV,
    detected: list[str],
    missing: list[str],
    has_metrics: bool,
    has_verbs: bool,
    total_skills: int,
    has_dates: bool,
    kd: float,
) -> list[Recommendation]:
    recs: list[Recommendation] = []

    # Priorité 1 — Fort (bloquant pour le parseur ATS)
    if cv.layout_detected == "two_columns":
        recs.append(Recommendation(
            priority=1, impact="Fort",
            action="Convertir en 1 colonne",
            why="Les parseurs ATS ignorent souvent la colonne droite — données perdues",
        ))
    _section_labels = {
        "experience": "Expérience professionnelle",
        "skills": "Compétences",
        "education": "Formation",
    }
    for s in missing:
        label = _section_labels.get(s, s.capitalize())
        recs.append(Recommendation(
            priority=1, impact="Fort",
            action=f"Ajouter une section « {label} »",
            why="Section obligatoire pour passer les filtres ATS automatiques",
        ))

    # Priorité 2 — Moyen (impact sur le score de matching)
    if not has_metrics and cv.experience:
        recs.append(Recommendation(
            priority=2, impact="Moyen",
            action="Ajouter des métriques chiffrées dans les expériences",
            why="Différencie les profils à compétences équivalentes (%, volumes, délais)",
        ))
    if not cv.projects and cv.experience:
        recs.append(Recommendation(
            priority=2, impact="Moyen",
            action="Créer une section Projets avec stack et résultats",
            why="Signal fort de pratique autonome, très valorisé en tech",
        ))
    if not cv.summary:
        recs.append(Recommendation(
            priority=2, impact="Moyen",
            action="Ajouter une accroche/profil en début de CV",
            why="Aide les recruteurs à positionner rapidement votre profil",
        ))
    if not has_dates and cv.experience:
        recs.append(Recommendation(
            priority=2, impact="Moyen",
            action="Compléter les dates de début/fin dans les expériences",
            why="Les ATS utilisent les dates pour calculer les années d'expérience",
        ))

    # Priorité 3 — Faible (polissage)
    if not has_verbs:
        recs.append(Recommendation(
            priority=3, impact="Faible",
            action="Débuter les bullets par des verbes ou noms d'action",
            why="Style attendu par les recruteurs (réalisé, développé, optimisé…)",
        ))
    if total_skills < 10:
        recs.append(Recommendation(
            priority=3, impact="Faible",
            action=f"Enrichir les compétences ({total_skills} détectées, cible ≥ 10)",
            why="Plus de mots-clés techniques = meilleur taux de correspondance ATS",
        ))
    if cv.word_count < 300:
        recs.append(Recommendation(
            priority=3, impact="Faible",
            action=f"Développer le contenu du CV ({cv.word_count} mots, cible ≥ 300)",
            why="CV trop court — enrichir avec contexte et bullets détaillés",
        ))
    if kd > 0.35:
        recs.append(Recommendation(
            priority=3, impact="Faible",
            action="Réduire la répétition de mots-clés techniques",
            why=f"Densité actuelle {kd:.0%} — risque de sur-optimisation détectée par certains ATS",
        ))

    return recs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_metrics(cv: NormalizedCV) -> bool:
    full_text = " ".join(
        b for e in cv.experience for b in e.bullets
    ) + " ".join(
        b for p in cv.projects for b in p.metrics
    )
    return any(p.search(full_text) for p in METRIC_PATTERNS)


def _keyword_density(cv: NormalizedCV) -> float:
    if cv.word_count == 0:
        return 0.0
    text_lower = cv.raw_text.lower()
    count = 0
    for skill_list in SKILL_CATEGORIES.values():
        for skill in skill_list:
            count += len(re.findall(r"\b" + re.escape(skill.lower()) + r"\b", text_lower))
    return count / cv.word_count


def _has_action_signals(cv: NormalizedCV) -> bool:
    bullets_text = " ".join(b for e in cv.experience for b in e.bullets).lower()
    if not bullets_text:
        bullets_text = cv.raw_text.lower()
    words = set(re.findall(r"\b\w+\b", bullets_text))
    return bool(words & ({v.lower() for v in _ACTION_VERBS} | _ACTION_NOUNS_FR))


# ---------------------------------------------------------------------------
# Career statistics
# ---------------------------------------------------------------------------


def _parse_ym(date_str: Optional[str]) -> Optional[tuple[int, int]]:
    if not date_str or date_str == "present":
        return None
    try:
        if "/" in date_str:
            month, year = date_str.split("/")
            return (int(year), int(month))
        parts = date_str.split("-")
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 1)
    except ValueError:
        return None


def _career_stats(cv: NormalizedCV) -> dict:
    total_months = sum(e.duration_months or 0 for e in cv.experience)
    total_years = round(total_months / 12, 1)

    start_years = []
    for e in cv.experience:
        parsed = _parse_ym(e.date_start)
        if parsed:
            start_years.append(parsed[0])
    career_start = min(start_years) if start_years else None

    most_recent = None
    if cv.experience:
        first = cv.experience[0]
        parts = [p for p in [first.title, first.company] if p]
        if parts:
            most_recent = " @ ".join(parts)

    edu_months = sum(e.duration_months or 0 for e in cv.education)
    edu_years = round(edu_months / 12)

    return {
        "total_experience_years": total_years,
        "career_start_year": career_start,
        "most_recent_role": most_recent,
        "education_years": edu_years,
    }


def _detect_career_gaps(cv: NormalizedCV) -> list[str]:
    """Return descriptions of gaps > 12 months in the combined experience+education timeline."""
    today = date.today()

    def to_abs(year: int, month: int) -> int:
        return year * 12 + month

    intervals: list[tuple[int, int]] = []
    for e in cv.experience:
        if not e.date_start:
            continue
        s = _parse_ym(e.date_start)
        if s is None:
            continue
        if e.is_current or not e.date_end or e.date_end == "present":
            end = (today.year, today.month)
        else:
            end = _parse_ym(e.date_end)
        if end is None:
            continue
        intervals.append((to_abs(*s), to_abs(*end)))

    for e in cv.education:
        if not e.date_start:
            continue
        s = _parse_ym(e.date_start)
        if s is None:
            continue
        if e.is_current or not e.date_end or e.date_end == "present":
            end = (today.year, today.month)
        else:
            end = _parse_ym(e.date_end)
        if end is None:
            continue
        intervals.append((to_abs(*s), to_abs(*end)))

    if len(intervals) < 2:
        return []

    intervals.sort(key=lambda x: x[0])
    gaps: list[str] = []
    for i in range(len(intervals) - 1):
        gap_start_abs = intervals[i][1]
        gap_end_abs = intervals[i + 1][0]
        gap_months = gap_end_abs - gap_start_abs
        if gap_months > 12:
            gap_start_year = gap_start_abs // 12
            gap_end_year = gap_end_abs // 12
            gap_years = round(gap_months / 12, 1)
            gaps.append(
                f"{gap_start_year}–{gap_end_year} : {gap_years:.0f} an(s) sans activité détectée"
            )
    return gaps
