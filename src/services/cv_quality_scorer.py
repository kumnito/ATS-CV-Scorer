"""CV quality scorer: produces a CVQualityReport from a NormalizedCV."""

import re
from datetime import date
from typing import Optional

from src.core.lexicons import ACTION_VERBS_EN, ACTION_VERBS_FR, METRIC_PATTERNS, SKILL_CATEGORIES
from src.core.schemas import CVQualityReport, NormalizedCV

_REQUIRED_SECTIONS = ["experience", "education", "skills"]
_ALL_SECTIONS = ["summary", "experience", "education", "skills", "projects", "certifications", "languages"]

_ACTION_VERBS = ACTION_VERBS_EN | {v.rstrip("é") for v in ACTION_VERBS_FR} | ACTION_VERBS_FR


class CVQualityScorer:
    def score(self, cv: NormalizedCV) -> CVQualityReport:
        detected = _detected_sections(cv)
        missing = [s for s in _REQUIRED_SECTIONS if s not in detected]

        score_structure = _score_structure(cv, detected)
        score_contenu = _score_contenu(cv, detected)
        score_global = round(score_structure * 0.40 + score_contenu * 0.60)

        kd = _keyword_density(cv)
        has_metrics = _has_metrics(cv)
        career = _career_stats(cv)
        gaps = _detect_career_gaps(cv)
        recs = _build_recommendations(cv, detected, missing, kd, has_metrics, cv.word_count)

        return CVQualityReport(
            score_structure=score_structure,
            score_contenu=score_contenu,
            score_global=score_global,
            sections_detectees=detected,
            sections_manquantes=missing,
            layout=cv.layout_detected,
            word_count=cv.word_count,
            has_metrics=has_metrics,
            keyword_density=round(kd, 3),
            recommendations=recs,
            total_experience_years=career["total_experience_years"],
            career_start_year=career["career_start_year"],
            most_recent_role=career["most_recent_role"],
            education_years=career["education_years"],
            career_gaps=gaps,
        )


# ---------------------------------------------------------------------------
# Section detection
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


# ---------------------------------------------------------------------------
# Structure score (0-100)
# ---------------------------------------------------------------------------


def _score_structure(cv: NormalizedCV, detected: list[str]) -> int:
    score = 0

    # Required sections +15 each
    for s in _REQUIRED_SECTIONS:
        if s in detected:
            score += 15

    # Layout bonus: single column preferred by ATS parsers
    if cv.layout_detected == "single_column":
        score += 20
    else:
        score += 5

    # Section order: skills before education (easier ATS keyword matching)
    if "skills" in detected and "education" in detected:
        sections_order = detected[:]
        if sections_order.index("skills") < sections_order.index("education"):
            score += 10

    # Complete header (+15 if name + title + email + phone all present)
    h = cv.header
    if h.name and h.title and h.email and h.phone:
        score += 15

    return min(100, score)


# ---------------------------------------------------------------------------
# Content score (0-100)
# ---------------------------------------------------------------------------


def _score_contenu(cv: NormalizedCV, detected: list[str]) -> int:
    score = 0

    if cv.summary:
        score += 10

    if cv.projects:
        score += 15

    if _has_metrics(cv):
        score += 15

    kd = _keyword_density(cv)
    if 0.15 <= kd <= 0.25:
        score += 20
    elif 0.10 <= kd < 0.15 or 0.25 < kd <= 0.35:
        score += 10

    if 500 <= cv.word_count <= 750:
        score += 15
    elif 400 <= cv.word_count < 500 or 750 < cv.word_count <= 900:
        score += 8

    if _has_action_verbs(cv):
        score += 10

    # Date completeness: +10 if all experience entries have dates
    if cv.experience:
        dated = sum(1 for e in cv.experience if e.date_start)
        if dated == len(cv.experience):
            score += 5

    # Chronological order: +5 if experience is sorted most-recent-first
    if _is_sorted_chronologically(cv.experience):
        score += 5

    return min(100, score)


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


def _has_action_verbs(cv: NormalizedCV) -> bool:
    bullets_text = " ".join(b for e in cv.experience for b in e.bullets).lower()
    if not bullets_text:
        bullets_text = cv.raw_text.lower()
    words = set(re.findall(r"\b\w+\b", bullets_text))
    return bool(words & {v.lower() for v in _ACTION_VERBS})


def _is_sorted_chronologically(experience: list) -> bool:
    dated = [e for e in experience if e.date_start]
    if len(dated) < 2:
        return True
    starts = []
    for e in dated:
        parts = e.date_start.split("-")
        starts.append(int(parts[0]) * 12 + (int(parts[1]) if len(parts) > 1 else 1))
    return all(starts[i] >= starts[i + 1] for i in range(len(starts) - 1))


# ---------------------------------------------------------------------------
# Career statistics
# ---------------------------------------------------------------------------


def _parse_ym(date_str: Optional[str]) -> Optional[tuple[int, int]]:
    if not date_str or date_str == "present":
        return None
    parts = date_str.split("-")
    return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 1)


def _career_stats(cv: NormalizedCV) -> dict:
    total_months = sum(e.duration_months or 0 for e in cv.experience)
    total_years = round(total_months / 12, 1)

    start_years = []
    for e in cv.experience:
        if e.date_start:
            parts = e.date_start.split("-")
            start_years.append(int(parts[0]))
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

    # Build intervals as (start_month, end_month) in absolute months from year 1
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


# ---------------------------------------------------------------------------
# Recommendations (ordered by estimated ATS impact)
# ---------------------------------------------------------------------------


def _build_recommendations(
    cv: NormalizedCV,
    detected: list[str],
    missing: list[str],
    kd: float,
    has_metrics: bool,
    word_count: int,
) -> list[str]:
    recs: list[str] = []

    if "experience" in missing:
        recs.append("Ajouter une section Expérience professionnelle (+15 pts structure)")
    if "skills" in missing:
        recs.append("Ajouter une section Compétences clairement identifiée (+15 pts structure)")
    if "education" in missing:
        recs.append("Ajouter une section Formation/Éducation (+15 pts structure)")

    if cv.layout_detected == "two_columns":
        recs.append("Convertir en CV une colonne : certains parseurs ATS ne lisent pas les colonnes multiples (+15 pts structure)")

    h = cv.header
    missing_header = [f for f, v in [("nom", h.name), ("titre", h.title), ("email", h.email), ("téléphone", h.phone)] if not v]
    if missing_header:
        recs.append(f"Compléter l'en-tête ({', '.join(missing_header)} manquant(s)) (+15 pts structure)")

    if not cv.summary:
        recs.append("Ajouter un profil/accroche en début de CV (+10 pts contenu)")

    if not cv.projects:
        recs.append("Ajouter une section Projets avec stack technique et métriques (+15 pts contenu)")

    if not has_metrics:
        recs.append("Quantifier les résultats dans les bullets d'expérience (%, chiffres, impact) (+15 pts contenu)")

    if kd < 0.10:
        recs.append("Densifier les mots-clés techniques (actuellement < 10 % des mots) (+20 pts contenu)")
    elif kd > 0.35:
        recs.append("Réduire la répétition de mots-clés techniques (> 35 % = risque de sur-optimisation)")

    if word_count < 400:
        recs.append(f"CV trop court ({word_count} mots) — enrichir avec contexte et bullets détaillés (cible : 500-750 mots)")
    elif word_count > 900:
        recs.append(f"CV trop long ({word_count} mots) — condenser à 500-750 mots pour faciliter la lecture ATS")

    if not _has_action_verbs(cv):
        recs.append("Commencer les bullets par des verbes d'action (dirigé, développé, optimisé…) (+10 pts contenu)")

    return recs


def _has_action_verbs(cv: NormalizedCV) -> bool:
    bullets_text = " ".join(b for e in cv.experience for b in e.bullets).lower()
    if not bullets_text:
        bullets_text = cv.raw_text.lower()
    words = set(re.findall(r"\b\w+\b", bullets_text))
    return bool(words & {v.lower() for v in _ACTION_VERBS})
