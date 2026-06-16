"""Evaluates SectorProfile criteria against a NormalizedCV.

Each detection function signature: (cv: NormalizedCV, criterion: Criterion) -> CriterionResult
Scores are binary: 100 (met) or 0 (not met).
Unknown detection_fn → score=0, graceful degradation (no exception).
"""

import re
from typing import Callable

from src.core.lexicons import METRIC_PATTERNS
from src.core.schemas import CriterionResult, NormalizedCV
from src.core.sector_profiles import Criterion

# ---------------------------------------------------------------------------
# Habilitation keywords (ATS-visible evidence of regulatory qualifications)
# ---------------------------------------------------------------------------

_HABILITATION_KEYWORDS: list[str] = [
    "caces", "habilitation", "habilité", "habilitation électrique",
    "permis", "carte professionnelle", "carte pro",
    "fimo", "fco", "carte conducteur",
    "bpjeps", "ssiap", "ssiap1", "ssiap2", "ssiap3",
    "haccp", "hygiène alimentaire",
    "carte t", "carte g", "loi hoguet",
    "autorisation de conduite",
    "agrément", "agrément préfectoral",
    "certification", "certifié",
    "diplôme d'état", "de infirmier",
    "attestation", "attestation de formation",
    "h0", "b0", "br", "bc", "b1", "b2", "hv", "be",
    "risque chimique",
    "travaux en hauteur", "port du harnais",
]


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def _make_result(
    criterion: Criterion,
    scored: bool,
    evidence: list[str],
) -> CriterionResult:
    s = 100 if scored else 0
    return CriterionResult(
        criterion_id=criterion.id,
        label=criterion.label,
        weight=criterion.weight,
        required=criterion.required,
        score=s,
        evidence=evidence,
        weighted_score=s * criterion.weight / 100,
    )


def has_experience(cv: NormalizedCV, criterion: Criterion) -> CriterionResult:
    count = len(cv.experience)
    scored = count >= 1
    evidence = [f"{count} expérience(s) détectée(s)"] if scored else ["Aucune expérience détectée"]
    return _make_result(criterion, scored, evidence)


def has_education(cv: NormalizedCV, criterion: Criterion) -> CriterionResult:
    count = len(cv.education)
    scored = count >= 1
    evidence = [f"{count} formation(s) / diplôme(s) détecté(s)"] if scored else ["Aucune formation détectée"]
    return _make_result(criterion, scored, evidence)


def has_summary(cv: NormalizedCV, criterion: Criterion) -> CriterionResult:
    scored = bool(cv.summary)
    evidence = ["Accroche/profil présent"] if scored else ["Accroche/profil absente"]
    return _make_result(criterion, scored, evidence)


def has_dates(cv: NormalizedCV, criterion: Criterion) -> CriterionResult:
    if not cv.experience:
        return _make_result(criterion, False, ["Aucune expérience — dates non vérifiables"])
    dated = sum(1 for e in cv.experience if e.date_start)
    ratio = dated / len(cv.experience)
    scored = ratio >= 0.5
    evidence = [f"{dated}/{len(cv.experience)} expériences datées ({ratio:.0%})"]
    return _make_result(criterion, scored, evidence)


def has_sufficient_words(cv: NormalizedCV, criterion: Criterion) -> CriterionResult:
    wc = cv.word_count
    scored = wc >= 300
    evidence = [f"{wc} mots détectés (seuil : 300)"]
    return _make_result(criterion, scored, evidence)


def has_contact(cv: NormalizedCV, criterion: Criterion) -> CriterionResult:
    h = cv.header
    found: list[str] = []
    if h.email:
        found.append("Email détecté")
    if h.phone:
        found.append("Téléphone détecté")
    scored = len(found) >= 1
    evidence = found if found else ["Ni email ni téléphone détectés"]
    return _make_result(criterion, scored, evidence)


def has_tech_skills(cv: NormalizedCV, criterion: Criterion) -> CriterionResult:
    total = len(cv.skills_flat)
    scored = total >= 5
    evidence = [f"{total} compétences techniques détectées"]
    return _make_result(criterion, scored, evidence)


def has_sector_skills(cv: NormalizedCV, criterion: Criterion) -> CriterionResult:
    if not criterion.keywords:
        scored = len(cv.skills_flat) >= 3
        evidence = [f"{len(cv.skills_flat)} compétences sectorielles"]
        return _make_result(criterion, scored, evidence)
    check = (cv.raw_text + " " + " ".join(cv.skills_flat)).lower()
    found = [kw for kw in criterion.keywords if kw.lower() in check]
    scored = len(found) >= 1
    evidence = [f"Mots-clés trouvés : {', '.join(found)}"] if found else ["Aucun mot-clé sectoriel détecté"]
    return _make_result(criterion, scored, evidence)


def has_metrics(cv: NormalizedCV, criterion: Criterion) -> CriterionResult:
    full_text = " ".join(b for e in cv.experience for b in e.bullets) + " ".join(
        b for p in cv.projects for b in p.metrics
    )
    scored = any(p.search(full_text) for p in METRIC_PATTERNS)
    evidence = ["Métriques chiffrées présentes"] if scored else ["Aucune métrique chiffrée détectée"]
    return _make_result(criterion, scored, evidence)


def has_projects(cv: NormalizedCV, criterion: Criterion) -> CriterionResult:
    count = len(cv.projects)
    scored = count >= 1
    evidence = [f"{count} projet(s) documenté(s)"] if scored else ["Aucun projet documenté"]
    return _make_result(criterion, scored, evidence)


def has_profile_keywords(cv: NormalizedCV, criterion: Criterion) -> CriterionResult:
    if not criterion.keywords:
        return _make_result(criterion, False, ["Aucun mot-clé métier configuré"])
    check = (cv.raw_text + " " + " ".join(cv.skills_flat)).lower()
    found = [kw for kw in criterion.keywords if kw.lower() in check]
    scored = len(found) / len(criterion.keywords) >= 0.3
    pct = len(found) / len(criterion.keywords)
    evidence = [f"{len(found)}/{len(criterion.keywords)} mots-clés métier ({pct:.0%})"]
    return _make_result(criterion, scored, evidence)


def has_habilitations(cv: NormalizedCV, criterion: Criterion) -> CriterionResult:
    check = cv.raw_text.lower()
    found = [kw for kw in _HABILITATION_KEYWORDS if kw in check]
    scored = len(found) >= 1
    evidence = [f"Habilitations trouvées : {', '.join(found[:5])}"] if found else ["Aucune habilitation détectée"]
    return _make_result(criterion, scored, evidence)


def has_languages(cv: NormalizedCV, criterion: Criterion) -> CriterionResult:
    count = len(cv.languages)
    scored = count >= 1
    evidence = [f"{count} langue(s) détectée(s)"] if scored else ["Aucune langue étrangère détectée"]
    return _make_result(criterion, scored, evidence)


def has_ai_criterion(cv: NormalizedCV, criterion: Criterion) -> CriterionResult:
    """Keyword-based evaluation for AI-generated criteria (detection_fn='has_ai_criterion')."""
    if not criterion.keywords:
        return _make_result(criterion, False, ["Aucun mot-clé configuré"])
    check = cv.raw_text.lower()
    found = [kw for kw in criterion.keywords if kw.lower() in check]
    scored = len(found) >= 1
    evidence = (
        [f"Mots-clés trouvés : {', '.join(found)}"]
        if found
        else ["Aucun mot-clé détecté dans le CV"]
    )
    return _make_result(criterion, scored, evidence)


# ---------------------------------------------------------------------------
# Registry — detection_fn name → function
# ---------------------------------------------------------------------------

_EVAL_FUNCTIONS: dict[str, Callable] = {
    "has_experience":       has_experience,
    "has_education":        has_education,
    "has_summary":          has_summary,
    "has_dates":            has_dates,
    "has_sufficient_words": has_sufficient_words,
    "has_contact":          has_contact,
    "has_tech_skills":      has_tech_skills,
    "has_sector_skills":    has_sector_skills,
    "has_metrics":          has_metrics,
    "has_projects":         has_projects,
    "has_profile_keywords": has_profile_keywords,
    "has_habilitations":    has_habilitations,
    "has_languages":        has_languages,
    "has_ai_criterion":     has_ai_criterion,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_criteria(
    cv: NormalizedCV,
    sector_result: "SectorDetectionResult",  # type: ignore[name-defined]  # noqa: F821
) -> list[CriterionResult]:
    """Evaluate all criteria for the detected sector profile.

    Imports are deferred to avoid circular dependency with sector_registry.
    Graceful degradation: unknown detection_fn → score=0, no exception.
    """
    from src.core.sector_registry import ALL_PROFILES, GENERIC_PROFILE  # deferred
    from src.services.sector_detector import SectorDetectionResult  # deferred

    profile = ALL_PROFILES.get(sector_result.profile_id) or GENERIC_PROFILE
    results: list[CriterionResult] = []

    for criterion in profile.criteria:
        fn = _EVAL_FUNCTIONS.get(criterion.detection_fn)
        if fn is None:
            results.append(CriterionResult(
                criterion_id=criterion.id,
                label=criterion.label,
                weight=criterion.weight,
                required=criterion.required,
                score=0,
                evidence=[f"Fonction '{criterion.detection_fn}' non implémentée"],
                weighted_score=0.0,
            ))
        else:
            results.append(fn(cv, criterion))

    return results
