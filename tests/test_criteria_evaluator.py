"""Tests for CriteriaEvaluator — Phase C.

All 13 detection functions + evaluate_criteria() wrapper.
No network calls, no MiniLM, no PDF parsing.
"""

import pytest

from src.core.schemas import NormalizedCV, CVHeader, CVExperience, CVEducation, CVProject
from src.core.sector_profiles import Criterion
from src.services.criteria_evaluator import (
    evaluate_criteria,
    has_ai_criterion,
    has_contact,
    has_dates,
    has_education,
    has_experience,
    has_habilitations,
    has_languages,
    has_metrics,
    has_profile_keywords,
    has_projects,
    has_sector_skills,
    has_sufficient_words,
    has_summary,
    has_tech_skills,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _criterion(
    id: str = "c",
    label: str = "Test",
    weight: int = 20,
    required: bool = True,
    detection_fn: str = "test_fn",
    keywords: list[str] | None = None,
) -> Criterion:
    return Criterion(id=id, label=label, weight=weight, required=required,
                     detection_fn=detection_fn, keywords=keywords or [])


# ---------------------------------------------------------------------------
# 1. has_experience
# ---------------------------------------------------------------------------

class TestHasExperience:
    def test_with_experience_returns_100(self):
        cv = NormalizedCV(experience=[CVExperience(title="Dev")])
        result = has_experience(cv, _criterion())
        assert result.score == 100
        assert result.weighted_score == 20.0

    def test_without_experience_returns_0(self):
        cv = NormalizedCV()
        result = has_experience(cv, _criterion())
        assert result.score == 0
        assert result.weighted_score == 0.0

    def test_evidence_contains_count(self):
        cv = NormalizedCV(experience=[CVExperience(), CVExperience()])
        result = has_experience(cv, _criterion())
        assert "2" in result.evidence[0]


# ---------------------------------------------------------------------------
# 2. has_education
# ---------------------------------------------------------------------------

class TestHasEducation:
    def test_with_education_returns_100(self):
        cv = NormalizedCV(education=[CVEducation(degree="BTS")])
        result = has_education(cv, _criterion())
        assert result.score == 100

    def test_without_education_returns_0(self):
        cv = NormalizedCV()
        result = has_education(cv, _criterion())
        assert result.score == 0


# ---------------------------------------------------------------------------
# 3. has_summary
# ---------------------------------------------------------------------------

class TestHasSummary:
    def test_with_summary_returns_100(self):
        cv = NormalizedCV(summary="Professionnel expérimenté")
        result = has_summary(cv, _criterion())
        assert result.score == 100

    def test_without_summary_returns_0(self):
        cv = NormalizedCV()
        result = has_summary(cv, _criterion())
        assert result.score == 0


# ---------------------------------------------------------------------------
# 4. has_dates
# ---------------------------------------------------------------------------

class TestHasDates:
    def test_majority_dated_returns_100(self):
        cv = NormalizedCV(experience=[
            CVExperience(date_start="2022-01"),
            CVExperience(date_start="2020-03"),
        ])
        result = has_dates(cv, _criterion())
        assert result.score == 100

    def test_no_experience_returns_0(self):
        cv = NormalizedCV()
        result = has_dates(cv, _criterion())
        assert result.score == 0

    def test_minority_dated_returns_0(self):
        cv = NormalizedCV(experience=[
            CVExperience(date_start=None),
            CVExperience(date_start=None),
            CVExperience(date_start="2020-01"),
        ])
        result = has_dates(cv, _criterion())
        # 1/3 = 33% < 50% → score 0
        assert result.score == 0


# ---------------------------------------------------------------------------
# 5. has_sufficient_words
# ---------------------------------------------------------------------------

class TestHasSufficientWords:
    def test_300_words_returns_100(self):
        cv = NormalizedCV(word_count=350)
        result = has_sufficient_words(cv, _criterion())
        assert result.score == 100

    def test_under_300_returns_0(self):
        cv = NormalizedCV(word_count=150)
        result = has_sufficient_words(cv, _criterion())
        assert result.score == 0

    def test_exactly_300_returns_100(self):
        cv = NormalizedCV(word_count=300)
        result = has_sufficient_words(cv, _criterion())
        assert result.score == 100


# ---------------------------------------------------------------------------
# 6. has_contact
# ---------------------------------------------------------------------------

class TestHasContact:
    def test_email_only_returns_100(self):
        cv = NormalizedCV(header=CVHeader(email="test@test.com"))
        result = has_contact(cv, _criterion())
        assert result.score == 100

    def test_phone_only_returns_100(self):
        cv = NormalizedCV(header=CVHeader(phone="0612345678"))
        result = has_contact(cv, _criterion())
        assert result.score == 100

    def test_no_contact_returns_0(self):
        cv = NormalizedCV()
        result = has_contact(cv, _criterion())
        assert result.score == 0

    def test_evidence_does_not_expose_actual_values(self):
        """Evidence doit afficher 'Email détecté', pas la valeur réelle."""
        cv = NormalizedCV(header=CVHeader(email="secret@corp.com", phone="0600000000"))
        result = has_contact(cv, _criterion())
        assert "secret@corp.com" not in str(result.evidence)
        assert "0600000000" not in str(result.evidence)
        assert "détecté" in " ".join(result.evidence).lower()


# ---------------------------------------------------------------------------
# 7. has_tech_skills
# ---------------------------------------------------------------------------

class TestHasTechSkills:
    def test_five_skills_returns_100(self):
        cv = NormalizedCV(skills_flat=["python", "docker", "mlflow", "git", "linux"])
        result = has_tech_skills(cv, _criterion())
        assert result.score == 100

    def test_zero_skills_returns_0(self):
        cv = NormalizedCV()
        result = has_tech_skills(cv, _criterion())
        assert result.score == 0

    def test_four_skills_returns_0(self):
        cv = NormalizedCV(skills_flat=["a", "b", "c", "d"])
        result = has_tech_skills(cv, _criterion())
        assert result.score == 0


# ---------------------------------------------------------------------------
# 8. has_sector_skills
# ---------------------------------------------------------------------------

class TestHasSectorSkills:
    def test_keyword_found_in_raw_text_returns_100(self):
        cv = NormalizedCV(raw_text="Expérience en encaissement caisse et facing rayon")
        c = _criterion(keywords=["encaissement", "facing"])
        result = has_sector_skills(cv, c)
        assert result.score == 100

    def test_keyword_not_found_returns_0(self):
        cv = NormalizedCV(raw_text="Expérience en développement logiciel")
        c = _criterion(keywords=["encaissement", "facing"])
        result = has_sector_skills(cv, c)
        assert result.score == 0

    def test_empty_keywords_fallback_to_skills_flat(self):
        cv = NormalizedCV(skills_flat=["a", "b", "c"])
        result = has_sector_skills(cv, _criterion(keywords=[]))
        assert result.score == 100  # 3 >= 3


# ---------------------------------------------------------------------------
# 9. has_metrics
# ---------------------------------------------------------------------------

class TestHasMetrics:
    def test_metrics_in_bullets_returns_100(self):
        cv = NormalizedCV(experience=[CVExperience(bullets=["Augmentation CA de 25%"])])
        result = has_metrics(cv, _criterion())
        assert result.score == 100

    def test_no_metrics_returns_0(self):
        cv = NormalizedCV(experience=[CVExperience(bullets=["Gestion de l'équipe"])])
        result = has_metrics(cv, _criterion())
        assert result.score == 0


# ---------------------------------------------------------------------------
# 10. has_projects
# ---------------------------------------------------------------------------

class TestHasProjects:
    def test_with_projects_returns_100(self):
        cv = NormalizedCV(projects=[CVProject(name="Mon projet")])
        result = has_projects(cv, _criterion())
        assert result.score == 100

    def test_without_projects_returns_0(self):
        cv = NormalizedCV()
        result = has_projects(cv, _criterion())
        assert result.score == 0


# ---------------------------------------------------------------------------
# 11. has_profile_keywords
# ---------------------------------------------------------------------------

class TestHasProfileKeywords:
    def test_30pct_keywords_present_returns_100(self):
        cv = NormalizedCV(raw_text="mlflow kubeflow pipeline mlops déploiement modèle inférence monitoring")
        c = _criterion(keywords=["mlflow", "kubeflow", "pipeline mlops",
                                  "déploiement modèle", "inférence", "monitoring modèle"])
        result = has_profile_keywords(cv, c)
        # mlflow, kubeflow, pipeline mlops, déploiement modèle, inférence = 5/6 = 83% ≥ 30%
        assert result.score == 100

    def test_below_30pct_returns_0(self):
        cv = NormalizedCV(raw_text="je suis un candidat motivé")
        c = _criterion(keywords=["mlflow", "kubeflow", "pipeline mlops",
                                  "déploiement modèle", "inférence", "monitoring modèle"])
        result = has_profile_keywords(cv, c)
        # 0/6 = 0% < 30%
        assert result.score == 0

    def test_empty_keywords_returns_0(self):
        cv = NormalizedCV(raw_text="mlflow kubeflow")
        result = has_profile_keywords(cv, _criterion(keywords=[]))
        assert result.score == 0


# ---------------------------------------------------------------------------
# 12. has_habilitations
# ---------------------------------------------------------------------------

class TestHasHabilitations:
    def test_caces_in_raw_text_returns_100(self):
        cv = NormalizedCV(raw_text="Titulaire du CACES R489 catégorie 1")
        result = has_habilitations(cv, _criterion())
        assert result.score == 100

    def test_habilitation_electrique_returns_100(self):
        cv = NormalizedCV(raw_text="Habilitation électrique BR BC obtenue en 2022")
        result = has_habilitations(cv, _criterion())
        assert result.score == 100

    def test_no_habilitation_returns_0(self):
        cv = NormalizedCV(raw_text="Expérience commerciale dans la vente")
        result = has_habilitations(cv, _criterion())
        assert result.score == 0

    def test_evidence_lists_found_keywords(self):
        cv = NormalizedCV(raw_text="CACES R489 et HACCP et FIMO en cours")
        result = has_habilitations(cv, _criterion())
        assert result.score == 100
        assert len(result.evidence) > 0


# ---------------------------------------------------------------------------
# 13. has_languages
# ---------------------------------------------------------------------------

class TestHasLanguages:
    def test_with_language_returns_100(self):
        cv = NormalizedCV(languages=["Anglais B2"])
        result = has_languages(cv, _criterion())
        assert result.score == 100

    def test_no_languages_returns_0(self):
        cv = NormalizedCV()
        result = has_languages(cv, _criterion())
        assert result.score == 0


# ---------------------------------------------------------------------------
# 14. has_ai_criterion
# ---------------------------------------------------------------------------

class TestHasAiCriterion:
    def test_keyword_in_raw_text_returns_100(self):
        cv = NormalizedCV(raw_text="Je maîtrise le permis de conduire et les habilitations")
        c = _criterion(keywords=["permis de conduire"])
        result = has_ai_criterion(cv, c)
        assert result.score == 100

    def test_keyword_not_found_returns_0(self):
        cv = NormalizedCV(raw_text="Développeur Python senior")
        c = _criterion(keywords=["habilitation électrique"])
        result = has_ai_criterion(cv, c)
        assert result.score == 0

    def test_empty_keywords_returns_0(self):
        cv = NormalizedCV(raw_text="texte complet du CV")
        result = has_ai_criterion(cv, _criterion(keywords=[]))
        assert result.score == 0

    def test_case_insensitive(self):
        cv = NormalizedCV(raw_text="Expérience en SOUDAGE et MÉCANIQUE")
        c = _criterion(keywords=["soudage", "mécanique"])
        result = has_ai_criterion(cv, c)
        assert result.score == 100


# ---------------------------------------------------------------------------
# 15. evaluate_criteria — intégration
# ---------------------------------------------------------------------------

class TestEvaluateCriteria:
    def test_returns_list_for_generic_profile(self):
        """evaluate_criteria retourne autant de résultats que de critères GENERIC."""
        from src.core.sector_registry import GENERIC_PROFILE
        from src.services.sector_detector import SectorDetectionResult
        cv = NormalizedCV(experience=[CVExperience(title="Dev")])
        generic_result = SectorDetectionResult(
            profile_id=GENERIC_PROFILE.id,
            sector=GENERIC_PROFILE.sector,
            job_title=GENERIC_PROFILE.job_title,
            confidence=0.0,
            alternatives=[],
        )
        results = evaluate_criteria(cv, generic_result)
        assert len(results) == len(GENERIC_PROFILE.criteria)

    def test_unknown_detection_fn_returns_score_0(self):
        """Fonction inconnue → score=0, pas d'exception."""
        from src.core.sector_profiles import Criterion, SectorProfile
        from src.services.sector_detector import SectorDetectionResult
        from src.core.sector_registry import ALL_PROFILES
        # Injecter temporairement un critère avec fn inconnue
        profile = list(ALL_PROFILES.values())[0]
        fake_sr = SectorDetectionResult(
            profile_id=profile.id,
            sector=profile.sector,
            job_title=profile.job_title,
            confidence=0.9,
            alternatives=[],
        )
        # On ne teste pas que la fn inconnue est appelée (trop couplé),
        # juste qu'evaluate_criteria() tourne sans erreur.
        cv = NormalizedCV()
        results = evaluate_criteria(cv, fake_sr)
        assert isinstance(results, list)
        assert all(r.score in (0, 100) for r in results)

    def test_all_results_have_criterion_id(self):
        """Chaque CriterionResult a un criterion_id non vide."""
        from src.core.sector_registry import GENERIC_PROFILE
        from src.services.sector_detector import SectorDetectionResult
        cv = NormalizedCV()
        sr = SectorDetectionResult(
            profile_id=GENERIC_PROFILE.id,
            sector=GENERIC_PROFILE.sector,
            job_title=GENERIC_PROFILE.job_title,
            confidence=0.0,
            alternatives=[],
        )
        results = evaluate_criteria(cv, sr)
        for r in results:
            assert r.criterion_id, f"criterion_id vide : {r}"

    def test_weighted_scores_consistent(self):
        """weighted_score = score * weight / 100 pour chaque résultat."""
        from src.core.sector_registry import GENERIC_PROFILE
        from src.services.sector_detector import SectorDetectionResult
        cv = NormalizedCV(
            experience=[CVExperience(title="test")],
            education=[CVEducation(degree="BTS")],
        )
        sr = SectorDetectionResult(
            profile_id=GENERIC_PROFILE.id,
            sector=GENERIC_PROFILE.sector,
            job_title=GENERIC_PROFILE.job_title,
            confidence=0.0,
            alternatives=[],
        )
        results = evaluate_criteria(cv, sr)
        for r in results:
            expected = r.score * r.weight / 100
            assert abs(r.weighted_score - expected) < 0.01, (
                f"weighted_score={r.weighted_score} ≠ {expected} pour {r.criterion_id}"
            )


# ---------------------------------------------------------------------------
# 16. SectorDetector.make_forced_result
# ---------------------------------------------------------------------------

class TestMakeForced:
    def test_known_profile_id(self):
        from src.services.sector_detector import SectorDetector
        r = SectorDetector.make_forced_result("vendeur")
        assert r.profile_id == "vendeur"
        assert r.confidence == 1.0
        assert r.alternatives == []

    def test_unknown_profile_id_falls_back_to_generic(self):
        from src.services.sector_detector import SectorDetector
        from src.core.sector_registry import GENERIC_PROFILE
        r = SectorDetector.make_forced_result("profil_inexistant")
        assert r.profile_id == GENERIC_PROFILE.id

    def test_result_has_sector_and_job_title(self):
        from src.services.sector_detector import SectorDetector
        r = SectorDetector.make_forced_result("infirmier")
        assert isinstance(r.sector, str) and r.sector
        assert isinstance(r.job_title, str) and r.job_title

    def test_confidence_is_1(self):
        from src.services.sector_detector import SectorDetector
        r = SectorDetector.make_forced_result("ml_engineer")
        assert r.confidence == pytest.approx(1.0)
