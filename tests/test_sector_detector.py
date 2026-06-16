"""Tests for SectorDetector — Phase A infrastructure.

MiniLM is never loaded: all title-scoring calls are mocked via
patch.object(detector, '_compute_title_scores', ...). Only
_score_skills and _score_experience run against real logic.
"""

import pytest
from unittest.mock import patch

from src.core.schemas import NormalizedCV, CVExperience, CVSkills
from src.core.sector_profiles import Criterion, SectorProfile
from src.core.sector_registry import (
    ALL_PROFILES,
    GENERIC_PROFILE,
    SECTOR_DISPLAY_NAMES,
    SECTORS,
    PROFILE_BY_SECTOR,
)
from src.services.sector_detector import SectorDetector, SectorDetectionResult, _CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ml_cv() -> NormalizedCV:
    return NormalizedCV(
        job_title="Machine Learning Engineer",
        skills_flat=["pytorch", "mlflow", "kubeflow", "docker", "python"],
        experience=[CVExperience(bullets=[
            "Déploiement modèle PyTorch sur Kubernetes",
            "Pipeline MLops avec Kubeflow et MLflow tracking",
        ])],
    )


def _vendeur_cv() -> NormalizedCV:
    return NormalizedCV(
        job_title="Vendeur conseil",
        sections={"skills": "encaissement facing merchandising point de vente fidélisation clientèle rayon"},
        experience=[CVExperience(bullets=[
            "Atteinte des objectifs vente à 120%",
            "Encaissement et gestion de caisse",
            "Facing rayon et merchandising produits",
        ])],
    )


def _infirmier_cv() -> NormalizedCV:
    return NormalizedCV(
        job_title="Infirmière diplômée d'État",
        sections={"skills": "soins infirmiers perfusion pansements complexes prescriptions médicales transmissions infirmières"},
        experience=[CVExperience(bullets=[
            "Soins infirmiers en service de médecine interne",
            "Perfusion intraveineuse et pansements complexes",
            "Transmissions infirmières en équipe pluridisciplinaire",
        ])],
    )


def _empty_cv() -> NormalizedCV:
    return NormalizedCV()


# ---------------------------------------------------------------------------
# 1. _score_skills — logique interne
# ---------------------------------------------------------------------------

class TestScoreSkills:
    def test_tech_tokens_in_skills_flat(self):
        """MLflow et Kubeflow dans skills_flat → score > 0 pour ml_engineer."""
        detector = SectorDetector()
        cv = NormalizedCV(skills_flat=["pytorch", "mlflow", "kubeflow"])
        kw = ALL_PROFILES["ml_engineer"].detection_keywords
        assert detector._score_skills(cv, kw) > 0.0

    def test_all_keywords_in_section_text(self):
        """Tous les mots-clés vendeur dans sections['skills'] → score = 1.0."""
        detector = SectorDetector()
        kw = ALL_PROFILES["vendeur"].detection_keywords
        cv = NormalizedCV(sections={"skills": " ".join(kw)})
        assert detector._score_skills(cv, kw) == pytest.approx(1.0)

    def test_non_tech_section_text(self):
        """Section skills avec termes retail → score vendeur élevé."""
        detector = SectorDetector()
        cv = NormalizedCV(sections={"skills": "encaissement facing merchandising rayon"})
        kw = ALL_PROFILES["vendeur"].detection_keywords
        assert detector._score_skills(cv, kw) >= 0.5

    def test_cross_profile_no_match(self):
        """Skills ML ne matchent pas les keywords vendeur."""
        detector = SectorDetector()
        cv = NormalizedCV(skills_flat=["pytorch", "mlflow", "docker"])
        kw = ALL_PROFILES["vendeur"].detection_keywords
        assert detector._score_skills(cv, kw) == pytest.approx(0.0)

    def test_empty_keywords_returns_zero(self):
        """Profile sans detection_keywords → score = 0 (pas d'erreur)."""
        detector = SectorDetector()
        assert detector._score_skills(_ml_cv(), []) == pytest.approx(0.0)

    def test_case_insensitive_matching(self):
        """La comparaison est insensible à la casse."""
        detector = SectorDetector()
        cv = NormalizedCV(sections={"skills": "ENCAISSEMENT FACING MERCHANDISING"})
        kw = ["encaissement", "facing", "merchandising"]
        assert detector._score_skills(cv, kw) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. _score_experience — logique interne
# ---------------------------------------------------------------------------

class TestScoreExperience:
    def test_keywords_in_bullets(self):
        """Mots-clés vendeur dans les bullets → score > 0."""
        detector = SectorDetector()
        cv = NormalizedCV(experience=[CVExperience(bullets=[
            "Encaissement des clients en caisse",
            "Facing et mise en rayon",
        ])])
        kw = ALL_PROFILES["vendeur"].detection_keywords
        assert detector._score_experience(cv, kw) > 0.0

    def test_no_experience_returns_zero(self):
        """Pas d'expérience + sections vides → score = 0."""
        detector = SectorDetector()
        cv = NormalizedCV()
        assert detector._score_experience(cv, ALL_PROFILES["vendeur"].detection_keywords) == pytest.approx(0.0)

    def test_experience_section_fallback(self):
        """Sans bullets, sections['experience'] sert de fallback."""
        detector = SectorDetector()
        cv = NormalizedCV(sections={"experience": "encaissement facing merchandising"})
        kw = ALL_PROFILES["vendeur"].detection_keywords
        assert detector._score_experience(cv, kw) > 0.0

    def test_infirmier_keywords_in_bullets(self):
        """Bullets soins infirmiers → score infirmier élevé."""
        detector = SectorDetector()
        cv = _infirmier_cv()
        kw = ALL_PROFILES["infirmier"].detection_keywords
        score = detector._score_experience(cv, kw)
        assert score > 0.3


# ---------------------------------------------------------------------------
# 3. detect() — titre + skills + expérience combinés (mock title scores)
# ---------------------------------------------------------------------------

class TestDetect:
    def test_detect_ml_engineer(self):
        """Mock titre fort ML → profil ml_engineer détecté."""
        cv = _ml_cv()
        detector = SectorDetector()
        with patch.object(detector, "_compute_title_scores", return_value={"ml_engineer": 0.92}):
            result = detector.detect(cv)
        assert result.profile_id == "ml_engineer"
        assert result.sector == "Informatique & Digital"
        assert result.confidence > _CONFIDENCE_THRESHOLD

    def test_detect_vendeur(self):
        """Mock titre vendeur + skills/experience matchent → vendeur détecté."""
        cv = _vendeur_cv()
        detector = SectorDetector()
        with patch.object(detector, "_compute_title_scores", return_value={"vendeur": 0.88}):
            result = detector.detect(cv)
        assert result.profile_id == "vendeur"
        assert result.confidence > _CONFIDENCE_THRESHOLD

    def test_detect_infirmier(self):
        """Mock titre infirmier + skills soins → infirmier détecté."""
        cv = _infirmier_cv()
        detector = SectorDetector()
        with patch.object(detector, "_compute_title_scores", return_value={"infirmier": 0.95}):
            result = detector.detect(cv)
        assert result.profile_id == "infirmier"

    def test_detect_fallback_low_confidence(self):
        """CV vide + aucun titre → confidence < seuil → non_detecte."""
        cv = _empty_cv()
        detector = SectorDetector()
        # job_title=None → _compute_title_scores natural returns {}
        result = detector.detect(cv)
        assert result.profile_id == GENERIC_PROFILE.id
        assert result.confidence < _CONFIDENCE_THRESHOLD

    def test_detect_no_job_title_skills_drive(self):
        """Sans job_title, les skills discriminants suffisent à dépasser le seuil."""
        kw = ALL_PROFILES["vendeur"].detection_keywords
        cv = NormalizedCV(
            job_title=None,
            # sections["skills"] contient exactement les mots-clés vendeur
            sections={"skills": " ".join(kw)},
            experience=[CVExperience(bullets=["Encaissement caisse", "Facing rayon"])],
        )
        detector = SectorDetector()
        # job_title=None → _compute_title_scores returns {} without loading MiniLM
        result = detector.detect(cv)
        assert result.profile_id == "vendeur"
        assert result.confidence >= _CONFIDENCE_THRESHOLD

    def test_detect_ambiguous_assistant_low_confidence(self):
        """Titre 'assistant' ambigu + aucune skill spécifique → generic."""
        cv = NormalizedCV(job_title="Assistant")
        detector = SectorDetector()
        # Simule un titre peu discriminant : scores faibles pour tous
        with patch.object(detector, "_compute_title_scores",
                          return_value={"assistant_administratif": 0.5, "charge_mission_rh": 0.4}):
            result = detector.detect(cv)
        # 0.4 * 0.5 + 0.35 * 0 + 0.25 * 0 = 0.20 < 0.3
        assert result.profile_id == GENERIC_PROFILE.id
        assert result.confidence < _CONFIDENCE_THRESHOLD

    def test_detect_alternatives_distinct_and_not_best(self):
        """Les 3 alternatives sont distinctes et n'incluent pas le profil principal."""
        cv = _ml_cv()
        detector = SectorDetector()
        with patch.object(detector, "_compute_title_scores", return_value={
            "ml_engineer": 0.92, "data_scientist": 0.70, "data_analyst": 0.55, "devops": 0.40,
        }):
            result = detector.detect(cv)
        alt_ids = [a[0] for a in result.alternatives]
        assert len(alt_ids) == 3
        assert len(set(alt_ids)) == 3                # toutes distinctes
        assert result.profile_id not in alt_ids       # le meilleur absent des alternatives

    def test_detect_confidence_always_in_bounds(self):
        """La confidence est toujours dans [0.0, 1.0] quelle que soit l'entrée."""
        detector = SectorDetector()
        for cv in [_ml_cv(), _empty_cv(), _vendeur_cv()]:
            with patch.object(detector, "_compute_title_scores", return_value={}):
                result = detector.detect(cv)
            assert 0.0 <= result.confidence <= 1.0

    def test_detect_confidence_at_threshold_uses_best_profile(self):
        """À exactement 0.3 (non strict), le meilleur profil est retourné, pas le générique."""
        # 0.4 * 0.75 + 0 + 0 = 0.300 exactement → non_detecte condition: < 0.3 → False
        cv = NormalizedCV(job_title="Opérateur de production")
        detector = SectorDetector()
        with patch.object(detector, "_compute_title_scores",
                          return_value={"operateur_production": 0.75}):
            result = detector.detect(cv)
        assert result.profile_id == "operateur_production"
        assert result.confidence == pytest.approx(0.3, abs=0.001)

    def test_detect_just_below_threshold_uses_generic(self):
        """Juste sous 0.3 → GENERIC_PROFILE."""
        # 0.4 * 0.72 = 0.288 < 0.3
        cv = NormalizedCV(job_title="Opérateur de production")
        detector = SectorDetector()
        with patch.object(detector, "_compute_title_scores",
                          return_value={"operateur_production": 0.72}):
            result = detector.detect(cv)
        assert result.profile_id == GENERIC_PROFILE.id

    def test_detect_multilingual_cv_french_keywords_match(self):
        """CV avec sections mélangées FR/EN — keywords FR matchent dans sections."""
        cv = NormalizedCV(
            job_title=None,
            sections={"skills": "infirmier soins infirmiers perfusion pansements complexes RN nursing"},
        )
        kw = ALL_PROFILES["infirmier"].detection_keywords
        detector = SectorDetector()
        score = detector._score_skills(cv, kw)
        assert score > 0.0  # au moins "soins infirmiers" et "perfusion" matchent

    def test_detect_result_has_all_fields(self):
        """SectorDetectionResult contient tous les champs requis."""
        detector = SectorDetector()
        with patch.object(detector, "_compute_title_scores", return_value={"vendeur": 0.9}):
            result = detector.detect(_vendeur_cv())
        assert isinstance(result.profile_id, str)
        assert isinstance(result.sector, str)
        assert isinstance(result.job_title, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.alternatives, list)
        assert len(result.alternatives) == 3


# ---------------------------------------------------------------------------
# 4. Sanity checks — registre
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_profiles_have_aliases(self):
        """Chaque SectorProfile a au moins 4 aliases."""
        for pid, profile in ALL_PROFILES.items():
            assert len(profile.aliases) >= 4, f"{pid}: seulement {len(profile.aliases)} aliases"

    def test_all_profiles_have_detection_keywords(self):
        """Chaque SectorProfile a au moins 4 detection_keywords."""
        for pid, profile in ALL_PROFILES.items():
            assert len(profile.detection_keywords) >= 4, (
                f"{pid}: seulement {len(profile.detection_keywords)} keywords"
            )

    def test_all_profile_ids_in_all_profiles(self):
        """Chaque profile_id listé dans SECTORS existe dans ALL_PROFILES."""
        for sector_key, pids in SECTORS.items():
            for pid in pids:
                assert pid in ALL_PROFILES, f"SECTORS[{sector_key}] référence {pid!r} absent de ALL_PROFILES"

    def test_generic_profile_has_six_criteria(self):
        """GENERIC_PROFILE possède exactement 6 critères."""
        assert len(GENERIC_PROFILE.criteria) == 6

    def test_generic_criteria_weights_sum_to_100(self):
        """La somme des poids des critères génériques vaut 100."""
        total = sum(c.weight for c in GENERIC_PROFILE.criteria)
        assert total == 100

    def test_generic_has_required_and_optional(self):
        """GENERIC_PROFILE a des critères required ET optional."""
        required = [c for c in GENERIC_PROFILE.criteria if c.required]
        optional = [c for c in GENERIC_PROFILE.criteria if not c.required]
        assert len(required) >= 1
        assert len(optional) >= 1

    def test_profile_by_sector_covers_all_sectors(self):
        """PROFILE_BY_SECTOR contient toutes les valeurs de SECTOR_DISPLAY_NAMES."""
        for display_name in SECTOR_DISPLAY_NAMES.values():
            assert display_name in PROFILE_BY_SECTOR, f"{display_name} absent de PROFILE_BY_SECTOR"

    def test_sector_count(self):
        """Le registre couvre au moins 20 secteurs."""
        assert len(SECTORS) >= 20

    def test_profile_count(self):
        """Le registre contient au moins 120 profils."""
        assert len(ALL_PROFILES) >= 120
