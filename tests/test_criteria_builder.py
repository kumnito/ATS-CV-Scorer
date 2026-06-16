"""Tests for CriteriaBuilder — Phase B.

Validates:
- 23 sector templates couverts
- Σ weights = 100 pour chaque template et chaque profil built
- Overrides profil-spécifiques (ml_engineer, infirmier, chauffeur_pl, cuisinier…)
- Auto-injection de detection_keywords dans eval_profile_keywords
- Intégrité du registre après build (toutes les criteria[] sont peuplées)
"""

import pytest

from src.core.sector_profiles import Criterion, SectorProfile
from src.core.sector_registry import ALL_PROFILES, GENERIC_PROFILE, SECTORS
from src.services.criteria_builder import (
    CriteriaBuilder,
    _CRITERION_DEFS,
    _PROFILE_OVERRIDES,
    _SECTOR_TEMPLATES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(profile_id: str, sector_key: str) -> list[Criterion]:
    builder = CriteriaBuilder()
    return builder.build_for_profile(ALL_PROFILES[profile_id], sector_key)


def _sector_key_of(profile_id: str) -> str:
    for sk, pids in SECTORS.items():
        if profile_id in pids:
            return sk
    raise KeyError(f"{profile_id!r} absent de SECTORS")


# ---------------------------------------------------------------------------
# 1. Templates sectoriels — couverture et sommes
# ---------------------------------------------------------------------------

class TestSectorTemplates:
    def test_all_sector_keys_have_templates(self):
        """Chaque sector_key du registre a un template dans CriteriaBuilder."""
        for sk in SECTORS:
            assert sk in _SECTOR_TEMPLATES, f"Pas de template pour sector_key={sk!r}"

    def test_all_sector_templates_sum_to_100(self):
        """Chaque template a des poids qui totalisent exactement 100."""
        for sk, spec in _SECTOR_TEMPLATES.items():
            total = sum(w for w, _ in spec.values())
            assert total == 100, f"Template {sk!r} : Σ={total} ≠ 100"

    def test_all_template_criterion_ids_are_known(self):
        """Tous les IDs de critères référencés dans les templates existent."""
        for sk, spec in _SECTOR_TEMPLATES.items():
            for cid in spec:
                assert cid in _CRITERION_DEFS, f"Template {sk!r} référence cid inconnu {cid!r}"

    def test_template_count(self):
        """Au moins 23 templates sectoriels."""
        assert len(_SECTOR_TEMPLATES) >= 23


# ---------------------------------------------------------------------------
# 2. Profils overrides — sommes et structure
# ---------------------------------------------------------------------------

class TestProfileOverrides:
    def test_all_overrides_sum_to_100(self):
        """Chaque override profile-spécifique somme à 100."""
        for pid, spec in _PROFILE_OVERRIDES.items():
            total = sum(w for w, _ in spec.values())
            assert total == 100, f"Override {pid!r} : Σ={total} ≠ 100"

    def test_all_override_criterion_ids_are_known(self):
        """Tous les IDs dans les overrides existent dans _CRITERION_DEFS."""
        for pid, spec in _PROFILE_OVERRIDES.items():
            for cid in spec:
                assert cid in _CRITERION_DEFS, f"Override {pid!r} : cid inconnu {cid!r}"


# ---------------------------------------------------------------------------
# 3. build_for_profile — profils clés
# ---------------------------------------------------------------------------

class TestBuildForProfile:
    def test_build_ml_engineer_weights_sum_to_100(self):
        criteria = _build("ml_engineer", "informatique_digital")
        assert sum(c.weight for c in criteria) == 100

    def test_build_ml_engineer_has_skills_tech(self):
        criteria = _build("ml_engineer", "informatique_digital")
        ids = [c.id for c in criteria]
        assert "eval_skills_tech" in ids

    def test_build_ml_engineer_skills_tech_weight_higher_than_template(self):
        """Override ml_engineer : eval_skills_tech a 25 pts (vs 20 dans le template)."""
        criteria = _build("ml_engineer", "informatique_digital")
        skill_w = next(c.weight for c in criteria if c.id == "eval_skills_tech")
        template_w = _SECTOR_TEMPLATES["informatique_digital"]["eval_skills_tech"][0]
        assert skill_w > template_w

    def test_build_vendeur_has_profile_keywords(self):
        """Commerce : eval_profile_keywords présent."""
        criteria = _build("vendeur", "commerce_distribution")
        assert any(c.id == "eval_profile_keywords" for c in criteria)

    def test_build_vendeur_profile_keywords_injected(self):
        """Les detection_keywords de vendeur sont injectés dans eval_profile_keywords."""
        profile = ALL_PROFILES["vendeur"]
        criteria = _build("vendeur", "commerce_distribution")
        kw_criterion = next(c for c in criteria if c.id == "eval_profile_keywords")
        for kw in profile.detection_keywords:
            assert kw in kw_criterion.keywords

    def test_build_vendeur_weights_sum_to_100(self):
        assert sum(c.weight for c in _build("vendeur", "commerce_distribution")) == 100

    def test_build_infirmier_formation_highest_weight(self):
        """Infirmier : eval_formation a le poids le plus élevé (diplôme d'État critique)."""
        criteria = _build("infirmier", "sante_social")
        by_weight = sorted(criteria, key=lambda c: c.weight, reverse=True)
        assert by_weight[0].id == "eval_formation"

    def test_build_infirmier_weights_sum_to_100(self):
        assert sum(c.weight for c in _build("infirmier", "sante_social")) == 100

    def test_build_chauffeur_pl_habilitations_required(self):
        """Chauffeur PL : eval_habilitations est required (permis + FIMO/FCO)."""
        criteria = _build("chauffeur_pl", "transport_logistique")
        hab = next(c for c in criteria if c.id == "eval_habilitations")
        assert hab.required is True

    def test_build_chauffeur_pl_habilitations_weight_highest(self):
        """Override chauffeur_pl : habilitations a le poids le plus élevé."""
        criteria = _build("chauffeur_pl", "transport_logistique")
        by_weight = sorted(criteria, key=lambda c: c.weight, reverse=True)
        assert by_weight[0].id == "eval_habilitations"

    def test_build_chauffeur_spl_same_structure_as_pl(self):
        """chauffeur_spl et chauffeur_pl ont les mêmes IDs de critères."""
        cpl = [c.id for c in _build("chauffeur_pl", "transport_logistique")]
        cspl = [c.id for c in _build("chauffeur_spl", "transport_logistique")]
        assert sorted(cpl) == sorted(cspl)

    def test_build_electricien_habilitations_weight_gte_25(self):
        """Électricien bâtiment : habilitations électriques ≥ 25 pts."""
        criteria = _build("electricien_batiment", "btp")
        hab = next(c for c in criteria if c.id == "eval_habilitations")
        assert hab.weight >= 25

    def test_build_electricien_habilitations_required(self):
        criteria = _build("electricien_batiment", "btp")
        hab = next(c for c in criteria if c.id == "eval_habilitations")
        assert hab.required is True

    def test_build_cuisinier_has_habilitations(self):
        """Cuisinier : eval_habilitations présent (HACCP)."""
        criteria = _build("cuisinier", "hotellerie_restauration")
        assert any(c.id == "eval_habilitations" for c in criteria)

    def test_build_cuisinier_weights_sum_to_100(self):
        assert sum(c.weight for c in _build("cuisinier", "hotellerie_restauration")) == 100

    def test_generic_profile_untouched(self):
        """build_for_profile retourne les critères existants pour non_detecte."""
        builder = CriteriaBuilder()
        result = builder.build_for_profile(GENERIC_PROFILE, "generic")
        assert len(result) == 6
        assert sum(c.weight for c in result) == 100


# ---------------------------------------------------------------------------
# 4. Intégrité du registre après init
# ---------------------------------------------------------------------------

class TestRegistryAfterInit:
    def test_all_profiles_have_criteria(self):
        """Tous les profils dans ALL_PROFILES ont des critères peuplés."""
        for pid, profile in ALL_PROFILES.items():
            assert len(profile.criteria) > 0, f"{pid}: criteria[] vide après init"

    def test_all_profiles_criteria_sum_to_100(self):
        """Tous les profils ont des poids qui totalisent 100."""
        for pid, profile in ALL_PROFILES.items():
            total = sum(c.weight for c in profile.criteria)
            assert total == 100, f"{pid}: Σ weights = {total}"

    def test_all_profiles_have_at_least_one_required(self):
        """Chaque profil a au moins un critère required."""
        for pid, profile in ALL_PROFILES.items():
            has_req = any(c.required for c in profile.criteria)
            assert has_req, f"{pid}: aucun critère required"

    def test_no_duplicate_criterion_ids_per_profile(self):
        """Aucun doublon d'ID de critère au sein d'un profil."""
        for pid, profile in ALL_PROFILES.items():
            ids = [c.id for c in profile.criteria]
            assert len(ids) == len(set(ids)), f"{pid}: doublons d'IDs : {ids}"

    def test_all_criterion_detection_fns_are_strings(self):
        """Tous les detection_fn sont des chaînes non vides."""
        for pid, profile in ALL_PROFILES.items():
            for c in profile.criteria:
                assert isinstance(c.detection_fn, str) and c.detection_fn, (
                    f"{pid}/{c.id}: detection_fn vide ou non-string"
                )

    def test_all_criterion_weights_positive(self):
        """Tous les poids sont des entiers strictement positifs."""
        for pid, profile in ALL_PROFILES.items():
            for c in profile.criteria:
                assert c.weight > 0, f"{pid}/{c.id}: weight={c.weight}"


# ---------------------------------------------------------------------------
# 5. Cas limites
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_unknown_sector_raises_valueerror(self):
        """sector_key inconnu et profil sans override → ValueError."""
        builder = CriteriaBuilder()
        fake_profile = SectorProfile(
            id="fake_job",
            sector="Secteur inconnu",
            job_title="Métier fictif",
            aliases=["Métier fictif"],
            detection_keywords=["mot-clé"],
        )
        with pytest.raises(ValueError, match="No criteria template"):
            builder.build_for_profile(fake_profile, "secteur_inexistant")

    def test_eval_profile_keywords_empty_when_no_detection_keywords(self):
        """Profile sans detection_keywords : eval_profile_keywords.keywords = []."""
        builder = CriteriaBuilder()
        profile = SectorProfile(
            id="test_no_kw",
            sector="Informatique & Digital",
            job_title="Test",
            aliases=["Test", "Test 2", "Test 3", "Test 4"],
            detection_keywords=[],
        )
        criteria = builder.build_for_profile(profile, "informatique_digital")
        kw_c = next((c for c in criteria if c.id == "eval_profile_keywords"), None)
        if kw_c is not None:
            assert kw_c.keywords == []
