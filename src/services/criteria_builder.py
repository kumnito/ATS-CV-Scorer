"""Builds sector-specific criteria lists for SectorProfile objects.

Usage in sector_registry.py (called once at module load):
    builder = CriteriaBuilder()
    for sector_key, pids in SECTORS.items():
        for pid in pids:
            ALL_PROFILES[pid].criteria = builder.build_for_profile(ALL_PROFILES[pid], sector_key)

CriteriaBuilder never calls ESCO at runtime. Detection keywords from each
profile are auto-injected into the 'eval_profile_keywords' criterion when
present in the sector template.
"""

from src.core.sector_profiles import Criterion, SectorProfile

# ---------------------------------------------------------------------------
# Criterion definitions — (label, detection_fn, default_keywords)
# ---------------------------------------------------------------------------

_CRITERION_DEFS: dict[str, tuple[str, str, list[str]]] = {
    "eval_experience":       ("Expériences professionnelles",       "has_experience",        ["expérience", "emploi", "poste"]),
    "eval_formation":        ("Formation académique",                "has_education",         ["formation", "diplôme", "école"]),
    "eval_summary":          ("Accroche / profil",                   "has_summary",           ["profil", "accroche", "objectif"]),
    "eval_dates":            ("Datation des expériences",            "has_dates",             ["date", "période", "depuis"]),
    "eval_word_count":       ("Volume du CV (≥ 300 mots)",           "has_sufficient_words",  []),
    "eval_contact":          ("Coordonnées complètes",               "has_contact",           ["email", "téléphone"]),
    "eval_skills_tech":      ("Compétences techniques",              "has_tech_skills",       []),
    "eval_skills_sector":    ("Compétences sectorielles",            "has_sector_skills",     []),
    "eval_metrics":          ("Résultats chiffrés",                  "has_metrics",           []),
    "eval_projects":         ("Projets réalisés",                    "has_projects",          []),
    "eval_profile_keywords": ("Mots-clés métier",                    "has_profile_keywords",  []),
    "eval_habilitations":    ("Habilitations & certifications",      "has_habilitations",     ["habilitation", "certification", "permis"]),
    "eval_languages":        ("Langues étrangères",                  "has_languages",         ["anglais", "espagnol", "allemand"]),
}

# ---------------------------------------------------------------------------
# Sector templates — sector_key → {criterion_id: (weight, required)}
# Each template must sum to 100.
# ---------------------------------------------------------------------------

_SECTOR_TEMPLATES: dict[str, dict[str, tuple[int, bool]]] = {
    "industrie_manufacturiere": {
        "eval_experience":    (25, True),
        "eval_habilitations": (20, True),
        "eval_formation":     (20, True),
        "eval_skills_sector": (15, False),
        "eval_dates":         (10, False),
        "eval_word_count":    (10, False),
    },
    "btp": {
        "eval_experience":    (25, True),
        "eval_habilitations": (20, True),
        "eval_formation":     (20, True),
        "eval_skills_sector": (20, False),
        "eval_dates":         (15, False),
    },
    "agroalimentaire": {
        "eval_experience":    (25, True),
        "eval_formation":     (20, True),
        "eval_habilitations": (20, True),
        "eval_skills_sector": (20, False),
        "eval_dates":         (15, False),
    },
    "energie_environnement": {
        "eval_experience":    (20, True),
        "eval_formation":     (20, True),
        "eval_habilitations": (25, True),
        "eval_skills_tech":   (20, False),
        "eval_dates":         (15, False),
    },
    "commerce_distribution": {
        "eval_experience":        (25, True),
        "eval_formation":         (15, True),
        "eval_skills_sector":     (20, True),
        "eval_profile_keywords":  (20, False),
        "eval_metrics":           (15, False),
        "eval_word_count":        (5,  False),
    },
    "transport_logistique": {
        "eval_experience":    (25, True),
        "eval_habilitations": (25, True),
        "eval_formation":     (15, True),
        "eval_dates":         (20, False),
        "eval_skills_sector": (15, False),
    },
    "hotellerie_restauration": {
        "eval_experience":    (30, True),
        "eval_formation":     (20, True),
        "eval_skills_sector": (20, False),
        "eval_languages":     (15, False),
        "eval_word_count":    (15, False),
    },
    "sante_social": {
        "eval_formation":     (30, True),
        "eval_experience":    (20, True),
        "eval_habilitations": (20, True),
        "eval_skills_sector": (15, False),
        "eval_summary":       (15, False),
    },
    "education_formation": {
        "eval_formation":     (30, True),
        "eval_experience":    (25, True),
        "eval_summary":       (20, False),
        "eval_skills_sector": (15, False),
        "eval_metrics":       (10, False),
    },
    "securite": {
        "eval_experience":    (25, True),
        "eval_habilitations": (30, True),
        "eval_formation":     (20, True),
        "eval_dates":         (15, False),
        "eval_word_count":    (10, False),
    },
    "nettoyage_services": {
        "eval_experience":    (30, True),
        "eval_habilitations": (20, True),
        "eval_formation":     (20, True),
        "eval_skills_sector": (15, False),
        "eval_dates":         (15, False),
    },
    "coiffure_esthetique": {
        "eval_formation":     (30, True),
        "eval_experience":    (25, True),
        "eval_skills_sector": (25, False),
        "eval_summary":       (10, False),
        "eval_word_count":    (10, False),
    },
    "immobilier": {
        "eval_experience":    (25, True),
        "eval_habilitations": (20, True),
        "eval_formation":     (15, True),
        "eval_metrics":       (20, False),
        "eval_summary":       (10, False),
        "eval_word_count":    (10, False),
    },
    "banque_assurance": {
        "eval_experience":    (25, True),
        "eval_formation":     (25, True),
        "eval_habilitations": (15, True),
        "eval_metrics":       (20, False),
        "eval_summary":       (15, False),
    },
    "finance_comptabilite": {
        "eval_experience":    (25, True),
        "eval_formation":     (25, True),
        "eval_habilitations": (10, False),
        "eval_skills_tech":   (20, False),
        "eval_metrics":       (20, False),
    },
    "rh_recrutement": {
        "eval_experience":    (25, True),
        "eval_formation":     (20, True),
        "eval_summary":       (20, False),
        "eval_skills_sector": (20, False),
        "eval_metrics":       (15, False),
    },
    "juridique_administratif": {
        "eval_experience":    (25, True),
        "eval_formation":     (30, True),
        "eval_skills_sector": (20, False),
        "eval_summary":       (15, False),
        "eval_word_count":    (10, False),
    },
    "marketing_communication": {
        "eval_experience":    (25, True),
        "eval_formation":     (15, True),
        "eval_projects":      (20, False),
        "eval_metrics":       (20, False),
        "eval_skills_tech":   (10, False),
        "eval_summary":       (10, False),
    },
    "informatique_digital": {
        "eval_experience":        (20, True),
        "eval_formation":         (15, True),
        "eval_skills_tech":       (20, True),
        "eval_projects":          (15, False),
        "eval_profile_keywords":  (15, False),
        "eval_metrics":           (10, False),
        "eval_summary":           (5,  False),
    },
    "tourisme_loisirs": {
        "eval_experience":    (25, True),
        "eval_formation":     (20, True),
        "eval_languages":     (25, False),
        "eval_skills_sector": (20, False),
        "eval_summary":       (10, False),
    },
    "sport": {
        "eval_formation":     (30, True),
        "eval_experience":    (25, True),
        "eval_habilitations": (20, True),
        "eval_skills_sector": (15, False),
        "eval_summary":       (10, False),
    },
    "culture_medias": {
        "eval_experience":    (25, True),
        "eval_projects":      (25, False),
        "eval_formation":     (15, True),
        "eval_skills_tech":   (20, False),
        "eval_summary":       (15, False),
    },
    "artisanat": {
        "eval_experience":    (30, True),
        "eval_formation":     (25, True),
        "eval_habilitations": (20, True),
        "eval_skills_sector": (15, False),
        "eval_word_count":    (10, False),
    },
}

# ---------------------------------------------------------------------------
# Profile-specific overrides — applied instead of the sector template
# ---------------------------------------------------------------------------

_PROFILE_OVERRIDES: dict[str, dict[str, tuple[int, bool]]] = {
    "ml_engineer": {
        "eval_experience":        (20, True),
        "eval_formation":         (15, True),
        "eval_skills_tech":       (25, True),
        "eval_projects":          (20, False),
        "eval_profile_keywords":  (10, False),
        "eval_metrics":           (5,  False),
        "eval_summary":           (5,  False),
    },
    "infirmier": {
        "eval_formation":     (35, True),
        "eval_experience":    (25, True),
        "eval_habilitations": (20, True),
        "eval_skills_sector": (10, False),
        "eval_summary":       (10, False),
    },
    "chauffeur_pl": {
        "eval_experience":    (20, True),
        "eval_habilitations": (35, True),
        "eval_formation":     (15, True),
        "eval_dates":         (20, False),
        "eval_skills_sector": (10, False),
    },
    "chauffeur_spl": {
        "eval_experience":    (20, True),
        "eval_habilitations": (35, True),
        "eval_formation":     (15, True),
        "eval_dates":         (20, False),
        "eval_skills_sector": (10, False),
    },
    "electricien_batiment": {
        "eval_experience":    (25, True),
        "eval_habilitations": (30, True),
        "eval_formation":     (20, True),
        "eval_skills_sector": (15, False),
        "eval_dates":         (10, False),
    },
    "cuisinier": {
        "eval_experience":    (30, True),
        "eval_formation":     (25, True),
        "eval_skills_sector": (25, False),
        "eval_habilitations": (10, True),
        "eval_word_count":    (10, False),
    },
}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class CriteriaBuilder:
    """Builds sector-appropriate Criterion lists for SectorProfile objects."""

    def build_for_profile(self, profile: SectorProfile, sector_key: str) -> list[Criterion]:
        """Return a criteria list for *profile*.

        Profile overrides take priority over sector templates.
        Auto-injects profile.detection_keywords into the 'eval_profile_keywords'
        criterion (if present in the spec).
        """
        if profile.id == "non_detecte":
            # GENERIC_PROFILE is hardcoded inline in sector_registry.py
            return profile.criteria

        spec = _PROFILE_OVERRIDES.get(profile.id) or _SECTOR_TEMPLATES.get(sector_key)
        if spec is None:
            raise ValueError(
                f"No criteria template for sector_key={sector_key!r} (profile {profile.id!r})"
            )

        criteria: list[Criterion] = []
        for cid, (weight, required) in spec.items():
            if cid not in _CRITERION_DEFS:
                raise ValueError(f"Unknown criterion id {cid!r}")
            label, detection_fn, default_kw = _CRITERION_DEFS[cid]
            kw = list(profile.detection_keywords) if cid == "eval_profile_keywords" else list(default_kw)
            criteria.append(Criterion(
                id=cid,
                label=label,
                weight=weight,
                required=required,
                detection_fn=detection_fn,
                keywords=kw,
            ))

        total = sum(c.weight for c in criteria)
        if total != 100:
            raise ValueError(
                f"Profile {profile.id!r}: criteria weights sum to {total}, expected 100"
            )

        return criteria
