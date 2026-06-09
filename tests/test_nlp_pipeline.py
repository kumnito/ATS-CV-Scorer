import pytest

from src.services.nlp_pipeline import (
    NLPPipeline,
    _clean_text,
    _detect_sections,
    _estimate_experience_years,
    _extract_job_title,
    _extract_location,
    _extract_sector,
    _extract_skills,
)
from src.core.schemas import NormalizedCV, CVExperience

SAMPLE_CV = """John Doe
john@example.com | +33 6 00 00 00 00

SUMMARY
Experienced ML Engineer with 5 years in NLP and computer vision.

EXPERIENCE
Senior ML Engineer — TechCorp  2020 - Present
- Built NLP pipelines using Python and PyTorch
- Deployed models on AWS with Docker and Kubernetes

ML Engineer — StartupXYZ  2018 - 2020
- Developed scikit-learn classification models
- Automated pipelines with Apache Airflow

EDUCATION
MSc Computer Science — Université Paris Saclay  2017 - 2018

SKILLS
Python, PyTorch, TensorFlow, scikit-learn, Docker, Kubernetes, AWS, SQL, Git
"""


def test_clean_text_removes_extra_whitespace():
    dirty = "hello   world\n\n\n\nfoo"
    result = _clean_text(dirty)
    assert "   " not in result
    assert "\n\n\n" not in result


def test_detect_sections_finds_experience():
    sections = _detect_sections(SAMPLE_CV)
    assert "experience" in sections
    assert "TechCorp" in sections["experience"]


def test_detect_sections_finds_skills():
    sections = _detect_sections(SAMPLE_CV)
    assert "skills" in sections
    assert "Python" in sections["skills"]


def test_detect_sections_finds_education():
    sections = _detect_sections(SAMPLE_CV)
    assert "education" in sections


def test_extract_skills_finds_python():
    skills = _extract_skills(SAMPLE_CV)
    assert "python" in skills
    assert "pytorch" in skills
    assert "docker" in skills


def test_estimate_experience_years():
    exp_text = "ML Engineer — TechCorp  2018 - 2022\nData Scientist 2016 - 2018"
    years = _estimate_experience_years(exp_text)
    assert years == 6.0


def test_estimate_experience_years_with_present():
    from datetime import datetime

    current = datetime.now().year
    exp_text = "Engineer 2020 - Present"
    years = _estimate_experience_years(exp_text)
    assert years == float(current - 2020)


def test_extract_job_title_finds_title_with_seniority():
    header = "Alex Martin\nML Engineer Junior - Paris, France\nalex@example.com | +33 6 00 00 00 00"
    assert _extract_job_title(header) == "ML Engineer Junior"


def test_extract_job_title_skips_person_name_line():
    header = "Alex Martin\nFull-Stack Developer | Lyon\nalex@example.com"
    assert _extract_job_title(header) == "Full-Stack Developer"


def test_extract_job_title_returns_none_without_keyword():
    header = "Alex Martin\nalex@example.com | +33 6 00 00 00 00"
    assert _extract_job_title(header) is None


def test_extract_location_prefers_entity_appearing_first_in_header():
    header = "Alex Martin\nML Engineer Junior - Paris, France"
    assert _extract_location(header, {"locations": ["France", "Paris"]}) == "Paris"


def test_extract_location_falls_back_to_first_global_entity():
    assert _extract_location("", {"locations": ["Lyon", "Paris"]}) == "Lyon"


def test_extract_location_returns_none_without_entities():
    assert _extract_location("Alex Martin\nDeveloper", {"locations": []}) is None


def test_extract_location_strips_noisy_lowercase_prefix_from_entity_span():
    # en_core_web_sm (an English model) sometimes folds a stray French
    # lowercase word into a GPE span on non-English text, e.g. tagging
    # "opérationnelle - Paris" as a single entity. Only the capitalized
    # (proper-noun) run is a genuine place name.
    header = "Alex Martin\nIngénieure Machine Learning opérationnelle - Paris, France"
    assert (
        _extract_location(header, {"locations": ["opérationnelle - Paris", "France"]})
        == "Paris"
    )


def test_extract_location_discards_entity_without_capitalized_run():
    header = "Alex Martin\nDeveloper opérationnelle"
    assert _extract_location(header, {"locations": ["opérationnelle"]}) is None


def test_extract_location_does_not_swallow_following_line():
    # The capitalized-word run must stop at the end of the line — otherwise
    # it bleeds into whatever follows the address (e.g. a job-title line
    # starting with a capital letter), producing "Croix Ingénieur Machine
    # Learning" instead of just "Croix".
    header = (
        "Alex Martin\n"
        "314/1 rue Jean Jaurès\n"
        "59170, CROIX\n"
        "Ingénieur Machine Learning - à la recherche d'un poste"
    )
    assert _extract_location(header, {"locations": []}) == "Croix"


def test_extract_location_prefers_postal_address_over_noisy_ner_entities():
    # French CVs commonly carry a full postal address in the header. This is
    # a far more reliable signal than spaCy's GPE/LOC entities, which
    # en_core_web_sm (an English model) misreads on French text — e.g.
    # tagging the job-title word "DÉVELOPPEUR" as a place.
    header = "Alex Martin\nDÉVELOPPEUR FULL-STACK\n314/1 rue Jean Jaurès\n59170, CROIX"
    assert (
        _extract_location(header, {"locations": ["DÉVELOPPEUR", "France"]}) == "Croix"
    )


# ---------------------------------------------------------------------------
# Bug fixes — regression tests (2026-06-09)
# ---------------------------------------------------------------------------


def test_extract_location_filters_language_names():
    # "Anglais" is classified as GPE by en_core_web_sm on French CVs — must be blocked.
    assert _extract_location("Keo PEN\npen.keoh@gmail.com", {"locations": ["Anglais"]}) is None


def test_extract_location_filters_soft_skill_words():
    assert _extract_location("Jane Doe\nDeveloper", {"locations": ["Autonome"]}) is None


def test_extract_location_filters_multiple_blocked_keeps_real_city():
    result = _extract_location(
        "Jane Doe\nDeveloper - Paris",
        {"locations": ["Anglais", "Français", "Paris"]},
    )
    assert result == "Paris"


def test_extract_job_title_finds_french_retail_title():
    header = "Keo PEN\nVendeur Conseiller client\npen.keoh@gmail.com"
    assert _extract_job_title(header) == "Vendeur Conseiller client"


def test_extract_job_title_finds_conseiller():
    header = "Marie Dupont\nConseillère commerciale\nmarie@example.com"
    assert _extract_job_title(header) == "Conseillère commerciale"


def test_detect_sections_maps_qualite_to_skills():
    cv = "Keo PEN\n\nEXPERIENCE\nVendeur SANDRO 2024\n\nQUALITÉ\nAutonome\nRigoureux\n"
    sections = _detect_sections(cv)
    assert "skills" in sections
    assert "Autonome" in sections["skills"]


def test_detect_sections_maps_passion_to_interests():
    cv = "Keo PEN\n\nEXPERIENCE\nVendeur SANDRO 2024\n\nPASSION\nMusique\nSport\n"
    sections = _detect_sections(cv)
    assert "interests" in sections
    assert "Musique" in sections["interests"]


def test_detect_sections_passion_not_in_experience():
    cv = "Keo PEN\n\nEXPERIENCE\nVendeur SANDRO 2024\n\nPASSION\nMusique\nSport\n"
    sections = _detect_sections(cv)
    assert "Musique" not in sections.get("experience", "")


@pytest.fixture(scope="module")
def pipeline():
    return NLPPipeline()


def test_parse_cv_returns_parsed_cv(pipeline):
    parsed = pipeline.parse_cv(SAMPLE_CV)
    assert parsed.raw_text
    assert "experience" in parsed.sections
    assert "python" in parsed.skills
    assert parsed.experience_years is not None
    assert parsed.experience_years > 0


def test_parse_cv_extracts_job_title_and_location(pipeline):
    cv_text = (
        "Alex Martin\n"
        "ML Engineer Junior - Paris, France\n"
        "alex.martin@example.com | +33 6 12 34 56 78\n\n" + SAMPLE_CV
    )
    parsed = pipeline.parse_cv(cv_text)
    assert parsed.job_title == "ML Engineer Junior"
    assert parsed.location == "Paris"


# ---------------------------------------------------------------------------
# Proposition A — fallback experience[0].title dans parse_normalized
# ---------------------------------------------------------------------------

def test_parse_normalized_uses_experience_title_when_header_title_missing(pipeline):
    """Quand header.title est None, le titre de la première expérience est utilisé."""
    from src.core.schemas import CVExperience, CVHeader, NormalizedCV

    cv = NormalizedCV(
        header=CVHeader(name="Keo PEN"),  # pas de title
        experience=[
            CVExperience(title="Conseiller de vente", company="Sandro"),
        ],
        raw_text="Conseiller de vente Sandro service client vente",
    )
    parsed = pipeline.parse_normalized(cv)
    assert parsed.job_title == "Conseiller de vente"


def test_parse_normalized_prefers_header_title_over_experience_title(pipeline):
    """header.title prend la priorité sur experience[0].title si défini."""
    from src.core.schemas import CVExperience, CVHeader, NormalizedCV

    cv = NormalizedCV(
        header=CVHeader(name="Jean Dupont", title="Data Analyst"),
        experience=[
            CVExperience(title="Développeur logiciel", company="Acme"),
        ],
        raw_text="Data Analyst python pandas sql",
    )
    parsed = pipeline.parse_normalized(cv)
    assert parsed.job_title == "Data Analyst"


# ---------------------------------------------------------------------------
# Proposition C — détection des compétences commerce
# ---------------------------------------------------------------------------

def test_extract_skills_detects_retail_commerce_skills():
    """Les skills du domaine commerce (vente, service client…) sont détectées."""
    retail_text = (
        "Conseiller de vente — Boutique mode\n"
        "Accueil clientèle, vente, service client, encaissement, inventaire.\n"
        "Maîtrise du merchandising et fidélisation des clients."
    )
    skills = _extract_skills(retail_text)
    assert "vente" in skills
    assert "service client" in skills
    assert "merchandising" in skills
    assert "encaissement" in skills


def test_parse_normalized_includes_commerce_skills_for_retail_cv(pipeline):
    """Un CV retail doit avoir des compétences commerce dans ParsedCV.skills."""
    from src.core.schemas import CVExperience, CVHeader, CVSkills, NormalizedCV

    cv = NormalizedCV(
        header=CVHeader(name="Keo PEN", title="Conseiller de vente"),
        experience=[CVExperience(title="Conseiller de vente", company="Sandro")],
        skills=CVSkills(commerce=["vente", "service client", "merchandising"]),
        raw_text="vente service client merchandising encaissement",
    )
    parsed = pipeline.parse_normalized(cv)
    assert "vente" in parsed.skills
    assert "service client" in parsed.skills


# ---------------------------------------------------------------------------
# _extract_sector
# ---------------------------------------------------------------------------

def test_extract_sector_from_experience_keywords():
    cv = NormalizedCV(
        experience=[CVExperience(title="Conseiller de vente", bullets=["gestion du magasin", "stock"])]
    )
    assert _extract_sector(cv) == "magasin"


def test_extract_sector_mode_from_company_context():
    cv = NormalizedCV(
        experience=[CVExperience(title="Vendeur", company="Sandro", bullets=["collection mode", "textile"])]
    )
    assert _extract_sector(cv) == "mode"


def test_extract_sector_transport_from_bullets():
    cv = NormalizedCV(
        experience=[CVExperience(title="Chauffeur", bullets=["livraison de colis", "transport routier"])]
    )
    assert _extract_sector(cv) == "transport"


def test_extract_sector_returns_none_when_no_match():
    cv = NormalizedCV(
        experience=[CVExperience(title="ML Engineer", company="Google", bullets=["deep learning", "python"])]
    )
    assert _extract_sector(cv) is None


def test_extract_sector_returns_none_with_no_experience():
    assert _extract_sector(NormalizedCV()) is None


def test_parse_normalized_sets_sector_field(pipeline):
    from src.core.schemas import CVHeader, CVSkills

    cv = NormalizedCV(
        header=CVHeader(name="Test", title="Vendeur"),
        experience=[CVExperience(title="Vendeur", company="Auchan", bullets=["grande distribution", "magasin"])],
        raw_text="Vendeur Auchan magasin grande distribution",
    )
    parsed = pipeline.parse_normalized(cv)
    assert parsed.sector == "magasin"
