import pytest

from src.services.nlp_pipeline import (
    NLPPipeline,
    _clean_text,
    _detect_sections,
    _estimate_experience_years,
    _extract_skills,
)

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
