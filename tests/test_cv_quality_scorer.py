"""Tests for CVQualityScorer."""

import pytest

from src.core.schemas import (
    CVEducation,
    CVExperience,
    CVHeader,
    CVProject,
    CVQualityReport,
    CVSkills,
    NormalizedCV,
)
from src.services.cv_quality_scorer import CVQualityScorer


def _make_full_cv() -> NormalizedCV:
    return NormalizedCV(
        header=CVHeader(
            name="Alice Dupont",
            title="ML Engineer",
            email="alice@example.com",
            phone="+33 6 12 34 56 78",
            location="Paris",
            github="github.com/alicedupont",
        ),
        summary="ML Engineer with 3 years of experience in NLP and MLOps.",
        skills=CVSkills(
            ml=["pytorch", "scikit-learn", "transformers"],
            mlops=["docker", "mlflow", "airflow"],
            cloud=["aws", "gcp"],
            languages=["python", "sql"],
            data=["pandas", "postgresql"],
            other=["git"],
        ),
        experience=[
            CVExperience(
                title="ML Engineer",
                company="TechCorp",
                period="2021 - 2024",
                date_start="2021-01",
                date_end="2024-01",
                duration_months=36,
                is_current=False,
                years=3.0,
                bullets=["Built NLP pipelines with 30% accuracy improvement", "Deployed models serving 10k users/day"],
            ),
            CVExperience(
                title="Data Scientist",
                company="StartupXYZ",
                period="2019 - 2021",
                date_start="2019-06",
                date_end="2021-01",
                duration_months=19,
                is_current=False,
                years=1.6,
                bullets=["Reduced inference latency by 40%"],
            ),
        ],
        education=[
            CVEducation(
                degree="MSc Machine Learning",
                school="Université Paris Saclay",
                year="2019",
                date_start="2017-09",
                date_end="2019-06",
                duration_months=21,
            )
        ],
        projects=[
            CVProject(
                name="Open Source NLP Toolkit",
                description="Text classification library",
                stack=["python", "pytorch"],
                url="github.com/alicedupont/nlp-toolkit",
                metrics=["500+ stars", "10k downloads"],
            )
        ],
        raw_text="Alice Dupont ML Engineer python pytorch scikit-learn docker aws mlflow airflow " * 20,
        layout_detected="single_column",
        word_count=600,
    )


def _make_minimal_cv() -> NormalizedCV:
    return NormalizedCV(
        header=CVHeader(name="Bob"),
        raw_text="Bob developer python",
        layout_detected="single_column",
        word_count=3,
    )


@pytest.fixture(scope="module")
def scorer():
    return CVQualityScorer()


@pytest.fixture(scope="module")
def full_report(scorer):
    return scorer.score(_make_full_cv())


@pytest.fixture(scope="module")
def minimal_report(scorer):
    return scorer.score(_make_minimal_cv())


# ---------------------------------------------------------------------------
# Schema validity
# ---------------------------------------------------------------------------


def test_full_report_is_quality_report(full_report):
    assert isinstance(full_report, CVQualityReport)


def test_minimal_report_is_quality_report(minimal_report):
    assert isinstance(minimal_report, CVQualityReport)


# ---------------------------------------------------------------------------
# score_global formula: 40% structure + 60% contenu
# ---------------------------------------------------------------------------


def test_score_global_formula(scorer):
    cv = _make_full_cv()
    report = scorer.score(cv)
    expected = round(report.score_structure * 0.40 + report.score_contenu * 0.60)
    assert report.score_global == expected


def test_score_global_bounded(full_report):
    assert 0 <= full_report.score_global <= 100


def test_score_structure_bounded(full_report):
    assert 0 <= full_report.score_structure <= 100


def test_score_contenu_bounded(full_report):
    assert 0 <= full_report.score_contenu <= 100


def test_has_metrics_true_when_bullets_quantified(full_report):
    assert full_report.has_metrics is True


def test_has_metrics_false_when_no_metrics(minimal_report):
    assert minimal_report.has_metrics is False


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def test_sections_detectees_include_experience(full_report):
    assert "experience" in full_report.sections_detectees


def test_sections_detectees_include_skills(full_report):
    assert "skills" in full_report.sections_detectees


def test_sections_manquantes_reflect_absent_sections(minimal_report):
    for s in ["experience", "education", "skills"]:
        assert s in minimal_report.sections_manquantes


# ---------------------------------------------------------------------------
# word_count passthrough
# ---------------------------------------------------------------------------


def test_word_count_matches_input(full_report):
    assert full_report.word_count == 600


def test_word_count_minimal(minimal_report):
    assert minimal_report.word_count == 3


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


def test_recommendations_nonempty_for_minimal_cv(minimal_report):
    assert len(minimal_report.recommendations) > 0


def test_recommendations_contain_sections_advice_for_minimal(minimal_report):
    joined = " ".join(minimal_report.recommendations).lower()
    assert "section" in joined or "compétences" in joined or "expérience" in joined


def test_full_cv_fewer_recommendations_than_minimal(full_report, minimal_report):
    assert len(full_report.recommendations) < len(minimal_report.recommendations)


# ---------------------------------------------------------------------------
# Career stats
# ---------------------------------------------------------------------------


def test_total_experience_years(full_report):
    # 36 months + 19 months = 55 months ≈ 4.6 years
    assert full_report.total_experience_years > 0


def test_career_start_year(full_report):
    assert full_report.career_start_year == 2019


def test_most_recent_role(full_report):
    assert full_report.most_recent_role is not None
    assert "TechCorp" in full_report.most_recent_role or "ML Engineer" in full_report.most_recent_role


def test_career_gaps_empty_for_continuous_career(scorer):
    cv = NormalizedCV(
        experience=[
            CVExperience(date_start="2022-01", date_end="2024-01", duration_months=24),
            CVExperience(date_start="2020-01", date_end="2022-01", duration_months=24),
        ],
        raw_text="x",
        word_count=1,
    )
    report = scorer.score(cv)
    assert report.career_gaps == []


def test_career_gaps_detected_for_long_gap(scorer):
    cv = NormalizedCV(
        experience=[
            CVExperience(date_start="2022-01", date_end="2024-01", duration_months=24),
            CVExperience(date_start="2015-01", date_end="2017-01", duration_months=24),
        ],
        raw_text="x",
        word_count=1,
    )
    report = scorer.score(cv)
    assert len(report.career_gaps) == 1
    assert "2017" in report.career_gaps[0]
