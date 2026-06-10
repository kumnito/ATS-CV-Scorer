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
    Recommendation,
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
# ATSReadability
# ---------------------------------------------------------------------------


def test_ats_readability_layout_single_col(full_report):
    assert full_report.ats_readability.layout == "single_column"
    assert full_report.ats_readability.layout_label == "✅ Optimal"


def test_ats_readability_layout_label_two_columns(scorer):
    cv = _make_full_cv()
    cv = cv.model_copy(update={"layout_detected": "two_columns"})
    report = scorer.score(cv)
    assert report.ats_readability.layout_label == "⚠️ Risque parseur"


def test_ats_readability_sections_found_include_experience_and_skills(full_report):
    assert "experience" in full_report.ats_readability.sections_found
    assert "skills" in full_report.ats_readability.sections_found


def test_ats_readability_sections_missing_for_minimal(minimal_report):
    for s in ["experience", "education", "skills"]:
        assert s in minimal_report.ats_readability.sections_missing


def test_ats_readability_is_machine_readable_full(full_report):
    assert full_report.ats_readability.is_machine_readable is True


def test_ats_readability_is_machine_readable_false_minimal(minimal_report):
    assert minimal_report.ats_readability.is_machine_readable is False


def test_ats_readability_extraction_method_passthrough(full_report):
    assert full_report.ats_readability.extraction_method == "pdfplumber"


# ---------------------------------------------------------------------------
# ProfileStrength — score
# ---------------------------------------------------------------------------


def test_profile_strength_score_bounded(full_report):
    assert 0 <= full_report.profile_strength.score <= 100


def test_profile_strength_level_solide_for_full_cv(full_report):
    assert full_report.profile_strength.level == "Solide"


def test_profile_strength_level_a_renforcer_for_minimal(minimal_report):
    assert minimal_report.profile_strength.level == "À renforcer"


def test_profile_strength_correct_level(scorer):
    cv = NormalizedCV(
        skills=CVSkills(ml=["pytorch"] * 10),
        experience=[CVExperience(title="Dev", date_start="2022-01", date_end="2024-01", duration_months=24)],
        raw_text="python pytorch " * 30,
        word_count=310,
    )
    report = scorer.score(cv)
    assert report.profile_strength.level in ("Correct", "Solide")
    assert report.profile_strength.score >= 50


# ---------------------------------------------------------------------------
# ProfileStrength — strengths & improvements
# ---------------------------------------------------------------------------


def test_profile_strength_strengths_nonempty_for_full(full_report):
    assert len(full_report.profile_strength.strengths) > 0


def test_profile_strength_improvements_nonempty_for_minimal(minimal_report):
    assert len(minimal_report.profile_strength.improvements) > 0


def test_profile_strength_metrics_in_strengths_for_full(full_report):
    joined = " ".join(full_report.profile_strength.strengths).lower()
    assert "métrique" in joined


def test_profile_strength_metrics_in_improvements_for_minimal(minimal_report):
    joined = " ".join(minimal_report.profile_strength.improvements).lower()
    assert "métrique" in joined or "quantifi" in joined


def test_profile_strength_skills_in_strengths_when_rich(full_report):
    joined = " ".join(full_report.profile_strength.strengths).lower()
    assert "skill" in joined or "compétence" in joined


def test_profile_strength_skills_in_improvements_when_missing(minimal_report):
    joined = " ".join(minimal_report.profile_strength.improvements).lower()
    assert "compétence" in joined


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


def test_recommendations_nonempty_for_minimal(minimal_report):
    assert len(minimal_report.recommendations) > 0


def test_recommendations_are_recommendation_objects(minimal_report):
    assert all(isinstance(r, Recommendation) for r in minimal_report.recommendations)


def test_recommendations_contain_section_advice_for_minimal(minimal_report):
    joined = " ".join(r.action for r in minimal_report.recommendations).lower()
    assert any(kw in joined for kw in ["section", "compétences", "expérience", "formation"])


def test_full_cv_fewer_recommendations_than_minimal(full_report, minimal_report):
    assert len(full_report.recommendations) < len(minimal_report.recommendations)


def test_recommendations_priority_values_valid(minimal_report):
    for r in minimal_report.recommendations:
        assert r.priority in (1, 2, 3)


def test_recommendations_impact_values_valid(minimal_report):
    for r in minimal_report.recommendations:
        assert r.impact in ("Fort", "Moyen", "Faible")


def test_two_column_layout_triggers_fort_recommendation(scorer):
    cv = _make_full_cv()
    cv = cv.model_copy(update={"layout_detected": "two_columns"})
    report = scorer.score(cv)
    fort_actions = [r.action for r in report.recommendations if r.impact == "Fort"]
    assert any("colonne" in a.lower() for a in fort_actions)


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


def test_score_handles_mm_yyyy_dates_without_crashing(scorer):
    """Le Vision LLM peut renvoyer des dates au format MM/YYYY au lieu de YYYY-MM."""
    cv = NormalizedCV(
        experience=[
            CVExperience(date_start="06/2010", date_end="06/2015", duration_months=60),
            CVExperience(date_start="2017-01", date_end="2020-01", duration_months=36),
        ],
        raw_text="x",
        word_count=1,
    )
    report = scorer.score(cv)
    assert report.career_start_year == 2010


# ---------------------------------------------------------------------------
# Action nouns FR (nouveaux signaux acceptés)
# ---------------------------------------------------------------------------


def test_action_nouns_fr_count_as_action_signals(scorer):
    cv = NormalizedCV(
        experience=[
            CVExperience(
                title="Vendeur",
                date_start="2022-01",
                date_end="2024-01",
                duration_months=24,
                bullets=["Ingestion de données clients", "Contrôle qualité produits"],
            )
        ],
        skills=CVSkills(other=["excel"] * 10),
        raw_text="Ingestion de données clients Contrôle qualité " * 5,
        word_count=310,
    )
    report = scorer.score(cv)
    # has_verbs = True (via action nouns) → pts_verbs = 10 → score >= 50
    assert report.profile_strength.score >= 50
