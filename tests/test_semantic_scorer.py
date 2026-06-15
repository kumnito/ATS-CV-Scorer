import pytest

from src.core.schemas import NormalizedCV
from src.services.semantic_scorer import SemanticScorer


@pytest.fixture(scope="module")
def scorer() -> SemanticScorer:
    return SemanticScorer()


def _parsed_cv(raw_text: str) -> NormalizedCV:
    return NormalizedCV(
        raw_text=raw_text,
        sections={"experience": "...", "education": "...", "skills": "python docker"},
        skills_flat=["python", "docker"],
    )


def test_score_returns_overall_score_in_range(scorer):
    cv = _parsed_cv("Python developer with machine learning experience.")
    result = scorer.score(cv, "We are looking for a Python developer with ML skills.")
    assert 0 <= result.overall_score <= 100


def test_score_with_precomputed_embedding_matches_default(scorer):
    cv = _parsed_cv("Python developer with machine learning experience.")
    jd = "We are looking for a Python developer with ML skills."

    cv_embedding = scorer.encode_cv(cv.raw_text)
    result_with_cache = scorer.score(cv, jd, cv_embedding=cv_embedding)
    result_default = scorer.score(cv, jd)

    assert result_with_cache.breakdown.semantic_similarity == pytest.approx(
        result_default.breakdown.semantic_similarity
    )


def test_score_many_matches_individual_scores(scorer):
    cv = _parsed_cv("Python developer with machine learning experience.")
    descriptions = [
        "We are looking for a Python developer with ML skills.",
        "Looking for a chef de cuisine with 5 years experience.",
    ]

    cv_embedding = scorer.encode_cv(cv.raw_text)
    batch_results = scorer.score_many(cv, descriptions, cv_embedding=cv_embedding)
    individual_results = [scorer.score(cv, jd, cv_embedding=cv_embedding) for jd in descriptions]

    assert len(batch_results) == 2
    for batch, individual in zip(batch_results, individual_results):
        assert batch.breakdown.semantic_similarity == pytest.approx(
            individual.breakdown.semantic_similarity
        )
        assert batch.overall_score == pytest.approx(individual.overall_score)


def test_score_many_empty_descriptions_returns_empty_list(scorer):
    cv = _parsed_cv("Python developer.")
    assert scorer.score_many(cv, []) == []


def test_truncation_handles_text_longer_than_max_seq_length(scorer):
    long_text = " ".join(["python"] * 1000)
    cv = _parsed_cv(long_text)
    result = scorer.score(cv, "Python developer wanted.")
    assert 0 <= result.overall_score <= 100
