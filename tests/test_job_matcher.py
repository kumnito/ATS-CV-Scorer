from src.core.schemas import JobListing, ParsedCV, ScoreBreakdown, ScoringResult
from src.services.job_matcher import _build_query, find_matching_jobs

_BREAKDOWN = ScoreBreakdown(
    semantic_similarity=0.5, keyword_match=0.5, structure_completeness=0.5
)


def _job(title: str = "ML Engineer", location: str = "Lille") -> JobListing:
    return JobListing(
        title=title,
        company="Acme",
        location=location,
        description="Build ML models.",
        url="https://example.com/job",
    )


class _FakeJobSearch:
    def __init__(self, by_location: dict[str | None, list[JobListing]]):
        self.by_location = by_location
        self.calls: list[tuple[str, str | None, int | None]] = []

    def search(self, query, location=None, distance=None, max_results=20):
        self.calls.append((query, location, distance))
        return self.by_location.get(location, [])


class _FakeScorer:
    def score(self, parsed_cv, description):
        score = 90.0 if "build" in description.lower() else 30.0
        return ScoringResult(overall_score=score, breakdown=_BREAKDOWN)


def _parsed_cv(job_title="ML Engineer", location="Croix", postal_code=None) -> ParsedCV:
    return ParsedCV(raw_text="...", job_title=job_title, location=location, postal_code=postal_code)


def test_build_query_strips_seniority_qualifiers():
    assert _build_query(_parsed_cv(job_title="ML Engineer Junior")) == "ML Engineer"


def test_build_query_falls_back_to_skills_without_job_title():
    cv = ParsedCV(raw_text="...", skills=["python", "pytorch", "docker", "sql"])
    assert _build_query(cv) == "python pytorch docker"


def test_find_matching_jobs_scores_and_ranks_by_overall_score():
    job_search = _FakeJobSearch({"Croix": [_job(), _job(title="Data Scientist")]})
    matches = find_matching_jobs(_parsed_cv(), job_search, _FakeScorer())

    assert [m.job.title for m in matches] == ["ML Engineer", "Data Scientist"]
    assert matches[0].scoring_result.overall_score == 90.0


def test_find_matching_jobs_searches_around_cv_location_within_30km():
    job_search = _FakeJobSearch({"Croix": [_job()]})

    find_matching_jobs(_parsed_cv(location="Croix"), job_search, _FakeScorer())

    assert job_search.calls == [("ML Engineer", "Croix", 30)]


def test_find_matching_jobs_region_takes_priority_over_cv_location():
    # Quand une région est sélectionnée, elle est utilisée directement —
    # évite les mauvais géocodages Adzuna (ex. "Croix" → Pont-Croix/Finistère).
    job_search = _FakeJobSearch(
        {"Croix": [_job(location="Pont-Croix")], "Hauts-de-France": [_job(location="Lille")]}
    )

    matches = find_matching_jobs(
        _parsed_cv(location="Croix"),
        job_search,
        _FakeScorer(),
        region="Hauts-de-France",
    )

    assert len(matches) == 1
    assert matches[0].job.location == "Lille"
    assert job_search.calls == [("ML Engineer", "Hauts-de-France", None)]


def test_find_matching_jobs_uses_city_when_no_region_selected():
    job_search = _FakeJobSearch({"Croix": [_job(location="Croix")]})

    matches = find_matching_jobs(_parsed_cv(location="Croix"), job_search, _FakeScorer())

    assert len(matches) == 1
    assert matches[0].job.location == "Croix"
    assert job_search.calls == [("ML Engineer", "Croix", 30)]


def test_find_matching_jobs_uses_postal_code_with_location():
    job_search = _FakeJobSearch({"59170 Croix": [_job()]})

    find_matching_jobs(_parsed_cv(location="Croix", postal_code="59170"), job_search, _FakeScorer())

    assert job_search.calls == [("ML Engineer", "59170 Croix", 30)]


def test_find_matching_jobs_uses_region_directly_when_cv_has_no_location():
    job_search = _FakeJobSearch({"Hauts-de-France": [_job(location="Lille")]})

    matches = find_matching_jobs(
        _parsed_cv(location=None), job_search, _FakeScorer(), region="Hauts-de-France"
    )

    assert len(matches) == 1
    assert job_search.calls == [("ML Engineer", "Hauts-de-France", None)]


def test_find_matching_jobs_returns_empty_list_without_location_or_region():
    job_search = _FakeJobSearch({})

    matches = find_matching_jobs(_parsed_cv(location=None), job_search, _FakeScorer())

    assert matches == []
    assert job_search.calls == []


def test_find_matching_jobs_returns_empty_list_without_query():
    job_search = _FakeJobSearch({})
    cv = ParsedCV(raw_text="...", job_title=None, skills=[], location="Croix")

    assert find_matching_jobs(cv, job_search, _FakeScorer()) == []
    assert job_search.calls == []
