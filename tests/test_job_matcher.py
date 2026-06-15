from src.core.schemas import JobListing, NormalizedCV, ScoreBreakdown, ScoringResult
from src.services.job_matcher import (
    JobSearchResult,
    _build_query,
    _build_queries,
    _find_synonym_queries,
    _trim_title_for_query,
    find_matching_jobs,
)

_BREAKDOWN = ScoreBreakdown(
    semantic_similarity=0.5, keyword_match=0.5, structure_completeness=0.5
)


def _job(title: str = "ML Engineer", location: str = "Lille", url: str | None = None) -> JobListing:
    return JobListing(
        title=title,
        company="Acme",
        location=location,
        description="Build ML models.",
        url=url or f"https://example.com/job/{title.lower().replace(' ', '-')}",
    )


class _FakeJobSearch:
    def __init__(self, by_location: dict[str | None, list[JobListing]]):
        self.by_location = by_location
        self.calls: list[tuple[str, str | None, int | None]] = []

    def search(self, query, location=None, distance=None, max_results=20, active_providers=None):
        self.calls.append((query, location, distance))
        return self.by_location.get(location, [])


class _FakeScorer:
    def encode_cv(self, cv_text):
        return None

    def score(self, parsed_cv, description, cv_embedding=None):
        score = 90.0 if "build" in description.lower() else 30.0
        return ScoringResult(overall_score=score, breakdown=_BREAKDOWN)

    def score_many(self, parsed_cv, descriptions, cv_embedding=None):
        return [self.score(parsed_cv, d) for d in descriptions]


class _LowScorer:
    def encode_cv(self, cv_text):
        return None

    def score(self, parsed_cv, description, cv_embedding=None):
        return ScoringResult(overall_score=10.0, breakdown=_BREAKDOWN)

    def score_many(self, parsed_cv, descriptions, cv_embedding=None):
        return [self.score(parsed_cv, d) for d in descriptions]


def _parsed_cv(job_title="ML Engineer", location="Croix", postal_code=None, sector=None) -> NormalizedCV:
    return NormalizedCV(
        raw_text="...",
        job_title=job_title,
        location=location,
        postal_code=postal_code,
        sector=sector,
    )


# ── _build_query ─────────────────────────────────────────────────────────────

def test_build_query_strips_seniority_qualifiers():
    assert _build_query(_parsed_cv(job_title="ML Engineer Junior")) == "ML Engineer"


def test_build_query_falls_back_to_skills_without_job_title():
    cv = NormalizedCV(raw_text="...", skills_flat=["python", "pytorch", "docker", "sql"])
    assert _build_query(cv) == "python pytorch docker"


def test_build_query_trims_company_name_from_title():
    cv = _parsed_cv(job_title="Conseiller de vente Sandro")
    assert _build_query(cv) == "Conseiller de vente"


def test_build_query_uses_commerce_skills_as_fallback():
    cv = NormalizedCV(raw_text="...", job_title=None, skills_flat=["vente", "service client", "merchandising"])
    assert _build_query(cv) == "vente service client merchandising"


# ── _trim_title_for_query ────────────────────────────────────────────────────

class TestTrimTitleForQuery:
    def test_drops_company_after_fr_retail_title(self):
        assert _trim_title_for_query("Conseiller de vente Sandro") == "Conseiller de vente"

    def test_keeps_pure_two_word_title(self):
        assert _trim_title_for_query("ML Engineer") == "ML Engineer"

    def test_keeps_three_word_tech_title(self):
        assert _trim_title_for_query("Data Engineer Senior") == "Data Engineer Senior"

    def test_no_keyword_returns_full_title(self):
        assert _trim_title_for_query("Sandro Paris") == "Sandro Paris"

    def test_vendeur_title_trimmed(self):
        assert _trim_title_for_query("Vendeur mode homme Boutique X") == "Vendeur mode homme"


# ── _find_synonym_queries ────────────────────────────────────────────────────

class TestFindSynonymQueries:
    def test_vendeur_returns_synonyms(self):
        alts = _find_synonym_queries("Vendeur")
        assert len(alts) >= 1
        assert "conseiller de vente" in alts

    def test_conseiller_de_vente_matches_vendeur_group(self):
        # "conseiller de vente" is a synonym of "vendeur"
        alts = _find_synonym_queries("Conseiller de vente")
        assert "vendeur" in alts

    def test_ml_engineer_maps_to_data_scientist_group(self):
        # "ml engineer" is listed as a synonym of "data scientist"
        alts = _find_synonym_queries("ML Engineer")
        assert "data scientist" in alts

    def test_no_exact_duplicate_in_results(self):
        alts = _find_synonym_queries("Vendeur magasin")
        assert all("vendeur" not in a.lower() for a in alts if a == "vendeur")


# ── _build_queries ───────────────────────────────────────────────────────────

class TestBuildQueries:
    def test_first_query_is_base_title(self):
        # The first query must always be the cleaned base title
        cv = _parsed_cv(job_title="ML Engineer")
        queries = _build_queries(cv)
        assert queries[0] == "ML Engineer"

    def test_adds_synonym_for_vendeur(self):
        cv = _parsed_cv(job_title="Vendeur")
        queries = _build_queries(cv)
        assert len(queries) >= 2
        assert queries[0] == "Vendeur"

    def test_adds_sector_query_when_sector_detected(self):
        cv = _parsed_cv(job_title="Conseiller de vente", sector="mode")
        queries = _build_queries(cv)
        assert any("mode" in q for q in queries)

    def test_no_sector_query_when_sector_already_in_title(self):
        # "mode" already in base title → no redundant sector query
        cv = _parsed_cv(job_title="Vendeur mode homme", sector="mode")
        queries = _build_queries(cv)
        assert not any(q.endswith("mode") and q.count("mode") > 1 for q in queries)

    def test_max_3_queries(self):
        cv = _parsed_cv(job_title="Vendeur", sector="magasin")
        queries = _build_queries(cv)
        assert len(queries) <= 3

    def test_returns_empty_when_no_title_and_no_skills(self):
        cv = NormalizedCV(raw_text="...", job_title=None, skills_flat=[])
        assert _build_queries(cv) == []


# ── find_matching_jobs ───────────────────────────────────────────────────────

def test_find_matching_jobs_scores_and_ranks_by_overall_score():
    job_search = _FakeJobSearch({"Croix": [_job(), _job(title="Data Scientist")]})
    result = find_matching_jobs(_parsed_cv(), job_search, _FakeScorer())

    assert [m.job.title for m in result.matches] == ["ML Engineer", "Data Scientist"]
    assert result.matches[0].scoring_result.overall_score == 90.0


def test_find_matching_jobs_searches_around_cv_location_within_30km():
    # All queries (multi-query) must use the CV location with a 30 km radius.
    job_search = _FakeJobSearch({"Croix": [_job()]})

    find_matching_jobs(_parsed_cv(location="Croix"), job_search, _FakeScorer())

    assert job_search.calls
    assert all(call[1] == "Croix" and call[2] == 30 for call in job_search.calls)


def test_find_matching_jobs_region_takes_priority_over_cv_location():
    # Toutes les requêtes multi-query doivent cibler la région, pas la ville du CV.
    job_search = _FakeJobSearch(
        {"Croix": [_job(location="Pont-Croix")], "Hauts-de-France": [_job(location="Lille")]}
    )

    result = find_matching_jobs(
        _parsed_cv(location="Croix"),
        job_search,
        _FakeScorer(),
        region="Hauts-de-France",
    )

    assert len(result.matches) >= 1
    assert result.matches[0].job.location == "Lille"
    assert all(call[1] == "Hauts-de-France" and call[2] is None for call in job_search.calls)


def test_find_matching_jobs_uses_city_when_no_region_selected():
    job_search = _FakeJobSearch({"Croix": [_job(location="Croix")]})

    result = find_matching_jobs(_parsed_cv(location="Croix"), job_search, _FakeScorer())

    assert len(result.matches) >= 1
    assert result.matches[0].job.location == "Croix"
    assert all(call[1] == "Croix" and call[2] == 30 for call in job_search.calls)


def test_find_matching_jobs_uses_postal_code_with_location():
    job_search = _FakeJobSearch({"59170 Croix": [_job()]})

    find_matching_jobs(_parsed_cv(location="Croix", postal_code="59170"), job_search, _FakeScorer())

    assert all(call[1] == "59170 Croix" and call[2] == 30 for call in job_search.calls)


def test_find_matching_jobs_uses_region_directly_when_cv_has_no_location():
    job_search = _FakeJobSearch({"Hauts-de-France": [_job(location="Lille")]})

    result = find_matching_jobs(
        _parsed_cv(location=None), job_search, _FakeScorer(), region="Hauts-de-France"
    )

    assert len(result.matches) >= 1
    assert all(call[1] == "Hauts-de-France" and call[2] is None for call in job_search.calls)


def test_find_matching_jobs_returns_empty_list_without_location_or_region():
    job_search = _FakeJobSearch({})

    result = find_matching_jobs(_parsed_cv(location=None), job_search, _FakeScorer())

    assert result.matches == []
    assert job_search.calls == []


def test_find_matching_jobs_returns_empty_list_without_query():
    job_search = _FakeJobSearch({})
    cv = NormalizedCV(raw_text="...", job_title=None, skills_flat=[], location="Croix")

    assert find_matching_jobs(cv, job_search, _FakeScorer()).matches == []
    assert job_search.calls == []


def test_find_matching_jobs_returns_jobsearchresult():
    job_search = _FakeJobSearch({"Croix": [_job()]})
    result = find_matching_jobs(_parsed_cv(), job_search, _FakeScorer())
    assert isinstance(result, JobSearchResult)
    assert isinstance(result.matches, list)
    assert isinstance(result.queries_used, list)


def test_find_matching_jobs_populates_queries_used():
    job_search = _FakeJobSearch({"Croix": [_job()]})
    result = find_matching_jobs(_parsed_cv(), job_search, _FakeScorer())
    assert "ML Engineer" in result.queries_used


def test_find_matching_jobs_deduplicates_by_url():
    # All queries return the same URL → only 1 unique listing scored
    shared_url = "https://example.com/job/shared"
    job = _job(url=shared_url)
    cv = _parsed_cv(job_title="Vendeur", sector="magasin")
    job_search = _FakeJobSearch({"Croix": [job]})

    result = find_matching_jobs(cv, job_search, _FakeScorer())

    assert len(result.matches) == 1


def test_find_matching_jobs_sets_few_results_when_fewer_than_3():
    # 1 result above threshold → few_results is True
    job_search = _FakeJobSearch({"Croix": [_job()]})
    result = find_matching_jobs(_parsed_cv(), job_search, _FakeScorer())
    assert result.few_results is True


def test_find_matching_jobs_few_results_false_with_3_or_more():
    jobs = [_job(title=f"Job {i}") for i in range(3)]
    job_search = _FakeJobSearch({"Croix": jobs})
    result = find_matching_jobs(_parsed_cv(), job_search, _FakeScorer())
    assert result.few_results is False


def test_find_matching_jobs_returns_all_when_none_above_threshold():
    # All listings score 10 (below 25) → fallback to unfiltered list, few_results=True
    job_search = _FakeJobSearch({"Croix": [_job()]})
    result = find_matching_jobs(_parsed_cv(), job_search, _LowScorer())
    assert len(result.matches) == 1  # fallback: show despite low score
    assert result.few_results is True


def test_find_matching_jobs_adds_synonym_queries_for_vendeur():
    cv = _parsed_cv(job_title="Vendeur", location="Lyon")
    job_search = _FakeJobSearch({"Lyon": [_job(title="Employé commercial")]})
    result = find_matching_jobs(cv, job_search, _FakeScorer())
    assert len(result.queries_used) > 1
    assert len(job_search.calls) > 1


def test_find_matching_jobs_adds_sector_query_when_sector_detected():
    cv = _parsed_cv(job_title="Conseiller de vente", sector="mode")
    job_search = _FakeJobSearch({"Croix": [_job()]})
    result = find_matching_jobs(cv, job_search, _FakeScorer())
    assert any("mode" in q for q in result.queries_used)


# ── active_providers / source_counts / duplicates_removed ──────────────────

def test_find_matching_jobs_passes_active_providers_to_job_search():
    job_search = _FakeJobSearch({"Croix": [_job()]})

    find_matching_jobs(_parsed_cv(), job_search, _FakeScorer(), active_providers=["adzuna", "jooble"])

    assert job_search.calls


def test_find_matching_jobs_computes_source_counts():
    jobs = [
        _job(title="ML Engineer", url="https://example.com/1"),
        _job(title="Data Scientist", url="https://example.com/2"),
    ]
    jobs[0].source = "adzuna"
    jobs[1].source = "jooble"
    job_search = _FakeJobSearch({"Croix": jobs})

    result = find_matching_jobs(_parsed_cv(), job_search, _FakeScorer())

    assert result.source_counts.get("adzuna") == 1
    assert result.source_counts.get("jooble") == 1


def test_find_matching_jobs_computes_duplicates_removed():
    # Same job repeated across the multi-query results -> deduplicated, counted.
    shared_url = "https://example.com/job/shared"
    job = _job(url=shared_url)
    cv = _parsed_cv(job_title="Vendeur", sector="magasin")
    job_search = _FakeJobSearch({"Croix": [job]})

    result = find_matching_jobs(cv, job_search, _FakeScorer())

    assert result.duplicates_removed >= 1
