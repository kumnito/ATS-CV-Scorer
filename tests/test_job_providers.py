import httpx

from src.core.schemas import JobListing
from src.services.job_providers.adzuna import AdzunaProvider
from src.services.job_providers.jooble import JoobleProvider, _parse_salary_min
from src.services.job_providers.orchestrator import JobSearchOrchestrator

_SAMPLE_PAYLOAD = {
    "results": [
        {
            "title": "ML Engineer",
            "company": {"display_name": "Acme Corp"},
            "location": {"display_name": "Paris, Ile-de-France"},
            "description": "Build and ship ML models in production.",
            "redirect_url": "https://www.adzuna.fr/details/123",
            "salary_min": 40000.0,
            "salary_max": 60000.0,
            "contract_type": "permanent",
        }
    ]
}


def _client_with_payload(payload: dict, status_code: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_adzuna_provider_search_tags_source():
    provider = AdzunaProvider(
        app_id="id", app_key="key", client=_client_with_payload(_SAMPLE_PAYLOAD)
    )

    listings = provider.search(query="ML Engineer", location="Paris")

    assert len(listings) == 1
    assert listings[0].source == "adzuna"
    assert listings[0].source_color == "#6366f1"


def test_adzuna_provider_check_availability_without_credentials():
    provider = AdzunaProvider(client=_client_with_payload(_SAMPLE_PAYLOAD))

    available, latency_ms = provider.check_availability()

    assert available is False
    assert latency_ms == 0.0


def test_adzuna_provider_check_availability_success():
    provider = AdzunaProvider(
        app_id="id", app_key="key", client=_client_with_payload(_SAMPLE_PAYLOAD)
    )

    available, latency_ms = provider.check_availability()

    assert available is True
    assert latency_ms >= 0.0


def test_adzuna_provider_check_availability_http_error():
    provider = AdzunaProvider(
        app_id="id", app_key="key", client=_client_with_payload({}, status_code=500)
    )

    available, _ = provider.check_availability()

    assert available is False


_JOOBLE_PAYLOAD = {
    "totalCount": 1,
    "jobs": [
        {
            "title": "ML Engineer",
            "location": "Paris, France",
            "snippet": "Build and ship ML models in production.",
            "salary": "45000 - 60000 EUR",
            "company": "Acme Corp",
            "link": "https://jooble.org/jdp/123",
            "type": "Full-time",
        }
    ],
}


def _client_with_post_payload(payload: dict, status_code: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_jooble_provider_search_maps_jobs_to_job_listings():
    provider = JoobleProvider(api_key="key", client=_client_with_post_payload(_JOOBLE_PAYLOAD))

    listings = provider.search(query="ML Engineer", location="Paris")

    assert len(listings) == 1
    listing = listings[0]
    assert listing.title == "ML Engineer"
    assert listing.company == "Acme Corp"
    assert listing.location == "Paris, France"
    assert listing.description == "Build and ship ML models in production."
    assert listing.url == "https://jooble.org/jdp/123"
    assert listing.salary_min == 45000.0
    assert listing.source == "jooble"
    assert listing.source_color == "#10b981"


def test_jooble_provider_search_returns_empty_list_without_api_key():
    provider = JoobleProvider(client=_client_with_post_payload(_JOOBLE_PAYLOAD))

    assert provider.search(query="ML Engineer") == []


def test_jooble_provider_search_returns_empty_list_without_query():
    provider = JoobleProvider(api_key="key", client=_client_with_post_payload(_JOOBLE_PAYLOAD))

    assert provider.search(query="   ") == []


def test_jooble_provider_search_returns_empty_list_on_http_error():
    provider = JoobleProvider(api_key="key", client=_client_with_post_payload({}, status_code=500))

    assert provider.search(query="ML Engineer") == []


def test_jooble_provider_check_availability_reflects_api_key_presence():
    assert JoobleProvider(api_key="key").check_availability() == (True, 0.0)
    assert JoobleProvider(api_key="").check_availability() == (False, 0.0)


def test_parse_salary_min_extracts_first_number():
    assert _parse_salary_min("45000 - 60000 EUR") == 45000.0
    assert _parse_salary_min("") is None
    assert _parse_salary_min("Negotiable") is None


class _FakeProvider:
    """Minimal JobProvider stand-in for orchestrator tests."""

    def __init__(self, name: str, listings: list[JobListing], available: tuple[bool, float] = (True, 1.0)):
        self.name = name
        self.color = "#000000"
        self._listings = listings
        self._available = available

    def check_availability(self) -> tuple[bool, float]:
        return self._available

    def search(self, query: str, location: str | None = None, max_results: int = 20) -> list[JobListing]:
        return self._listings


def _listing(url: str, source: str) -> JobListing:
    return JobListing(
        title="ML Engineer",
        company="Acme",
        location="Paris",
        description="desc",
        url=url,
        source=source,
    )


def test_orchestrator_check_all_availability():
    providers = [
        _FakeProvider("alpha", [], available=(True, 5.0)),
        _FakeProvider("beta", [], available=(False, 0.0)),
    ]
    orchestrator = JobSearchOrchestrator(providers)

    results = orchestrator.check_all_availability()

    assert results["alpha"] == (True, 5.0)
    assert results["beta"] == (False, 0.0)


def test_orchestrator_search_merges_and_dedupes_by_url():
    shared_url = "https://example.com/job/1"
    providers = [
        _FakeProvider("alpha", [_listing(shared_url, "alpha"), _listing("https://example.com/job/2", "alpha")]),
        _FakeProvider("beta", [_listing(shared_url, "beta")]),
    ]
    orchestrator = JobSearchOrchestrator(providers)

    results = orchestrator.search(query="ML Engineer", location="Paris", active_providers=["alpha", "beta"])

    urls = [r.url for r in results]
    assert urls == ["https://example.com/job/1", "https://example.com/job/2"]


def test_orchestrator_search_skips_inactive_providers():
    providers = [_FakeProvider("alpha", [_listing("https://example.com/job/1", "alpha")])]
    orchestrator = JobSearchOrchestrator(providers)

    results = orchestrator.search(query="ML Engineer", location="Paris", active_providers=["beta"])

    assert results == []


def test_orchestrator_search_handles_provider_failure_gracefully():
    class _BrokenProvider(_FakeProvider):
        def search(self, query, location=None, max_results=20):
            raise RuntimeError("boom")

    providers = [
        _BrokenProvider("broken", []),
        _FakeProvider("alpha", [_listing("https://example.com/job/1", "alpha")]),
    ]
    orchestrator = JobSearchOrchestrator(providers)

    results = orchestrator.search(query="ML Engineer", location="Paris", active_providers=["broken", "alpha"])

    assert len(results) == 1
    assert results[0].url == "https://example.com/job/1"
