import httpx

from src.core.schemas import JobListing
from src.services.job_providers.adzuna import AdzunaProvider
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
