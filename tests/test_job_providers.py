import httpx

from src.core.schemas import JobListing
from src.services.job_providers.adzuna import AdzunaProvider
from src.services.job_providers.france_travail import (
    FranceTravailProvider,
    _extract_department_code,
    _parse_salary_min as _ft_parse_salary_min,
)
from src.services.job_providers.jooble import JoobleProvider, _parse_salary_min
from src.services.job_providers.oauth2_token_manager import OAuth2TokenManager
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

    def search(
        self, query: str, location: str | None = None, max_results: int = 20, distance: int | None = None
    ) -> list[JobListing]:
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
        def search(self, query, location=None, max_results=20, distance=None):
            raise RuntimeError("boom")

    providers = [
        _BrokenProvider("broken", []),
        _FakeProvider("alpha", [_listing("https://example.com/job/1", "alpha")]),
    ]
    orchestrator = JobSearchOrchestrator(providers)

    results = orchestrator.search(query="ML Engineer", location="Paris", active_providers=["broken", "alpha"])

    assert len(results) == 1
    assert results[0].url == "https://example.com/job/1"


# ---------------------------------------------------------------------------
# OAuth2TokenManager
# ---------------------------------------------------------------------------

_TOKEN_PAYLOAD = {"access_token": "tok-123", "expires_in": 3600, "token_type": "Bearer"}


def _token_client(payload: dict = _TOKEN_PAYLOAD, status_code: int = 200):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status_code, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def test_oauth2_token_manager_caches_token():
    client, calls = _token_client()
    manager = OAuth2TokenManager(
        client_id="id", client_secret="secret", token_url="https://auth.example/token", scope="scope"
    )
    manager._client = client

    token1 = manager.get_token()
    token2 = manager.get_token()

    assert token1 == "tok-123"
    assert token2 == "tok-123"
    assert len(calls) == 1


def test_oauth2_token_manager_refresh_always_calls_token_endpoint():
    client, calls = _token_client()
    manager = OAuth2TokenManager(
        client_id="id", client_secret="secret", token_url="https://auth.example/token", scope="scope"
    )
    manager._client = client

    manager.refresh()
    manager.refresh()

    assert len(calls) == 2


def test_oauth2_token_manager_returns_none_on_http_error():
    client, _ = _token_client(payload={}, status_code=401)
    manager = OAuth2TokenManager(
        client_id="id", client_secret="secret", token_url="https://auth.example/token", scope="scope"
    )
    manager._client = client

    assert manager.get_token() is None


# ---------------------------------------------------------------------------
# FranceTravailProvider
# ---------------------------------------------------------------------------

_FT_SEARCH_PAYLOAD = {
    "resultats": [
        {
            "intitule": "Data Scientist",
            "entreprise": {"nom": "Acme"},
            "lieuTravail": {"libelle": "Lille - 59"},
            "description": "Analyse de données et modèles ML.",
            "origineOffre": {"urlOrigine": "https://www.francetravail.fr/offres/123"},
            "salaire": {"libelle": "Mensuel de 2200.0 Euros à 2500.0 Euros"},
        }
    ]
}


def _ft_client(search_payload: dict = _FT_SEARCH_PAYLOAD, search_status: int = 200):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if "access_token" in str(request.url):
            return httpx.Response(200, json=_TOKEN_PAYLOAD)
        return httpx.Response(search_status, json=search_payload)

    return httpx.Client(transport=httpx.MockTransport(handler)), requests


def test_france_travail_provider_search_maps_results_to_job_listings():
    client, requests = _ft_client()
    provider = FranceTravailProvider(client_id="id", client_secret="secret", client=client)

    listings = provider.search(query="Data Scientist", location="59170 Croix")

    assert len(listings) == 1
    listing = listings[0]
    assert listing.title == "Data Scientist"
    assert listing.company == "Acme"
    assert listing.location == "Lille - 59"
    assert listing.url == "https://www.francetravail.fr/offres/123"
    assert listing.salary_min == 2200.0
    assert listing.source == "france_travail"
    assert listing.source_color == "#2563eb"

    search_request = requests[-1]
    assert search_request.url.params["departement"] == "59"
    assert search_request.headers["authorization"] == "Bearer tok-123"


def test_france_travail_provider_company_defaults_when_missing():
    payload = {
        "resultats": [
            {
                "intitule": "Data Scientist",
                "entreprise": {},
                "lieuTravail": {"libelle": "Lille"},
                "description": "desc",
                "origineOffre": {"urlOrigine": "https://www.francetravail.fr/offres/124"},
                "salaire": {},
            }
        ]
    }
    client, _ = _ft_client(search_payload=payload)
    provider = FranceTravailProvider(client_id="id", client_secret="secret", client=client)

    listings = provider.search(query="Data Scientist")

    assert listings[0].company == "Non précisé"
    assert listings[0].salary_min is None


def test_france_travail_provider_search_returns_empty_without_credentials():
    client, _ = _ft_client()
    provider = FranceTravailProvider(client=client)

    assert provider.search(query="Data Scientist") == []


def test_france_travail_provider_search_returns_empty_without_query():
    client, _ = _ft_client()
    provider = FranceTravailProvider(client_id="id", client_secret="secret", client=client)

    assert provider.search(query="   ") == []


def test_france_travail_provider_search_returns_empty_when_token_refresh_fails():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = FranceTravailProvider(client_id="id", client_secret="secret", client=client)

    assert provider.search(query="Data Scientist") == []


def test_france_travail_provider_search_returns_empty_on_quota_error():
    client, _ = _ft_client(search_status=429)
    provider = FranceTravailProvider(client_id="id", client_secret="secret", client=client)

    assert provider.search(query="Data Scientist") == []


def test_france_travail_provider_check_availability_success():
    client, _ = _ft_client()
    provider = FranceTravailProvider(client_id="id", client_secret="secret", client=client)

    available, latency_ms = provider.check_availability()

    assert available is True
    assert latency_ms >= 0.0


def test_france_travail_provider_check_availability_without_credentials():
    client, _ = _ft_client()
    provider = FranceTravailProvider(client=client)

    assert provider.check_availability() == (False, 0.0)


def test_france_travail_provider_check_availability_token_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = FranceTravailProvider(client_id="id", client_secret="secret", client=client)

    assert provider.check_availability() == (False, 0.0)


def test_extract_department_code_from_postal_code():
    assert _extract_department_code("59170 Croix") == "59"


def test_extract_department_code_from_department_only():
    assert _extract_department_code("59") == "59"


def test_extract_department_code_returns_none_for_region_name():
    assert _extract_department_code("Hauts-de-France") is None


def test_extract_department_code_returns_none_without_location():
    assert _extract_department_code(None) is None


def test_france_travail_parse_salary_min_extracts_first_number():
    assert _ft_parse_salary_min("Mensuel de 2200.0 Euros à 2500.0 Euros") == 2200.0
    assert _ft_parse_salary_min("") is None


def test_orchestrator_search_defaults_to_all_providers_when_active_providers_not_given():
    providers = [
        _FakeProvider("alpha", [_listing("https://example.com/job/1", "alpha")]),
        _FakeProvider("beta", [_listing("https://example.com/job/2", "beta")]),
    ]
    orchestrator = JobSearchOrchestrator(providers)

    results = orchestrator.search(query="ML Engineer", location="Paris")

    urls = {r.url for r in results}
    assert urls == {"https://example.com/job/1", "https://example.com/job/2"}
