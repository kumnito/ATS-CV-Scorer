import httpx

from src.services.job_search import JobSearchService

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
        },
        {
            "title": "Data Scientist",
            "company": {"display_name": "DataCo"},
            "location": {"display_name": "Lyon"},
            "description": "Analyze data and build predictive models.",
            "redirect_url": "https://www.adzuna.fr/details/456",
        },
    ]
}


def _client_with_payload(payload: dict, status_code: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_search_maps_results_to_job_listings():
    svc = JobSearchService(
        app_id="id", app_key="key", client=_client_with_payload(_SAMPLE_PAYLOAD)
    )

    listings = svc.search(query="ML Engineer", location="Paris")

    assert len(listings) == 2
    first = listings[0]
    assert first.title == "ML Engineer"
    assert first.company == "Acme Corp"
    assert first.location == "Paris, Ile-de-France"
    assert first.url == "https://www.adzuna.fr/details/123"
    assert first.salary_min == 40000.0
    assert first.salary_max == 60000.0
    assert first.contract_type == "permanent"


def test_search_handles_missing_optional_fields():
    svc = JobSearchService(
        app_id="id", app_key="key", client=_client_with_payload(_SAMPLE_PAYLOAD)
    )

    listings = svc.search(query="Data Scientist")

    second = listings[1]
    assert second.salary_min is None
    assert second.salary_max is None
    assert second.contract_type is None


def test_search_returns_empty_list_without_credentials():
    svc = JobSearchService(
        app_id="", app_key="", client=_client_with_payload(_SAMPLE_PAYLOAD)
    )

    assert svc.search(query="ML Engineer") == []


def test_search_returns_empty_list_without_query():
    svc = JobSearchService(
        app_id="id", app_key="key", client=_client_with_payload(_SAMPLE_PAYLOAD)
    )

    assert svc.search(query="   ") == []


def test_search_passes_distance_param_only_when_location_and_distance_given():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_SAMPLE_PAYLOAD)

    svc = JobSearchService(
        app_id="id",
        app_key="key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    svc.search(query="ML Engineer", location="Croix", distance=30)
    svc.search(query="ML Engineer", location="Hauts-de-France")
    svc.search(query="ML Engineer")

    assert captured[0].url.params["where"] == "Croix"
    assert captured[0].url.params["distance"] == "30"
    assert "where" in captured[1].url.params
    assert "distance" not in captured[1].url.params
    assert "where" not in captured[2].url.params
    assert "distance" not in captured[2].url.params


def test_search_returns_empty_list_on_http_error():
    svc = JobSearchService(
        app_id="id", app_key="key", client=_client_with_payload({}, status_code=500)
    )

    assert svc.search(query="ML Engineer") == []
