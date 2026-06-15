"""Backward-compatible wrapper around AdzunaProvider.

Existing callers (job_matcher, api/server, app.py) construct
``JobSearchService(app_id=..., app_key=..., country=..., client=...)`` and call
``.search(query, location, distance, max_results)`` — this delegates to
``AdzunaProvider`` so the Provider pattern lives in one place.
"""

from typing import Optional

import httpx

from src.core.schemas import JobListing
from src.services.job_providers.adzuna import AdzunaProvider, _to_job_listing, _BASE_URL  # noqa: F401

__all__ = ["JobSearchService", "_to_job_listing", "_BASE_URL"]


class JobSearchService:
    def __init__(
        self,
        app_id: str = "",
        app_key: str = "",
        country: str = "fr",
        client: httpx.Client | None = None,
    ) -> None:
        self._provider = AdzunaProvider(app_id=app_id, app_key=app_key, country=country, client=client)

    def search(
        self,
        query: str,
        location: Optional[str] = None,
        distance: Optional[int] = None,
        max_results: int = 20,
    ) -> list[JobListing]:
        return self._provider.search(
            query=query, location=location, distance=distance, max_results=max_results
        )
