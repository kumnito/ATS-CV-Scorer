import logging
import time
from typing import Optional

import httpx

from src.core.schemas import JobListing
from src.services.job_providers.base import JobProvider

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.adzuna.com/v1/api/jobs"


class AdzunaProvider(JobProvider):
    name = "adzuna"
    color = "#6366f1"

    def __init__(
        self,
        app_id: str = "",
        app_key: str = "",
        country: str = "fr",
        client: httpx.Client | None = None,
    ) -> None:
        self.app_id = app_id
        self.app_key = app_key
        self.country = country
        self._client = client or httpx.Client(timeout=15.0)

    def check_availability(self) -> tuple[bool, float]:
        if not self.app_id or not self.app_key:
            return False, 0.0

        start = time.monotonic()
        try:
            response = self._client.get(
                f"{_BASE_URL}/{self.country}/search/1",
                params={
                    "app_id": self.app_id,
                    "app_key": self.app_key,
                    "results_per_page": 1,
                    "content-type": "application/json",
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Adzuna availability check failed: %s", exc)
            return False, (time.monotonic() - start) * 1000

        return True, (time.monotonic() - start) * 1000

    def search(
        self,
        query: str,
        location: Optional[str] = None,
        max_results: int = 20,
        distance: Optional[int] = None,
    ) -> list[JobListing]:
        if not self.app_id or not self.app_key:
            logger.warning("Adzuna credentials missing — skipping job search.")
            return []
        if not query.strip():
            return []

        params = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "what": query,
            "results_per_page": max_results,
            "content-type": "application/json",
        }
        if location:
            params["where"] = location
            if distance is not None:
                params["distance"] = distance

        url = f"{_BASE_URL}/{self.country}/search/1"

        try:
            response = self._client.get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Adzuna search failed for query=%r: %s", query, exc)
            return []

        results = response.json().get("results", [])
        return [_to_job_listing(item) for item in results]


def _to_job_listing(item: dict) -> JobListing:
    company = item.get("company") or {}
    location = item.get("location") or {}
    return JobListing(
        title=item.get("title", "").strip(),
        company=company.get("display_name", "").strip(),
        location=location.get("display_name", "").strip(),
        description=item.get("description", "").strip(),
        url=item.get("redirect_url", ""),
        salary_min=item.get("salary_min"),
        salary_max=item.get("salary_max"),
        contract_type=item.get("contract_type"),
        source=AdzunaProvider.name,
        source_color=AdzunaProvider.color,
    )
