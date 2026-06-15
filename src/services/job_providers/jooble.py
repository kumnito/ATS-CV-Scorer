import logging
import re
from typing import Optional

import httpx

from src.core.schemas import JobListing
from src.services.job_providers.base import JobProvider

logger = logging.getLogger(__name__)

_BASE_URL = "https://jooble.org/api/"

_SALARY_NUMBER_RE = re.compile(r"[\d][\d\s.,]*")


class JoobleProvider(JobProvider):
    name = "jooble"
    color = "#10b981"

    def __init__(
        self,
        api_key: str = "",
        country: str = "fr",
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.country = country
        self._client = client or httpx.Client(timeout=15.0)

    def check_availability(self) -> tuple[bool, float]:
        # Jooble has no dedicated health endpoint — availability is whether a key is configured.
        return bool(self.api_key), 0.0

    def search(
        self,
        query: str,
        location: Optional[str] = None,
        max_results: int = 20,
    ) -> list[JobListing]:
        if not self.api_key:
            logger.warning("Jooble API key missing — skipping job search.")
            return []
        if not query.strip():
            return []

        payload: dict = {"keywords": query}
        if location:
            payload["location"] = location

        url = f"{_BASE_URL}{self.api_key}"

        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Jooble search failed for query=%r: %s", query, exc)
            return []

        jobs = response.json().get("jobs", [])
        return [_to_job_listing(item) for item in jobs[:max_results]]


def _parse_salary_min(salary: str) -> Optional[float]:
    if not salary:
        return None
    match = _SALARY_NUMBER_RE.search(salary)
    if not match:
        return None
    number = match.group(0).strip().replace(" ", "").replace(",", "").replace(".", "")
    try:
        return float(number)
    except ValueError:
        return None


def _to_job_listing(item: dict) -> JobListing:
    return JobListing(
        title=(item.get("title") or "").strip(),
        company=(item.get("company") or "").strip(),
        location=(item.get("location") or "").strip(),
        description=(item.get("snippet") or "").strip(),
        url=item.get("link", ""),
        salary_min=_parse_salary_min(item.get("salary", "")),
        source=JoobleProvider.name,
        source_color=JoobleProvider.color,
    )
