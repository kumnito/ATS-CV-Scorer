import logging
from concurrent.futures import ThreadPoolExecutor

from src.core.schemas import JobListing
from src.services.job_providers.base import JobProvider

logger = logging.getLogger(__name__)


class JobSearchOrchestrator:
    """Fans a search out across multiple JobProvider sources and merges results."""

    def __init__(self, providers: list[JobProvider]) -> None:
        self.providers = providers

    def check_all_availability(self) -> dict[str, tuple[bool, float]]:
        if not self.providers:
            return {}

        results: dict[str, tuple[bool, float]] = {}
        with ThreadPoolExecutor(max_workers=len(self.providers)) as executor:
            futures = {
                executor.submit(provider.check_availability): provider
                for provider in self.providers
            }
            for future, provider in futures.items():
                try:
                    results[provider.name] = future.result()
                except Exception as exc:  # noqa: BLE001 — provider failures must not crash the check
                    logger.warning("Availability check failed for %s: %s", provider.name, exc)
                    results[provider.name] = (False, 0.0)
        return results

    def search(
        self,
        query: str,
        location: str | None = None,
        active_providers: list[str] | None = None,
        max_results: int = 20,
        distance: int | None = None,
    ) -> list[JobListing]:
        if active_providers is None:
            active_providers = [provider.name for provider in self.providers]

        seen_urls: set[str] = set()
        all_listings: list[JobListing] = []

        for provider in self.providers:
            if provider.name not in active_providers:
                continue
            try:
                listings = provider.search(
                    query=query, location=location, max_results=max_results, distance=distance
                )
            except Exception as exc:  # noqa: BLE001 — one provider's failure must not break the rest
                logger.warning("Provider %s search failed: %s", provider.name, exc)
                continue

            for listing in listings:
                if listing.url not in seen_urls:
                    seen_urls.add(listing.url)
                    all_listings.append(listing)

        return all_listings
