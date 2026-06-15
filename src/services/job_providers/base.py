from abc import ABC, abstractmethod
from typing import Optional

from src.core.schemas import JobListing


class JobProvider(ABC):
    """Common interface for job-listing sources (Adzuna, Jooble, France Travail, ...)."""

    name: str
    color: str

    @abstractmethod
    def check_availability(self) -> tuple[bool, float]:
        """Return (available, latency_ms)."""
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        location: Optional[str] = None,
        max_results: int = 20,
        distance: Optional[int] = None,
    ) -> list[JobListing]:
        ...
