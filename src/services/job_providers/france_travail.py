import logging
import re
import time
from typing import Optional

import httpx

from src.core.schemas import JobListing
from src.services.job_providers.base import JobProvider
from src.services.job_providers.oauth2_token_manager import OAuth2TokenManager

logger = logging.getLogger(__name__)

_TOKEN_URL = "https://entreprise.francetravail.fr/connexion/oauth2/access_token?realm=/partenaire"
_SEARCH_URL = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
_SCOPE = "api_offresdemploiv2 o2dsoffre"

_SEARCH_RADIUS_KM = 30
_SEARCH_RANGE = "0-14"  # 15 results

_SALARY_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


class FranceTravailProvider(JobProvider):
    name = "france_travail"
    color = "#2563eb"

    def __init__(
        self,
        client_id: str = "",
        client_secret: str = "",
        client: httpx.Client | None = None,
        token_manager: OAuth2TokenManager | None = None,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self._client = client or httpx.Client(timeout=15.0)
        self._token_manager = token_manager or OAuth2TokenManager(
            client_id=client_id,
            client_secret=client_secret,
            token_url=_TOKEN_URL,
            scope=_SCOPE,
            client=self._client,
        )

    def check_availability(self) -> tuple[bool, float]:
        if not self.client_id or not self.client_secret:
            return False, 0.0

        start = time.monotonic()
        token = self._token_manager.refresh()
        latency_ms = (time.monotonic() - start) * 1000

        if token is None:
            return False, 0.0
        return True, latency_ms

    def search(
        self,
        query: str,
        location: Optional[str] = None,
        max_results: int = 20,
    ) -> list[JobListing]:
        if not self.client_id or not self.client_secret:
            logger.warning("France Travail credentials missing — skipping job search.")
            return []
        if not query.strip():
            return []

        token = self._token_manager.get_token()
        if token is None:
            logger.warning("France Travail: unable to obtain an OAuth2 token — skipping job search.")
            return []

        params: dict = {
            "motsCles": query,
            "rayon": _SEARCH_RADIUS_KM,
            "range": _SEARCH_RANGE,
        }
        department = _extract_department_code(location)
        if department:
            params["departement"] = department

        try:
            response = self._client.get(
                _SEARCH_URL,
                params=params,
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                logger.warning("France Travail quota exceeded for query=%r: %s", query, exc)
            else:
                logger.warning("France Travail search failed for query=%r: %s", query, exc)
            return []
        except httpx.HTTPError as exc:
            logger.warning("France Travail search failed for query=%r: %s", query, exc)
            return []

        results = response.json().get("resultats", [])
        return [_to_job_listing(item) for item in results[:max_results]]


def _extract_department_code(location: Optional[str]) -> Optional[str]:
    """Derive a French department code from a postal code (e.g. "59170 Croix" -> "59").

    France Travail's `commune` param expects an INSEE code, which isn't available
    from the CV pipeline — `departement` (derived from the postal code) is used
    instead. Free-text locations (region names, cities without digits) are ignored
    and the search falls back to nationwide (no location filter).
    """
    if not location:
        return None
    digits = re.sub(r"\D", "", location)
    if len(digits) >= 5:
        return digits[:2]
    if len(digits) in (2, 3):
        return digits
    return None


def _parse_salary_min(salary: str) -> Optional[float]:
    if not salary:
        return None
    match = _SALARY_NUMBER_RE.search(salary)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return None


def _to_job_listing(item: dict) -> JobListing:
    entreprise = item.get("entreprise") or {}
    lieu = item.get("lieuTravail") or {}
    origine = item.get("origineOffre") or {}
    salaire = item.get("salaire") or {}

    company = (entreprise.get("nom") or "").strip() or "Non précisé"

    return JobListing(
        title=(item.get("intitule") or "").strip(),
        company=company,
        location=(lieu.get("libelle") or "").strip(),
        description=(item.get("description") or "").strip(),
        url=origine.get("urlOrigine", ""),
        salary_min=_parse_salary_min(salaire.get("libelle", "")),
        source=FranceTravailProvider.name,
        source_color=FranceTravailProvider.color,
    )
