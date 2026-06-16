import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.core.lexicons import JOB_TITLE_RE, JOB_TITLE_SYNONYMS
from src.core.schemas import NormalizedCV, RankedJobMatch
from src.services.semantic_scorer import SemanticScorer

# Seniority qualifiers narrow Adzuna's `what` search too aggressively
# (e.g. "ML Engineer Junior" returns 0 results where "ML Engineer" returns many).
# Stripped only from the *search query* — NormalizedCV.job_title keeps the full title.
_SENIORITY_RE = re.compile(
    r"\b(junior|senior|lead|principal|staff|intern|associate)\b", re.IGNORECASE
)

# Adzuna's geocoding only resolves well-known places — small towns/suburbs
# (e.g. "Croix" near Lille) silently return 0 results. Offering the 13
# metropolitan French regions as a manual fallback keeps the search scoped
# to a place the user actually cares about, instead of the geocoding gap
# forcing a blind nationwide search (which surfaced irrelevant Paris-area
# listings for a CV located near Lille).
FRANCE_REGIONS: list[str] = [
    "Auvergne-Rhône-Alpes",
    "Bourgogne-Franche-Comté",
    "Bretagne",
    "Centre-Val de Loire",
    "Corse",
    "Grand Est",
    "Hauts-de-France",
    "Île-de-France",
    "Normandie",
    "Nouvelle-Aquitaine",
    "Occitanie",
    "Pays de la Loire",
    "Provence-Alpes-Côte d'Azur",
]

# Search radius (km) around the location detected in the CV — keeps results
# relevant to where the candidate actually lives rather than scattered
# nationwide.
_CV_LOCATION_RADIUS_KM = 30

# Offers scoring below this threshold are considered low-quality matches.
_MIN_SCORE_THRESHOLD = 25.0

# If fewer than this many offers pass the threshold, signal the UI to show
# a "few results" warning inviting the user to broaden their search.
_FEW_RESULTS_THRESHOLD = 3


@dataclass
class JobSearchResult:
    matches: list[RankedJobMatch]
    queries_used: list[str]
    location_used: Optional[str]
    few_results: bool = False
    source_counts: dict[str, int] = field(default_factory=dict)
    duplicates_removed: int = 0


def _trim_title_for_query(title: str) -> str:
    """Drop trailing company name from a job title string.

    Keeps words up to the last job-title keyword + 2 following words, which
    covers most patterns like "Conseiller de vente Sandro" → "Conseiller de
    vente" or "ML Engineer ACME Corp" → "ML Engineer ACME" (acceptable).
    Falls back to the full title if no keyword is found.
    """
    words = title.split()
    last_kw = max(
        (i for i, w in enumerate(words) if JOB_TITLE_RE.search(w)),
        default=len(words) - 1,
    )
    return " ".join(words[: last_kw + 3])


def _build_query(parsed_cv: NormalizedCV) -> str:
    if parsed_cv.job_title:
        cleaned = _SENIORITY_RE.sub("", parsed_cv.job_title)
        cleaned = _trim_title_for_query(re.sub(r"\s+", " ", cleaned).strip())
        if cleaned:
            return cleaned
    return " ".join(parsed_cv.skills_flat[:3])


def _find_synonym_queries(title: str) -> list[str]:
    """Return up to 2 alternative queries from JOB_TITLE_SYNONYMS for the given title."""
    title_lower = title.lower()
    for canonical, synonyms in JOB_TITLE_SYNONYMS.items():
        if canonical in title_lower or any(s in title_lower for s in synonyms):
            return [t for t in [canonical] + synonyms if t.lower() not in title_lower][:2]
    return []


def _build_queries(parsed_cv: NormalizedCV) -> list[str]:
    """Build 1–3 query variants: base title, synonym, and/or title+sector."""
    base = _build_query(parsed_cv)
    if not base:
        return []

    queries = [base]

    # Synonym variant — diversifies beyond the exact extracted title
    if parsed_cv.job_title:
        alts = _find_synonym_queries(parsed_cv.job_title)
        if alts and alts[0] not in queries:
            queries.append(alts[0])

    # Sector-enriched variant — only when the sector word is not already in base
    if parsed_cv.sector and parsed_cv.job_title and parsed_cv.sector.lower() not in base.lower():
        cleaned = _SENIORITY_RE.sub("", parsed_cv.job_title)
        trimmed = _trim_title_for_query(re.sub(r"\s+", " ", cleaned).strip())
        enriched = f"{trimmed} {parsed_cv.sector}".strip()
        if enriched and enriched not in queries:
            queries.append(enriched)

    return queries[:3]


def find_matching_jobs(
    parsed_cv: NormalizedCV,
    job_search: Any,
    scorer: SemanticScorer,
    max_results: int = 20,
    region: str | None = None,
    cv_embedding: Optional[np.ndarray] = None,
    active_providers: Optional[list[str]] = None,
) -> JobSearchResult:
    queries = _build_queries(parsed_cv)
    if not queries:
        return JobSearchResult(matches=[], queries_used=[], location_used=None)

    # Determine location and distance for Adzuna
    location_used: Optional[str] = None
    distance: Optional[int] = None
    if region:
        # User's explicit region always takes priority — avoids Adzuna
        # geocoding ambiguous city names (e.g. "Croix" → Pont-Croix/Finistère
        # instead of Croix/59 near Lille) regardless of whether the city
        # search would return results.
        location_used = region
    elif parsed_cv.location:
        # Include postal code when available for precise Adzuna geocoding:
        # "59170 Croix" is unambiguous, "Croix" alone is not.
        location_used = (
            f"{parsed_cv.postal_code} {parsed_cv.location}".strip()
            if parsed_cv.postal_code
            else parsed_cv.location
        )
        distance = _CV_LOCATION_RADIUS_KM
    else:
        return JobSearchResult(matches=[], queries_used=queries, location_used=None)

    # Run each query and collect unique listings by URL
    results_per_query = max(5, max_results // len(queries))
    seen_urls: set[str] = set()
    all_listings = []

    total_listings_seen = 0
    for query in queries:
        listings = job_search.search(
            query=query,
            location=location_used,
            distance=distance,
            max_results=results_per_query,
            active_providers=active_providers,
        )
        total_listings_seen += len(listings)
        for listing in listings:
            if listing.url not in seen_urls:
                seen_urls.add(listing.url)
                all_listings.append(listing)
    duplicates_removed = total_listings_seen - len(all_listings)

    # Score all unique listings in a single batch encoding pass
    if cv_embedding is None:
        cv_embedding = scorer.encode_cv(parsed_cv.raw_text)
    scoring_results = scorer.score_many(
        parsed_cv, [listing.description for listing in all_listings], cv_embedding=cv_embedding
    )
    scored = [
        RankedJobMatch(job=listing, scoring_result=result)
        for listing, result in zip(all_listings, scoring_results)
    ]
    scored.sort(key=lambda m: m.scoring_result.overall_score, reverse=True)

    # Filter by minimum quality threshold — return all as fallback if none pass
    filtered = [m for m in scored if m.scoring_result.overall_score >= _MIN_SCORE_THRESHOLD]
    few_results = bool(scored) and len(filtered) < _FEW_RESULTS_THRESHOLD

    final_matches = filtered if filtered else scored
    source_counts = dict(Counter(m.job.source for m in final_matches))

    return JobSearchResult(
        matches=final_matches,
        queries_used=queries,
        location_used=location_used,
        few_results=few_results,
        source_counts=source_counts,
        duplicates_removed=duplicates_removed,
    )
