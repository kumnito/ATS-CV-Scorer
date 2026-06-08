import re

from src.core.schemas import ParsedCV, RankedJobMatch
from src.services.job_search import JobSearchService
from src.services.semantic_scorer import SemanticScorer

# Seniority qualifiers narrow Adzuna's `what` search too aggressively
# (e.g. "ML Engineer Junior" returns 0 results where "ML Engineer" returns many).
# Stripped only from the *search query* — ParsedCV.job_title keeps the full title.
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


def _build_query(parsed_cv: ParsedCV) -> str:
    if parsed_cv.job_title:
        cleaned = _SENIORITY_RE.sub("", parsed_cv.job_title)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            return cleaned
    return " ".join(parsed_cv.skills[:3])


def find_matching_jobs(
    parsed_cv: ParsedCV,
    job_search: JobSearchService,
    scorer: SemanticScorer,
    max_results: int = 20,
    region: str | None = None,
) -> list[RankedJobMatch]:
    query = _build_query(parsed_cv)
    if not query.strip():
        return []

    listings: list = []
    if parsed_cv.location:
        listings = job_search.search(
            query=query,
            location=parsed_cv.location,
            distance=_CV_LOCATION_RADIUS_KM,
            max_results=max_results,
        )
    if not listings and region:
        # Adzuna's geocoding misses small towns/suburbs (e.g. "Croix" near
        # Lille returns 0 results while "Hauts-de-France" returns several) —
        # fall back to the user-chosen region rather than guessing nationwide
        # and surfacing irrelevant listings far from the candidate.
        listings = job_search.search(
            query=query, location=region, max_results=max_results
        )

    matches = [
        RankedJobMatch(
            job=listing, scoring_result=scorer.score(parsed_cv, listing.description)
        )
        for listing in listings
    ]
    matches.sort(key=lambda m: m.scoring_result.overall_score, reverse=True)
    return matches
