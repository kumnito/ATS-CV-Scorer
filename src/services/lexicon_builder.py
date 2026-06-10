"""Builds skill/job lexicons from external sources (ESCO API + HuggingFace datasets).

Run directly to refresh lexicons_generated.json:
    python -m src.services.lexicon_builder [--force]
    make update-lexicons
"""

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_ESCO_BASE = "https://ec.europa.eu/esco/api"
_OUTPUT_PATH = Path(__file__).parent.parent.parent / "lexicons_generated.json"

_TARGET_OCCUPATIONS: list[str] = [
    # Tech / data
    "machine learning engineer",
    "data engineer",
    "software developer",
    "data analyst",
    "mlops engineer",
    # Vente / commerce
    "sales representative",
    "commercial",
    # RH
    "human resources manager",
    "recruitment consultant",
    # Ingénierie généraliste
    "civil engineer",
    "mechanical engineer",
    # Santé / fitness
    "fitness trainer",
    "healthcare worker",
]

# Words so generic that they cannot disambiguate an occupation query from its
# ESCO result.  Used by _occupation_matches_query to detect bad matches like
# "machine learning engineer" → "packing machinery engineer".
_TRIVIAL_OCCUPATION_WORDS: frozenset[str] = frozenset({
    "engineer", "developer", "scientist", "analyst", "specialist",
    "technician", "manager", "officer", "consultant", "architect",
})

# Keywords used to assign a raw skill string to a category bucket.
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "ml": [
        # Frameworks & libraries
        "machine learning", "deep learning", "neural", "tensorflow", "pytorch",
        "keras", "scikit", "llm", "nlp", "computer vision", "reinforcement",
        "xgboost", "lightgbm", "transformers", "hugging face", "langchain",
        "rag", "stable diffusion", "bert", "gpt", "generative",
        # ESCO-style descriptors
        "recommender", "classification", "clustering", "regression",
        "prediction model", "natural language", "statistical model",
        "machine translation", "image recognition", "pattern recognition",
    ],
    "mlops": [
        # Tools
        "mlflow", "kubeflow", "airflow", "docker", "kubernetes", "ci/cd",
        "jenkins", "terraform", "helm", "ansible", "prefect", "dvc",
        "bentoml", "triton", "github actions", "gitlab ci",
        # ESCO-style descriptors
        "pipeline", "deployment", "monitoring", "container", "orchestrat",
        "infrastructure", "continuous integration", "version control",
        "model deployment", "model monitoring",
    ],
    "cloud": [
        # Platforms
        "aws", "azure", "gcp", "google cloud", "databricks", "snowflake",
        "bigquery", "sagemaker", "vertex ai",
        # ESCO-style descriptors
        "cloud computing", r"\bcloud\b", "cloud service", "cloud platform",
        "cloud storage", "serverless",
    ],
    "languages": [
        "python", "java", "javascript", "typescript", r"\bc\+\+", r"\bc#",
        "golang", "rust", "scala", "kotlin", "swift", "php", "ruby",
        "bash", "shell scripting", "sql", "matlab", r"\br\b",
        # ESCO-style descriptors
        "programming language", "scripting",
    ],
    "data": [
        # Tools & databases
        "pandas", "numpy", "scipy", "postgresql", "mysql", "mongodb",
        "redis", "elasticsearch", "cassandra", "dbt", "kafka", "spark",
        "hadoop", "tableau", "power bi", "looker", "plotly", "matplotlib",
        "seaborn", "polars",
        # ESCO-style descriptors (broad — checked last within this category)
        r"\bdata\b", "statistics", "analytical", "query language",
        "data warehouse", "data analysis", "data mining", "etl",
        "information extraction", "database scheme", "data quality",
        "data model", "data process", "data collection",
        "online analytical",
        "business intelligence",
        "statistical analysis",
        "unstructured data",
        "big data",
    ],
}

# Verb prefixes that ESCO prepends to skill titles ("use Python", "develop models").
# Stripping them yields the underlying tool/concept name.
_SKILL_VERB_PREFIX_RE = re.compile(
    r"^(use|apply|develop|create|manage|design|implement|build|configure|"
    r"install|operate|write|program|code|analys[ei]|work with|utiliz[ei]|"
    r"employ|deploy|set up|set-up|execute|perform|collect|process|handle|"
    r"establish|report|interpret|extract|normalise|normalize|identify|"
    r"categori[sz][ei]|categoris[ei]|ensure|support|provide|document)\s+",
    re.IGNORECASE,
)


# Single-word terms so generic that matching them against CV text produces noise.
_SKILL_STOPWORDS: frozenset[str] = frozenset({
    "data", "software", "systems", "process", "information", "services",
    "applications", "methods", "tools", "results", "requirements", "techniques",
})


def _normalize_skill(title: str) -> Optional[str]:
    """Strip verb prefix and reject phrases that are too long or too generic."""
    cleaned = _SKILL_VERB_PREFIX_RE.sub("", title.strip().lower())
    words = cleaned.split()
    if not cleaned or len(words) > 3:
        return None
    # Reject single-word generic terms that would match in almost every CV
    if len(words) == 1 and cleaned in _SKILL_STOPWORDS:
        return None
    return cleaned


def _categorize(skill: str) -> str:
    """Assign a skill string to the best matching category bucket."""
    skill_lower = skill.lower()
    for category, patterns in _CATEGORY_KEYWORDS.items():
        if any(re.search(p, skill_lower) for p in patterns):
            return category
    return "other"


def _merge_partial(target: dict, partial: dict) -> None:
    """Merge a partial result dict into target in-place (no duplicates)."""
    for cat, skills in partial.get("skill_categories", {}).items():
        bucket: list = target["skill_categories"].setdefault(cat, [])
        seen = set(bucket)
        for s in skills:
            if s not in seen:
                bucket.append(s)
                seen.add(s)
    for field in ("job_titles", "action_verbs_en", "action_verbs_fr"):
        existing: list = target[field]
        seen_existing = set(existing)
        for item in partial.get(field, []):
            if item not in seen_existing:
                existing.append(item)
                seen_existing.add(item)


def _detect_skill_field(columns: list[str]) -> Optional[str]:
    """Return the first column name that looks like it contains skill strings."""
    for candidate in ("skill", "skills", "name", "label", "text", "title"):
        if candidate in columns:
            return candidate
    return None


class LexiconBuilder:
    def __init__(
        self,
        output_path: Path = _OUTPUT_PATH,
        timeout: float = 15.0,
    ) -> None:
        self._output = Path(output_path)
        self._timeout = timeout

    def build(self, force_refresh: bool = False) -> dict:
        """Fetch lexicons from all sources and persist to JSON.

        If force_refresh is False and the output file already exists, the
        cached version is returned immediately (no network calls).
        """
        if not force_refresh and self._output.exists():
            try:
                return json.loads(self._output.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        result: dict = {
            "skill_categories": {
                cat: [] for cat in ["ml", "mlops", "cloud", "languages", "data", "other"]
            },
            "job_titles": [],
            "action_verbs_fr": [],
            "action_verbs_en": [],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sources": [],
        }

        for fetch_fn, source_name in [
            (self._fetch_esco, "esco"),
            (self._fetch_huggingface, "huggingface"),
        ]:
            try:
                partial = fetch_fn()
                _merge_partial(result, partial)
                result["sources"].append(source_name)
                logger.info("Loaded lexicons from %s", source_name)
            except Exception as exc:
                logger.warning("Skipping %s source: %s", source_name, exc)

        self._output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result

    # ------------------------------------------------------------------
    # ESCO source (Priority 1)
    # ------------------------------------------------------------------

    def _fetch_esco(self) -> dict:
        skills_by_category: dict[str, list[str]] = {
            cat: [] for cat in list(_CATEGORY_KEYWORDS) + ["other"]
        }
        job_titles: list[str] = []

        with httpx.Client(base_url=_ESCO_BASE, timeout=self._timeout) as client:
            for query in _TARGET_OCCUPATIONS:
                occ_uri, occ_label = self._esco_best_occupation(client, query)
                if not occ_uri:
                    logger.debug("ESCO: no matching occupation for %r (skipped)", query)
                    continue

                if occ_label and occ_label not in job_titles:
                    job_titles.append(occ_label)

                occ_en = self._esco_get_occupation(client, occ_uri, "en")
                if not occ_en:
                    continue
                for link in occ_en.get("_links", {}).get("hasEssentialSkill", []):
                    raw = link.get("title", "")
                    normalized = _normalize_skill(raw)
                    if normalized:
                        cat = _categorize(normalized)
                        if normalized not in skills_by_category[cat]:
                            skills_by_category[cat].append(normalized)

        return {
            "skill_categories": skills_by_category,
            "job_titles": job_titles,
            "action_verbs_en": [],
            "action_verbs_fr": [],
        }

    def _esco_best_occupation(
        self, client: httpx.Client, query: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Return (uri, title) of the first ESCO occupation that shares at least
        one non-trivial domain word with the query (avoids fuzzy mismatch like
        'machine learning engineer' → 'packing machinery engineer').
        """
        resp = client.get(
            "/search",
            params={"text": query, "language": "en", "type": "occupation", "limit": 3},
        )
        resp.raise_for_status()
        results = resp.json().get("_embedded", {}).get("results", [])

        query_words = {w.lower() for w in query.split()} - _TRIVIAL_OCCUPATION_WORDS
        for r in results:
            title = r.get("title", "")
            title_words = {w.lower() for w in title.split()} - _TRIVIAL_OCCUPATION_WORDS
            if not query_words or (query_words & title_words):
                return r["uri"], title.lower()

        return None, None

    def _esco_get_occupation(
        self, client: httpx.Client, uri: str, language: str
    ) -> Optional[dict]:
        resp = client.get("/resource/occupation", params={"uri": uri, "language": language})
        if resp.status_code != 200:
            return None
        return resp.json()

    # ------------------------------------------------------------------
    # HuggingFace source (Priority 2)
    # ------------------------------------------------------------------

    def _fetch_huggingface(self) -> dict:
        try:
            from datasets import load_dataset  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "datasets package not installed — run: pip install datasets"
            ) from exc

        ds = load_dataset("jackboyla/all-job-skills", split="train")
        skill_field = _detect_skill_field(ds.column_names)
        if not skill_field:
            raise RuntimeError(
                f"Cannot find skill column in dataset. Available columns: {ds.column_names}"
            )

        skills_by_category: dict[str, list[str]] = {
            cat: [] for cat in list(_CATEGORY_KEYWORDS) + ["other"]
        }
        for row in ds:
            raw = str(row.get(skill_field, "")).strip().lower()
            if not raw or len(raw.split()) > 4:
                continue
            cat = _categorize(raw)
            if raw not in skills_by_category[cat]:
                skills_by_category[cat].append(raw)

        return {
            "skill_categories": skills_by_category,
            "job_titles": [],
            "action_verbs_en": [],
            "action_verbs_fr": [],
        }


def print_stats(data: dict) -> None:
    cats = data.get("skill_categories", {})
    total = sum(len(v) for v in cats.values())
    breakdown = ", ".join(f"{k}: {len(v)}" for k, v in cats.items())
    print(f"✓ {total} compétences ({breakdown})")
    print(f"✓ {len(data.get('job_titles', []))} métiers")
    print(
        f"✓ {len(data.get('action_verbs_en', []))} verbes EN, "
        f"{len(data.get('action_verbs_fr', []))} verbes FR"
    )
    sources = data.get("sources", [])
    print(f"✓ Sources : {sources if sources else ['(aucune — fallback hardcodé)']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    builder = LexiconBuilder()
    result = builder.build(force_refresh="--force" in sys.argv or "-f" in sys.argv)
    print_stats(result)
