"""Shared NLP lexicons — imported by cv_transformer and nlp_pipeline.

Hardcoded entries are always authoritative. If lexicons_generated.json exists
(produced by `make update-lexicons`), its entries are merged in at import time
with hardcoded entries taking priority — no regressions possible.
"""

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

# Cosine-similarity threshold for matching ESCO multi-word skill phrases
# (e.g. "produce sales reports") against CV text via sentence-transformers.
# Calibrated on the benchmark corpus (make benchmark --calibrate): 0.51 is
# the lowest value that brings the HF non-tech average to "Correct" (>=50)
# while avoiding the worst false positives (e.g. "sales promotion
# techniques" on a backend-engineer CV) without regressing synth_* CVs.
SEMANTIC_SKILL_MATCH_THRESHOLD: float = 0.51

SECTION_HEADERS: dict[str, str] = {
    "summary": r"^(summary|objective|profile|about me|professional summary|career objective|executive summary|about|profil|accroche|résumé|présentation|à propos)",
    "experience": r"^(experience|work experience|employment|professional experience|career history|work history|employment history|professional background|expérience|expériences professionnelles?|parcours professionnel)",
    "education": r"^(education|academic|qualification|academic background|educational background|studies|academic qualifications|formation|formations?|études|diplôme|diplômes?)",
    "skills": r"^(skills|technical skills|core competencies|competencies|expertise|technologies|tech stack|tools & technologies|key skills|compétences?|compétences techniques?|technologies?|qualités?|aptitudes?|atouts|savoir-être|savoir-faire)",
    "projects": r"^(projects|key projects|personal projects|professional projects|notable projects|side projects|projets?|réalisations?)",
    "certifications": r"^(certifications?|certificates?|awards?|achievements?|licenses?|accreditations?|certifications?|diplômes?|distinctions?)",
    "languages": r"^(languages?|spoken languages?|language skills?|langues?)",
    "interests": r"^(interests?|hobbies?|passions?|personal interests?|loisirs?|centres? d'intérêts?|activités? extra-professionnelles?|vie extra-professionnelle)",
    "contact": r"^(contact|contact information|contact details|personal information|personal details|coordonnées?|informations? personnelles?|informations? de contact)",
}

# Mutable — extended at import time by _merge_generated() if lexicons_generated.json exists.
SKILL_CATEGORIES: dict[str, list[str]] = {
    "ml": [
        "tensorflow",
        "pytorch",
        "scikit-learn",
        "keras",
        "xgboost",
        "lightgbm",
        "hugging face",
        "transformers",
        "langchain",
        "llm",
        "nlp",
        "computer vision",
        "deep learning",
        "machine learning",
        "reinforcement learning",
        "openai",
        "llamaindex",
        "rag",
        "stable diffusion",
    ],
    "mlops": [
        "mlflow",
        "kubeflow",
        "airflow",
        "docker",
        "kubernetes",
        "ci/cd",
        "github actions",
        "gitlab ci",
        "jenkins",
        "terraform",
        "helm",
        "ansible",
        "prefect",
        "dvc",
        "bentoml",
        "triton",
    ],
    "cloud": [
        "aws",
        "azure",
        "gcp",
        "google cloud",
        "databricks",
        "snowflake",
        "bigquery",
        "sagemaker",
        "vertex ai",
        "azure ml",
    ],
    "languages": [
        "python",
        "java",
        "javascript",
        "typescript",
        "c++",
        "c#",
        "go",
        "rust",
        "scala",
        "r",
        "kotlin",
        "swift",
        "php",
        "ruby",
        "bash",
        "sql",
        "matlab",
        "julia",
    ],
    "data": [
        "pandas",
        "numpy",
        "scipy",
        "postgresql",
        "mysql",
        "mongodb",
        "redis",
        "elasticsearch",
        "sqlite",
        "cassandra",
        "dbt",
        "neo4j",
        "dynamodb",
        "kafka",
        "spark",
        "hadoop",
        "tableau",
        "power bi",
        "looker",
        "plotly",
        "matplotlib",
        "seaborn",
        "polars",
    ],
    "other": [
        "react",
        "vue",
        "angular",
        "django",
        "flask",
        "fastapi",
        "spring",
        "express",
        "next.js",
        "nuxt",
        "laravel",
        "rails",
        "asp.net",
        "git",
        "github",
        "gitlab",
        "jira",
        "confluence",
        "grafana",
        "prometheus",
        "datadog",
        "linux",
        "streamlit",
        "gradio",
    ],
    "commerce": [
        # Activités commerciales (FR)
        "vente", "merchandising", "prospection", "négociation",
        "fidélisation", "encaissement",
        # Relation / service client (FR)
        "service client", "relation client", "conseil client",
        # Gestion opérationnelle (FR)
        "gestion de stock", "inventaire", "réassort",
        # English equivalents (offres Adzuna souvent en anglais)
        "sales", "customer service", "retail", "cashier",
        "inventory management",
        # Outils transverses
        "crm", "pos",
    ],
}

JOB_TITLE_KEYWORDS: list[str] = [
    # English
    "engineer",
    "developer",
    "scientist",
    "analyst",
    "manager",
    "designer",
    "architect",
    "consultant",
    "specialist",
    "lead",
    "director",
    "researcher",
    "administrator",
    "technician",
    "officer",
    "coordinator",
    "assistant",
    "operator",
    # French — tech
    "ingénieur",
    "développeur",
    "analyste",
    "responsable",
    "chef",
    "concepteur",
    "consultant",
    "chargé",
    "coordinateur",
    "gestionnaire",
    "technicien",
    "directeur",
    "superviseur",
    "formateur",
    # French — commerce / retail / générique
    "vendeur",
    "vendeuse",
    "conseiller",
    "conseillère",
    "commercial",
    "commerciale",
    "assistant",
    "assistante",
    "opérateur",
    "opératrice",
    "hôte",
    "hôtesse",
    "caissier",
    "caissière",
    "logisticien",
    "magasinier",
    "stockman",
    "préparateur",
    "livreur",
    "agent",
    "chargée",
    "conducteur",
    "conductrice",
    "chauffeur",
    "chauffeure",
    "exploitant",
]

# Hardcoded base action-verb sets (mutable — _merge_generated() may extend them).
_ACTION_VERBS_EN_BASE: set[str] = {
    "led", "built", "designed", "developed", "implemented", "optimized",
    "automated", "deployed", "architected", "mentored", "delivered",
    "reduced", "increased", "launched", "created", "managed", "improved",
    "established", "coordinated", "negotiated", "supervised", "trained",
    "analyzed", "engineered", "integrated", "migrated", "refactored",
    "streamlined", "accelerated", "achieved", "drove", "transformed",
    "scaled", "secured", "maintained", "produced", "published",
}

_ACTION_VERBS_FR_BASE: set[str] = {
    "dirigé", "conçu", "développé", "implémenté", "optimisé", "automatisé",
    "déployé", "architecturé", "encadré", "livré", "réduit", "augmenté",
    "lancé", "piloté", "géré", "amélioré", "établi", "coordonné",
    "supervisé", "formé", "analysé", "intégré", "migré", "refactorisé",
    "accéléré", "réalisé", "transformé", "sécurisé", "maintenu", "produit",
    "créé", "mis en place", "assuré", "accompagné", "structuré",
}

PERSON_NAME_RE = re.compile(r"^[A-Z][A-Za-zà-öù-ÿ'-]+$")

PROPER_NOUN_RUN_RE = re.compile(
    r"[A-ZÀ-Ý][\wà-öù-ÿ'-]*(?:[\s,-]+[A-ZÀ-Ý][\wà-öù-ÿ'-]*)*"
)

# Matches two orderings of FR postal address:
#   postal-first : "59170 Croix"  → group(1)=postal, group(2)=city
#   city-first   : "Croix, 59170" → group(3)=city,   group(4)=postal
# Callers must use: city = m.group(2) or m.group(3)
#                   postal = m.group(1) or m.group(4)
_PC = r"\b(\d{5})\b"
_CITY = r"([A-ZÀ-Ý][\wà-öù-ÿ'-]*(?:[ \t-][A-ZÀ-Ý][\wà-öù-ÿ'-]*)*)"
POSTAL_CODE_CITY_RE = re.compile(
    rf"(?:{_PC}[ \t,–—-]*{_CITY}|{_CITY}[ \t]*,[ \t]*{_PC})"
)
# Postal-first only variant: "59170 Croix" — more reliable than the combined
# regex because the city-first alternative can consume street names as "cities"
# (e.g. "Jean Jaurès, 59170" matches before "59170 Croix").
POSTAL_FIRST_RE = re.compile(rf"{_PC}[ \t,–—-]*{_CITY}")

TITLE_SPLIT_RE = re.compile(r"\s+[-–—,]\s+|\s*\|\s*")

YEAR_RANGE_RE = re.compile(
    r"\b(19|20)(\d{2})\s*[-–—]\s*((19|20)(\d{2})|present|current|now|today|aujourd|en\s*cours|présent)\b",
    re.IGNORECASE,
)

# Words that spaCy (en_core_web_sm) frequently misclassifies as GPE/LOC on French CVs.
LOCATION_BLOCKLIST: frozenset[str] = frozenset({
    # Language names / demonyms
    "français", "anglais", "espagnol", "allemand", "italien", "portugais",
    "chinois", "japonais", "arabe", "russe", "néerlandais", "néerlandaise",
    "flamand", "catalan", "hindi", "coréen", "vietnamien", "turc", "polonais",
    "français natif", "anglais courant", "bilingue",
    # Common soft-skill / quality words misclassified
    "autonome", "rigoureux", "dynamique", "motivé", "passionné", "organisé",
    "polyvalent", "réactif", "créatif", "curieux", "ponctuel", "sérieux",
    # Generic words that appear capitalised in CV headers
    "contact", "email", "adresse", "téléphone", "linkedin", "github",
    "mobile", "permis",
    # Street-adjacent proper nouns common in French addresses (frequent false positives)
    "jaurès", "jean jaurès", "gambetta", "clemenceau", "foch",
    "de gaulle", "pasteur", "victor hugo", "napoleon",
})

# Regex patterns for metric/quantified achievement detection
METRIC_PATTERNS: list[re.Pattern] = [
    re.compile(r"\d+\s*%"),
    re.compile(r"\d+x\b", re.IGNORECASE),
    re.compile(r"[$€£]\s*\d+"),
    re.compile(r"\d+\s*(users?|clients?|customers?|requests?|transactions?|utilisateurs?|clients?)", re.IGNORECASE),
    re.compile(r"\b(increased|decreased|reduced|improved|grew|saved|generated|delivered)\b.{0,60}\d+", re.IGNORECASE),
    re.compile(r"\b(augmenté|réduit|amélioré|généré|économisé|livré)\b.{0,60}\d+", re.IGNORECASE),
    re.compile(r"\d+\s*(ms|seconds?|minutes?|hours?|days?|ms|heures?|jours?)\b", re.IGNORECASE),
    re.compile(r"\d+\s*k\b|\d{4,}"),
]

# Job title synonyms — maps a canonical title fragment to alternative search terms.
# Used by job_matcher._build_queries() to diversify Adzuna queries.
JOB_TITLE_SYNONYMS: dict[str, list[str]] = {
    "vendeur": ["conseiller de vente", "commercial terrain", "chargé de clientèle", "attaché commercial"],
    "développeur": ["software engineer", "ingénieur logiciel", "programmeur"],
    "data scientist": ["ml engineer", "ingénieur machine learning", "analyste data"],
    "conducteur": ["chauffeur", "transporteur", "livreur"],
    "opérateur": ["technicien de production", "agent de fabrication", "conducteur de ligne"],
}

# Sector keywords — maps keyword lists to a sector label added to Adzuna queries.
# Scanned against experience text (titles + company + bullets) for sector enrichment.
SECTOR_KEYWORDS: list[tuple[list[str], str]] = [
    (["magasin", "boutique", "grande distribution", "supermarché", "hypermarché", "enseigne"], "magasin"),
    (["mode", "textile", "prêt-à-porter", "habillement", "collection", "luxe", "maroquinerie"], "mode"),
    (["restauration", "restaurant", "cuisine", "hôtellerie", "fast-food", "café"], "restauration"),
    (["transport", "livraison", "logistique", "chauffeur", "fret", "supply chain"], "transport"),
    (["production", "fabrication", "industrie", "usine", "atelier", "manufacture"], "industrie"),
    (["btp", "chantier", "construction", "bâtiment", "travaux"], "btp"),
    (["santé", "médical", "hôpital", "soins", "infirmier", "clinique", "pharmacie"], "santé"),
    (["banque", "finance", "assurance", "comptabilité", "audit"], "finance"),
]


# ---------------------------------------------------------------------------
# Merge generated lexicons at import time (silent fallback if file absent)
# ---------------------------------------------------------------------------

# ESCO multi-word skill phrases eligible for semantic matching (e.g. "produce
# sales reports"), and their precomputed sentence-transformers embeddings.
# Populated by _load_generated_embeddings() if lexicons_embeddings.npy exists
# and matches embedded_skills (both produced by `make update-lexicons`).
GENERATED_SKILL_PHRASES: list[str] = []
GENERATED_SKILL_EMBEDDINGS: Optional[np.ndarray] = None

_EMBEDDINGS_PATH = Path(__file__).parent.parent.parent / "lexicons_embeddings.npy"


def _load_generated_embeddings(embedded_skills: list[str]) -> None:
    global GENERATED_SKILL_PHRASES, GENERATED_SKILL_EMBEDDINGS
    if not embedded_skills:
        return

    # Fast path: load from cache.
    if _EMBEDDINGS_PATH.exists():
        try:
            embeddings = np.load(_EMBEDDINGS_PATH)
            if embeddings.shape[0] == len(embedded_skills):
                GENERATED_SKILL_PHRASES = embedded_skills
                GENERATED_SKILL_EMBEDDINGS = embeddings
                return
        except (OSError, ValueError):
            pass  # Cache corrupt — recompute below.

    # Cache absent or corrupt: encode with the shared MiniLM model.
    # Persists the result so subsequent startups use the fast path.
    try:
        from src.core.model_registry import get_minilm
        embeddings = get_minilm().encode(embedded_skills, convert_to_numpy=True)
        try:
            np.save(_EMBEDDINGS_PATH, embeddings)
        except OSError:
            pass  # Read-only filesystem — keep in-memory embeddings only.
        GENERATED_SKILL_PHRASES = embedded_skills
        GENERATED_SKILL_EMBEDDINGS = embeddings
    except Exception:
        pass  # sentence-transformers unavailable — skip semantic matching.


def _merge_generated() -> tuple[list[str], list[str]]:
    """Load lexicons_generated.json and merge into module-level structures.

    Returns (extra_verbs_en, extra_verbs_fr) — new verbs not already in the
    hardcoded base sets, to be unioned into the final ACTION_VERBS frozensets.

    As a side effect, populates GENERATED_SKILL_PHRASES and
    GENERATED_SKILL_EMBEDDINGS for semantic skill matching (see
    src/services/semantic_skill_matcher.py).
    """
    _path = Path(__file__).parent.parent.parent / "lexicons_generated.json"
    if not _path.exists():
        return [], []
    try:
        data = json.loads(_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return [], []

    _load_generated_embeddings(data.get("embedded_skills", []))

    # Skills: extend each category with generated entries not in hardcoded set.
    hardcoded_skills: set[str] = {
        s.lower() for cat in SKILL_CATEGORIES.values() for s in cat
    }
    for cat, skills in data.get("skill_categories", {}).items():
        bucket = SKILL_CATEGORIES.setdefault(cat, [])
        for s in skills:
            s_low = s.lower()
            if s_low not in hardcoded_skills:
                bucket.append(s)
                hardcoded_skills.add(s_low)

    extra_en = [
        v.lower() for v in data.get("action_verbs_en", [])
        if v.lower() not in _ACTION_VERBS_EN_BASE
    ]
    extra_fr = [
        v.lower() for v in data.get("action_verbs_fr", [])
        if v.lower() not in _ACTION_VERBS_FR_BASE
    ]
    return extra_en, extra_fr


# ---------------------------------------------------------------------------
# Derived symbols — computed by init_lexicons(), called once below
# ---------------------------------------------------------------------------

# Flat list of all skills across all categories (for backward-compatible keyword matching).
ALL_SKILLS: list[str] = []
# Single compiled alternation regex matching any skill in ALL_SKILLS (built by init_lexicons()).
ALL_SKILLS_RE: re.Pattern = re.compile(r"(?!)")
JOB_TITLE_RE: re.Pattern = re.compile(r"(?!)")
ACTION_VERBS_EN: frozenset[str] = frozenset()
ACTION_VERBS_FR: frozenset[str] = frozenset()

_initialized = False


def init_lexicons() -> None:
    """Merge lexicons_generated.json and compute derived lexicon structures.

    Idempotent — safe to call multiple times (e.g. explicitly at startup from
    app.py / src/api/server.py) and is also invoked once at module import time
    so that `from src.core.lexicons import ALL_SKILLS` keeps working for
    callers that don't invoke it explicitly.
    """
    global ALL_SKILLS, ALL_SKILLS_RE, JOB_TITLE_RE, ACTION_VERBS_EN, ACTION_VERBS_FR, _initialized
    if _initialized:
        return

    extra_verbs_en, extra_verbs_fr = _merge_generated()

    all_skills_set: set[str] = {s.lower() for cat in SKILL_CATEGORIES.values() for s in cat}
    ALL_SKILLS = sorted(all_skills_set)
    ALL_SKILLS_RE = re.compile(
        r"\b(" + "|".join(re.escape(s) for s in ALL_SKILLS) + r")\b"
    )

    JOB_TITLE_RE = re.compile(
        r"\b(" + "|".join(JOB_TITLE_KEYWORDS) + r")\b", re.IGNORECASE
    )

    ACTION_VERBS_EN = frozenset(_ACTION_VERBS_EN_BASE | set(extra_verbs_en))
    ACTION_VERBS_FR = frozenset(_ACTION_VERBS_FR_BASE | set(extra_verbs_fr))

    _initialized = True


init_lexicons()
