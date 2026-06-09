"""Shared NLP lexicons — imported by cv_transformer and nlp_pipeline."""

import re

SECTION_HEADERS: dict[str, str] = {
    "summary": r"^(summary|objective|profile|about me|professional summary|career objective|executive summary|about|profil|accroche|résumé|présentation|à propos)",
    "experience": r"^(experience|work experience|employment|professional experience|career history|work history|employment history|professional background|expérience|expériences professionnelles?|parcours professionnel)",
    "education": r"^(education|academic|qualification|academic background|educational background|studies|academic qualifications|formation|formations?|études|diplôme|diplômes?)",
    "skills": r"^(skills|technical skills|core competencies|competencies|expertise|technologies|tech stack|tools & technologies|key skills|compétences?|compétences techniques?|technologies?|qualités?|aptitudes?|atouts|savoir-être|savoir-faire)",
    "projects": r"^(projects|key projects|personal projects|professional projects|notable projects|side projects|projets?|réalisations?)",
    "certifications": r"^(certifications?|certificates?|awards?|achievements?|licenses?|accreditations?|certifications?|diplômes?|distinctions?)",
    "languages": r"^(languages?|spoken languages?|language skills?|langues?)",
    "interests": r"^(interests?|hobbies?|passions?|personal interests?|loisirs?|centres? d'intérêts?|activités? extra-professionnelles?|vie extra-professionnelle)",
}

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
}

# Flat set of all skills across all categories (for backward-compatible keyword matching).
_ALL_SKILLS_SET: set[str] = {s.lower() for cat in SKILL_CATEGORIES.values() for s in cat}
ALL_SKILLS: list[str] = sorted(_ALL_SKILLS_SET)

# Kept for backward compatibility — nlp_pipeline used this dict structure.
TECH_SKILLS: dict[str, list[str]] = {
    "ml": SKILL_CATEGORIES["ml"],
    "mlops": SKILL_CATEGORIES["mlops"],
    "cloud": SKILL_CATEGORIES["cloud"],
    "languages": SKILL_CATEGORIES["languages"],
    "data": SKILL_CATEGORIES["data"],
    "other": SKILL_CATEGORIES["other"],
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
]

JOB_TITLE_RE = re.compile(
    r"\b(" + "|".join(JOB_TITLE_KEYWORDS) + r")\b", re.IGNORECASE
)

PERSON_NAME_RE = re.compile(r"^[A-Z][A-Za-zà-öù-ÿ'-]+$")

PROPER_NOUN_RUN_RE = re.compile(
    r"[A-ZÀ-Ý][\wà-öù-ÿ'-]*(?:[\s,-]+[A-ZÀ-Ý][\wà-öù-ÿ'-]*)*"
)

POSTAL_CODE_CITY_RE = re.compile(
    r"\b(\d{5})\b[ \t,–—-]*([A-ZÀ-Ý][\wà-öù-ÿ'-]*(?:[ \t-][A-ZÀ-Ý][\wà-öù-ÿ'-]*)*)"
)

TITLE_SPLIT_RE = re.compile(r"\s+[-–—,]\s+|\s*\|\s*")

YEAR_RANGE_RE = re.compile(
    r"\b(19|20)(\d{2})\s*[-–—]\s*((19|20)(\d{2})|present|current|now|today|aujourd|en\s*cours|présent)\b",
    re.IGNORECASE,
)

# Words that spaCy (en_core_web_sm) frequently misclassifies as GPE/LOC on French CVs.
# Used by nlp_pipeline._extract_location to filter NER false positives.
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
})

# Action verbs for quality scoring
ACTION_VERBS_EN: frozenset[str] = frozenset(
    {
        "led", "built", "designed", "developed", "implemented", "optimized",
        "automated", "deployed", "architected", "mentored", "delivered",
        "reduced", "increased", "launched", "created", "managed", "improved",
        "established", "coordinated", "negotiated", "supervised", "trained",
        "analyzed", "engineered", "integrated", "migrated", "refactored",
        "streamlined", "accelerated", "achieved", "drove", "transformed",
        "scaled", "secured", "maintained", "produced", "published",
    }
)

ACTION_VERBS_FR: frozenset[str] = frozenset(
    {
        "dirigé", "conçu", "développé", "implémenté", "optimisé", "automatisé",
        "déployé", "architecturé", "encadré", "livré", "réduit", "augmenté",
        "lancé", "piloté", "géré", "amélioré", "établi", "coordonné",
        "supervisé", "formé", "analysé", "intégré", "migré", "refactorisé",
        "accéléré", "réalisé", "transformé", "sécurisé", "maintenu", "produit",
        "créé", "mis en place", "dirigé", "assuré", "accompagné", "structuré",
    }
)

# Regex patterns for metric/quantified achievement detection
METRIC_PATTERNS: list[re.Pattern] = [
    re.compile(r"\d+\s*%"),
    re.compile(r"\d+x\b", re.IGNORECASE),
    re.compile(r"[$€£]\s*\d+"),
    re.compile(r"\d+\s*(users?|clients?|customers?|requests?|transactions?|utilisateurs?|clients?)", re.IGNORECASE),
    re.compile(r"\b(increased|decreased|reduced|improved|grew|saved|generated|delivered)\b.{0,60}\d+", re.IGNORECASE),
    re.compile(r"\b(augmenté|réduit|amélioré|généré|économisé|livré)\b.{0,60}\d+", re.IGNORECASE),
    re.compile(r"\d+\s*(ms|seconds?|minutes?|hours?|days?|ms|heures?|jours?)\b", re.IGNORECASE),
    re.compile(r"\d+\s*k\b|\d{4,}"),  # "50k" or large numbers (5000+)
]
