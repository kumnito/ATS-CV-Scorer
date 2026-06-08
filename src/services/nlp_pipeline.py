import re
from collections import Counter
from datetime import datetime
from typing import Optional

import spacy

from src.core.schemas import ParsedCV

SECTION_HEADERS: dict[str, str] = {
    "summary": r"^(summary|objective|profile|about me|professional summary|career objective|executive summary|about)",
    "experience": r"^(experience|work experience|employment|professional experience|career history|work history|employment history|professional background)",
    "education": r"^(education|academic|qualification|academic background|educational background|studies|academic qualifications)",
    "skills": r"^(skills|technical skills|core competencies|competencies|expertise|technologies|tech stack|tools & technologies|key skills)",
    "projects": r"^(projects|key projects|personal projects|professional projects|notable projects|side projects)",
    "certifications": r"^(certifications?|certificates?|awards?|achievements?|licenses?|accreditations?)",
    "languages": r"^(languages?|spoken languages?|language skills?)",
}

TECH_SKILLS: dict[str, list[str]] = {
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
    ],
    "frameworks": [
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
    ],
    "ml_ai": [
        "tensorflow",
        "pytorch",
        "scikit-learn",
        "keras",
        "xgboost",
        "lightgbm",
        "pandas",
        "numpy",
        "scipy",
        "hugging face",
        "transformers",
        "langchain",
        "openai",
        "llm",
        "nlp",
        "computer vision",
        "deep learning",
        "machine learning",
        "reinforcement learning",
        "mlflow",
        "kubeflow",
    ],
    "cloud_devops": [
        "aws",
        "azure",
        "gcp",
        "google cloud",
        "docker",
        "kubernetes",
        "terraform",
        "helm",
        "ansible",
        "jenkins",
        "github actions",
        "gitlab ci",
        "ci/cd",
        "airflow",
        "kafka",
        "spark",
        "hadoop",
        "databricks",
    ],
    "databases": [
        "postgresql",
        "mysql",
        "mongodb",
        "redis",
        "elasticsearch",
        "sqlite",
        "cassandra",
        "bigquery",
        "snowflake",
        "dbt",
        "neo4j",
        "dynamodb",
    ],
    "tools": [
        "git",
        "github",
        "gitlab",
        "jira",
        "confluence",
        "tableau",
        "power bi",
        "looker",
        "grafana",
        "prometheus",
        "datadog",
        "linux",
    ],
}

_ALL_SKILLS = sorted({s.lower() for cat in TECH_SKILLS.values() for s in cat})

YEAR_RANGE_RE = re.compile(
    r"\b(19|20)(\d{2})\s*[-–—]\s*((19|20)(\d{2})|present|current|now|today)\b",
    re.IGNORECASE,
)


class NLPPipeline:
    def __init__(self, model: str = "en_core_web_sm") -> None:
        self.nlp = spacy.load(model)

    def parse_cv(self, text: str) -> ParsedCV:
        cleaned = _clean_text(text)
        sections = _detect_sections(cleaned)
        doc = self.nlp(cleaned[:100_000])

        return ParsedCV(
            raw_text=cleaned,
            sections=sections,
            entities=_extract_entities(doc),
            skills=_extract_skills(cleaned),
            experience_years=_estimate_experience_years(sections.get("experience", "")),
            keywords=_extract_keywords(doc),
        )


# --- helpers (module-level for testability) ---


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _detect_sections(text: str) -> dict[str, str]:
    lines = text.split("\n")
    buckets: dict[str, list[str]] = {"header": []}
    current = "header"

    for line in lines:
        stripped = line.strip()
        matched = _match_section_header(stripped)
        if matched:
            current = matched
            buckets.setdefault(current, [])
        else:
            buckets.setdefault(current, []).append(stripped)

    return {k: "\n".join(v).strip() for k, v in buckets.items() if any(v)}


def _match_section_header(line: str) -> Optional[str]:
    if not line or len(line) > 60:
        return None
    normalized = line.lower().rstrip(":").strip()
    # Accept lines that look like headers: all-caps, title-cased, or ending with ':'
    looks_like_header = (
        line.isupper() or line.endswith(":") or re.match(r"^[A-Z][A-Za-z\s&/]+$", line)
    )
    if not looks_like_header:
        return None
    for section, pattern in SECTION_HEADERS.items():
        if re.match(pattern, normalized, re.IGNORECASE):
            return section
    return None


def _extract_entities(doc) -> dict[str, list[str]]:
    mapping = {
        "PERSON": "persons",
        "ORG": "organizations",
        "GPE": "locations",
        "LOC": "locations",
        "DATE": "dates",
    }
    result: dict[str, list[str]] = {v: [] for v in mapping.values()}
    for ent in doc.ents:
        key = mapping.get(ent.label_)
        if key:
            result[key].append(ent.text)
    return {k: list(dict.fromkeys(v)) for k, v in result.items()}


def _extract_skills(text: str) -> list[str]:
    text_lower = text.lower()
    return sorted(
        skill
        for skill in _ALL_SKILLS
        if re.search(r"\b" + re.escape(skill) + r"\b", text_lower)
    )


def _estimate_experience_years(experience_text: str) -> Optional[float]:
    if not experience_text:
        return None

    current_year = datetime.now().year
    total = 0.0

    for m in YEAR_RANGE_RE.finditer(experience_text):
        start = int(m.group(1) + m.group(2))
        end_raw = m.group(3).lower()
        if end_raw in ("present", "current", "now", "today"):
            end = current_year
        else:
            end = int(m.group(4) + m.group(5)) if m.group(4) else int(end_raw)

        if 1970 <= start <= current_year and start <= end <= current_year + 1:
            total += end - start

    return round(total, 1) if total > 0 else None


def _extract_keywords(doc) -> list[str]:
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and token.pos_ in ("NOUN", "PROPN", "ADJ")
        and len(token.text) > 2
    ]
    return [word for word, _ in Counter(tokens).most_common(50)]
