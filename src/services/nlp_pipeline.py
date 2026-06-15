import re
from collections import Counter
from datetime import datetime
from typing import Optional

import spacy

from src.core.lexicons import (
    ALL_SKILLS,
    JOB_TITLE_RE,
    LOCATION_BLOCKLIST,
    POSTAL_CODE_CITY_RE,
    PROPER_NOUN_RUN_RE,
    SECTION_HEADERS,
    SECTOR_KEYWORDS,
    TITLE_SPLIT_RE,
    YEAR_RANGE_RE,
    PERSON_NAME_RE,
)
from src.core.schemas import NormalizedCV, ParsedCV

_ALL_SKILLS = ALL_SKILLS


class NLPPipeline:
    def __init__(self, model: str = "en_core_web_sm") -> None:
        self.nlp = spacy.load(model)

    def parse_normalized(self, normalized_cv: NormalizedCV) -> ParsedCV:
        """Build a ParsedCV from a NormalizedCV, preferring structured fields.

        Structured fields (job_title, location, skills, experience_years,
        sections) are taken directly from NormalizedCV where available —
        bypassing the brittle regex/NER path that is unreliable on French CVs.
        spaCy is still used for entities and keyword extraction on raw_text.
        """
        cleaned = _clean_text(normalized_cv.raw_text)
        doc = self.nlp(cleaned[:100_000])
        entities = _extract_entities(doc)

        sections = _build_sections_from_normalized(normalized_cv)
        all_skills = normalized_cv.skills.flat()

        exp_months = sum(e.duration_months or 0 for e in normalized_cv.experience)
        exp_years: Optional[float] = None
        if exp_months > 0:
            exp_years = round(exp_months / 12, 1)
        else:
            exp_years = _estimate_experience_years(sections.get("experience", ""))

        job_title = (
            normalized_cv.header.title
            # Most recent experience title is cleaner than header title for CVs
            # where the header line mixes the title and the company name without
            # a separator (e.g. "Conseiller de vente Sandro").
            or (normalized_cv.experience[0].title if normalized_cv.experience else None)
            or _extract_job_title(sections.get("header", ""))
        )
        location = normalized_cv.header.location or _extract_location(sections.get("header", ""), entities)
        postal_code = normalized_cv.header.postal_code
        sector = _extract_sector(normalized_cv)

        return ParsedCV(
            raw_text=cleaned,
            sections=sections,
            entities=entities,
            skills=all_skills or _extract_skills(cleaned),
            experience_years=exp_years,
            keywords=_extract_keywords(doc),
            job_title=job_title,
            location=location,
            postal_code=postal_code,
            sector=sector,
        )


# --- helpers (module-level for testability) ---


def _extract_sector(normalized_cv: NormalizedCV) -> Optional[str]:
    """Detect professional sector from experience entries (titles, company, bullets)."""
    if not normalized_cv.experience:
        return None
    experience_text = " ".join(
        " ".join(filter(None, [e.title, e.company] + list(e.bullets)))
        for e in normalized_cv.experience
    ).lower()
    for keywords, sector in SECTOR_KEYWORDS:
        if any(kw in experience_text for kw in keywords):
            return sector
    return None


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


def _looks_like_person_name(line: str) -> bool:
    words = line.split()
    return (
        1 < len(words) <= 4
        and all(PERSON_NAME_RE.match(w) for w in words)
        and not JOB_TITLE_RE.search(line)
    )


def _extract_job_title(header_text: str) -> Optional[str]:
    for line in header_text.split("\n"):
        stripped = line.strip()
        if (
            not stripped
            or _looks_like_person_name(stripped)
            or not JOB_TITLE_RE.search(stripped)
        ):
            continue
        for segment in TITLE_SPLIT_RE.split(stripped):
            segment = segment.strip()
            if segment and JOB_TITLE_RE.search(segment):
                return segment
    return None


def _clean_location_candidate(raw: str) -> Optional[str]:
    runs = PROPER_NOUN_RUN_RE.findall(raw)
    if not runs:
        return None
    return max(runs, key=len).strip(" ,-")


def _extract_location(
    header_text: str, entities: dict[str, list[str]]
) -> Optional[str]:
    postal_match = POSTAL_CODE_CITY_RE.search(header_text)
    if postal_match:
        return postal_match.group(2).strip().title()

    locations = list(
        dict.fromkeys(
            filter(
                None,
                (
                    _clean_location_candidate(loc)
                    for loc in entities.get("locations", [])
                    if loc.lower().strip() not in LOCATION_BLOCKLIST
                ),
            )
        )
    )
    if not locations:
        return None
    in_header = [
        (header_text.index(loc), loc) for loc in locations if loc in header_text
    ]
    if in_header:
        return min(in_header)[1]
    return locations[0]


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
        elif re.search(r"(aujourd|en\s*cours|présent|actuel)", end_raw, re.IGNORECASE):
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


def _build_sections_from_normalized(cv: NormalizedCV) -> dict[str, str]:
    sections: dict[str, str] = {}
    if cv.header.name or cv.header.title:
        parts = [p for p in [cv.header.name, cv.header.title, cv.header.email, cv.header.phone, cv.header.location] if p]
        sections["header"] = "\n".join(parts)
    if cv.summary:
        sections["summary"] = cv.summary
    if cv.experience:
        exp_lines = []
        for e in cv.experience:
            line_parts = [p for p in [e.title, e.company, e.period] if p]
            exp_lines.append(" — ".join(line_parts))
            exp_lines.extend(e.bullets)
        sections["experience"] = "\n".join(exp_lines)
    if cv.education:
        edu_lines = [f"{e.degree} {e.school} {e.year or ''}".strip() for e in cv.education]
        sections["education"] = "\n".join(edu_lines)
    if cv.skills.flat():
        sections["skills"] = ", ".join(cv.skills.flat())
    if cv.projects:
        sections["projects"] = "\n".join(p.name for p in cv.projects)
    if cv.languages:
        sections["languages"] = ", ".join(cv.languages)
    return sections
