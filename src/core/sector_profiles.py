"""Dataclasses for sector-based CV scoring profiles."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Criterion:
    id: str
    label: str
    weight: int          # weights sum to 100 per profile
    required: bool       # True = mandatory for ATS eligibility
    detection_fn: str    # function name resolved by CVQualityScorer in Phase B
    keywords: list[str]  # associated keywords for evidence extraction


@dataclass
class SectorProfile:
    id: str                        # e.g. "ml_engineer"
    sector: str                    # e.g. "Informatique & Digital"
    job_title: str                 # e.g. "Machine Learning Engineer"
    aliases: list[str]             # title variants — FR-first, then EN
    detection_keywords: list[str]  # discriminant terms, never generic
    criteria: list[Criterion] = field(default_factory=list)
    esco_occupation_uri: Optional[str] = None
