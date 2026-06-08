from typing import Optional
from pydantic import BaseModel, Field


class ParsedCV(BaseModel):
    raw_text: str
    sections: dict[str, str] = Field(default_factory=dict)
    entities: dict[str, list[str]] = Field(default_factory=dict)
    skills: list[str] = Field(default_factory=list)
    experience_years: Optional[float] = None
    keywords: list[str] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    semantic_similarity: float
    keyword_match: float
    structure_completeness: float
    ai_feedback_score: Optional[float] = None


class ScoringResult(BaseModel):
    overall_score: float
    breakdown: ScoreBreakdown
    matched_keywords: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    feedback: Optional[str] = None


class ATSResponse(BaseModel):
    scoring_result: ScoringResult
    parsed_cv: ParsedCV
    processing_time_seconds: float
