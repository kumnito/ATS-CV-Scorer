from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# CV Transformer output models
# ---------------------------------------------------------------------------


class CVHeader(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    postal_code: Optional[str] = None
    github: Optional[str] = None
    linkedin: Optional[str] = None


class CVSkills(BaseModel):
    ml: list[str] = Field(default_factory=list)
    mlops: list[str] = Field(default_factory=list)
    cloud: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    data: list[str] = Field(default_factory=list)
    other: list[str] = Field(default_factory=list)

    def flat(self) -> list[str]:
        return self.ml + self.mlops + self.cloud + self.languages + self.data + self.other


class CVExperience(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    period: Optional[str] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None  # "present" when is_current is True
    duration_months: Optional[int] = None
    is_current: bool = False
    years: Optional[float] = None
    bullets: list[str] = Field(default_factory=list)


class CVEducation(BaseModel):
    degree: str = ""
    school: str = ""
    year: Optional[str] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    duration_months: Optional[int] = None
    is_current: bool = False
    skills: list[str] = Field(default_factory=list)


class CVProject(BaseModel):
    name: str = ""
    description: Optional[str] = None
    stack: list[str] = Field(default_factory=list)
    url: Optional[str] = None
    metrics: list[str] = Field(default_factory=list)


class NormalizedCV(BaseModel):
    header: CVHeader = Field(default_factory=CVHeader)
    summary: Optional[str] = None
    skills: CVSkills = Field(default_factory=CVSkills)
    experience: list[CVExperience] = Field(default_factory=list)
    education: list[CVEducation] = Field(default_factory=list)
    projects: list[CVProject] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    raw_text: str = ""
    layout_detected: str = "single_column"
    word_count: int = 0


# ---------------------------------------------------------------------------
# CV Quality Report
# ---------------------------------------------------------------------------


class CVQualityReport(BaseModel):
    score_structure: int
    score_contenu: int
    score_global: int  # 40% structure + 60% contenu
    sections_detectees: list[str] = Field(default_factory=list)
    sections_manquantes: list[str] = Field(default_factory=list)
    layout: str
    word_count: int
    has_metrics: bool
    keyword_density: float
    recommendations: list[str] = Field(default_factory=list)
    # Derived career fields
    total_experience_years: float = 0.0
    career_start_year: Optional[int] = None
    most_recent_role: Optional[str] = None
    education_years: int = 0
    career_gaps: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Existing models (unchanged)
# ---------------------------------------------------------------------------


class ParsedCV(BaseModel):
    raw_text: str
    sections: dict[str, str] = Field(default_factory=dict)
    entities: dict[str, list[str]] = Field(default_factory=dict)
    skills: list[str] = Field(default_factory=list)
    experience_years: Optional[float] = None
    keywords: list[str] = Field(default_factory=list)
    job_title: Optional[str] = None
    location: Optional[str] = None
    postal_code: Optional[str] = None


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


class JobListing(BaseModel):
    title: str
    company: str
    location: str
    description: str
    url: str
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    contract_type: Optional[str] = None


class RankedJobMatch(BaseModel):
    job: JobListing
    scoring_result: ScoringResult
