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
    commerce: list[str] = Field(default_factory=list)

    def flat(self) -> list[str]:
        return self.ml + self.mlops + self.cloud + self.languages + self.data + self.other + self.commerce


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
    extraction_method: str = "pdfplumber"   # "pdfplumber" | "ocr" | "vision_llm"
    extraction_confidence: float = 1.0       # 0.0 – 1.0

    # Champs enrichis par NLPPipeline.parse_normalized() — résolution des
    # valeurs (titre, localisation, secteur...) via fallback regex/NER/spaCy
    # sur raw_text, en complément des champs structurés ci-dessus.
    sections: dict[str, str] = Field(default_factory=dict)
    entities: dict[str, list[str]] = Field(default_factory=dict)
    keywords: list[str] = Field(default_factory=list)
    skills_flat: list[str] = Field(default_factory=list)
    experience_years: Optional[float] = None
    job_title: Optional[str] = None
    location: Optional[str] = None
    postal_code: Optional[str] = None
    sector: Optional[str] = None


# ---------------------------------------------------------------------------
# CV Quality Report — 3 axes indépendants
# ---------------------------------------------------------------------------


class ATSReadability(BaseModel):
    layout: str                    # "single_column" | "two_columns"
    layout_label: str              # "✅ Optimal" | "⚠️ Risque parseur"
    sections_found: list[str] = Field(default_factory=list)
    sections_missing: list[str] = Field(default_factory=list)
    extraction_method: str = "pdfplumber"
    is_machine_readable: bool      # True si word_count >= 150 mots extraits


class ProfileStrength(BaseModel):
    level: str          # "Solide" | "Correct" | "À renforcer"
    score: int          # 0-100

    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    priority: int        # 1 = Fort, 2 = Moyen, 3 = Faible
    impact: str          # "Fort" | "Moyen" | "Faible"
    action: str
    why: str


class CriterionResult(BaseModel):
    criterion_id: str
    label: str
    weight: int
    required: bool
    score: int                          # 0 ou 100 (binaire)
    evidence: list[str] = Field(default_factory=list)
    weighted_score: float = 0.0         # score * weight / 100


class CVQualityReport(BaseModel):
    ats_readability: ATSReadability
    profile_strength: ProfileStrength
    recommendations: list[Recommendation] = Field(default_factory=list)

    # Timeline carrière (conservé)
    total_experience_years: float = 0.0
    career_start_year: Optional[int] = None
    most_recent_role: Optional[str] = None
    education_years: int = 0
    career_gaps: list[str] = Field(default_factory=list)

    # Extraction pipeline (conservé pour compatibilité)
    extraction_method: str = "pdfplumber"
    extraction_confidence: float = 1.0

    # Phase B — scoring adaptatif par secteur/métier
    detected_sector: Optional[str] = None
    detected_profile: Optional[str] = None
    detection_confidence: float = 0.0
    criteria_results: list[CriterionResult] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    semantic_similarity: float
    keyword_match: float
    structure_completeness: float


class ScoringResult(BaseModel):
    overall_score: float
    breakdown: ScoreBreakdown
    matched_keywords: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    feedback: Optional[str] = None


class ATSResponse(BaseModel):
    scoring_result: ScoringResult
    parsed_cv: NormalizedCV
    processing_time_seconds: float
    detected_sector: Optional[str] = None
    detected_profile: Optional[str] = None
    detection_confidence: float = 0.0
    criteria_results: list[CriterionResult] = Field(default_factory=list)


class JobListing(BaseModel):
    title: str
    company: str
    location: str
    description: str
    url: str
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    contract_type: Optional[str] = None
    source: str = "adzuna"
    source_color: str = "#6366f1"


class RankedJobMatch(BaseModel):
    job: JobListing
    scoring_result: ScoringResult
