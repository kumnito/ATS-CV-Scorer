import os
import tempfile
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.core.schemas import ATSResponse, RankedJobMatch
from src.services.claude_feedback import ClaudeFeedback
from src.services.job_matcher import find_matching_jobs
from src.services.job_search import JobSearchService
from src.services.nlp_pipeline import NLPPipeline
from src.services.pdf_extractor import PDFExtractor
from src.services.semantic_scorer import SemanticScorer

app = FastAPI(title="ATS CV Scorer API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_pdf_extractor = PDFExtractor()
_nlp_pipeline = NLPPipeline()
_semantic_scorer = SemanticScorer()
_job_search_service = JobSearchService(
    app_id=settings.adzuna_id,
    app_key=settings.adzuna_api_key,
    country=settings.adzuna_country,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/score", response_model=ATSResponse)
async def score_cv(
    cv_file: UploadFile = File(..., description="PDF résumé/CV"),
    job_description: str = Form(..., description="Full job description text"),
    include_feedback: bool = Form(False, description="Include Claude AI feedback"),
):
    if cv_file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    content = await cv_file.read()
    if len(content) > settings.max_pdf_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"File exceeds {settings.max_pdf_size_mb} MB limit."
        )

    start = time.perf_counter()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        cv_text = _pdf_extractor.extract(tmp_path)
        if not cv_text:
            raise HTTPException(
                status_code=422, detail="Could not extract text from PDF."
            )

        parsed_cv = _nlp_pipeline.parse_cv(cv_text)
        scoring_result = _semantic_scorer.score(parsed_cv, job_description)

        if include_feedback and settings.anthropic_api_key:
            feedback_svc = ClaudeFeedback()
            scoring_result.feedback = feedback_svc.generate_feedback(
                parsed_cv, job_description, scoring_result
            )

        return ATSResponse(
            scoring_result=scoring_result,
            parsed_cv=parsed_cv,
            processing_time_seconds=round(time.perf_counter() - start, 3),
        )
    finally:
        os.unlink(tmp_path)


@app.post("/find-jobs", response_model=list[RankedJobMatch])
async def find_jobs(
    cv_file: UploadFile = File(..., description="PDF résumé/CV"),
    max_results: int = Form(
        20, description="Maximum number of job listings to fetch and score"
    ),
):
    if cv_file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    content = await cv_file.read()
    if len(content) > settings.max_pdf_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"File exceeds {settings.max_pdf_size_mb} MB limit."
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        cv_text = _pdf_extractor.extract(tmp_path)
        if not cv_text:
            raise HTTPException(
                status_code=422, detail="Could not extract text from PDF."
            )

        parsed_cv = _nlp_pipeline.parse_cv(cv_text)
        return find_matching_jobs(
            parsed_cv, _job_search_service, _semantic_scorer, max_results=max_results
        )
    finally:
        os.unlink(tmp_path)
