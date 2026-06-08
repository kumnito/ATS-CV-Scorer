import re
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.core.schemas import ParsedCV, ScoreBreakdown, ScoringResult

_STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "that",
    "this",
    "these",
    "those",
    "we",
    "you",
    "they",
    "it",
    "he",
    "she",
    "our",
    "your",
    "their",
    "its",
}

_IMPORTANT_SECTIONS = {"experience", "education", "skills"}
_NICE_SECTIONS = {"summary", "projects", "certifications"}


class SemanticScorer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def score(self, parsed_cv: ParsedCV, job_description: str) -> ScoringResult:
        semantic = self._semantic_similarity(parsed_cv.raw_text, job_description)
        jd_keywords = _extract_jd_keywords(job_description)
        keyword = _keyword_match_score(parsed_cv.raw_text, jd_keywords)
        structure = _structure_score(parsed_cv)

        overall = round(semantic * 35 + keyword * 40 + structure * 25, 1)
        cv_lower = parsed_cv.raw_text.lower()

        return ScoringResult(
            overall_score=overall,
            breakdown=ScoreBreakdown(
                semantic_similarity=round(semantic * 100, 1),
                keyword_match=round(keyword * 100, 1),
                structure_completeness=round(structure * 100, 1),
            ),
            matched_keywords=[kw for kw in jd_keywords if kw.lower() in cv_lower][:20],
            missing_keywords=[kw for kw in jd_keywords if kw.lower() not in cv_lower][
                :20
            ],
        )

    def _semantic_similarity(self, cv_text: str, jd_text: str) -> float:
        cv_emb = self.model.encode([cv_text[:512]], convert_to_numpy=True)
        jd_emb = self.model.encode([jd_text[:512]], convert_to_numpy=True)
        score = cosine_similarity(cv_emb, jd_emb)[0][0]
        return float(np.clip(score, 0.0, 1.0))


def _keyword_match_score(cv_text: str, jd_keywords: list[str]) -> float:
    if not jd_keywords:
        return 0.5
    cv_lower = cv_text.lower()
    matched = sum(1 for kw in jd_keywords if kw.lower() in cv_lower)
    return matched / len(jd_keywords)


def _structure_score(parsed_cv: ParsedCV) -> float:
    score = sum(
        0.25
        for s in _IMPORTANT_SECTIONS
        if s in parsed_cv.sections and parsed_cv.sections[s]
    )
    score += sum(
        0.083
        for s in _NICE_SECTIONS
        if s in parsed_cv.sections and parsed_cv.sections[s]
    )
    if parsed_cv.skills:
        score += min(len(parsed_cv.skills) / 10, 0.15)
    return min(1.0, score)


def _extract_jd_keywords(jd_text: str) -> list[str]:
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.]+\b", jd_text)
    filtered = [w for w in words if w.lower() not in _STOP_WORDS and len(w) > 2]
    counter = Counter(w.lower() for w in filtered)
    return [word for word, _ in counter.most_common(40)]
