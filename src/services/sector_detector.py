"""Sector detection from a NormalizedCV.

SectorDetector.detect() scores each registered SectorProfile across three
signals (title × 0.4, skills × 0.35, experience × 0.25) and returns the
best match with its top-3 alternatives.

MiniLM is loaded lazily and shared at module level. The aliases embedding
matrix is built once per SectorDetector instance on first detect() call.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.core.schemas import NormalizedCV
from src.core.sector_registry import ALL_PROFILES, GENERIC_PROFILE

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD: float = 0.3

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


@dataclass
class SectorDetectionResult:
    profile_id: str                       # e.g. "ml_engineer"
    sector: str                           # e.g. "Informatique & Digital"
    job_title: str                        # e.g. "Machine Learning Engineer"
    confidence: float                     # 0.0 – 1.0
    alternatives: list[tuple[str, float]] # top-3 other profiles [(id, score)]


class SectorDetector:
    """Detects the best matching SectorProfile for a NormalizedCV.

    Parameters
    ----------
    model:
        Optional pre-loaded SentenceTransformer (injected in tests to skip
        MiniLM startup).
    """

    def __init__(self, model=None) -> None:
        self._injected_model = model
        self._aliases_matrix: Optional[np.ndarray] = None
        self._alias_profile_ids: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, cv: NormalizedCV) -> SectorDetectionResult:
        title_scores = self._compute_title_scores(cv)

        scores: dict[str, float] = {}
        for pid, profile in ALL_PROFILES.items():
            s_title = title_scores.get(pid, 0.0)
            s_skills = self._score_skills(cv, profile.detection_keywords)
            s_exp = self._score_experience(cv, profile.detection_keywords)
            scores[pid] = 0.4 * s_title + 0.35 * s_skills + 0.25 * s_exp

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_id, best_score = ranked[0]
        confidence = min(1.0, max(0.0, best_score))

        logger.debug(
            "SectorDetector: best=%s (%.3f), title_keys=%d",
            best_id, confidence, len(title_scores),
        )

        if confidence < _CONFIDENCE_THRESHOLD:
            return SectorDetectionResult(
                profile_id=GENERIC_PROFILE.id,
                sector=GENERIC_PROFILE.sector,
                job_title=GENERIC_PROFILE.job_title,
                confidence=round(confidence, 3),
                alternatives=[(pid, round(s, 3)) for pid, s in ranked[:3]],
            )

        best_profile = ALL_PROFILES[best_id]
        return SectorDetectionResult(
            profile_id=best_id,
            sector=best_profile.sector,
            job_title=best_profile.job_title,
            confidence=round(confidence, 3),
            alternatives=[(pid, round(s, 3)) for pid, s in ranked[1:4]],
        )

    # ------------------------------------------------------------------
    # Title scoring — single MiniLM encode + batch cosine sim
    # ------------------------------------------------------------------

    def _compute_title_scores(self, cv: NormalizedCV) -> dict[str, float]:
        if not cv.job_title:
            return {}

        self._ensure_aliases_matrix()
        if self._aliases_matrix is None or len(self._aliases_matrix) == 0:
            return {}

        model = self._get_model()
        title_emb = model.encode([cv.job_title], convert_to_numpy=True)
        sims = cosine_similarity(title_emb, self._aliases_matrix)[0]

        result: dict[str, float] = {}
        for i, pid in enumerate(self._alias_profile_ids):
            if pid not in result or sims[i] > result[pid]:
                result[pid] = float(sims[i])
        return result

    # ------------------------------------------------------------------
    # Skills scoring — keyword presence in skills_flat + skills section
    # ------------------------------------------------------------------

    def _score_skills(self, cv: NormalizedCV, keywords: list[str]) -> float:
        if not keywords:
            return 0.0
        # skills_flat: tech terms (NLPPipeline-detected)
        # sections["skills"]: raw skills section text (non-tech CVs list terms here)
        check_text = (
            " ".join(cv.skills_flat) + " " + cv.sections.get("skills", "")
        ).lower()
        matched = sum(1 for kw in keywords if kw.lower() in check_text)
        return min(1.0, matched / len(keywords))

    # ------------------------------------------------------------------
    # Experience scoring — keyword frequency in bullets + experience section
    # ------------------------------------------------------------------

    def _score_experience(self, cv: NormalizedCV, keywords: list[str]) -> float:
        if not keywords:
            return 0.0
        bullets = " ".join(b for e in cv.experience for b in e.bullets).lower()
        exp_section = cv.sections.get("experience", "").lower()
        check_text = (bullets + " " + exp_section).strip()
        if not check_text:
            check_text = cv.raw_text.lower()
        matched = sum(1 for kw in keywords if kw.lower() in check_text)
        return min(1.0, matched / len(keywords))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_forced_result(profile_id: str) -> "SectorDetectionResult":
        """Build a SectorDetectionResult for a manually chosen profile_id."""
        from src.core.sector_registry import ALL_PROFILES, GENERIC_PROFILE  # deferred
        profile = ALL_PROFILES.get(profile_id) or GENERIC_PROFILE
        return SectorDetectionResult(
            profile_id=profile.id,
            sector=profile.sector,
            job_title=profile.job_title,
            confidence=1.0,
            alternatives=[],
        )

    def _get_model(self):
        if self._injected_model is not None:
            return self._injected_model
        return _get_model()

    def _ensure_aliases_matrix(self) -> None:
        if self._aliases_matrix is not None:
            return

        profile_aliases = [
            (pid, alias)
            for pid, profile in ALL_PROFILES.items()
            for alias in profile.aliases
        ]

        if not profile_aliases:
            self._aliases_matrix = np.zeros((0, 384), dtype=np.float32)
            return

        pids = [pa[0] for pa in profile_aliases]
        texts = [pa[1] for pa in profile_aliases]

        model = self._get_model()
        logger.info("SectorDetector: encoding %d aliases…", len(texts))
        self._aliases_matrix = model.encode(texts, convert_to_numpy=True)
        self._alias_profile_ids = pids
        logger.info("SectorDetector: aliases matrix ready (%s)", self._aliases_matrix.shape)
