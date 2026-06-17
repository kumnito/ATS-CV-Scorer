"""Process-wide SentenceTransformer registry.

Provides a single shared instance of all-MiniLM-L6-v2 across
SemanticScorer, SectorDetector, and semantic_skill_matcher —
avoiding three separate model loads (~180 MB RAM saving).
"""
import threading

from sentence_transformers import SentenceTransformer

_minilm_model: SentenceTransformer | None = None
_lock = threading.Lock()


def get_minilm() -> SentenceTransformer:
    """Return the process-wide SentenceTransformer('all-MiniLM-L6-v2') instance."""
    global _minilm_model
    if _minilm_model is None:
        with _lock:
            if _minilm_model is None:
                _minilm_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _minilm_model
