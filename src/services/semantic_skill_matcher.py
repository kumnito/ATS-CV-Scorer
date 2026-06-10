"""Semantic matching of ESCO multi-word skill phrases against CV text.

Substring matching (see _ALL_SKILLS in nlp_pipeline) works well for short
tool/product names ("python", "aws") but fails for ESCO "essential skill"
phrases ("produce sales reports", "communicate with customers"), which
almost never appear verbatim in a CV. This module compares sentence
embeddings instead: each CV line is encoded and compared via cosine
similarity to the precomputed embeddings of generated ESCO phrases
(GENERATED_SKILL_EMBEDDINGS, populated from lexicons_embeddings.npy).

If no generated embeddings are available (lexicons_embeddings.npy absent —
e.g. make update-lexicons was never run), match_semantic_skills returns []
without loading the sentence-transformers model.
"""

from sklearn.metrics.pairwise import cosine_similarity

from src.core.lexicons import (
    GENERATED_SKILL_EMBEDDINGS,
    GENERATED_SKILL_PHRASES,
    SEMANTIC_SKILL_MATCH_THRESHOLD,
)

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def match_semantic_skills(
    text: str, threshold: float = SEMANTIC_SKILL_MATCH_THRESHOLD
) -> list[str]:
    """Return generated ESCO skill phrases whose embedding matches a CV line.

    A phrase is matched if its cosine similarity to the best-matching CV
    line is >= threshold.
    """
    if GENERATED_SKILL_EMBEDDINGS is None or not GENERATED_SKILL_PHRASES:
        return []

    lines = [
        line.strip() for line in text.splitlines() if len(line.strip().split()) >= 2
    ]
    if not lines:
        return []

    model = _get_model()
    line_embeddings = model.encode(lines, convert_to_numpy=True)
    similarities = cosine_similarity(line_embeddings, GENERATED_SKILL_EMBEDDINGS)
    best_per_phrase = similarities.max(axis=0)

    return [
        phrase
        for phrase, score in zip(GENERATED_SKILL_PHRASES, best_per_phrase)
        if score >= threshold
    ]
