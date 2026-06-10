"""Tests for semantic skill matching — sentence-transformers model is mocked."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.services import semantic_skill_matcher
from src.services.semantic_skill_matcher import match_semantic_skills


def _mock_model(line_embeddings: np.ndarray) -> MagicMock:
    model = MagicMock()
    model.encode.return_value = line_embeddings
    return model


def test_returns_empty_when_no_generated_embeddings():
    with patch.object(semantic_skill_matcher, "GENERATED_SKILL_EMBEDDINGS", None):
        assert match_semantic_skills("Some CV text with several words") == []


def test_returns_empty_when_no_multi_word_lines():
    embeddings = np.array([[1.0, 0.0]])
    with (
        patch.object(semantic_skill_matcher, "GENERATED_SKILL_EMBEDDINGS", embeddings),
        patch.object(
            semantic_skill_matcher, "GENERATED_SKILL_PHRASES", ["sales strategies"]
        ),
    ):
        # Single-word lines are skipped before the model is even loaded.
        assert match_semantic_skills("Python\nDocker\nSQL") == []


def test_matches_phrase_above_threshold():
    skill_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    line_embeddings = np.array([[1.0, 0.0]])  # identical to first skill -> sim=1.0

    with (
        patch.object(
            semantic_skill_matcher, "GENERATED_SKILL_EMBEDDINGS", skill_embeddings
        ),
        patch.object(
            semantic_skill_matcher,
            "GENERATED_SKILL_PHRASES",
            ["customer relationship management", "engineering principles"],
        ),
        patch.object(
            semantic_skill_matcher,
            "_get_model",
            return_value=_mock_model(line_embeddings),
        ),
    ):
        result = match_semantic_skills(
            "Manage long-term client relationships", threshold=0.5
        )

    assert result == ["customer relationship management"]


def test_phrase_below_threshold_is_excluded():
    skill_embeddings = np.array([[1.0, 0.0]])
    line_embeddings = np.array([[0.0, 1.0]])  # orthogonal -> sim=0.0

    with (
        patch.object(
            semantic_skill_matcher, "GENERATED_SKILL_EMBEDDINGS", skill_embeddings
        ),
        patch.object(
            semantic_skill_matcher,
            "GENERATED_SKILL_PHRASES",
            ["sales promotion techniques"],
        ),
        patch.object(
            semantic_skill_matcher,
            "_get_model",
            return_value=_mock_model(line_embeddings),
        ),
    ):
        result = match_semantic_skills("Two word line here", threshold=0.5)

    assert result == []
