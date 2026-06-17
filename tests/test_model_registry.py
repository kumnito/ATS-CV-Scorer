"""Tests for the process-wide MiniLM singleton (model_registry)."""

from src.core.model_registry import get_minilm


def test_repeated_calls_return_same_instance():
    assert get_minilm() is get_minilm()


def test_singleton_shared_across_services():
    """SemanticScorer, SectorDetector, and semantic_skill_matcher share one instance."""
    from src.services.sector_detector import SectorDetector
    from src.services.semantic_scorer import SemanticScorer
    from src.services.semantic_skill_matcher import _get_model as ssm_get_model

    registry_model = get_minilm()
    assert SemanticScorer().model is registry_model
    assert SectorDetector()._get_model() is registry_model
    assert ssm_get_model() is registry_model
