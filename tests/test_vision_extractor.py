"""Tests for VisionExtractor (cascade level 3 — Claude Vision LLM)."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.services.vision_extractor import VisionExtractor


def test_vision_json_parsing():
    """extract() parse correctement la réponse JSON de Claude."""
    extractor = VisionExtractor()
    vision_json = """{
  "header": {"name": "Alice Martin", "email": "alice@example.com",
             "title": "Data Scientist", "phone": null, "location": "Paris",
             "postal_code": null, "github": null, "linkedin": null},
  "summary": "Expérimentée en NLP et ML.",
  "skills": {"ml": ["pytorch", "scikit-learn"], "mlops": ["docker"],
             "cloud": ["aws"], "languages": ["python"], "data": ["sql"],
             "other": [], "commerce": []},
  "experience": [{"title": "Data Scientist", "company": "TechCorp",
                  "date_start": "2021", "date_end": "2024",
                  "is_current": false, "bullets": ["Développé des modèles NLP"]}],
  "education": [{"degree": "Master ML", "school": "Sorbonne",
                 "date_start": "2019", "date_end": "2021", "is_current": false}],
  "projects": [], "languages": ["Français", "Anglais"]
}"""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=vision_json)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    # Patch pdf2image (non installé) + Anthropic client + settings
    mock_img = MagicMock()
    mock_img.save = lambda buf, format: buf.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    mock_pdf2image = MagicMock()
    mock_pdf2image.convert_from_path.return_value = [mock_img]

    with patch("src.services.vision_extractor.settings") as mock_settings, \
         patch.dict(sys.modules, {"pdf2image": mock_pdf2image}), \
         patch("anthropic.Anthropic", return_value=mock_client):
        mock_settings.anthropic_api_key = "sk-test-key"
        mock_settings.claude_model = "claude-sonnet-4-6"
        cv = extractor.extract("dummy.pdf")

    assert cv.header.name == "Alice Martin"
    assert cv.header.email == "alice@example.com"
    assert "pytorch" in cv.skills.ml
    assert cv.extraction_method == "vision_llm"
    assert cv.extraction_confidence == 0.95


def test_vision_skipped_without_api_key():
    """extract() lève RuntimeError si aucune clé API."""
    extractor = VisionExtractor()
    mock_pdf2image = MagicMock()
    with patch("src.services.vision_extractor.settings") as mock_settings, \
         patch.dict(sys.modules, {"pdf2image": mock_pdf2image}):
        mock_settings.anthropic_api_key = ""
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            extractor.extract("dummy.pdf")
