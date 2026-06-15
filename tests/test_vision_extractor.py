"""Tests for VisionExtractor (cascade level 3 — Claude Vision LLM)."""

import sys
from unittest.mock import MagicMock, patch

import anthropic
import httpx
import pytest

from src.services.vision_extractor import VisionExtractionError, VisionExtractor


def _mock_pdf2image() -> MagicMock:
    mock_img = MagicMock()
    mock_img.save = lambda buf, format: buf.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    mock_pdf2image = MagicMock()
    mock_pdf2image.convert_from_path.return_value = [mock_img]
    return mock_pdf2image


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


def test_vision_extractor_uses_configured_timeout():
    """L'appel Anthropic Vision est borné par settings.vision_timeout_seconds."""
    extractor = VisionExtractor()
    vision_json = '{"header": {"name": "Alice"}, "skills": {}, "experience": [], "education": []}'

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=vision_json)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    mock_img = MagicMock()
    mock_img.save = lambda buf, format: buf.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    mock_pdf2image = MagicMock()
    mock_pdf2image.convert_from_path.return_value = [mock_img]

    with patch("src.services.vision_extractor.settings") as mock_settings, \
         patch.dict(sys.modules, {"pdf2image": mock_pdf2image}), \
         patch("anthropic.Anthropic", return_value=mock_client) as mock_anthropic:
        mock_settings.anthropic_api_key = "sk-test-key"
        mock_settings.claude_model = "claude-sonnet-4-6"
        mock_settings.vision_timeout_seconds = 60
        extractor.extract("dummy.pdf")

    mock_anthropic.assert_called_once_with(api_key="sk-test-key", timeout=60)


def test_vision_raises_extraction_error_on_rate_limit():
    extractor = VisionExtractor()
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request)
    rate_limit_error = anthropic.RateLimitError("rate limited", response=response, body=None)

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = rate_limit_error

    with patch("src.services.vision_extractor.settings") as mock_settings, \
         patch.dict(sys.modules, {"pdf2image": _mock_pdf2image()}), \
         patch("anthropic.Anthropic", return_value=mock_client):
        mock_settings.anthropic_api_key = "sk-test-key"
        mock_settings.claude_model = "claude-sonnet-4-6"
        mock_settings.vision_timeout_seconds = 60
        with pytest.raises(VisionExtractionError, match="surchargé"):
            extractor.extract("dummy.pdf")


def test_vision_raises_extraction_error_on_connection_error():
    extractor = VisionExtractor()
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    connection_error = anthropic.APIConnectionError(request=request)

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = connection_error

    with patch("src.services.vision_extractor.settings") as mock_settings, \
         patch.dict(sys.modules, {"pdf2image": _mock_pdf2image()}), \
         patch("anthropic.Anthropic", return_value=mock_client):
        mock_settings.anthropic_api_key = "sk-test-key"
        mock_settings.claude_model = "claude-sonnet-4-6"
        mock_settings.vision_timeout_seconds = 60
        with pytest.raises(VisionExtractionError, match="contacter"):
            extractor.extract("dummy.pdf")


def test_vision_raises_extraction_error_on_api_status_error():
    extractor = VisionExtractor()
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(500, request=request)
    status_error = anthropic.APIStatusError("server error", response=response, body=None)

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = status_error

    with patch("src.services.vision_extractor.settings") as mock_settings, \
         patch.dict(sys.modules, {"pdf2image": _mock_pdf2image()}), \
         patch("anthropic.Anthropic", return_value=mock_client):
        mock_settings.anthropic_api_key = "sk-test-key"
        mock_settings.claude_model = "claude-sonnet-4-6"
        mock_settings.vision_timeout_seconds = 60
        with pytest.raises(VisionExtractionError, match="erreur"):
            extractor.extract("dummy.pdf")


def test_vision_skipped_without_api_key():
    """extract() lève RuntimeError si aucune clé API."""
    extractor = VisionExtractor()
    mock_pdf2image = MagicMock()
    with patch("src.services.vision_extractor.settings") as mock_settings, \
         patch.dict(sys.modules, {"pdf2image": mock_pdf2image}):
        mock_settings.anthropic_api_key = ""
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            extractor.extract("dummy.pdf")
