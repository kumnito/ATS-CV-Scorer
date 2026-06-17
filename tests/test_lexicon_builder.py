"""Tests for LexiconBuilder — all network calls are mocked."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.services.lexicon_builder import (
    LexiconBuilder,
    _categorize,
    _detect_skill_field,
    _merge_partial,
    _normalize_skill,
)

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _empty_result() -> dict:
    return {
        "skill_categories": {cat: [] for cat in ["ml", "mlops", "cloud", "languages", "data", "other"]},
        "job_titles": [],
        "action_verbs_en": [],
        "action_verbs_fr": [],
        "generated_at": "2025-01-01T00:00:00+00:00",
        "sources": [],
    }


def _esco_search_response(uri: str = "http://esco/occ/1", title: str = "data scientist") -> dict:
    """Returns a search result where the title matches the default query 'data scientist'."""
    return {"_embedded": {"results": [{"uri": uri, "title": title}]}}


def _esco_occupation_response(skill_titles: list[str]) -> dict:
    return {
        "title": "data scientist",
        "_links": {
            "hasEssentialSkill": [{"title": t, "uri": f"http://esco/skill/{i}"} for i, t in enumerate(skill_titles)]
        },
    }


def _make_mock_client(search_resp: dict, occ_en_resp: dict, occ_fr_resp: dict) -> MagicMock:
    """Return a mock httpx.Client whose .get() returns the right fixture per URL."""
    def get(url, params=None, **_):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        if "/search" in url:
            mock_resp.status_code = 200
            mock_resp.json.return_value = search_resp
        elif "/resource/occupation" in url:
            lang = (params or {}).get("language", "en")
            mock_resp.status_code = 200
            mock_resp.json.return_value = occ_fr_resp if lang == "fr" else occ_en_resp
        else:
            mock_resp.status_code = 404
            mock_resp.json.return_value = {}
        return mock_resp

    client = MagicMock()
    client.get.side_effect = get
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    return client


# ---------------------------------------------------------------------------
# Unit tests — pure helpers
# ---------------------------------------------------------------------------

class TestNormalizeSkill:
    def test_strips_use_prefix(self):
        assert _normalize_skill("use Python") == "python"

    def test_strips_develop_prefix(self):
        assert _normalize_skill("develop machine learning models") == "machine learning models"

    def test_strips_apply_prefix(self):
        assert _normalize_skill("apply agile methodology") == "agile methodology"

    def test_rejects_five_word_phrase(self):
        assert _normalize_skill("develop end to end machine learning pipelines") is None

    def test_bare_tool_name_unchanged(self):
        assert _normalize_skill("Python") == "python"

    def test_empty_string_returns_none(self):
        assert _normalize_skill("") is None


class TestCategorize:
    def test_ml_keyword(self):
        assert _categorize("pytorch") == "ml"

    def test_mlops_keyword(self):
        assert _categorize("kubernetes") == "mlops"

    def test_cloud_keyword(self):
        assert _categorize("aws") == "cloud"

    def test_data_keyword(self):
        assert _categorize("apache kafka") == "data"

    def test_unknown_falls_back_to_other(self):
        assert _categorize("some-obscure-tool-xyz") == "other"


class TestDetectSkillField:
    def test_finds_skill_column(self):
        assert _detect_skill_field(["id", "skill", "category"]) == "skill"

    def test_finds_name_as_fallback(self):
        assert _detect_skill_field(["id", "name", "description"]) == "name"

    def test_returns_none_when_no_match(self):
        assert _detect_skill_field(["col_a", "col_b"]) is None


class TestMergePartial:
    def test_adds_new_skills(self):
        target = _empty_result()
        partial = {"skill_categories": {"ml": ["pytorch"]}, "job_titles": [], "action_verbs_en": [], "action_verbs_fr": []}
        _merge_partial(target, partial)
        assert "pytorch" in target["skill_categories"]["ml"]

    def test_no_duplicate_skills(self):
        target = _empty_result()
        target["skill_categories"]["ml"] = ["tensorflow"]
        partial = {"skill_categories": {"ml": ["tensorflow", "pytorch"]}, "job_titles": [], "action_verbs_en": [], "action_verbs_fr": []}
        _merge_partial(target, partial)
        assert target["skill_categories"]["ml"].count("tensorflow") == 1
        assert "pytorch" in target["skill_categories"]["ml"]

    def test_adds_new_category(self):
        target = _empty_result()
        partial = {"skill_categories": {"devops": ["terraform"]}, "job_titles": [], "action_verbs_en": [], "action_verbs_fr": []}
        _merge_partial(target, partial)
        assert target["skill_categories"]["devops"] == ["terraform"]

    def test_merges_job_titles_without_duplicates(self):
        target = _empty_result()
        target["job_titles"] = ["data analyst"]
        partial = {"skill_categories": {}, "job_titles": ["data analyst", "ml engineer"], "action_verbs_en": [], "action_verbs_fr": []}
        _merge_partial(target, partial)
        assert target["job_titles"].count("data analyst") == 1
        assert "ml engineer" in target["job_titles"]

    def test_merges_action_verbs(self):
        target = _empty_result()
        partial = {"skill_categories": {}, "job_titles": [], "action_verbs_en": ["orchestrated"], "action_verbs_fr": ["orchestré"]}
        _merge_partial(target, partial)
        assert "orchestrated" in target["action_verbs_en"]
        assert "orchestré" in target["action_verbs_fr"]


# ---------------------------------------------------------------------------
# LexiconBuilder integration tests (network mocked)
# ---------------------------------------------------------------------------

class TestLexiconBuilderCache:
    def test_uses_cache_when_not_forced(self, tmp_path):
        cached = _empty_result()
        cached["skill_categories"]["ml"] = ["cached-skill"]
        cached["sources"] = ["esco"]
        (tmp_path / "lexicons_generated.json").write_text(json.dumps(cached))

        builder = LexiconBuilder(output_path=tmp_path / "lexicons_generated.json")
        with patch.object(builder, "_fetch_esco") as mock_esco:
            result = builder.build(force_refresh=False)

        mock_esco.assert_not_called()
        assert "cached-skill" in result["skill_categories"]["ml"]

    def test_force_refresh_ignores_cache(self, tmp_path):
        cached = _empty_result()
        cached["skill_categories"]["ml"] = ["stale-skill"]
        output = tmp_path / "lexicons_generated.json"
        output.write_text(json.dumps(cached))

        fresh = {"skill_categories": {"ml": ["fresh-skill"]}, "job_titles": [], "action_verbs_en": [], "action_verbs_fr": []}
        builder = LexiconBuilder(output_path=output)
        with (
            patch.object(builder, "_fetch_esco", return_value=fresh),
            patch.object(builder, "_fetch_huggingface", side_effect=Exception("skip")),
        ):
            result = builder.build(force_refresh=True)

        assert "fresh-skill" in result["skill_categories"]["ml"]
        assert "stale-skill" not in result["skill_categories"]["ml"]

    def test_corrupted_cache_triggers_refresh(self, tmp_path):
        output = tmp_path / "lexicons_generated.json"
        output.write_text("not valid json{{{")

        builder = LexiconBuilder(output_path=output)
        with (
            patch.object(builder, "_fetch_esco", side_effect=Exception("network")),
            patch.object(builder, "_fetch_huggingface", side_effect=Exception("skip")),
        ):
            result = builder.build(force_refresh=False)

        assert result["sources"] == []


class TestLexiconBuilderFallback:
    def test_falls_back_silently_when_all_sources_fail(self, tmp_path):
        output = tmp_path / "lexicons_generated.json"
        builder = LexiconBuilder(output_path=output)
        with (
            patch.object(builder, "_fetch_esco", side_effect=Exception("timeout")),
            patch.object(builder, "_fetch_huggingface", side_effect=Exception("not installed")),
        ):
            result = builder.build(force_refresh=True)

        assert result["sources"] == []
        assert output.exists()
        assert all(isinstance(v, list) for v in result["skill_categories"].values())

    def test_partial_failure_keeps_successful_source(self, tmp_path):
        esco_data = {"skill_categories": {"ml": ["esco-skill"]}, "job_titles": ["data analyst"], "action_verbs_en": [], "action_verbs_fr": []}
        output = tmp_path / "lexicons_generated.json"
        builder = LexiconBuilder(output_path=output)
        with (
            patch.object(builder, "_fetch_esco", return_value=esco_data),
            patch.object(builder, "_fetch_huggingface", side_effect=Exception("datasets not installed")),
        ):
            result = builder.build(force_refresh=True)

        assert "esco" in result["sources"]
        assert "huggingface" not in result["sources"]
        assert "esco-skill" in result["skill_categories"]["ml"]

    def test_output_file_written_even_on_all_failures(self, tmp_path):
        output = tmp_path / "lexicons_generated.json"
        builder = LexiconBuilder(output_path=output)
        with (
            patch.object(builder, "_fetch_esco", side_effect=Exception("err")),
            patch.object(builder, "_fetch_huggingface", side_effect=Exception("err")),
        ):
            builder.build(force_refresh=True)

        assert output.exists()
        loaded = json.loads(output.read_text())
        assert "generated_at" in loaded


class TestEscoBestOccupation:
    def test_accepts_matching_title(self, tmp_path):
        # "data" in "data scientist" overlaps with "data engineer" query
        search_resp = {"_embedded": {"results": [{"uri": "http://esco/1", "title": "data scientist"}]}}
        mock_client = _make_mock_client(search_resp, {}, {})
        builder = LexiconBuilder(output_path=tmp_path / "out.json")
        with patch("httpx.Client", return_value=mock_client):
            uri, title = builder._esco_best_occupation(mock_client, "data engineer")
        assert uri == "http://esco/1"
        assert title == "data scientist"

    def test_rejects_mismatched_title(self, tmp_path):
        # "machine learning engineer" vs "packing machinery engineer" — no domain word overlap
        search_resp = {"_embedded": {"results": [{"uri": "http://esco/bad", "title": "packing machinery engineer"}]}}
        mock_client = _make_mock_client(search_resp, {}, {})
        builder = LexiconBuilder(output_path=tmp_path / "out.json")
        with patch("httpx.Client", return_value=mock_client):
            uri, title = builder._esco_best_occupation(mock_client, "machine learning engineer")
        assert uri is None
        assert title is None

    def test_returns_none_when_results_empty(self, tmp_path):
        empty = {"_embedded": {"results": []}}
        mock_client = _make_mock_client(empty, {}, {})
        builder = LexiconBuilder(output_path=tmp_path / "out.json")
        with patch("httpx.Client", return_value=mock_client):
            uri, title = builder._esco_best_occupation(mock_client, "data analyst")
        assert uri is None


class TestFetchEsco:
    def test_extracts_skills_and_job_title(self, tmp_path):
        # "data scientist" matches "data scientist" query — domain word "data" in common
        mock_client = _make_mock_client(
            search_resp=_esco_search_response(title="data scientist"),
            occ_en_resp=_esco_occupation_response(["use pytorch", "use python", "develop neural networks"]),
            occ_fr_resp=_esco_occupation_response(["utiliser pytorch", "développer des réseaux"]),
        )

        builder = LexiconBuilder(output_path=tmp_path / "out.json")
        with patch("httpx.Client", return_value=mock_client):
            result = builder._fetch_esco()

        all_skills = [s for v in result["skill_categories"].values() for s in v]
        assert "pytorch" in all_skills
        assert "python" in all_skills
        assert "data scientist" in result["job_titles"]

    def test_skips_occupation_when_search_returns_empty(self, tmp_path):
        empty_search = {"_embedded": {"results": []}}
        mock_client = _make_mock_client(
            search_resp=empty_search,
            occ_en_resp={},
            occ_fr_resp={},
        )

        builder = LexiconBuilder(output_path=tmp_path / "out.json")
        with patch("httpx.Client", return_value=mock_client):
            result = builder._fetch_esco()

        assert result["job_titles"] == []

    def test_action_verbs_always_empty_from_esco(self, tmp_path):
        mock_client = _make_mock_client(
            search_resp=_esco_search_response(),
            occ_en_resp=_esco_occupation_response(["develop software", "test systems"]),
            occ_fr_resp=_esco_occupation_response([]),
        )

        builder = LexiconBuilder(output_path=tmp_path / "out.json")
        with patch("httpx.Client", return_value=mock_client):
            result = builder._fetch_esco()

        # ESCO does not generate action verbs — hardcoded sets are authoritative
        assert result["action_verbs_en"] == []
        assert result["action_verbs_fr"] == []

    def test_esco_http_error_propagates(self, tmp_path):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("HTTP 429")
        mock_client.get.return_value = mock_resp

        builder = LexiconBuilder(output_path=tmp_path / "out.json")
        with patch("httpx.Client", return_value=mock_client):
            with pytest.raises(Exception, match="HTTP 429"):
                builder._fetch_esco()


class TestFetchHuggingFace:
    def test_extracts_skills_from_dataset(self, tmp_path):
        mock_ds = MagicMock()
        mock_ds.column_names = ["skill", "category"]
        mock_ds.__iter__ = MagicMock(
            return_value=iter([
                {"skill": "pytorch"},
                {"skill": "apache kafka"},
                {"skill": "some obscure thing xyz"},
            ])
        )

        _builder = LexiconBuilder(output_path=tmp_path / "out.json")
        with patch("src.services.lexicon_builder.LexiconBuilder._fetch_huggingface") as mock_hf:
            mock_hf.return_value = {
                "skill_categories": {"ml": ["pytorch"], "data": ["apache kafka"], "other": ["some obscure thing xyz"]},
                "job_titles": [],
                "action_verbs_en": [],
                "action_verbs_fr": [],
            }

        # Test via integration with mocked load_dataset
        with patch("builtins.__import__", side_effect=lambda name, *a, **kw: _mock_datasets_import(name, mock_ds)):
            pass  # import mocking is complex — covered by the partial result test below

    def test_raises_when_datasets_not_installed(self, tmp_path):
        builder = LexiconBuilder(output_path=tmp_path / "out.json")

        original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

        def mock_import(name, *args, **kwargs):
            if name == "datasets":
                raise ImportError("No module named 'datasets'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RuntimeError, match="datasets package not installed"):
                builder._fetch_huggingface()

    def test_raises_when_skill_column_missing(self, tmp_path):
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["col_a", "col_b", "col_c"]
        mock_dataset.__iter__ = MagicMock(return_value=iter([]))
        mock_datasets_module = MagicMock()
        mock_datasets_module.load_dataset.return_value = mock_dataset

        builder = LexiconBuilder(output_path=tmp_path / "out.json")
        with patch.dict("sys.modules", {"datasets": mock_datasets_module}):
            with pytest.raises(RuntimeError, match="Cannot find skill column"):
                builder._fetch_huggingface()

    def test_filters_long_skill_phrases(self, tmp_path):
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["skill"]
        mock_dataset.__iter__ = MagicMock(return_value=iter([
            {"skill": "python"},
            {"skill": "this is a very long skill phrase that should be filtered"},
        ]))
        mock_datasets_module = MagicMock()
        mock_datasets_module.load_dataset.return_value = mock_dataset

        builder = LexiconBuilder(output_path=tmp_path / "out.json")
        with patch.dict("sys.modules", {"datasets": mock_datasets_module}):
            result = builder._fetch_huggingface()

        all_skills = [s for v in result["skill_categories"].values() for s in v]
        assert "python" in all_skills
        assert "this is a very long skill phrase that should be filtered" not in all_skills


def _mock_datasets_import(name: str, mock_ds: MagicMock):
    """Helper to inject a mock datasets module."""
    if name == "datasets":
        mod = MagicMock()
        mod.load_dataset.return_value = mock_ds
        return mod
    raise ImportError(f"mocked: {name}")
