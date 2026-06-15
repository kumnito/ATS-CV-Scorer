"""Tests for CVTransformer.

Single-column tests run against the committed fixture PDF.
Two-column tests are skipped if CV_kumnito_two_columns.pdf is not present
(the user must drop it in tests/fixtures/ before those tests can run).
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.budget_guard import BudgetGuard
from src.core.schemas import NormalizedCV
from src.services.cv_transformer import CVTransformer, extract_date_range

FIXTURES = Path(__file__).parent / "fixtures"
SINGLE_COLUMN_PDF = FIXTURES / "sample_cv_ml_engineer_junior.pdf"
TWO_COLUMNS_PDF = FIXTURES / "CV_kumnito_two_columns.pdf"

TWO_COL_MISSING = not TWO_COLUMNS_PDF.exists()


@pytest.fixture(autouse=True)
def _isolated_budget_guard(tmp_path, monkeypatch):
    """Isole le BudgetGuard partagé pour éviter de consommer le quota réel."""
    guard = BudgetGuard(limit=300, path=tmp_path / "claude_quota.json")
    monkeypatch.setattr("src.services.cv_transformer.budget_guard", guard)
    yield guard


@pytest.fixture(scope="module")
def transformer():
    return CVTransformer()


def _transform_pdfplumber_only(transformer, pdf_path: str) -> "NormalizedCV":
    """Force le chemin pdfplumber pur (pas d'OCR, pas de Vision LLM).

    Utilisé dans les fixtures de tests sur PDFs réels : ces tests vérifient
    la logique de parsing pdfplumber (layout, sections, entités) et ne doivent
    pas déclencher de vrais appels OCR/API ni être perturbés par les fallbacks.
    """
    with patch("src.services.cv_transformer.settings") as ms, \
         patch.object(transformer, "_extract_text_ocr", return_value=""):
        ms.anthropic_api_key = ""
        return transformer.transform(pdf_path)


@pytest.fixture(scope="module")
def single_col_cv(transformer):
    return _transform_pdfplumber_only(transformer, str(SINGLE_COLUMN_PDF))


# ---------------------------------------------------------------------------
# Single-column fixture tests
# ---------------------------------------------------------------------------


def test_single_col_returns_normalized_cv(single_col_cv):
    assert isinstance(single_col_cv, NormalizedCV)


def test_single_col_layout_detected(single_col_cv):
    assert single_col_cv.layout_detected == "single_column"


def test_single_col_raw_text_nonempty(single_col_cv):
    assert len(single_col_cv.raw_text) > 100


def test_single_col_word_count(single_col_cv):
    assert single_col_cv.word_count > 50


def test_single_col_skills_found(single_col_cv):
    all_skills = single_col_cv.skills.flat()
    assert len(all_skills) > 0, "Au moins une compétence doit être détectée"


def test_single_col_skills_have_python(single_col_cv):
    all_skills = [s.lower() for s in single_col_cv.skills.flat()]
    assert "python" in all_skills


def test_parse_skills_adds_semantic_matches_to_other(transformer):
    with patch(
        "src.services.cv_transformer.match_semantic_skills",
        return_value=["customer relationship management"],
    ):
        skills = transformer._parse_skills("Some CV text about managing client accounts.")

    assert "customer relationship management" in skills.other


def test_parse_skills_does_not_duplicate_substring_matches(transformer):
    with patch(
        "src.services.cv_transformer.match_semantic_skills",
        return_value=["python"],
    ):
        skills = transformer._parse_skills("Experienced Python developer.")

    assert skills.other.count("python") == 0
    assert skills.flat().count("python") == 1


def test_parse_skills_passes_custom_semantic_threshold(transformer):
    with patch(
        "src.services.cv_transformer.match_semantic_skills", return_value=[]
    ) as mock_match:
        transformer._parse_skills("Some text here", semantic_threshold=0.7)

    mock_match.assert_called_once_with("Some text here", threshold=0.7)


def test_single_col_header_has_email(single_col_cv):
    assert single_col_cv.header.email is not None


def test_single_col_experience_or_education_nonempty(single_col_cv):
    assert single_col_cv.experience or single_col_cv.education


def test_single_col_pydantic_valid(single_col_cv):
    dumped = single_col_cv.model_dump()
    rebuilt = NormalizedCV(**dumped)
    assert rebuilt.layout_detected == single_col_cv.layout_detected


# ---------------------------------------------------------------------------
# Date extraction unit tests (no PDF needed)
# ---------------------------------------------------------------------------


def test_date_range_years_only():
    ds, de, dm, ic = extract_date_range("2020 - 2023")
    assert ds == "2020"
    assert de == "2023"
    assert dm == 36
    assert ic is False


def test_date_range_present():
    from datetime import date
    ds, de, dm, ic = extract_date_range("Jan 2021 – Present")
    assert ds is not None
    assert ic is True
    assert de == "present"
    assert dm and dm > 0


def test_date_range_french_months():
    ds, de, dm, ic = extract_date_range("Jan 2019 – Mar 2020")
    assert ds == "2019-01"
    assert de == "2020-03"
    assert dm == 14
    assert ic is False


def test_date_range_en_cours():
    ds, de, dm, ic = extract_date_range("2022 – en cours")
    assert ic is True
    assert ds is not None


def test_date_range_no_date():
    ds, de, dm, ic = extract_date_range("Python, Docker, Kubernetes")
    assert ds is None
    assert de is None
    assert dm is None
    assert ic is False


def test_date_range_single_year():
    ds, de, dm, ic = extract_date_range("2023")
    assert ds == "2023"


def test_date_range_explicit_months():
    ds, de, dm, ic = extract_date_range("Stage de 6 mois")
    assert dm == 6


# ---------------------------------------------------------------------------
# Two-column fixture tests (skipped if PDF not present)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_col_cv(transformer):
    if TWO_COL_MISSING:
        pytest.skip("CV_kumnito_two_columns.pdf not in fixtures/")
    return _transform_pdfplumber_only(transformer, str(TWO_COLUMNS_PDF))


@pytest.mark.skipif(TWO_COL_MISSING, reason="CV_kumnito_two_columns.pdf not in fixtures/")
def test_two_col_layout_detected(two_col_cv):
    assert two_col_cv.layout_detected == "two_columns", (
        f"Layout détecté : {two_col_cv.layout_detected!r} — vérifier que le PDF est bien bi-colonne"
    )


@pytest.mark.skipif(TWO_COL_MISSING, reason="CV_kumnito_two_columns.pdf not in fixtures/")
def test_two_col_valid_normalized_cv(two_col_cv):
    assert isinstance(two_col_cv, NormalizedCV)
    assert two_col_cv.word_count > 50


@pytest.mark.skipif(TWO_COL_MISSING, reason="CV_kumnito_two_columns.pdf not in fixtures/")
def test_two_col_location_from_full_doc(two_col_cv):
    loc = two_col_cv.header.location
    if loc is not None:
        import re
        assert re.match(r"^[A-ZÀ-Ý][a-zA-Zà-öù-ÿ\s\-]+$", loc), (
            f"Localisation inattendue (artefact ?) : {loc!r}"
        )


@pytest.mark.skipif(TWO_COL_MISSING, reason="CV_kumnito_two_columns.pdf not in fixtures/")
def test_two_col_experience_nonempty(two_col_cv):
    assert two_col_cv.experience, "Des expériences professionnelles doivent être détectées"


@pytest.mark.skipif(TWO_COL_MISSING, reason="CV_kumnito_two_columns.pdf not in fixtures/")
def test_two_col_skills_nonempty(two_col_cv):
    assert two_col_cv.skills.flat(), "Des compétences doivent être détectées"


@pytest.mark.skipif(TWO_COL_MISSING, reason="CV_kumnito_two_columns.pdf not in fixtures/")
def test_two_col_header_nonempty(two_col_cv):
    h = two_col_cv.header
    assert any([h.name, h.email, h.phone, h.location]), "L'en-tête doit contenir au moins un champ"


# ---------------------------------------------------------------------------
# CV-KEO-PEN.pdf — regression tests (2026-06-09)
# ---------------------------------------------------------------------------

KEO_PEN_PDF = FIXTURES / "CV-KEO-PEN.pdf"
KEO_PEN_MISSING = not KEO_PEN_PDF.exists()


@pytest.fixture(scope="module")
def keo_pen_cv(transformer):
    return _transform_pdfplumber_only(transformer, str(KEO_PEN_PDF))


@pytest.mark.skipif(KEO_PEN_MISSING, reason="CV-KEO-PEN.pdf not in fixtures/")
def test_keo_pen_layout_two_columns(keo_pen_cv):
    assert keo_pen_cv.layout_detected == "two_columns"


@pytest.mark.skipif(KEO_PEN_MISSING, reason="CV-KEO-PEN.pdf not in fixtures/")
def test_keo_pen_name_includes_pen(keo_pen_cv):
    # Bug 3 : le nom fragmenté "Keo" (col. gauche) + "PEN" (col. droite) doit être fusionné.
    assert keo_pen_cv.header.name is not None
    assert "PEN" in keo_pen_cv.header.name.upper()


@pytest.mark.skipif(KEO_PEN_MISSING, reason="CV-KEO-PEN.pdf not in fixtures/")
def test_keo_pen_experience_contains_sandro(keo_pen_cv):
    companies = [e.company for e in keo_pen_cv.experience if e.company]
    assert any("SANDRO" in (c or "").upper() for c in companies)


@pytest.mark.skipif(KEO_PEN_MISSING, reason="CV-KEO-PEN.pdf not in fixtures/")
def test_keo_pen_passion_section_not_in_experience(keo_pen_cv):
    # Bug 5 : "Musique", "Sport" ne doivent pas apparaître comme entreprises.
    companies = [(e.company or "").lower() for e in keo_pen_cv.experience]
    assert "musique" not in companies
    assert "sport" not in companies


# ---------------------------------------------------------------------------
# Pipeline en cascade — extraction_quality + fallbacks
# ---------------------------------------------------------------------------

_SAMPLE_OCR_TEXT = """John Doe
john@example.com | +33 6 00 00 00 00

EXPÉRIENCES PROFESSIONNELLES

Ingénieur Machine Learning
TechCorp  2021 - 2024
- Développement de modèles NLP avec Python et PyTorch
- Déploiement de pipelines MLOps sur AWS avec Docker et Kubernetes
- Optimisation des performances des modèles de deep learning
- Animation de revues techniques hebdomadaires avec les équipes produit
- Mise en place de suivi des expériences avec MLflow et DVC
- Réduction du temps d'inférence de 40 % via BentoML et cache Redis

Data Scientist
StartupAI  2019 - 2021
- Modèles de classification supervisée avec scikit-learn et XGBoost
- Analyse exploratoire et visualisation avec pandas et matplotlib
- Construction de pipelines de données avec Apache Airflow

FORMATION

Master Informatique — Spécialité Machine Learning
Université Paris Saclay  2017 - 2019
Cours : apprentissage automatique, traitement du langage naturel, vision par ordinateur

COMPÉTENCES

Langages : python java javascript sql bash
ML/DL : pytorch tensorflow scikit-learn xgboost transformers langchain
MLOps : docker kubernetes mlflow airflow dvc github actions ci/cd
Cloud : aws azure google cloud
Bases de données : postgresql mysql redis mongodb

LANGUES

Français natif · Anglais courant (TOEIC 900)
"""


class TestExtractionQuality:
    def test_good_quality(self):
        words = [{"text": f"w{i}"} for i in range(150)]
        assert CVTransformer._extraction_quality(words) == "good"

    def test_partial_quality(self):
        words = [{"text": f"w{i}"} for i in range(80)]
        assert CVTransformer._extraction_quality(words) == "partial"

    def test_failed_quality(self):
        words = [{"text": f"w{i}"} for i in range(10)]
        assert CVTransformer._extraction_quality(words) == "failed"

    def test_boundary_good(self):
        words = [{"text": f"w{i}"} for i in range(150)]
        assert CVTransformer._extraction_quality(words) == "good"

    def test_boundary_partial(self):
        words = [{"text": f"w{i}"} for i in range(50)]
        assert CVTransformer._extraction_quality(words) == "partial"

    def test_empty_is_failed(self):
        assert CVTransformer._extraction_quality([]) == "failed"


class TestTransformFromText:
    """Chemin OCR dégradé — texte brut sans info positionnelle."""

    @pytest.fixture(scope="class")
    def cv(self):
        return CVTransformer()._transform_from_text(
            _SAMPLE_OCR_TEXT, extraction_method="ocr", extraction_confidence=0.85
        )

    def test_returns_normalized_cv(self, cv):
        assert isinstance(cv, NormalizedCV)

    def test_extraction_method_is_ocr(self, cv):
        assert cv.extraction_method == "ocr"

    def test_extraction_confidence_stored(self, cv):
        assert cv.extraction_confidence == 0.85

    def test_layout_single_column(self, cv):
        assert cv.layout_detected == "single_column"

    def test_word_count_reasonable(self, cv):
        assert cv.word_count > 30

    def test_skills_detected(self, cv):
        assert len(cv.skills.flat()) > 0

    def test_python_in_skills(self, cv):
        assert "python" in cv.skills.flat()

    def test_experience_detected(self, cv):
        assert len(cv.experience) > 0

    def test_header_email_extracted(self, cv):
        assert cv.header.email == "john@example.com"


class TestOCRFallback:
    """OCR déclenché quand confiance pdfplumber < 0.85 (seuil ≈ 380 mots).

    Cela couvre à la fois les cas "peu de mots" ET les PDFs graphiques/Canva
    qui passent le seuil de 150 mots mais restent incomplets.
    """

    def _no_api_key(self):
        """Context manager : désactive Vision LLM pour les tests OCR purs."""
        return patch("src.services.cv_transformer.settings",
                     **{"anthropic_api_key": ""})

    def test_ocr_triggered_on_low_word_count(self):
        t = CVTransformer()
        few_words = [[{"text": f"w{i}", "x0": 0, "top": i * 5, "x1": 20,
                       "size": 12.0, "fontname": "Arial"} for i in range(60)]]

        with patch("src.services.cv_transformer.settings") as ms, \
             patch.object(t, "_extract_words", return_value=few_words), \
             patch.object(t, "_extract_text_ocr", return_value=_SAMPLE_OCR_TEXT) as mock_ocr:
            ms.anthropic_api_key = ""
            cv = t.transform("dummy.pdf")

        mock_ocr.assert_called_once_with("dummy.pdf")
        assert cv.extraction_method == "ocr"

    def test_ocr_triggered_on_partial_confidence(self):
        """OCR déclenché même si pdfplumber ≥150 mots mais confiance < 0.85."""
        t = CVTransformer()
        partial_words = [[{"text": f"w{i}", "x0": 0, "top": i * 5, "x1": 20,
                           "size": 12.0, "fontname": "Arial"} for i in range(250)]]

        with patch("src.services.cv_transformer.settings") as ms, \
             patch.object(t, "_extract_words", return_value=partial_words), \
             patch.object(t, "_extract_text_ocr", return_value="") as mock_ocr:
            ms.anthropic_api_key = ""
            t.transform("dummy.pdf")

        mock_ocr.assert_called_once_with("dummy.pdf")

    def test_ocr_wins_when_it_extracts_significantly_more(self):
        """OCR gagne quand il dépasse pdfplumber de >10% ET ≥150 mots."""
        t = CVTransformer()
        # 150 mots pdfplumber → seuil = 150 × 1.10 = 165
        # OCR retourne _SAMPLE_OCR_TEXT (188 mots > 165) → OCR gagne
        partial_words = [[{"text": f"w{i}", "x0": 0, "top": i * 5, "x1": 20,
                           "size": 12.0, "fontname": "Arial"} for i in range(150)]]

        with patch("src.services.cv_transformer.settings") as ms, \
             patch.object(t, "_extract_words", return_value=partial_words), \
             patch.object(t, "_extract_text_ocr", return_value=_SAMPLE_OCR_TEXT):
            ms.anthropic_api_key = ""
            cv = t.transform("dummy.pdf")

        assert cv.extraction_method == "ocr"

    def test_pdfplumber_kept_when_ocr_does_not_improve(self):
        """OCR doit dépasser pdfplumber de >10% pour le supplanter, sinon pdfplumber gardé."""
        t = CVTransformer()
        # 250 mots pdfplumber → OCR doit dépasser 250 × 1.10 = 275 pour gagner.
        # On donne OCR = 260 mots (gain < 10%) → pdfplumber conservé.
        partial_words = [[{"text": f"w{i}", "x0": 0, "top": i * 5, "x1": 20,
                           "size": 12.0, "fontname": "Arial"} for i in range(250)]]
        ocr_260 = " ".join([f"word{i}" for i in range(260)])

        with patch("src.services.cv_transformer.settings") as ms, \
             patch.object(t, "_extract_words", return_value=partial_words), \
             patch.object(t, "_extract_text_ocr", return_value=ocr_260):
            ms.anthropic_api_key = ""
            cv = t.transform("dummy.pdf")

        assert cv.extraction_method == "pdfplumber"
        assert cv.extraction_confidence == round(250 / 450.0, 2)

    def test_ocr_called_on_failed_extraction(self):
        t = CVTransformer()
        empty_words: list[list[dict]] = [[]]

        with patch("src.services.cv_transformer.settings") as ms, \
             patch.object(t, "_extract_words", return_value=empty_words), \
             patch.object(t, "_extract_text_ocr", return_value=_SAMPLE_OCR_TEXT):
            ms.anthropic_api_key = ""
            cv = t.transform("dummy.pdf")

        assert cv.extraction_method == "ocr"

    def test_pdfplumber_direct_when_high_confidence(self):
        # ≥450 mots → confiance = 1.0 ≥ 0.85 → pas d'appel OCR ni Vision LLM
        t = CVTransformer()
        rich_words = [[{"text": f"w{i}", "x0": 0, "top": i * 5, "x1": 20,
                        "size": 12.0, "fontname": "Arial"} for i in range(500)]]

        with patch.object(t, "_extract_words", return_value=rich_words), \
             patch.object(t, "_extract_text_ocr") as mock_ocr:
            cv = t.transform("dummy.pdf")

        mock_ocr.assert_not_called()
        assert cv.extraction_method == "pdfplumber"
        assert cv.extraction_confidence == 1.0

    def test_pdfplumber_confidence_calibrated(self):
        # Confiance = min(1, mots/450) ; OCR tenté mais vide → Vision LLM bloqué (pas de clé)
        t = CVTransformer()
        fake_words = [{"x0": 10, "top": i * 15, "text": "word"} for i in range(300)]
        with patch("src.services.cv_transformer.settings") as ms, \
             patch.object(t, "_extract_words", return_value=[fake_words]), \
             patch.object(t, "_extract_text_ocr", return_value=""):
            ms.anthropic_api_key = ""
            cv = t.transform("dummy.pdf")
        assert cv.extraction_method == "pdfplumber"
        assert cv.extraction_confidence == round(min(1.0, 300 / 450.0), 2)

    def test_ocr_result_has_skills(self):
        t = CVTransformer()
        few_words: list[list[dict]] = [[]]

        with patch("src.services.cv_transformer.settings") as ms, \
             patch.object(t, "_extract_words", return_value=few_words), \
             patch.object(t, "_extract_text_ocr", return_value=_SAMPLE_OCR_TEXT):
            ms.anthropic_api_key = ""
            cv = t.transform("dummy.pdf")

        assert "python" in cv.skills.flat()

    def test_fallback_to_pdfplumber_when_ocr_unavailable(self):
        """Si OCR lève une exception, la cascade utilise pdfplumber en dernier recours."""
        t = CVTransformer()
        few_words = [[{"text": f"w{i}", "x0": 0, "top": i * 5, "x1": 20,
                       "size": 12.0, "fontname": "Arial"} for i in range(60)]]

        with patch.object(t, "_extract_words", return_value=few_words), \
             patch.object(t, "_extract_text_ocr", side_effect=RuntimeError("tesseract absent")), \
             patch.object(t, "_transform_from_vision", side_effect=RuntimeError("no api key")):
            cv = t.transform("dummy.pdf")

        assert isinstance(cv, NormalizedCV)
        assert cv.extraction_method == "pdfplumber"


class TestVisionLLMFallback:
    """Niveau 3 de la cascade : Vision LLM déclenché si confiance < 0.85 ET clé API présente."""

    def _mock_vision_cv_rich(self) -> NormalizedCV:
        """CV Vision LLM avec richesse structurelle élevée (5 expériences, 15 compétences)."""
        from src.core.schemas import CVExperience, CVSkills
        return NormalizedCV(
            header={"name": "Jane Smith", "email": "jane@example.com", "title": "ML Engineer"},
            summary="Ingénieure ML expérimentée.",
            experience=[
                CVExperience(title=f"Poste {i}", company=f"Boîte {i}",
                             date_start="2020", date_end="2023")
                for i in range(5)
            ],
            skills=CVSkills(
                ml=["pytorch", "sklearn", "xgboost", "transformers", "spacy"],
                mlops=["docker", "mlflow", "dvc", "airflow"],
                cloud=["aws", "gcp"],
                languages=["python", "sql"],
                data=["pandas", "spark"],
            ),
            extraction_method="vision_llm",
            extraction_confidence=0.95,
        )

    def _mock_vision_cv_poor(self) -> NormalizedCV:
        """CV Vision LLM avec richesse structurelle faible (pas d'expériences ni compétences)."""
        return NormalizedCV(
            header={"name": "Jane Smith"},
            extraction_method="vision_llm",
            extraction_confidence=0.95,
        )

    def test_vision_triggered_below_threshold_with_api_key(self):
        """Vision LLM appelé quand confiance < 0.85 ET clé disponible ET Vision LLM plus riche."""
        t = CVTransformer()
        empty_words: list[list[dict]] = [[]]
        with patch("src.services.cv_transformer.settings") as mock_settings, \
             patch.object(t, "_extract_words", return_value=empty_words), \
             patch.object(t, "_extract_text_ocr", return_value="just a few words here"), \
             patch.object(t, "_transform_from_vision", return_value=self._mock_vision_cv_rich()) as mock_vis:
            mock_settings.anthropic_api_key = "sk-test"
            cv = t.transform("dummy.pdf")

        mock_vis.assert_called_once_with("dummy.pdf")
        assert cv.extraction_method == "vision_llm"

    def test_vision_not_triggered_without_api_key(self):
        """Sans clé API, Vision LLM n'est jamais appelé, même si confiance < 0.85."""
        t = CVTransformer()
        empty_words: list[list[dict]] = [[]]

        with patch("src.services.cv_transformer.settings") as mock_settings, \
             patch.object(t, "_extract_words", return_value=empty_words), \
             patch.object(t, "_extract_text_ocr", return_value="just a few words"), \
             patch.object(t, "_transform_from_vision") as mock_vis:
            mock_settings.anthropic_api_key = ""
            t.transform("dummy.pdf")

        mock_vis.assert_not_called()

    def test_vision_not_triggered_when_disabled_via_settings(self):
        """VISION_LLM_ENABLED=false bloque le niveau 3 même avec une clé API valide."""
        t = CVTransformer()
        empty_words: list[list[dict]] = [[]]

        with patch("src.services.cv_transformer.settings") as mock_settings, \
             patch.object(t, "_extract_words", return_value=empty_words), \
             patch.object(t, "_extract_text_ocr", return_value="just a few words"), \
             patch.object(t, "_transform_from_vision") as mock_vis:
            mock_settings.anthropic_api_key = "sk-test"
            mock_settings.vision_llm_enabled = False
            t.transform("dummy.pdf")

        mock_vis.assert_not_called()

    def test_vision_not_triggered_when_allow_vision_false(self):
        """allow_vision=False bloque le niveau 3 (quota par session atteint)."""
        t = CVTransformer()
        empty_words: list[list[dict]] = [[]]

        with patch("src.services.cv_transformer.settings") as mock_settings, \
             patch.object(t, "_extract_words", return_value=empty_words), \
             patch.object(t, "_extract_text_ocr", return_value="just a few words"), \
             patch.object(t, "_transform_from_vision") as mock_vis:
            mock_settings.anthropic_api_key = "sk-test"
            mock_settings.vision_llm_enabled = True
            t.transform("dummy.pdf", allow_vision=False)

        mock_vis.assert_not_called()

    def test_vision_not_triggered_when_confidence_sufficient(self):
        """Vision LLM non appelé si pdfplumber/OCR atteignent confiance ≥ 0.85."""
        t = CVTransformer()
        rich_ocr = " ".join([f"word{i}" for i in range(400)])

        with patch("src.services.cv_transformer.settings") as mock_settings, \
             patch.object(t, "_extract_words", return_value=[[]]), \
             patch.object(t, "_extract_text_ocr", return_value=rich_ocr), \
             patch.object(t, "_transform_from_vision") as mock_vis:
            mock_settings.anthropic_api_key = "sk-test"
            t.transform("dummy.pdf")

        mock_vis.assert_not_called()

    def test_vision_wins_when_structurally_richer(self):
        """Vision LLM retenu si son score de richesse structurelle dépasse pdfplumber/OCR."""
        t = CVTransformer()
        partial_ocr = " ".join([f"word{i}" for i in range(200)])

        with patch("src.services.cv_transformer.settings") as mock_settings, \
             patch.object(t, "_extract_words", return_value=[[]]), \
             patch.object(t, "_extract_text_ocr", return_value=partial_ocr), \
             patch.object(t, "_transform_from_vision", return_value=self._mock_vision_cv_rich()) as mock_vis:
            mock_settings.anthropic_api_key = "sk-test"
            cv = t.transform("dummy.pdf")

        mock_vis.assert_called_once()
        assert cv.extraction_method == "vision_llm"

    def test_vision_not_used_when_structurally_poorer(self):
        """Vision LLM ignoré si sa richesse structurelle est inférieure au meilleur résultat."""
        from src.core.schemas import CVExperience, CVSkills
        t = CVTransformer()
        # OCR → pdfplumber/OCR avec 3 expériences et 6 compétences
        partial_ocr = " ".join([f"word{i}" for i in range(200)])

        # Simuler que _transform_from_text retourne un CV riche
        rich_ocr_cv = NormalizedCV(
            experience=[CVExperience(title=f"P{i}", company=f"C{i}") for i in range(3)],
            skills=CVSkills(ml=["pytorch", "sklearn", "xgboost"],
                            mlops=["docker", "mlflow", "dvc"]),
            extraction_method="ocr",
            extraction_confidence=0.44,
        )

        with patch("src.services.cv_transformer.settings") as mock_settings, \
             patch.object(t, "_extract_words", return_value=[[]]), \
             patch.object(t, "_extract_text_ocr", return_value=partial_ocr), \
             patch.object(t, "_transform_from_text", return_value=rich_ocr_cv), \
             patch.object(t, "_transform_from_vision", return_value=self._mock_vision_cv_poor()):
            mock_settings.anthropic_api_key = "sk-test"
            cv = t.transform("dummy.pdf")

        assert cv.extraction_method == "ocr"

    def test_vision_json_parsing(self):
        """_transform_from_vision parse correctement la réponse JSON de Claude."""
        t = CVTransformer()
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

        with patch("src.services.cv_transformer.settings") as mock_settings, \
             patch.dict(sys.modules, {"pdf2image": mock_pdf2image}), \
             patch("anthropic.Anthropic", return_value=mock_client):
            mock_settings.anthropic_api_key = "sk-test-key"
            mock_settings.claude_model = "claude-sonnet-4-6"
            cv = t._transform_from_vision("dummy.pdf")

        assert cv.header.name == "Alice Martin"
        assert cv.header.email == "alice@example.com"
        assert "pytorch" in cv.skills.ml
        assert cv.extraction_method == "vision_llm"
        assert cv.extraction_confidence == 0.95

    def test_vision_skipped_without_api_key(self):
        """_transform_from_vision lève RuntimeError si aucune clé API."""
        t = CVTransformer()
        mock_pdf2image = MagicMock()
        with patch("src.services.cv_transformer.settings") as mock_settings, \
             patch.dict(sys.modules, {"pdf2image": mock_pdf2image}):
            mock_settings.anthropic_api_key = ""
            with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
                t._transform_from_vision("dummy.pdf")
