"""Tests for CVTransformer.

Single-column tests run against the committed fixture PDF.
Two-column tests are skipped if CV_kumnito_two_columns.pdf is not present
(the user must drop it in tests/fixtures/ before those tests can run).
"""

import os
from pathlib import Path

import pytest

from src.core.schemas import NormalizedCV
from src.services.cv_transformer import CVTransformer, extract_date_range

FIXTURES = Path(__file__).parent / "fixtures"
SINGLE_COLUMN_PDF = FIXTURES / "sample_cv_ml_engineer_junior.pdf"
TWO_COLUMNS_PDF = FIXTURES / "CV_kumnito_two_columns.pdf"

TWO_COL_MISSING = not TWO_COLUMNS_PDF.exists()


@pytest.fixture(scope="module")
def transformer():
    return CVTransformer()


@pytest.fixture(scope="module")
def single_col_cv(transformer):
    return transformer.transform(str(SINGLE_COLUMN_PDF))


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


@pytest.mark.skipif(TWO_COL_MISSING, reason="CV_kumnito_two_columns.pdf not in fixtures/")
def test_two_col_layout_detected(transformer):
    cv = transformer.transform(str(TWO_COLUMNS_PDF))
    assert cv.layout_detected == "two_columns", (
        f"Layout détecté : {cv.layout_detected!r} — vérifier que le PDF est bien bi-colonne"
    )


@pytest.mark.skipif(TWO_COL_MISSING, reason="CV_kumnito_two_columns.pdf not in fixtures/")
def test_two_col_valid_normalized_cv(transformer):
    cv = transformer.transform(str(TWO_COLUMNS_PDF))
    assert isinstance(cv, NormalizedCV)
    assert cv.word_count > 50


@pytest.mark.skipif(TWO_COL_MISSING, reason="CV_kumnito_two_columns.pdf not in fixtures/")
def test_two_col_location_from_full_doc(transformer):
    cv = transformer.transform(str(TWO_COLUMNS_PDF))
    loc = cv.header.location
    # La localisation est recherchée dans tout le document (pas seulement l'en-tête).
    # Si une adresse postale FR (code postal + ville) est présente n'importe où dans
    # le PDF, elle doit être détectée. None est correct uniquement si le CV n'en contient pas.
    if loc is not None:
        import re
        assert re.match(r"^[A-ZÀ-Ý][a-zA-Zà-öù-ÿ\s\-]+$", loc), (
            f"Localisation inattendue (artefact ?) : {loc!r}"
        )


@pytest.mark.skipif(TWO_COL_MISSING, reason="CV_kumnito_two_columns.pdf not in fixtures/")
def test_two_col_experience_nonempty(transformer):
    cv = transformer.transform(str(TWO_COLUMNS_PDF))
    assert cv.experience, "Des expériences professionnelles doivent être détectées"


@pytest.mark.skipif(TWO_COL_MISSING, reason="CV_kumnito_two_columns.pdf not in fixtures/")
def test_two_col_skills_nonempty(transformer):
    cv = transformer.transform(str(TWO_COLUMNS_PDF))
    assert cv.skills.flat(), "Des compétences doivent être détectées"


@pytest.mark.skipif(TWO_COL_MISSING, reason="CV_kumnito_two_columns.pdf not in fixtures/")
def test_two_col_header_nonempty(transformer):
    cv = transformer.transform(str(TWO_COLUMNS_PDF))
    h = cv.header
    assert any([h.name, h.email, h.phone, h.location]), "L'en-tête doit contenir au moins un champ"


# ---------------------------------------------------------------------------
# CV-KEO-PEN.pdf — regression tests (2026-06-09)
# ---------------------------------------------------------------------------

KEO_PEN_PDF = FIXTURES / "CV-KEO-PEN.pdf"
KEO_PEN_MISSING = not KEO_PEN_PDF.exists()


@pytest.fixture(scope="module")
def keo_pen_cv(transformer):
    return transformer.transform(str(KEO_PEN_PDF))


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
