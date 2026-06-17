"""Generates synthetic two-column CV PDFs for regression testing.

Usage: python tests/fixtures/generate_two_column_fixtures.py

Two PDFs are produced:
- synth_two_columns_simple.pdf  — standard two-column layout
- synth_two_columns_sidebar.pdf — sidebar + main column, name split across
                                   columns (regression: cross-column name join
                                   + PASSIONS section not leaking into experience)
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

W, H = A4  # 595.27 × 841.89 pt
FIXTURES = Path(__file__).parent

LEFT_X = 50.0
RIGHT_X = 310.0  # gap between ~200 (max left x0) and 310 → ~110 pt > 25 pt threshold


def _t(c: canvas.Canvas, x: float, y: float, text: str, size: int = 11, bold: bool = False) -> None:
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    c.drawString(x, y, text)


def generate_simple() -> None:
    """Standard two-column CV: skills+languages in left col, experience+education in right."""
    path = FIXTURES / "synth_two_columns_simple.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)

    # ── Header zone (full width, y > ~673 so pdfplumber top < body_y_start) ──
    _t(c, LEFT_X, 800, "Sophie DURAND", size=16, bold=True)
    _t(c, LEFT_X, 780, "sophie.durand@example.com")
    _t(c, LEFT_X, 764, "75008 Paris")

    # ── Left column (x = 50) ──
    _t(c, LEFT_X, 640, "SKILLS", size=12, bold=True)
    _t(c, LEFT_X, 620, "python sql bash docker")
    _t(c, LEFT_X, 604, "kubernetes aws azure gcp")
    _t(c, LEFT_X, 588, "pytorch tensorflow keras")
    _t(c, LEFT_X, 572, "scikit-learn pandas numpy")
    _t(c, LEFT_X, 556, "mlflow airflow fastapi")
    _t(c, LEFT_X, 540, "github actions ci/cd")
    _t(c, LEFT_X, 524, "postgresql mysql redis")

    _t(c, LEFT_X, 494, "LANGUAGES", size=12, bold=True)
    _t(c, LEFT_X, 474, "French native")
    _t(c, LEFT_X, 458, "English fluent C1")
    _t(c, LEFT_X, 442, "Spanish intermediate B1")

    # ── Right column (x = 310) ──
    _t(c, RIGHT_X, 640, "EXPERIENCE", size=12, bold=True)
    _t(c, RIGHT_X, 620, "ML Engineer")
    _t(c, RIGHT_X, 604, "TechCorp Paris")
    _t(c, RIGHT_X, 588, "2021 - Present")
    _t(c, RIGHT_X, 572, "NLP pipeline Python PyTorch")
    _t(c, RIGHT_X, 556, "CI/CD Docker Kubernetes AWS")
    _t(c, RIGHT_X, 540, "Data Scientist")
    _t(c, RIGHT_X, 524, "DataCorp Lyon")
    _t(c, RIGHT_X, 508, "2019 - 2021")
    _t(c, RIGHT_X, 492, "pandas scikit-learn XGBoost")
    _t(c, RIGHT_X, 476, "SQL dashboards analytics")

    _t(c, RIGHT_X, 446, "EDUCATION", size=12, bold=True)
    _t(c, RIGHT_X, 426, "Master Data Science")
    _t(c, RIGHT_X, 410, "Sorbonne Paris")
    _t(c, RIGHT_X, 394, "2017 - 2019")
    _t(c, RIGHT_X, 374, "Bachelor Mathematics")
    _t(c, RIGHT_X, 358, "Universite Lyon 1")
    _t(c, RIGHT_X, 342, "2015 - 2017")

    c.save()
    print(f"Generated: {path}")


def generate_sidebar() -> None:
    """Sidebar CV with two regression scenarios:

    - Bug 3: name "Keo" (left col) + "PEN" (right col) split at same vertical
      position — must be joined into "Keo PEN" via cross-column name detection.
    - Bug 5: PASSIONS section (Musique, Sport) must NOT leak into experience
      companies — requires the "interests" section header to be recognised.
    """
    path = FIXTURES / "synth_two_columns_sidebar.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)

    # ── Name split across columns (body zone: y=640, top ≈ 189 > body_y_start ≈ 97) ──
    _t(c, LEFT_X, 640, "Keo", size=18, bold=True)
    _t(c, RIGHT_X, 640, "PEN", size=18, bold=True)
    _t(c, LEFT_X, 615, "keo.pen@example.com")

    # ── Left sidebar (x = 50) ──
    _t(c, LEFT_X, 585, "SKILLS", size=12, bold=True)
    _t(c, LEFT_X, 565, "python docker sql")
    _t(c, LEFT_X, 549, "react javascript css")
    _t(c, LEFT_X, 533, "tensorflow pytorch")
    _t(c, LEFT_X, 517, "postgresql redis")
    _t(c, LEFT_X, 501, "github actions linux")

    _t(c, LEFT_X, 471, "LANGUAGES", size=12, bold=True)
    _t(c, LEFT_X, 451, "Khmer native")
    _t(c, LEFT_X, 435, "French fluent")
    _t(c, LEFT_X, 419, "English intermediate")

    # ── Right main column (x = 310) ──
    _t(c, RIGHT_X, 585, "EXPERIENCE", size=12, bold=True)
    _t(c, RIGHT_X, 565, "Software Engineer")
    _t(c, RIGHT_X, 549, "SANDRO Paris")
    _t(c, RIGHT_X, 533, "2019 - 2023")
    _t(c, RIGHT_X, 517, "Web application development")
    _t(c, RIGHT_X, 501, "Automated tests pytest docker")

    _t(c, RIGHT_X, 475, "Junior Developer")
    _t(c, RIGHT_X, 459, "Kenzo Paris")
    _t(c, RIGHT_X, 443, "2016 - 2019")
    _t(c, RIGHT_X, 427, "Legacy system integration")

    _t(c, RIGHT_X, 397, "PASSIONS", size=12, bold=True)
    _t(c, RIGHT_X, 377, "Musique")
    _t(c, RIGHT_X, 361, "Sport")

    c.save()
    print(f"Generated: {path}")


if __name__ == "__main__":
    generate_simple()
    generate_sidebar()
