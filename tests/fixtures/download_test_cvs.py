"""Builds the benchmark CV corpus used by tests/benchmark_ats.py.

1. Downloads 10 real-world resumes from the HuggingFace dataset
   "ahmedheakl/resume-atlas" — 2 per category group (tech, data, sales,
   engineering, non-tech) — and renders each as a single-column text PDF.
2. Generates 5 synthetic PDFs covering the layouts CVTransformer must
   handle: 2 single-column (native text), 2 two-column (simulated via
   absolute positioning), 1 French "reconversion professionnelle" CV.

All PDFs are written to tests/fixtures/sample_cvs/.

    python tests/fixtures/download_test_cvs.py
"""

import sys
from pathlib import Path
from typing import Optional
from unittest.mock import patch
from xml.sax.saxutils import escape

from fpdf import FPDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.cv_quality_scorer import CVQualityScorer  # noqa: E402
from src.services.cv_transformer import CVTransformer  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "sample_cvs"

# Map our 5 benchmark groups to 2 categories each from the resume-atlas
# dataset (43 categories total).
CATEGORY_GROUPS: dict[str, list[str]] = {
    "tech": ["Information Technology", "DevOps"],
    "data": ["Data Science", "Database"],
    "sales": ["Sales", "Business Analyst"],
    "engineering": ["Civil Engineer", "Mechanical Engineer"],
    "non-tech": ["Human Resources", "Health and Fitness"],
}

# resume-atlas texts are flat, lowercase, punctuation-free word blobs with no
# typographic structure. To turn them into realistic CVs, we locate these
# keyword phrases (most specific first) and treat each as a section break.
_SECTION_KEYWORD_PHRASES: dict[str, list[str]] = {
    "summary": [
        "professional summary",
        "career objective",
        "executive summary",
        "summary",
    ],
    "experience": [
        "professional experience",
        "work experience",
        "work history",
        "employment history",
        "career history",
        "experience",
    ],
    "skills": [
        "technical skills",
        "key skills",
        "core competencies",
        "general skills",
        "skills",
    ],
    "education": ["education"],
    "projects": ["key projects", "personal projects", "projects"],
    "certifications": ["certifications", "certificates", "certification"],
    "languages": ["languages"],
    "interests": ["interests"],
}

_SECTION_TITLES: dict[str, str] = {
    "summary": "SUMMARY",
    "experience": "PROFESSIONAL EXPERIENCE",
    "education": "EDUCATION",
    "skills": "SKILLS",
    "projects": "PROJECTS",
    "certifications": "CERTIFICATIONS",
    "languages": "LANGUAGES",
    "interests": "INTERESTS",
}

# Skip the first few words (name/contact line) when looking for section
# keywords, so they cannot be mistaken for a section header.
_MIN_SECTION_OFFSET = 3

_TITLE_STYLE = ParagraphStyle(
    name="CVTitle", fontName="Helvetica-Bold", fontSize=16, spaceAfter=12
)
_SECTION_STYLE = ParagraphStyle(
    name="SectionHeader",
    fontName="Helvetica-Bold",
    fontSize=13,
    spaceBefore=10,
    spaceAfter=6,
)
_BODY_STYLE = ParagraphStyle(
    name="CVBody", fontName="Helvetica", fontSize=10, leading=14
)


def _slug(text: str) -> str:
    return text.lower().replace(" ", "_")


def _text_to_pdf(text: str, output_path: Path, title: str = "") -> None:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    if title:
        pdf.set_font("Helvetica", style="B", size=14)
        pdf.multi_cell(0, 8, title)
        pdf.ln(2)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, text)
    pdf.output(str(output_path))


def _split_resume_text(text: str) -> list[tuple[Optional[str], str]]:
    """Splits a flat resume-atlas blob into (section_name, content) chunks.

    Returns a leading (None, header_text) chunk for the contact-info
    preamble, followed by one entry per detected section, ordered by where
    the matching keyword appears in the source text.
    """
    words = text.split()
    n = len(words)
    claimed: set[int] = set()
    candidates: list[tuple[int, int, str]] = []

    for section, phrases in _SECTION_KEYWORD_PHRASES.items():
        for phrase in phrases:
            phrase_words = phrase.split()
            plen = len(phrase_words)
            for i in range(_MIN_SECTION_OFFSET, n - plen + 1):
                if any(j in claimed for j in range(i, i + plen)):
                    continue
                if words[i : i + plen] == phrase_words:
                    candidates.append((i, plen, section))
                    claimed.update(range(i, i + plen))
                    break
            else:
                continue
            break

    candidates.sort()

    chunks: list[tuple[Optional[str], str]] = []
    header_end = candidates[0][0] if candidates else n
    header_text = " ".join(words[:header_end])
    if header_text:
        chunks.append((None, header_text))

    for idx, (start, plen, section) in enumerate(candidates):
        content_start = start + plen
        content_end = candidates[idx + 1][0] if idx + 1 < len(candidates) else n
        content = " ".join(words[content_start:content_end]).strip()
        if content:
            chunks.append((section, content))

    return chunks


def _render_structured_pdf(
    title: str, chunks: list[tuple[Optional[str], str]], output_path: Path
) -> None:
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
    )
    story: list = [Paragraph(escape(title), _TITLE_STYLE)]
    for section, content in chunks:
        if section is not None:
            story.append(Paragraph(_SECTION_TITLES[section], _SECTION_STYLE))
        story.append(Paragraph(escape(content), _BODY_STYLE))
        story.append(Spacer(1, 6))
    doc.build(story)


def _validate_pdf(output_path: Path) -> None:
    """Checks word_count > 150 and >= 3 sections detected, logging a warning otherwise."""
    with patch("src.core.config.settings.anthropic_api_key", ""):
        cv = CVTransformer().transform(str(output_path))
    report = CVQualityScorer().score(cv)
    sections = report.ats_readability.sections_found
    if cv.word_count <= 150 or len(sections) < 3:
        print(
            f"  ATTENTION: {output_path.name} -> word_count={cv.word_count}, "
            f"sections={sections}"
        )


def download_huggingface_samples() -> None:
    from datasets import load_dataset

    needed = {cat: 1 for cats in CATEGORY_GROUPS.values() for cat in cats}
    collected: dict[str, list[str]] = {cat: [] for cat in needed}

    print("Téléchargement de ahmedheakl/resume-atlas (streaming)...")
    ds = load_dataset("ahmedheakl/resume-atlas", split="train", streaming=True)
    for row in ds:
        cat = row["Category"]
        if cat in needed and len(collected[cat]) < needed[cat]:
            collected[cat].append(row["Text"])
        if all(len(collected[c]) >= n for c, n in needed.items()):
            break

    for group, categories in CATEGORY_GROUPS.items():
        for category in categories:
            texts = collected.get(category, [])
            if not texts:
                print(f"  ATTENTION: aucun CV trouvé pour la catégorie '{category}'")
                continue
            filename = f"hf_{group}_{_slug(category)}.pdf"
            output_path = OUTPUT_DIR / filename
            chunks = _split_resume_text(texts[0])
            _render_structured_pdf(category, chunks, output_path)
            _validate_pdf(output_path)
            print(f"  OK {filename}")


# ---------------------------------------------------------------------------
# Synthetic PDFs
# ---------------------------------------------------------------------------

_BACKEND_ENGINEER_CV = """Sophie Lambert
Senior Backend Engineer - Lyon, France
sophie.lambert@example.com | +33 6 11 22 33 44 | linkedin.com/in/sophielambert

SUMMARY
Senior Backend Engineer with 7 years of experience designing and scaling
distributed systems in Python and Go. Strong focus on API reliability,
observability and developer experience.

EXPERIENCE
Senior Backend Engineer - PaySphere (Lyon)                       2021 - Present
- Designed and shipped a payments API processing 2M transactions/day,
  reducing p99 latency by 35% through caching and query optimization.
- Migrated a monolithic Django service to a set of FastAPI microservices,
  cutting deployment time from 45 to 8 minutes.
- Mentored 3 junior engineers and led the adoption of code review standards.

Backend Engineer - ShopLine (Lyon)                                2018 - 2021
- Built a recommendation service in Python and Redis, increasing
  cross-sell conversion by 12%.
- Implemented automated testing with pytest, raising coverage from 40% to 85%.
- Set up CI/CD pipelines with GitLab CI and Docker for 6 services.

EDUCATION
Master's Degree in Computer Science - INSA Lyon                  2014 - 2018

SKILLS
Python, Go, FastAPI, Django, PostgreSQL, Redis, Docker, Kubernetes, AWS,
GitLab CI, gRPC, Kafka, pytest, Linux, Terraform

PROJECTS
Order Routing Engine - PaySphere
- Rebuilt the order routing engine using Kafka streams, improving
  throughput by 3x and reducing failed deliveries by 18%.

LANGUAGES
French (native), English (fluent, C1)
"""

_MARKETING_MANAGER_CV = """Julien Moreau
Marketing Manager - Bordeaux, France
julien.moreau@example.com | +33 6 22 33 44 55 | linkedin.com/in/julienmoreau

SUMMARY
Results-driven Marketing Manager with 8 years of experience leading digital
campaigns and brand strategy for retail and consumer goods companies.

EXPERIENCE
Marketing Manager - GreenLeaf Retail (Bordeaux)                   2020 - Present
- Led a rebranding campaign that increased brand awareness by 27% over
  12 months, measured through quarterly surveys.
- Managed a team of 5 marketing specialists and a 400k EUR annual budget.
- Launched an email marketing automation program generating 1.2M EUR
  in incremental revenue.

Digital Marketing Specialist - UrbanWear (Bordeaux)               2016 - 2020
- Grew social media following from 5k to 80k across Instagram and TikTok.
- Coordinated influencer partnerships, improving conversion rate by 9%.
- Analyzed campaign performance using Google Analytics and built monthly
  reporting dashboards.

EDUCATION
Master's Degree in Marketing - Kedge Business School               2014 - 2016

SKILLS
Digital marketing, SEO, SEM, Google Analytics, Google Ads, Meta Ads,
HubSpot, content strategy, brand management, budget management,
email marketing, A/B testing, market research, project management

PROJECTS
GreenLeaf Loyalty Program
- Designed and launched a loyalty program reaching 50k active members
  within the first year, driving a 15% increase in repeat purchases.

LANGUAGES
French (native), English (fluent, B2), Spanish (intermediate, B1)
"""


def _write_two_column_cv(
    header_lines: list[tuple[str, bool]],
    left_lines: list[tuple[str, bool]],
    right_lines: list[tuple[str, bool]],
    output_path: Path,
) -> None:
    """Renders a simulated 2-column CV using absolute positioning.

    `header_lines` spans the full width at the top of the page (kept within
    the top 20% so it does not bias CVTransformer's layout-gap detection).
    `left_lines`/`right_lines` are rendered in two side-by-side columns with
    a gap wide enough (>25pt) for the layout detector to pick up.
    """
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    y = 15.0
    for text, bold in header_lines:
        pdf.set_font("Helvetica", style="B" if bold else "", size=12 if bold else 11)
        pdf.set_xy(15, y)
        pdf.multi_cell(180, 6, text)
        y = pdf.get_y() + 1

    left_x, left_w = 15.0, 60.0
    right_x, right_w = 90.0, 105.0
    left_y = right_y = 65.0  # > 20% of 297mm page height

    for text, bold in left_lines:
        pdf.set_font("Helvetica", style="B" if bold else "", size=10)
        pdf.set_xy(left_x, left_y)
        pdf.multi_cell(left_w, 5, text)
        left_y = pdf.get_y() + (1.5 if not text.strip() else 0.5)

    for text, bold in right_lines:
        pdf.set_font("Helvetica", style="B" if bold else "", size=11)
        pdf.set_xy(right_x, right_y)
        pdf.multi_cell(right_w, 5.5, text)
        right_y = pdf.get_y() + (1.5 if not text.strip() else 0.5)

    pdf.output(str(output_path))


def _build_data_analyst_two_columns(output_path: Path) -> None:
    header = [
        ("Camille Durand", True),
        ("Data Analyst - Nantes, France", False),
        ("camille.durand@example.com | +33 6 33 44 55 66", False),
    ]
    left = [
        ("CONTACT", True),
        ("camille.durand@example.com", False),
        ("+33 6 33 44 55 66", False),
        ("linkedin.com/in/camilledurand", False),
        ("", False),
        ("SKILLS", True),
        (
            "SQL, Python, pandas, numpy, Tableau, Power BI, Excel, "
            "scikit-learn, dbt, Airflow, Snowflake, statistics, "
            "data visualization, A/B testing",
            False,
        ),
        ("", False),
        ("LANGUAGES", True),
        ("French (native), English (fluent, C1)", False),
        ("", False),
        ("EDUCATION", True),
        ("Master's Degree in Statistics", False),
        ("Universite de Nantes - 2017 - 2019", False),
    ]
    right = [
        ("SUMMARY", True),
        (
            "Data Analyst with 5 years of experience turning raw data into "
            "actionable insights for e-commerce and logistics teams. Skilled "
            "in SQL, Python and dashboarding tools.",
            False,
        ),
        ("", False),
        ("EXPERIENCE", True),
        ("Data Analyst - FreightLink (Nantes)                2021 - Present", True),
        (
            "- Built a Tableau dashboard tracking delivery KPIs across 12 "
            "warehouses, reducing late deliveries by 14%.",
            False,
        ),
        (
            "- Automated weekly reporting with Python and Airflow, saving "
            "6 hours of manual work per week.",
            False,
        ),
        (
            "- Designed SQL data models in Snowflake using dbt, improving "
            "query performance by 40%.",
            False,
        ),
        ("", False),
        ("Junior Data Analyst - RetailPlus (Nantes)          2019 - 2021", True),
        (
            "- Analyzed customer churn data with pandas and scikit-learn, "
            "identifying segments responsible for 60% of churn.",
            False,
        ),
        (
            "- Created A/B testing reports that informed pricing decisions "
            "across 3 product lines.",
            False,
        ),
        ("", False),
        ("PROJECTS", True),
        ("Demand Forecasting Model", True),
        (
            "- Built a forecasting model with scikit-learn that reduced "
            "stockouts by 22% during peak season.",
            False,
        ),
    ]
    _write_two_column_cv(header, left, right, output_path)


def _build_product_manager_two_columns(output_path: Path) -> None:
    header = [
        ("Thomas Girard", True),
        ("Product Manager - Toulouse, France", False),
        ("thomas.girard@example.com | +33 6 44 55 66 77", False),
    ]
    left = [
        ("CONTACT", True),
        ("thomas.girard@example.com", False),
        ("+33 6 44 55 66 77", False),
        ("linkedin.com/in/thomasgirard", False),
        ("", False),
        ("SKILLS", True),
        (
            "Product strategy, roadmapping, agile, scrum, Jira, Figma, "
            "user research, A/B testing, SQL, stakeholder management, "
            "OKRs, product analytics, Amplitude",
            False,
        ),
        ("", False),
        ("LANGUAGES", True),
        ("French (native), English (fluent, C1)", False),
        ("", False),
        ("EDUCATION", True),
        ("Master's Degree in Engineering", False),
        ("ENSEEIHT Toulouse - 2013 - 2016", False),
    ]
    right = [
        ("SUMMARY", True),
        (
            "Product Manager with 8 years of experience leading B2B SaaS "
            "products from discovery to launch, working closely with "
            "engineering, design and sales teams.",
            False,
        ),
        ("", False),
        ("EXPERIENCE", True),
        ("Senior Product Manager - CloudOps (Toulouse)       2020 - Present", True),
        (
            "- Launched a billing automation feature adopted by 70% of "
            "customers within 6 months, increasing retention by 9%.",
            False,
        ),
        (
            "- Defined and tracked OKRs across a team of 12 engineers, "
            "improving quarterly delivery predictability by 30%.",
            False,
        ),
        (
            "- Ran 40+ user interviews to prioritize the roadmap, cutting "
            "support tickets related to onboarding by 25%.",
            False,
        ),
        ("", False),
        ("Product Manager - DevTools Inc (Toulouse)          2016 - 2020", True),
        (
            "- Owned the API platform roadmap, growing developer adoption "
            "from 200 to 3,500 active integrations.",
            False,
        ),
        (
            "- Coordinated with design to redesign the onboarding flow, "
            "reducing time-to-first-value by 45%.",
            False,
        ),
        ("", False),
        ("PROJECTS", True),
        ("Self-serve Billing Portal", True),
        (
            "- Drove the launch of a self-serve billing portal that reduced "
            "billing support requests by 35%.",
            False,
        ),
    ]
    _write_two_column_cv(header, left, right, output_path)


_RECONVERSION_FR_CV = """Nathalie Petit
Developpeuse Web Junior (en reconversion) - Croix, France
nathalie.petit@example.com | +33 6 55 66 77 88 | linkedin.com/in/nathaliepetit

PROFIL
Apres 10 ans en tant que comptable, reconversion reussie vers le
developpement web full-stack. Diplomee d'une formation intensive
Developpeur Web et Web Mobile, je recherche un poste de developpeuse
junior pour mettre en pratique mes competences en JavaScript, React et
Node.js, tout en valorisant ma rigueur et mon sens de l'organisation
acquis en comptabilite.

FORMATION
Formation Developpeur Web et Web Mobile - O'clock (a distance)    2024 - 2025
- Projet de fin de formation : application de gestion de budget
  personnel en React, Node.js et PostgreSQL, deployee sur Render.

DUT Gestion des Entreprises et des Administrations - IUT Lille    2012 - 2014

EXPERIENCE
Comptable - Cabinet Dujardin (Croix)                              2014 - 2024
- Geree la comptabilite de plus de 50 clients PME, avec un taux
  d'erreur inferieur a 1%.
- Automatise la saisie des factures avec des macros Excel, reduisant
  le temps de traitement de 30%.
- Forme 4 nouveaux collaborateurs aux outils internes.

PROJETS
Gestionnaire de budget personnel
- Application web en React et Node.js permettant de suivre les depenses
  mensuelles et de generer des rapports automatiques, testee par 15
  utilisateurs beta.

Site vitrine pour artisan local
- Site vitrine responsive en HTML, CSS et JavaScript, ameliorant la
  visibilite en ligne du client (+40% de visites en 3 mois).

COMPETENCES
JavaScript, React, Node.js, Express, PostgreSQL, HTML, CSS, Git,
GitHub, API REST, Excel, comptabilite, gestion administrative

LANGUES
Francais (natif), Anglais (intermediaire, B1)

CENTRES D'INTERETS
Lecture, randonnee, benevolat associatif
"""


def generate_synthetic_pdfs() -> None:
    print("Generation des CVs synthetiques...")

    _text_to_pdf(
        _BACKEND_ENGINEER_CV,
        OUTPUT_DIR / "synth_single_column_backend_engineer.pdf",
    )
    print("  OK synth_single_column_backend_engineer.pdf")

    _text_to_pdf(
        _MARKETING_MANAGER_CV,
        OUTPUT_DIR / "synth_single_column_marketing_manager.pdf",
    )
    print("  OK synth_single_column_marketing_manager.pdf")

    _build_data_analyst_two_columns(OUTPUT_DIR / "synth_two_columns_data_analyst.pdf")
    print("  OK synth_two_columns_data_analyst.pdf")

    _build_product_manager_two_columns(
        OUTPUT_DIR / "synth_two_columns_product_manager.pdf"
    )
    print("  OK synth_two_columns_product_manager.pdf")

    _text_to_pdf(
        _RECONVERSION_FR_CV,
        OUTPUT_DIR / "synth_reconversion_fr.pdf",
    )
    print("  OK synth_reconversion_fr.pdf")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    download_huggingface_samples()
    generate_synthetic_pdfs()
    print(f"\nTermine. CVs ecrits dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
