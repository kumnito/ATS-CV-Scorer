"""Hugging Face Spaces entry point — Gradio interface for ATS CV Scorer."""

import logging

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

import gradio as gr
import gradio_client.utils as _gradio_client_utils

from src.core.budget_guard import budget_guard
from src.core.config import settings
from src.core.lexicons import init_lexicons
from src.core.schemas import CVQualityReport, NormalizedCV
from src.services.claude_feedback import ClaudeBudgetExceeded, ClaudeFeedback, ClaudeServiceError
from src.services.cv_quality_scorer import CVQualityScorer
from src.services.cv_transformer import CVTransformer
from src.services.job_matcher import FRANCE_REGIONS, find_matching_jobs
from src.services.job_search import JobSearchService
from src.services.nlp_pipeline import NLPPipeline
from src.services.semantic_scorer import SemanticScorer


def _patch_gradio_client_bool_schema() -> None:
    # gradio_client 1.3.0 (bundled with gradio==4.44.0) assumes JSON-schema nodes
    # are always dicts. Pydantic 2.11+ emits `additionalProperties: true` (a bare
    # bool) for `dict[str, Any]`-shaped fields, which crashes the route Gradio
    # itself queries on launch to verify localhost is reachable.
    original = _gradio_client_utils._json_schema_to_python_type

    def patched(schema, defs):
        if isinstance(schema, bool):
            return "Any"
        return original(schema, defs)

    _gradio_client_utils._json_schema_to_python_type = patched


_patch_gradio_client_bool_schema()

init_lexicons()

_cv_transformer = CVTransformer()
_nlp_pipeline = NLPPipeline()
_semantic_scorer = SemanticScorer()
_cv_quality_scorer = CVQualityScorer()
_job_search_service = JobSearchService(
    app_id=settings.adzuna_id,
    app_key=settings.adzuna_api_key,
    country=settings.adzuna_country,
)

logger.info("budget_guard | quota Claude global restant : %d/%d",
            budget_guard.get_remaining(), budget_guard.limit)

MAX_JOB_RESULTS = 10

# Limites par session (gr.State) pour protéger le budget ANTHROPIC_API_KEY de la démo.
MAX_VISION_CALLS_PER_SESSION = 3
MAX_FEEDBACK_CALLS_PER_SESSION = 5

_DEMO_LIMIT_MD = (
    "⚠️ **Limite de la démo atteinte.** Pour un usage illimité, clonez le projet "
    "et ajoutez votre propre clé API.\n\n"
    "[📦 Voir le projet sur GitHub](https://github.com/kumnito/ATS-CV-Scorer)"
)

# ── Thème & CSS — Tech Dashboard ─────────────────────────────────────────────

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
    # gr.themes.Base (4.44.0) only wraps font_mono in a list when it's a str —
    # a bare GoogleFont object hits `for fontfam in font_mono` on a non-iterable.
    font_mono=[gr.themes.GoogleFont("JetBrains Mono")],
)

# Palette propre à l'app — basée sur les variables CSS natives du thème Gradio
# (--body-text-color, --background-fill-*, --border-color-*), qui basculent
# automatiquement entre light et dark mode. Aucune couleur de texte/fond n'est
# hardcodée en hex, sauf les accents indigo (lisibles sur les deux fonds) et
# les paires bg/texte des badges, qui ont un correctif .dark dédié.
CUSTOM_CSS = """
.gradio-container {
    --app-border: var(--border-color-primary);
    --app-card-bg: var(--background-fill-primary);
    --app-text-secondary: var(--body-text-color-subdued);
    --app-badge-bg: #eef2ff;
    --app-badge-text: #4338ca;
    background: var(--background-fill-secondary);
}

.dark .gradio-container {
    --app-badge-bg: #312e81;
    --app-badge-text: #c7d2fe;
}

/* Cards résultats */
.result-card, .metric-card {
    border: 0.5px solid var(--app-border);
    border-radius: 12px;
    padding: 1.25rem;
    background: var(--app-card-bg);
    color: var(--body-text-color);
}

/* Grille de métriques (4 cards horizontales) */
.metrics-grid {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.metric-card {
    flex: 1;
    min-width: 150px;
}
.metric-label {
    font-size: 12px;
    color: var(--app-text-secondary);
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.metric-value {
    font-size: 28px;
    font-weight: 600;
    line-height: 1.2;
    color: var(--body-text-color);
}
.metric-sub {
    font-size: 12px;
    color: var(--app-text-secondary);
    margin-top: 2px;
}

/* Onglets — accent indigo sur l'onglet actif */
.tab-nav button.selected, div[role="tablist"] button.selected {
    border-bottom: 2px solid #6366f1 !important;
    color: #6366f1 !important;
    font-weight: 600;
}

/* Bouton "Analyser" — style outline discret */
.analyze-outline {
    background: transparent !important;
    border: 1px solid var(--app-border) !important;
    color: var(--app-text-secondary) !important;
}
.analyze-outline:hover {
    border-color: #6366f1 !important;
    color: #6366f1 !important;
}

/* Zone d'upload du CV */
.cv-upload .wrap {
    border: 1.5px dashed var(--app-border) !important;
    border-radius: 12px;
    transition: border-color 0.15s ease, background-color 0.15s ease;
}
.cv-upload .wrap:hover {
    border-color: #6366f1 !important;
    background-color: var(--app-badge-bg);
}

/* Badges génériques */
.skill-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 8px;
    background: var(--app-badge-bg);
    color: var(--app-badge-text);
    margin: 2px;
}
.extraction-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 500;
    padding: 4px 10px;
    border-radius: 8px;
    margin-bottom: 0.75rem;
}

/* Timeline carrière */
.timeline-entry {
    display: flex;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 0.5px solid var(--app-border);
    color: var(--body-text-color);
}
.timeline-entry:last-child {
    border-bottom: none;
}
.timeline-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-top: 4px;
    flex-shrink: 0;
}
.timeline-title {
    font-weight: 500;
    color: var(--body-text-color);
}
.timeline-meta {
    font-size: 12px;
    color: var(--app-text-secondary);
}
.timeline-badge {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 8px;
    background: var(--app-badge-bg);
    color: var(--app-badge-text);
    align-self: center;
    white-space: nowrap;
}
"""


def _score_color(score: float) -> str:
    return "🟢" if score >= 70 else ("🟡" if score >= 45 else "🔴")


def _score_color_hex(score: float) -> str:
    if score >= 70:
        return "#22c55e"
    if score >= 50:
        return "#f59e0b"
    return "#ef4444"


def _progress_bar_html(label: str, value: int, color: str = "#6366f1", max_val: int = 100) -> str:
    pct = max(0, min(100, round(value / max_val * 100)))
    return (
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">'
        f'<span style="width:80px;font-size:12px;color:var(--app-text-secondary)">{label}</span>'
        '<div style="flex:1;height:6px;background:var(--app-border);border-radius:3px">'
        f'<div style="width:{pct}%;height:100%;background:{color};border-radius:3px"></div>'
        "</div>"
        f'<span style="font-size:12px;font-weight:500;width:32px;text-align:right">{value}</span>'
        "</div>"
    )


def _metric_card(label: str, value: str, sub: str) -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-sub">{sub}</div>'
        "</div>"
    )


def _extraction_badge_html(report: CVQualityReport) -> str:
    m, c = report.extraction_method, report.extraction_confidence
    if m == "pdfplumber":
        bg, color, label = "#dcfce7", "#16a34a", "✅ Extraction native"
    elif m == "ocr":
        bg, color, label = "#fef3c7", "#b45309", "⚠️ Extraction OCR"
    else:
        bg, color, label = "#e0e7ff", "#4338ca", "🤖 Extraction Vision IA"
    return (
        f'<span class="extraction-badge" style="background:{bg};color:{color}">'
        f"{label} · confiance {c:.0%}</span>"
    )


def _skill_badge(skill: str) -> str:
    return f'<span class="skill-badge">{skill}</span>'


_SKILL_CATEGORY_LABELS = {
    "ml": "ML",
    "mlops": "MLOps",
    "cloud": "Cloud",
    "data": "Data",
    "languages": "Langages",
    "other": "Autres",
    "commerce": "Commerce",
}


def _format_skill_badges(cv: NormalizedCV) -> str:
    skills_dict = cv.skills.model_dump()
    parts = []
    for key, label in _SKILL_CATEGORY_LABELS.items():
        values = skills_dict.get(key, [])
        if not values:
            continue
        badges = "".join(_skill_badge(s) for s in values)
        parts.append(
            f'<div style="margin-bottom:6px"><strong style="font-size:12px">{label}</strong><br>{badges}</div>'
        )
    if not parts:
        return "<em>Aucune compétence détectée</em>"
    return "".join(parts)


def _format_quality_report(cv: NormalizedCV, report: CVQualityReport) -> str:
    ats = report.ats_readability
    ps = report.profile_strength

    sections_str = " · ".join(ats.sections_found) if ats.sections_found else "—"
    readability_label = "✅ Oui" if ats.is_machine_readable else f"⚠️ Non ({cv.word_count} mots < 150)"

    table_rows = [
        f"| Format | {ats.layout_label} |",
        f"| Sections trouvées | ✅ {sections_str} |",
    ]
    if ats.sections_missing:
        table_rows.append(f"| Sections manquantes | ❌ {' · '.join(ats.sections_missing)} |")
    table_rows.append(f"| Lisibilité machine | {readability_label} |")

    lines = [
        "## 📋 Lisibilité ATS",
        "",
        "| Critère | Statut |",
        "|---|---|",
        *table_rows,
        "",
    ]

    # Solidité du profil
    level_emoji = {"Solide": "🟢", "Correct": "🟡", "À renforcer": "🔴"}.get(ps.level, "")
    lines += [
        f"## 💼 Solidité du profil — {level_emoji} {ps.level}",
        _progress_bar_html("Profil", ps.score, color=_score_color_hex(ps.score)),
        "",
    ]
    if ps.strengths:
        lines.append("✅ **Points forts**")
        lines.extend(f"- {s}" for s in ps.strengths)
        lines.append("")
    if ps.improvements:
        lines.append("💡 **À améliorer**")
        lines.extend(f"- {imp}" for imp in ps.improvements)
        lines.append("")

    # Recommandations prioritaires
    if report.career_gaps:
        lines += ["### ⚠️ Trous de carrière détectés"]
        lines.extend(f"- {gap}" for gap in report.career_gaps)
        lines.append("")

    if report.recommendations:
        _impact_emoji = {"Fort": "🔴", "Moyen": "🟠", "Faible": "🟡"}
        recs_sorted = sorted(report.recommendations, key=lambda r: r.priority)
        lines.append("## 🎯 Actions prioritaires")
        for i, rec in enumerate(recs_sorted, 1):
            emoji = _impact_emoji.get(rec.impact, "⚪")
            lines.append(f"{i}. {emoji} **{rec.action}** — {rec.why}")
        lines.append("")

    return "\n".join(lines)


def _format_metrics_html(cv: NormalizedCV, report: CVQualityReport) -> str:
    ats = report.ats_readability
    ps = report.profile_strength

    layout_icon = "✅" if ats.layout == "single_column" else "⚠️"
    layout_sub = "1 colonne" if ats.layout == "single_column" else "2 colonnes"

    skills_count = len(cv.skills.flat())
    density = round(skills_count / max(cv.word_count, 1) * 100)

    exp_value = f"{report.total_experience_years:.0f} an(s)" if report.total_experience_years else "—"
    exp_sub = f"depuis {report.career_start_year}" if report.career_start_year else "—"

    cards = [
        _metric_card("Lisibilité ATS", layout_icon, layout_sub),
        _metric_card("Profil", f"{ps.score}/100", ps.level),
        _metric_card("Mots-clés", f"{density}%", f"{skills_count} skills"),
        _metric_card("Expérience", exp_value, exp_sub),
    ]
    return _extraction_badge_html(report) + f'<div class="metrics-grid">{"".join(cards)}</div>'


def _timeline_entry_html(dot_color: str, title: str, meta: str, badge: str = "") -> str:
    badge_html = f'<span class="timeline-badge">{badge}</span>' if badge else ""
    return (
        '<div class="timeline-entry">'
        f'<div class="timeline-dot" style="background:{dot_color}"></div>'
        '<div style="flex:1">'
        f'<div class="timeline-title">{title}</div>'
        f'<div class="timeline-meta">{meta}</div>'
        "</div>"
        f"{badge_html}"
        "</div>"
    )


def _format_timeline(cv: NormalizedCV, report: CVQualityReport) -> str:
    if not cv.experience and not cv.education:
        return ""

    header_lines = ["## 📅 Timeline carrière"]
    if report.total_experience_years > 0:
        header_lines.append(f"**Expérience totale : {report.total_experience_years:.1f} an(s)**")
    if report.career_start_year:
        header_lines.append(f"Premier poste détecté : {report.career_start_year}")

    events: list[tuple[int, int, str]] = []
    for e in cv.experience:
        label_parts = [p for p in [e.title, e.company] if p]
        title = " @ ".join(label_parts) if label_parts else "Poste non détecté"
        badge = f"{e.years:.0f} an(s)" if e.years else ""
        period = e.period or (
            f"{e.date_start or '?'} – {'présent' if e.is_current else (e.date_end or '?')}"
        )
        s_year = int(e.date_start.split("-")[0]) if e.date_start else 0
        s_month = int(e.date_start.split("-")[1]) if e.date_start and "-" in e.date_start else 1
        events.append((s_year, s_month, _timeline_entry_html("#22c55e", title, period, badge)))

    for e in cv.education:
        title = " — ".join(p for p in [e.degree, e.school] if p) or "Formation non détectée"
        badge = f"{round((e.duration_months or 0) / 12):.0f} an(s)" if e.duration_months else ""
        period = e.year or (
            f"{e.date_start or '?'} – {'présent' if e.is_current else (e.date_end or '?')}"
        )
        s_year = int(e.date_start.split("-")[0]) if e.date_start else 0
        s_month = int(e.date_start.split("-")[1]) if e.date_start and "-" in e.date_start else 1
        events.append((s_year, s_month, _timeline_entry_html("#6366f1", title, period, badge)))

    events.sort(key=lambda x: (x[0], x[1]), reverse=True)
    entries_html = [html for _, _, html in events]

    for gap in report.career_gaps:
        entries_html.append(
            _timeline_entry_html("#f59e0b", "⚠️ Période sans activité détectée", gap)
        )

    return "\n".join(header_lines) + "\n\n" + "".join(entries_html)


def _format_profile_summary(parsed_cv, normalized_cv: NormalizedCV) -> str:
    """Profil détecté affiché dans l'onglet Analyse."""
    name = normalized_cv.header.name or ""
    title = parsed_cv.job_title or normalized_cv.header.title or ""
    location = parsed_cv.location or normalized_cv.header.location or ""

    header = " · ".join(p for p in [name, title] if p) or "Profil détecté"
    meta = []
    if location:
        meta.append(f"📍 {location}")

    lines = [f"### 👤 {header}"]
    if meta:
        lines.append("  ·  ".join(meta))
    return "\n".join(lines)


def _format_cv_context_strip(parsed_cv, report: CVQualityReport) -> str:
    """Bandeau de contexte compact affiché en haut de l'onglet Recherche."""
    parts = []
    if parsed_cv.job_title:
        parts.append(f"**{parsed_cv.job_title}**")
    if parsed_cv.location:
        parts.append(f"📍 {parsed_cv.location}")
    score = report.profile_strength.score
    color = _score_color(score)
    parts.append(f"{color} Score CV : **{score}/100**")
    return "✅ Profil chargé — " + " · ".join(parts)


# ── Handler 1 : traitement du CV ─────────────────────────────────────────────

def on_cv_upload(cv_file, vision_calls):
    _reset = (
        "", "", "",          # quality_md, timeline_md, profile_md
        None, None, None,    # parsed_cv_state, normalized_cv_state, quality_report_state
        "⚠️ Uploadez votre CV en onglet 1 pour commencer.",  # cv_context_md
        gr.update(interactive=False),                         # search_btn
        gr.update(value=""),                                  # job_title_input
        vision_calls,                                         # vision_calls_state
        None,                                                  # cv_embedding_state
        "", "",                                               # metrics_html, skills_html
    )

    if cv_file is None:
        return _reset

    allow_vision = vision_calls < MAX_VISION_CALLS_PER_SESSION
    normalized_cv = _cv_transformer.transform(cv_file.name, allow_vision=allow_vision)
    if normalized_cv.extraction_method == "vision_llm":
        vision_calls += 1

    if not normalized_cv.raw_text:
        return (
            "❌ Impossible d'extraire le texte du PDF.",
            "", "",
            None, None, None,
            "❌ Échec de l'extraction PDF — vérifiez que le fichier n'est pas protégé.",
            gr.update(interactive=False),
            gr.update(value=""),
            vision_calls,
            None,
            "", "",
        )

    parsed_cv = _nlp_pipeline.parse_normalized(normalized_cv)
    quality_report = _cv_quality_scorer.score(normalized_cv)
    cv_embedding = _semantic_scorer.encode_cv(normalized_cv.raw_text)

    quality_md = _format_quality_report(normalized_cv, quality_report)
    if not allow_vision and normalized_cv.extraction_confidence < 0.85:
        quality_md += f"\n\n---\n\n{_DEMO_LIMIT_MD}"

    return (
        quality_md,
        _format_timeline(normalized_cv, quality_report),
        _format_profile_summary(parsed_cv, normalized_cv),
        parsed_cv,
        normalized_cv,
        quality_report,
        _format_cv_context_strip(parsed_cv, quality_report),
        gr.update(interactive=True),
        gr.update(value=parsed_cv.job_title or ""),
        vision_calls,
        cv_embedding,
        _format_metrics_html(normalized_cv, quality_report),
        _format_skill_badges(normalized_cv),
    )


def on_cv_clear():
    return (
        "", "", "",
        None, None, None,
        "⚠️ Uploadez votre CV en onglet 1 pour commencer.",
        gr.update(interactive=False),
        gr.update(value=""),
        0,
        None,
        "", "",
    )


# ── Handler 2 : recherche et scoring des offres ───────────────────────────────

def _empty_job_slots() -> list:
    result = []
    for _ in range(MAX_JOB_RESULTS):
        result += _slot_updates()
    return result


def _format_job_card(rank: int, match) -> str:
    job, result = match.job, match.scoring_result
    score = result.overall_score
    color = _score_color_hex(score)

    meta = [
        part
        for part in (
            f"<strong>{job.company}</strong>" if job.company else None,
            f"📍 {job.location}" if job.location else None,
            job.contract_type,
        )
        if part
    ]
    if job.salary_min or job.salary_max:
        lo = f"{int(job.salary_min):,}".replace(",", " ") if job.salary_min else "?"
        hi = f"{int(job.salary_max):,}".replace(",", " ") if job.salary_max else "?"
        meta.append(f"💰 {lo} – {hi} €/an")

    return (
        '<div class="result-card" style="display:flex;gap:1rem;align-items:flex-start;margin-bottom:0.75rem">'
        f'<div style="font-size:22px;font-weight:500;color:{color};min-width:48px">{score:.0f}</div>'
        '<div style="flex:1">'
        f'<div style="font-weight:600;margin-bottom:4px">#{rank} — {job.title}</div>'
        f'<div style="font-size:13px;color:var(--app-text-secondary);margin-bottom:6px">{" · ".join(meta)}</div>'
        f'<a href="{job.url}" target="_blank">Voir l\'offre complète ↗</a>'
        "</div>"
        "</div>"
    )


def _slot_updates(rank: int | None = None, match=None):
    if match is None:
        return [
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(visible=False),
            gr.update(value="", visible=False),
        ]
    return [
        gr.update(visible=True),
        gr.update(value=_format_job_card(rank, match)),
        gr.update(visible=True),
        gr.update(value="", visible=False),
    ]


def on_search(parsed_cv, region, title_override, cv_embedding):
    if parsed_cv is None:
        status = "⚠️ Uploadez d'abord votre CV en onglet 1."
        return [status, [], ""] + _empty_job_slots()

    # Apply user's title override before searching
    if title_override and title_override.strip():
        parsed_cv = parsed_cv.model_copy(update={"job_title": title_override.strip()})

    result = find_matching_jobs(
        parsed_cv,
        _job_search_service,
        _semantic_scorer,
        cv_embedding=cv_embedding,
        max_results=MAX_JOB_RESULTS,
        region=region or None,
    )
    matches = result.matches

    # Build the query display line
    query_display = ""
    if result.queries_used:
        main_q = result.queries_used[0]
        query_display = f'🔍 Requête Adzuna : **"{main_q}"**'
        if len(result.queries_used) > 1:
            extras = ", ".join(f'"{q}"' for q in result.queries_used[1:])
            query_display += f" · variantes : {extras}"
        if result.location_used:
            loc_radius = " (30 km)" if parsed_cv.location and not region else ""
            query_display += f" · 📍 {result.location_used}{loc_radius}"

    # Build the profile summary line for the status header
    if region:
        profile_line = f"**Profil :** {parsed_cv.job_title or '_non déterminé_'} · 🗺️ {region}"
    else:
        profile_line = f"**Profil :** {parsed_cv.job_title or '_non déterminé_'}"
        if parsed_cv.location:
            loc_display = (
                f"{parsed_cv.postal_code} {parsed_cv.location}".strip()
                if parsed_cv.postal_code
                else parsed_cv.location
            )
            profile_line += f" · 📍 {loc_display} (rayon 30 km)"

    few_results_note = (
        "\n\n⚠️ Peu d'offres correspondantes trouvées — "
        "essayez d'élargir le rayon géographique ou de préciser votre titre de poste."
        if result.few_results
        else ""
    )

    if matches:
        status = (
            f"## ✅ {len(matches)} offre(s) trouvée(s), classées par score de compatibilité décroissant\n"
            f"{profile_line}{few_results_note}"
        )
    else:
        status = (
            f"## 😕 Aucune offre ne correspond à votre profil pour le moment\n{profile_line}\n\n"
            "Essayez avec un CV mentionnant plus explicitement votre **intitulé de poste**, "
            "vos **compétences clés** et votre **ville**, ou réessayez un peu plus tard."
        )

    outputs = [status, matches, query_display]
    for i in range(MAX_JOB_RESULTS):
        outputs += _slot_updates(rank=i + 1, match=matches[i]) if i < len(matches) else _slot_updates()
    return outputs


# ── Handler 3 : feedback Claude par offre (à la demande) ─────────────────────

_ANALYZE_LOADING_MD = (
    "⏳ **Analyse en cours…** Claude examine cette offre au regard de votre "
    "profil — merci de patienter quelques secondes."
)


def _make_analyze_handler(index: int):
    def _analyze_one(parsed_cv, matches, feedback_calls):
        if not matches or index >= len(matches):
            yield gr.update(value="", visible=False), feedback_calls
            return

        if feedback_calls >= MAX_FEEDBACK_CALLS_PER_SESSION:
            yield gr.update(value=_DEMO_LIMIT_MD, visible=True), feedback_calls
            return

        yield gr.update(value=_ANALYZE_LOADING_MD, visible=True), feedback_calls

        match = matches[index]
        try:
            cf = ClaudeFeedback()
            feedback = cf.generate_feedback(
                parsed_cv, match.job.description, match.scoring_result
            )
            feedback_calls += 1
        except ValueError:
            feedback = (
                "⚠️ Configurez `ANTHROPIC_API_KEY` dans le fichier `.env` pour générer "
                "un feedback personnalisé pour cette offre."
            )
        except ClaudeBudgetExceeded:
            feedback = "⚠️ Service IA temporairement indisponible — réessayez plus tard."
        except ClaudeServiceError as exc:
            feedback = f"⚠️ {exc}"
        except Exception as exc:
            logger.warning("analyze_one | erreur inattendue lors du feedback Claude : %s", exc)
            feedback = "⚠️ Feedback Claude indisponible pour le moment — réessayez plus tard."
        yield gr.update(value=feedback, visible=True), feedback_calls

    return _analyze_one


# ── Interface Gradio ──────────────────────────────────────────────────────────

with gr.Blocks(
    title="ATS CV Scorer",
    theme=THEME,
    css=CUSTOM_CSS,
) as demo:
    gr.Markdown("# 📋 ATS CV Scorer")

    parsed_cv_state = gr.State(None)
    normalized_cv_state = gr.State(None)
    quality_report_state = gr.State(None)
    matches_state = gr.State([])
    vision_calls_state = gr.State(0)
    feedback_calls_state = gr.State(0)
    cv_embedding_state = gr.State(None)

    with gr.Tabs():

        # ── Onglet 1 : Analyse du CV ──────────────────────────────────────────
        with gr.Tab("📋 Analyse du CV"):
            gr.Markdown(
                "Uploadez votre CV pour obtenir immédiatement une **analyse qualité ATS** — "
                "lisibilité parseur, solidité du profil, timeline carrière et actions prioritaires. "
                "Aucun appel réseau externe n'est déclenché à cette étape."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    cv_input = gr.File(
                        label="CV (PDF)", file_types=[".pdf"], elem_classes=["cv-upload"]
                    )
                    profile_md = gr.Markdown()
                    skills_html = gr.HTML()
                with gr.Column(scale=2):
                    metrics_html = gr.HTML()
                    quality_md = gr.Markdown()
                    timeline_md = gr.Markdown()

        # ── Onglet 2 : Recherche d'offres ─────────────────────────────────────
        with gr.Tab("🔍 Recherche d'offres"):
            gr.Markdown(
                "Recherchez des offres correspondant à votre profil via **Adzuna**, "
                "scorées automatiquement par similarité sémantique. Demandez ensuite un "
                "**feedback Claude personnalisé** pour chaque offre à la demande — "
                "aucun appel IA n'est déclenché tant que vous ne cliquez pas sur « Analyser »."
            )
            cv_context_md = gr.Markdown("⚠️ Uploadez votre CV en onglet 1 pour commencer.")
            with gr.Row():
                job_title_input = gr.Textbox(
                    label="Intitulé de poste (modifiable avant la recherche)",
                    placeholder="ex. Vendeur, Data Scientist, Développeur Python…",
                    scale=2,
                )
                region_dropdown = gr.Dropdown(
                    choices=FRANCE_REGIONS,
                    label="Région (prioritaire si sélectionnée — remplace la localisation du CV)",
                    value=None,
                    scale=2,
                )
                search_btn = gr.Button(
                    "🔍 Rechercher des offres correspondantes",
                    variant="primary",
                    interactive=False,
                    scale=1,
                )
            search_query_md = gr.Markdown()
            search_status_md = gr.Markdown()

            job_slots = []
            for _ in range(MAX_JOB_RESULTS):
                with gr.Group(visible=False) as group:
                    job_md = gr.Markdown()
                    slot_analyze_btn = gr.Button(
                        "🔎 Analyser cette offre",
                        size="sm",
                        visible=False,
                        elem_classes=["analyze-outline"],
                    )
                    slot_feedback_md = gr.Markdown(visible=False)
                job_slots.append((group, job_md, slot_analyze_btn, slot_feedback_md))

    # ── Câblage des événements ────────────────────────────────────────────────

    _upload_outputs = [
        quality_md,
        timeline_md,
        profile_md,
        parsed_cv_state,
        normalized_cv_state,
        quality_report_state,
        cv_context_md,
        search_btn,
        job_title_input,
        vision_calls_state,
        cv_embedding_state,
        metrics_html,
        skills_html,
    ]

    cv_input.upload(
        fn=on_cv_upload, inputs=[cv_input, vision_calls_state], outputs=_upload_outputs
    )
    cv_input.clear(fn=on_cv_clear, outputs=_upload_outputs)

    _search_outputs = [search_status_md, matches_state, search_query_md]
    for group, job_md, slot_analyze_btn, slot_feedback_md in job_slots:
        _search_outputs += [group, job_md, slot_analyze_btn, slot_feedback_md]

    search_btn.click(
        fn=on_search,
        inputs=[parsed_cv_state, region_dropdown, job_title_input, cv_embedding_state],
        outputs=_search_outputs,
    )

    for idx, (_, _, slot_analyze_btn, slot_feedback_md) in enumerate(job_slots):
        slot_analyze_btn.click(
            fn=_make_analyze_handler(idx),
            inputs=[parsed_cv_state, matches_state, feedback_calls_state],
            outputs=[slot_feedback_md, feedback_calls_state],
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
