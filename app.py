"""Hugging Face Spaces entry point — Gradio interface for ATS CV Scorer."""

import gradio as gr
import gradio_client.utils as _gradio_client_utils

from src.core.config import settings
from src.core.schemas import CVQualityReport, NormalizedCV
from src.services.claude_feedback import ClaudeFeedback
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

_cv_transformer = CVTransformer()
_nlp_pipeline = NLPPipeline()
_semantic_scorer = SemanticScorer()
_cv_quality_scorer = CVQualityScorer()
_job_search_service = JobSearchService(
    app_id=settings.adzuna_id,
    app_key=settings.adzuna_api_key,
    country=settings.adzuna_country,
)

MAX_JOB_RESULTS = 10


def _score_color(score: float) -> str:
    return "🟢" if score >= 70 else ("🟡" if score >= 45 else "🔴")


def _bar(value: int, max_val: int = 100, width: int = 20) -> str:
    filled = round(value / max_val * width)
    return "█" * filled + "░" * (width - filled)


def _format_quality_report(cv: NormalizedCV, report: CVQualityReport) -> str:
    layout_badge = (
        "✅ 1 colonne (optimal ATS)" if report.layout == "single_column" else "⚠️ 2 colonnes (risque parseur ATS)"
    )

    lines = [
        "## 📊 Qualité du CV",
        "",
        f"**Score global : {report.score_global}/100**",
        f"`{_bar(report.score_global)}` {report.score_global}%",
        "",
        "| Dimension | Score | Barre |",
        "|---|---|---|",
        f"| Structure | {report.score_structure}/100 | `{_bar(report.score_structure, width=10)}` |",
        f"| Contenu   | {report.score_contenu}/100 | `{_bar(report.score_contenu, width=10)}` |",
        "",
        f"📐 Layout : {layout_badge}  ·  📝 {report.word_count} mots  ·  📊 Densité mots-clés : {report.keyword_density:.1%}",
        "",
    ]

    if report.sections_detectees:
        lines += [
            "### ✅ Sections détectées",
            "  ".join(f"`{s}`" for s in report.sections_detectees),
            "",
        ]

    if report.sections_manquantes:
        lines += [
            "### ❌ Sections manquantes",
            "  ".join(f"`{s}`" for s in report.sections_manquantes),
            "",
        ]

    badges = []
    if report.has_metrics:
        badges.append("✅ Métriques quantifiées")
    else:
        badges.append("❌ Pas de métriques")
    lines += ["  ".join(badges), ""]

    if report.career_gaps:
        lines += ["### ⚠️ Trous de carrière détectés"]
        for gap in report.career_gaps:
            lines.append(f"- {gap}")
        lines.append("")

    if report.recommendations:
        lines += ["### 💡 Recommandations (par impact ATS décroissant)"]
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    return "\n".join(lines)


def _format_timeline(cv: NormalizedCV, report: CVQualityReport) -> str:
    if not cv.experience and not cv.education:
        return ""

    lines = ["## 📅 Timeline carrière"]
    if report.total_experience_years > 0:
        lines.append(f"**Expérience totale : {report.total_experience_years:.1f} an(s)**")
    if report.career_start_year:
        lines.append(f"Premier poste détecté : {report.career_start_year}")
    lines.append("")

    events: list[tuple[int, int, str]] = []
    for e in cv.experience:
        label_parts = [p for p in [e.title, e.company] if p]
        label = " @ ".join(label_parts) if label_parts else "Poste non détecté"
        dur = f" ({e.years:.0f} an(s))" if e.years else ""
        period = e.period or (
            f"{e.date_start or '?'} – {'présent' if e.is_current else (e.date_end or '?')}"
        )
        s_year = int(e.date_start.split("-")[0]) if e.date_start else 0
        s_month = int(e.date_start.split("-")[1]) if e.date_start and "-" in e.date_start else 1
        events.append((s_year, s_month, f"💼 **{period}** : {label}{dur}"))

    for e in cv.education:
        label = " — ".join(p for p in [e.degree, e.school] if p) or "Formation non détectée"
        dur = f" ({round((e.duration_months or 0) / 12):.0f} an(s))" if e.duration_months else ""
        period = e.year or (
            f"{e.date_start or '?'} – {'présent' if e.is_current else (e.date_end or '?')}"
        )
        s_year = int(e.date_start.split("-")[0]) if e.date_start else 0
        s_month = int(e.date_start.split("-")[1]) if e.date_start and "-" in e.date_start else 1
        events.append((s_year, s_month, f"🎓 **{period}** : {label}{dur}"))

    events.sort(key=lambda x: (x[0], x[1]), reverse=True)
    for _, _, label in events:
        lines.append(f"- {label}")

    if report.career_gaps:
        lines.append("")
        lines.append("**Périodes sans activité détectée :**")
        for gap in report.career_gaps:
            lines.append(f"  - ⚠️ {gap}")

    return "\n".join(lines)


def _format_profile_summary(parsed_cv, normalized_cv: NormalizedCV) -> str:
    """Profil détecté affiché dans l'onglet Analyse."""
    name = normalized_cv.header.name or ""
    title = parsed_cv.job_title or normalized_cv.header.title or ""
    location = parsed_cv.location or normalized_cv.header.location or ""

    header = " · ".join(p for p in [name, title] if p) or "Profil détecté"
    meta = []
    if location:
        meta.append(f"📍 {location}")
    top_skills = (parsed_cv.skills or [])[:6]
    if top_skills:
        meta.append("🛠 " + ", ".join(top_skills))

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
    color = _score_color(report.score_global)
    parts.append(f"{color} Score CV : **{report.score_global}/100**")
    return "✅ Profil chargé — " + " · ".join(parts)


# ── Handler 1 : traitement du CV ─────────────────────────────────────────────

def on_cv_upload(cv_file):
    _reset = (
        "", "", "",          # quality_md, timeline_md, profile_md
        None, None, None,    # parsed_cv_state, normalized_cv_state, quality_report_state
        "⚠️ Uploadez votre CV en onglet 1 pour commencer.",  # cv_context_md
        gr.update(interactive=False),                         # search_btn
    )

    if cv_file is None:
        return _reset

    normalized_cv = _cv_transformer.transform(cv_file.name)
    if not normalized_cv.raw_text:
        return (
            "❌ Impossible d'extraire le texte du PDF.",
            "", "",
            None, None, None,
            "❌ Échec de l'extraction PDF — vérifiez que le fichier n'est pas protégé.",
            gr.update(interactive=False),
        )

    parsed_cv = _nlp_pipeline.parse_normalized(normalized_cv)
    quality_report = _cv_quality_scorer.score(normalized_cv)

    return (
        _format_quality_report(normalized_cv, quality_report),
        _format_timeline(normalized_cv, quality_report),
        _format_profile_summary(parsed_cv, normalized_cv),
        parsed_cv,
        normalized_cv,
        quality_report,
        _format_cv_context_strip(parsed_cv, quality_report),
        gr.update(interactive=True),
    )


def on_cv_clear():
    return (
        "", "", "",
        None, None, None,
        "⚠️ Uploadez votre CV en onglet 1 pour commencer.",
        gr.update(interactive=False),
    )


# ── Handler 2 : recherche et scoring des offres ───────────────────────────────

def _empty_job_slots() -> list:
    result = []
    for _ in range(MAX_JOB_RESULTS):
        result += _slot_updates()
    return result


def _format_job_card(rank: int, match) -> str:
    job, result = match.job, match.scoring_result
    meta = [
        part
        for part in (
            f"**{job.company}**" if job.company else None,
            f"📍 {job.location}" if job.location else None,
            job.contract_type,
        )
        if part
    ]
    if job.salary_min or job.salary_max:
        lo = f"{int(job.salary_min):,}".replace(",", " ") if job.salary_min else "?"
        hi = f"{int(job.salary_max):,}".replace(",", " ") if job.salary_max else "?"
        meta.append(f"💰 {lo} – {hi} €/an")

    return f"""### {_score_color(result.overall_score)} #{rank} — {job.title} *({result.overall_score:.0f}/100)*
{" · ".join(meta)}

[Voir l'offre complète ↗]({job.url})
"""


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


def on_search(parsed_cv, region):
    if parsed_cv is None:
        status = "⚠️ Uploadez d'abord votre CV en onglet 1."
        return [status, []] + _empty_job_slots()

    matches = find_matching_jobs(
        parsed_cv,
        _job_search_service,
        _semantic_scorer,
        max_results=MAX_JOB_RESULTS,
        region=region or None,
    )

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

    if matches:
        status = (
            f"## ✅ {len(matches)} offre(s) trouvée(s), classées par score de compatibilité décroissant\n"
            f"{profile_line}"
        )
    else:
        status = (
            f"## 😕 Aucune offre ne correspond à votre profil pour le moment\n{profile_line}\n\n"
            "Essayez avec un CV mentionnant plus explicitement votre **intitulé de poste**, "
            "vos **compétences clés** et votre **ville**, ou réessayez un peu plus tard."
        )

    outputs = [status, matches]
    for i in range(MAX_JOB_RESULTS):
        outputs += _slot_updates(rank=i + 1, match=matches[i]) if i < len(matches) else _slot_updates()
    return outputs


# ── Handler 3 : feedback Claude par offre (à la demande) ─────────────────────

_ANALYZE_LOADING_MD = (
    "⏳ **Analyse en cours…** Claude examine cette offre au regard de votre "
    "profil — merci de patienter quelques secondes."
)


def _make_analyze_handler(index: int):
    def _analyze_one(parsed_cv, matches):
        if not matches or index >= len(matches):
            yield gr.update(value="", visible=False)
            return

        yield gr.update(value=_ANALYZE_LOADING_MD, visible=True)

        match = matches[index]
        try:
            cf = ClaudeFeedback()
            feedback = cf.generate_feedback(
                parsed_cv, match.job.description, match.scoring_result
            )
        except ValueError:
            feedback = (
                "⚠️ Configurez `ANTHROPIC_API_KEY` dans le fichier `.env` pour générer "
                "un feedback personnalisé pour cette offre."
            )
        except Exception as exc:
            feedback = f"⚠️ Feedback Claude indisponible : {exc}"
        yield gr.update(value=feedback, visible=True)

    return _analyze_one


# ── Interface Gradio ──────────────────────────────────────────────────────────

with gr.Blocks(
    title="ATS CV Scorer",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
) as demo:
    gr.Markdown("# 📋 ATS CV Scorer")

    parsed_cv_state = gr.State(None)
    normalized_cv_state = gr.State(None)
    quality_report_state = gr.State(None)
    matches_state = gr.State([])

    with gr.Tabs():

        # ── Onglet 1 : Analyse du CV ──────────────────────────────────────────
        with gr.Tab("📋 Analyse du CV"):
            gr.Markdown(
                "Uploadez votre CV pour obtenir immédiatement une **analyse qualité ATS** — "
                "score structure/contenu, timeline carrière, compétences détectées et recommandations. "
                "Aucun appel réseau externe n'est déclenché à cette étape."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    cv_input = gr.File(label="CV (PDF)", file_types=[".pdf"])
                    profile_md = gr.Markdown()
                with gr.Column(scale=2):
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
                region_dropdown = gr.Dropdown(
                    choices=FRANCE_REGIONS,
                    label="Région (prioritaire si sélectionnée — remplace la localisation du CV pour la recherche Adzuna)",
                    value=None,
                    scale=3,
                )
                search_btn = gr.Button(
                    "🔍 Rechercher des offres correspondantes",
                    variant="primary",
                    interactive=False,
                    scale=1,
                )
            search_status_md = gr.Markdown()

            job_slots = []
            for _ in range(MAX_JOB_RESULTS):
                with gr.Group(visible=False) as group:
                    job_md = gr.Markdown()
                    slot_analyze_btn = gr.Button(
                        "🔎 Analyser cette offre", size="sm", visible=False
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
    ]

    cv_input.upload(fn=on_cv_upload, inputs=[cv_input], outputs=_upload_outputs)
    cv_input.clear(fn=on_cv_clear, outputs=_upload_outputs)

    _search_outputs = [search_status_md, matches_state]
    for group, job_md, slot_analyze_btn, slot_feedback_md in job_slots:
        _search_outputs += [group, job_md, slot_analyze_btn, slot_feedback_md]

    search_btn.click(
        fn=on_search,
        inputs=[parsed_cv_state, region_dropdown],
        outputs=_search_outputs,
    )

    for idx, (_, _, slot_analyze_btn, slot_feedback_md) in enumerate(job_slots):
        slot_analyze_btn.click(
            fn=_make_analyze_handler(idx),
            inputs=[parsed_cv_state, matches_state],
            outputs=[slot_feedback_md],
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
