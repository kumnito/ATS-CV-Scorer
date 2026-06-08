"""Hugging Face Spaces entry point — Gradio interface for ATS CV Scorer."""

import gradio as gr
import gradio_client.utils as _gradio_client_utils

from src.core.config import settings
from src.services.claude_feedback import ClaudeFeedback
from src.services.job_matcher import FRANCE_REGIONS, find_matching_jobs
from src.services.job_search import JobSearchService
from src.services.nlp_pipeline import NLPPipeline
from src.services.pdf_extractor import PDFExtractor
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

_pdf_extractor = PDFExtractor()
_nlp_pipeline = NLPPipeline()
_semantic_scorer = SemanticScorer()
_job_search_service = JobSearchService(
    app_id=settings.adzuna_id,
    app_key=settings.adzuna_api_key,
    country=settings.adzuna_country,
)

MAX_JOB_RESULTS = 10


def _score_color(score: float) -> str:
    return "🟢" if score >= 70 else ("🟡" if score >= 45 else "🔴")


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


def _build_search_response(status: str, parsed_cv, matches: list) -> list:
    outputs = [status, parsed_cv, matches]
    for i in range(MAX_JOB_RESULTS):
        if i < len(matches):
            outputs += _slot_updates(rank=i + 1, match=matches[i])
        else:
            outputs += _slot_updates()
    return outputs


def search_jobs(cv_file, region):
    if cv_file is None:
        return _build_search_response("⚠️ Veuillez uploader un CV.", None, [])

    cv_text = _pdf_extractor.extract(cv_file.name)
    if not cv_text:
        return _build_search_response(
            "❌ Impossible d'extraire le texte du PDF.", None, []
        )

    parsed_cv = _nlp_pipeline.parse_cv(cv_text)
    matches = find_matching_jobs(
        parsed_cv,
        _job_search_service,
        _semantic_scorer,
        max_results=MAX_JOB_RESULTS,
        region=region or None,
    )

    profile_line = f"**Profil détecté :** {parsed_cv.job_title or '_non déterminé_'}"
    if parsed_cv.location:
        profile_line += f" · 📍 {parsed_cv.location} (rayon 30 km)"
    if region:
        profile_line += f" · 🗺️ {region}"

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

    return _build_search_response(status, parsed_cv, matches)


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


with gr.Blocks(
    title="ATS CV Scorer",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
) as demo:
    gr.Markdown(
        """# 📋 ATS CV Scorer — Recherche automatique d'offres
        Uploadez votre CV : le pipeline NLP en extrait votre **titre de poste**, vos
        **compétences** et votre **localisation**, recherche des offres correspondantes
        via l'API **Adzuna**, puis les score automatiquement contre votre profil avec un
        moteur de similarité sémantique. Vous pouvez ensuite demander un **feedback Claude
        personnalisé pour chaque offre, à la demande** — aucun appel IA n'est déclenché
        tant que vous ne cliquez pas sur "Analyser".
        """
    )

    parsed_cv_state = gr.State(None)
    matches_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            job_cv_input = gr.File(label="CV (PDF)", file_types=[".pdf"])
            region_dropdown = gr.Dropdown(
                choices=FRANCE_REGIONS,
                label="Région (optionnel — repli si la localisation du CV ne donne aucun résultat à 30 km)",
                value=None,
            )
            search_btn = gr.Button(
                "🔍 Rechercher des offres correspondantes", variant="primary"
            )
            search_status_md = gr.Markdown()

        with gr.Column(scale=2):
            job_slots = []
            for _ in range(MAX_JOB_RESULTS):
                with gr.Group(visible=False) as group:
                    job_md = gr.Markdown()
                    slot_analyze_btn = gr.Button(
                        "🔎 Analyser cette offre", size="sm", visible=False
                    )
                    slot_feedback_md = gr.Markdown(visible=False)
                job_slots.append((group, job_md, slot_analyze_btn, slot_feedback_md))

    search_outputs = [search_status_md, parsed_cv_state, matches_state]
    for group, job_md, slot_analyze_btn, slot_feedback_md in job_slots:
        search_outputs += [group, job_md, slot_analyze_btn, slot_feedback_md]

    search_btn.click(
        fn=search_jobs, inputs=[job_cv_input, region_dropdown], outputs=search_outputs
    )

    for idx, (_, _, slot_analyze_btn, slot_feedback_md) in enumerate(job_slots):
        slot_analyze_btn.click(
            fn=_make_analyze_handler(idx),
            inputs=[parsed_cv_state, matches_state],
            outputs=[slot_feedback_md],
        )

if __name__ == "__main__":
    demo.launch()
