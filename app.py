"""Hugging Face Spaces entry point — Gradio interface for ATS CV Scorer."""

import os
import tempfile
import time

import gradio as gr

from src.services.claude_feedback import ClaudeFeedback
from src.services.nlp_pipeline import NLPPipeline
from src.services.pdf_extractor import PDFExtractor
from src.services.semantic_scorer import SemanticScorer

_pdf_extractor = PDFExtractor()
_nlp_pipeline = NLPPipeline()
_semantic_scorer = SemanticScorer()


def analyze(cv_file, job_description: str, api_key: str):
    if cv_file is None:
        return "⚠️ Veuillez uploader un CV.", "", "", ""
    if not job_description.strip():
        return "⚠️ Veuillez coller une offre d'emploi.", "", "", ""

    start = time.perf_counter()

    cv_text = _pdf_extractor.extract(cv_file.name)
    if not cv_text:
        return "❌ Impossible d'extraire le texte du PDF.", "", "", ""

    parsed_cv = _nlp_pipeline.parse_cv(cv_text)
    result = _semantic_scorer.score(parsed_cv, job_description)

    feedback_md = "_Ajoutez une clé API Claude pour obtenir des recommandations personnalisées._"
    effective_key = api_key.strip() or os.getenv("ANTHROPIC_API_KEY", "")
    if effective_key:
        try:
            cf = ClaudeFeedback(api_key=effective_key)
            feedback_md = cf.generate_feedback(parsed_cv, job_description, result)
            result.feedback = feedback_md
        except Exception as exc:
            feedback_md = f"⚠️ Feedback Claude indisponible : {exc}"

    elapsed = round(time.perf_counter() - start, 2)

    score_color = "🟢" if result.overall_score >= 70 else ("🟡" if result.overall_score >= 45 else "🔴")
    score_md = f"""## {score_color} Score ATS global : **{result.overall_score} / 100**

| Critère | Score |
|---|---|
| Correspondance mots-clés | {result.breakdown.keyword_match:.1f} / 100 |
| Similarité sémantique | {result.breakdown.semantic_similarity:.1f} / 100 |
| Complétude structure | {result.breakdown.structure_completeness:.1f} / 100 |

*Analysé en {elapsed}s — sections détectées : {", ".join(parsed_cv.sections.keys()) or "aucune"}*
"""

    keywords_md = f"""### ✅ Mots-clés trouvés ({len(result.matched_keywords)})
{", ".join(result.matched_keywords) or "_Aucun_"}

### ❌ Mots-clés manquants ({len(result.missing_keywords)})
{", ".join(result.missing_keywords) or "_Aucun_"}
"""

    skills_md = f"""**Compétences détectées ({len(parsed_cv.skills)}) :**
{", ".join(parsed_cv.skills) or "_Aucune_"}

**Expérience estimée :** {f"{parsed_cv.experience_years} ans" if parsed_cv.experience_years else "_Non déterminée_"}
"""

    return score_md, keywords_md, skills_md, feedback_md


with gr.Blocks(
    title="ATS CV Scorer",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
) as demo:
    gr.Markdown(
        """# 📋 ATS CV Scorer
        Analysez l'adéquation de votre CV avec une offre d'emploi grâce au NLP et à Claude AI.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            cv_input = gr.File(label="CV (PDF)", file_types=[".pdf"])
            jd_input = gr.Textbox(
                label="Offre d'emploi",
                placeholder="Collez ici le texte complet de l'offre…",
                lines=10,
            )
            api_key_input = gr.Textbox(
                label="Clé API Claude (optionnel)",
                placeholder="sk-ant-…",
                type="password",
            )
            analyze_btn = gr.Button("Analyser", variant="primary")

        with gr.Column(scale=2):
            score_out = gr.Markdown(label="Score ATS")
            with gr.Tabs():
                with gr.Tab("Mots-clés"):
                    keywords_out = gr.Markdown()
                with gr.Tab("Compétences"):
                    skills_out = gr.Markdown()
                with gr.Tab("Feedback Claude"):
                    feedback_out = gr.Markdown()

    analyze_btn.click(
        fn=analyze,
        inputs=[cv_input, jd_input, api_key_input],
        outputs=[score_out, keywords_out, skills_out, feedback_out],
    )

    gr.Examples(
        examples=[
            [None, "We are looking for a Senior ML Engineer proficient in Python, PyTorch, and MLflow. Experience with Kubernetes and AWS is required. Strong NLP background preferred.", ""],
        ],
        inputs=[cv_input, jd_input, api_key_input],
    )

if __name__ == "__main__":
    demo.launch()
