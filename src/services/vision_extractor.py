"""Claude Vision LLM extraction — cascade level 3 of CVTransformer.

Used when pdfplumber/OCR confidence stays below threshold: renders the first
page of the PDF as an image and asks Claude to extract a structured CV as
JSON, which is then validated into a NormalizedCV.
"""

import base64
import io
import json
import logging
import re

import anthropic

from src.core.config import settings
from src.core.schemas import NormalizedCV

logger = logging.getLogger(__name__)


class VisionExtractionError(Exception):
    """Levée quand l'appel Claude Vision échoue (rate limit, erreur serveur, réseau).

    Le message porté par cette exception est destiné à l'utilisateur final
    (français, non technique) ; le détail technique est loggé séparément.
    """


class VisionExtractor:
    """Extracts a NormalizedCV from a PDF page image via Claude Vision."""

    def extract(self, pdf_path: str) -> NormalizedCV:
        """Extraire le NormalizedCV via Claude Vision (image du PDF).

        Nécessite ANTHROPIC_API_KEY dans .env.
        Coût estimé : ~0.01 $ / CV (claude-sonnet-4-6, 1 image).
        """
        try:
            from pdf2image import convert_from_path  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("pdf2image requis pour le fallback Vision LLM") from exc

        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY absent — Vision LLM désactivé")

        images = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=1)
        if not images:
            raise RuntimeError("Impossible de convertir le PDF en image")

        buf = io.BytesIO()
        images[0].save(buf, format="PNG")
        img_b64 = base64.standard_b64encode(buf.getvalue()).decode()

        client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.vision_timeout_seconds,
        )

        system_prompt = (
            "Tu es un extracteur de CV expert. "
            "Extrais TOUTES les informations visibles dans ce CV, même partielles. "
            "Ne rien omettre. Retourne UNIQUEMENT un JSON valide, sans markdown."
        )
        user_prompt = (
            "Extrais ce CV avec le maximum de détails et retourne un JSON avec ces clés exactes :\n"
            '{"header": {"name": null, "title": null, "email": null, "phone": null, '
            '"location": null, "postal_code": null, "github": null, "linkedin": null}, '
            '"summary": null, '
            '"skills": {"ml": [], "mlops": [], "cloud": [], "languages": [], '
            '"data": [], "other": [], "commerce": []}, '
            '"experience": [{"title": null, "company": null, "date_start": null, '
            '"date_end": null, "is_current": false, "bullets": []}], '
            '"education": [{"degree": "", "school": "", "date_start": null, '
            '"date_end": null, "is_current": false}], '
            '"projects": [], "languages": []}\n\n'
            "Instructions importantes :\n"
            "- skills : inclure TOUS les outils visibles, y compris ceux mentionnés "
            "dans les sections formation/éducation (ex. XGBoost, FastAPI, Docker appris en formation "
            "→ les mettre dans skills, pas seulement dans education).\n"
            "- experience.bullets : retranscrire chaque point de manière complète.\n"
            "- education : inclure TOUTES les formations, certifications et MOOCs visibles.\n"
            "- projects : inclure les projets avec stack technique et métriques si présents.\n"
            "- languages : liste de strings simples (ex. [\"Français\", \"Anglais\"])."
        )

        try:
            response = client.messages.create(
                model=settings.claude_model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": img_b64},
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                }],
            )
        except anthropic.RateLimitError as exc:
            logger.warning("vision_extractor | rate limit Claude : %s", exc)
            raise VisionExtractionError(
                "Le service IA Vision est temporairement surchargé."
            ) from exc
        except anthropic.APIConnectionError as exc:
            logger.warning("vision_extractor | erreur de connexion Claude : %s", exc)
            raise VisionExtractionError(
                "Impossible de contacter le service IA Vision."
            ) from exc
        except anthropic.APIStatusError as exc:
            logger.warning("vision_extractor | erreur API Claude (%s) : %s", exc.status_code, exc)
            raise VisionExtractionError(
                "Le service IA Vision a rencontré une erreur."
            ) from exc

        raw_json = response.content[0].text
        # Strip markdown code fences if present
        raw_json = re.sub(r"```(?:json)?\s*", "", raw_json).strip()
        start, end = raw_json.find("{"), raw_json.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("Aucun JSON dans la réponse Vision LLM")

        data = json.loads(raw_json[start:end])

        # Reconstruct raw_text from all text fields for skill matching and word count comparison.
        # All items are converted to str — Claude peut varier la structure de ses listes.
        def _s(v: object) -> str:
            return str(v) if v is not None else ""

        def _strs(lst: object) -> list[str]:
            if not isinstance(lst, list):
                return []
            return [_s(i) for i in lst if i is not None and str(i).strip()]

        raw_parts: list[str] = []
        hdr = data.get("header") or {}
        raw_parts.extend(filter(None, [hdr.get("name"), hdr.get("title"),
                                       hdr.get("location"), hdr.get("linkedin")]))
        if data.get("summary"):
            raw_parts.append(_s(data["summary"]))
        for exp in data.get("experience", []):
            if not isinstance(exp, dict):
                continue
            raw_parts.extend(filter(None, [exp.get("title"), exp.get("company")]))
            raw_parts.extend(_strs(exp.get("bullets")))
        for edu in data.get("education", []):
            if not isinstance(edu, dict):
                continue
            raw_parts.extend(filter(None, [edu.get("degree"), edu.get("school")]))
        for proj in data.get("projects", []):
            if isinstance(proj, dict):
                raw_parts.extend(filter(None, [proj.get("name"), proj.get("description")]))
            elif isinstance(proj, str):
                raw_parts.append(proj)
        skills_data = data.get("skills") or {}
        for skill_list in skills_data.values():
            raw_parts.extend(_strs(skill_list))
        raw_parts.extend(_strs(data.get("languages")))
        data["raw_text"] = " ".join(s for s in raw_parts if isinstance(s, str) and s.strip())
        data["layout_detected"] = "single_column"
        data["word_count"] = len(data["raw_text"].split())
        data["extraction_method"] = "vision_llm"
        data["extraction_confidence"] = 0.95

        # Normaliser les champs liste[str] — Claude peut retourner des dicts à la place
        def _flatten_str_list(lst: object) -> list[str]:
            if not isinstance(lst, list):
                return []
            result = []
            for item in lst:
                if isinstance(item, str):
                    result.append(item)
                elif isinstance(item, dict):
                    # Prendre la première valeur de type str trouvée
                    for v in item.values():
                        if isinstance(v, str) and v.strip():
                            result.append(v)
                            break
            return result

        data["languages"] = _flatten_str_list(data.get("languages"))

        return NormalizedCV.model_validate(data)


def build_raw_text(cv: NormalizedCV) -> str:
    """Reconstruire raw_text depuis tous les champs structurés d'un NormalizedCV.

    Utilisé après Vision LLM pour obtenir un word_count fidèle au contenu
    réel du CV plutôt qu'au texte JSON brut.
    """
    parts: list[str] = []

    if cv.header.name:
        parts.append(cv.header.name)
    if cv.header.title:
        parts.append(cv.header.title)
    if cv.header.location:
        parts.append(cv.header.location)

    if cv.summary:
        parts.append(cv.summary)

    for category, skills in cv.skills.model_dump().items():
        if isinstance(skills, list):
            parts.extend(str(s) for s in skills if s)

    for exp in cv.experience:
        if exp.title:
            parts.append(exp.title)
        if exp.company:
            parts.append(exp.company)
        if exp.period:
            parts.append(exp.period)
        parts.extend(exp.bullets)

    for edu in cv.education:
        if edu.degree:
            parts.append(edu.degree)
        if edu.school:
            parts.append(edu.school)
        parts.extend(edu.skills)

    for proj in cv.projects:
        if proj.name:
            parts.append(proj.name)
        if proj.description:
            parts.append(proj.description)
        parts.extend(proj.stack)
        parts.extend(proj.metrics)

    return " ".join(p for p in parts if p and str(p).strip())


def richness_score(cv: NormalizedCV) -> float:
    """Score de richesse structurelle d'un NormalizedCV.

    Utilisé pour comparer Vision LLM vs pdfplumber/OCR sans dépendre du
    word_count brut (qui compte le texte décoratif pour pdfplumber et les
    noms de clés JSON pour Vision LLM).
    """
    score = 0.0
    score += len(cv.experience) * 10
    score += len(cv.education) * 8
    score += sum(len(v) for v in [
        cv.skills.ml,
        cv.skills.mlops,
        cv.skills.cloud,
        cv.skills.languages,
        cv.skills.data,
        cv.skills.other,
    ]) * 2
    score += len(cv.projects) * 10
    score += 15 if cv.header.email else 0
    score += 15 if cv.header.title else 0
    score += 10 if cv.summary else 0
    return score
