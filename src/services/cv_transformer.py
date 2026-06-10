"""Layout-aware PDF → NormalizedCV transformer.

Reads character-level positional data from pdfplumber (x0, top, font size,
font name) to detect single-column vs two-column layouts and reconstruct a
semantically-ordered text stream before NLP parsing.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

import pdfplumber

from src.core.lexicons import (
    ACTION_VERBS_EN,
    ACTION_VERBS_FR,
    JOB_TITLE_RE,
    METRIC_PATTERNS,
    POSTAL_CODE_CITY_RE,
    SECTION_HEADERS,
    SKILL_CATEGORIES,
    TITLE_SPLIT_RE,
)
from src.core.config import settings
from src.core.schemas import (
    CVEducation,
    CVExperience,
    CVHeader,
    CVProject,
    CVSkills,
    NormalizedCV,
)
from src.services.semantic_skill_matcher import match_semantic_skills

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(
    r"(?:\+33\s?|0)(?:[67]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{2}|\d[\s.\-]?\d{2}[\s.\-]?\d{2}[\s.\-]?\d{2}[\s.\-]?\d{2})"
)
_GITHUB_RE = re.compile(r"github\.com/[\w\-]+(?:/[\w\-]+)?", re.IGNORECASE)
_LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w\-]+", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE)

_BULLET_CHARS = frozenset("•–-—▪◦◆▸→★*·")
_BULLET_RE = re.compile(r"^[\s]*[•–\-—▪◦◆▸→★*·]\s*")

_MONTH_MAP: dict[str, int] = {
    "jan": 1, "janv": 1, "janvier": 1, "january": 1,
    "feb": 2, "fév": 2, "fevr": 2, "févr": 2, "février": 2, "february": 2,
    "mar": 3, "mars": 3, "march": 3,
    "avr": 4, "avril": 4, "april": 4, "apr": 4,
    "mai": 5, "may": 5,
    "jun": 6, "juin": 6, "june": 6,
    "jul": 7, "juil": 7, "juillet": 7, "july": 7,
    "aug": 8, "aou": 8, "août": 8, "aout": 8, "august": 8,
    "sep": 9, "sept": 9, "septembre": 9, "september": 9,
    "oct": 10, "octobre": 10, "october": 10,
    "nov": 11, "novembre": 11, "november": 11,
    "dec": 12, "déc": 12, "décembre": 12, "december": 12,
}

_CURRENT_RE = re.compile(
    r"\b(present|aujourd[''`]?hui|en\s*cours|current|maintenant|now|présent|actuel)\b",
    re.IGNORECASE,
)

_MONTH_YEAR_RE = re.compile(
    r"\b(jan(?:v(?:ier)?)?|fév(?:r(?:ier)?)?|fevr?(?:ier)?|mars?|avr(?:il)?|mai|"
    r"juin|juil(?:let)?|ao[uû]t|sept?(?:embre)?|oct(?:obre)?|nov(?:embre)?|"
    r"d[eé]c(?:embre)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"[\s./]*([12]\d{3})\b",
    re.IGNORECASE,
)

_YEAR_ONLY_RE = re.compile(r"\b(19|20)(\d{2})\b")

_DATE_RANGE_RE = re.compile(
    r"("
    r"(?:(?:jan|fév|fevr?|mars?|avr|mai|juin|juil|ao[uû]t|sept?|oct|nov|d[eé]c|"
    r"jan(?:v)?|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\.?\s+)?"
    r"[12]\d{3}"
    r")"
    r"\s*[-–—/]\s*"
    r"("
    r"(?:(?:jan|fév|fevr?|mars?|avr|mai|juin|juil|ao[uû]t|sept?|oct|nov|d[eé]c|"
    r"jan(?:v)?|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\.?\s+)?"
    r"(?:[12]\d{3}|present|aujourd[''`]?hui|en\s*cours|current|maintenant|now|présent|actuel)"
    r")",
    re.IGNORECASE,
)

_EXPLICIT_DURATION_RE = re.compile(
    r"\b(\d+)\s*(?:mois|months?)\b|\b(\d+)(?:[.,]\d+)?\s*(?:ans?|years?)\b",
    re.IGNORECASE,
)

_REQUIRED_SECTIONS = {"experience", "education", "skills"}
_PREFERRED_ORDER = ["skills", "education"]  # skills should appear before education ideally


@dataclass
class _Line:
    text: str
    max_size: float
    is_bold: bool
    x0: float
    top: float
    column: str = "full"  # "left", "right", "full"


@dataclass
class _LayoutInfo:
    layout_type: str  # "single_column" | "two_columns"
    gutter_x: Optional[float] = None  # x position of column split
    body_y_start: float = 0.0  # y au-dessus duquel c'est l'en-tête (bandeau coloré)


# ---------------------------------------------------------------------------
# Vision LLM richness score
# ---------------------------------------------------------------------------


def _vision_richness_score(cv: "NormalizedCV") -> float:
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


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class CVTransformer:
    """Transform a PDF CV into a structured NormalizedCV.

    Uses positional character data from pdfplumber (x0, top, font size,
    font name) to detect layout and reconstruct correct reading order before
    parsing sections, header, skills, experience, education, and projects.
    """

    # ------------------------------------------------------------------
    # Public entry point — cascade : pdfplumber → OCR → Vision LLM
    # ------------------------------------------------------------------

    def transform(self, pdf_path: str, allow_vision: bool = True) -> NormalizedCV:
        """Extraire un NormalizedCV depuis un PDF, avec fallback automatique.

        Cascade à 3 niveaux :
          1. pdfplumber confiance ≥ 0.85 → retourner directement
          2. pdfplumber < 0.85 → OCR ; OCR gagne si +10 % de mots
             si confiance résultante ≥ 0.85 → pas de Vision LLM
          3. confiance < 0.85 + clé Anthropic + VISION_LLM_ENABLED + allow_vision
             → Vision LLM. Vision LLM gagne si son score de richesse
             structurelle dépasse celui du meilleur résultat pdfplumber/OCR

        `allow_vision` permet à l'appelant de désactiver le niveau 3 pour cet
        appel (ex. quota Vision LLM par session déjà atteint), indépendamment
        de la configuration globale.

        Le layout_detected est toujours calculé depuis les données pdfplumber
        réelles et injecté dans le résultat final, quelle que soit la méthode
        d'extraction retenue.
        """
        _CONFIDENCE_THRESHOLD = 0.85

        words_per_page = self._extract_words(pdf_path)
        all_words = [w for page in words_per_page for w in page]
        pdf_confidence = round(min(1.0, len(all_words) / 450.0), 2)
        logger.info("cascade | pdfplumber: %d mots, confiance %.2f", len(all_words), pdf_confidence)

        # Layout toujours détecté sur le PDF réel, indépendamment de la cascade
        layout_info = self._detect_layout(words_per_page)
        real_layout = layout_info.layout_type

        def _with_real_layout(cv: NormalizedCV) -> NormalizedCV:
            if cv.layout_detected != real_layout:
                return cv.model_copy(update={"layout_detected": real_layout})
            return cv

        # --- Niveau 1 : pdfplumber suffisant ---
        if pdf_confidence >= _CONFIDENCE_THRESHOLD:
            logger.info("cascade | niveau 1 → pdfplumber direct")
            return self._transform_from_words(
                words_per_page,
                extraction_method="pdfplumber",
                extraction_confidence=pdf_confidence,
            )

        # --- Niveau 2 : OCR ---
        ocr_text = ""
        try:
            ocr_text = self._extract_text_ocr(pdf_path)
        except Exception as exc:
            logger.info("cascade | OCR indisponible : %s", exc)

        ocr_word_count = len(ocr_text.split()) if ocr_text else 0
        logger.info("cascade | OCR: %d mots", ocr_word_count)

        _OCR_WINS_THRESHOLD = 1.10
        if ocr_word_count > len(all_words) * _OCR_WINS_THRESHOLD and ocr_word_count >= 150:
            best_confidence = round(min(1.0, ocr_word_count / 450.0), 2)
            use_ocr = True
            logger.info("cascade | meilleur = OCR (%d mots, conf %.2f)", ocr_word_count, best_confidence)
        elif len(all_words) >= 150:
            best_confidence = pdf_confidence
            use_ocr = False
            logger.info("cascade | meilleur = pdfplumber (%d mots, conf %.2f)", len(all_words), best_confidence)
        else:
            best_word_count = max(len(all_words), ocr_word_count)
            best_confidence = round(min(1.0, best_word_count / 450.0), 2)
            use_ocr = ocr_word_count >= len(all_words)
            logger.info("cascade | meilleur = %s (%d mots, conf %.2f)",
                        "OCR" if use_ocr else "pdfplumber", best_word_count, best_confidence)

        if best_confidence >= _CONFIDENCE_THRESHOLD:
            logger.info("cascade | niveau 2 → confiance suffisante, pas de Vision LLM")
            if use_ocr:
                return _with_real_layout(self._transform_from_text(ocr_text, "ocr", best_confidence))
            return self._transform_from_words(words_per_page, "pdfplumber", best_confidence)

        # --- Construction du meilleur résultat pdfplumber/OCR pour comparaison ---
        if use_ocr and ocr_word_count >= 30:
            best_cv = _with_real_layout(self._transform_from_text(ocr_text, "ocr", best_confidence))
        elif len(all_words) >= 30:
            best_cv = self._transform_from_words(words_per_page, "pdfplumber", pdf_confidence)
        else:
            best_cv = NormalizedCV(extraction_method="pdfplumber", extraction_confidence=0.0,
                                   layout_detected=real_layout)

        # --- Niveau 3 : Vision LLM (comparaison par richesse structurelle) ---
        if settings.anthropic_api_key and settings.vision_llm_enabled and allow_vision:
            logger.info("cascade | niveau 3 → Vision LLM déclenché (conf %.2f < %.2f)",
                        best_confidence, _CONFIDENCE_THRESHOLD)
            best_richness = _vision_richness_score(best_cv)
            logger.info("cascade | %s richness: %.0f", best_cv.extraction_method, best_richness)
            try:
                vision_cv = self._transform_from_vision(pdf_path)
                # Reconstruire raw_text et word_count depuis les champs structurés
                enriched_raw = self._build_raw_text_from_normalized(vision_cv)
                vision_cv = vision_cv.model_copy(update={
                    "raw_text": enriched_raw,
                    "word_count": len(enriched_raw.split()),
                    "layout_detected": real_layout,
                })
                vision_richness = _vision_richness_score(vision_cv)
                winner = "vision" if vision_richness > best_richness else best_cv.extraction_method
                logger.info("cascade | vision richness: %.0f → retenu: %s", vision_richness, winner)
                if vision_richness > best_richness:
                    return vision_cv
            except Exception as exc:
                logger.warning("cascade | Vision LLM échoué : %s", exc)
        else:
            logger.info(
                "cascade | niveau 3 ignoré (clé=%s, vision_llm_enabled=%s, allow_vision=%s)",
                bool(settings.anthropic_api_key), settings.vision_llm_enabled, allow_vision,
            )

        return best_cv

    def _build_raw_text_from_normalized(self, cv: NormalizedCV) -> str:
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

    # ------------------------------------------------------------------
    # Step 0 — Extraction quality assessment
    # ------------------------------------------------------------------

    @staticmethod
    def _extraction_quality(words: list[dict]) -> str:
        n = len(words)
        if n >= 150:
            return "good"
        if n >= 50:
            return "partial"
        return "failed"

    # ------------------------------------------------------------------
    # Step 0b — OCR extraction (pytesseract + pdf2image, lazy import)
    # ------------------------------------------------------------------

    def _extract_text_ocr(self, pdf_path: str) -> str:
        """Convertir le PDF en images et extraire le texte via Tesseract.

        Dépendances systèmes requises :
            sudo apt-get install tesseract-ocr tesseract-ocr-fra poppler-utils
        Dépendances Python (requirements-ocr.txt ou pip install) :
            pytesseract>=0.3.10  pdf2image>=1.17.0  Pillow>=10.0.0
        """
        try:
            from pdf2image import convert_from_path  # type: ignore[import]
            import pytesseract  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "OCR non disponible — installer les dépendances : "
                "pip install pytesseract pdf2image Pillow  et  "
                "apt-get install tesseract-ocr tesseract-ocr-fra poppler-utils"
            ) from exc

        images = convert_from_path(pdf_path, dpi=300)
        if not images:
            return ""

        parts: list[str] = []
        for img in images:
            page_text = pytesseract.image_to_string(
                img, lang="fra+eng", config="--psm 1 --oem 1"
            )
            parts.append(page_text)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Step 0c — Vision LLM extraction (Claude API)
    # ------------------------------------------------------------------

    def _transform_from_vision(self, pdf_path: str) -> NormalizedCV:
        """Extraire le NormalizedCV via Claude Vision (image du PDF).

        Nécessite ANTHROPIC_API_KEY dans .env.
        Coût estimé : ~0.01 $ / CV (claude-sonnet-4-6, 1 image).
        """
        import base64
        import io
        import json

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

        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=settings.anthropic_api_key)

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

    # ------------------------------------------------------------------
    # Core pipeline — pdfplumber words → NormalizedCV
    # ------------------------------------------------------------------

    def _transform_from_words(
        self,
        words_per_page: list[list[dict]],
        extraction_method: str,
        extraction_confidence: float,
    ) -> NormalizedCV:
        if not words_per_page or not any(words_per_page):
            return NormalizedCV(
                extraction_method=extraction_method,
                extraction_confidence=0.0,
            )

        layout_info = self._detect_layout(words_per_page)
        lines = self._build_annotated_lines(words_per_page, layout_info)

        if not lines:
            return NormalizedCV(
                extraction_method=extraction_method,
                extraction_confidence=0.0,
            )

        raw_text = _lines_to_raw_text(lines)
        body_size = _estimate_body_size(lines)
        section_map = self._split_into_sections(lines, body_size)

        # On two-column CVs, the left-column-first linearization can push
        # right-column header fragments (name, title) into non-header sections
        # because those right-column lines are emitted after the left column's
        # section headers. Instead of modifying section_map (which would
        # corrupt experience parsing), we collect the top-of-page lines from
        # ALL sections and pass them to _parse_header alongside the header
        # section lines. "Top of page" = lines whose top is below the 12th
        # percentile of all line tops — empirically covers the name/title area.
        if layout_info.layout_type == "two_columns" and lines:
            sorted_tops = sorted(l.top for l in lines)
            top_threshold = sorted_tops[max(0, int(len(sorted_tops) * 0.12))]
            header_lines_for_parsing = list(section_map.get("header", []))
            seen_ids = {id(l) for l in header_lines_for_parsing}
            for l in lines:
                if l.top <= top_threshold and id(l) not in seen_ids:
                    header_lines_for_parsing.append(l)
                    seen_ids.add(id(l))
            header_lines_for_parsing.sort(key=lambda l: l.top)
        else:
            header_lines_for_parsing = section_map.get("header", [])

        header = self._parse_header(header_lines_for_parsing)
        if not header.location:
            header = header.model_copy(update={
                "location": _extract_location_from_text(raw_text),
                "postal_code": _extract_postal_code_from_text(raw_text),
            })
        summary = _lines_to_raw_text(section_map.get("summary", [])).strip() or None

        if not summary:
            _hdr_lines = section_map.get("header", [])
            _accroche = [
                l for l in _hdr_lines
                if len(l.text.strip()) > 60
                and len(l.text.split()) > 8
                and not _EMAIL_RE.search(l.text)
                and not _PHONE_RE.search(l.text)
                and not _GITHUB_RE.search(l.text)
                and not _LINKEDIN_RE.search(l.text)
                and not _URL_RE.search(l.text)
                and not POSTAL_CODE_CITY_RE.search(l.text)
            ]
            summary = _lines_to_raw_text(_accroche).strip() or None

        skills = self._parse_skills(raw_text)
        experience = self._parse_experience(section_map.get("experience", []))
        education = self._parse_education(section_map.get("education", []))
        projects = self._parse_projects(section_map.get("projects", []))
        languages = self._parse_language_list(section_map.get("languages", []))
        word_count = len(raw_text.split())

        return NormalizedCV(
            header=header,
            summary=summary,
            skills=skills,
            experience=experience,
            education=education,
            projects=projects,
            languages=languages,
            raw_text=raw_text,
            layout_detected=layout_info.layout_type,
            word_count=word_count,
            extraction_method=extraction_method,
            extraction_confidence=extraction_confidence,
        )

    # ------------------------------------------------------------------
    # Degraded pipeline — plain text (OCR) → NormalizedCV
    # ------------------------------------------------------------------

    def _transform_from_text(
        self,
        text: str,
        extraction_method: str,
        extraction_confidence: float,
    ) -> NormalizedCV:
        """Construire un NormalizedCV depuis un texte brut (sans info positionnelle).

        Utilisé pour le chemin OCR : crée des _Line fictifs (taille/gras uniformes)
        afin de réutiliser toute la logique de détection de sections et de parsing
        existante.  La détection d'en-tête repose uniquement sur les patterns texte
        (majuscules, deux-points, Title Case) sans signal typographique.
        """
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        raw_lines = text.split("\n")
        lines: list[_Line] = [
            _Line(
                text=raw_line,
                max_size=12.0,
                is_bold=False,
                x0=0.0,
                top=float(i * 15),
                column="full",
            )
            for i, raw_line in enumerate(raw_lines)
        ]

        raw_text = _lines_to_raw_text(lines)
        section_map = self._split_into_sections(lines, body_size=12.0)

        header = self._parse_header(section_map.get("header", []))
        if not header.location:
            header = header.model_copy(update={
                "location": _extract_location_from_text(raw_text),
                "postal_code": _extract_postal_code_from_text(raw_text),
            })

        summary = _lines_to_raw_text(section_map.get("summary", [])).strip() or None
        if not summary:
            _hdr = [
                l for l in section_map.get("header", [])
                if len(l.text.strip()) > 60
                and len(l.text.split()) > 8
                and not _EMAIL_RE.search(l.text)
                and not _PHONE_RE.search(l.text)
                and not _GITHUB_RE.search(l.text)
                and not _LINKEDIN_RE.search(l.text)
            ]
            summary = _lines_to_raw_text(_hdr).strip() or None

        skills = self._parse_skills(raw_text)
        experience = self._parse_experience(section_map.get("experience", []))
        education = self._parse_education(section_map.get("education", []))
        projects = self._parse_projects(section_map.get("projects", []))
        languages = self._parse_language_list(section_map.get("languages", []))
        word_count = len(raw_text.split())

        return NormalizedCV(
            header=header,
            summary=summary,
            skills=skills,
            experience=experience,
            education=education,
            projects=projects,
            languages=languages,
            raw_text=raw_text,
            layout_detected="single_column",
            word_count=word_count,
            extraction_method=extraction_method,
            extraction_confidence=extraction_confidence,
        )

    # ------------------------------------------------------------------
    # Step 1 — Extract words with positional + style attributes
    # ------------------------------------------------------------------

    def _extract_words(self, pdf_path: str) -> list[list[dict]]:
        pages: list[list[dict]] = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    try:
                        words = page.extract_words(
                            x_tolerance=8,   # >3 pour grouper les lettres espacées (artefact PDF letter-spacing)
                            y_tolerance=3,
                            extra_attrs=["size", "fontname"],
                        )
                        pages.append(words or [])
                    except Exception:
                        pages.append([])
        except Exception:
            return []
        return pages

    # ------------------------------------------------------------------
    # Step 2 — Detect layout (single vs two columns)
    # ------------------------------------------------------------------

    def _detect_layout(self, words_per_page: list[list[dict]]) -> _LayoutInfo:
        all_words = [w for page in words_per_page for w in page]
        if not all_words:
            return _LayoutInfo("single_column")

        page_width = max((w.get("x1") or w["x0"]) for w in all_words)
        if page_width <= 0:
            return _LayoutInfo("single_column")

        # Exclure le bandeau d'en-tête (≈ 20 % supérieurs de la page) pour éviter
        # que les wraps de l'accroche (qui s'étendent loin vers la droite dans la
        # colonne gauche) ne biaisent le calcul du gutter.
        page_height = max((w.get("bottom") or w["top"] + 12) for w in all_words)
        body_y_start = 0.20 * page_height

        # Utiliser les x0 de TOUS les mots du corps (pas seulement les débuts de
        # ligne) pour avoir une distribution fidèle de la répartition horizontale.
        body_x0s: list[float] = [
            w["x0"]
            for page in words_per_page
            for w in page
            if w["top"] > body_y_start
        ]

        if len(body_x0s) < 5:
            return _LayoutInfo("single_column")

        # Chercher le plus grand gap dans la zone centrale [15 %, 85 %] de la page.
        lo, hi = 0.15 * page_width, 0.85 * page_width
        central_starts = sorted(set(round(x, 1) for x in body_x0s if lo <= x <= hi))
        if len(central_starts) < 2:
            return _LayoutInfo("single_column")

        gaps = [
            (central_starts[i + 1] - central_starts[i], central_starts[i], central_starts[i + 1])
            for i in range(len(central_starts) - 1)
        ]
        max_gap, gap_lo, gap_hi = max(gaps, key=lambda g: g[0])

        if max_gap < 25:
            return _LayoutInfo("single_column")

        # Utiliser la moyenne du gap.  Les wraps larges dans la zone d'en-tête
        # (top < body_y_start) ne sont pas divisés par ce gutter — ils sont
        # émis en ordre vertical avant les colonnes du corps (voir
        # _build_annotated_lines), ce qui évite qu'ils contaminent les sections.
        gutter_x = (gap_lo + gap_hi) / 2
        left_count = sum(1 for x in body_x0s if x < gutter_x)
        right_count = sum(1 for x in body_x0s if x >= gutter_x)
        total = len(body_x0s)

        if left_count / total >= 0.10 and right_count / total >= 0.10:
            return _LayoutInfo("two_columns", gutter_x=gutter_x, body_y_start=body_y_start)
        return _LayoutInfo("single_column")

    # ------------------------------------------------------------------
    # Step 3 — Build annotated lines with column info
    # ------------------------------------------------------------------

    def _build_annotated_lines(
        self, words_per_page: list[list[dict]], layout: _LayoutInfo
    ) -> list[_Line]:
        lines: list[_Line] = []
        for page_words in words_per_page:
            if not page_words:
                continue
            raw_groups = _group_words_into_raw_lines(page_words)

            if layout.layout_type == "two_columns" and layout.gutter_x:
                gutter = layout.gutter_x
                body_y = layout.body_y_start
                # Les lignes dans la zone d'en-tête (bandeau coloré, top < body_y)
                # sont émises sans division par le gutter — leur texte peut
                # s'étendre sur toute la largeur de la colonne gauche.  Elles
                # apparaissent avant les lignes du corps dans l'ordre vertical,
                # ce qui garantit qu'elles tombent dans la section "header" et
                # non dans les sections "experience" ou "education".
                header_groups: list[tuple[float, list[dict]]] = []
                left_lines: list[tuple[float, list[dict]]] = []
                right_lines: list[tuple[float, list[dict]]] = []
                for top, words in raw_groups:
                    if top < body_y:
                        header_groups.append((top, words))
                    else:
                        lw = sorted([w for w in words if w["x0"] < gutter], key=lambda w: w["x0"])
                        rw = sorted([w for w in words if w["x0"] >= gutter], key=lambda w: w["x0"])
                        if lw:
                            left_lines.append((top, lw))
                        if rw:
                            right_lines.append((top, rw))
                for top, hw in sorted(header_groups, key=lambda x: x[0]):
                    lines.append(_words_to_line(hw, "full"))
                for top, lw in sorted(left_lines, key=lambda x: x[0]):
                    lines.append(_words_to_line(lw, "left"))
                for top, rw in sorted(right_lines, key=lambda x: x[0]):
                    lines.append(_words_to_line(rw, "right"))
            else:
                for top, words in raw_groups:
                    lines.append(_words_to_line(words, "full"))
        return lines

    # ------------------------------------------------------------------
    # Step 4 — Detect section boundaries by typography + text patterns
    # ------------------------------------------------------------------

    def _split_into_sections(
        self, lines: list[_Line], body_size: float
    ) -> dict[str, list[_Line]]:
        buckets: dict[str, list[_Line]] = {"header": []}
        current = "header"

        for line in lines:
            if not line.text.strip():
                buckets.setdefault(current, []).append(line)
                continue
            if _is_section_header(line, body_size):
                matched = _match_section_name(line.text)
                if matched:
                    current = matched
                    buckets.setdefault(current, [])
                    continue
            buckets.setdefault(current, []).append(line)

        return {k: v for k, v in buckets.items() if any(l.text.strip() for l in v)}

    # ------------------------------------------------------------------
    # Step 5 — Parse each section into structured objects
    # ------------------------------------------------------------------

    def _parse_header(self, lines: list[_Line]) -> CVHeader:
        full_text = " ".join(l.text for l in lines)
        all_text = "\n".join(l.text for l in lines)

        email = _first_match(_EMAIL_RE, all_text)
        phone = _first_match(_PHONE_RE, all_text)
        github_m = _GITHUB_RE.search(all_text)
        github = github_m.group(0) if github_m else None
        linkedin_m = _LINKEDIN_RE.search(all_text)
        linkedin = linkedin_m.group(0) if linkedin_m else None

        location = _extract_location_from_text(all_text)
        postal_code = _extract_postal_code_from_text(all_text)

        # Name: largest-size non-empty line (typically the very first big text)
        name_line = max(
            (l for l in lines if l.text.strip() and not _EMAIL_RE.search(l.text) and not _PHONE_RE.search(l.text)),
            key=lambda l: l.max_size,
            default=None,
        )
        name = name_line.text.strip() if name_line else None

        # On two-column CVs, the name can be split across columns (e.g. "Keo"
        # left / "PEN" right) when the font uses letter-spacing that forces
        # pdfplumber to emit individual character tokens per column.
        # Re-join if a line in the opposite column sits at the same top (±15px).
        if name_line and name_line.column in ("left", "right"):
            opp_col = "right" if name_line.column == "left" else "left"
            complement = next(
                (
                    l for l in lines
                    if l.column == opp_col
                    and abs(l.top - name_line.top) <= 15
                    and l.text.strip()
                    and not _EMAIL_RE.search(l.text)
                    and not _PHONE_RE.search(l.text)
                    and not _GITHUB_RE.search(l.text)
                    and not _LINKEDIN_RE.search(l.text)
                ),
                None,
            )
            if complement:
                left_part = name if name_line.column == "left" else complement.text.strip()
                right_part = complement.text.strip() if name_line.column == "left" else name
                name = f"{left_part} {right_part}".strip()

        # Title: first non-name line matching a job title keyword
        title = None
        for l in lines:
            t = l.text.strip()
            if not t or t == name:
                continue
            if _EMAIL_RE.search(t) or _PHONE_RE.search(t) or _GITHUB_RE.search(t) or _LINKEDIN_RE.search(t):
                continue
            if JOB_TITLE_RE.search(t):
                for seg in TITLE_SPLIT_RE.split(t):
                    seg = seg.strip()
                    if seg and JOB_TITLE_RE.search(seg):
                        title = seg
                        break
            if title:
                break

        return CVHeader(
            name=name,
            title=title,
            email=email,
            phone=phone,
            location=location,
            postal_code=postal_code,
            github=github,
            linkedin=linkedin,
        )

    def _parse_skills(
        self, raw_text: str, semantic_threshold: Optional[float] = None
    ) -> CVSkills:
        text_lower = raw_text.lower()
        result: dict[str, list[str]] = {cat: [] for cat in SKILL_CATEGORIES}
        for cat, skill_list in SKILL_CATEGORIES.items():
            for skill in skill_list:
                if re.search(r"\b" + re.escape(skill.lower()) + r"\b", text_lower):
                    result[cat].append(skill)

        # ESCO multi-word phrases (e.g. "produce sales reports") rarely appear
        # verbatim — matched semantically instead of by substring.
        existing = {s.lower() for skills in result.values() for s in skills}
        kwargs = {} if semantic_threshold is None else {"threshold": semantic_threshold}
        for phrase in match_semantic_skills(raw_text, **kwargs):
            if phrase.lower() not in existing:
                result.setdefault("other", []).append(phrase)
                existing.add(phrase.lower())

        return CVSkills(**result)

    def _parse_experience(self, lines: list[_Line]) -> list[CVExperience]:
        return _parse_timed_entries(lines, is_education=False)

    def _parse_education(self, lines: list[_Line]) -> list[CVEducation]:
        raw_entries = _parse_timed_entries(lines, is_education=True)
        edu_list: list[CVEducation] = []
        for exp in raw_entries:
            degree = exp.title or ""
            school = exp.company or ""
            # Nettoyer les labels "Role:" résidus dans school (ex : "Data Engineer:")
            if school.rstrip().endswith(":"):
                school = ""
            # Rejeter les entrées sans date dont degree et school sont des labels de rôle
            # (ex : "Monitoring :", "CLOUD") — captées par _parse_skills() via raw_text.
            if not exp.date_start and not exp.duration_months:
                if degree.rstrip().endswith(":") or (not school and not degree):
                    continue
                if JOB_TITLE_RE.search(degree.rstrip(":")) and not exp.bullets:
                    # Titre de poste sans date ni bullets → pas une formation
                    continue
            edu_list.append(
                CVEducation(
                    degree=degree,
                    school=school,
                    year=exp.period,
                    date_start=exp.date_start,
                    date_end=exp.date_end,
                    duration_months=exp.duration_months,
                    is_current=exp.is_current,
                    skills=_extract_tech_terms(" ".join(exp.bullets)),
                )
            )
        return edu_list

    def _parse_projects(self, lines: list[_Line]) -> list[CVProject]:
        if not lines:
            return []
        projects: list[CVProject] = []
        entries = _split_by_blank_or_bold(lines)
        for entry_lines in entries:
            if not entry_lines:
                continue
            text = " ".join(l.text for l in entry_lines)
            name = entry_lines[0].text.strip()
            bullets = [
                _BULLET_RE.sub("", l.text).strip()
                for l in entry_lines[1:]
                if _BULLET_RE.match(l.text)
            ]
            url_m = _URL_RE.search(text)
            url = url_m.group(0) if url_m else None
            metrics = [
                l.text.strip()
                for l in entry_lines
                if any(p.search(l.text) for p in METRIC_PATTERNS)
            ]
            projects.append(
                CVProject(
                    name=name,
                    description=" ".join(bullets) if bullets else None,
                    stack=_extract_tech_terms(text),
                    url=url,
                    metrics=metrics,
                )
            )
        return projects

    def _parse_language_list(self, lines: list[_Line]) -> list[str]:
        langs: list[str] = []
        for l in lines:
            # Each non-empty line is a language entry; strip bullet chars
            t = _BULLET_RE.sub("", l.text).strip()
            if t and len(t) < 60:
                langs.append(t)
        return langs


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _group_words_into_raw_lines(words: list[dict]) -> list[tuple[float, list[dict]]]:
    """Group words by y-proximity (±3 px) and sort each group left-to-right."""
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    groups: list[tuple[float, list[dict]]] = []
    current: list[dict] = [sorted_words[0]]
    current_top: float = sorted_words[0]["top"]
    for w in sorted_words[1:]:
        if abs(w["top"] - current_top) <= 3:
            current.append(w)
        else:
            groups.append((current_top, sorted(current, key=lambda w: w["x0"])))
            current = [w]
            current_top = w["top"]
    groups.append((current_top, sorted(current, key=lambda w: w["x0"])))
    return groups


def _words_to_line(words: list[dict], column: str) -> _Line:
    text = " ".join(w["text"] for w in words).strip()
    sizes = [float(w.get("size") or 10.0) for w in words if w.get("size")]
    fontnames = [str(w.get("fontname") or "") for w in words]
    max_size = max(sizes) if sizes else 10.0
    is_bold = any("bold" in fn.lower() for fn in fontnames)
    x0 = min(w["x0"] for w in words)
    top = min(w["top"] for w in words)
    return _Line(text=text, max_size=max_size, is_bold=is_bold, x0=x0, top=top, column=column)


def _lines_to_raw_text(lines: list[_Line]) -> str:
    raw = "\n".join(l.text for l in lines)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def _estimate_body_size(lines: list[_Line]) -> float:
    sizes = [l.max_size for l in lines if l.text.strip() and l.max_size > 0]
    if not sizes:
        return 10.0
    # Mode: most frequent size
    from collections import Counter
    rounded = [round(s, 1) for s in sizes]
    return Counter(rounded).most_common(1)[0][0]


def _is_section_header(line: _Line, body_size: float) -> bool:
    t = line.text.strip()
    if not t or len(t) > 80:
        return False
    # Typography signal
    typo = line.max_size > body_size * 1.08 or (line.is_bold and line.max_size >= body_size * 0.95)
    # Text pattern signal (all-caps, ends with colon, or Title Case short line)
    text_pattern = (
        t.isupper()
        or t.endswith(":")
        or bool(re.match(r"^[A-ZÀ-Ý][A-Za-zÀ-ÿ\s&/éèêàûùâôî]+$", t))
    )
    return typo or text_pattern


def _match_section_name(text: str) -> Optional[str]:
    normalized = text.lower().rstrip(":").strip()
    for section, pattern in SECTION_HEADERS.items():
        if re.match(pattern, normalized, re.IGNORECASE):
            return section
    return None


def _first_match(pattern: re.Pattern, text: str) -> Optional[str]:
    m = pattern.search(text)
    return m.group(0).strip() if m else None


def _extract_location_from_text(text: str) -> Optional[str]:
    """Retourne la ville si une adresse postale FR est détectée (code postal + ville)."""
    m = POSTAL_CODE_CITY_RE.search(text)
    if m:
        return m.group(2).strip().title()
    return None


def _extract_postal_code_from_text(text: str) -> Optional[str]:
    """Retourne le code postal à 5 chiffres si une adresse postale FR est détectée."""
    m = POSTAL_CODE_CITY_RE.search(text)
    if m:
        return m.group(1)
    return None


def _extract_tech_terms(text: str) -> list[str]:
    """Return known skill terms found in text (flat list)."""
    text_lower = text.lower()
    found = []
    for skill_list in SKILL_CATEGORIES.values():
        for skill in skill_list:
            if re.search(r"\b" + re.escape(skill.lower()) + r"\b", text_lower):
                found.append(skill)
    return found


# ---------------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------------


def _parse_date_str(raw: str) -> Optional[tuple[int, int]]:
    """Return (year, month) from a raw date string.

    Month is 0 when only a year is detected (sentinel for "year-only").
    Month defaults to 0 for year-only dates so callers can preserve the
    original format in date_start/date_end strings.
    """
    raw = raw.strip()
    if _CURRENT_RE.match(raw):
        return None  # caller handles "present"
    # Month + year
    m = _MONTH_YEAR_RE.match(raw)
    if m:
        month_str = m.group(1)[:4].lower()
        month_str = month_str.replace("é", "e").replace("û", "u").replace("è", "e")
        month = _MONTH_MAP.get(month_str[:3], _MONTH_MAP.get(month_str[:4], 1))
        year = int(m.group(2))
        return (year, month)
    # Year only
    m = _YEAR_ONLY_RE.search(raw)
    if m:
        return (int(m.group(0)), 0)
    return None


def _months_between(start: tuple[int, int], end: tuple[int, int]) -> int:
    s_month = start[1] or 1
    e_month = end[1] or 1
    return max(0, (end[0] - start[0]) * 12 + (e_month - s_month))


def _date_tuple_to_str(t: tuple[int, int]) -> str:
    if t[1] == 0:
        return str(t[0])
    return f"{t[0]}-{t[1]:02d}"


def extract_date_range(text: str) -> tuple[Optional[str], Optional[str], Optional[int], bool]:
    """Return (date_start, date_end, duration_months, is_current)."""
    # Explicit duration ("3 mois", "2 ans")
    dur_m = _EXPLICIT_DURATION_RE.search(text)
    if dur_m:
        months_val = int(dur_m.group(1) or 0)
        years_val = float(dur_m.group(2) or 0)
        duration = months_val or int(years_val * 12)
        if duration > 0:
            return (None, None, duration, False)

    # Date range
    for m in _DATE_RANGE_RE.finditer(text):
        raw_start, raw_end = m.group(1), m.group(2)
        is_current = bool(_CURRENT_RE.search(raw_end))
        start = _parse_date_str(raw_start)
        if start is None:
            continue
        if is_current:
            today = date.today()
            end_tuple = (today.year, today.month)
            duration = _months_between(start, end_tuple)
            return (_date_tuple_to_str(start), "present", duration, True)
        end = _parse_date_str(raw_end)
        if end is None:
            continue
        duration = _months_between(start, end)
        return (_date_tuple_to_str(start), _date_tuple_to_str(end), duration, False)

    # Single year (e.g. "2023" on its own line)
    year_matches = [m.group(0) for m in _YEAR_ONLY_RE.finditer(text)]
    if len(year_matches) == 1:
        return (year_matches[0], None, None, False)
    if len(year_matches) >= 2:
        y_start = int(year_matches[0])
        y_end = int(year_matches[-1])
        duration = (y_end - y_start) * 12
        return (str(y_start), str(y_end), duration if duration > 0 else None, False)

    return (None, None, None, False)


# ---------------------------------------------------------------------------
# Entry parsing (experience / education)
# ---------------------------------------------------------------------------


def _is_bullet(text: str) -> bool:
    return bool(_BULLET_RE.match(text)) or (text.startswith("-") and len(text) > 2)


def _contains_date(text: str) -> bool:
    return bool(_DATE_RANGE_RE.search(text) or _MONTH_YEAR_RE.search(text) or _YEAR_ONLY_RE.search(text))


def _split_by_blank_or_bold(lines: list[_Line]) -> list[list[_Line]]:
    """Split a list of lines into chunks separated by blank lines or bold lines."""
    chunks: list[list[_Line]] = []
    current: list[_Line] = []
    for l in lines:
        if not l.text.strip():
            if current:
                chunks.append(current)
                current = []
        else:
            current.append(l)
    if current:
        chunks.append(current)
    return chunks


def _parse_timed_entries(lines: list[_Line], is_education: bool = False) -> list[CVExperience]:
    """Parse experience or education lines into a list of CVExperience entries."""
    if not lines:
        return []

    from collections import Counter

    # Taille de corps locale : mode des tailles non nulles de toutes les lignes.
    # Utilisée pour distinguer les vrais en-têtes typographiques (taille > 1.2×corps)
    # des noms d'entreprises bold mais de taille normale.
    _sizes = [round(l.max_size, 1) for l in lines if l.text.strip() and l.max_size > 0]
    _body_size = Counter(_sizes).most_common(1)[0][0] if _sizes else 10.0
    _header_size_threshold = _body_size * 1.2

    entries: list[CVExperience] = []
    entry_lines: list[_Line] = []

    def flush() -> None:
        if not entry_lines:
            return
        entry = _build_entry(entry_lines)
        # Filtrer les blocs parasites : prénom/nom isolé, fragment sans date ni
        # bullets (ex : "PEN", "professionnelles", ligne unique d'outil).
        total_words = sum(len(l.text.split()) for l in entry_lines)
        if total_words < 3 and not entry.date_start and not entry.duration_months and not entry.bullets:
            entry_lines.clear()
            return
        if entry.title or entry.company or entry.period or entry.bullets:
            entries.append(entry)
        entry_lines.clear()

    for line in lines:
        t = line.text.strip()
        if not t:
            continue
        # A date-containing non-bullet short line signals a new entry or date boundary
        if _contains_date(t) and not _is_bullet(t) and len(t) < 100:
            # If there's already a date in current entry, start new entry
            current_has_date = any(_contains_date(l.text) for l in entry_lines)
            if current_has_date and entry_lines:
                flush()
            entry_lines.append(line)
        elif (
            line.is_bold
            and not _is_bullet(t)
            and len(t) < 80
            and entry_lines
            and line.max_size >= _header_size_threshold
        ):
            # Ligne typographiquement grande et bold = nouvel en-tête d'entrée.
            # Les noms d'entreprises bold mais de taille normale (≈ body_size)
            # ne déclenchent plus de flush — ils seront rattachés à l'entrée en cours.
            flush()
            entry_lines.append(line)
        else:
            entry_lines.append(line)

    flush()

    # Sort by date_start descending (most recent first)
    def sort_key(e: CVExperience) -> tuple:
        if e.date_start:
            parts = e.date_start.split("-")
            return (-int(parts[0]), -int(parts[1]) if len(parts) > 1 else 0)
        return (0, 0)

    entries.sort(key=sort_key)
    return entries


def _build_entry(lines: list[_Line]) -> CVExperience:
    """Build a CVExperience from a list of raw lines belonging to one entry."""
    title: Optional[str] = None
    company: Optional[str] = None
    period: Optional[str] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    duration_months: Optional[int] = None
    is_current = False
    bullets: list[str] = []

    for line in lines:
        t = line.text.strip()
        if not t:
            continue

        if _is_bullet(t):
            bullets.append(_BULLET_RE.sub("", t).strip())
            continue

        if _contains_date(t):
            ds, de, dm, ic = extract_date_range(t)
            if ds or dm:
                period = t
                date_start = ds
                date_end = de
                duration_months = dm
                is_current = ic
            continue

        # Non-bullet, non-date short line: try to extract title / company
        if JOB_TITLE_RE.search(t):
            if title is None:
                # Could be "Title | Company" or "Title - Company"
                parts = TITLE_SPLIT_RE.split(t)
                title = parts[0].strip()
                if len(parts) > 1:
                    company = parts[1].strip()
            elif company is None:
                company = t
        elif company is None and len(t) < 80:
            company = t

    years = round(duration_months / 12, 1) if duration_months else None
    return CVExperience(
        title=title,
        company=company,
        period=period,
        date_start=date_start,
        date_end=date_end,
        duration_months=duration_months,
        is_current=is_current,
        years=years,
        bullets=bullets,
    )
