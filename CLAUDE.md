# ATS CV Scorer — Contexte projet

Projet portfolio (parcours ML Engineer DataScientest) : analyse un CV (PDF)
via un pipeline NLP + scoring sémantique + détection sectorielle +
feedback IA Claude. Déployé sur Hugging Face Spaces.

## Stack

| Couche | Tech |
|---|---|
| Extraction PDF | pdfplumber → Tesseract OCR → Vision LLM Claude, cascade 3 niveaux |
| NLP | spaCy `en_core_web_sm` + regex (lexiques FR/EN dans `lexicons.py`) |
| Scoring sémantique | sentence-transformers `all-MiniLM-L6-v2` |
| Détection secteur | MiniLM cosine sim · 134 profils · 9 secteurs (`sector_registry.py`) |
| Scoring ATS | `CVQualityScorer` — critères pondérés par profil sectoriel |
| Feedback IA | Anthropic Claude `claude-sonnet-4-6` · prompt sectoriel · réponse FR |
| Recherche d'offres | Adzuna (`ADZUNA_ID`/`ADZUNA_API_KEY`) · Jooble · France Travail OAuth2 |
| API REST | FastAPI v0.3.0 (`src/api/server.py`) — `/score`, `/find-jobs`, `/health` |
| UI | Gradio 4.44 (`app.py`) — 2 onglets, thème Tech Dashboard custom |

## Structure

```
src/
  core/
    config.py          — Settings (pydantic-settings, lit .env)
    schemas.py         — modèles Pydantic (NormalizedCV, CVQualityReport,
                         ATSResponse, SectorDetectionResult, CriterionResult…)
    lexicons.py        — lexiques FR/EN (skills, secteurs, regex pré-compilées)
                         init_lexicons() idempotent, appelé au démarrage
    sector_profiles.py — dataclass SectorProfile
    sector_registry.py — 134 profils en 9 secteurs + ALL_PROFILES + GENERIC_PROFILE
    budget_guard.py    — quota global appels Claude (fichier JSON persistant)
  services/
    cv_transformer.py      — extraction PDF cascade 3 niveaux, layout-aware
    vision_extractor.py    — niveau 3 Vision LLM
    nlp_pipeline.py        — NLPPipeline.parse_normalized() → NormalizedCV enrichi
    semantic_scorer.py     — SemanticScorer (MiniLM)
    semantic_skill_matcher.py — matching sémantique compétences ESCO
    sector_detector.py     — SectorDetector (MiniLM n-gram, matrix aliases)
    criteria_builder.py    — CriteriaBuilder : instancie les critères par profil
    criteria_evaluator.py  — fonctions d'évaluation binaires (0/100)
    cv_quality_scorer.py   — CVQualityScorer : orchestre critères + ProfileStrength
    claude_feedback.py     — ClaudeFeedback : prompt sectoriel enrichi, budget guard
    lexicon_builder.py     — LexiconBuilder (ESCO) — utilisé off-line, pas en runtime
    job_matcher.py         — find_matching_jobs() : multi-requêtes, dédup, seuil
    job_providers/
      base.py              — ABC JobProvider
      adzuna.py            — AdzunaProvider
      jooble.py            — JoobleProvider
      france_travail.py    — FranceTravailProvider (OAuth2)
      oauth2_token_manager.py — cache token thread-safe
      orchestrator.py      — JobSearchOrchestrator (fan-out parallèle)
  api/
    server.py              — FastAPI : /score (pipeline complet), /find-jobs, /health
  ui/
    pipeline_diagram.py    — schéma pipeline animé (HTML/CSS/SVG) + BRIDGE_JS
                             get_pipeline_html() rendu unique · get_stage_signal() canal JS
app.py                     — UI Gradio (point d'entrée HF Spaces)
tests/
  test_*.py                — un fichier par service, mocks httpx.MockTransport
  benchmark_sectoriel.py   — rapport CSV : détection × scoring par CV du corpus
  fixtures/
    sample_cvs/            — 15 CVs PDF (corpus benchmark)
    benchmark_sectoriel.csv
```

## Conventions

- `make lint` / `make format` couvrent `src/` et `tests/` uniquement — **pas `app.py`** (code legacy non formaté par ruff). `src/ui/` est couvert.
- **Bridge JS pipeline diagram** : injecter du JS persistant via `gr.Blocks(head=f"<script>{JS}</script>")`, **jamais** `gr.Blocks(js=...)` — le paramètre `js=` en Gradio 4.44 passe le code via `new Function()` et brise silencieusement les event handlers (aucun upload ne fonctionne).
- Secrets : `.env` (jamais commité), `.env.example` reste vide. **Ne jamais afficher la valeur d'un secret** (`cut -d'=' -f1` pour vérifier la présence, jamais `cat`/`grep` sur les valeurs).
- pydantic-settings mappe `nom_du_champ` ↔ `NOM_DU_CHAMP` (insensible à la casse, sans préfixe).
- Réponses Claude forcées en français via `_SYSTEM_PROMPT` (`claude_feedback.py`).
- `init_lexicons()` doit être appelé avant tout import de service. Elle est idempotente.
- `NLPPipeline.parse_normalized(cv)` retourne un **nouveau** `NormalizedCV` via `model_copy(update={...})` — l'objet `cv` original n'est pas muté. Toujours utiliser le retour.
- `scorer.score()` doit recevoir le `parsed_cv` (retour de `parse_normalized`), pas le `normalized_cv` brut — `skills_flat` est vide sur le transformer brut.

## Pipeline complet

```
PDF
 → CVTransformer (pdfplumber → OCR → Vision LLM)
 → NormalizedCV (raw_text, header, experience, education, projects, skills)
 → NLPPipeline.parse_normalized()
 → NormalizedCV enrichi (job_title, location, postal_code, skills_flat, sector)
 → SectorDetector.detect()                  → SectorDetectionResult
 → CVQualityScorer.score(cv, sector_result) → CVQualityReport
   (ATSReadability + ProfileStrength + list[CriterionResult] + Recommendations)
 → [UI Onglet 1 : qualité affichée immédiatement]
 → JobSearchOrchestrator.search()           → list[JobListing] (3 sources)
 → SemanticScorer.score_many()              → list[RankedJobMatch]
 → [UI Onglet 2 : offres classées]
 → ClaudeFeedback.generate_feedback(        → str markdown
     cv, job_desc, scoring_result,
     sector_result=..., criteria_results=...)
   [à la demande, par offre]
```

## Détection sectorielle

`score = 0.40 × title_sim + 0.35 × skills_kw + 0.25 × exp_kw`

- `title_sim` : fenêtres n-gram 2–4 mots sur `job_title` bruité → cosine sim max contre matrix d'aliases MiniLM (batch encodé à l'init, lazy).
- `skills_kw` : proportion `detection_keywords` trouvés dans `skills_flat + sections["skills"]`.
- `exp_kw` : proportion trouvés dans bullets experience + `sections["experience"]`.
- Seuil : 0.30. En dessous → `GENERIC_PROFILE` (`non_detecte`).
- `detection_keywords` en FR **et** EN pour tous les profils `informatique_digital` (CVs anglophones).
- Correction manuelle : dropdown UI → `SectorDetector.make_forced_result(profile_id)`.

## Feedback Claude sectoriel

`ClaudeFeedback.generate_feedback()` accepte `sector_result` et `criteria_results` (optionnels, rétrocompatible) :

- Quand fournis : prompt inclut profil détecté + confiance + critères obligatoires KO (label + evidence) + critères recommandés KO. Instructions Claude : 3 parties (POINTS FORTS / POINTS À AMÉLIORER / CONSEIL PRIORITAIRE), max 250 mots.
- Sans `sector_result` : prompt générique original (comportement pré-Phase F).
- `app.py` transmet `sector_result_state` + `quality_report_state.criteria_results` au handler "Analyser cette offre".

## API REST (v0.3.0)

- `/score` : CVTransformer + NLPPipeline + SectorDetector + CVQualityScorer + SemanticScorer + ClaudeFeedback (optionnel). `ATSResponse` inclut `detected_sector`, `detected_profile`, `detection_confidence`, `criteria_results`.
- `/find-jobs` : CVTransformer + NLPPipeline + JobSearchOrchestrator (Adzuna + Jooble) + SemanticScorer.
- `/health` : `claude_budget_remaining`.

## Pitfalls connus

- **`en_core_web_sm` sur texte FR** : NER GPE/LOC produit des faux positifs (noms de langues, noms de rues). Solution : cascade `_cascade_location()` dans `nlp_pipeline.py` : `POSTAL_FIRST_RE` (code-avant-ville) → `POSTAL_CODE_CITY_RE` (ville-avant) → NER filtré (`LOCATION_BLOCKLIST` + `STREET_TOKENS`). Même priorité appliquée dans `cv_transformer._extract_location_from_text()`. **`POSTAL_CODE_CITY_RE` seul ne suffit pas** : il matche "Jean Jaurès, 59170" avant "59170 Croix" — toujours essayer `POSTAL_FIRST_RE` en premier.
- **`gr.Blocks(js=...)` Gradio 4.44** : brise silencieusement les event handlers (aucun upload ne fonctionne). Utiliser `head=f"<script>{JS}</script>"` à la place.
- **Adzuna géocode mal les petites villes** : utiliser le code postal (`"59170 Croix"` au lieu de `"Croix"`). La région manuelle Gradio prend toujours la priorité sur la localisation CV.
- **`job_title` bruité** : le champ peut contenir nom, téléphone, email. Le n-gram windowing (2–4 mots) dans `_compute_title_scores` permet d'extraire "devops engineer" d'une chaîne bruitée.
- **France Travail OAuth2** : l'application francetravail.io doit être **abonnée** à l'API "Offres d'emploi v2" (souscription distincte de la création de l'app). Sinon : `invalid_client` sur le token endpoint même avec des credentials valides.
- **`font_mono` Gradio 4.44** : doit être une liste (`[gr.themes.GoogleFont(...)]`), pas un objet nu.
- **Variables CSS Gradio dark mode** : utiliser `--body-text-color(-subdued)`, `--background-fill-primary/secondary`, `--border-color-primary`. Les noms comme `--color-text-primary` n'existent pas dans cette version.

## État actuel

- **396 tests, 10 skippés** (PDFs privés `CV_kumnito_two_columns.pdf` et `CV-KEO-PEN.pdf` absents du repo).
- **Benchmark sectoriel** : détection ≥ 80% sur le corpus de 15 CVs. `make benchmark-sectoriel` génère `tests/fixtures/benchmark_sectoriel.csv`.
- **Déployé** : HF Spaces `voroman/ats-cv-scorer`, remote `hf`.
- **Pipeline diagram animé** : `src/ui/pipeline_diagram.py` — schéma 7 blocs + connecteurs SVG `stroke-dashoffset` (animation draw 2s/étape). Bridge `MutationObserver` injecté via `head=<script>` pour transitions CSS zero-clignotement. Schéma rendu une seule fois (yield 1), JS mis à jour via `stage_signal_html` caché.

## Dette technique résiduelle

| Dette | Impact | Effort |
|---|---|---|
| Lexiques ESCO snapshot statique | `lexicons_generated.json` + `lexicons_embeddings.npy` versionnés (7 KB + 164 KB) — snapshot à régénérer via `make update-lexicons` quand l'ESCO évolue | Faible — commit + push suffit |
| 10 tests skippés structurellement | Couverture layout 2 colonnes non régressée en CI | Faible — générer des fixtures synthétiques avec reportlab |
| France Travail absent de `/find-jobs` API | Orchestrateur API : Adzuna + Jooble seulement | Faible — ajouter `FranceTravailProvider` à `_orchestrator` dans `server.py` |
