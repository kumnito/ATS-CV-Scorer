# ATS CV Scorer — Contexte projet

Projet portfolio (parcours ML Engineer) : analyse un CV (PDF) face à une offre
d'emploi via un pipeline NLP + scoring sémantique + feedback IA Claude.

## Stack

| Couche | Tech |
|---|---|
| Extraction PDF | pdfplumber |
| NLP | spaCy `en_core_web_sm` |
| Scoring sémantique | sentence-transformers (`all-MiniLM-L6-v2`) |
| Feedback IA | Anthropic Claude (`claude-sonnet-4-6`, prompt caching) |
| Recherche d'offres | Adzuna (`ADZUNA_ID` / `ADZUNA_API_KEY`, pays `fr`) |
| API REST | FastAPI (`src/api/server.py`) |
| UI | Gradio (`app.py`, point d'entrée Hugging Face Spaces) |

## Structure

- `src/core/` — `config.py` (Settings via pydantic-settings, lit `.env`), `schemas.py` (modèles Pydantic)
- `src/services/` — `pdf_extractor`, `nlp_pipeline`, `semantic_scorer`, `claude_feedback`, `job_search` (client Adzuna), `job_matcher` (orchestration recherche+scoring)
- `src/api/server.py` — endpoints `/score` et `/find-jobs`
- `app.py` — UI Gradio unique (HF Spaces)
- `tests/` — un fichier de tests par service, mocks via `httpx.MockTransport` (jamais d'appel réseau réel dans les tests)

## Conventions

- `make lint` / `make format` couvrent `src/` et `tests/` uniquement (pas `app.py` — code legacy non formaté par ruff, ne pas reformater en dehors du scope d'une tâche).
- Secrets : `.env` (jamais commité), `.env.example` reste vide. **Ne jamais afficher la valeur d'un secret** (utiliser `cut -d'=' -f1` pour vérifier la présence d'une variable, jamais `cat`/`grep` sur les valeurs).
- pydantic-settings mappe `nom_du_champ` ↔ `NOM_DU_CHAMP` (insensible à la casse, sans préfixe).
- Réponses Claude forcées en français via le system prompt (`_SYSTEM_PROMPT` dans `claude_feedback.py`), quelle que soit la langue du CV/de l'offre.

## Travaux réalisés (session du 2026-06-08)

### 1. Inversion du flux : "CV → recherche automatique d'offres → scoring"
Nouveau mode unique dans `app.py` : upload du CV seul → extraction
titre de poste/localisation/compétences (NLP) → recherche d'offres via
**Adzuna** → scoring sémantique → classement → feedback Claude **à la
demande** (bouton « Analyser cette offre » par offre, aucun coût IA tant
que l'utilisateur ne clique pas).

Nouveaux fichiers : `src/services/job_search.py` (`JobSearchService`,
client Adzuna via `httpx`), `src/services/job_matcher.py`
(`find_matching_jobs`, `_build_query`). Nouveaux schémas `JobListing` /
`RankedJobMatch` + champs `ParsedCV.job_title` / `ParsedCV.location`.
Endpoint `/find-jobs` ajouté à l'API FastAPI. L'ancien onglet "Score CV ↔
Offre" et le champ "Clé API Claude" ont été supprimés à la demande de
l'utilisateur (UI simplifiée à un seul mode).

### 2. Correction de l'extraction de localisation (3 itérations)
`en_core_web_sm` est un modèle **anglais** : appliqué à des CV
français, son NER GPE/LOC produit des faux positifs en cascade
("opérationnelle", "DÉVELOPPEUR" détectés comme lieux). Solution finale
dans `_extract_location` (`nlp_pipeline.py`) — stratégie en cascade :
1. **Adresse postale FR** (`POSTAL_CODE_CITY_RE`) : code postal à 5
   chiffres + ville capitalisée, sur la même ligne — signal quasi-fiable
   pour les CV français qui incluent l'adresse complète.
2. **Fallback NER filtré** (`PROPER_NOUN_RUN_RE`) : isole la plus longue
   séquence de mots à initiale majuscule au sein de chaque entité GPE/LOC
   bruitée (filtre les adjectifs/mots-clés de poste mal classés).

### 3. Recherche d'offres géo-scopée (Adzuna géocode mal les petites villes)
"Croix" (commune de la métropole lilloise) ne géocode pas du tout chez
Adzuna (0 résultat quel que soit le rayon), d'où un repli initial
"national" qui remontait des offres hors-sujet (Paris). Solution dans
`find_matching_jobs` (`job_matcher.py`) :
1. Recherche `where=<location CV>` + `distance=30` (km) — précision locale.
2. Si vide **et** région choisie par l'utilisateur : repli sur cette
   région (`FRANCE_REGIONS`, 13 régions métropolitaines, menu déroulant
   Gradio `region_dropdown`). **Plus de repli national aveugle.**

`JobSearchService.search` accepte désormais un paramètre `distance`
(transmis à Adzuna uniquement avec une `location`).

### 4. Feedback Claude forcé en français
Ajout au `_SYSTEM_PROMPT` de `claude_feedback.py` :
"Always respond in French, regardless of the language of the CV or job
description." — vérifié en direct (CV/offre en anglais → analyse rendue
intégralement en français).

### 5. Indicateur de chargement pendant l'analyse Claude
`_make_analyze_handler` converti en générateur (pattern Gradio standard
pour les mises à jour progressives) : yield immédiat d'un message
"⏳ Analyse en cours…" (visible), puis yield du feedback final une fois
l'appel Claude terminé.

### 6. Divers
- Bouton "🔎 Analyser cette offre avec Claude" → "🔎 Analyser cette offre".
- Message "aucune offre trouvée" reformulé sans détails techniques internes
  (`ADZUNA_ID`/`.env`) — orienté utilisateur final.

## Tests ajoutés cette session

`tests/test_job_search.py` (6), `tests/test_job_matcher.py` (8),
+ tests d'extraction titre/localisation dans `tests/test_nlp_pipeline.py`
(10, dont régressions sur les faux positifs NER et l'adresse postale).
Total : 34 tests, tous verts (`make test`). Aucun appel réseau réel dans
la suite (mocks `httpx.MockTransport`).
