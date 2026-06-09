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

---

## Travaux réalisés (session du 2026-06-08 — session 2)

### 1. Nouveau transformer layout-aware (`src/services/cv_transformer.py`)
`CVTransformer.transform(pdf_path)` → `NormalizedCV`. Lit les positions
des caractères pdfplumber (`x0`, `top`, `size`, `fontname`) pour :
- détecter la **mise en page** (1 colonne / 2 colonnes) via analyse des
  gaps en x0 dans la zone centrale de la page ;
- reconstruire l'**ordre de lecture** correct (colonne gauche en entier,
  puis colonne droite) avant de parser les sections ;
- extraire un `CVHeader` structuré (nom, titre, email, téléphone,
  localisation, github, linkedin) via regex dédiées ;
- parser expériences, formations, projets et compétences en objets
  Pydantic (`CVExperience`, `CVEducation`, `CVProject`, `CVSkills`).

Règle de localisation **stricte** dans le transformer : adresse postale
FR uniquement (`POSTAL_CODE_CITY_RE`) — pas de fallback NER, pour éviter
les faux positifs sur CV français. La région manuelle Gradio prend le
relais dans `job_matcher.py`.

### 2. Nouveau scorer qualité CV (`src/services/cv_quality_scorer.py`)
`CVQualityScorer.score(cv: NormalizedCV)` → `CVQualityReport` :
- **score_structure** (0-100) : sections obligatoires, layout, complétude
  de l'en-tête, ordre sections (skills avant education) ;
- **score_contenu** (0-100) : accroche, projets, métriques quantifiées,
  densité mots-clés (cible 15-25 %), longueur (cible 500-750 mots),
  verbes d'action ;
- **score_global** = 40 % structure + 60 % contenu ;
- **statistiques carrière** : années d'expérience totales, année de début,
  dernier poste, années de formation, détection de trous > 12 mois ;
- **recommandations** ordonnées par impact ATS décroissant.

Intégré dans `app.py` (`_format_quality_report`, `_format_timeline`) :
l'analyse qualité est affichée à chaque upload de CV, avant même la
recherche d'offres.

### 3. Nouveau fichier de lexiques (`src/core/lexicons.py`)
Extrait de `nlp_pipeline.py` et enrichi : `SKILL_CATEGORIES` (dict
catégorisé ml/mlops/cloud/languages/data/other), `ALL_SKILLS` (liste
plate pour backward-compat), `ACTION_VERBS_EN/FR`, `METRIC_PATTERNS`,
`JOB_TITLE_RE`, `PERSON_NAME_RE`, `PROPER_NOUN_RUN_RE`,
`POSTAL_CODE_CITY_RE`, `TITLE_SPLIT_RE`, `YEAR_RANGE_RE`. Importé par
`cv_transformer.py`, `cv_quality_scorer.py` et `nlp_pipeline.py`.

### 4. Généralisation du scorer — suppression de la logique GitHub
À la demande de l'utilisateur (outil généraliste, pas dédié à un poste) :
- `has_github` retiré de `CVQualityReport` (schéma Pydantic) ;
- bonus +10 pts contenu supprimé dans `_score_contenu` ;
- recommandation "Ajouter un lien GitHub" supprimée de
  `_build_recommendations` ;
- badge "✅ GitHub / ❌ GitHub absent" retiré de `app.py`.
`CVHeader.github` (extraction URL) conservé — métadonnée utile sans
impacter le score. `"github"` et `"github actions"` restent dans les
lexiques de compétences.

## Tests ajoutés (session 2)

`tests/test_cv_quality_scorer.py` et `tests/test_cv_transformer.py`
(nouveaux fichiers). Total suite : **77 tests, tous verts** (`make test`).
2 tests `has_github` retirés lors de la généralisation du scorer.

---

## Travaux réalisés (session du 2026-06-09)

### Refonte UI — séparation en deux onglets Gradio

`app.py` restructuré : la fonction `search_jobs()` monolithique est
remplacée par deux handlers distincts câblés sur des événements Gradio
séparés.

**Onglet 1 — "📋 Analyse du CV"**
- Déclenché par `cv_input.upload` → `on_cv_upload(cv_file)`
- Pipeline : `CVTransformer` → `NLPPipeline` → `CVQualityScorer`
- Affiche immédiatement : rapport qualité, timeline carrière, profil
  détecté (nom, titre, localisation, top 6 compétences)
- Utilisable sans clé Adzuna (aucun appel réseau externe)
- `cv_input.clear` → `on_cv_clear()` : remet les états à None et désactive
  le bouton de recherche

**Onglet 2 — "🔍 Recherche d'offres"**
- Bouton "Rechercher" démarre `interactive=False` ; activé par
  `on_cv_upload` dès qu'un CV est traité avec succès
- Bandeau `cv_context_md` : profil condensé (titre, localisation, score
  global) mis à jour à chaque upload — donne le contexte sans forcer
  l'aller-retour vers l'onglet 1
- Déclenché par `search_btn.click` → `on_search(parsed_cv_state, region)`
- Pipeline : `find_matching_jobs` + scoring sémantique (lit `parsed_cv_state`
  déjà calculé — pas de re-traitement du CV)
- Relancer la recherche sur une autre région ne recalcule pas le CV

**Nouveau `gr.State`**
- `quality_report_state` ajouté (stocke `CVQualityReport`) pour usage
  futur éventuel dans l'onglet 2

**Aucun changement** dans les services, schémas ou tests — la séparation
est entièrement dans `app.py`.

---

### Corrections pipeline — CV-KEO-PEN.pdf (CV retail, 2 colonnes, sans code postal)

Score initial : 11/100. Cinq incohérences identifiées et corrigées.

**1. Faux positif localisation — `location='Anglais'`**
`en_core_web_sm` classait le niveau de langue "Anglais" comme entité GPE.
Correction : nouveau `LOCATION_BLOCKLIST` dans `src/core/lexicons.py`
(frozenset des noms de langues, soft-skills, mots génériques d'en-tête CV).
Filtrage appliqué dans `_extract_location()` (`nlp_pipeline.py`) avant
toute tentative de nettoyage de l'entité.

**2. `job_title=None` pour les titres retail français**
`JOB_TITLE_RE` ne couvrait pas les métiers courants du commerce/vente.
Correction : `JOB_TITLE_KEYWORDS` dans `lexicons.py` enrichi avec vendeur,
vendeuse, conseiller, conseillère, commercial, assistant, hôte, caissier,
logisticien, magasinier, stockman, etc.

**3. `name='Keo'` au lieu de `'Keo PEN'` — nom épelé en interlettrage**
Le nom "K e o  P E N" était fragmenté : "Keo" en colonne gauche (section
`header`), "PEN" entraîné en colonne droite (fusionné dans section
`languages`). Le complément de nom n'était jamais retrouvé.
Correction dans `CVTransformer._parse_header()` (`cv_transformer.py`) :
- Reclaim des lignes dans les 12 premiers percentiles des tops verticaux
  de la page (toutes sections), transmises à `_parse_header()` **sans
  modifier `section_map`** (évite de contaminer les autres sections) ;
- Lors de la fusion colonne gauche + colonne droite, la partie issue de la
  colonne gauche est placée en premier : `left_part + " " + right_part`.

**4. Section 'QUALITÉ' non reconnue comme compétences**
Le pattern `skills` dans `SECTION_HEADERS` n'incluait pas les variantes
françaises courantes. Correction : ajout de `qualités?|aptitudes?|atouts|
savoir-être|savoir-faire` dans le pattern `skills`.

**5. Section 'PASSION' parsée comme expériences**
`CVTransformer` n'avait pas de clé `interests` dans `SECTION_HEADERS` —
les lignes tombaient dans la section courante (expérience). Correction :
nouvelle clé `"interests"` ajoutée (`passions?|loisirs?|hobbies?|centres?
d'intérêts?|...`). La section est reconnue et arrête l'accumulation dans
experience ; elle est silencieusement ignorée par le transformer (pas parsée
en objets Pydantic), ce qui est le comportement voulu.

**Tests de régression ajoutés** dans `tests/test_nlp_pipeline.py` (8
tests : blocklist localisation ×3, titres retail FR ×2, qualité→skills ×1,
passion→interests ×2) et `tests/test_cv_transformer.py` (4 tests KEO-PEN :
layout, fusion nom "PEN" inclus, SANDRO en experience, PASSION hors
experience).

---

### Corrections géocodage Adzuna — priorité région + code postal

Problème : `where=Croix` retournait des offres depuis Pont-Croix (Finistère)
au lieu de Croix (Nord, métropole lilloise). L'ancien mécanisme "repli sur
la région si résultats vides" était inefficace car Adzuna renvoyait 10
résultats — de la mauvaise région.

**1. La région sélectionnée manuellement prend toujours la priorité**
Dans `find_matching_jobs()` (`job_matcher.py`) : si `region` est fourni,
la recherche est faite directement sur la région, sans passer par la
localisation du CV. L'ancien `if not listings and region` (fallback) est
supprimé. Tests `test_job_matcher.py` réécrits en conséquence :
- `test_find_matching_jobs_region_takes_priority_over_cv_location`
- `test_find_matching_jobs_uses_city_when_no_region_selected`

**2. Code postal ajouté à la requête Adzuna pour lever l'ambiguïté**
`POSTAL_CODE_CITY_RE` reformaté : groupe(1) = code postal, groupe(2) =
ville (auparavant groupe(1) = ville). Tous les appels `group(1)` pour la
ville mis à jour en `group(2)` dans `cv_transformer.py` et
`nlp_pipeline.py`.

Nouveau champ `postal_code: Optional[str]` ajouté à `CVHeader` et
`ParsedCV` (`schemas.py`). Propagé via :
- `CVTransformer._parse_header()` → `CVHeader.postal_code`
- `NLPPipeline.parse_normalized()` → `ParsedCV.postal_code`

Quand disponible : `where=f"{postal_code} {city}"` (ex. `"59170 Croix"`)
dans `find_matching_jobs()`. Label `region_dropdown` mis à jour dans
`app.py` pour refléter la priorité explicite. Bandeau profil affiche le
code postal quand présent.

Nouveau test : `test_find_matching_jobs_uses_postal_code_with_location`.

**Total suite après ces corrections : 90 tests, tous verts.**
