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

---

## Travaux réalisés (session du 2026-06-09 — session 3)

### Amélioration de la pertinence Adzuna — 5 axes

#### 1. Détection du secteur (`src/core/lexicons.py`, `nlp_pipeline.py`, `schemas.py`)
Nouveau `SECTOR_KEYWORDS` dans `lexicons.py` : 8 secteurs (magasin, mode,
restauration, transport, industrie, btp, santé, finance). Nouvelle fonction
`_extract_sector(NormalizedCV)` dans `nlp_pipeline.py` : scan du texte des
expériences (titre + company + bullets) pour détecter le secteur dominant.
Propagé dans `parse_normalized()` → champ `sector: Optional[str]` ajouté à
`ParsedCV`.

#### 2. Multi-requêtes Adzuna + déduplication par URL (`job_matcher.py`)
`JOB_TITLE_SYNONYMS` ajouté dans `lexicons.py` (vendeur/développeur/data
scientist/conducteur/opérateur). Nouveau `_find_synonym_queries(title)`
et `_build_queries(parsed_cv)` → génèrent 1-3 requêtes Adzuna (titre de
base + synonyme + titre×secteur). Chaque requête reçoit
`max_results = max(5, max_results // nb_requêtes)`. Les offres de toutes
les requêtes sont dédupliquées par URL avant scoring.

#### 3. Filtre de seuil qualité + signal "peu d'offres" (`job_matcher.py`)
`_MIN_SCORE_THRESHOLD = 25.0` — les offres sous ce score sont filtrées.
Fallback : si aucune offre ne passe le seuil, toutes sont retournées (pas
de page blanche). `few_results: bool` dans `JobSearchResult` — True quand
`len(filtered) < 3 and bool(scored)`. Affiché dans `app.py` via
`few_results_note`.

#### 4. Nouveau type de retour `JobSearchResult` (dataclass)
Remplace `list[RankedJobMatch]`. Champs : `matches`, `queries_used`,
`location_used`, `few_results`. Toutes les occurrences dans `app.py` et les
tests mises à jour.

#### 5. UI — champ titre modifiable + affichage requête (`app.py`)
- `job_title_input: gr.Textbox` pré-rempli au chargement du CV (output de
  `on_cv_upload`) — l'utilisateur peut corriger le titre avant la recherche.
- `search_query_md: gr.Markdown` — affiché après la recherche :
  `🔍 Requête Adzuna : "vendeur" · variantes : "conseiller de vente" · 📍 Croix (30 km)`
- `on_search` accepte désormais `title_override` ; si non vide, remplace
  `parsed_cv.job_title` via `model_copy(update={...})` avant l'appel
  `find_matching_jobs`.

### Tests ajoutés (session 3)
- `test_job_matcher.py` : 14 nouveaux tests (synonymes, secteur, dédup,
  seuil, few_results) + mise à jour des assertions `calls` → vérification
  "toutes les requêtes utilisent la bonne localisation" (plus d'assertion
  sur une liste exacte).
- `test_nlp_pipeline.py` : 6 nouveaux tests `_extract_sector`.

**Total suite : 161 tests, tous verts.**

---

## Travaux réalisés (session du 2026-06-09 — session 4)

### 1. Cascade d'extraction PDF 3 niveaux (`src/services/cv_transformer.py`)

Refonte de `CVTransformer.transform()` :

**Niveau 1 — pdfplumber** : confiance = `min(1.0, mots_extraits / 450)`.
Si ≥ 0.85 → retour immédiat.

**Niveau 2 — OCR Tesseract** : déclenché si confiance < 0.85. Ne remplace
pdfplumber que si `ocr_mots > pdf_mots × 1.10 AND ocr_mots ≥ 150` (seuil
+10% pour préserver l'avantage layout de pdfplumber en cas de gain marginal).

**Niveau 3 — Vision LLM Claude** : déclenché si confiance finale < 0.85 ET
`settings.anthropic_api_key` présent. Gagne sur `_vision_richness_score()`
(score structurel : expériences × 10, éducation × 8, skills × 2, projets × 10,
email +15, titre +15, summary +10) — pas sur le word_count.

**Invariant layout** : `_detect_layout()` tourne toujours sur les données
pdfplumber réelles. Résultat injecté via `model_copy(update={"layout_detected":
real_layout})` dans tous les chemins (Vision LLM retourne toujours
"single_column", ce qui écraserait sinon la vraie détection).

**Post-Vision LLM** : `_build_raw_text_from_normalized()` reconstruit raw_text
depuis les champs structurés pour un word_count précis.

**Logging** : `import logging; logger = logging.getLogger(__name__)` dans
cv_transformer.py. `app.py` : `logging.basicConfig(level=logging.INFO, ...)`.

**Tests** : `_transform_pdfplumber_only()` patche `settings.anthropic_api_key = ""`
ET mocke `_extract_text_ocr → ""` pour forcer le chemin pdfplumber dans les
fixtures lisant de vrais PDFs. Tests Vision LLM basés sur richesse structurelle
(`mock_vision_cv_rich` / `mock_vision_cv_poor`), pas sur le word_count.
`pdf2image` mocké via `patch.dict(sys.modules, {"pdf2image": mock_pdf2image})`.

**3 tests connus en échec** (non-régressifs, pre-existants dans la session) :
`test_two_col_layout_detected`, `test_keo_pen_layout_two_columns`,
`test_keo_pen_experience_contains_sandro`.

### 2. Logging détaillé dans CVQualityScorer

`logger.info()` ajouté dans `CVQualityScorer.score()` pour diagnostiquer les
critères du score contenus : word_count, skills total, has_metrics, has_verbs,
has_projects, has_dates, plus le détail de chaque `pts_*`.

### 3. Refonte CVQualityReport — 3 axes indépendants

**Nouveaux modèles Pydantic** dans `src/core/schemas.py` :

```python
class ATSReadability(BaseModel):
    layout: str              # "single_column" | "two_columns"
    layout_label: str        # "✅ Optimal" | "⚠️ Risque parseur"
    sections_found: list[str]
    sections_missing: list[str]
    extraction_method: str
    is_machine_readable: bool  # word_count >= 150

class ProfileStrength(BaseModel):
    level: str    # "Solide" | "Correct" | "À renforcer"
    score: int    # 0-100
    strengths: list[str]
    improvements: list[str]

class Recommendation(BaseModel):
    priority: int   # 1=Fort, 2=Moyen, 3=Faible
    impact: str     # "Fort" | "Moyen" | "Faible"
    action: str
    why: str
```

**`CVQualityReport` simplifié** : `ats_readability`, `profile_strength`,
`list[Recommendation]`, timeline carrière (conservée), `extraction_method` +
`extraction_confidence` (compatibilité). Supprimés : `score_global`,
`score_structure`, `score_contenu`, `keyword_density`, `has_metrics`,
`sections_detectees`, `sections_manquantes`, `layout`, `word_count`.

**Scoring ProfileStrength recalibré FR (8 critères, max 100 pts) :**
- word_count ≥ 300 → +15
- skills_total ≥ 10 → +20
- experience_count ≥ 1 → +15
- summary présent → +10
- dates ≥ 50% des entrées expérience → +10
- action_verbs OU action_nouns FR → +10 (nouveau : frozenset `_ACTION_NOUNS_FR`
  accepte "Ingestion", "Contrôle", "Automatisation", etc.)
- métriques présentes → +10
- projets présents → +10

Niveaux : "Solide" ≥ 75, "Correct" ≥ 50, "À renforcer" < 50.

**Recommandations** ordonnées par priorité :
1 (Fort) — bloquant ATS : layout 2 colonnes, sections manquantes
2 (Moyen) — impact matching : métriques, projets, summary, dates
3 (Faible) — polissage : verbes, skills < 10, mots < 300, sur-optimisation

**UI `app.py`** : `_format_quality_report()` réécrit en 3 blocs Markdown
(📋 Lisibilité ATS · 💼 Solidité du profil · 🎯 Actions prioritaires).
`_format_cv_context_strip()` utilise `report.profile_strength.score`.
`_extraction_badge()` supprimée (info intégrée dans la table ATS).

**Tests** : `tests/test_cv_quality_scorer.py` réécrit — 32 tests (vs 21 avant),
tous verts. Couverture : ATSReadability, ProfileStrength (score + level +
strengths/improvements), Recommendations (schema, priorité, impact), action
nouns FR, career stats, career gaps.

**Total suite (hors cv_transformer) : 146 tests verts.**

---

## Travaux réalisés (session du 2026-06-15)

### 1. Refactor : suppression de `ParsedCV`, `NormalizedCV` unifié

`ParsedCV` (schéma séparé dans `src/core/schemas.py`) supprimé. `NormalizedCV`
enrichi avec les champs auparavant portés par `ParsedCV` : `sections`,
`entities`, `keywords`, `skills_flat`, `experience_years`, `job_title`,
`location`, `postal_code`, `sector`.

`NLPPipeline.parse_normalized(cv: NormalizedCV) -> NormalizedCV` (au lieu de
`-> ParsedCV`) : retourne `normalized_cv.model_copy(update={...})` avec ces
champs résolus. `_extract_skills` réécrit autour de `ALL_SKILLS_RE` (regex
pré-compilée, remplace le scan O(n) sur `ALL_SKILLS`).

`src/core/lexicons.py` : la fusion `lexicons_generated.json` + le calcul des
structures dérivées (`ALL_SKILLS`, `ALL_SKILLS_RE`, `JOB_TITLE_RE`,
`ACTION_VERBS_EN/FR`) sont déplacés dans `init_lexicons()` — idempotent,
appelée explicitement au démarrage de `app.py` et `src/api/server.py`
(plus de calcul au moment de l'import du module).

Tous les services (`semantic_scorer`, `claude_feedback`, `job_matcher`) et
`app.py` utilisent `NormalizedCV` + `.skills_flat` (suppression de
`.skills` liste plate sur ce type). `ATSResponse.parsed_cv` est désormais
`NormalizedCV`.

`src/api/server.py` : nouveau context manager async `_tmp_pdf(cv_file)`
(validation MIME + taille + écriture/suppression du fichier temporaire),
factorisé pour `/score` et `/find-jobs` (suppression de code dupliqué).

Suppression de code mort : `NLPPipeline.parse_cv()`, `JOB_TITLES`,
`TECH_SKILLS`, `ai_feedback_score` (ScoreBreakdown).

**Total suite : 234/234 tests verts.**

### 2. Refonte UI — thème "Tech Dashboard" (`app.py`)

Nouveau thème Gradio custom : `gr.themes.Base(primary_hue=indigo,
secondary_hue=slate, neutral_hue=slate, font=Inter, font_mono=[GoogleFont
("JetBrains Mono")])` + `CUSTOM_CSS` injecté via `gr.Blocks(theme=THEME,
css=CUSTOM_CSS)`.

**Piège gradio 4.44.0** : `font_mono` doit être passé en **liste**
(`[gr.themes.GoogleFont(...)]`) — un `GoogleFont` nu fait planter
`gr.themes.Base.__init__` (`TypeError: 'GoogleFont' object is not
iterable`), car le wrap automatique en liste ne s'applique qu'à `font`,
pas à `font_mono`, dans cette version.

**Nouveaux éléments** :
- `_format_metrics_html()` (`gr.HTML`, onglet 1, avant les détails) : 4
  cards horizontales (Lisibilité ATS, Profil, Mots-clés, Expérience) +
  badge méthode d'extraction (`_extraction_badge_html` : ✅ native / ⚠️ OCR
  / 🤖 Vision IA + confiance %).
- `_format_skill_badges()` (`gr.HTML` séparé) : compétences en badges,
  groupées par catégorie (ML/MLOps/Cloud/Data/Langages/Autres/Commerce) —
  remplace la liste plate retirée de `_format_profile_summary`.
- `_format_timeline()` réécrit en HTML (`.timeline-entry`/dot/title/meta/
  badge) : point vert = expérience, indigo = formation, amber = trou de
  carrière (`career_gaps`).
- `_format_job_card()` réécrit : `.result-card`, score affiché en grand
  (22px) coloré par seuil via `_score_color_hex()` (≥70 vert `#22c55e`,
  ≥50 amber `#f59e0b`, <50 rouge `#ef4444`).
- `_progress_bar_html()` : barre HTML générique, remplace l'ancien rendu
  Markdown `███░░░` (utilisée pour la barre "Profil").
- `elem_classes=["cv-upload"]` sur `cv_input` (zone dashed + hover),
  `elem_classes=["analyze-outline"]` sur les boutons "Analyser cette offre".

`_upload_outputs`, `on_cv_upload`, `on_cv_clear` étendus avec les deux
nouveaux `gr.HTML()` (`metrics_html`, `skills_html`).

Commit `28901f2`, push `origin` + `hf`.

### 3. Correctif contraste dark mode (`app.py`)

**Symptôme** : sur HF Spaces en dark mode, titres de sections et texte des
cards quasi invisibles — seuls badges et scores (couleurs auto-portées)
restaient lisibles.

**Cause** : `CUSTOM_CSS` forçait des fonds clairs hardcodés (fond de page
`#f8fafc`, cards `#ffffff`) pendant que le texte Gradio (`--body-text-
color`) bascule en clair sous `.dark` → texte clair sur fond clair forcé.

**Fix** : toutes les variables `--app-*` remappées sur les variables
natives Gradio (qui basculent automatiquement avec la classe `.dark`) :
- `--app-border` → `var(--border-color-primary)`
- `--app-card-bg` → `var(--background-fill-primary)`
- `--app-text-secondary` → `var(--body-text-color-subdued)`
- fond `.gradio-container` → `var(--background-fill-secondary)`
- `.result-card`/`.metric-card`/`.timeline-entry` reçoivent
  `color: var(--body-text-color)` (hérité par `.metric-value`,
  `.timeline-title`, titre/score d'offre).
- `--app-badge-bg`/`--app-badge-text` (indigo-50/700) : override
  `.dark .gradio-container { ... }` → indigo-900/200.
- Accents `#4338ca` (faible contraste sur fond sombre) remplacés par
  `#6366f1` (onglet actif, hover bouton "Analyser").

**Note** : les noms `--color-text-primary`, `--color-background-info`,
etc. (parfois cités dans des specs UI) **n'existent pas** dans gradio
4.44 — utiliser `--body-text-color(-subdued)`, `--background-fill-
primary/secondary`, `--border-color-primary`, `--color-accent(-soft)`.

234/234 tests verts (app.py hors suite). Commit `e4c701f`, push `origin` +
`hf`. README mis à jour (commit `4f8bccc`) avec une section "UI — Tech
Dashboard".

---

## Travaux réalisés (session du 2026-06-15 — session 2) : recherche multi-sources

### Phase 1-3 : Provider pattern (Adzuna, Jooble, France Travail)

Nouveau package `src/services/job_providers/` :

- `base.py` — ABC `JobProvider` : `name`, `color`, `check_availability() ->
  tuple[bool, float]` (disponibilité + latence ms), `search(query,
  location=None, max_results=20, distance=None) -> list[JobListing]`.
- `adzuna.py` — `AdzunaProvider`, extrait de l'ancien `JobSearchService`.
  Couleur indigo `#6366f1`.
- `jooble.py` — `JoobleProvider` (API POST, clé `JOOBLE_API_KEY`). Couleur
  verte `#10b981`. `distance` → `radius` dans le payload.
- `france_travail.py` — `FranceTravailProvider`, OAuth2 client credentials
  (`FRANCE_TRAVAIL_CLIENT_ID`/`_SECRET`). Couleur bleue `#2563eb`. Localisation
  résolue en département via `_extract_department_code` (code postal → 2
  premiers chiffres). `distance` → `rayon` (défaut 30).
- `oauth2_token_manager.py` — `OAuth2TokenManager` thread-safe
  (`threading.Lock`) : `get_token()` (cache, buffer expiration 60s) vs
  `refresh()` (forcé, utilisé par `check_availability()`).
- `orchestrator.py` — `JobSearchOrchestrator(providers)` :
  `check_all_availability()` (ThreadPoolExecutor, parallèle),
  `search(query, location=None, active_providers=None, max_results=20,
  distance=None)` — fusionne + déduplique par URL, dégradation gracieuse si
  un provider échoue ou est inactif. `active_providers=None` → tous les
  providers.

`JobListing` (`schemas.py`) : nouveaux champs `source: str = "adzuna"`,
`source_color: str = "#6366f1"`.

Commits `948511e` (Adzuna provider), `6f51cea` (Jooble), `1799b6b`
(France Travail OAuth2).

### Phase 4 : UI multi-sources (`app.py`)

**Initialisation** : `_providers` (liste des 3 providers) → `_orchestrator
= JobSearchOrchestrator(providers=_providers)`. `_PROVIDER_LABELS` (dict
nom → (label affiché, tagline)). `_default_active_providers()` : Adzuna
toujours actif, Jooble/France Travail actifs par défaut seulement si leurs
credentials sont configurés.

**`job_matcher.find_matching_jobs`** : nouveau paramètre
`active_providers`, transmis à `job_search.search(...)`. `JobSearchResult`
enrichi de `source_counts: dict[str, int]` (via `Counter`) et
`duplicates_removed: int` (calculé sur l'ensemble multi-requêtes).
`JobSearchService.search` (single-provider wrapper, conservé pour
`src/api/server.py`) accepte `active_providers` mais l'ignore.

**UI (`app.py`)** :
- `sources_html` (`gr.HTML`) : 3 cards statut (✅ vert + latence / ⏳ amber
  / ❌ rouge), peuplées par `demo.load(fn=on_load_check_providers, ...)` —
  `check_all_availability()` tourne **au chargement de l'onglet**, pas à la
  recherche.
- `active_providers_checkbox` (`gr.CheckboxGroup`) : sources actives,
  valeur par défaut `_default_active_providers()`, transmise à `on_search`
  → `find_matching_jobs(..., active_providers=...)`.
- `_format_job_card` : pill colorée par source (`_source_pill_html`, classe
  CSS `.source-pill`, couleur = `job.source_color` + fond à 10% d'opacité
  `{color}1a`).
- `source_stats_md` : ligne "Adzuna (N) · Jooble (N) · France Travail (N) ·
  doublons retirés (N)" via `_format_source_stats`.
- `source_filter_dropdown` + `salary_filter_checkbox` : filtrent
  `matches_state` côté client (`on_filter_change` → `_render_job_slots`),
  **sans relancer la recherche**.
- `query_display` rendu source-agnostic ("🔍 Requête :" au lieu de "Requête
  Adzuna").
- `on_search` : remplace `_job_search_service` par `_orchestrator`,
  réinitialise les filtres (source = "Toutes sources", salaire = False) à
  chaque nouvelle recherche.

**Tests** : signatures `search()` mises à jour partout (providers, fakes,
orchestrator) pour inclure `distance` et `active_providers`. 269/269 tests
verts. Validation `make run` end-to-end via `gradio_client`/`httpx` brut
(upload CV → 15 offres, stats "Adzuna (10) · Jooble (5) · doublons retirés
(4)", filtre par source fonctionnel). France Travail répond ❌ en local
(token OAuth2 400 — credentials/scope à vérifier séparément, hors scope de
cette session).

Commit prévu : `feat: UI multi-sources (Adzuna + Jooble + France Travail) +
statut API temps réel`.
