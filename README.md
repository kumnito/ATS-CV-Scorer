---
title: ATS CV Scorer
emoji: 📋
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Score your CV, find matching jobs, get AI feedback
---

# ATS CV Scorer

Portfolio project — ML Engineer track.

Upload your CV (PDF) only: the pipeline extracts your **job title**,
**skills** and **location**, automatically searches matching job listings via
the **Adzuna** API, scores each one against your profile with a semantic
similarity engine, and ranks them. You can then request a **personalized
Claude AI analysis per listing, on demand** — no AI call is triggered until
you click "Analyser cette offre".

## Stack

| Layer | Tech |
|---|---|
| PDF extraction | pdfplumber (layout-aware — single/two-column detection) |
| NLP | spaCy `en_core_web_sm` |
| Semantic scoring | `sentence-transformers` (all-MiniLM-L6-v2) |
| CV quality scoring | custom rule-based scorer (structure + content, 0-100) |
| AI feedback | Anthropic Claude API (responses in French) |
| Job search | Adzuna API (`httpx`) |
| API | FastAPI |
| UI | Gradio |
| Deployment | Hugging Face Spaces |

## How it works

1. **Extraction** — `CVTransformer` uses pdfplumber's character-level
   positional data to detect layout (single / two-column), reconstruct
   reading order, and produce a fully structured `NormalizedCV` (header,
   skills, experience, education, projects).
2. **CV quality analysis** — `CVQualityScorer` immediately produces a
   quality report: structure score, content score, detected/missing
   sections, keyword density, career timeline, gaps, and ATS
   recommendations — visible before any job search.
3. **NLP** — spaCy detects the job title and location from the header
   (postal address pattern first, filtered NER as fallback — `en_core_web_sm`
   is an English model and unreliable on French text).
4. **Job search** — Adzuna is queried around the detected location (30 km
   radius). If that yields nothing (Adzuna's geocoding misses small
   towns/suburbs), you can pick a French region from the dropdown as a
   manual fallback — no blind nationwide search.
5. **Scoring** — each listing is scored against your CV with the semantic
   engine and ranked by overall compatibility.
6. **AI feedback** — click "Analyser cette offre" on any listing to get a
   personalized, French-language Claude analysis (skill gaps, ATS keyword
   suggestions, structural improvements).

## Score breakdown

| Component | Weight |
|---|---|
| Keyword match | 40 % |
| Semantic similarity | 35 % |
| CV structure completeness | 25 % |

Claude AI feedback is optional and requires an `ANTHROPIC_API_KEY`.
Job search requires `ADZUNA_ID` / `ADZUNA_API_KEY` (free Adzuna developer
account — register on their site to obtain credentials).

## Local setup

```bash
cp .env.example .env      # add ANTHROPIC_API_KEY and ADZUNA_ID / ADZUNA_API_KEY
make install
make run                  # Gradio UI → http://localhost:7860
make dev                  # FastAPI  → http://localhost:8000/docs
make test
```
