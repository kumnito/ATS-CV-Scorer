---
title: ATS CV Scorer
emoji: 📋
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Analyse your CV against job descriptions with NLP + Claude AI
---

# ATS CV Scorer

Portfolio project — ML Engineer track.

## Stack

| Layer | Tech |
|---|---|
| PDF extraction | pdfplumber |
| NLP | spaCy `en_core_web_sm` |
| Semantic scoring | `sentence-transformers` (all-MiniLM-L6-v2) |
| AI feedback | Anthropic Claude API |
| API | FastAPI |
| UI | Gradio |
| Deployment | Hugging Face Spaces |

## Score breakdown

| Component | Weight |
|---|---|
| Keyword match | 40 % |
| Semantic similarity | 35 % |
| CV structure completeness | 25 % |

Claude AI feedback is optional and requires an `ANTHROPIC_API_KEY`.

## Local setup

```bash
cp .env.example .env      # add your ANTHROPIC_API_KEY
make install
make run                  # Gradio UI → http://localhost:7860
make dev                  # FastAPI  → http://localhost:8000/docs
make test
```
