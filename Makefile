.PHONY: install install-dev run dev test lint format clean update-lexicons install-ocr benchmark

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV)/bin/activate
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -r requirements-dev.txt

run: install
	$(PYTHON) app.py

dev: install
	$(VENV)/bin/uvicorn src.api.server:app --reload --port 8000

test: install-dev
	$(VENV)/bin/pytest tests/ -v

lint: install
	$(VENV)/bin/ruff check src/ tests/

format: install
	$(VENV)/bin/ruff format src/ tests/

update-lexicons: install
	$(PYTHON) -m src.services.lexicon_builder --force

install-ocr: install
	sudo apt-get install -y tesseract-ocr tesseract-ocr-fra poppler-utils
	$(PIP) install pytesseract pdf2image Pillow

benchmark: install-dev
	$(PYTHON) tests/benchmark_ats.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
