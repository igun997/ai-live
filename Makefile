SHELL := /bin/bash
VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn

.PHONY: help setup install-cuda download-models run dev test lint clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: $(VENV)/bin/activate ## Create venv and install dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "Setup complete. Copy .env.example to .env and add your GEMINI_API_KEY:"
	@echo "  cp .env.example .env"
	@echo ""

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install-cuda: setup ## Install PyTorch with CUDA 12.4 support
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

download-models: setup ## Download Whisper and Kokoro models
	$(PYTHON) -c "\
from faster_whisper import WhisperModel; \
print('Downloading Whisper small model...'); \
model = WhisperModel('small', device='cpu', compute_type='int8'); \
print('Whisper model ready.')"
	$(PYTHON) -c "\
from kokoro import KPipeline; \
print('Initializing Kokoro TTS (English)...'); \
p = KPipeline(lang_code='a'); \
print('Kokoro English pipeline ready.'); \
print('Initializing Kokoro TTS (French)...'); \
p = KPipeline(lang_code='f'); \
print('Kokoro French pipeline ready.')"
	@echo "All models downloaded."

run: ## Run the server (production)
	$(UVICORN) src.main:app --host 0.0.0.0 --port 8000

dev: ## Run the server with auto-reload (development)
	$(UVICORN) src.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir src

test: ## Run tests
	$(PYTHON) -m pytest tests/ -v

lint: ## Run linter
	$(PYTHON) -m ruff check src/ tests/

clean: ## Remove venv, caches, and generated files
	rm -rf $(VENV)
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
	rm -rf outputs/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
