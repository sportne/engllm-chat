VENV = .venv
PYTHON = $(VENV)/bin/python

SRC_DIR := src
TEST_DIR := tests
PACKAGE := engllm_chat

.PHONY: \
    help setup-venv install-dev \
    format format-check \
    lint typecheck \
    test coverage smoke-chat smoke-ollama-chat smoke-pex \
    package clean ci

OLLAMA_BASE_URL ?= http://127.0.0.1:11434
OLLAMA_MODEL ?= qwen2.5-coder:7b-instruct-q4_K_M
SMOKE_PROVIDER ?= ollama
SMOKE_MODEL ?=
SMOKE_BASE_URL ?=

help:
	@echo "engllm-chat Makefile targets:"
	@echo "  make setup-venv   - Create virtual environment"
	@echo "  make install-dev  - Install project with dev dependencies"
	@echo "  make format       - Run Ruff formatting"
	@echo "  make format-check - Check Ruff formatting"
	@echo "  make lint         - Run ruff linting"
	@echo "  make typecheck    - Run mypy static checks"
	@echo "  make test         - Run tests without coverage"
	@echo "  make coverage     - Run tests with coverage threshold"
	@echo "  make smoke-chat   - Run the opt-in provider-backed chat workflow smoke test"
	@echo "  make smoke-ollama-chat - Run the opt-in local Ollama chat workflow smoke test"
	@echo "  make smoke-pex    - Build the .pex artifact and smoke-check the packaged CLI"
	@echo "  make package      - Build source and wheel"
	@echo "  make package-pex  - Build the single-file .pex artifact"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make ci           - Run format-check, lint, typecheck, and one coverage-backed test pass"

setup-venv:
	python3 -m venv $(VENV)

install-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .[dev]

format:
	$(PYTHON) -m ruff format $(SRC_DIR) $(TEST_DIR)

format-check:
	$(PYTHON) -m ruff format --check $(SRC_DIR) $(TEST_DIR)

lint:
	$(PYTHON) -m ruff check $(SRC_DIR) $(TEST_DIR)

typecheck:
	$(PYTHON) -m mypy $(SRC_DIR)

test:
	$(PYTHON) -m pytest -o addopts=""

coverage:
	$(PYTHON) -m pytest -o addopts="" tests --cov=$(PACKAGE) --cov-report=term-missing --cov-fail-under=90

smoke-chat:
	$(PYTHON) -m engllm_chat.smoke_chat --provider "$(SMOKE_PROVIDER)" --directory . $(if $(SMOKE_MODEL),--model "$(SMOKE_MODEL)") $(if $(SMOKE_BASE_URL),--base-url "$(SMOKE_BASE_URL)") --require-tool-call

smoke-ollama-chat:
	$(PYTHON) -m engllm_chat.smoke_chat --provider ollama --directory . --model "$(OLLAMA_MODEL)" --base-url "$(OLLAMA_BASE_URL)" --require-tool-call

package:
	$(PYTHON) -m build

package-pex:
	$(PYTHON) scripts/build_pex.py --project-root .

smoke-pex:
	$(PYTHON) scripts/build_pex.py --project-root . --smoke

clean:
	rm -rf build dist *.egg-info

ci: format-check lint typecheck coverage
