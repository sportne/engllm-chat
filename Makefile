VENV = .venv
PYTHON = $(VENV)/bin/python

SRC_DIR := src
TEST_DIR := tests
PACKAGE := engllm_chat

.PHONY: \
    help setup-venv install-dev \
    format format-check \
    lint typecheck \
    test coverage \
    package clean ci

help:
	@echo "engllm-chat Makefile targets:"
	@echo "  make setup-venv   - Create virtual environment"
	@echo "  make install-dev  - Install project with dev dependencies"
	@echo "  make format       - Run black and isort"
	@echo "  make format-check - Check black and isort formatting"
	@echo "  make lint         - Run ruff linting"
	@echo "  make typecheck    - Run mypy static checks"
	@echo "  make test         - Run tests without coverage"
	@echo "  make coverage     - Run tests with coverage threshold"
	@echo "  make package      - Build source and wheel"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make ci           - Run format-check, lint, typecheck, and one coverage-backed test pass"

setup-venv:
	python3 -m venv $(VENV)

install-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .[dev]

format:
	$(PYTHON) -m black $(SRC_DIR) $(TEST_DIR)
	$(PYTHON) -m isort $(SRC_DIR) $(TEST_DIR)

format-check:
	$(PYTHON) -m black --check $(SRC_DIR) $(TEST_DIR)
	$(PYTHON) -m isort --check-only $(SRC_DIR) $(TEST_DIR)

lint:
	$(PYTHON) -m ruff check $(SRC_DIR) $(TEST_DIR)

typecheck:
	$(PYTHON) -m mypy $(SRC_DIR)

test:
	$(PYTHON) -m pytest -o addopts=""

coverage:
	$(PYTHON) -m pytest -o addopts="" tests --cov=$(PACKAGE) --cov-report=term-missing --cov-fail-under=90

package:
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build

clean:
	rm -rf build dist *.egg-info

ci: format-check lint typecheck coverage
