PYTHON  ?= python3.11
VENV    ?= .venv
PIP     := $(VENV)/bin/pip
PY      := $(VENV)/bin/python
PYTEST  := $(VENV)/bin/pytest
BLACK   := $(VENV)/bin/black
RUFF    := $(VENV)/bin/ruff
MYPY    := $(VENV)/bin/mypy
BANDIT  := $(VENV)/bin/bandit

.DEFAULT_GOAL := help

.PHONY: help venv install test lint format security docker clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n",$$1,$$2}'

venv:  ## Create virtualenv
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv  ## Install all dependencies
	$(PIP) install -e ".[dev,api,train]"

test:  ## Run pytest with coverage
	$(PYTEST) tests/ -v --cov=. --cov-report=term-missing --cov-fail-under=70

lint:  ## Run ruff linter + mypy type checker
	$(RUFF) check . --select=E,W,F,I
	$(MYPY) api.py inference/ platform/ --ignore-missing-imports --no-error-summary

format:  ## Auto-format with black + ruff fix
	$(BLACK) .
	$(RUFF) check . --fix

security:  ## Bandit security scan (zero high/medium issues)
	$(BANDIT) -r api.py inference/ platform/ ui/ vl_trainer.py \
	  -ll -ii --exit-zero -f txt

docker-build:  ## Build Docker image
	docker compose build

docker-up:  ## Start all services
	docker compose up -d

docker-down:  ## Stop all services
	docker compose down

clean:  ## Remove build artefacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage
