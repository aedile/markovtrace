.DEFAULT_GOAL := help
SHELL := /bin/bash

.PHONY: help install lint format-check test test-cov ci-local

help: ## Show this help text and exit
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

install: ## uv sync --extra dev --extra test
	uv sync --extra dev --extra test

lint: ## Run ruff, mypy --strict, bandit
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy --strict src/
	uv run bandit -c pyproject.toml -r src/

format-check: ## Verify formatting only
	uv run ruff format --check .

test: ## Run pytest
	uv run pytest

test-cov: ## Run pytest with coverage gate (80%)
	uv run pytest --cov=src/markovtrace --cov-report=term-missing --cov-fail-under=80

ci-local: lint test-cov ## Run the full local CI battery (lint + tests with coverage)
