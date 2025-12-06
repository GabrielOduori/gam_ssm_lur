# Makefile for GAM-SSM-LUR development

.PHONY: help install install-dev test test-quick lint format check clean docs build publish publish-test example reproduce

PYTHON := python
PIP := pip
PYTEST := pytest
BLACK := black
RUFF := ruff
MYPY := mypy

# Default target
help:
	@echo "GAM-SSM-LUR Development Commands"
	@echo "================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install package"
	@echo "  make install-dev  Install with development dependencies"
	@echo ""
	@echo "Quality:"
	@echo "  make test         Run tests with coverage"
	@echo "  make lint         Run linters (ruff, mypy)"
	@echo "  make format       Format code with black"
	@echo "  make check        Run all checks (format, lint, test)"
	@echo ""
	@echo "Build:"
	@echo "  make build        Build distribution packages"
	@echo "  make docs         Build documentation"
	@echo "  make clean        Remove build artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make example      Run basic example"
	@echo "  make reproduce    Reproduce paper results"

# Installation
install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install

# Testing
test:
	$(PYTEST) tests/ -v --cov=gam_ssm_lur --cov-report=term-missing --cov-report=html

test-quick:
	$(PYTEST) tests/ -v -x --tb=short

# Code quality
lint:
	$(RUFF) check src/ tests/ experiments/
	$(MYPY) src/gam_ssm_lur/ --ignore-missing-imports

format:
	$(BLACK) src/ tests/ experiments/
	$(RUFF) check src/ tests/ experiments/ --fix

check: format lint test

# Build
build: clean
	$(PYTHON) -m build

docs:
	@if [ -d docs ]; then \
		cd docs && make html; \
	else \
		echo "docs/ directory not found; nothing to build."; \
	fi

# Clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Examples
example:
	$(PYTHON) experiments/01_basic_usage.py

reproduce:
	$(PYTHON) experiments/reproduce_paper.py --data-dir data/ --output-dir results/

# Publishing (use with caution)
publish-test:
	$(PYTHON) -m twine upload --repository testpypi dist/*

publish:
	$(PYTHON) -m twine upload dist/*
