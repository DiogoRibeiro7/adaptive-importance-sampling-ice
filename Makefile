# Makefile for Safe-ICE development

.PHONY: help install test lint format docs clean build publish docker

# Variables
PYTHON := python
POETRY := poetry
PACKAGE := safe_ice
DOCKER_IMAGE := safe-ice
DOCKER_TAG := latest

# Default target
help:
	@echo "Safe-ICE Development Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install    Install package and dependencies"
	@echo "  test       Run tests with coverage"
	@echo "  lint       Run linting and type checking"
	@echo "  format     Format code with black and isort"
	@echo "  docs       Build documentation"
	@echo "  clean      Clean build artifacts"
	@echo "  build      Build distribution packages"
	@echo "  publish    Publish to PyPI"
	@echo "  docker     Build Docker image"
	@echo ""
	@echo "Development workflow:"
	@echo "  make install  # Set up development environment"
	@echo "  make test     # Run tests"
	@echo "  make format   # Format code"
	@echo "  make lint     # Check code quality"

# Installation
install:
	@echo "Installing Safe-ICE and dependencies..."
	$(POETRY) install --with dev,docs

install-all:
	@echo "Installing Safe-ICE with all extras..."
	$(POETRY) install --with dev,docs --all-extras

# Testing
test:
	@echo "Running tests..."
	$(POETRY) run pytest tests/ -v --cov=$(PACKAGE) --cov-report=term --cov-report=html

test-quick:
	@echo "Running quick tests..."
	$(POETRY) run pytest tests/ -v -m "not slow"

test-integration:
	@echo "Running integration tests..."
	$(POETRY) run pytest tests/test_integration.py -v

benchmark:
	@echo "Running benchmarks..."
	$(POETRY) run python examples/performance_benchmark.py

# Code Quality
lint:
	@echo "Running linters..."
	$(POETRY) run ruff check $(PACKAGE)
	$(POETRY) run mypy $(PACKAGE) --strict

format:
	@echo "Formatting code..."
	$(POETRY) run black $(PACKAGE) tests examples
	$(POETRY) run isort $(PACKAGE) tests examples

check: format lint test
	@echo "All checks passed!"

# Documentation
docs:
	@echo "Building documentation..."
	cd docs && $(MAKE) clean && $(MAKE) html
	@echo "Documentation built at docs/build/html/index.html"

docs-serve:
	@echo "Serving documentation..."
	cd docs/build/html && $(PYTHON) -m http.server 8000

docs-watch:
	@echo "Watching documentation for changes..."
	$(POETRY) run sphinx-autobuild docs/source docs/build/html

# Building
clean:
	@echo "Cleaning build artifacts..."
	rm -rf dist/ build/ *.egg-info
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build: clean
	@echo "Building distribution packages..."
	$(POETRY) build
	@echo "Checking packages..."
	$(POETRY) run twine check dist/*

# Publishing
publish-test: build
	@echo "Publishing to Test PyPI..."
	$(POETRY) run twine upload --repository testpypi dist/*

publish: build
	@echo "Publishing to PyPI..."
	@echo "Are you sure? This will upload to the real PyPI! [y/N]"
	@read -r REPLY; \
	if [ "$$REPLY" = "y" ] || [ "$$REPLY" = "Y" ]; then \
		$(POETRY) run twine upload dist/*; \
	else \
		echo "Aborted."; \
	fi

# Docker
docker-build:
	@echo "Building Docker image..."
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run:
	@echo "Running Docker container..."
	docker run --rm -it $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-jupyter:
	@echo "Building Jupyter Docker image..."
	docker build -f Dockerfile.jupyter -t $(DOCKER_IMAGE)-jupyter:$(DOCKER_TAG) .
	@echo "Starting Jupyter Lab..."
	docker run --rm -p 8888:8888 $(DOCKER_IMAGE)-jupyter:$(DOCKER_TAG)

docker-docs:
	@echo "Building docs Docker image..."
	docker build -f Dockerfile.docs -t $(DOCKER_IMAGE)-docs:$(DOCKER_TAG) .
	@echo "Serving documentation..."
	docker run --rm -p 8000:8000 $(DOCKER_IMAGE)-docs:$(DOCKER_TAG)

docker-compose-up:
	@echo "Starting all services with docker-compose..."
	docker-compose up -d

docker-compose-down:
	@echo "Stopping all services..."
	docker-compose down

# Version Management
version-patch:
	@echo "Bumping patch version..."
	$(PYTHON) scripts/bump_version.py patch

version-minor:
	@echo "Bumping minor version..."
	$(PYTHON) scripts/bump_version.py minor

version-major:
	@echo "Bumping major version..."
	$(PYTHON) scripts/bump_version.py major

# Development Helpers
dev-setup: install
	@echo "Setting up development environment..."
	$(POETRY) run pre-commit install
	@echo "Development environment ready!"

examples:
	@echo "Running example scripts..."
	$(POETRY) run python examples/basic_usage.py
	$(POETRY) run python examples/high_dimensional.py

demo:
	@echo "Running Safe-ICE demo..."
	$(POETRY) run safe-ice demo

# CI/CD Helpers
ci-test:
	@echo "Running CI tests..."
	$(POETRY) run pytest tests/ -v --cov=$(PACKAGE) --cov-report=xml --cov-report=term

ci-lint:
	@echo "Running CI linting..."
	$(POETRY) run ruff check $(PACKAGE) --format=github
	$(POETRY) run mypy $(PACKAGE)

# Release Process
release: check build
	@echo "Preparing release..."
	@echo "1. Update CHANGELOG.md"
	@echo "2. Bump version: make version-[patch|minor|major]"
	@echo "3. Commit changes"
	@echo "4. Tag release: git tag v0.1.0"
	@echo "5. Push: git push && git push --tags"
	@echo "6. Publish: make publish"
	@echo ""
	@echo "See scripts/release_checklist.md for detailed instructions"

.PHONY: all
all: clean install test lint docs build
	@echo "Full build completed successfully!"