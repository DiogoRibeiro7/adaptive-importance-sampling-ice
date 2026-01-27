# Contributing to Safe-ICE

Thank you for your interest in contributing to Safe-ICE! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)
- [Questions?](#questions)

## Getting Started

We welcome contributions of all kinds:
- Bug fixes
- New features
- Documentation improvements
- Test coverage improvements
- Performance optimizations
- Example scripts

Before making significant changes, please open an issue to discuss your proposed changes.

## Development Setup

### 1. Fork and Clone

Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/YOUR-USERNAME/adaptive-importance-sampling-ice.git
cd adaptive-importance-sampling-ice
```

### 2. Install Poetry

This project uses Poetry for dependency management. Install it if you haven't:

```bash
pip install poetry
```

or follow the official installation guide: https://python-poetry.org/docs/#installation

### 3. Install Dependencies

Install the project and its dependencies:

```bash
poetry install
```

This will create a virtual environment and install all required dependencies.

### 4. Install Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
poetry run pre-commit install
```

This will automatically run formatters and linters before each commit.

### 5. Verify Installation

Run the tests to ensure everything is working:

```bash
poetry run pytest
```

## Code Style

We maintain strict code quality standards:

### Formatting
- **Black**: Code formatter with line length 88
- **isort**: Import sorting, Black-compatible
- Run: `poetry run black .` and `poetry run isort .`

### Linting
- **ruff**: Fast Python linter
- Run: `poetry run ruff check .`

### Type Checking
- **mypy**: Static type checking in strict mode
- All functions must have type hints
- Run: `poetry run mypy safe_ice`

### Documentation
- All public functions/classes must have docstrings
- Use NumPy-style docstrings
- Include parameter types and return types
- Add usage examples for complex functions

### Example Docstring

```python
def estimate_failure_probability(
    self,
    initial_params: Optional[vMFNMParameters] = None
) -> Tuple[float, NDArrayF, NDArrayF]:
    """Estimate failure probability using Safe-ICE algorithm.

    Parameters
    ----------
    initial_params : vMFNMParameters, optional
        Initial parameters for the vMFNM distribution.
        If None, uses default initialization.

    Returns
    -------
    pf : float
        Estimated failure probability
    samples : NDArrayF
        Generated samples, shape (n_total_samples, dimension)
    weights : NDArrayF
        Importance weights, shape (n_total_samples,)

    Examples
    --------
    >>> ice = SafeICE(g, dimension=2)
    >>> pf, samples, weights = ice.estimate_failure_probability()
    >>> print(f"Failure probability: {pf:.2e}")
    """
```

## Testing

### Running Tests

Run all tests:
```bash
poetry run pytest
```

Run with coverage:
```bash
poetry run pytest --cov=safe_ice --cov-report=html
```

Run specific test file:
```bash
poetry run pytest tests/test_distributions.py
```

Run with markers:
```bash
poetry run pytest -m "not slow"  # Skip slow tests
poetry run pytest -m integration  # Run only integration tests
```

### Writing Tests

- Place unit tests in `tests/`
- Name test files as `test_*.py`
- Use pytest fixtures for common setup
- Aim for >80% code coverage
- Test edge cases and error conditions

Example test:
```python
import pytest
import numpy as np
from safe_ice import SafeICE

def test_safe_ice_dimension():
    """Test SafeICE respects dimension parameter."""
    def g(u):
        return 3.5 - np.linalg.norm(u, axis=-1)

    ice = SafeICE(g, dimension=5)
    assert ice.dimension == 5
```

## Submitting Changes

### 1. Create a Feature Branch

Create a branch from `develop`:

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `test/` - Test improvements
- `perf/` - Performance improvements

### 2. Make Your Changes

- Write clear, concise commit messages
- Include tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Your Changes

Commit messages should follow this format:
```
type: Brief description (max 50 chars)

Longer description if needed. Explain the problem this
commit is solving and why this approach was chosen.

Fixes #123  # Reference issues if applicable
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/changes
- `perf`: Performance improvements
- `refactor`: Code restructuring
- `style`: Formatting changes
- `chore`: Maintenance tasks

### 4. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub:
- Target the `develop` branch
- Provide a clear description
- Reference any related issues
- Ensure CI checks pass

### 5. Pull Request Checklist

Before submitting:
- [ ] Tests pass locally (`poetry run pytest`)
- [ ] Code is formatted (`poetry run black .`)
- [ ] Imports are sorted (`poetry run isort .`)
- [ ] Linting passes (`poetry run ruff check .`)
- [ ] Type checking passes (`poetry run mypy safe_ice`)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (for significant changes)

## Reporting Issues

### Bug Reports

When reporting bugs, please include:
1. Python version and OS
2. Complete error message and traceback
3. Minimal reproducible example
4. Expected vs actual behavior
5. Steps to reproduce

Use this template:
```markdown
**Environment:**
- Python version:
- OS:
- Safe-ICE version:

**Description:**
[Clear description of the bug]

**To Reproduce:**
```python
# Minimal code to reproduce
```

**Expected behavior:**
[What should happen]

**Actual behavior:**
[What actually happens]

**Error message:**
```
[Complete traceback]
```
```

### Security Issues

For security vulnerabilities, please email the maintainers directly instead of opening a public issue.

## Feature Requests

For feature requests, please:
1. Check existing issues/PRs first
2. Provide clear use case
3. Explain why this feature would be useful
4. Consider implementation approach

## Questions?

- Check the documentation first
- Search existing issues
- Open a discussion issue for general questions
- Tag issues appropriately

## Code of Conduct

Please note that this project adheres to a Code of Conduct. By participating, you are expected to:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Accept feedback gracefully

## Recognition

Contributors will be recognized in:
- The AUTHORS file
- Release notes
- Project documentation

Thank you for contributing to Safe-ICE!