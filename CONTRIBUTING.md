# Contributing to AniRec

Thank you for your interest in contributing! This document explains the development workflow.

## Development Setup

```bash
git clone https://github.com/your-org/AniRec.git
cd AniRec

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

## Running Tests

```bash
# All tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Fast unit tests only (skip slow/integration)
pytest tests/ -m "not slow and not integration"
```

## Code Style

We use **ruff** for linting and formatting:

```bash
# Check
ruff check src/ tests/

# Auto-fix
ruff check --fix src/ tests/
```

## Security Scan

```bash
bandit -r src/ -ll
```

## Submitting a Pull Request

1. Fork the repository and create a branch: `git checkout -b feat/my-feature`
2. Make your changes, add tests for new functionality
3. Run `pytest tests/ -v` and ensure all tests pass
4. Run `ruff check src/ tests/` — no errors allowed
5. Update `CHANGELOG.md` with a summary of your change
6. Open a PR against `main` using the PR template

## Commit Message Convention

```
type(scope): short summary

Longer description if needed.

Fixes #123
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `ci`, `chore`

## Versioning

AniRec follows [Semantic Versioning](https://semver.org/). The version is set in `setup.py`.
