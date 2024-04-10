# Contributing to Time-Series Forecasting & Anomaly Lab

Thank you for considering contributing to this project!

## How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Run tests**: `pytest tests/`
5. **Run linters**: `ruff check src/ && black --check src/`
6. **Commit with conventional commits**: `git commit -m "feat: add amazing feature"`
7. **Push to your fork**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pre-commit install
```

## Code Style

- Follow PEP 8
- Use Black for formatting (line length: 100)
- Use Ruff for linting
- Add type hints where appropriate
- Write docstrings for all public functions

## Testing

- Write unit tests for new functions
- Add integration tests for new features
- Maintain test coverage above 80%
- Run `pytest tests/ -v --cov=src`

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions or fixes
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

## Pull Request Process

1. Update README.md with details of changes if needed
2. Update CHANGELOG.md under "Unreleased"
3. Ensure CI passes (linting, tests, Docker build)
4. Request review from maintainers
5. Squash commits before merging

## Questions?

Open an issue or contact the maintainer at dhieddine.barhoumi@gmail.com
