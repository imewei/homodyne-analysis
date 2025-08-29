# Contributing to Homodyne Analysis Package

Thank you for your interest in contributing to the Homodyne Analysis Package! This guide
provides essential information for contributors.

## Development Setup

### Prerequisites

- Python 3.12 or higher
- Git

### Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/imewei/homodyne.git
   cd homodyne
   ```

1. **Install development dependencies:**

   ```bash
   make dev-install
   # or manually:
   pip install -e ".[all,dev,docs]"
   ```

1. **Run tests to verify setup:**

   ```bash
   make test
   # or manually:
   pytest -v
   ```

## Development Workflow

### Code Quality

We maintain high code quality standards:

- **Formatting:** Use Black for code formatting: `make format`
- **Linting:** Use flake8 and mypy: `make lint`
- **Testing:** Maintain test coverage and add tests for new features: `make test-all`

### Development Commands

We provide a comprehensive Makefile for common development tasks:

```bash
make help          # Show all available commands
make dev-install   # Install with all development dependencies
make test          # Run tests
make test-all      # Run tests with coverage
make lint          # Check code quality
make format        # Format code
make clean         # Clean all artifacts
make docs          # Build documentation
make build         # Build distribution packages
```

## Repository Cleanup

### Cleaning Development Artifacts

The repository should be kept clean of development artifacts. Use these commands:

#### Quick Clean

```bash
make clean
```

#### Manual Clean (Advanced)

```bash
# Remove all untracked files except data and results
git clean -xfd --exclude=data --exclude=homodyne_results

# Or clean specific artifact types:
make clean-pyc     # Remove Python bytecode
make clean-test    # Remove test artifacts  
make clean-build   # Remove build artifacts
```

#### Files That Should Be Ignored

The following files/directories are automatically ignored via `.gitignore`:

- **Bytecode:** `__pycache__/`, `*.py[cod]`
- **Build artifacts:** `build/`, `dist/`, `*.egg-info/`
- **Test artifacts:** `.pytest_cache/`, `.coverage*`, `htmlcov/`
- **IDE/Editor:** `.mypy_cache/`, `.idea/`, `.vscode/`, `.DS_Store`
- **Experimental data:** `data/`, `homodyne_results*/`, `my_config*.json`
- **Logs:** `run.log`, `*.tmp`, `*.bak`

### Before Committing

Always clean your working directory before committing:

```bash
make clean
git status  # Verify working tree is clean
```

## Testing

### Running Tests

```bash
# Basic test run
make test

# Full test suite with coverage
make test-all

# Run specific test categories
pytest -v -m "not slow"           # Skip slow tests
pytest -v -m "mcmc"              # Only MCMC tests
pytest -v -m "performance"       # Only performance tests
```

### Test Categories

Our tests are organized by markers:

- `slow`: Time-intensive tests
- `integration`: Integration tests
- `mcmc`: MCMC-specific tests
- `performance`: Performance benchmarks
- `regression`: Regression tests

### Writing Tests

- Place tests in `homodyne/tests/`
- Use descriptive test names
- Add appropriate markers
- Maintain or improve test coverage

## Documentation

### Building Documentation

```bash
make docs        # Build HTML documentation
make docs-serve  # Serve docs locally at http://localhost:8000
```

Documentation is built using Sphinx and hosted on Read the Docs.

## Submitting Changes

1. **Create a feature branch:**

   ```bash
   git checkout -b feature/your-feature-name
   ```

1. **Make your changes** following the development guidelines

1. **Clean up and test:**

   ```bash
   make clean
   make test-all
   make lint
   ```

1. **Commit your changes:**

   ```bash
   git add .
   git commit -m "Descriptive commit message"
   ```

1. **Push and create a pull request:**

   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style Guidelines

- **Python:** Follow PEP 8, enforced by Black formatter
- **Docstrings:** Use NumPy style docstrings
- **Type hints:** Use comprehensive type annotations
- **Imports:** Organize imports logically, use absolute imports
- **Comments:** Write clear, concise comments for complex logic

## Release Process

For maintainers releasing new versions:

```bash
# 1. Update version in homodyne/__init__.py
# 2. Clean and test
make clean
make test-all
make lint

# 3. Build and check distribution
make build
make check

# 4. Tag and release (maintainers only)
git tag v0.x.x
git push origin v0.x.x
make upload  # Upload to PyPI
```

## Getting Help

- **Issues:** Report bugs and request features on
  [GitHub Issues](https://github.com/imewei/homodyne/issues)
- **Documentation:** Check the [online documentation](https://homodyne.readthedocs.io/)
- **Email:** Contact the maintainers at wchen@anl.gov

## License

By contributing, you agree that your contributions will be licensed under the MIT
License.
