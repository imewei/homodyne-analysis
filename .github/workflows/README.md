# GitHub Actions Workflows

This directory contains GitHub Actions workflows for the homodyne repository.

## 🚀 Current Status

**Updated workflows are now active** with configurations matching the current codebase:

- **ci.yml**: Core CI pipeline with tests, quality checks, and build validation
- **quality.yml**: Code quality checks with relaxed mypy for scientific computing
- **ReadTheDocs**: Automatic documentation deployment from `main` branch
- **Local Development**: Pre-commit hooks and local testing tools

## 📁 Disabled Workflows

Comprehensive CI/CD workflows have been temporarily disabled but preserved in
`workflows-disabled/`:

- `ci.yml` - Full test suite, quality checks, and build validation
- `code-quality.yml` - Advanced code analysis and dependency scanning
- `performance.yml` - Performance testing and benchmarking
- `metrics.yml` - Code metrics and statistics tracking
- `release.yml` - Automated release management

## 🔄 Active Workflows

Current active workflows:

**ci.yml** - Main CI pipeline:
- Multi-OS testing (Ubuntu, Windows, macOS)
- Python 3.12 and 3.13 support
- MCMC mocking for clean tests
- Package building and metadata validation
- Security scanning with Bandit and pip-audit

**quality.yml** - Code quality checks:
- Pre-commit hooks validation
- Relaxed MyPy type checking (scientific computing friendly)
- Code formatting (Black, isort, Ruff)
- Security auditing

## 📖 ReadTheDocs Configuration

Documentation is automatically built and deployed via ReadTheDocs:

- **Configuration**: `.readthedocs.yaml`
- **Triggers**: Automatic on push to `main` branch
- **Live docs**: https://homodyne.readthedocs.io/
- **Build process**: `cd docs && make clean && make html`

## 🧹 Post-MCMC Cleanup

The active workflows reflect complete MCMC removal:

- **No MCMC dependencies**: PyMC, ArviZ, corner, PyTensor, and JAX removed
- **Mock imports**: sys.modules mocking prevents MCMC import errors
- **Focus**: Classical (Nelder-Mead, Gurobi) and Robust optimization only
- **Relaxed MyPy**: Scientific computing friendly type checking
- **Clean tests**: All tests run with MCMC modules disabled

## 🔧 Local Development

Use these commands for local development and testing:

```bash
# Run tests with MCMC mocking (as used in Makefile)
make test

# Quality checks
make lint
make format
make type-check

# Documentation
cd docs && make html
```
