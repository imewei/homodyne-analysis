# GitHub Actions Workflows

This directory contains GitHub Actions workflows for the homodyne repository.

## 🚀 Current Status

**All workflows are currently disabled** and moved to `workflows-disabled/` for
reference. The project relies on:

- **ReadTheDocs**: Automatic documentation deployment from `main` branch
- **Local Development**: Pre-commit hooks and local testing tools
- **Manual Quality Assurance**: Developer-driven code quality checks

## 📁 Disabled Workflows

Comprehensive CI/CD workflows have been temporarily disabled but preserved in
`workflows-disabled/`:

- `ci.yml` - Full test suite, quality checks, and build validation
- `code-quality.yml` - Advanced code analysis and dependency scanning
- `performance.yml` - Performance testing and benchmarking
- `metrics.yml` - Code metrics and statistics tracking
- `release.yml` - Automated release management

## 🔄 Re-enabling Workflows

To re-enable CI/CD workflows:

1. Move desired workflow files from `workflows-disabled/` to `workflows/`
1. Update dependency installation commands to exclude MCMC packages
1. Review and update Python version matrix (currently supports 3.12+)
1. Verify all referenced dependency groups exist in `pyproject.toml`

## 📖 ReadTheDocs Configuration

Documentation is automatically built and deployed via ReadTheDocs:

- **Configuration**: `.readthedocs.yaml`
- **Triggers**: Automatic on push to `main` branch
- **Live docs**: https://homodyne.readthedocs.io/
- **Build process**: `cd docs && make clean && make html`

## 🧹 MCMC Cleanup Notes

The disabled workflows have been updated to reflect MCMC removal:

- Removed references to PyMC, ArviZ, corner, and PyTensor dependencies
- Updated installation commands to exclude MCMC-related extras
- Test matrices focus on core functionality (classical + robust optimization)
- Documentation builds no longer attempt to import MCMC modules

## 🔧 Local Development

With workflows disabled, use these local commands:

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
