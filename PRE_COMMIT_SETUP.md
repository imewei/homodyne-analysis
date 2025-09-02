# Pre-commit Hooks Setup Guide

This project uses [pre-commit](https://pre-commit.com/) hooks to ensure consistent code
quality, formatting, and security standards across all contributions.

*Updated: 2025-09-01 - Enhanced testing framework integration and improved code quality
standards*

## Quick Setup (Unified System)

1. **Install with development dependencies**:

   ```bash
   pip install homodyne-analysis[dev]

   # Setup unified development environment
   homodyne-post-install --shell zsh --advanced
   ```

1. **Install pre-commit hooks**:

   ```bash
   pre-commit install
   ```

1. **Validate installation**:

   ```bash
   # Test unified system + pre-commit setup
   homodyne-validate
   pre-commit run --all-files
   ```

**That's it!** Hooks will now run automatically on every commit, and you have the full
unified development environment.

## Manual Usage

### Run on All Files

```bash
# Run all hooks on all files
pre-commit run --all-files

# Run a specific hook on all files
pre-commit run black --all-files
pre-commit run ruff --all-files
```

### Run on Staged Files Only

```bash
# Run hooks on currently staged files
pre-commit run

# Run specific hook on staged files
pre-commit run flake8
```

## Configured Hooks

### Code Formatting

- **Black**: Python code formatter (88 character line length)
- **isort**: Import statement sorting (black profile)
- **Ruff Format**: Fast Python formatter written in Rust

### Code Quality & Linting

- **Ruff**: Extremely fast Python linter with auto-fixes (✅ Active)
- **MyPy**: Static type checking (excluding tests) (⚠️ Improved stubs)
- ~~**Flake8**: Style guide enforcement~~ (Replaced by Ruff for better performance)

### Security

- **Bandit**: Security vulnerability scanner
  - Generates `bandit_report.json`
  - Skips common false positives in scientific code

### File Quality

- **Pre-commit hooks**: Built-in file quality checks
  - Trailing whitespace removal
  - End-of-file fixing
  - YAML/JSON/TOML validation
  - Merge conflict detection
  - Large file detection (max 1MB)

### Documentation

- **mdformat**: Markdown formatter (88 character wrap)
- **Prettier**: YAML and JSON formatting

### Jupyter Notebooks

- **nbqa-black**: Black formatting for notebooks
- **nbqa-isort**: Import sorting for notebooks

## Hook Configuration

All hooks are configured in `.pre-commit-config.yaml` with project-specific settings:

- **Line length**: 88 characters (Black standard)
- **Import profile**: Black-compatible
- **Security level**: Medium and above
- **Type checking**: Enabled with scientific dependencies
- **Exclusions**: Tests, build directories, generated files

## Bypassing Hooks

### Skip All Hooks (Emergency Use Only)

```bash
git commit --no-verify -m "Emergency commit message"
```

### Skip Specific Hooks

```bash
# Set environment variable to skip specific hooks
SKIP=mypy,bandit git commit -m "Skip type checking and security scan"
```

## Troubleshooting

### Hook Failures

If a hook fails:

1. **Review the output** - hooks often auto-fix issues
1. **Stage the fixes**: `git add .`
1. **Commit again**: The hooks should pass now

### Common Issues

**Black/Ruff formatting conflicts:**

```bash
# Run both formatters to resolve conflicts
pre-commit run black --all-files
pre-commit run ruff-format --all-files
```

**MyPy type checking errors:**

```bash
# Fix type issues or add type ignore comments
# MyPy excludes tests by default
```

**Bandit security warnings:**

```bash
# Review security warnings in bandit_report.json
# Add # nosec comments for false positives
```

### Updating Hooks

```bash
# Update hook versions to latest
pre-commit autoupdate

# Reinstall hooks after updates
pre-commit install
```

## Integration with Development Workflow

### Recommended Development Flow (Unified System)

1. **Setup environment**: `homodyne-post-install --shell zsh --advanced`
1. **Validate system**: `homodyne-validate`
1. **Make your changes**
1. **Run tests**:
   ```bash
   # Fast tests only (recommended for development)
   pytest homodyne/tests/ -m "fast" -x --tb=line -q

   # Unit tests with coverage
   pytest homodyne/tests/unit/ -v --cov=homodyne --cov-report=term-missing

   # All tests excluding slow/integration/mcmc
   pytest homodyne/tests/ -m "not slow and not integration and not mcmc" -x --tb=line -q
   ```
1. **Stage files**: `git add .`
1. **Commit**: `git commit -m "Your message"`
   - Pre-commit hooks run automatically and may modify files
   - If files are modified, stage and commit again
1. **Push**: `git push`

**Advanced development commands**:

```bash
# Use unified system commands for development
homodyne-validate --test completion     # Test shell completion
homodyne-validate --test gpu            # Test GPU setup
homodyne-gpu-optimize --benchmark       # Optimize development environment

# Testing framework with markers (v0.7.2)
pytest homodyne/tests/ -m "ci"          # Run CI-suitable tests
pytest homodyne/tests/ -m "unit"        # Run unit tests only
pytest homodyne/tests/ -m "system"      # Run system tests only
pytest homodyne/tests/ -m "fast"        # Run fast tests only
pytest homodyne/tests/ -m "regression"  # Run regression tests
```

### CI/CD Integration

Pre-commit hooks run in GitHub Actions to ensure code quality standards are maintained
across all contributions.

## Benefits

- **Consistent formatting** across all contributors
- **Early error detection** before code review
- **Security scanning** to catch vulnerabilities
- **Reduced review time** with automated quality checks
- **Professional code standards** maintained automatically

## Support

For issues with pre-commit setup:

1. Check the [official pre-commit documentation](https://pre-commit.com/)
1. Review hook-specific documentation for individual tools
1. Open an issue in the project repository

______________________________________________________________________

**Note**: Pre-commit hooks are designed to help maintain code quality while being
minimally intrusive to the development workflow. Most issues are auto-fixed, requiring
only re-staging and committing.
