# Pre-commit Hooks Setup Guide

This project uses [pre-commit](https://pre-commit.com/) hooks to ensure consistent code quality, formatting, and security standards across all contributions.

## Quick Setup

1. **Install pre-commit** (included in dev dependencies):
   ```bash
   pip install homodyne-analysis[dev]
   # or
   pip install pre-commit
   ```

2. **Install the hooks**:
   ```bash
   pre-commit install
   ```

3. **That's it!** Hooks will now run automatically on every commit.

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
- **Flake8**: Style guide enforcement
- **Ruff**: Extremely fast Python linter with auto-fixes
- **MyPy**: Static type checking (excluding tests)

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
2. **Stage the fixes**: `git add .`
3. **Commit again**: The hooks should pass now

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

### Recommended Development Flow
1. Make your changes
2. Run tests: `pytest homodyne/tests/`
3. Stage files: `git add .`
4. Commit: `git commit -m "Your message"`
   - Hooks run automatically and may modify files
   - If files are modified, stage and commit again
5. Push: `git push`

### CI/CD Integration
Pre-commit hooks run in GitHub Actions to ensure code quality standards are maintained across all contributions.

## Benefits

- **Consistent formatting** across all contributors
- **Early error detection** before code review
- **Security scanning** to catch vulnerabilities
- **Reduced review time** with automated quality checks
- **Professional code standards** maintained automatically

## Support

For issues with pre-commit setup:
1. Check the [official pre-commit documentation](https://pre-commit.com/)
2. Review hook-specific documentation for individual tools
3. Open an issue in the project repository

---

**Note**: Pre-commit hooks are designed to help maintain code quality while being minimally intrusive to the development workflow. Most issues are auto-fixed, requiring only re-staging and committing.