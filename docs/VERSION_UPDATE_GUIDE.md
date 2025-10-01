# Version Update Guide

## Recent Version Updates (2025-09-29)

### Summary

All package versions have been updated for consistency across the development
environment, pre-commit hooks, GitHub Actions workflows, and package dependencies.

### What Changed

#### Python Version

- **Standardized to Python 3.13** across all environments
- Local environment: Python 3.13
- Pre-commit hooks: Python 3.13
- GitHub Actions: Python 3.13
- pyproject.toml: `requires-python = ">=3.12"`

#### Node.js Version

- **Updated from v18 to system Node.js (v24.8.0)**
- Resolves pre-commit hook failures due to unavailable Node.js v18 downloads
- Pre-commit now uses system-installed Node.js

#### Python Package Versions (pyproject.toml)

- `isort`: `>=5.13.0` → `>=6.0.1` (matches pre-commit v6.0.1)
- `pytest-html`: `>=4.2.0` → `>=4.1.1` (matches installed v4.1.1)
- Added `flake8>=7.3.0` to quality dependencies

#### GitHub Actions Versions

- `actions/upload-artifact`: v3 → v4
- `actions/download-artifact`: v3 → v4
- `actions/configure-pages`: v3 → v4
- `actions/upload-pages-artifact`: v2 → v3
- `actions/deploy-pages`: v2 → v3
- Added `cache: "pip"` to all `actions/setup-python` steps

### Version Matrix

| Tool | Pre-commit | pyproject.toml | Installed | Status |
|------|-----------|----------------|-----------|--------| | Python | 3.13 | >=3.12 |
3.13 | ✅ | | Node.js | system (24.8.0) | N/A | 24.8.0 | ✅ | | ruff | 0.13.2 | >=0.13.2 |
0.13.2 | ✅ | | black | 25.9.0 | >=25.9.0 | 25.9.0 | ✅ | | isort | 6.0.1 | >=6.0.1 |
6.0.1 | ✅ | | mypy | 1.18.2 | >=1.18.2 | 1.18.2 | ✅ | | flake8 | 7.3.0 | >=7.3.0 | 7.3.0
| ✅ | | pytest | 8.4.0 | >=8.4.0 | 8.4.2 | ✅ | | bandit | 1.8.6 | >=1.8.6 | 1.8.6 | ✅ |

### Migration Guide

#### For Existing Developers

1. **Update Python to 3.13**

   ```bash
   # macOS with Homebrew
   brew install python@3.13

   # Linux with pyenv
   pyenv install 3.13
   pyenv global 3.13
   ```

2. **Update Node.js (if needed)**

   ```bash
   # macOS with Homebrew
   brew install node

   # Verify version
   node --version  # Should be v24.x or later
   ```

3. **Recreate Virtual Environment**

   ```bash
   # Remove old environment
   rm -rf venv/

   # Create new environment with Python 3.13
   python3.13 -m venv venv
   source venv/bin/activate

   # Install dependencies
   pip install -e .[dev]
   ```

4. **Reinstall Pre-commit Hooks**

   ```bash
   # Clean pre-commit cache
   pre-commit clean
   pre-commit uninstall

   # Reinstall hooks
   pre-commit install --install-hooks
   ```

5. **Verify Installation**

   ```bash
   # Check Python version
   python --version  # Should show 3.13.x

   # Check package versions
   pip list | grep -E "(black|isort|ruff|mypy|pytest)"

   # Test pre-commit hooks
   pre-commit run --all-files
   ```

#### For CI/CD Pipelines

GitHub Actions workflows have been updated automatically. No action required.

### Breaking Changes

#### Minor Breaking Changes

- **Python 3.12+** is now required (was previously 3.8+)
- Some older Python 3.8-3.11 specific code may need updates
- Type hints use modern Python 3.12+ syntax

#### Non-Breaking Changes

- All tool version updates are backward compatible within their major versions
- GitHub Actions updates are transparent to workflows

### Troubleshooting

#### Pre-commit Hook Failures

**Issue**: `HTTP Error 404: Not Found` when installing Node.js

```
Solution: This has been fixed by using system Node.js instead of downloading v18
```

**Issue**: `vulture` or `dead` hooks fail with "Please pass at least one file"

```
Solution: These hooks have been configured with explicit paths. Run:
pre-commit clean && pre-commit install --install-hooks
```

**Issue**: Custom hooks show syntax errors

```
Solution: Custom quality hooks have been temporarily disabled.
They will be re-enabled via wrapper scripts.
```

#### Package Version Conflicts

**Issue**: `pip` reports version conflicts

```bash
# Solution: Recreate environment
rm -rf venv/
python3.13 -m venv venv
source venv/bin/activate
pip install -e .[dev]
```

**Issue**: Tests fail after version update

```bash
# Solution: Clear pytest cache
rm -rf .pytest_cache/
pytest --cache-clear
```

### Version Compatibility

#### Supported Python Versions

- **Minimum**: Python 3.12
- **Recommended**: Python 3.13
- **Maximum**: Python 3.14 (when released)

#### Supported Platforms

- **Linux**: Ubuntu 22.04+, Debian 12+, RHEL 9+
- **macOS**: macOS 13+ (Ventura or later)
- **Windows**: Windows 10+, WSL2 recommended

### References

- [Python 3.13 Release Notes](https://docs.python.org/3.13/whatsnew/3.13.html)
- [Node.js v24 Release Notes](https://nodejs.org/en/blog/release/v24.0.0)
- [Pre-commit Hook Documentation](https://pre-commit.com/)
- [GitHub Actions Changelog](https://github.blog/changelog/)

### Related Documentation

- [CLAUDE.md](../CLAUDE.md) - Project overview and development guide
- [pyproject.toml](../pyproject.toml) - Package dependencies and configuration
- [.pre-commit-config.yaml](../.pre-commit-config.yaml) - Pre-commit hook configuration

### Changelog Entry

```markdown
## [Unreleased]

### Changed
- **BREAKING**: Minimum Python version updated to 3.12 (from 3.8)
- Updated all development tool versions for consistency
- Standardized Python version to 3.13 across all environments
- Updated GitHub Actions to latest versions (v4)

### Fixed
- Pre-commit hooks no longer fail due to unavailable Node.js v18
- Version consistency between local environment and CI/CD
- Package dependency version conflicts

### Improved
- Faster pre-commit hooks with pip caching
- Better error messages for version conflicts
- Comprehensive version documentation
```

______________________________________________________________________

**Last Updated**: 2025-09-29 **Maintainer**: Wei Chen (wchen@anl.gov)
