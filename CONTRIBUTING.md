# Contributing to Homodyne Scattering Analysis Package

We welcome contributions to the Homodyne Scattering Analysis Package! This document provides guidelines for contributing to the project.

## Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/homodyne-analysis.git
   cd homodyne-analysis
   ```
3. **Create a development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -e .[dev]
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
5. **Make your changes** and **add tests**
6. **Run the test suite**:
   ```bash
   python homodyne/run_tests.py
   ```
7. **Submit a pull request**

## Development Setup

### Required Dependencies

```bash
pip install -e .[dev]
```

This installs all development dependencies including:
- pytest for testing
- black for code formatting
- flake8 for linting
- mypy for type checking
- sphinx for documentation

### Optional Dependencies

For full functionality testing:
```bash
pip install -e .[all]  # Includes MCMC dependencies
```

## Code Style

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Imports**: Use absolute imports when possible
- **Type hints**: Required for all new public functions
- **Docstrings**: NumPy style for all public functions

### Formatting

We use [Black](https://black.readthedocs.io/) for code formatting:

```bash
black homodyne/
```

### Linting

We use [flake8](https://flake8.pycqa.org/) for linting:

```bash
flake8 homodyne/
```

### Type Checking

We use [mypy](http://mypy-lang.org/) for static type checking:

```bash
mypy homodyne/
```

## Testing

### Running Tests

```bash
# Run all tests
python homodyne/run_tests.py

# Run fast tests only (exclude slow integration tests)
python homodyne/run_tests.py --fast

# Run with coverage
python homodyne/run_tests.py --coverage

# Run tests in parallel
python homodyne/run_tests.py --parallel 4

# Run specific tests
python homodyne/run_tests.py -k "test_config"
```

### Writing Tests

- Add tests for all new functionality
- Use descriptive test names: `test_static_isotropic_mode_parameter_validation`
- Place tests in the appropriate `homodyne/tests/test_*.py` file
- Use pytest fixtures for common setup
- Mark slow tests with `@pytest.mark.slow`
- Mark tests requiring MCMC with `@pytest.mark.mcmc`

Example test:
```python
import pytest
from homodyne import ConfigManager

def test_config_manager_validates_parameters():
    \"\"\"Test that ConfigManager properly validates parameter bounds.\"\"\"
    config = ConfigManager()
    # Test implementation here
    assert config.validate_parameters() is True
```

## Documentation

### Building Documentation

```bash
cd sphinx_docs
make html
```

The documentation will be built in `sphinx_docs/_build/html/`.

### Writing Documentation

- Use NumPy-style docstrings for all public functions
- Include examples in docstrings when helpful
- Update README.md for user-facing changes
- Add new pages to `sphinx_docs/` for major features

### Docstring Format

```python
def analyze_correlation_data(data: np.ndarray, config: dict) -> dict:
    """
    Analyze correlation data using specified configuration.
    
    Parameters
    ----------
    data : np.ndarray
        Experimental correlation data with shape (n_angles, n_times, n_times)
    config : dict
        Analysis configuration parameters
        
    Returns
    -------
    dict
        Analysis results including fitted parameters and statistics
        
    Examples
    --------
    >>> data = load_experimental_data("experiment.h5")
    >>> config = {"static_mode": True, "static_submode": "isotropic"}
    >>> results = analyze_correlation_data(data, config)
    """
```

## Pull Request Process

1. **Create a descriptive PR title**:
   - ✅ "Add support for custom angle filtering ranges"
   - ❌ "Fix bug"

2. **Write a clear description**:
   - What changes were made?
   - Why were they needed?
   - Any breaking changes?
   - Related issues?

3. **Ensure all checks pass**:
   - Tests pass
   - Code is formatted with Black
   - No linting errors
   - Documentation builds successfully

4. **Request review** from maintainers

5. **Address feedback** promptly

### PR Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)  
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] All new and existing tests pass

## Checklist
- [ ] My code follows the code style of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
```

## Issue Reporting

When reporting issues:

1. **Use the issue templates** when available
2. **Search existing issues** first
3. **Provide minimal reproducible examples**
4. **Include environment information**:
   - Python version
   - Package versions (`pip list`)
   - Operating system
   - Error messages and stack traces

## Code Review Guidelines

### For Contributors
- Keep PRs focused and reasonably sized
- Respond to review comments promptly
- Be open to feedback and suggestions

### For Reviewers
- Be constructive and specific in feedback
- Explain the "why" behind suggestions
- Recognize good contributions
- Focus on code quality, not personal style preferences

## Release Process

Releases are managed by maintainers:

1. Update version numbers in `homodyne/__init__.py` and `pyproject.toml`
2. Update `CHANGELOG.md` with release notes
3. Create and push git tag
4. GitHub Actions will automatically build and deploy

## Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Open a GitHub Issue
- **Security issues**: Email maintainers directly

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

## Recognition

Contributors are recognized in:
- GitHub contributors page
- CHANGELOG.md for significant contributions
- Package acknowledgments for major features

Thank you for contributing to the Homodyne Scattering Analysis Package!
