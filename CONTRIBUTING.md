# Contributing to Homodyne Analysis

We welcome contributions to the Homodyne Scattering Analysis Package! This document provides guidelines for contributing code, documentation, and reporting issues.

## 🚀 Quick Start for Contributors

### Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/homodyne.git
cd homodyne

# 2. Create a virtual environment
python -m venv homodyne-dev
source homodyne-dev/bin/activate  # Linux/macOS
# or: homodyne-dev\Scripts\activate  # Windows

# 3. Install in development mode with all dependencies
pip install -e ".[dev]"

# 4. Set up unified development environment
homodyne-post-install --shell zsh --gpu --advanced

# 5. Install pre-commit hooks
pre-commit install

# 6. Verify installation
homodyne-validate --quick
pytest -c pytest-quick.ini
```

### Development Workflow

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes and run tests
pytest -m "not slow" -x --tb=line -q

# Run code quality checks
make format     # Format code (black, isort, ruff)
make lint       # Run linting and type checks

# Commit with pre-commit hooks
git add .
git commit -m "feat: add your feature description"

# Push and create pull request
git push origin feature/your-feature-name
```

## 📋 Development Guidelines

### Code Style and Standards

We maintain high code quality through automated tools:

- **Black**: Code formatting (88-character line length)
- **isort**: Import organization  
- **Ruff**: Fast linting with auto-fixes
- **MyPy**: Type checking (gradual typing)
- **Bandit**: Security analysis for scientific code

```bash
# Format your code before committing
make format

# Check code quality
make lint

# Run security scan
bandit -r homodyne/
```

### Testing Requirements

All contributions must include comprehensive tests:

```bash
# Quick development tests (< 30 seconds)
pytest -m "fast" -x --tb=line -q

# Unit tests only
pytest -m "unit" 

# Skip slow tests during development
pytest -m "not slow" -x --tb=line -q

# Full test suite (CI equivalent)
pytest -c pytest-full.ini
```

**Test Categories:**
- `fast`: Quick tests (< 1 second each)
- `unit`: Isolated component tests
- `integration`: Multi-component interactions
- `mcmc`: Bayesian analysis tests (requires specialized setup)
- `gpu`: GPU acceleration tests (Linux + NVIDIA only)
- `performance`: Regression detection

### Performance Considerations

When making performance-related changes:

```bash
# Run performance benchmarks
pytest -m "performance" --benchmark-only

# Check for regressions
pytest homodyne/tests/performance/

# GPU optimization (Linux only)
homodyne-gpu-optimize --benchmark --report
```

## 🔧 Types of Contributions

### 1. Bug Fixes

**Process:**
1. Create an issue describing the bug with minimal reproduction case
2. Fork the repository and create a fix branch: `fix/issue-number-description`
3. Write tests that demonstrate the bug and verify the fix
4. Ensure all existing tests still pass
5. Submit a pull request with clear description

**Requirements:**
- Include regression test that fails before fix and passes after
- Update relevant docstrings if API behavior changes
- Add changelog entry in `CHANGELOG.md`

### 2. New Features

**Process:**
1. Open an issue to discuss the feature before implementation
2. Create feature branch: `feature/feature-name`
3. Implement with comprehensive tests and documentation
4. Update configuration templates if needed
5. Add examples to documentation

**Requirements:**
- Full test coverage for new functionality
- Updated API documentation with examples
- Performance impact assessment
- Backward compatibility considerations

### 3. Documentation Improvements

**Process:**
1. Documentation lives in `docs/` (Sphinx) and inline docstrings
2. Follow NumPy docstring style for consistency
3. Include working code examples
4. Test documentation builds locally

```bash
# Build documentation locally
cd docs/
make html
open _build/html/index.html  # View results
```

**Requirements:**
- Clear, concise writing
- Working code examples (tested with doctest)
- Cross-references to related functionality
- Screenshots for UI-related changes

### 4. Performance Improvements

**Process:**
1. Profile existing code to identify bottlenecks
2. Implement optimization with before/after benchmarks
3. Ensure numerical accuracy is maintained
4. Update performance baselines if significant improvement

**Requirements:**
- Benchmark results showing improvement
- Verification that scientific accuracy is preserved
- Memory usage impact assessment
- Regression tests for edge cases

## 🧪 Testing Guidelines

### Test Organization

```
homodyne/tests/
├── unit/           # Isolated component tests
├── integration/    # Multi-component interactions
├── system/         # CLI and full system tests
├── mcmc/          # Bayesian analysis tests
├── performance/   # Benchmarking and regression
└── fixtures/      # Shared test data and utilities
```

### Writing Tests

**Example Test Structure:**

```python
import pytest
from homodyne.analysis.core import HomodyneAnalysisCore
from homodyne.core.config import ConfigManager

class TestHomodyneAnalysisCore:
    """Test suite for core analysis functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Provide mock configuration for testing."""
        return ConfigManager.from_dict({
            "analysis_mode": "static_isotropic",
            "parameters": {...}
        })
    
    def test_classical_optimization_basic(self, mock_config):
        """Test basic classical optimization functionality."""
        analysis = HomodyneAnalysisCore(mock_config)
        results = analysis.optimize_classical()
        
        assert results["success"] is True
        assert "parameters" in results
        assert results["chi_squared"] > 0
    
    @pytest.mark.slow
    def test_mcmc_sampling_convergence(self, mock_config):
        """Test MCMC sampling convergence (marked as slow)."""
        # Implementation for comprehensive MCMC test
        pass
        
    @pytest.mark.performance
    def test_optimization_performance(self, benchmark):
        """Performance regression test."""
        # Use pytest-benchmark for timing
        pass
```

### Test Data and Fixtures

- Use `homodyne/tests/fixtures/` for shared test data
- Create realistic but minimal test cases
- Mock external dependencies and file I/O when possible
- Provide clear fixture documentation

## 📖 Documentation Standards

### Docstring Style (NumPy Format)

```python
def analyze_correlation_function(
    c2_data: np.ndarray,
    angles: np.ndarray,
    times: np.ndarray,
    method: str = "classical"
) -> Dict[str, Any]:
    """Analyze time-dependent correlation functions.
    
    Fits correlation data using specified optimization method to extract
    transport coefficients and their uncertainties.
    
    Parameters
    ----------
    c2_data : np.ndarray, shape (n_angles, n_times, n_times)
        Two-time correlation function data c₂(φ, t₁, t₂)
    angles : np.ndarray, shape (n_angles,)
        Scattering angles in degrees
    times : np.ndarray, shape (n_times,)
        Time points in seconds
    method : str, default="classical"
        Optimization method: "classical", "robust", "mcmc", or "all"
        
    Returns
    -------
    Dict[str, Any]
        Results dictionary containing:
        - parameters: Fitted parameter values with uncertainties
        - chi_squared: Goodness of fit metric
        - method_info: Method-specific analysis details
        
    Raises
    ------
    ValueError
        If input arrays have incompatible shapes
    OptimizationError
        If fitting fails to converge
        
    Examples
    --------
    >>> import numpy as np
    >>> from homodyne import analyze_correlation_function
    >>> 
    >>> # Generate synthetic data
    >>> angles = np.linspace(0, 90, 10)
    >>> times = np.logspace(-3, 1, 50)
    >>> c2_data = generate_synthetic_data(angles, times)
    >>>
    >>> # Analyze with classical optimization
    >>> results = analyze_correlation_function(c2_data, angles, times)
    >>> print(f"Diffusion coefficient: {results['parameters']['D0']:.2e}")
    
    References
    ----------
    .. [1] He et al. "Transport coefficient approach for characterizing 
           nonequilibrium dynamics in soft matter." PNAS 121.31 (2024).
    """
```

### API Documentation

- Document all public functions, classes, and methods
- Include parameter types, shapes for arrays
- Provide realistic usage examples
- Reference relevant scientific literature
- Cross-reference related functionality

## 🔄 Pull Request Process

### Before Submitting

1. **Run the full test suite:**
   ```bash
   pytest -c pytest-full.ini
   make lint
   ```

2. **Update documentation:**
   ```bash
   # Build docs to check for issues
   cd docs/ && make html
   ```

3. **Update changelog:**
   Add entry to `CHANGELOG.md` under "Unreleased" section

### Pull Request Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)  
- [ ] Breaking change (fix or feature that breaks existing functionality)
- [ ] Documentation update

## Testing
- [ ] Added tests that demonstrate fix/feature works
- [ ] All existing tests pass
- [ ] Performance impact assessed (if applicable)

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-reviewed code changes
- [ ] Updated documentation as needed
- [ ] Added changelog entry
- [ ] No merge conflicts
```

### Review Process

All submissions require review by project maintainers:

1. **Automated checks:** CI tests, linting, security scans
2. **Code review:** Functionality, style, documentation
3. **Testing verification:** Test coverage and quality
4. **Performance assessment:** Impact on existing functionality

## 🏗️ Architecture Guidelines

### Package Structure

```
homodyne/
├── core/               # Core computational kernels
│   ├── config.py      # Configuration management
│   ├── kernels.py     # Numba-accelerated calculations
│   └── io_utils.py    # Data input/output utilities
├── analysis/           # Main analysis engine
│   └── core.py        # HomodyneAnalysisCore class
├── optimization/       # Optimization methods
│   ├── classical.py   # Classical methods (Nelder-Mead, Gurobi)
│   ├── robust.py      # Robust methods (Wasserstein, Scenario)
│   ├── mcmc.py        # Bayesian MCMC (PyMC backend)
│   └── mcmc_gpu.py    # GPU-accelerated MCMC (NumPyro)
├── runtime/           # Runtime utilities
│   ├── gpu/           # GPU optimization tools
│   └── shell/         # Shell completion system
└── tests/             # Comprehensive test suite
```

### Design Patterns

1. **Lazy Loading:** Heavy dependencies imported only when needed
2. **Configuration-Driven:** All analysis controlled via JSON configs
3. **Modular Optimization:** Pluggable optimization backends
4. **Error Handling:** Graceful degradation for missing dependencies
5. **Performance:** Numba JIT compilation for computational kernels

## 🐛 Issue Reporting

### Bug Reports

**Template:**

```markdown
**Bug Description**
Clear description of the issue.

**Reproduction Steps**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What should happen.

**Actual Behavior**
What actually happens.

**Environment**
- Python version: 
- Package version: 
- Operating system:
- Key dependencies: homodyne-validate output

**Additional Context**
Any other relevant information.
```

### Feature Requests

**Template:**

```markdown
**Feature Summary**
Brief description of requested feature.

**Motivation**
Why is this feature needed? What problem does it solve?

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
Other approaches you've considered.

**Additional Context**
Examples, references, related issues.
```

## 📊 Performance Expectations

### Benchmarks and Baselines

We maintain performance baselines for regression detection:

```bash
# Run benchmarks and compare to baselines
pytest homodyne/tests/performance/ --benchmark-compare

# Update baselines after approved performance changes
pytest homodyne/tests/performance/ --benchmark-save=new_baseline
```

**Performance Targets:**
- Correlation function calculation: < 500μs per evaluation
- Classical optimization: < 2 minutes for typical datasets
- MCMC sampling: < 30 minutes for 4000 samples
- Memory usage: < 2GB for standard analyses

## 🔒 Security Considerations

We use automated security scanning but contributors should be aware of:

- No hardcoded credentials or secrets
- Validate all user inputs
- Use secure file handling practices
- Follow scientific computing security best practices

```bash
# Run security scan
bandit -r homodyne/
pip-audit
```

## 📋 Code Review Checklist

### For Contributors

- [ ] Code follows style guidelines (black, isort, ruff)
- [ ] Tests cover new functionality
- [ ] Documentation updated
- [ ] No performance regressions
- [ ] Backward compatibility maintained
- [ ] Security scan passes

### For Reviewers

- [ ] Functionality works as described
- [ ] Code is readable and well-documented
- [ ] Tests are comprehensive and meaningful
- [ ] No security issues
- [ ] Performance impact acceptable
- [ ] Design fits with overall architecture

## 🎉 Recognition

Contributors are recognized in:
- `AUTHORS.md` file
- Release notes for significant contributions  
- Package metadata for maintainers
- Documentation acknowledgments

Thank you for contributing to advancing X-ray photon correlation spectroscopy analysis!

---

## 📞 Getting Help

- **Documentation**: https://homodyne.readthedocs.io/
- **Issues**: https://github.com/imewei/homodyne/issues
- **Discussions**: GitHub Discussions tab
- **Email**: wchen@anl.gov (maintainers)

For quick questions, check existing issues and documentation first. For complex technical discussions, GitHub Discussions is preferred over email.