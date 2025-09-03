# Homodyne Test Suite Documentation

## 📚 Overview

The Homodyne test suite provides comprehensive validation of XPCS analysis components through a well-organized, hierarchical structure optimized for maintainability, performance, and developer experience.

## 🚀 Quick Start

### Running Tests

```bash
# Quick development tests (< 10 seconds)
pytest -c pytest-quick.ini

# Full test suite with coverage
pytest -c pytest-full.ini

# CI-optimized testing
pytest -c pytest-ci.ini

# Specific test categories
pytest homodyne/tests/unit        # Unit tests only
pytest homodyne/tests/integration # Integration tests
pytest -m "not slow"              # Skip slow tests
```

## 📁 Directory Structure

```
tests/
├── unit/              (20 files) - Isolated component testing
│   ├── core/         (4 files) - Core functionality tests
│   ├── analysis/     (3 files) - Analysis algorithm tests
│   ├── optimization/ (6 files) - Optimization algorithm tests
│   │   ├── test_classical.py - Classical optimization methods
│   │   ├── test_robust.py - Robust optimization methods  
│   │   ├── test_mcmc.py - Isolated CPU MCMC backend tests (updated)
│   │   ├── test_mcmc_gpu.py - Isolated GPU MCMC backend tests (updated)
│   │   ├── test_mcmc_cross_validation.py - Backend isolation validation (new)
│   │   └── __init__.py
│   └── utils/        (7 files) - Utility function tests
├── integration/       (8 files) - Component integration tests
├── system/           (16 files) - System-level tests
│   ├── cli/         (3 files) - Command-line interface tests
│   ├── gpu/         (3 files) - GPU functionality tests
│   ├── installation/ (4 files) - Installation and setup tests
│   └── advanced/    (1 file) - Advanced feature tests
├── performance/      (5 files) - Performance and benchmark tests
├── mcmc/            (14 files) - MCMC-specific tests
├── regression/      (2 files) - Regression tests
└── fixtures/        - Shared test fixtures and utilities
```

**Total**: 65 test files organized by purpose and scope

## 🎯 Test Execution Profiles

### Quick Development (`pytest-quick.ini`)
- **Scope**: Unit tests only, skip slow tests
- **Target**: < 10 seconds
- **Use Case**: Rapid development feedback

### Full Suite (`pytest-full.ini`)  
- **Scope**: All tests with coverage reports
- **Target**: < 5 minutes
- **Use Case**: Pre-commit verification

### CI Optimized (`pytest-ci.ini`)
- **Scope**: Unit + Integration tests
- **Features**: Parallel execution, skip slow/GPU/MCMC
- **Target**: < 2 minutes
- **Use Case**: CI/CD pipelines

## 🏷️ Test Markers

### Automatic Markers
- `@pytest.mark.unit` - All tests in `unit/`
- `@pytest.mark.integration` - All tests in `integration/`
- `@pytest.mark.mcmc` - All tests in `mcmc/`
- `@pytest.mark.slow` - Tests taking > 5 seconds
- `@pytest.mark.gpu` - Tests requiring GPU acceleration

### Usage Examples
```bash
pytest -m "fast"                 # Fast tests only
pytest -m "not slow"             # Exclude slow tests  
pytest -m "unit and fast"        # Fast unit tests
pytest -m "not mcmc"             # Non-MCMC tests
```

## 🔬 Isolated MCMC Backend Testing Framework

### Core Test Files

#### **Isolated GPU MCMC Tests** (`test_mcmc_gpu.py` - 50+ tests)
- **Backend Isolation**: Verifies complete separation from PyMC/PyTensor
- **Environment Setup**: JAX/NumPyro availability, GPU detection, CPU fallback within JAX ecosystem
- **Sampler Initialization**: Configuration validation, performance features
- **Model Creation**: NumPyro model construction, prior distributions
- **API Compatibility**: Interface consistency with isolated CPU backend

#### **Isolated CPU MCMC Tests** (`test_mcmc.py` - 25+ tests)
- **Backend Isolation**: Verifies complete separation from JAX/NumPyro
- **Pure PyMC Implementation**: CPU-only PyTensor configuration
- **Data Saving Updates**: Updated for isolated backend architecture
- **Cross-Platform Compatibility**: Linux, macOS, Windows validation

#### **Backend Isolation Validation** (`test_mcmc_cross_validation.py` - 15+ classes)
- **Import Isolation**: Verifies no cross-contamination between backends
- **Environment Configuration**: Tests isolated environment variable setup
- **API Compatibility**: Constructor signatures, attributes, method interfaces
- **Error Handling Consistency**: Consistent behavior across isolated backends

### Running MCMC Tests

```bash
# All isolated MCMC backend tests
pytest homodyne/tests/unit/optimization/ -v

# Isolated GPU backend tests only
pytest homodyne/tests/unit/optimization/test_mcmc_gpu.py -v

# Isolated CPU backend tests only
pytest homodyne/tests/unit/optimization/test_mcmc.py -v

# Backend isolation validation
pytest homodyne/tests/unit/optimization/test_mcmc_cross_validation.py -v

# Skip slow benchmarks
pytest homodyne/tests/unit/optimization/ -k "not slow" -q
```

### MCMC Test Dependencies

Tests automatically skip when dependencies are unavailable:

```bash
SKIPPED [1] test_mcmc_gpu.py:152: MCMC GPU module not available
SKIPPED [1] test_mcmc.py:592: MCMC module not available
```

### MCMC Testing Best Practices

1. **Dependency Mocking**: Use proper JAX/PyMC mocking
2. **Performance Marking**: Mark slow tests with `@pytest.mark.slow`
3. **Cross-Platform Testing**: Ensure tests work with/without GPU hardware

## 📊 Test Coverage

### Core Functionality
- **Unit Tests**: Isolated algorithms with mocked dependencies
- **Integration Tests**: Component interactions and workflows
- **System Tests**: Full system behavior including CLI

### Recent Coverage Additions

#### **Isolated MCMC Backend Testing Framework (Phase 5)**
- Added comprehensive isolated GPU MCMC backend testing (50+ tests)
- Updated isolated CPU MCMC backend tests for architecture changes  
- Backend isolation validation between completely separated CPU/GPU implementations
- Dependency-aware test skipping for isolated PyMC and JAX/NumPyro backends

#### **System Components (Phase 3)**
- **GPU Optimizer Tests** (25 tests): Hardware detection, optimization
- **System Validator Tests** (40 tests): Environment validation
- **Uninstall Scripts Tests** (32 tests): Safe cleanup procedures

### Coverage Reports
```bash
# Generate HTML coverage report
pytest --cov=homodyne --cov-report=html
open htmlcov/index.html

# Terminal coverage with missing lines
pytest --cov=homodyne --cov-report=term-missing
```

## 🔧 Development Guide

### Adding New Tests

1. **Determine Category**: Unit → `unit/`, Integration → `integration/`, MCMC → `mcmc/`
2. **Use Shared Fixtures**: Leverage fixtures from `conftest.py`
3. **Mark Appropriately**: Use markers for categorization

```python
# Example test file
import pytest
from homodyne.core import new_feature

def test_new_feature_basic():
    """Test basic functionality."""
    assert new_feature.compute(1, 2) == 3

@pytest.mark.slow
def test_new_feature_large_data():
    """Test with large dataset."""
    # Implementation
```

### Performance-Conscious Testing

```python
@pytest.mark.fast
def test_simple_function():
    """Fast unit test - should complete < 1 second."""
    pass

def test_with_performance_constraint(performance_checker):
    """Test with performance requirements."""
    with performance_checker(max_time=0.5):
        # Must complete within 0.5 seconds
        pass
```

## ⚡ Performance Optimizations

### Parallel Execution
```bash
pytest -n auto              # Auto-detect CPU cores
pytest -n 4                 # Specific worker count
pytest --dist=loadscope     # By test scope
```

### Memory Management
- Automatic cleanup of test artifacts
- Memory usage monitoring (when psutil available)
- Garbage collection between tests

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: pip install -e .[test]
      - name: Run tests
        run: pytest -c pytest-ci.ini

  mcmc-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        deps: 
          - name: "PyMC only"
            install: "pymc arviz"
          - name: "JAX only" 
            install: "jax numpyro"
          - name: "Full MCMC"
            install: "pymc arviz jax numpyro"
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e .[test]
          pip install ${{ matrix.deps.install }}
      - name: Run MCMC tests
        run: pytest homodyne/tests/unit/optimization/ -k "not slow" -v
```

## 🐛 Troubleshooting

### Common Issues

#### Import Errors
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Slow Test Execution
```bash
pytest --durations=10       # Identify slow tests
pytest -m "not slow"        # Skip slow tests
```

#### MCMC Tests Failing
```bash
pip install pymc arviz      # Install PyMC dependencies
pytest -m "not mcmc"        # Skip if not needed
```

#### GPU Tests on CPU Systems
```bash
pytest -m "not gpu"         # Skip GPU tests
```

### Debug Commands
```bash
pytest --durations=10       # Show test durations
pytest -v --tb=long         # Verbose with full tracebacks
pytest --collect-only       # Show test collection
```

## 📈 Optimization History

### Phase 1-2: Foundation ✅
- Removed obsolete tests, fixed failing tests
- Consolidated duplicate logic, streamlined coverage

### Phase 3: Coverage Expansion ✅  
- Added 97 new comprehensive tests
- Covered 3 previously untested modules
- Enhanced GPU and system testing

### Phase 4: Performance & Standards ✅
- Created test execution profiles
- Implemented automatic categorization
- Enhanced CI/CD integration (31% performance improvement)

### Phase 5: Isolated MCMC Backend Testing ✅
- Added comprehensive isolated GPU MCMC backend testing (50+ tests)
- Updated isolated CPU MCMC backend tests for architecture separation
- Backend isolation validation between completely separated implementations
- Enhanced dependency-aware test skipping for isolated backends

## 🎯 Best Practices

1. **Keep Tests Fast**: Aim for < 1 second per unit test
2. **Mock External Dependencies**: Use fixtures and mocks for I/O
3. **Test One Thing**: Each test should verify a single behavior
4. **Use Descriptive Names**: `test_feature_behavior_when_condition`
5. **Leverage Fixtures**: Reuse common setup via conftest.py
6. **Mark Appropriately**: Use markers for test categorization
7. **Clean Up**: Ensure tests don't leave artifacts
8. **Document Complex Tests**: Add docstrings for non-obvious logic

## 📚 Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Pytest-xdist for Parallel Testing](https://github.com/pytest-dev/pytest-xdist)
- [Pytest Best Practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html)

---

*This test suite provides a solid foundation for reliable, fast, and maintainable testing of the Homodyne analysis framework. The comprehensive isolated MCMC backend testing framework ensures robust validation of completely separated CPU and GPU Bayesian analysis implementations, preventing PyTensor/JAX conflicts while maintaining full functionality.*