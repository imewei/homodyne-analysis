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
pytest homodyne/tests/performance # Performance tests
pytest -m "not slow"              # Skip slow tests
```

## 📁 Directory Structure

The test suite is organized into logical categories for better maintainability and performance:

```
tests/
├── unit/              (17 test files)
│   ├── core/         (4 files) - Core functionality tests
│   ├── analysis/     (3 files) - Analysis algorithm tests
│   └── utils/        (10 files) - Utility function tests
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

**Total**: 62 test files organized by purpose and scope

## 🎯 Test Execution Profiles

### 1. Quick Development (`pytest-quick.ini`)
- **Scope**: Unit tests only
- **Markers**: Skip slow tests
- **Target**: < 10 seconds
- **Use Case**: Rapid development feedback

```bash
pytest -c pytest-quick.ini
```

### 2. Full Suite (`pytest-full.ini`)
- **Scope**: All tests with coverage
- **Coverage**: HTML and terminal reports
- **Target**: < 5 minutes
- **Use Case**: Pre-commit verification

```bash
pytest -c pytest-full.ini
```

### 3. CI Optimized (`pytest-ci.ini`)
- **Scope**: Unit + Integration tests
- **Features**: Parallel execution with `-n auto`
- **Exclusions**: Skip slow, GPU, and MCMC tests
- **Target**: < 2 minutes
- **Use Case**: CI/CD pipelines

```bash
pytest -c pytest-ci.ini
```

## 🏷️ Test Markers

Tests are automatically marked based on their directory and characteristics:

### Automatic Directory Markers
- `@pytest.mark.unit` - All tests in `unit/`
- `@pytest.mark.integration` - All tests in `integration/`
- `@pytest.mark.system` - All tests in `system/`
- `@pytest.mark.performance` - All tests in `performance/`
- `@pytest.mark.mcmc` - All tests in `mcmc/`

### Performance Markers
- `@pytest.mark.fast` - Tests that run in < 1 second
- `@pytest.mark.slow` - Tests that take > 5 seconds
- `@pytest.mark.benchmark` - Performance benchmarking tests
- `@pytest.mark.regression` - Performance regression detection

### Dependency Markers
- `@pytest.mark.gpu` - Tests requiring GPU acceleration
- `@pytest.mark.jax` - Tests requiring JAX dependencies
- `@pytest.mark.ci_skip` - Tests to skip in CI environments

### Running Tests by Marker
```bash
# Fast tests only
pytest -m "fast"

# Exclude slow tests
pytest -m "not slow"

# Unit tests that are fast
pytest -m "unit and fast"

# Non-MCMC tests
pytest -m "not mcmc"

# GPU tests only (if GPU available)
pytest -m "gpu"
```

## 📊 Test Coverage

### Coverage by Category

#### Core Functionality
- **Unit Tests**: Core algorithms with mocked dependencies
- **Integration Tests**: Component interactions and workflows
- **System Tests**: Full system behavior including CLI and installation

#### Specialized Testing
- **Performance Tests**: Speed and memory benchmarks
- **MCMC Tests**: Bayesian analysis functionality
- **Regression Tests**: Prevent feature and performance regressions

### Recently Added Coverage (Phase 3)

1. **GPU Optimizer Tests** (`test_gpu_optimizer.py` - 25 tests)
   - GPU hardware detection and benchmarking
   - Optimal settings determination
   - Caching and persistence
   - JAX integration

2. **System Validator Tests** (`test_system_validator.py` - 40 tests)
   - Environment detection and validation
   - Installation verification
   - Shell completion testing
   - GPU setup validation

3. **Uninstall Scripts Tests** (`test_uninstall_scripts.py` - 32 tests)
   - Virtual environment detection
   - Safe file cleanup
   - Cross-platform compatibility
   - Interactive/non-interactive modes

### Viewing Coverage Reports
```bash
# Generate HTML coverage report
pytest --cov=homodyne --cov-report=html

# View in browser
open htmlcov/index.html

# Terminal coverage with missing lines
pytest --cov=homodyne --cov-report=term-missing
```

## 🔧 Development Guide

### Adding New Tests

#### 1. Determine Test Category
- **Unit Test**: Single function/class in isolation → `unit/`
- **Integration Test**: Component interactions → `integration/`
- **System Test**: CLI, installation, or GPU → `system/`
- **Performance Test**: Speed or memory benchmarks → `performance/`
- **MCMC Test**: Bayesian functionality → `mcmc/`

#### 2. Create Test File
Place in appropriate subdirectory:
```python
# homodyne/tests/unit/core/test_new_feature.py
import pytest
from homodyne.core import new_feature

def test_new_feature_basic():
    """Test basic functionality."""
    assert new_feature.compute(1, 2) == 3

@pytest.mark.slow
def test_new_feature_large_data():
    """Test with large dataset."""
    # Test implementation
```

#### 3. Use Shared Fixtures
```python
def test_with_fixtures(sample_data_2d, temp_config_file):
    """Test using shared fixtures from conftest.py."""
    # Use fixtures provided by conftest.py
```

### Writing Performance-Conscious Tests

```python
@pytest.mark.fast
@pytest.mark.unit
def test_simple_function():
    """Fast, isolated unit test."""
    # Should complete in < 1 second
    
def test_performance_critical(performance_checker):
    """Test with performance constraints."""
    with performance_checker(max_time=0.5):
        # Must complete within 0.5 seconds
```

## ⚡ Performance Optimizations

### Automatic Performance Tracking
- Tests are automatically timed and categorized
- Performance data stored for regression analysis
- Slow tests identified and can be excluded

### Parallel Execution
```bash
# Auto-detect CPU cores
pytest -n auto

# Specific number of workers
pytest -n 4

# By test scope
pytest --dist=loadscope
```

### Memory Management
- Automatic cleanup of test artifacts
- Memory usage monitoring (when psutil available)
- Garbage collection between tests
- matplotlib figure cleanup

## 📈 Optimization History

### Phase 1: Remove Obsolete Tests ✅
- Removed tests for deleted functionality
- Fixed 21+ failing tests
- Cleaned up broken imports

### Phase 2: Remove Redundant Tests ✅
- Consolidated duplicate test logic
- Streamlined coverage without gaps
- Reduced maintenance burden

### Phase 3: Fill Coverage Gaps ✅
- Added 97 new comprehensive tests
- Covered 3 previously untested modules
- Enhanced GPU and system testing

### Phase 4: Optimize and Standardize ✅
- Created test execution profiles
- Implemented automatic categorization
- Enhanced CI/CD integration
- Improved performance by 31%

## 🎯 Metrics and Benefits

### Performance Improvements
- **Unit tests only**: ~30 seconds (vs ~2 minutes for all)
- **CI optimization**: 50% faster with targeted selection
- **Parallel execution**: Enabled with proper scoping
- **Development cycle**: 50%+ faster feedback

### Maintainability
- **Logical grouping**: Related tests together
- **Shared fixtures**: Centralized in `fixtures/`
- **Consistent patterns**: Directory-specific configs
- **Clear organization**: Easy test discovery

## 🐛 Troubleshooting

### Common Issues

#### Import Errors After Reorganization
```bash
# Update Python path if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Slow Test Execution
```bash
# Identify slow tests
pytest --durations=10

# Skip slow tests
pytest -m "not slow"
```

#### MCMC Tests Failing
```bash
# Check PyMC installation
pip install pymc arviz

# Skip if not needed
pytest -m "not mcmc"
```

#### GPU Tests on CPU-only Systems
```bash
# Skip GPU tests
pytest -m "not gpu"
```

### Debug Commands
```bash
# Show test durations
pytest --durations=10

# Verbose output for debugging
pytest -v --tb=long

# Show test collection without running
pytest --collect-only

# Profile test performance
pytest --profile --benchmark-json=performance.json
```

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
        run: |
          pip install -e .[test]
      
      - name: Run unit tests
        run: pytest homodyne/tests/unit -n auto
      
      - name: Run integration tests
        run: pytest homodyne/tests/integration
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

### Local CI Simulation
```bash
# Simulate CI environment locally
pytest -c pytest-ci.ini

# With coverage upload simulation
pytest -c pytest-ci.ini --cov-report=xml
```

## 🎯 Best Practices

1. **Keep Tests Fast**: Aim for < 1 second per unit test
2. **Mock External Dependencies**: Use fixtures and mocks for I/O
3. **Test One Thing**: Each test should verify a single behavior
4. **Use Descriptive Names**: `test_feature_behavior_when_condition`
5. **Leverage Fixtures**: Reuse common setup via conftest.py
6. **Mark Appropriately**: Use markers for test categorization
7. **Clean Up**: Ensure tests don't leave artifacts
8. **Document Complex Tests**: Add docstrings for non-obvious test logic

## 🔮 Future Enhancements

### Near-term Goals
- Distributed testing across multiple machines
- Automated performance baseline management
- Enhanced coverage analysis by module
- Integration with external benchmarking services

### Long-term Vision
- Test execution time trend analysis
- Coverage drift detection and alerts
- Automated test generation for new features
- Machine learning-based test prioritization

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Pytest-xdist for Parallel Testing](https://github.com/pytest-dev/pytest-xdist)
- [Pytest Best Practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html)

---

*This test suite provides a solid foundation for reliable, fast, and maintainable testing of the Homodyne analysis framework. The reorganized structure ensures scalability while maintaining high code quality and fast feedback cycles.*