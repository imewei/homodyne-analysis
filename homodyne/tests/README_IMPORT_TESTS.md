# Import Verification and Startup Performance Test Infrastructure

This directory contains comprehensive test infrastructure for validating import usage and monitoring startup performance in the homodyne package.

## 📁 Files Overview

### Core Test Files

1. **`test_import_verification.py`**
   - Comprehensive import verification tests
   - Detects unused imports across the entire codebase
   - Validates import dependency chains
   - Checks for circular imports
   - Tests import structure consistency

2. **`test_startup_performance.py`**
   - Startup timing benchmarks and regression tests
   - Memory usage monitoring during imports
   - Lazy loading effectiveness tests
   - Conditional import performance validation
   - Progressive loading performance analysis

### Utility Scripts

3. **`import_analyzer.py`** *(executable)*
   - Standalone import analysis utility
   - Can be run independently for import cleanup
   - Generates automated cleanup scripts
   - Supports external tool integration (autoflake, unimport)

4. **`establish_baselines.py`** *(executable)*
   - Creates performance baselines for regression testing
   - Measures current import performance
   - Compares with historical baselines
   - Generates performance reports

## 🚀 Quick Start

### Run All Import Verification Tests
```bash
# Run import verification tests
python -m pytest homodyne/tests/test_import_verification.py -v

# Run only unused import detection
python -m pytest homodyne/tests/test_import_verification.py::TestImportVerification::test_no_unused_imports -v
```

### Run Startup Performance Tests
```bash
# Run all startup performance tests
python -m pytest homodyne/tests/test_startup_performance.py -m performance -v

# Run specific performance test
python -m pytest homodyne/tests/test_startup_performance.py::TestStartupPerformance::test_basic_import_performance -v
```

### Analyze Imports with Standalone Utility
```bash
# Quick analysis (check only)
python homodyne/tests/import_analyzer.py --check-only

# Detailed analysis with suggestions
python homodyne/tests/import_analyzer.py --verbose

# Generate cleanup script
python homodyne/tests/import_analyzer.py --cleanup-script cleanup_imports.py

# Export full analysis to JSON
python homodyne/tests/import_analyzer.py --output import_analysis.json
```

### Establish Performance Baselines
```bash
# Create initial baselines
python homodyne/tests/establish_baselines.py

# View baseline comparison
python homodyne/tests/establish_baselines.py
```

## 📊 Test Categories and Markers

### Import Verification Tests

- **`@pytest.mark.integration`**: Full import dependency validation
- **`@pytest.mark.performance`**: Import timing tests
- Tests for:
  - Unused imports detection
  - Broken import chains
  - Circular dependencies
  - Import structure consistency
  - Fast critical imports

### Startup Performance Tests

- **`@pytest.mark.performance`**: All performance benchmarks
- **`@pytest.mark.slow`**: Comprehensive performance suites
- Tests for:
  - Basic import performance (< 2s)
  - Warm import performance (< 10ms)
  - Progressive loading (no performance cliffs)
  - Lazy loading effectiveness (> 2x speedup)
  - Memory usage (< 100MB basic import)
  - Performance regression detection

## 🔧 Configuration

### Test Execution Options

```bash
# Skip slow tests
python -m pytest -m "not slow"

# Run only performance tests
python -m pytest -m performance

# Run with detailed timing
python -m pytest --durations=10

# Run specific import tests
python -m pytest -k "import" -v
```

### Performance Thresholds

Current performance expectations:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Basic import time | < 2.0s | `import homodyne` |
| Warm import time | < 0.01s | Already loaded modules |
| Memory usage | < 50MB | Basic import memory delta |
| Lazy loading overhead | < 5x | Accessing lazy objects vs basic import |
| Stage import time | < 3.0s | Each progressive loading stage |

### Memory Usage Limits

| Component | Memory Limit | Description |
|-----------|-------------|-------------|
| Basic import | 50MB | Memory delta for `import homodyne` |
| Core modules | 25MB | Individual core module imports |
| Total peak | 500MB | Maximum process memory during tests |

## 📈 Baseline Management

### Baseline Files

- **`startup_baselines.json`**: Performance baselines for regression testing
- Automatically created by `establish_baselines.py`
- Updated when significant performance changes are validated

### Baseline Structure

```json
{
  "metadata": {
    "created_at": "timestamp",
    "python_version": "3.12.0",
    "platform": "darwin"
  },
  "individual_modules": {
    "homodyne": {
      "cold_import": {"import_time": 0.45, "memory_delta_mb": 12.3},
      "warm_import": {"avg_time": 0.002, "iterations": 5}
    }
  },
  "progressive_loading": {...},
  "lazy_loading": {...},
  "summary": {
    "avg_import_time": 0.8,
    "max_import_time": 1.5,
    "total_modules_tested": 7
  }
}
```

### Updating Baselines

```bash
# Create new baselines (compares with existing)
python homodyne/tests/establish_baselines.py

# View current vs baseline comparison
python homodyne/tests/establish_baselines.py
```

## 🔍 Import Analysis Details

### Unused Import Detection

The import analyzer uses AST parsing to detect:

1. **Unused regular imports**: `import module` not referenced
2. **Unused from imports**: `from module import name` not used
3. **Unused aliases**: `import module as alias` where alias not used
4. **Context tracking**: Where imports are used (functions, classes, module level)

### Analysis Capabilities

- **Dependency chains**: Maps import relationships
- **Circular import detection**: Identifies problematic circular dependencies
- **Optimization suggestions**: Recommends import improvements
- **External tool integration**: Works with autoflake, unimport

### Cleanup Script Generation

```bash
# Generate automated cleanup script
python homodyne/tests/import_analyzer.py --cleanup-script cleanup.py

# IMPORTANT: Review the script before running!
# The script modifies source files
python cleanup.py  # Only after review
```

## 🎯 Integration with CI/CD

### GitHub Actions Integration

Add to your workflow:

```yaml
- name: Import Verification
  run: |
    python -m pytest homodyne/tests/test_import_verification.py -v

- name: Startup Performance Tests
  run: |
    python -m pytest homodyne/tests/test_startup_performance.py -m performance -v

- name: Check for Unused Imports
  run: |
    python homodyne/tests/import_analyzer.py --check-only
  # This will exit with code 1 if unused imports are found
```

### Pre-commit Integration

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: import-verification
      name: Import Verification
      entry: python -m pytest homodyne/tests/test_import_verification.py::TestImportVerification::test_no_unused_imports -x
      language: system
      pass_filenames: false
```

### Performance Regression Detection

```bash
# Run regression tests
python -m pytest homodyne/tests/test_startup_performance.py::TestStartupPerformance::test_startup_regression_detection -v

# Update baselines after validated performance changes
python homodyne/tests/establish_baselines.py
```

## 🐛 Troubleshooting

### Common Issues

1. **Import test failures**:
   - Check for recently added unused imports
   - Verify all imports are actually needed
   - Consider if imports are used in ways the AST parser doesn't detect

2. **Performance test failures**:
   - Run on a quiet system (no background processes)
   - Check if new dependencies were added
   - Verify test environment is consistent

3. **False positives in unused imports**:
   - Some imports may be needed for side effects
   - Type checking imports may appear unused
   - Add to allowed exceptions in test configuration

### Debugging Import Issues

```bash
# Verbose analysis
python homodyne/tests/import_analyzer.py --verbose --external-tools

# Check specific file
python -c "
from homodyne.tests.test_import_verification import ImportVerificationSuite
suite = ImportVerificationSuite(Path('homodyne'))
result = suite.analyze_file(Path('homodyne/specific_file.py'))
print(result)
"
```

### Performance Debugging

```bash
# Detailed startup analysis
python homodyne/tests/test_startup_performance.py

# Check memory usage
python -c "
from homodyne.tests.test_startup_performance import StartupBenchmarkSuite
suite = StartupBenchmarkSuite()
result = suite.measure_cold_import('homodyne')
print(f'Import time: {result.get('import_time')}s')
print(f'Memory usage: {result.get('memory_delta_mb')}MB')
"
```

## 📋 Maintenance

### Regular Tasks

1. **Weekly**: Run import analysis to catch new unused imports
2. **After major changes**: Update performance baselines
3. **Before releases**: Run full performance regression tests
4. **Monthly**: Review and clean up any accumulated technical debt

### Updating Test Infrastructure

When adding new modules or changing import structure:

1. Update performance baselines
2. Add new critical modules to test lists
3. Adjust performance thresholds if needed
4. Update this documentation

### Contributing

When adding new tests:

1. Use appropriate markers (`@pytest.mark.performance`, etc.)
2. Follow existing test patterns
3. Add documentation for new test capabilities
4. Update baseline establishment script if needed

---

## 🏆 Success Criteria

This test infrastructure ensures:

- ✅ **Zero unused imports** in production code
- ✅ **Fast startup times** (< 2s for basic import)
- ✅ **Efficient memory usage** (< 50MB for basic import)
- ✅ **No performance regressions** in import times
- ✅ **Effective lazy loading** (> 2x speedup)
- ✅ **No circular dependencies** in import chains
- ✅ **Consistent import structure** across the package

This infrastructure provides the foundation for maintaining clean, efficient import structure and fast startup performance in the homodyne package.
