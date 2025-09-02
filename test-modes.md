# Test Execution Modes

This document describes the different test execution configurations available for the homodyne project.

## Quick Reference

```bash
# Fast development tests (< 30 seconds)
pytest -c pytest-quick.ini

# Standard CI tests (< 3 minutes) 
pytest -c pytest-ci.ini

# Full test suite excluding benchmarks (< 5 minutes)
pytest -c pytest-full.ini

# Performance benchmarks only (5-15 minutes)
pytest -c pytest-benchmarks.ini
```

## Configuration Details

### 1. pytest-quick.ini
**Purpose**: Rapid development feedback  
**Target time**: < 30 seconds  
**Includes**: Unit tests marked as "fast"  
**Excludes**: Integration, MCMC, performance, slow tests  
**Use case**: Development TDD cycle

### 2. pytest-ci.ini  
**Purpose**: Continuous Integration  
**Target time**: < 3 minutes  
**Includes**: Unit and regression tests  
**Excludes**: Performance, system, integration, MCMC tests  
**Features**: Parallel execution, coverage reporting  

### 3. pytest-full.ini
**Purpose**: Complete functional testing  
**Target time**: < 5 minutes  
**Includes**: All tests except benchmarks  
**Excludes**: Only benchmark tests  
**Features**: Full coverage, parallel execution  

### 4. pytest-benchmarks.ini
**Purpose**: Performance benchmarking  
**Target time**: 5-15 minutes  
**Includes**: Only benchmark and performance tests  
**Features**: pytest-benchmark integration, performance tracking  

## Performance Optimizations Applied

### Phase 1: Reduced Computation
- MCMC draws: 10000+ → 20 (99% reduction)
- MCMC chains: 2-8 → 1 (75% reduction) 
- MCMC tuning: 500-1000 → 10 (98% reduction)

### Phase 2: Caching & Parallelization  
- Session-scoped test data caching
- Parallel test execution with pytest-xdist
- Memory-efficient data structures

### Phase 3: Smart Test Selection
- Automatic test marking based on execution time
- Separate benchmark execution
- CI-optimized test selection

## Expected Performance Improvements

| Configuration | Before | After | Improvement |
|---------------|--------|-------|-------------|
| Quick tests   | 2-3 min | 20-30s | 75-80% |
| CI tests      | 5-7 min | 2-3 min | 50-60% |
| Full tests    | 7+ min | 3-5 min | 40-50% |
| Benchmarks    | Mixed in | Separate | Isolated |

## Usage Examples

```bash
# Development workflow
pytest -c pytest-quick.ini src/tests/unit/

# Pre-commit checks  
pytest -c pytest-ci.ini

# Release testing
pytest -c pytest-full.ini

# Performance analysis
pytest -c pytest-benchmarks.ini --benchmark-compare

# Custom test selection
pytest -m "not slow and not benchmark" --maxfail=3

# Show detailed test outcomes including skipped reasons
pytest -rA  # Shows all outcomes (passed, failed, skipped, etc.) with reasons

# Show only skipped test reasons
pytest -rs  # Shows only skipped tests with reasons
```

## Test Markers

- `@pytest.mark.fast`: Tests that run in < 1 second
- `@pytest.mark.slow`: Tests that take > 1 second  
- `@pytest.mark.benchmark`: Performance benchmark tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.mcmc`: MCMC-specific tests
- `@pytest.mark.performance`: Performance-related tests
- `@pytest.mark.unit`: Unit tests (isolated)
- `@pytest.mark.ci`: Tests suitable for CI