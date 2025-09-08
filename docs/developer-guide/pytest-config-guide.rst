Pytest Configuration Guide
===========================

This document explains the different pytest configuration files available in the homodyne project and how to use them effectively for different testing scenarios.

Configuration Files Overview
-----------------------------

1. pytest-quick.ini - Rapid Development Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Usage:** ``pytest -c pytest-quick.ini``

- **Purpose:** Fast feedback during development
- **Scope:** Unit tests only (``homodyne/tests/unit``)
- **Filters:** Excludes slow, MCMC, and benchmark tests
- **Timeout:** 60 seconds
- **Max Failures:** 5 (fail fast)
- **Output:** Quiet mode with short tracebacks

**Best for:** Day-to-day development and quick validation

2. pytest-ci.ini - Continuous Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Usage:** ``pytest -c pytest-ci.ini``

- **Purpose:** GitHub Actions and CI environments
- **Scope:** Unit and regression tests
- **Filters:** Excludes slow, integration, MCMC, and benchmark tests
- **Parallel:** Auto-scaling with ``pytest-xdist``
- **Coverage:** 75% minimum with XML reports
- **Timeout:** 300 seconds (5 minutes)

**Best for:** Automated testing in CI/CD pipelines

3. pytest-full.ini - Complete Test Suite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Usage:** ``pytest -c pytest-full.ini``

- **Purpose:** Comprehensive testing with coverage
- **Scope:** All tests except benchmarks
- **Coverage:** 80% minimum with HTML reports
- **Parallel:** Auto-scaling execution
- **Timeout:** 600 seconds (10 minutes)
- **Output:** Verbose mode with detailed reporting

**Best for:** Pre-release testing and comprehensive validation

4. pytest-benchmarks.ini - Performance Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Usage:** ``pytest -c pytest-benchmarks.ini``

- **Purpose:** Performance benchmarks and regression testing
- **Scope:** Performance, MCMC, core analysis tests
- **Benchmarking:** Full benchmark suite with histograms
- **Timeout:** 1200 seconds (20 minutes)
- **Environment:** Controlled threading for reproducible results

**Best for:** Performance analysis and optimization validation

New Test Markers
-----------------

The updated configurations include support for new test markers related to recent enhancements:

Computational Method Markers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``hybrid_irls``: Tests for hybrid limited-iteration IRLS approach
- ``weighted_refit``: Tests for weighted refit functionality  
- ``mad_estimation``: Tests for MAD (Median Absolute Deviation) estimation
- ``angle_filtering``: Tests for angle filtering optimization features

Usage Examples
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run only hybrid IRLS tests
   pytest -m "hybrid_irls"

   # Run quick tests excluding specific features
   pytest -c pytest-quick.ini -m "not hybrid_irls"

   # Run performance tests including hybrid IRLS
   pytest -c pytest-benchmarks.ini -m "performance or hybrid_irls"

   # Run CI tests with coverage
   pytest -c pytest-ci.ini --cov-report=term-missing

Configuration Updates
---------------------

Key Improvements
~~~~~~~~~~~~~~~~

1. **Better timeout handling** across all configurations
2. **Enhanced marker filtering** for improved test selection
3. **Improved coverage settings** with better exclusion patterns
4. **Performance-oriented benchmark settings** with controlled environment
5. **Support for new hybrid IRLS features**

Coverage Improvements
~~~~~~~~~~~~~~~~~~~~~

- Excludes runtime files and conftest.py from coverage
- Better exclusion patterns for debug/info logging
- Support for abstract methods and property exclusions

Environment Settings
~~~~~~~~~~~~~~~~~~~~

The benchmark configuration now includes controlled threading for reproducible performance measurements:

.. code-block:: bash

   NUMBA_NUM_THREADS=1
   OMP_NUM_THREADS=1 
   OPENBLAS_NUM_THREADS=1
   MKL_NUM_THREADS=1

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. **Unknown marker warnings**: Add new markers to the configuration files
2. **Timeout issues**: Adjust timeout values for your environment
3. **Coverage failures**: Review exclusion patterns if coverage drops unexpectedly

Performance Tips
~~~~~~~~~~~~~~~~

- Use ``pytest-quick.ini`` for development iteration
- Use ``pytest-ci.ini`` for automated testing
- Run ``pytest-benchmarks.ini`` periodically to check performance regressions
- Use ``pytest-full.ini`` before releases for comprehensive validation

Integration with CLAUDE.md
---------------------------

These configurations work seamlessly with the commands documented in ``CLAUDE.md``:

.. code-block:: bash

   # Quick test run (now uses pytest-quick.ini internally)
   pytest -m "not slow" -x --tb=line -q

   # Full test suite (uses pytest-full.ini style settings)
   pytest --cov=homodyne --cov-report=term-missing

   # CI-style testing (uses pytest-ci.ini approach)
   pytest -c pytest-ci.ini

Configuration Selection Guide
-----------------------------

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Configuration
     - Time Limit
     - Use Case
     - Coverage
   * - ``pytest-quick.ini``
     - < 60s
     - Development iteration
     - No coverage
   * - ``pytest-ci.ini``
     - < 300s
     - CI/CD pipelines
     - 75% minimum
   * - ``pytest-full.ini``
     - < 600s
     - Pre-release validation
     - 80% minimum
   * - ``pytest-benchmarks.ini``
     - < 1200s
     - Performance analysis
     - No coverage

Advanced Configuration Details
------------------------------

Marker Filtering Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~

Each configuration uses different marker combinations:

- **Quick**: ``-m "not slow and not mcmc and not benchmark"``
- **CI**: ``-m "not slow and not integration and not mcmc and not benchmark"``
- **Full**: ``-m "not benchmark"`` (runs all except benchmarks)
- **Benchmarks**: ``-m "benchmark or performance or slow or hybrid_irls"``

Parallel Execution Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~

All configurations except quick use ``-n auto`` for parallel execution:

- **CI**: Uses ``--dist=loadscope`` for optimal load balancing
- **Full**: Standard parallel execution
- **Benchmarks**: Sequential execution for accurate timing

Coverage Configuration
~~~~~~~~~~~~~~~~~~~~~~

The full configuration includes comprehensive coverage settings:

.. code-block:: ini

   [coverage:run]
   source = homodyne
   omit =
       */tests/*
       */test_*
       */__pycache__/*
       */runtime/*
       */conftest.py

   [coverage:report]
   exclude_lines =
       pragma: no cover
       def __repr__
       def __str__
       raise AssertionError
       raise NotImplementedError
       if __name__ == .__main__.:
       if TYPE_CHECKING:
       @abstractmethod
       @abstractproperty
       pass
       logger\.debug
       logger\.info
       warnings\.warn