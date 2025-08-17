Contributing Guide
==================

We welcome contributions to the homodyne package! This guide provides comprehensive information for developers who want to contribute to the project, from bug reports to major feature additions.

Getting Started
---------------

Development Environment Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Fork and Clone the Repository**

.. code-block:: bash

   # Fork the repository on GitHub, then clone your fork
   git clone https://github.com/your-username/homodyne-analysis.git
   cd homodyne-analysis

**2. Set Up Development Environment**

.. code-block:: bash

   # Create development environment
   python -m venv dev_env
   source dev_env/bin/activate  # On Windows: dev_env\Scripts\activate

   # Install development dependencies
   pip install numpy scipy matplotlib numba
   pip install pymc arviz pytensor  # For MCMC features
   pip install pytest pytest-cov   # For testing

**3. Install Package in Development Mode**

.. code-block:: bash

   # Install in editable mode
   pip install -e .

**4. Verify Installation**

.. code-block:: bash

   # Run test suite to verify setup
   python homodyne/run_tests.py --fast

Repository Structure
~~~~~~~~~~~~~~~~~~~~

Understanding the codebase organization:

.. code-block:: text

   homodyne/
   ├── run_homodyne.py              # Main CLI entry point
   ├── create_config.py            # Configuration generator
   ├── benchmark_performance.py    # Performance benchmarking
   ├── homodyne/
   │   ├── __init__.py             # Package exports
   │   ├── plotting.py            # Visualization utilities
   │   ├── run_tests.py          # Test runner
   │   ├── core/                 # Core functionality
   │   │   ├── config.py        # Configuration management
   │   │   ├── kernels.py       # Computational kernels
   │   │   └── io_utils.py      # Data I/O utilities
   │   ├── analysis/            # Analysis engines
   │   │   └── core.py         # Main analysis class
   │   ├── optimization/        # Optimization methods
   │   │   ├── classical.py    # Classical optimization
   │   │   └── mcmc.py         # Bayesian MCMC
   │   └── tests/              # Test suite
   │       ├── conftest.py     # Test configuration
   │       ├── fixtures.py     # Test fixtures
   │       └── test_*.py       # Test modules
   └── sphinx_docs/             # Documentation source

Types of Contributions
----------------------

Bug Reports
~~~~~~~~~~~

**How to Report Bugs**:

1. **Search existing issues** to avoid duplicates
2. **Use the bug report template** with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs. actual behavior
   - System information (OS, Python version, package versions)
   - Configuration files (sanitized)
   - Complete error tracebacks

**Example Bug Report**:

.. code-block:: text

   **Bug Description**
   MCMC sampling fails with R-hat convergence warnings on flow analysis

   **Steps to Reproduce**
   1. Use config_laminar_flow.json with default settings
   2. Run: python run_homodyne.py --laminar-flow --method mcmc
   3. MCMC completes but shows R-hat > 1.1 warnings

   **Expected Behavior**
   MCMC should converge with R-hat < 1.1

   **System Information**
   - Python 3.9.7
   - NumPy 1.21.0
   - PyMC 4.0.1
   - macOS 12.0.1

Feature Requests
~~~~~~~~~~~~~~~~

**How to Request Features**:

1. **Check existing feature requests** and roadmap
2. **Provide clear justification** for the feature
3. **Describe the use case** and expected benefits
4. **Consider implementation complexity** and backwards compatibility

**Example Feature Request**:

.. code-block:: text

   **Feature Request: GPU Acceleration Support**
   
   **Use Case**
   Large-scale XPCS datasets (>10GB) would benefit from GPU acceleration
   
   **Proposed Implementation**
   - CuPy backend for NumPy operations
   - GPU-accelerated correlation function calculation
   - Optional dependency with CPU fallback
   
   **Benefits**
   - 10-100x speedup for large datasets
   - Better scalability for high-throughput analysis

Documentation Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Types of Documentation Contributions**:

- **Tutorial improvements**: Clearer explanations, more examples
- **API documentation**: Better docstrings, parameter descriptions
- **Troubleshooting guides**: Solutions to common problems
- **Performance guides**: Optimization tips and best practices

**Documentation Standards**:

- Use reStructuredText (RST) format
- Follow existing style and organization
- Include code examples with expected output
- Test all code examples before submission

Code Contributions
~~~~~~~~~~~~~~~~~~

**Types of Code Contributions**:

- **Bug fixes**: Resolve identified issues
- **Performance improvements**: Optimize algorithms or implementations
- **New features**: Add functionality following project design patterns
- **Test improvements**: Increase coverage or test quality

Development Workflow
--------------------

Git Workflow
~~~~~~~~~~~~

**1. Create Feature Branch**

.. code-block:: bash

   # Update main branch
   git checkout main
   git pull upstream main

   # Create feature branch
   git checkout -b feature/your-feature-name

**2. Make Changes**

- Write code following project conventions
- Add tests for new functionality
- Update documentation as needed
- Commit changes with clear messages

**3. Test Changes**

.. code-block:: bash

   # Run full test suite
   python homodyne/run_tests.py

   # Run tests with coverage
   python homodyne/run_tests.py --coverage

   # Test specific functionality
   python homodyne/run_tests.py -k "your_test_pattern"

**4. Submit Pull Request**

- Push branch to your fork
- Create pull request with clear description
- Link to related issues
- Request code review

Coding Standards
~~~~~~~~~~~~~~~

**Python Style Guide**:

- Follow PEP 8 with line length limit of 100 characters
- Use type hints for function signatures
- Write comprehensive docstrings (NumPy style)

**Code Quality**:

- Maintain backwards compatibility when possible
- Write unit tests for new functionality
- Use meaningful variable and function names
- Add comments for complex algorithms

**Example Code Style**:

.. code-block:: python

   def compute_correlation_function(
       experimental_data: np.ndarray,
       parameters: Dict[str, float],
       time_points: np.ndarray,
       use_numba: bool = True
   ) -> np.ndarray:
       """
       Compute theoretical correlation function g1(t1, t2).

       Parameters
       ----------
       experimental_data : np.ndarray
           Experimental correlation data g2(t1, t2)
       parameters : Dict[str, float]
           Physical parameters for the model
       time_points : np.ndarray
           Time points for correlation function
       use_numba : bool, optional
           Enable Numba JIT compilation (default: True)

       Returns
       -------
       np.ndarray
           Theoretical correlation function g1(t1, t2)

       Examples
       --------
       >>> params = {'D0': 1e-12, 'alpha': 1.0}
       >>> times = np.linspace(0, 1, 100)
       >>> g1 = compute_correlation_function(data, params, times)
       """
       # Implementation here
       pass

Testing Guidelines
-----------------

Test Categories
~~~~~~~~~~~~~~

**Unit Tests**: Test individual functions and classes

.. code-block:: python

   def test_parameter_validation():
       """Test parameter validation in configuration loading."""
       with pytest.raises(ValueError, match="Invalid parameter value"):
           config = ConfigManager(invalid_config_path)

**Integration Tests**: Test complete workflows

.. code-block:: python

   def test_complete_analysis_workflow():
       """Test end-to-end analysis pipeline."""
       config = create_test_config()
       analysis = HomodyneAnalysisCore(config)
       results = analysis.optimize_classical()
       
       assert results.success
       assert len(results.x) == len(config.get_active_parameters())

**Performance Tests**: Verify performance requirements

.. code-block:: python

   def test_numba_performance():
       """Test that Numba provides expected speedup."""
       config_no_numba = create_config(use_numba_jit=False)
       config_with_numba = create_config(use_numba_jit=True)
       
       time_no_numba = benchmark_analysis(config_no_numba)
       time_with_numba = benchmark_analysis(config_with_numba)
       
       speedup = time_no_numba / time_with_numba
       assert speedup > 2.0, f"Insufficient speedup: {speedup:.2f}x"

Test Writing Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Test Structure**:

.. code-block:: python

   def test_feature_description():
       """Clear description of what is being tested."""
       
       # Arrange: Set up test conditions
       config = create_test_config()
       expected_result = calculate_expected_result()
       
       # Act: Execute the functionality
       analysis = HomodyneAnalysisCore(config)
       actual_result = analysis.method_under_test()
       
       # Assert: Verify correctness
       np.testing.assert_allclose(
           actual_result, expected_result, rtol=1e-6,
           err_msg="Method produces incorrect results"
       )

**Test Coverage Requirements**:

- New code must have ≥80% test coverage
- Critical computational kernels must have ≥95% coverage
- All public API methods must have 100% coverage

Running Tests Locally
~~~~~~~~~~~~~~~~~~~~~

**Basic Test Execution**:

.. code-block:: bash

   # Run all tests
   python homodyne/run_tests.py

   # Run fast tests only
   python homodyne/run_tests.py --fast

   # Run with verbose output
   python homodyne/run_tests.py --verbose

**Advanced Testing**:

.. code-block:: bash

   # Run with coverage reporting
   python homodyne/run_tests.py --coverage

   # Run specific test patterns
   python homodyne/run_tests.py -k "test_config"

   # Run parallel tests
   python homodyne/run_tests.py --parallel 4

Performance Considerations
--------------------------

Performance Guidelines
~~~~~~~~~~~~~~~~~~~~~~

**Computational Efficiency**:

- Use NumPy vectorized operations
- Enable Numba JIT compilation for computational kernels
- Profile code to identify bottlenecks
- Consider memory usage for large datasets

**Example Optimized Code**:

.. code-block:: python

   from numba import jit
   import numpy as np

   @jit(nopython=True, cache=True)
   def compute_chi_squared_optimized(experimental, theoretical, uncertainties):
       """Numba-optimized chi-squared calculation."""
       chi_sq = 0.0
       n = len(experimental)
       
       for i in range(n):
           diff = experimental[i] - theoretical[i]
           chi_sq += (diff / uncertainties[i]) ** 2
       
       return chi_sq

**Performance Testing**:

.. code-block:: python

   def test_performance_regression():
       """Ensure no performance regression in core algorithms."""
       import time
       
       config = create_large_dataset_config()
       analysis = HomodyneAnalysisCore(config)
       
       start_time = time.time()
       results = analysis.optimize_classical()
       execution_time = time.time() - start_time
       
       # Should complete within reasonable time
       assert execution_time < MAX_EXECUTION_TIME, f"Performance regression: {execution_time}s"

Memory Management
~~~~~~~~~~~~~~~~

**Guidelines**:

- Use appropriate data types (float32 vs float64)
- Implement chunked processing for large datasets
- Monitor memory usage in tests
- Provide memory limit configuration options

**Example Memory-Efficient Code**:

.. code-block:: python

   def process_large_dataset(data, chunk_size=1000, memory_limit_gb=16):
       """Process large dataset in chunks to manage memory usage."""
       memory_limit_bytes = memory_limit_gb * 1024**3
       
       for i in range(0, len(data), chunk_size):
           chunk = data[i:i + chunk_size]
           
           # Monitor memory usage
           current_memory = get_memory_usage()
           if current_memory > memory_limit_bytes:
               # Reduce chunk size or implement other memory management
               pass
           
           # Process chunk
           result_chunk = process_chunk(chunk)
           yield result_chunk

Documentation Guidelines
------------------------

Docstring Standards
~~~~~~~~~~~~~~~~~~

Use NumPy-style docstrings for all public functions:

.. code-block:: python

   def analyze_correlation_data(
       data: np.ndarray,
       config: ConfigManager,
       method: str = "classical"
   ) -> AnalysisResults:
       """
       Analyze experimental correlation data using specified method.

       This function performs comprehensive analysis of XPCS correlation
       data using either classical optimization or Bayesian MCMC sampling.

       Parameters
       ----------
       data : np.ndarray
           Experimental correlation function g2(t1, t2) data
       config : ConfigManager
           Configuration object containing analysis parameters
       method : str, optional
           Analysis method ("classical", "mcmc", or "all"), default "classical"

       Returns
       -------
       AnalysisResults
           Object containing optimized parameters, goodness-of-fit statistics,
           and uncertainty estimates (for MCMC methods)

       Raises
       ------
       ValueError
           If data format is invalid or method is not recognized
       ConvergenceError
           If optimization fails to converge

       Examples
       --------
       >>> config = ConfigManager("analysis_config.json")
       >>> results = analyze_correlation_data(experimental_data, config)
       >>> print(f"Optimized D0: {results.parameters['D0']:.2e}")
       Optimized D0: 1.23e-12

       Notes
       -----
       The classical method uses Nelder-Mead simplex optimization, while
       MCMC employs NUTS sampling with automatic parameter tuning.

       References
       ----------
       .. [1] He et al., "Transport coefficient approach for characterizing 
              nonequilibrium dynamics in soft matter," PNAS 2024.
       """
       pass

Documentation Building
~~~~~~~~~~~~~~~~~~~~~~

**Build Documentation Locally**:

.. code-block:: bash

   # Install Sphinx and dependencies
   pip install sphinx sphinx-rtd-theme myst-parser

   # Build HTML documentation
   cd sphinx_docs
   make html

   # View documentation
   open _build/html/index.html  # On macOS
   # Or navigate to _build/html/index.html in your browser

**Documentation Structure**:

- **User Guide**: Installation, quickstart, usage examples
- **Advanced Topics**: Performance optimization, scaling theory
- **API Reference**: Complete function and class documentation
- **Developer Guide**: Contributing guidelines, testing, troubleshooting

Review Process
--------------

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~

**Before Submitting**:

- [ ] Code follows project style guidelines
- [ ] All tests pass locally
- [ ] New functionality has adequate test coverage
- [ ] Documentation is updated appropriately
- [ ] Performance impact is assessed
- [ ] Backwards compatibility is maintained

**Pull Request Template**:

.. code-block:: text

   ## Description
   Brief description of changes and motivation

   ## Type of Change
   - [ ] Bug fix (non-breaking change that fixes an issue)
   - [ ] New feature (non-breaking change that adds functionality)
   - [ ] Breaking change (change that would cause existing functionality to not work as expected)
   - [ ] Documentation update

   ## Testing
   - [ ] New tests added for new functionality
   - [ ] All existing tests pass
   - [ ] Performance impact assessed

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] Breaking changes documented

Code Review Process
~~~~~~~~~~~~~~~~~~

**Review Criteria**:

- **Correctness**: Code works as intended
- **Style**: Follows project conventions
- **Tests**: Adequate coverage and quality
- **Documentation**: Clear and complete
- **Performance**: No significant regressions

**Reviewer Guidelines**:

- Provide constructive feedback
- Explain reasoning behind suggestions
- Test changes locally when necessary
- Approve when all criteria are met

Release Process
---------------

Version Management
~~~~~~~~~~~~~~~~~

**Semantic Versioning**:

- **Major**: Breaking changes (X.0.0)
- **Minor**: New features, backwards compatible (0.X.0)
- **Patch**: Bug fixes, backwards compatible (0.0.X)

**Release Checklist**:

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Performance benchmarks run
- [ ] Migration guide updated (for major releases)
- [ ] Release notes prepared

Community Guidelines
--------------------

Code of Conduct
~~~~~~~~~~~~~~~

**Our Commitment**:

- Foster an open and welcoming environment
- Respect all contributors regardless of background
- Focus on constructive collaboration
- Maintain professional communication

**Expected Behavior**:

- Use inclusive language
- Respect differing viewpoints
- Accept constructive criticism gracefully
- Focus on what's best for the community

**Reporting Issues**:

Report unacceptable behavior to project maintainers. All complaints will be reviewed and investigated promptly and fairly.

Getting Help
~~~~~~~~~~~~

**Resources for Contributors**:

- **Documentation**: Comprehensive guides and API reference
- **Issues**: GitHub Issues for questions and bug reports  
- **Discussions**: GitHub Discussions for general questions
- **Code Review**: Learn from feedback on pull requests

**Mentoring**:

New contributors are welcome! Maintainers are available to help with:

- Understanding the codebase
- Identifying good first issues
- Code review and feedback
- Best practices and conventions

Thank You
---------

We appreciate all contributions to the homodyne package! Whether you're reporting bugs, improving documentation, or adding new features, your contributions help make this package better for the entire XPCS community.

**Recognition**:

Contributors are acknowledged in:

- Release notes for their contributions
- Documentation credits
- Project README contributors section

Together, we're building powerful tools for advancing X-ray photon correlation spectroscopy research!
