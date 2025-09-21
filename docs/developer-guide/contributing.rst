Contributing Guide
==================

We welcome contributions to the homodyne package! This guide explains how to contribute effectively.

Getting Started
---------------

**1. Fork and Clone**

.. code-block:: bash

   # Fork the repository on GitHub, then:
   git clone https://github.com/YOUR_USERNAME/homodyne.git
   cd homodyne

   # Add upstream remote
   git remote add upstream https://github.com/imewei/homodyne.git

**2. Development Setup**

.. code-block:: bash

   # Create development environment
   conda create -n homodyne-dev python=3.12
   conda activate homodyne-dev

   # Install in development mode
   pip install -e .[dev]

   # Install pre-commit hooks
   pre-commit install

**3. Verify Setup**

.. code-block:: bash

   # Run tests to verify everything works
   pytest homodyne/tests/ -v

   # Check code quality and security tools
   black --check homodyne/
   isort --check-only homodyne/
   flake8 homodyne/
   mypy homodyne/
   bandit -r homodyne/
   pip-audit

Development Workflow
--------------------

**1. Create Feature Branch**

.. code-block:: bash

   # Sync with upstream
   git fetch upstream
   git checkout main
   git merge upstream/main

   # Create feature branch
   git checkout -b feature/your-feature-name

**2. Make Changes**

- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed
- Keep commits focused and atomic

**3. Test Changes**

.. code-block:: bash

   # Run relevant tests
   pytest homodyne/tests/test_your_changes.py -v

   # Run full test suite
   pytest homodyne/tests/

   # Check coverage
   pytest homodyne/tests/ --cov=homodyne --cov-report=html

**4. Code Quality Checks**

.. code-block:: bash

   # Format code
   black homodyne/

   # Check linting
   flake8 homodyne/

   # Type checking
   mypy homodyne/

   # Check documentation
   cd docs/
   make html

**5. Commit and Push**

.. code-block:: bash

   # Stage changes
   git add .

   # Commit with descriptive message
   git commit -m "Add feature: brief description

   - Detailed description of changes
   - Why the change was made
   - Any breaking changes or migration notes"

   # Push to your fork
   git push origin feature/your-feature-name

**6. Create Pull Request**

- Open PR against the main branch
- Use the PR template
- Link related issues
- Request review from maintainers

Coding Standards
----------------

**Python Style**

Follow PEP 8 with these specifics:

.. code-block:: python

   # Line length: 88 characters (Black default)
   # Use Black for formatting
   # Use meaningful variable names

   # Good
   def compute_correlation_function(tau_values, model_parameters, scattering_vector):
       """Compute correlation function with given parameters."""
       pass

   # Avoid
   def compute_g1(t, p, q):
       pass

**Type Hints**

Use type hints for all public functions:

.. code-block:: python

   from typing import List, Optional, Tuple, Union
   import numpy as np

   def optimize_parameters(
       initial_params: List[float],
       bounds: Optional[List[Tuple[float, float]]] = None,
       method: str = "Nelder-Mead"
   ) -> Union[np.ndarray, None]:
       """Optimize model parameters."""
       pass

**Documentation**

Use NumPy-style docstrings:

.. code-block:: python

   def compute_chi_squared(
       experimental_data: np.ndarray,
       theoretical_data: np.ndarray,
       uncertainties: Optional[np.ndarray] = None
   ) -> float:
       """
       Compute chi-squared goodness of fit.

       Parameters
       ----------
       experimental_data : np.ndarray
           Experimental correlation data.
       theoretical_data : np.ndarray
           Theoretical model predictions.
       uncertainties : np.ndarray, optional
           Experimental uncertainties. If None, assumes uniform weighting.

       Returns
       -------
       float
           Chi-squared value.

       Examples
       --------
       >>> exp_data = np.array([1.0, 0.8, 0.6])
       >>> theory_data = np.array([0.98, 0.79, 0.61])
       >>> chi2 = compute_chi_squared(exp_data, theory_data)
       >>> print(f"Chi-squared: {chi2:.4f}")
       Chi-squared: 0.0014
       """
       pass

**Error Handling**

Use specific exception types:

.. code-block:: python

   from homodyne.utils import ConfigurationError, DataFormatError

   def load_configuration(config_path: str) -> dict:
       """Load and validate configuration file."""
       if not os.path.exists(config_path):
           raise FileNotFoundError(f"Configuration file not found: {config_path}")

       try:
           with open(config_path) as f:
               config = json.load(f)
       except json.JSONDecodeError as e:
           raise ConfigurationError(f"Invalid JSON in config file: {e}")

       if "analysis_settings" not in config:
           raise ConfigurationError("Missing required 'analysis_settings' section")

       return config

Testing Guidelines
------------------

**Test Coverage**

Aim for >90% test coverage for new code:

.. code-block:: python

   # Test all public functions
   # Test edge cases and error conditions
   # Test with realistic data

   class TestNewFeature:
       def test_basic_functionality(self):
           """Test basic feature operation."""
           pass

       def test_edge_cases(self):
           """Test boundary conditions."""
           pass

       def test_error_handling(self):
           """Test error conditions."""
           with pytest.raises(ValueError):
               invalid_operation()

       @pytest.mark.parametrize("param,expected", [
           (1.0, 2.0),
           (2.0, 4.0),
           (3.0, 6.0)
       ])
       def test_parameterized(self, param, expected):
           """Test with multiple parameter sets."""
           assert function(param) == expected

**Performance Tests**

Include performance tests for computationally intensive features:

.. code-block:: python

   @pytest.mark.benchmark
   def test_optimization_performance(self, benchmark):
       """Benchmark optimization performance."""
       result = benchmark(run_optimization, test_data)
       assert result.success

**Integration Tests**

Test complete workflows:

.. code-block:: python

   def test_complete_analysis_workflow(self, tmp_path):
       """Test end-to-end analysis workflow."""
       # Create test configuration
       config_file = create_test_config(tmp_path)

       # Run complete analysis
       analysis = HomodyneAnalysisCore(config_file)
       analysis.load_experimental_data()
       result = analysis.optimize_classical()

       # Verify results
       assert result.success
       assert result.fun < threshold

Documentation Guidelines
------------------------

**API Documentation**

- Document all public functions and classes
- Include examples in docstrings
- Use proper cross-references
- Keep documentation up-to-date with code changes

**User Guide Updates**

When adding new features:

1. Update relevant user guide sections
2. Add examples to the examples section
3. Update configuration documentation
4. Consider adding troubleshooting entries

**Developer Documentation**

For significant architectural changes:

1. Update architecture documentation
2. Document new design patterns
3. Update performance guidelines
4. Add troubleshooting information

Types of Contributions
----------------------

**Bug Fixes**

1. **Reproduce the issue** with a minimal example
2. **Add a test** that fails before the fix
3. **Implement the fix** with minimal changes
4. **Verify the test passes** after the fix
5. **Update documentation** if needed

**New Features**

1. **Discuss the feature** in an issue first
2. **Design the API** carefully
3. **Implement with tests** and documentation
4. **Consider backward compatibility**
5. **Update examples** if relevant

**Performance Improvements**

1. **Benchmark current performance** before changes
2. **Implement optimization** with tests
3. **Verify performance improvement** with benchmarks
4. **Ensure correctness** is maintained
5. **Document the improvement**

**Documentation Improvements**

1. **Identify unclear sections** or missing information
2. **Add examples** and clarifications
3. **Update for accuracy** with current code
4. **Test documentation builds** locally
5. **Check for broken links** or references

Pull Request Guidelines
-----------------------

**PR Title and Description**

Use clear, descriptive titles:

.. code-block:: text


Include comprehensive descriptions:

.. code-block:: text

   ## Summary
   Brief description of changes

   ## Changes Made
   - Specific change 1
   - Specific change 2

   ## Testing
   - How was this tested?
   - Any new test cases added?

   ## Breaking Changes
   - Any backward compatibility issues?

   ## Related Issues
   - Fixes #123
   - Related to #456

**Code Review Process**

1. **Self-review** your changes before submitting
2. **Respond to feedback** constructively
3. **Make requested changes** promptly
4. **Keep the PR focused** on a single feature/fix
5. **Rebase and squash** commits if requested

**Checklist**

Before submitting a PR:

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Change is backward compatible (or breaking changes are documented)
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the change and why it's needed

Release Process
---------------

**Versioning**

We follow semantic versioning (SemVer):

- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

**Release Checklist**

For maintainers:

1. Update version numbers
2. Update CHANGELOG.md
3. Run full test suite
4. Build and test documentation
5. Create release tag
6. Publish to PyPI
7. Update GitHub release notes

Community Guidelines
--------------------

**Code of Conduct**

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn
- Acknowledge contributions

**Communication**

- **GitHub Issues**: Bug reports, feature requests
- **Pull Requests**: Code contributions
- **Discussions**: General questions and ideas

**Recognition**

Contributors are recognized through:

- Git commit history
- CONTRIBUTORS.md file
- Release notes
- GitHub contributor statistics

Getting Help
------------

If you need help contributing:

1. **Read the documentation** thoroughly
2. **Search existing issues** for similar problems
3. **Ask questions** in GitHub Discussions
4. **Start with small contributions** to learn the workflow
5. **Join the community** and learn from other contributors

We appreciate all contributions, from bug reports to major features!
