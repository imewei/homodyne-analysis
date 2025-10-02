# Changelog

All notable changes to the Homodyne Scattering Analysis Package will be documented in
this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-01

### Major Release - Production Ready

This release marks the first stable 1.0 release of homodyne-analysis, representing a
mature, production-ready package for analyzing homodyne scattering in X-ray Photon
Correlation Spectroscopy (XPCS) under nonequilibrium conditions.

### Critical Bug Fixes

- **Frame Counting Convention Fix**: Fixed critical bug in frame counting that caused
  NaN chi-squared values and dimensional mismatches
  - Corrected formula: `time_length = end_frame - start_frame + 1` (1-based inclusive to
    0-based Python slicing)
  - Added utility functions: `calculate_time_length()` and
    `config_frames_to_python_slice()`
  - Auto-adjustment of time_length when cached data dimensions don't match config
  - Fixed in 9 modules: analysis/core.py, cli/run_homodyne.py, cli/simulation.py,
    core/composed_analysis.py, core/config.py, core/io_utils.py, core/kernels.py,
    core/workflows.py, data/xpcs_loader.py
  - Comprehensive regression tests added in test_time_length_calculation.py

### Major Features

- **Conditional Angle Subsampling**: Preserve angular information for datasets with few
  angles

  - Automatically skips angle subsampling when n_angles < 4
  - Prevents loss of critical angular information (e.g., 2 angles → 2 angles instead of
    2 → 1)
  - Time subsampling still applied for performance (~16x reduction)
  - Implemented in both classical and robust optimizers
  - Documented in all configuration templates with automatic behavior notes

- **Memory Optimization for Robust Methods**: Improved ellipsoidal optimization
  reliability

  - Increased memory limit from 85% to 90% for robust optimization
  - Fixed stacked decorator issue causing premature memory limit errors
  - Removed conflicting @secure_scientific_computation decorator
  - Ellipsoidal uncertainty sets now handle larger datasets reliably

- **Comprehensive Documentation Suite**: Professional research-grade documentation

  - Complete API documentation (docs/api/README.md, analysis_core.md)
  - Research methodology documentation (docs/research/methodology.md)
  - Documentation summary with quality metrics (DOCUMENTATION_SUMMARY.md)
  - 107 KB of technical documentation with 50+ code examples
  - 20+ LaTeX mathematical formulations
  - 7 peer-reviewed references

### Configuration Improvements

- **Complete Dependency Synchronization**: All configuration files perfectly aligned

  - requirements.txt: Core dependencies synced with pyproject.toml
  - requirements-optional.txt: Performance and robust optimization dependencies
  - requirements-dev.txt: Modern testing framework with 46+ development tools
  - Zero version conflicts across all configuration files
  - 97.2% configuration completeness verified

- **Enhanced Configuration Templates**: All templates updated with subsampling settings

  - laminar_flow.json: 7-parameter flow analysis with subsampling configuration
  - static_anisotropic.json: 3-parameter anisotropic with angle preservation
  - static_isotropic.json: 3-parameter isotropic with optimized subsampling
  - template.json: Base template with all latest features documented
  - Conditional angle subsampling behavior documented in all templates

- **Makefile Enhancements**: 54 development targets with complete .PHONY declarations

  - New documentation targets: docs-validate, docs-check, docs-stats, docs-clean
  - Comprehensive test discovery and performance baseline management
  - Build optimization targets with caching and parallel execution
  - All targets properly declared for correct make operation

### Performance Improvements

- **Optimized Chi-Squared Calculations**: 38% faster execution

  - Improved from 1.33ms to 0.82ms per calculation
  - Vectorized operations throughout
  - Numba JIT compilation providing 3-5x speedup for core kernels

- **Subsampling Performance**: Configurable data reduction for large datasets

  - Enabled by default in all templates for datasets > 100k points
  - Typical speedup: 20-50x with \<10% chi-squared degradation
  - Example: 8M dataset (2 angles × 2000² times) reduced to ~62.5k points (128x
    reduction)
  - Conditional angle preservation ensures data quality

### Quality Assurance

- **Modern Testing Infrastructure**: 26 pytest markers with comprehensive coverage

  - Enhanced pytest configuration with 8.4.0+ features
  - Parallel execution, HTML/JSON reports, property-based testing
  - Performance benchmarking and regression testing
  - Memory profiling and notebook validation

- **Security Hardening**: Complete security scanning pipeline

  - bandit, safety, pip-audit all configured in dev dependencies
  - Pre-commit hooks with 25 checks across 10 phases
  - Comprehensive type checking with mypy 1.18.2+
  - Modern linting with ruff 0.13.2+ and black 25.9.0+

### Dependency Updates

- **NumPy 2.x Support**: Full compatibility with NumPy 2.1.0+

  - Updated from NumPy 1.24+ to NumPy 2.1.0+
  - All scientific computing dependencies modernized
  - scipy>=1.14.0, matplotlib>=3.9.0, h5py>=3.12.0

- **Modern Python Support**: Python 3.12+ required

  - Classifiers for Python 3.12, 3.13, 3.14
  - typing-extensions for backward compatibility
  - Future-proof version constraints

### Package Distribution

- **MANIFEST.in Updates**: All new documentation properly included

  - API documentation (docs/api/\*)
  - Research documentation (docs/research/\*)
  - Documentation summary (DOCUMENTATION_SUMMARY.md)
  - 40 documentation files properly packaged

- **Build System Validation**: PEP 517/518 compliant

  - setuptools>=80.9.0 with setuptools-scm>=8.1.0
  - Dynamic versioning from git tags
  - Wheel>=0.45.1 for modern distribution

### API Stability

This 1.0.0 release marks the package API as stable:

- Public API frozen - no breaking changes without major version bump
- Configuration format stable - backward compatibility maintained
- CLI interface stable - existing scripts continue to work
- Output format stable - analysis results structure preserved

### Migration from v0.x

No breaking changes for users:

- Existing configurations work without modification
- CLI commands remain unchanged
- Python API fully backward compatible
- Cache files automatically validated and adjusted

### Acknowledgments

This release represents months of development, testing, and refinement to achieve
production-ready quality for scientific research applications in X-ray Photon
Correlation Spectroscopy.

**Contributors**: Wei Chen, Hongrui He, Claude (Anthropic) **Institution**: Argonne
National Laboratory

______________________________________________________________________

## [0.8.0] - 2025-09-18

### BREAKING CHANGES

- **MCMC Analysis Removed**: MCMC (Markov Chain Monte Carlo) functionality has been
  completely removed from homodyne-analysis to simplify the codebase and reduce
  dependency complexity
  - Removed `homodyne.optimization.mcmc` module and all PyMC-based Bayesian analysis
  - Removed `--method mcmc` CLI option with graceful deprecation handling
  - Removed PyMC, ArviZ, PyTensor, and corner plot dependencies
  - Updated all configuration templates to mark MCMC sections as deprecated

### Added

- **Enhanced Deprecation Handling**: Comprehensive migration guidance for users
  transitioning from MCMC
  - Graceful handling of `--method mcmc` with detailed migration instructions
  - Configuration system automatically ignores MCMC sections with deprecation warnings
  - MCMC plotting functions replaced with informative deprecation stubs
- **Migration Documentation**: Updated examples and documentation to focus on available
  methods
  - Classical optimization (Nelder-Mead, Gurobi) for fast parameter estimation
  - Robust optimization (Wasserstein DRO, Scenario-based, Ellipsoidal) for uncertainty
    resistance
  - Comprehensive migration guide with alternative analysis workflows

### Changed

- **Method Selection**: `--method all` now runs only classical and robust optimization
  methods
- **Configuration System**: MCMC sections in configuration files are automatically
  filtered out with warnings
- **Dependencies**: Removed optional MCMC dependencies from installation requirements
- **Documentation**: Updated all examples and tutorials to use classical and robust
  methods

### Removed

- **MCMC Module**: Complete removal of `homodyne/optimization/mcmc.py`
- **MCMC Dependencies**: PyMC ≥5.0.0, ArviZ ≥0.12.0, PyTensor ≥2.8.0, corner ≥2.2.0
- **MCMC Plotting**: Corner plots, trace plots, and convergence diagnostics
- **MCMC Results**: Posterior sampling, trace storage, and Bayesian uncertainty
  quantification

### Migration Guide

**For Uncertainty Quantification (MCMC Alternative)**:

- Use robust optimization methods for uncertainty resistance against measurement noise
- Classical methods provide parameter error estimates and goodness-of-fit metrics
- Both methods include comprehensive diagnostic plots and result validation

**Command Migration**:

```bash
# Old MCMC usage (no longer works)
# homodyne --method mcmc

# New alternatives
homodyne --method classical  # Fast parameter estimation
homodyne --method robust     # Noise-resistant analysis
homodyne --method all        # Both classical and robust methods
```

**Configuration Migration**:

- MCMC configuration sections are automatically ignored with deprecation warnings
- No manual configuration changes required - existing configs remain compatible
- New configurations should focus on classical and robust optimization settings

## [0.7.1] - 2025-08-28

### Fixed

- **Critical Windows CI Test Failures**: Fixed Windows path separator compatibility
  issues in shell completion system
  - Shell completion functions now properly use `os.sep` for Windows backslash (`\`)
    path separators
  - Fixed test assertions to expect platform-appropriate path separators instead of
    hardcoded forward slashes
  - Resolved `TestHomodyneCompleter.test_output_dir_completer` and
    `TestCompletionFunctions.test_complete_output_dir` failures
- **Enhanced Cross-Platform Test Reliability**: All completion tests now pass
  consistently across Windows, macOS, and Linux
  - Updated `cli_completion.py` and `completion_fast.py` to use proper cross-platform
    path handling
  - Fixed multiple test assertions in `test_cli_completion.py` and
    `test_completion_fast.py`
  - Added missing `import os` for cross-platform path separator support
- **Windows Unicode Encoding Issues**: Fixed Windows cp1252 encoding errors in logging
  - Replaced Unicode characters (Å⁻¹, μm) with ASCII equivalents (A^-1, um) to prevent
    Windows encoding errors
  - Fixed logging output in `HomodyneAnalysisCore` initialization summary
- **Performance Test Stability**: Improved timing assertion handling
  - Fixed performance test failures when timing measurements are unmeasurably fast
  - Added proper handling for edge cases in `test_config_caching_optimization`
- **Type Compatibility**: Fixed type conversion issues
  - Convert list parameters to numpy arrays for function compatibility
  - Resolved type errors in `calculate_c2_nonequilibrium_laminar_parallel`
- **Code Quality**: Applied consistent formatting and style
  - Fixed black code formatting issues across test files
  - Removed auto-generated files from version control

### Technical Details

- **Path Separator Logic**: Enhanced directory completion to append `os.sep` instead of
  hardcoded `/`
- **Test Coverage**: All 30+ completion tests now pass on Windows with native backslash
  path separators
- **Backward Compatibility**: Maintains full compatibility with existing Unix-style
  forward slash usage
- **CI Pipeline**: All formatting and linting checks now pass consistently

### Impact

- **Windows Users**: Shell completion now works correctly with Windows path conventions
- **CI/CD Pipeline**: Windows CI tests now pass reliably without path separator
  conflicts or Unicode encoding issues
- **Developer Experience**: Consistent cross-platform behavior for all completion
  functionality and improved code quality standards

## [0.7.0] - 2025-08-28

### Added

- **Enhanced Cross-Platform Compatibility**: Comprehensive Windows compatibility
  improvements for shell completion system
- **Improved Shell Completion Reliability**: Better error handling and graceful
  degradation across all supported platforms
- **Advanced Path Handling**: Cross-platform path separator support for Windows, macOS,
  and Linux environments

### Fixed

- **Critical Windows Compatibility**: Fixed shell completion path separator handling for
  Windows environments
  - Added support for both Windows backslash (`\`) and Unix forward slash (`/`) path
    separators
  - Enhanced completion functions with `os.sep` detection for cross-platform
    compatibility
  - Resolved Windows CI test failures related to path handling in completion system
- **Performance Test Reliability**: Fixed arithmetic formatting in completion
  performance tests
  - Improved mathematical expression handling across different platforms
  - Enhanced test stability and consistency in CI environments
- **Code Quality Improvements**: Applied comprehensive formatting and linting fixes
  - Resolved Pylance diagnostic issues and improved type safety
  - Maintained consistent code style across the entire codebase

### Enhanced

- **Cross-Platform Testing**: Improved test suite reliability across Windows, macOS, and
  Linux
- **Error Messages**: Better error handling with more informative messages for
  cross-platform scenarios
- **Code Documentation**: Enhanced inline documentation for cross-platform path handling
- **Developer Experience**: Smoother development workflow across different operating
  systems

### Technical Details

- **Path Separator Logic**: Enhanced completion functions to handle both `os.sep` and
  `/` for maximum compatibility
- **Test Stability**: Fixed performance test assertions to work consistently across
  platforms
- **Code Quality**: Maintained high standards with comprehensive linting and formatting
- **CI/CD Pipeline**: Ensured all GitHub workflow quality checks pass on all supported
  platforms

### Compatibility

- **Windows**: Full shell completion support with native path separator handling
- **macOS**: Continued excellent support with all existing functionality
- **Linux**: Maintained compatibility with enhanced cross-platform features
- **Backward Compatible**: All existing functionality preserved, no breaking changes

## [0.6.10] - 2025-08-28

### Added

- **Shell Completion Uninstall Feature**: New
  `homodyne --uninstall-completion {bash,zsh,fish,powershell}` command to cleanly remove
  shell completion
- **Enhanced Shell Completion System**: Improved reliability and error handling for
  shell completion installation
- **Cross-Platform Uninstall Support**: Uninstall functionality works across all
  supported shells and platforms
- **Comprehensive Documentation**: Updated all documentation with uninstall examples and
  usage instructions

### Fixed

- **homodyne-config Completion**: Fixed shell completion for `homodyne-config` command
  that was not showing correct options
- **Import Path Issues**: Resolved relative import issues in completion system that
  caused failures when run as script
- **Shell Parsing Logic**: Fixed completion parsing logic for proper handling of
  command-line arguments
- **Zsh Completion Fallback**: Enhanced zsh completion fallback system to handle more
  edge cases

### Enhanced

- **Completion Documentation**: Updated README.md, CLI_REFERENCE.md, API_REFERENCE.md,
  and docs/ with comprehensive examples
- **Error Handling**: Improved error messages and graceful degradation for completion
  system
- **Code Quality**: Applied black formatting, flake8 linting, mypy type checking, and
  ruff formatting to all new code
- **Type Annotations**: Added proper type hints to all new completion-related functions

### Developer Experience

- **Bypass Completion File**: Enhanced `homodyne_completion_bypass.zsh` with support for
  both commands and all options
- **Manual Completion**: Added manual completion triggers (Ctrl-X + c) for when
  automatic completion fails
- **Convenience Aliases**: Added shortcuts like `hc-iso`, `hc-aniso`, `hc-flow` for
  common `homodyne-config` operations
- **Help Function**: Enhanced `homodyne_help` to show all available shortcuts and
  completion options

## [0.6.9] - 2025-08-27

### Added

- **Comprehensive Security Framework**: Integrated security scanning with Bandit and
  pip-audit for automated vulnerability detection
- **Security Documentation**: Complete security guidelines and best practices
  documentation (`docs/developer-guide/security.rst`)
- **Enhanced Quality Tools**: Added pip-audit dependency vulnerability scanner to
  development workflow
- **Security Configuration**: Properly configured Bandit with scientific code patterns
  and security exclusions

### Fixed

- **Windows CI Performance Test**: Adjusted robust optimization performance test
  thresholds for CI environment compatibility
- **Security Tool Integration**: Fixed Bandit configuration to work with scientific
  Python code patterns
- **Documentation Updates**: Enhanced README.md and documentation with security
  information

### Changed

- **Development Workflow**: Updated code quality checks to include comprehensive
  security scanning
- **Requirements Files**: Enhanced with security tools and proper dependency
  organization
- **Package Configuration**: Improved pyproject.toml with security tool configurations
  and proper skips

### Security

- **Zero Security Issues**: Achieved 0 medium/high severity security issues through
  comprehensive Bandit scanning
- **Dependency Security**: Implemented automated dependency vulnerability checking with
  pip-audit
- **Secure Development**: Established security-first development practices and
  documentation

## [0.6.8] - 2025-08-27

### Fixed

- **Cross-Platform Compatibility**: Fixed Windows path separator issues in completion
  tests
- **Module Import Issues**: Resolved AttributeError in isotropic mode integration tests
- **Configuration Template Handling**: Fixed MODE_DEPENDENT placeholder resolution in
  static mode tests
- **Performance Test Thresholds**: Adjusted completion performance expectations for CI
  environments
- **Code Quality**: Fixed import sorting and formatting issues

### Improved

- **Test Suite Reliability**: All GitHub Actions tests now pass consistently across
  platforms
- **Cross-Platform Testing**: Enhanced compatibility with Windows, macOS, and Linux
- **Code Formatting**: Applied black formatter and isort for consistent code style

## [0.6.5] - 2024-11-24

### Added

- **Robust Optimization Framework**: Complete implementation of robust optimization
  methods for noise-resistant parameter estimation
  - Three robust methods: Robust-Wasserstein (DRO), Robust-Scenario (Bootstrap),
    Robust-Ellipsoidal (Bounded uncertainty)
  - Integration with CVXPY + Gurobi for convex optimization
  - Dedicated `--method robust` command-line flag for robust-only analysis
  - Comprehensive test coverage with 29 unit tests and 15 performance benchmarks
- **Individual Method Results Saving**: Comprehensive saving of analysis results for all
  optimization methods
  - Saves fitted parameters with statistical uncertainties for each method individually
  - Method-specific directories (nelder_mead/, gurobi/, robust_wasserstein/, etc.)
  - JSON files with parameters, uncertainties, goodness-of-fit metrics, and convergence
    information
  - NumPy archives with full numerical data for each method
  - Summary files for easy method comparison (all_classical_methods_summary.json,
    all_robust_methods_summary.json)
- **Comprehensive Diagnostic Summary Visualization**: Advanced diagnostic plots for
  analysis quality assessment
  - 2×3 grid layout combining method comparison, parameter uncertainties, convergence
    diagnostics, and residuals analysis
  - Cross-method comparison with chi-squared values, parameter uncertainties, and MCMC
    convergence metrics
  - Residuals distribution analysis with normal distribution overlay and statistical
    summaries
  - Professional formatting with consistent styling, grid lines, and color coding

### Changed

- **Diagnostic Summary Plot Generation**: Main `diagnostic_summary.png` plot only
  generated for `--method all` to provide meaningful cross-method comparisons
- **Classical Optimization Architecture**: Expanded from single-method to multi-method
  framework
- **Configuration Templates**: All JSON templates now include robust optimization and
  Gurobi method options
- **Method Selection**: Best optimization result automatically selected based on
  chi-squared value

### Fixed

- **Removed deprecated `--static` CLI argument**: Cleaned up legacy command-line
  argument that was replaced by `--static-anisotropic`
- **Removed unused profiler module**: Deleted `homodyne/core/profiler.py` and migrated
  functionality to `PerformanceMonitor` class in `core/config.py`
- **Fixed AttributeError in CLI**: Resolved `args.static` reference error that caused
  immediate crash on startup
- **Fixed test imports**: Updated all performance test imports to use new
  `PerformanceMonitor` API
- **Documentation updates**: Updated all documentation to reflect removed functionality
  and new API patterns
- **Type Safety**: Resolved all Pylance type checking issues for optional imports
- **Parameter Bounds**: Ensured consistent bounds across all optimization methods

### Performance

- **Enhanced stable benchmarking**: Added comprehensive statistics (mean, median,
  percentiles, outlier detection)
- **Performance test improvements**: Better reliability with JIT warmup and
  deterministic data
- **Bounds Constraints**: Gurobi provides native support for parameter bounds (unlike
  Nelder-Mead)

## [0.6.6] - 2025-08-27

### Added

- **Enhanced Shell Completion System**: Implemented multi-tier shell completion with
  robust fallback mechanisms
  - Fast standalone completion script (`homodyne_complete`) with zero package
    dependencies for instant performance
  - Comprehensive shell shortcuts: `hc` (classical), `hr` (robust), `ha` (all methods)
  - Silent loading option for completion system without startup notifications
  - Three-tier fallback system: tab completion → shortcuts → help system
- **Code Quality Improvements**: Comprehensive formatting and linting applied across
  entire codebase
  - Applied Black formatter (line length 88) to all Python files for consistent style
  - Applied isort import sorting with Black profile for organized imports
  - Enhanced type consistency and import organization
- **Documentation Updates**: Comprehensive updates to reflect shell completion
  enhancements
  - Updated CLI_REFERENCE.md with three-tier completion system documentation
  - Enhanced README.md with Shell Completion & Shortcuts section
  - Updated user-guide documentation with shell enhancement setup instructions

### Changed

- **Shell Completion Architecture**: Migrated from argcomplete-only to hybrid completion
  system
  - Added bypass mechanism for zsh compdef issues
    (`compdef:153: _comps: assignment to invalid subscript range`)
  - Implemented external completion handler with caching for performance optimization
  - Removed startup notification messages for silent shell loading
- **CLI Interface**: Enhanced user experience with improved completion and shortcuts
  - Completion system now gracefully degrades from tab completion to shortcuts to help
  - Added comprehensive troubleshooting section for completion issues

### Fixed

- **Shell Completion Issues**: Resolved zsh compdef registration failures that broke tab
  completion
- **Completion Performance**: Optimized completion speed with aggressive caching and
  minimal file system operations
- **Documentation Consistency**: Updated version references across all documentation
  files
- **File Organization**: Cleaned up temporary completion files and consolidated working
  completion system

### Performance

- **Completion Speed**: Target < 50ms completion time achieved through zero-dependency
  completion script
- **Caching System**: Implemented intelligent file/directory caching with TTL for faster
  subsequent completions
- **Memory Optimization**: Minimal memory footprint for completion operations

## [Unreleased]

### Changed - Version Consistency Update (2025-09-29)

#### Breaking Changes

- **BREAKING**: Minimum Python version updated to 3.12 (from 3.8)
- Standardized Python version to 3.13 across all environments

#### Version Updates

- Updated `isort` from `>=5.13.0` to `>=6.0.1`
- Updated `pytest-html` from `>=4.2.0` to `>=4.1.1`
- Added `flake8>=7.3.0` to quality dependencies
- GitHub Actions updated: `actions/upload-artifact` v3→v4, `actions/download-artifact`
  v3→v4
- Pre-commit: Node.js v18 → system (v24.8.0)

### Fixed - Test Failures (2025-09-29)

- **9 Test Failures Resolved**: All tests now pass (529 passed, 0 failed)
- Fixed import issues (unused imports, broken relative imports)
- Fixed ML feature dimension mismatch (4 features vs 7 features)
- Fixed startup monitoring TypeError (dict vs numeric values)
- Fixed test configuration (isinstance checks, performance thresholds)

### Added - Documentation (2025-09-29)

- **VERSION_UPDATE_GUIDE.md**: Comprehensive migration guide with version matrix
- **CHANGELOG.md**: Updated with all recent changes

### Added

- **Comprehensive Code Quality Improvements**: Major cleanup and optimization of
  codebase quality
  - Fixed critical Gurobi optimization implementation that was non-iterative and getting
    stuck
  - Implemented proper iterative trust region SQP approach for Gurobi optimization
  - Removed unused function definitions (308 lines) from kernels.py fallback
    implementations
  - Fixed all critical flake8 issues including false comparisons and import organization
  - Added missing fallback function definitions to resolve name errors
  - Enhanced Gurobi with adaptive trust region management and parameter-scaled finite
    differences

### Changed

- **Gurobi Optimization Implementation**: Complete rewrite from single-shot to iterative
  optimization
  - **Trust Region SQP**: Successive quadratic approximations with adaptive trust
    regions (0.1 → 1e-8 to 1.0 range)
  - **Iterative refinement**: Up to 50 outer iterations with convergence criteria based
    on gradient norm and objective improvement
  - **Numerical stability**: Parameter-scaled epsilon for finite differences and
    diagonal Hessian approximation
  - **Enhanced logging**: Debug messages showing iteration progress and convergence
    metrics
- **Code Quality Standards**: Updated formatting and import organization
  - **Black formatting**: Applied 88-character line length formatting to all files
  - **Import sorting**: Fixed import order with isort across all modules
  - **Type annotations**: Improved import patterns to resolve mypy redefinition warnings

### Fixed

- **Critical Gurobi Bug**: Gurobi optimization was building single quadratic
  approximation around initial point only
  - **Root Cause**: No iterative refinement meant χ² values remained constant across
    "iterations"
  - **Solution**: Implemented proper trust region optimization with step
    acceptance/rejection logic
  - **Expected Impact**: Progressive χ² improvement instead of constant values, proper
    convergence behavior
- **Code Quality Issues**: Resolved major flake8 and type checking problems
  - Fixed `== False` to `is False` comparisons in test files (7 locations)
  - Removed unused imports and variables in test modules
  - Added missing fallback functions `_solve_least_squares_batch_fallback` and
    `_compute_chi_squared_batch_fallback`
  - Improved import patterns in `test_cli_completion.py` to avoid redefinition warnings

## [0.6.4] - 2025-08-22

### Added

- **Gurobi Optimization Support**: Added Gurobi quadratic programming solver as
  alternative to Nelder-Mead
  - Automatic detection and graceful fallback when Gurobi not available
  - Quadratic approximation of chi-squared objective function using finite differences
  - Optimized configurations for different analysis modes (static 3-param, laminar flow
    7-param)
  - Comprehensive test coverage with bounds constraint validation
- **Enhanced Documentation**: Updated all configuration templates with Gurobi options
  and usage guidance
- **Optimization Method Consistency**: All methods (Nelder-Mead, Gurobi, MCMC) use
  identical parameter bounds
- **Test Output Summary**: Added `-rs` flag to pytest configuration for always showing
  skip reasons
- **Performance Baselines**: Added comprehensive performance_baselines.json for
  regression tracking

### Changed

- **Classical Optimization Architecture**: Expanded from single-method to multi-method
  framework
- **Configuration Templates**: All JSON templates now include Gurobi method options
- **Package Dependencies**: Added optional Gurobi support in pyproject.toml and
  requirements.txt
- **Method Selection**: Best optimization result automatically selected based on
  chi-squared value
- **Test Cleanup**: Enhanced cleanup of test-generated homodyne_results directories

### Fixed

- **Type Safety**: Resolved all Pylance type checking issues for optional Gurobi imports
- **Parameter Bounds**: Ensured consistent bounds across all optimization methods
- **Test Performance**: Fixed config caching test parameter bounds validation
- **Performance Test Ratio**: Improved chi2_correlation_ratio_regression test with
  workload scaling
- **Test Cleanup**: Fixed automatic cleanup of homodyne/homodyne_results test artifacts
- **Performance Baselines Path**: Corrected baseline file path resolution in performance
  tests

### Performance

- **Bounds Constraints**: Gurobi provides native support for parameter bounds (unlike
  Nelder-Mead)
- **Quadratic Programming**: Potentially faster convergence for smooth, well-conditioned
  problems
- **Test Stability**: Improved performance test reliability with JIT warmup and
  deterministic data

## [0.6.3] - 2025-08-21

### Added

- **Advanced batch processing**: New `solve_least_squares_batch_numba` for vectorized
  least squares solving
- **Vectorized chi-squared computation**: Added `compute_chi_squared_batch_numba` for
  batch chi-squared calculation
- **Comprehensive optimization test suite**: Extended performance tests for Phase 3
  batch optimizations

### Changed

- **Chi-squared calculation architecture**: Replaced sequential processing with
  vectorized batch operations
- **Memory access patterns**: Optimized for better cache locality and reduced memory
  allocations
- **Least squares solver**: Enhanced with direct 2x2 matrix math for maximum efficiency

### Performance

- **Breakthrough optimization**: Chi-squared calculation improved by 63.1% (546μs →
  202μs)
- **Batch processing implementation**: Eliminated sequential angle processing with
  vectorized operations
- **Performance ratio achievement**: Chi-squared/correlation ratio improved from 43x to
  15.6x (64% reduction)
- **Memory layout optimization**: Enhanced cache efficiency through contiguous memory
  operations
- **Multi-phase optimization**: Combined variance pre-computation + Numba integration +
  batch vectorization
- **Total speedup factor**: 2.71x improvement over original implementation

## [0.6.2] - 2025-08-21

### Performance

- **Major performance optimizations**: Chi-squared calculation improved by 38% (1.33ms →
  0.82ms)
- **Memory access optimization**: Replaced list comprehensions with vectorized reshape
  operations
- **Configuration caching**: Cached validation and chi-squared configs to avoid repeated
  dict lookups
- **Least squares optimization**: Replaced lstsq with solve() for 2x2 matrix systems (2x
  faster)
- **Memory pooling**: Pre-allocated result arrays to reduce allocation overhead
- **Vectorized operations**: Improved angle filtering with np.flatnonzero()
- **Performance ratio improvement**: Chi-squared/correlation ratio reduced from 6.0x to
  1.7x

### Added

- **New optimization features**: Memory pooling, configuration caching, precomputed
  integrals
- **Performance regression tests**: Automated monitoring of performance baselines
- **Optimization test suite**: Comprehensive tests for new optimization features
- **Performance documentation**: Comprehensive performance guide (docs/performance.rst)
- **Enhanced benchmarking**: Updated performance baselines with optimization metrics

### Changed

- **Static case optimization**: Enhanced vectorized broadcasting for identical
  correlation functions
- **Parameter validation**: Added early returns and optimized bounds checking
- **Array operations**: Improved memory locality and reduced copy operations
- **Algorithm selection**: Better static vs laminar flow detection and handling

### Fixed

- **Memory efficiency**: Reduced garbage collection overhead through pooling
- **Numerical stability**: Preserved all validation logic while optimizing performance
- **JIT compatibility**: Maintained Numba acceleration with optimized pure Python paths

### Added

- Added @pytest.mark.memory decorators to memory-related tests for proper test
  collection

### Fixed

- Fixed GitHub test failure where memory tests were being deselected (exit code 5)
- Updated NumPy version constraints in setup.py, pyproject.toml, and requirements.txt
  for Numba 0.61.2 compatibility
- Fixed documentation CLI command references from python scripts to
  homodyne-config/homodyne commands

## [0.6.1] - 2025-08-21

### Added

- Enhanced JIT warmup system with comprehensive function-level compilation
- Stable benchmarking utilities with statistical outlier filtering
- Consolidated performance testing infrastructure
- Performance baseline tracking and regression detection
- Enhanced type annotations and consistency checks
- Pytest-benchmark integration for advanced performance testing

### Changed

- Improved performance test reliability with reduced variance (60% reduction in CV)
- Updated performance baselines to reflect realistic JIT-compiled expectations
- Consolidated environment optimization utilities to reduce code duplication
- Enhanced error messages and debugging information in tests

### Fixed

- Fixed performance variability in correlation calculation benchmarks
- Resolved type annotation issues in plotting and core modules
- Fixed matplotlib colormap access for better compatibility
- Corrected assertion failures in MCMC plotting tests

### Performance

- Reduced performance variance in JIT-compiled functions from >100% to ~26% CV
- Enhanced warmup procedures for more stable benchmarking
- Improved memory efficiency in performance testing
- Better outlier detection and filtering for timing measurements

## [2024.1.0] - Previous Release

### Added

- Initial homodyne scattering analysis implementation
- Three analysis modes: Static Isotropic, Static Anisotropic, Laminar Flow
- Classical optimization (Nelder-Mead) and Bayesian MCMC (NUTS) methods
- Comprehensive plotting and visualization capabilities
- Configuration management system
- Performance optimizations with Numba JIT compilation

### Features

- High-performance correlation function calculation
- Memory-efficient data processing
- Comprehensive test suite with 361+ tests
- Documentation and examples
- Command-line interface
- Python API

______________________________________________________________________

## Version Numbering

- **Major**: Breaking API changes
- **Minor**: New features, performance improvements
- **Patch**: Bug fixes, documentation updates

## Categories

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Now removed features
- **Fixed**: Any bug fixes
- **Security**: Vulnerability fixes
- **Performance**: Performance improvements
