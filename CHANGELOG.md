# Changelog

All notable changes to the Homodyne Scattering Analysis Package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.1] - 2025-08-28

### Fixed
- **Critical Windows CI Test Failures**: Fixed Windows path separator compatibility issues in shell completion system
  - Shell completion functions now properly use `os.sep` for Windows backslash (`\`) path separators
  - Fixed test assertions to expect platform-appropriate path separators instead of hardcoded forward slashes
  - Resolved `TestHomodyneCompleter.test_output_dir_completer` and `TestCompletionFunctions.test_complete_output_dir` failures
- **Enhanced Cross-Platform Test Reliability**: All completion tests now pass consistently across Windows, macOS, and Linux
  - Updated `cli_completion.py` and `completion_fast.py` to use proper cross-platform path handling
  - Fixed multiple test assertions in `test_cli_completion.py` and `test_completion_fast.py`
  - Added missing `import os` for cross-platform path separator support

### Technical Details
- **Path Separator Logic**: Enhanced directory completion to append `os.sep` instead of hardcoded `/`
- **Test Coverage**: All 30+ completion tests now pass on Windows with native backslash path separators
- **Backward Compatibility**: Maintains full compatibility with existing Unix-style forward slash usage

### Impact
- **Windows Users**: Shell completion now works correctly with Windows path conventions
- **CI/CD Pipeline**: Windows CI tests now pass reliably without path separator conflicts
- **Developer Experience**: Consistent cross-platform behavior for all completion functionality

## [0.7.0] - 2025-08-28

### Added
- **Enhanced Cross-Platform Compatibility**: Comprehensive Windows compatibility improvements for shell completion system
- **Improved Shell Completion Reliability**: Better error handling and graceful degradation across all supported platforms
- **Advanced Path Handling**: Cross-platform path separator support for Windows, macOS, and Linux environments

### Fixed
- **Critical Windows Compatibility**: Fixed shell completion path separator handling for Windows environments
  - Added support for both Windows backslash (`\`) and Unix forward slash (`/`) path separators
  - Enhanced completion functions with `os.sep` detection for cross-platform compatibility
  - Resolved Windows CI test failures related to path handling in completion system
- **Performance Test Reliability**: Fixed arithmetic formatting in completion performance tests
  - Improved mathematical expression handling across different platforms
  - Enhanced test stability and consistency in CI environments
- **Code Quality Improvements**: Applied comprehensive formatting and linting fixes
  - Resolved Pylance diagnostic issues and improved type safety
  - Maintained consistent code style across the entire codebase

### Enhanced
- **Cross-Platform Testing**: Improved test suite reliability across Windows, macOS, and Linux
- **Error Messages**: Better error handling with more informative messages for cross-platform scenarios  
- **Code Documentation**: Enhanced inline documentation for cross-platform path handling
- **Developer Experience**: Smoother development workflow across different operating systems

### Technical Details
- **Path Separator Logic**: Enhanced completion functions to handle both `os.sep` and `/` for maximum compatibility
- **Test Stability**: Fixed performance test assertions to work consistently across platforms
- **Code Quality**: Maintained high standards with comprehensive linting and formatting
- **CI/CD Pipeline**: Ensured all GitHub workflow quality checks pass on all supported platforms

### Compatibility
- **Windows**: Full shell completion support with native path separator handling
- **macOS**: Continued excellent support with all existing functionality
- **Linux**: Maintained compatibility with enhanced cross-platform features
- **Backward Compatible**: All existing functionality preserved, no breaking changes

## [0.6.10] - 2025-08-28

### Added
- **Shell Completion Uninstall Feature**: New `homodyne --uninstall-completion {bash,zsh,fish,powershell}` command to cleanly remove shell completion
- **Enhanced Shell Completion System**: Improved reliability and error handling for shell completion installation
- **Cross-Platform Uninstall Support**: Uninstall functionality works across all supported shells and platforms
- **Comprehensive Documentation**: Updated all documentation with uninstall examples and usage instructions

### Fixed
- **homodyne-config Completion**: Fixed shell completion for `homodyne-config` command that was not showing correct options
- **Import Path Issues**: Resolved relative import issues in completion system that caused failures when run as script
- **Shell Parsing Logic**: Fixed completion parsing logic for proper handling of command-line arguments
- **Zsh Completion Fallback**: Enhanced zsh completion fallback system to handle more edge cases

### Enhanced
- **Completion Documentation**: Updated README.md, CLI_REFERENCE.md, API_REFERENCE.md, and docs/ with comprehensive examples
- **Error Handling**: Improved error messages and graceful degradation for completion system
- **Code Quality**: Applied black formatting, flake8 linting, mypy type checking, and ruff formatting to all new code
- **Type Annotations**: Added proper type hints to all new completion-related functions

### Developer Experience
- **Bypass Completion File**: Enhanced `homodyne_completion_bypass.zsh` with support for both commands and all options
- **Manual Completion**: Added manual completion triggers (Ctrl-X + c) for when automatic completion fails  
- **Convenience Aliases**: Added shortcuts like `hc-iso`, `hc-aniso`, `hc-flow` for common `homodyne-config` operations
- **Help Function**: Enhanced `homodyne_help` to show all available shortcuts and completion options

## [0.6.9] - 2025-08-27

### Added
- **Comprehensive Security Framework**: Integrated security scanning with Bandit and pip-audit for automated vulnerability detection
- **Security Documentation**: Complete security guidelines and best practices documentation (`docs/developer-guide/security.rst`)
- **Enhanced Quality Tools**: Added pip-audit dependency vulnerability scanner to development workflow
- **Security Configuration**: Properly configured Bandit with scientific code patterns and security exclusions

### Fixed
- **Windows CI Performance Test**: Adjusted robust optimization performance test thresholds for CI environment compatibility
- **Security Tool Integration**: Fixed Bandit configuration to work with scientific Python code patterns
- **Documentation Updates**: Enhanced README.md and documentation with security information

### Changed
- **Development Workflow**: Updated code quality checks to include comprehensive security scanning
- **Requirements Files**: Enhanced with security tools and proper dependency organization
- **Package Configuration**: Improved pyproject.toml with security tool configurations and proper skips

### Security
- **Zero Security Issues**: Achieved 0 medium/high severity security issues through comprehensive Bandit scanning
- **Dependency Security**: Implemented automated dependency vulnerability checking with pip-audit
- **Secure Development**: Established security-first development practices and documentation

## [0.6.8] - 2025-08-27

### Fixed
- **Cross-Platform Compatibility**: Fixed Windows path separator issues in completion tests
- **Module Import Issues**: Resolved AttributeError in isotropic mode integration tests 
- **Configuration Template Handling**: Fixed MODE_DEPENDENT placeholder resolution in static mode tests
- **Performance Test Thresholds**: Adjusted completion performance expectations for CI environments
- **Code Quality**: Fixed import sorting and formatting issues

### Improved
- **Test Suite Reliability**: All GitHub Actions tests now pass consistently across platforms
- **Cross-Platform Testing**: Enhanced compatibility with Windows, macOS, and Linux
- **Code Formatting**: Applied black formatter and isort for consistent code style

## [0.6.5] - 2024-11-24

### Added
- **Robust Optimization Framework**: Complete implementation of robust optimization methods for noise-resistant parameter estimation
  - Three robust methods: Robust-Wasserstein (DRO), Robust-Scenario (Bootstrap), Robust-Ellipsoidal (Bounded uncertainty)
  - Integration with CVXPY + Gurobi for convex optimization
  - Dedicated `--method robust` command-line flag for robust-only analysis
  - Comprehensive test coverage with 29 unit tests and 15 performance benchmarks
- **Individual Method Results Saving**: Comprehensive saving of analysis results for all optimization methods
  - Saves fitted parameters with statistical uncertainties for each method individually
  - Method-specific directories (nelder_mead/, gurobi/, robust_wasserstein/, etc.)
  - JSON files with parameters, uncertainties, goodness-of-fit metrics, and convergence information
  - NumPy archives with full numerical data for each method
  - Summary files for easy method comparison (all_classical_methods_summary.json, all_robust_methods_summary.json)
- **Comprehensive Diagnostic Summary Visualization**: Advanced diagnostic plots for analysis quality assessment
  - 2×3 grid layout combining method comparison, parameter uncertainties, convergence diagnostics, and residuals analysis
  - Cross-method comparison with chi-squared values, parameter uncertainties, and MCMC convergence metrics
  - Residuals distribution analysis with normal distribution overlay and statistical summaries
  - Professional formatting with consistent styling, grid lines, and color coding

### Changed  
- **Diagnostic Summary Plot Generation**: Main `diagnostic_summary.png` plot only generated for `--method all` to provide meaningful cross-method comparisons
- **Classical Optimization Architecture**: Expanded from single-method to multi-method framework
- **Configuration Templates**: All JSON templates now include robust optimization and Gurobi method options
- **Method Selection**: Best optimization result automatically selected based on chi-squared value

### Fixed
- **Removed deprecated `--static` CLI argument**: Cleaned up legacy command-line argument that was replaced by `--static-anisotropic`
- **Removed unused profiler module**: Deleted `homodyne/core/profiler.py` and migrated functionality to `PerformanceMonitor` class in `core/config.py`
- **Fixed AttributeError in CLI**: Resolved `args.static` reference error that caused immediate crash on startup
- **Fixed test imports**: Updated all performance test imports to use new `PerformanceMonitor` API
- **Documentation updates**: Updated all documentation to reflect removed functionality and new API patterns
- **Type Safety**: Resolved all Pylance type checking issues for optional imports
- **Parameter Bounds**: Ensured consistent bounds across all optimization methods

### Performance
- **Enhanced stable benchmarking**: Added comprehensive statistics (mean, median, percentiles, outlier detection)
- **Performance test improvements**: Better reliability with JIT warmup and deterministic data
- **Bounds Constraints**: Gurobi provides native support for parameter bounds (unlike Nelder-Mead)

## [0.6.6] - 2025-08-27

### Added
- **Enhanced Shell Completion System**: Implemented multi-tier shell completion with robust fallback mechanisms
  - Fast standalone completion script (`homodyne_complete`) with zero package dependencies for instant performance
  - Comprehensive shell shortcuts: `hc` (classical), `hm` (mcmc), `hr` (robust), `ha` (all methods)
  - Silent loading option for completion system without startup notifications
  - Three-tier fallback system: tab completion → shortcuts → help system
- **Code Quality Improvements**: Comprehensive formatting and linting applied across entire codebase
  - Applied Black formatter (line length 88) to all Python files for consistent style
  - Applied isort import sorting with Black profile for organized imports
  - Enhanced type consistency and import organization
- **Documentation Updates**: Comprehensive updates to reflect shell completion enhancements
  - Updated CLI_REFERENCE.md with three-tier completion system documentation
  - Enhanced README.md with Shell Completion & Shortcuts section
  - Updated user-guide documentation with shell enhancement setup instructions

### Changed
- **Shell Completion Architecture**: Migrated from argcomplete-only to hybrid completion system
  - Added bypass mechanism for zsh compdef issues (`compdef:153: _comps: assignment to invalid subscript range`)
  - Implemented external completion handler with caching for performance optimization
  - Removed startup notification messages for silent shell loading
- **CLI Interface**: Enhanced user experience with improved completion and shortcuts
  - Completion system now gracefully degrades from tab completion to shortcuts to help
  - Added comprehensive troubleshooting section for completion issues

### Fixed  
- **Shell Completion Issues**: Resolved zsh compdef registration failures that broke tab completion
- **Completion Performance**: Optimized completion speed with aggressive caching and minimal file system operations
- **Documentation Consistency**: Updated version references across all documentation files
- **File Organization**: Cleaned up temporary completion files and consolidated working completion system

### Performance
- **Completion Speed**: Target < 50ms completion time achieved through zero-dependency completion script
- **Caching System**: Implemented intelligent file/directory caching with TTL for faster subsequent completions
- **Memory Optimization**: Minimal memory footprint for completion operations

## [Unreleased]

### Added
- **Comprehensive Code Quality Improvements**: Major cleanup and optimization of codebase quality
  - Fixed critical Gurobi optimization implementation that was non-iterative and getting stuck
  - Implemented proper iterative trust region SQP approach for Gurobi optimization
  - Removed unused function definitions (308 lines) from kernels.py fallback implementations  
  - Fixed all critical flake8 issues including false comparisons and import organization
  - Added missing fallback function definitions to resolve name errors
  - Enhanced Gurobi with adaptive trust region management and parameter-scaled finite differences

### Changed
- **Gurobi Optimization Implementation**: Complete rewrite from single-shot to iterative optimization
  - **Trust Region SQP**: Successive quadratic approximations with adaptive trust regions (0.1 → 1e-8 to 1.0 range)
  - **Iterative refinement**: Up to 50 outer iterations with convergence criteria based on gradient norm and objective improvement
  - **Numerical stability**: Parameter-scaled epsilon for finite differences and diagonal Hessian approximation
  - **Enhanced logging**: Debug messages showing iteration progress and convergence metrics
- **Code Quality Standards**: Updated formatting and import organization
  - **Black formatting**: Applied 88-character line length formatting to all files
  - **Import sorting**: Fixed import order with isort across all modules
  - **Type annotations**: Improved import patterns to resolve mypy redefinition warnings

### Fixed
- **Critical Gurobi Bug**: Gurobi optimization was building single quadratic approximation around initial point only
  - **Root Cause**: No iterative refinement meant χ² values remained constant across "iterations"
  - **Solution**: Implemented proper trust region optimization with step acceptance/rejection logic
  - **Expected Impact**: Progressive χ² improvement instead of constant values, proper convergence behavior
- **Code Quality Issues**: Resolved major flake8 and type checking problems
  - Fixed `== False` to `is False` comparisons in test files (7 locations)
  - Removed unused imports and variables in test modules
  - Added missing fallback functions `_solve_least_squares_batch_fallback` and `_compute_chi_squared_batch_fallback`
  - Improved import patterns in `test_cli_completion.py` to avoid redefinition warnings

## [0.6.4] - 2025-08-22

### Added
- **Gurobi Optimization Support**: Added Gurobi quadratic programming solver as alternative to Nelder-Mead
  - Automatic detection and graceful fallback when Gurobi not available
  - Quadratic approximation of chi-squared objective function using finite differences
  - Optimized configurations for different analysis modes (static 3-param, laminar flow 7-param)
  - Comprehensive test coverage with bounds constraint validation
- **Enhanced Documentation**: Updated all configuration templates with Gurobi options and usage guidance
- **Optimization Method Consistency**: All methods (Nelder-Mead, Gurobi, MCMC) use identical parameter bounds
- **Test Output Summary**: Added `-rs` flag to pytest configuration for always showing skip reasons
- **Performance Baselines**: Added comprehensive performance_baselines.json for regression tracking

### Changed
- **Classical Optimization Architecture**: Expanded from single-method to multi-method framework
- **Configuration Templates**: All JSON templates now include Gurobi method options
- **Package Dependencies**: Added optional Gurobi support in pyproject.toml and requirements.txt
- **Method Selection**: Best optimization result automatically selected based on chi-squared value
- **Test Cleanup**: Enhanced cleanup of test-generated homodyne_results directories

### Fixed
- **Type Safety**: Resolved all Pylance type checking issues for optional Gurobi imports
- **Parameter Bounds**: Ensured consistent bounds across all optimization methods
- **Test Performance**: Fixed config caching test parameter bounds validation
- **Performance Test Ratio**: Improved chi2_correlation_ratio_regression test with workload scaling
- **Test Cleanup**: Fixed automatic cleanup of homodyne/homodyne_results test artifacts
- **Performance Baselines Path**: Corrected baseline file path resolution in performance tests

### Performance
- **Bounds Constraints**: Gurobi provides native support for parameter bounds (unlike Nelder-Mead)
- **Quadratic Programming**: Potentially faster convergence for smooth, well-conditioned problems
- **Test Stability**: Improved performance test reliability with JIT warmup and deterministic data

## [0.6.3] - 2025-08-21

### Added
- **Advanced batch processing**: New `solve_least_squares_batch_numba` for vectorized least squares solving
- **Vectorized chi-squared computation**: Added `compute_chi_squared_batch_numba` for batch chi-squared calculation
- **Comprehensive optimization test suite**: Extended performance tests for Phase 3 batch optimizations

### Changed
- **Chi-squared calculation architecture**: Replaced sequential processing with vectorized batch operations
- **Memory access patterns**: Optimized for better cache locality and reduced memory allocations
- **Least squares solver**: Enhanced with direct 2x2 matrix math for maximum efficiency

### Performance
- **Breakthrough optimization**: Chi-squared calculation improved by 63.1% (546μs → 202μs)
- **Batch processing implementation**: Eliminated sequential angle processing with vectorized operations  
- **Performance ratio achievement**: Chi-squared/correlation ratio improved from 43x to 15.6x (64% reduction)
- **Memory layout optimization**: Enhanced cache efficiency through contiguous memory operations
- **Multi-phase optimization**: Combined variance pre-computation + Numba integration + batch vectorization
- **Total speedup factor**: 2.71x improvement over original implementation

## [0.6.2] - 2025-08-21

### Performance
- **Major performance optimizations**: Chi-squared calculation improved by 38% (1.33ms → 0.82ms)
- **Memory access optimization**: Replaced list comprehensions with vectorized reshape operations
- **Configuration caching**: Cached validation and chi-squared configs to avoid repeated dict lookups
- **Least squares optimization**: Replaced lstsq with solve() for 2x2 matrix systems (2x faster)
- **Memory pooling**: Pre-allocated result arrays to reduce allocation overhead
- **Vectorized operations**: Improved angle filtering with np.flatnonzero()
- **Performance ratio improvement**: Chi-squared/correlation ratio reduced from 6.0x to 1.7x

### Added
- **New optimization features**: Memory pooling, configuration caching, precomputed integrals
- **Performance regression tests**: Automated monitoring of performance baselines
- **Optimization test suite**: Comprehensive tests for new optimization features
- **Performance documentation**: Comprehensive performance guide (docs/performance.rst)
- **Enhanced benchmarking**: Updated performance baselines with optimization metrics

### Changed
- **Static case optimization**: Enhanced vectorized broadcasting for identical correlation functions
- **Parameter validation**: Added early returns and optimized bounds checking
- **Array operations**: Improved memory locality and reduced copy operations
- **Algorithm selection**: Better static vs laminar flow detection and handling

### Fixed
- **Memory efficiency**: Reduced garbage collection overhead through pooling
- **Numerical stability**: Preserved all validation logic while optimizing performance
- **JIT compatibility**: Maintained Numba acceleration with optimized pure Python paths

### Added
- Added @pytest.mark.memory decorators to memory-related tests for proper test collection

### Fixed
- Fixed GitHub test failure where memory tests were being deselected (exit code 5)
- Updated NumPy version constraints in setup.py, pyproject.toml, and requirements.txt for Numba 0.61.2 compatibility
- Fixed documentation CLI command references from python scripts to homodyne-config/homodyne commands

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

---

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