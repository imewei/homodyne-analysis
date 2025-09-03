Changelog
=========

All notable changes to the Homodyne Scattering Analysis Package will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_, and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

## [0.7.2] - 2025-09-01

### Major: Unified System Integration & Enhanced Testing

**Complete architectural redesign** combining unified post-install system, enhanced testing framework, and streamlined user experience.

### Added

- **Unified Post-Install System**: Single command setup for all features
  - `homodyne-post-install --shell zsh --gpu --advanced` - one-command complete setup
  - `homodyne-post-install --interactive` - interactive feature selection
  - Automatic platform detection and environment-specific configuration

- **Advanced System Tools**: New CLI tools for optimization and validation
  - `homodyne-gpu-optimize` - Hardware-specific GPU optimization and benchmarking
  - `homodyne-validate` - Comprehensive system validation framework
  - `homodyne-cleanup` - Intelligent cleanup with dry-run support

- **Enhanced Virtual Environment Detection**: Multi-environment support
  - Support for conda, mamba, venv with backward compatibility
  - Unified detection logic across post-install and uninstall scripts

- **Comprehensive Test Marker System**: Organized testing for CI/CD optimization
  - `fast`/`slow`, `unit`/`integration`, `mcmc`, `gpu`, `system` markers
  - 263 focused CI tests instead of full suite for better performance

- **Smart GPU System**: Automatic detection and intelligent selection
  - Hardware-specific optimization profiles with memory management
  - Cross-platform fallback (Linux GPU, Windows/macOS CPU optimization)

- **Unified Shell Completion**: Cross-shell compatibility with consistent aliases
  - Single completion file supporting all shells
  - Unified aliases: `hm`, `hc`, `hr`, `ha` with smart GPU detection

### Fixed

- **Critical Windows Compatibility**: Fixed path separator issues in completion system
- **GPU Hardware Detection**: Proper CI testing with filesystem operation mocking
- **Pytest Configuration**: Resolved marker evaluation and collection conflicts
- **Cross-Platform Testing**: Enhanced Windows, macOS, and Linux compatibility
- **Code Quality**: Black formatting compliance across 92+ files

### Changed

- **Streamlined Architecture**: Consolidated codebase with merged installations
- **Enhanced User Experience**: One-command setup replacing multi-step configuration
- **Simplified Completion**: Automatic installation during package setup

### Removed

- **Legacy Shell Completion**: Cleaned up 600+ lines of unused completion logic
- **Manual CLI Options**: Removed `--install-completion` and `--uninstall-completion`

## [0.7.0] - 2025-08-28

### Enhanced Cross-Platform Compatibility

**Comprehensive Windows compatibility improvements** for shell completion system with better error handling and graceful degradation.

### Added

- **Cross-Platform Path Handling**: Support for Windows, macOS, and Linux path separators
- **Enhanced Error Messages**: Better cross-platform error handling with informative messages
- **Improved Test Reliability**: Enhanced test stability across all platforms

### Fixed

- **Windows Path Compatibility**: Fixed shell completion path separator handling
  - Support for both Windows backslash (`\`) and Unix forward slash (`/`)
  - Enhanced completion functions with `os.sep` detection
  - Resolved Windows CI test failures in completion system
- **Performance Tests**: Fixed arithmetic formatting in completion tests
- **Code Quality**: Comprehensive formatting and linting fixes with type safety improvements

### Compatibility

- **Windows**: Full shell completion with native path separator handling
- **macOS/Linux**: Maintained compatibility with enhanced cross-platform features
- **Backward Compatible**: No breaking changes

## [0.6.10] - 2025-08-28

### Shell Completion Enhancements

**Enhanced shell completion system** with uninstall functionality and improved cross-platform reliability.

### Added

- **Completion Uninstall**: New `homodyne --uninstall-completion` command for clean removal
- **Cross-Platform Support**: Enhanced reliability across bash, zsh, fish, and PowerShell
- **Developer Tools**: Manual completion triggers and convenience aliases
- **Enhanced Documentation**: Comprehensive examples and usage instructions

### Fixed

- **Command Completion**: Fixed `homodyne-config` completion options
- **Import Issues**: Resolved relative import failures in completion system
- **Shell Parsing**: Improved command-line argument handling
- **Zsh Fallback**: Enhanced fallback system for edge cases

## [0.6.9] - 2025-08-27

### Security & Quality Framework

**Comprehensive security framework** with integrated vulnerability scanning and best practices documentation.

### Added

- **Security Scanning**: Integrated Bandit and pip-audit for automated vulnerability detection
- **Security Documentation**: Complete guidelines in `docs/developer-guide/security.rst`
- **Quality Tools**: Enhanced development workflow with dependency scanning

### Fixed

- **CI Performance**: Adjusted test thresholds for CI environment compatibility
- **Tool Configuration**: Fixed Bandit configuration for scientific Python patterns

### Security

- **Zero Security Issues**: 0 medium/high severity issues through comprehensive scanning
- **Dependency Security**: Automated vulnerability checking with pip-audit
- **Secure Development**: Security-first practices and documentation

## [0.6.8] - 2025-08-27

### Stability & Compatibility Fixes

**Cross-platform reliability improvements** with enhanced test suite and consistent formatting.

### Fixed

- **Cross-Platform Issues**: Windows path separator fixes in completion tests
- **Import Errors**: Resolved AttributeError in isotropic mode integration tests
- **Template Handling**: Fixed MODE_DEPENDENT placeholder resolution
- **Performance Tests**: Adjusted thresholds for CI environments
- **Code Quality**: Import sorting and formatting consistency

### Improved

- **Test Reliability**: Consistent GitHub Actions tests across all platforms
- **Code Formatting**: Applied black formatter and isort throughout codebase

## [0.6.5] - 2024-11-24

### Robust Optimization Framework

**Complete robust optimization implementation** for noise-resistant parameter estimation with comprehensive visualization.

### Added

- **Robust Methods**: Three optimization approaches
  - Robust-Wasserstein (DRO), Robust-Scenario (Bootstrap), Robust-Ellipsoidal
  - CVXPY + Gurobi integration for convex optimization
  - Dedicated `--method robust` command-line flag

- **Individual Method Results**: Comprehensive saving system
  - Method-specific directories with JSON parameters and uncertainties
  - NumPy archives with complete numerical data
  - Summary files for cross-method comparison

- **Diagnostic Visualizations**: Advanced analysis quality assessment
  - 2×3 grid layout with method comparison and residuals analysis
  - Cross-method chi-squared comparison and convergence metrics
  - Professional formatting with consistent styling

### Changed

- **Architecture**: Expanded from single-method to multi-method framework
- **Templates**: All configurations include robust and Gurobi options
- **Method Selection**: Automatic selection based on chi-squared values

### Fixed

- **CLI Cleanup**: Removed deprecated `--static` argument and unused profiler module
- **Error Resolution**: Fixed AttributeError crashes and import issues
- **Type Safety**: Resolved Pylance type checking issues for optional imports

## [0.6.6] - 2025-08-27

### Enhanced Shell Completion System

**Multi-tier shell completion** with robust fallback mechanisms and performance optimization.

### Added

- **Shell Completion**: Multi-tier system with fallback mechanisms
  - Fast standalone script with zero dependencies (< 50ms target)
  - Comprehensive shortcuts: `hc`, `hm`, `hr`, `ha` for different methods
  - Three-tier fallback: tab completion → shortcuts → help system

- **Code Quality**: Comprehensive formatting and linting
  - Black formatter (88-char lines) and isort import sorting
  - Enhanced type consistency and import organization

### Changed

- **Completion Architecture**: Hybrid system replacing argcomplete-only approach
- **CLI Interface**: Graceful degradation with improved user experience

### Fixed

- **Zsh Issues**: Resolved compdef registration failures breaking tab completion
- **Performance**: Optimized speed with caching and minimal filesystem operations

### Performance

- **Speed**: < 50ms completion time with intelligent caching system
- **Memory**: Minimal footprint for completion operations

## [Unreleased]

### Isolated MCMC Backend Architecture

**Revolutionary backend separation** completely isolating PyMC CPU and NumPyro GPU implementations to prevent PyTensor/JAX conflicts while enabling optimal performance across all platforms.

### Added

- **Isolated CPU Backend (`mcmc_cpu_backend.py`)**: Pure PyMC implementation completely separated from JAX
  - Environment variable isolation with dedicated PyTensor CPU configuration
  - No JAX imports to prevent namespace contamination
  - Cross-platform compatibility (Linux, macOS, Windows)

- **Isolated GPU Backend (`mcmc_gpu_backend.py`)**: Pure NumPyro/JAX implementation completely separated from PyMC  
  - Dedicated JAX environment configuration with GPU detection and CPU fallback
  - No PyMC imports to prevent PyTensor conflicts
  - Hardware-optimized performance on Linux, CPU fallback on macOS/Windows

- **Backend Separation System**: Complete architectural redesign
  - Updated `run_homodyne.py` with `get_mcmc_backend()` for intelligent backend selection
  - Environment-based backend selection via `HOMODYNE_GPU_INTENT`
  - Isolated environment configuration per backend

- **Enhanced Package Structure**: Updated packaging for isolated architecture
  - Split MCMC dependencies: `[mcmc]`, `[mcmc-gpu]`, `[mcmc-all]` installation options
  - New requirements files for granular dependency management
  - Updated `pyproject.toml`, `MANIFEST.in`, and `setup.py` for isolated backends

- **Updated Testing Framework**: Comprehensive tests for isolated backends
  - Backend isolation verification tests
  - Updated test imports and descriptions to reflect architecture
  - Performance validation for both backends

- **Enhanced Installation System**: Updated post-install and cleanup scripts
  - `homodyne-post-install` messages reflect isolated backend architecture
  - `homodyne-cleanup` updated for isolated backend file management
  - Installation guides updated for new dependency structure

### Changed

- **MCMC Architecture**: Complete transformation from mixed to isolated backend system
  - `mcmc.py`: Now pure PyMC CPU implementation (removed JAX/environment configuration code)
  - `mcmc_gpu.py`: Remains pure NumPyro/JAX implementation (unchanged core logic)
  - Backend selection moved to dedicated wrapper modules for complete isolation

- **Command Usage**: Enhanced clarity for backend selection
  - `homodyne --method mcmc`: Uses isolated PyMC CPU backend
  - `homodyne-gpu --method mcmc`: Uses isolated NumPyro/JAX GPU backend with CPU fallback
  - Smart aliases (`hm`, `ha`) automatically select appropriate backend

- **Documentation**: Comprehensive updates reflecting isolated architecture
  - README.md updated with backend-specific installation and usage instructions
  - CLAUDE.md updated with isolated backend development guidelines
  - All .md files updated to reflect new architecture

### Fixed

- **PyTensor/JAX Conflicts**: Resolved critical namespace conflicts that occurred when both backends were imported
  - CPU backend: No longer imports JAX, completely isolated from JAX ecosystem
  - GPU backend: No longer imports PyMC/PyTensor, completely isolated from PyMC ecosystem  
  - Environment variables properly isolated per backend

- **macOS/Windows Compatibility**: Enhanced cross-platform support
  - GPU backend automatically falls back to JAX CPU mode on non-Linux systems
  - CPU backend works identically across all platforms
  - No more JAX GPU errors on macOS/Windows systems

- **Test Framework**: Updated test imports and expectations
  - Fixed test failures due to removed `_lazy_import_jax` from CPU backend
  - Updated attribute expectations (`trace` → `mcmc_trace`) to match implementation
  - Enhanced backend isolation verification tests

- **Installation**: Resolved dependency conflicts and installation issues
  - Users can now install CPU-only (`[mcmc]`) or GPU-only (`[mcmc-gpu]`) dependencies
  - No more mixed dependency conflicts during installation
  - Clean separation of PyMC and NumPyro dependency trees

### Benefits

- **Complete Conflict Resolution**: Eliminates all PyTensor/JAX namespace conflicts and import issues
- **Platform Flexibility**: 
  - Linux: Full GPU acceleration with CUDA 12.6+ for GPU backend
  - macOS/Windows: Stable CPU-only operation for both backends with JAX CPU fallback
- **Installation Granularity**: Users can install only the backend they need
  - `pip install homodyne-analysis[mcmc]` - CPU-only PyMC backend
  - `pip install homodyne-analysis[mcmc-gpu]` - GPU-capable NumPyro/JAX backend
  - `pip install homodyne-analysis[mcmc-all]` - Both backends for maximum flexibility
- **Development Independence**: Each backend can evolve independently without affecting the other
- **Performance Optimization**: Each backend optimized for its specific use case and environment

### Architecture Impact

- **File Structure**: 
  - `mcmc_cpu_backend.py`: 150+ lines of isolated CPU backend wrapper
  - `mcmc_gpu_backend.py`: 180+ lines of isolated GPU backend wrapper
  - `mcmc.py`: Cleaned 50+ lines of obsolete environment configuration
  - `mcmc_gpu.py`: Unchanged core implementation, enhanced isolation
- **Code Reduction**: Eliminated 100+ lines of mixed backend configuration code
- **Test Coverage**: Enhanced with 15+ new backend isolation tests
- **Documentation**: 500+ lines updated across all .md files

## [0.6.4] - 2025-08-22

### Gurobi Optimization Support

**Added Gurobi quadratic programming** as alternative to Nelder-Mead with automatic detection and graceful fallback.

### Added

- **Gurobi Solver**: Quadratic programming alternative to Nelder-Mead
  - Automatic detection with graceful fallback
  - Quadratic approximation using finite differences
  - Comprehensive test coverage with bounds validation

- **Enhanced Templates**: Updated configurations with Gurobi options
- **Performance Tracking**: Comprehensive baselines for regression detection

### Changed

- **Architecture**: Multi-method framework with automatic method selection
- **Dependencies**: Optional Gurobi support in package configuration
- **Test Cleanup**: Enhanced cleanup of generated results directories

### Fixed

- **Type Safety**: Resolved Pylance issues for optional Gurobi imports
- **Bounds Consistency**: Uniform parameter bounds across all methods
- **Test Reliability**: Improved performance test stability

### Performance

- **Native Bounds**: Gurobi provides built-in parameter bounds support
- **Convergence**: Potentially faster for smooth, well-conditioned problems

## [0.6.3] - 2025-08-21

### Performance Breakthrough

**Major performance optimizations** achieving 63.1% improvement in chi-squared calculations through vectorized batch processing.

### Added

- **Vectorized Processing**: Batch chi-squared and least squares computation
- **Advanced Algorithms**: `solve_least_squares_batch_numba` and `compute_chi_squared_batch_numba`
- **Performance Testing**: Extended test suite for batch optimization validation

### Changed

- **Architecture**: Vectorized batch operations replacing sequential processing
- **Memory Access**: Optimized cache locality and reduced allocations
- **Solver**: Direct 2x2 matrix math for maximum efficiency

### Performance

- **63.1% Improvement**: Chi-squared calculation (546μs → 202μs)
- **Ratio Improvement**: Chi-squared/correlation ratio (43x → 15.6x, 64% reduction)
- **Total Speedup**: 2.71x improvement over original implementation

## [0.6.2] - 2025-08-21

### Performance Optimizations

**Major performance improvements** with 38% faster chi-squared calculations and comprehensive optimization features.

### Added

- **Optimization Features**: Memory pooling, configuration caching, precomputed integrals
- **Performance Testing**: Regression tests and comprehensive benchmarking
- **Documentation**: Performance guide (docs/performance.rst)

### Changed

- **Memory Access**: Vectorized operations replacing list comprehensions
- **Algorithm Selection**: Better static vs laminar flow detection
- **Array Operations**: Improved locality and reduced copy operations

### Fixed

- **Test Collection**: Fixed memory test deselection issues
- **NumPy Compatibility**: Updated version constraints for Numba 0.61.2
- **Documentation**: Fixed CLI command references

### Performance

- **38% Improvement**: Chi-squared calculation (1.33ms → 0.82ms)
- **Ratio Improvement**: Chi-squared/correlation ratio (6.0x → 1.7x)
- **Memory Efficiency**: Reduced garbage collection through pooling

## [0.6.1] - 2025-08-21

### Performance Testing Framework

**Enhanced JIT warmup** and stable benchmarking with comprehensive performance infrastructure.

### Added

- **JIT Warmup**: Comprehensive function-level compilation system
- **Benchmarking**: Statistical outlier filtering and pytest-benchmark integration
- **Performance Tracking**: Baseline tracking and regression detection
- **Type Safety**: Enhanced annotations and consistency checks

### Changed

- **Test Reliability**: 60% reduction in performance variance (CV)
- **Baselines**: Updated to reflect realistic JIT-compiled expectations
- **Environment Optimization**: Consolidated utilities reducing code duplication

### Fixed

- **Variability**: Fixed correlation calculation benchmark inconsistencies
- **Type Issues**: Resolved annotation problems in plotting and core modules
- **Matplotlib**: Fixed colormap access for better compatibility

### Performance

- **Variance Reduction**: JIT functions from >100% to ~26% CV
- **Stability**: Enhanced warmup and outlier detection

## [2024.1.0] - Previous Release

### Initial Release

**Foundation release** with core homodyne scattering analysis implementation.

### Added

- **Analysis Modes**: Static Isotropic, Static Anisotropic, Laminar Flow
- **Optimization**: Classical (Nelder-Mead) and Bayesian MCMC (NUTS) methods
- **Visualization**: Comprehensive plotting and visualization capabilities
- **Configuration**: Management system with validation
- **Performance**: Numba JIT compilation optimizations
- **Testing**: Comprehensive suite with 361+ tests
- **Interface**: Command-line and Python API

---

## Versioning & Categories

This project adheres to [Semantic Versioning](https://semver.org/):

- **Major**: Breaking API changes
- **Minor**: New features, performance improvements  
- **Patch**: Bug fixes, documentation updates

**Change Categories**: Added, Changed, Fixed, Removed, Security, Performance
