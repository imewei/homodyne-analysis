# Changelog

All notable changes to the Homodyne Scattering Analysis Package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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