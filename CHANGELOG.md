# Changelog

All notable changes to the Homodyne Scattering Analysis Package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive README.md documentation converted from WARP.md
- Modern Python packaging with `pyproject.toml` and `setup.py`
- Sphinx documentation infrastructure with RTD theme
- GitHub Actions workflow for automated documentation deployment
- API documentation generation with `sphinx-apidoc`
- Custom CSS styling for documentation
- Package metadata and dependency management

### Changed
- Restructured documentation from WARP-specific to public-facing format
- Enhanced package discoverability with proper PyPI metadata
- Improved installation instructions with optional dependency groups

### Fixed
- Documentation build warnings and formatting issues
- Package structure for better import handling

## [6.0.0] - 2024-XX-XX

### Added
- **Triple Analysis Modes**: Static Isotropic (3 parameters), Static Anisotropic (3 parameters), and Laminar Flow (7 parameters) for comprehensive experimental coverage
- **Always-On Scaling Optimization**: Automatic g₂ = offset + contrast × g₁ fitting for scientifically accurate chi-squared calculations
- **Comprehensive Data Validation**: Experimental C2 data validation plots with standalone plotting capabilities
- **Enhanced Configuration System**: Mode-specific templates with intelligent defaults and metadata injection
- **Multiple Optimization Approaches**: Fast classical optimization (Nelder-Mead) for point estimates and robust Bayesian MCMC (NUTS) for full posterior distributions with uncertainty quantification
- **Performance Optimizations**: Numba JIT compilation for computational kernels, smart angle filtering, and memory-efficient data handling
- **Integrated Visualization**: Experimental data validation plots, parameter evolution tracking, MCMC convergence diagnostics, and corner plots for uncertainty visualization
- **Quality Assurance**: Extensive test coverage with pytest framework and performance benchmarking tools

### Core Features
- Support for nonequilibrium laminar flow analysis in X-ray Photon Correlation Spectroscopy (XPCS)
- Implementation of theoretical framework from He et al. PNAS 2024
- Time-dependent intensity correlation functions g₂(φ,t₁,t₂) analysis
- Transport coefficient characterization for soft matter systems
- Bayesian uncertainty quantification with MCMC sampling

### Technical Improvements
- Numba JIT compilation for 3-5x performance improvement
- Smart angle filtering for computational efficiency
- Memory-efficient data handling and caching
- JSON-based configuration system with validation
- Comprehensive error handling and logging

### Dependencies
- **Core**: numpy, scipy, matplotlib
- **Performance**: numba (recommended)
- **MCMC**: pymc, arviz, pytensor (optional)
- **Documentation**: sphinx, sphinx-rtd-theme, myst-parser

## [5.x.x] and Earlier

Previous versions focused on core algorithm development and initial implementation of homodyne scattering analysis capabilities. For detailed history of earlier versions, please refer to the git commit history.

---

**Legend:**
- `Added` for new features
- `Changed` for changes in existing functionality
- `Deprecated` for soon-to-be removed features
- `Removed` for now removed features
- `Fixed` for any bug fixes
- `Security` for vulnerability fixes
