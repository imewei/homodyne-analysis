# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the Homodyne Scattering Analysis Package - a high-performance Python package for analyzing homodyne scattering in X-ray Photon Correlation Spectroscopy (XPCS) under nonequilibrium conditions. The package implements theoretical frameworks for characterizing transport properties in flowing soft matter systems.

## Architecture

### Core Module Structure

- **`homodyne/core/`** - Foundation components
  - `config.py` - Configuration management with template system and parameter validation
  - `kernels.py` - Optimized computational kernels with Numba JIT compilation for correlation functions
  - `io_utils.py` - Data I/O handling for experimental data loading and result saving

- **`homodyne/analysis/`** - Main analysis engine
  - `core.py` - Primary analysis pipeline and chi-squared fitting implementation

- **`homodyne/optimization/`** - Multiple optimization approaches
  - `classical.py` - Classical methods (Nelder-Mead, Gurobi) with angle filtering
  - `robust.py` - Robust optimization (Wasserstein DRO, Scenario-based, Ellipsoidal) for noise-resistant analysis
  - `mcmc.py` - PyMC-based Bayesian parameter estimation with NUTS sampling

- **`homodyne/plotting.py`** - Comprehensive visualization for data validation and diagnostics

### Analysis Modes

The package supports three distinct physics-based analysis modes:

1. **Static Isotropic** (3 parameters) - Fastest, for isotropic systems with no angular dependence
2. **Static Anisotropic** (3 parameters) - Static systems with angular filtering enabled
3. **Laminar Flow** (7 parameters) - Full nonequilibrium flow analysis with shear effects

### Key Features

- **Three optimization approaches**: Classical (fast), Robust (noise-resistant), MCMC (uncertainty quantification)
- **High performance**: Numba JIT compilation provides 3-5x speedup
- **Noise resistance**: Robust optimization methods handle measurement uncertainty and outliers
- **Scientific accuracy**: Automatic g₂ = offset + contrast × g₁ fitting with consistent parameter bounds across all methods

## Development Commands

### Installation and Setup
```bash
# Development installation with all dependencies
make dev-install
# or manually: pip install -e ".[all,dev,docs]"

# Basic installation
make install
# or manually: pip install -e .
```

### Testing
```bash
# Run core tests
make test

# Run all tests with coverage
make test-all

# Run performance tests
make test-performance

# Run performance regression tests
make test-regression

# Fast test run (minimal dependencies)
make test-fast
```

### Code Quality
```bash
# Format code (Black, isort, Ruff)
make format

# Run linting (Ruff, mypy)
make lint

# Fix Ruff issues automatically
make ruff-fix

# Combined quality checks
make quality
```

### Performance Baseline Management
```bash
# Update performance baselines
make baseline-update

# Reset all baselines
make baseline-reset

# Generate performance report
make baseline-report
```

### Documentation
```bash
# Build documentation
make docs

# Serve documentation locally
make docs-serve
```

### Cleanup
```bash
# Clean all artifacts (preserves venv)
make clean

# Clean cache files only
make clean-cache

# Clean everything including venv
make clean-all
```

### Packaging
```bash
# Build distribution packages
make build

# Upload to PyPI
make upload

# Check package metadata
make check
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
make install-hooks
# or: pre-commit install

# Run hooks manually
make pre-commit
# or: pre-commit run --all-files
```

## CLI Usage

### Main Commands
```bash
# Primary analysis tool
homodyne [OPTIONS]

# Configuration generator
homodyne-config [OPTIONS]
```

### Common Analysis Workflows
```bash
# Create configuration for different modes
homodyne-config --mode static_isotropic    # 3-parameter, fastest
homodyne-config --mode static_anisotropic  # 3-parameter with angle filtering
homodyne-config --mode laminar_flow        # 7-parameter, full physics

# Run different optimization methods
homodyne --method classical    # Fast, point estimates
homodyne --method robust      # Noise-resistant optimization only
homodyne --method mcmc        # Bayesian uncertainty quantification
homodyne --method all         # All methods for comparison

# Data validation and plotting
homodyne --plot-experimental-data          # Validate input data quality
homodyne --plot-simulated-data            # Plot theoretical correlations
homodyne --plot-simulated-data --contrast 0.3 --offset 1.2  # With scaling
```

### Shell Completion
```bash
# Install completion (one-time setup)
homodyne --install-completion bash    # Linux/macOS
homodyne --install-completion zsh     # macOS default
homodyne --install-completion powershell  # Windows

# Available shortcuts
hc          # homodyne --method classical
hr          # homodyne --method robust
hm          # homodyne --method mcmc
ha          # homodyne --method all
```

## Configuration Management

### Configuration Files
- Use `homodyne-config` to generate template configurations
- Configuration files specify analysis mode and parameters
- Support for metadata (sample name, author, experiment details)

### Mode Selection Rules
```json
{
  "analysis_settings": {
    "static_mode": false,                    // → Laminar Flow (7 params)
    "static_mode": true,
    "static_submode": "isotropic"           // → Static Isotropic (3 params)
    "static_submode": "anisotropic"         // → Static Anisotropic (3 params)
  }
}
```

## Testing Framework

### Test Organization
- **Core tests**: Fast unit tests for basic functionality
- **Integration tests**: End-to-end analysis workflows
- **Performance tests**: Regression testing with baseline comparison
- **Markers**: Use pytest markers for test categorization (`slow`, `integration`, `performance`, `mcmc`, `jax`)

### Running Specific Test Categories
```bash
pytest -m "not slow"                    # Skip slow tests
pytest -m "performance"                 # Performance tests only
pytest -m "integration and not slow"    # Integration tests, exclude slow
```

## Dependencies and Optional Features

### Core Dependencies (always required)
- numpy, scipy, matplotlib

### Optional Feature Groups
- **Performance**: numba, jax (JIT compilation, GPU acceleration)
- **MCMC**: pymc, arviz, pytensor, corner (Bayesian analysis)
- **Robust**: cvxpy (robust optimization)
- **Gurobi**: gurobipy (commercial optimization solver, requires license)
- **Data**: xpcs-viewer (XPCS data handling)
- **Quality**: black, isort, ruff, mypy, bandit (code quality tools)
- **Test**: pytest suite with coverage and benchmarking
- **Docs**: sphinx documentation system

### Graceful Degradation
The package gracefully handles missing optional dependencies - features using unavailable dependencies will be disabled with appropriate warnings.

## Security and Code Quality

### Automated Quality Checks
- **Pre-commit hooks** enforce formatting, linting, and security scanning
- **Black**: Code formatting (88-character line length)
- **Ruff**: Fast linting with auto-fixes
- **Bandit**: Security vulnerability scanning (0 medium/high severity issues)
- **MyPy**: Type checking for scientific code patterns

### Security Configuration
- No hardcoded secrets or credentials
- Safe file operations with proper error handling
- Dependency vulnerability scanning with pip-audit
- Cross-platform compatibility (Windows, macOS, Linux)

## Performance Considerations

### Optimization Features
- **Numba JIT compilation**: 3-5x speedup for computational kernels
- **Vectorized operations**: NumPy-optimized array operations and angle filtering
- **Memory efficiency**: Lazy loading, memory-mapped files, smart caching
- **Angle filtering**: Reduces computational load in anisotropic modes

### Performance Testing
- Comprehensive regression testing with baseline comparison
- Benchmarking infrastructure with outlier filtering
- Memory profiling and optimization validation

## Output Structure

Results are organized in `./homodyne_results/` with method-specific subdirectories:
```
./homodyne_results/
├── homodyne_analysis_results.json  # Main results summary
├── run.log                         # Execution log
├── classical/                      # Classical optimization results
├── robust/                         # Robust optimization results
├── mcmc/                          # MCMC results with traces and plots
├── exp_data/                      # Experimental data validation plots
└── simulated_data/               # Simulated data visualization
```

Each method directory contains complete analysis results, fitted data arrays (NPZ format), parameters with uncertainties, and diagnostic visualizations.

## Physical Models

The package implements three key equations for correlation functions in nonequilibrium systems:

- **Equation 13**: Full nonequilibrium laminar flow
- **Equation S-75**: Equilibrium under constant shear
- **Equation S-76**: One-time correlation (Siegert relation)

Parameters represent time-dependent transport coefficients:
- **D(t)**: Anomalous diffusion coefficient
- **γ̇(t)**: Time-dependent shear rate
- **Scaling**: g₂ = offset + contrast × g₁ relationship

## Version Information

Current version: 0.7.1
- Python 3.12+ required
- Cross-platform Windows, macOS, Linux support
- Comprehensive pre-commit hooks and security scanning
- Enhanced shell completion with Windows compatibility