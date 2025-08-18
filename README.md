# Homodyne Scattering Analysis Package

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![Numba](https://img.shields.io/badge/Numba-JIT%20Accelerated-green)](https://numba.pydata.org/)

A comprehensive Python package for analyzing homodyne scattering in X-ray Photon Correlation Spectroscopy (XPCS) under nonequilibrium conditions. This package implements the theoretical framework described in [He et al. PNAS 2024](https://doi.org/10.1073/pnas.2401162121) for characterizing nonequilibrium dynamics in soft matter systems through detailed transport coefficient analysis.

## Overview

### Physical Context

The package analyzes time-dependent intensity correlation functions c₂(φ,t₁,t₂) for complex fluids under nonequilibrium laminar flow conditions. It captures the interplay between Brownian diffusion and advective shear flow, enabling quantitative characterization of transport properties in flowing soft matter systems.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Analysis Modes](#analysis-modes)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Configuration Management](#configuration-management)
  - [Data Validation](#data-validation)
- [Scaling Optimization](#scaling-optimization)
- [Optimization Methods](#optimization-methods)
- [Performance Optimization](#performance-optimization)
- [Testing](#testing)
- [Package Architecture](#package-architecture)
- [Troubleshooting](#troubleshooting)
- [Migration Guide](#migration-guide)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)

## Installation

### Development Installation

Since the package is not yet published to PyPI, install directly from the repository:

```bash
# Clone the repository
git clone https://github.com/imewei/homodyne.git
cd homodyne

# Set up Python environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Or with Miniforge3
mamba create -n homodyne python>=3.12
mamba activate homodyne

# Install core dependencies
pip install numpy scipy matplotlib

# For XPCS data handling (recommended)
pip install xpcs-viewer

# For enhanced performance (recommended)
pip install numba

# For Bayesian MCMC analysis
pip install pymc arviz pytensor

# Install the package in development mode
pip install -e .
```

### Dependencies

**Core Requirements**: `numpy`, `scipy`, `matplotlib`  
**Data Handling**: `xpcs-viewer` (specialized XPCS data loading and manipulation)  
**Performance Enhancement**: `numba` (provides 3-5x speedup via JIT compilation)  
**Bayesian Analysis**: `pymc`, `arviz`, `pytensor` (for MCMC sampling and diagnostics)  
**Optional Test Dependencies**: `pytest`, `pytest-cov`, `pytest-xdist`, `pytest-benchmark`, `pytest-mock`, `hypothesis`, `black`, `flake8`, `mypy`  
**Optional Documentation Dependencies**: `sphinx`, `sphinx-rtd-theme`, `myst-parser`, `sphinx-autodoc-typehints`, `numpydoc`

## Quick Start

```python
from homodyne import HomodyneAnalysisCore, ConfigManager

# Load configuration
config = ConfigManager("my_experiment.json")

# Initialize analysis
analysis = HomodyneAnalysisCore(config)

# Run classical optimization
results = analysis.optimize_classical()

# Or run both classical and MCMC
results = analysis.optimize_all()
```

Or from the command line:

```bash
# Basic analysis with isotropic mode (fastest)
python run_homodyne.py --static-isotropic --method classical

# Full flow analysis with uncertainty quantification
python run_homodyne.py --laminar-flow --method mcmc
```

## Analysis Modes

The homodyne analysis package supports three distinct analysis modes, each optimized for different experimental scenarios:

| Mode | Parameters | Angle Handling | Use Case | Speed | Command |
|------|------------|----------------|----------|-------|---------|
| **Static Isotropic** | 3 | Single dummy | Fastest, isotropic systems | ⭐⭐⭐ | `--static-isotropic` |
| **Static Anisotropic** | 3 | Filtering enabled | Static with angular deps | ⭐⭐ | `--static-anisotropic` |
| **Laminar Flow** | 7 | Full coverage | Flow & shear analysis | ⭐ | `--laminar-flow` |

### Static Isotropic Mode (3 parameters)
- **Physical Context**: Analysis of systems at equilibrium with isotropic scattering where results don't depend on scattering angle
- **Parameters**: 
  - D₀: Effective diffusion coefficient
  - α: Time exponent characterizing dynamic scaling
  - D_offset: Baseline diffusion component
- **Key Features**:
  - No angle filtering (automatically disabled)
  - No phi_angles_file loading (uses single dummy angle)
  - Fastest analysis mode
- **When to Use**: Isotropic samples, quick validation runs, preliminary analysis
- **Model**: `g₁(t₁,t₂) = exp(-q² ∫ᵗ²ᵗ¹ D(t)dt)` with no angular dependence

### Static Anisotropic Mode (3 parameters)
- **Physical Context**: Analysis of systems at equilibrium with angular dependence but no flow effects
- **Parameters**: D₀, α, D_offset (same as isotropic mode)
- **Key Features**:
  - Angle filtering enabled for optimization efficiency
  - phi_angles_file loaded for angle information
  - Per-angle scaling optimization
- **When to Use**: Static samples with measurable angular variations, moderate computational resources
- **Model**: Same as isotropic mode but with angle filtering to focus optimization on specific angular ranges

### Laminar Flow Mode (7 parameters) 
- **Physical Context**: Analysis of systems under controlled shear flow conditions with full physics model
- **Parameters**: 
  - D₀, α, D_offset: Same as static modes
  - γ̇₀: Characteristic shear rate
  - β: Shear rate exponent for flow scaling
  - γ̇_offset: Baseline shear component
  - φ₀: Angular offset parameter for flow geometry
- **Key Features**:
  - All flow and diffusion effects included
  - phi_angles_file required for angle-dependent flow effects
  - Complex parameter space with potential correlations
- **When to Use**: Systems under shear, nonequilibrium conditions, transport coefficient analysis
- **Model**: `g₁(t₁,t₂) = g₁_diff(t₁,t₂) × g₁_shear(t₁,t₂)` where shear effects are `sinc²(Φ)`

## Usage

### Command Line Interface

#### Main Analysis Runner

```bash
# Basic classical optimization with mode specification
python run_homodyne.py --static-isotropic --method classical
python run_homodyne.py --static-anisotropic --method mcmc
python run_homodyne.py --laminar-flow --method all

# Use custom configuration
python run_homodyne.py --config my_experiment.json --output-dir ./results

# Generate experimental data validation plots
python run_homodyne.py --plot-experimental-data --verbose

# Combine validation with analysis
python run_homodyne.py --plot-experimental-data --method all --verbose
```

#### Data Validation

Generate comprehensive validation plots of experimental C2 correlation data:

```bash
# Basic data validation
python run_homodyne.py --plot-experimental-data --config my_config.json

# Verbose validation with debug logging
python run_homodyne.py --plot-experimental-data --config my_config.json --verbose
```

**Output**: Creates validation plots in `./plots/data_validation/` including:
- Full 2D correlation function heatmaps g₂(t₁,t₂) for each angle
- Diagonal slices g₂(t,t) showing temporal decay
- Cross-sectional profiles at different time points
- Statistical summaries with data quality metrics

### Configuration Management

#### Template System
The package provides mode-specific configuration templates optimized for different analysis scenarios:

- **`config_static_isotropic.json`**: Optimized for isotropic analysis with single dummy angle
- **`config_static_anisotropic.json`**: Static analysis with angle filtering enabled
- **`config_laminar_flow.json`**: Full flow analysis with all 7 parameters
- **`config_template.json`**: Master template with comprehensive documentation

#### Configuration Creation
Generate analysis configurations using the enhanced `create_config.py`:

```bash
# Create isotropic static configuration (fastest)
python create_config.py --mode static_isotropic --sample protein_01

# Create anisotropic static configuration with metadata
python create_config.py --mode static_anisotropic --sample collagen \
                        --author "Your Name" --experiment "Static analysis"

# Create flow analysis configuration
python create_config.py --mode laminar_flow --sample microgel \
                        --experiment "Microgel dynamics under shear"

# Create with custom output file
python create_config.py --mode static_isotropic --output my_isotropic_config.json
```

#### Mode Selection Logic
Configuration files specify analysis mode through:

```json
{
  "analysis_settings": {
    "static_mode": true/false,
    "static_submode": "isotropic" | "anisotropic" | null
  }
}
```

**Mode Selection Rules**:
- `static_mode: false` → **Laminar Flow Mode**
- `static_mode: true, static_submode: "isotropic"` → **Static Isotropic Mode**  
- `static_mode: true, static_submode: "anisotropic"` → **Static Anisotropic Mode**
- `static_mode: true, static_submode: null` → **Static Anisotropic Mode** (default)

#### Active Parameters System
Specify which parameters to optimize and display in plots:

```json
{
  "initial_parameters": {
    "active_parameters": ["D0", "alpha", "D_offset"]
  }
}
```

**Mode-Specific Defaults**:
- **Static Modes**: `["D0", "alpha", "D_offset"]` (3 parameters)
- **Laminar Flow**: `["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]` (7 parameters)

### Data Validation and Quality Control

#### Integrated Validation
Enable experimental data plotting within the main analysis workflow:

```bash
python run_homodyne.py --plot-experimental-data --verbose
```

#### Config-based Validation
Enable experimental data plotting via configuration file:

```json
{
  "workflow_integration": {
    "analysis_workflow": {
      "plot_experimental_data_on_load": true
    }
  }
}
```

**Quality indicators to look for:**
- Mean values around 1.0 (expected for g₂ correlation functions)
- Enhanced diagonal values (should be higher than off-diagonal)
- Sufficient contrast (> 0.001) indicating dynamic signal
- Consistent structure across different angles

## Scaling Optimization

**Scaling optimization is now always enabled** across all analysis modes for scientifically accurate results.

### Mathematical Relationship
The scaling optimization determines the optimal relationship between experimental and theoretical correlation functions:

```
g₂ = offset + contrast × g₁
```

Where:
- **g₁**: Theoretical correlation function
- **g₂**: Experimental correlation function  
- **contrast**: Fitted scaling parameter (multiplicative factor)
- **offset**: Fitted baseline parameter (additive factor)

### Physical Significance
Scaling optimization accounts for systematic factors present in experimental data:
- **Instrumental response functions**: Detector and optical system responses
- **Background signals**: Electronic noise, scattered light, dark current
- **Detector gain variations**: Pixel-to-pixel sensitivity differences
- **Normalization differences**: Systematic differences in data processing

### Implementation
The optimal parameters are determined using least squares solution:
```python
A = np.vstack([theory, np.ones(len(theory))]).T
scaling, residuals, _, _ = np.linalg.lstsq(A, exp, rcond=None)
contrast, offset = scaling
fitted = theory * contrast + offset
```

This provides meaningful chi-squared statistics: `χ² = Σ(experimental - fitted)²/σ²`

## Optimization Methods

### Classical Optimization
- **Algorithm**: Nelder-Mead simplex method
- **Performance**: Fast execution (~minutes for typical datasets)
- **Output**: Point estimates with goodness-of-fit statistics
- **Best For**: Exploratory analysis, parameter screening, computational efficiency requirements
- **Command**: `--method classical`

### Bayesian MCMC Sampling
- **Algorithm**: NUTS (No-U-Turn Sampler) via PyMC
- **Performance**: Comprehensive but slower (~hours depending on data size)
- **Output**: Full posterior distributions, uncertainty quantification, convergence diagnostics
- **Best For**: Robust parameter estimation, uncertainty analysis, publication-quality results
- **Command**: `--method mcmc`
- **Additional Requirements**: `pip install pymc arviz pytensor`

### Combined Analysis
- **Recommended Workflow**: Use classical optimization for initial parameter estimates, then refine with MCMC for full uncertainty analysis
- **Command**: `--method all` (runs both methods sequentially)

**Note**: Scaling optimization (g₂ = offset + contrast × g₁) is always enabled in all methods for consistent and scientifically accurate chi-squared calculations.

## Performance Optimization

### Environment Variables
```bash
# Optimize BLAS/threading for performance
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Disable Intel SVML for Numba compatibility
export NUMBA_DISABLE_INTEL_SVML=1
```

### Configuration Tuning
- **Angle Filtering**: Enable in config to focus optimization on specific angular ranges ([-10°, 10°] and [170°, 190°])
- **Numba JIT**: Automatically enabled when available; provides 3-5x speedup
- **Memory Limits**: Adjust `memory_limit_gb` in configuration based on available RAM

### Performance Benchmarking
Use the comprehensive benchmarking suite to validate optimizations:

```bash
# Full performance analysis
python benchmark_performance.py --iterations 50 --size 1000

# Quick performance check
python benchmark_performance.py --fast
```

**Benchmarked Components**:
- Computational kernels with Numba JIT acceleration
- Optimized matrix operations and vectorized functions
- Configuration loading and caching performance
- Memory efficiency and allocation patterns

## Testing

The package includes a comprehensive test suite using pytest:

```bash
# Standard test run
python homodyne/run_tests.py

# Fast tests only (exclude slow integration tests)
python homodyne/run_tests.py --fast

# Run with coverage reporting
python homodyne/run_tests.py --coverage

# Run tests in parallel
python homodyne/run_tests.py --parallel 4

# Verbose test output
python homodyne/run_tests.py --verbose

# Test specific functionality
python homodyne/run_tests.py --markers "integration"
python homodyne/run_tests.py -k "static_mode"
```

**Test Categories**: Core functionality, mode-specific behavior, I/O operations, plotting, integration workflows  
**Enhanced Coverage**: Static mode analysis, isotropic mode integration, MCMC features, angle filtering  
**Quality Metrics**: Extensive coverage of critical code paths with mode-specific validation  
**Performance Tests**: Benchmarking and regression detection  
**Data Validation**: Ensuring numerical accuracy and consistency across modes

## Package Architecture

### Core Components

- **`homodyne/core/`**: Central infrastructure including configuration management (`ConfigManager`), optimized computational kernels, and flexible I/O utilities with JSON serialization support
- **`homodyne/analysis/`**: Main analysis engine (`HomodyneAnalysisCore`) handling experimental data loading, correlation function calculations, and chi-squared fitting
- **`homodyne/optimization/`**: Dual optimization framework with classical methods (`ClassicalOptimizer`) and Bayesian MCMC sampling (`MCMCSampler`)
- **`homodyne/plotting.py`**: Comprehensive visualization system for data validation, parameter analysis, and diagnostic plotting

### Key Classes and Functions

- **`ConfigManager`**: Robust JSON configuration handling with mode detection, template-based creation, validation, and runtime parameter override capabilities
- **`HomodyneAnalysisCore`**: Primary analysis engine managing experimental data loading, preprocessing, and chi-squared objective function calculations with mode-specific behavior
- **`ClassicalOptimizer`**: Scipy-based optimization with intelligent angle filtering and performance monitoring
- **`MCMCSampler`**: PyMC-based Bayesian parameter estimation using NUTS sampling with convergence diagnostics
- **Optimized Computational Kernels**: Enhanced performance kernels including `create_symmetric_matrix_optimized`, `matrix_vector_multiply_optimized`, `apply_scaling_vectorized`, `compute_chi_squared_fast`, and `exp_negative_vectorized`

### File Structure

```
homodyne/
├── run_homodyne.py              # Main CLI entry point with integrated data validation
├── create_config.py             # Enhanced configuration generator with mode selection
├── benchmark_performance.py     # Performance benchmarking suite
├── README.md                    # This comprehensive guide
├── LICENSE                      # MIT License
├── setup.py                     # Package setup configuration
├── pyproject.toml              # Modern Python packaging configuration
├── requirements.txt            # Package dependencies
├── MANIFEST.in                 # Package manifest
├── .gitignore                  # Git ignore patterns
├── docs/                       # Sphinx documentation
│   ├── Makefile               # Documentation build configuration
│   ├── conf.py                # Sphinx configuration
│   ├── index.rst              # Main documentation index
│   ├── _static/               # Static documentation assets
│   │   └── .gitkeep          # Git keep file for empty directory
│   ├── user-guide/            # User documentation
│   │   ├── installation.rst   # Installation guide
│   │   ├── quickstart.rst     # Quick start guide
│   │   ├── configuration.rst  # Configuration documentation
│   │   ├── analysis-modes.rst # Analysis modes documentation
│   │   └── examples.rst       # Usage examples
│   ├── api-reference/         # API documentation
│   │   ├── index.rst          # API reference index
│   │   ├── core.rst           # Core API documentation
│   │   ├── mcmc.rst           # MCMC API documentation
│   │   ├── models.rst         # Models API documentation
│   │   ├── utilities.rst      # Utilities API documentation
│   │   └── _autosummary/      # Auto-generated API docs
│   │       ├── homodyne.config.rst     # Config module docs
│   │       ├── homodyne.core.rst       # Core module docs
│   │       ├── homodyne.models.rst     # Models module docs
│   │       ├── homodyne.optimization.mcmc.rst  # MCMC docs
│   │       └── homodyne.utils.rst      # Utils module docs
│   └── developer-guide/       # Developer documentation
│       ├── index.rst          # Developer guide index
│       ├── architecture.rst   # System architecture
│       ├── contributing.rst   # Contributing guidelines
│       ├── testing.rst        # Testing documentation
│       ├── performance.rst    # Performance optimization
│       └── troubleshooting.rst # Troubleshooting guide
├── homodyne/                   # Main package
│   ├── __init__.py            # Package exports and version (v6.0)
│   ├── .coveragerc            # Test coverage configuration
│   ├── config_static_isotropic.json   # Template for isotropic analysis
│   ├── config_static_anisotropic.json # Template for anisotropic analysis
│   ├── config_laminar_flow.json       # Template for flow analysis
│   ├── config_template.json   # Master template with comprehensive documentation
│   ├── run_tests.py           # Enhanced test runner with coverage and parallel options
│   ├── plotting.py            # Comprehensive visualization utilities
│   ├── core/                  # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration management with mode detection
│   │   ├── kernels.py         # Computational kernels (enhanced with optimized functions)
│   │   └── io_utils.py        # Data I/O utilities
│   ├── analysis/              # Analysis engines
│   │   ├── __init__.py
│   │   └── core.py            # Main analysis class with mode-specific behavior
│   ├── optimization/          # Optimization methods
│   │   ├── __init__.py
│   │   ├── classical.py       # Scipy-based optimization
│   │   └── mcmc.py            # PyMC Bayesian sampling
│   └── tests/                 # Comprehensive test suite
│       ├── __init__.py
│       ├── conftest.py        # Pytest configuration
│       ├── fixtures.py        # Test fixtures and utilities
│       ├── test_angle_filtering.py      # Angle filtering functionality
│       ├── test_classical_config_reading.py  # Classical config tests
│       ├── test_config.py               # Configuration management tests
│       ├── test_config_integration.py   # Config integration tests
│       ├── test_config_json.py          # JSON configuration tests
│       ├── test_integration.py          # Integration testing
│       ├── test_io_utils.py             # I/O utilities tests
│       ├── test_isotropic_mode_integration.py  # Isotropic mode integration
│       ├── test_mcmc_angle_filtering.py # MCMC angle filtering tests
│       ├── test_mcmc_config_reading.py  # MCMC config tests
│       ├── test_mcmc_config_regression.py # MCMC regression tests
│       ├── test_mcmc_config_validation.py # MCMC validation tests
│       ├── test_mcmc_convergence_diagnostics.py # MCMC convergence tests
│       ├── test_mcmc_initial_parameters.py # MCMC parameter tests
│       ├── test_mcmc_parameter_bounds_regression.py # MCMC bounds tests
│       ├── test_mcmc_scaling_consistency.py # MCMC scaling tests
│       ├── test_per_angle_chi_squared.py # Per-angle analysis tests
│       ├── test_plotting.py             # Plotting functionality tests
│       ├── test_save_results.py         # Results saving tests
│       ├── test_static_mode.py          # Static mode functionality
│       ├── test_targeted_mcmc_features.py # Targeted MCMC tests
│       └── test_utils_mcmc.py           # MCMC utilities tests
└── my_config.json              # Example configuration file
```

## Troubleshooting

### Common Issues

**Missing Dependencies**:
```bash
# For classical optimization
pip install scipy numpy matplotlib

# For MCMC analysis
pip install pymc arviz pytensor

# For performance acceleration  
pip install numba
```

**Configuration Errors**:
- Ensure JSON configuration is valid (check with `python -m json.tool config.json`)
- Verify file paths exist in configuration
- Check parameter bounds and initial values are reasonable

**Memory Issues**:
- Reduce array sizes in configuration
- Enable angle filtering for large datasets
- Adjust `memory_limit_gb` setting
- Use `float32` instead of `float64` for data type

**Convergence Problems**:
- Adjust initial parameter values in configuration
- Increase maximum iterations for classical optimization
- For MCMC: check R-hat values and effective sample sizes in diagnostics

### Mode-Specific Issues

**"Angle filtering enabled but static_isotropic mode detected"**:
This is expected behavior - angle filtering is automatically disabled in isotropic mode regardless of configuration.

**"phi_angles_file not found" in static isotropic mode**:
This is expected - phi_angles_file is not loaded in isotropic mode. A dummy angle is used automatically.

**Slow optimization in laminar flow mode**:
Enable angle filtering to reduce computational cost by 3-5x with minimal accuracy loss.

**MCMC convergence problems with 7 parameters**:
- Increase tuning steps (`tune: 2000+`)
- Use better initial parameter estimates from classical optimization
- Increase target acceptance rate (`target_accept: 0.95`)

### Performance Tips

- **First Run**: Allow extra time for Numba JIT compilation warmup
- **Large Datasets**: Use isotropic mode when applicable (fastest), enable angle filtering in anisotropic/flow modes
- **Memory Constraints**: Use NPZ format instead of HDF5 for caching
- **Parallel Processing**: Set appropriate `num_threads` in configuration

## Migration Guide

### From Legacy Static Mode
If you have existing configurations with just `"static_mode": true`:

**Before** (legacy):
```json
{
  "analysis_settings": {
    "static_mode": true
  }
}
```

**After** (explicit):
```json
{
  "analysis_settings": {
    "static_mode": true,
    "static_submode": "anisotropic"
  }
}
```

**Backward Compatibility**: Legacy configurations automatically default to `"anisotropic"` mode.

### Configuration Updates
**Remove scaling optimization setting** (now always enabled):
```json
{
  "chi_squared_calculation": {
    "scaling_optimization": true  // Remove this line
  }
}
```

### Command Updates
**Updated CLI flags**:
- `--static` → `--static-anisotropic` (deprecated but still works)
- New: `--static-isotropic` for fastest analysis
- New: `--laminar-flow` for explicit flow mode

## Citation

If you use this package in your research, please cite:

```bibtex
@article{he2024transport,
  title={Transport coefficient approach for characterizing nonequilibrium dynamics in soft matter},
  author={He, Hongrui and Liang, Hao and Chu, Miaoqi and Jiang, Zhang and de Pablo, Juan J and Tirrell, Matthew V and Narayanan, Suresh and Chen, Wei},
  journal={Proceedings of the National Academy of Sciences},
  volume={121},
  number={31},
  pages={e2401162121},
  year={2024},
  publisher={National Academy of Sciences},
  doi={10.1073/pnas.2401162121}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please feel free to submit issues, feature requests, and pull requests.

### Development Setup

1. Clone the repository
2. Install in development mode: `pip install -e .`
3. Install development dependencies: `pip install pytest pytest-cov numba pymc arviz`
4. Run tests: `python homodyne/run_tests.py`

### Authors

- **Wei Chen** - *Argonne National Laboratory* - wchen@anl.gov
- **Hongrui He** - *Argonne National Laboratory*

### Acknowledgments

This work was supported by the U.S. Department of Energy, Office of Science, Basic Energy Sciences under contract DE-AC02-06CH11357. Use of the Advanced Photon Source, an Office of Science User Facility operated for the U.S. Department of Energy (DOE) Office of Science by Argonne National Laboratory.

## Documentation

📚 **Complete Documentation**: https://imewei.github.io/homodyne/

The documentation includes:
- **User Guide**: Installation, quickstart, configuration, and examples
- **API Reference**: Complete API documentation with auto-generated reference
- **Developer Guide**: Architecture, contributing guidelines, and troubleshooting
