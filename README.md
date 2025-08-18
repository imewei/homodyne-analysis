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

### PyPI Installation (Recommended)

Install the complete package with all features from PyPI:

```bash
# Full installation with all features (MCMC, performance, data handling)
pip install homodyne-analysis[all]
```

### Development Installation

For development or latest features, install directly from the repository:

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

# Install the package in development mode with all features
pip install -e .[all]
```

### Dependencies

**Core Requirements**: `numpy`, `scipy`, `matplotlib`  
**Data Handling**: `xpcs-viewer` (specialized XPCS data loading and manipulation)  
**Performance Enhancement**: `numba` (provides 3-5x speedup via JIT compilation)  
**Bayesian Analysis**: `pymc`, `arviz`, `pytensor` (for MCMC sampling and diagnostics)  
**Optional Test Dependencies**: `pytest`, `pytest-cov`, `pytest-xdist`, `pytest-benchmark`, `pytest-mock`, `hypothesis`, `black`, `flake8`, `mypy`  
**Optional Documentation Dependencies**: `sphinx`, `sphinx-rtd-theme`, `myst-parser`, `sphinx-autodoc-typehints`, `numpydoc`

## Quick Start

Installation:

```bash
# Full installation with all features (MCMC, performance, data handling)
pip install homodyne-analysis[all]
```

After installation, use the command line tools:

```bash
# Create a configuration file
homodyne-config --mode laminar_flow --sample my_sample --output my_config.json

# Run analysis
homodyne --config my_config.json --method all
```

Or use the Python API:

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

### Console Scripts (PyPI Installation)

After installing via PyPI, use the convenient console scripts:

```bash
# Main analysis command
homodyne --help

# Configuration creator
homodyne-config --help
```

### Command Line Interface

#### Main Analysis Runner

```bash
# Basic classical optimization with mode specification
homodyne --static-isotropic --method classical
homodyne --static-anisotropic --method mcmc
homodyne --laminar-flow --method all

# Use custom configuration and output directory
homodyne --config my_experiment.json --output-dir ./results

# Generate experimental data validation plots only (no fitting)
homodyne --plot-experimental-data --verbose

# Classical method with C2 heatmaps (saves to ./homodyne_results/classical/)
homodyne --method classical --plot-c2-heatmaps

# MCMC method with comprehensive uncertainty analysis (saves to ./homodyne_results/mcmc/)
homodyne --method mcmc --config my_experiment.json

# MCMC method with C2 heatmaps using posterior means (saves to ./homodyne_results/mcmc/)
homodyne --method mcmc --plot-c2-heatmaps

# Note: --plot-experimental-data now skips all fitting and saves plots to ./homodyne_results/exp_data/
```

#### Data Validation

Generate comprehensive validation plots of experimental C2 correlation data without performing any fitting:

```bash
# Basic data validation (plots only, no fitting)
homodyne --plot-experimental-data --config my_config.json

# Verbose validation with debug logging
homodyne --plot-experimental-data --config my_config.json --verbose
```

**Output**: Creates validation plots in `./homodyne_results/exp_data/` including:
- Full 2D correlation function heatmaps g₂(t₁,t₂) for each angle
- Diagonal slices g₂(t,t) showing temporal decay
- Cross-sectional profiles at different time points
- Statistical summaries with data quality metrics

**Note**: The `--plot-experimental-data` flag now:
- **Skips all fitting procedures** (classical and MCMC)
- **Saves plots to `./homodyne_results/exp_data/`** instead of `./plots/data_validation/`
- **Exits immediately** after generating experimental data plots

### Configuration Management

#### Template System
The package provides mode-specific configuration templates optimized for different analysis scenarios:

- **`config_static_isotropic.json`**: Optimized for isotropic analysis with single dummy angle
- **`config_static_anisotropic.json`**: Static analysis with angle filtering enabled
- **`config_laminar_flow.json`**: Full flow analysis with all 7 parameters
- **`config_template.json`**: Master template with comprehensive documentation

#### Configuration Creation
Generate analysis configurations using the `homodyne-config` command:

```bash
# Create isotropic static configuration (fastest)
homodyne-config --mode static_isotropic --sample protein_01

# Create anisotropic static configuration with metadata
homodyne-config --mode static_anisotropic --sample collagen \
                        --author "Your Name" --experiment "Static analysis"

# Create flow analysis configuration
homodyne-config --mode laminar_flow --sample microgel \
                        --experiment "Microgel dynamics under shear"

# Create with custom output file
homodyne-config --mode static_isotropic --output my_isotropic_config.json
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
homodyne --plot-experimental-data --verbose
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
- **Output**: Full posterior distributions, uncertainty quantification, convergence diagnostics, 3D surface plots
- **Best For**: Robust parameter estimation, uncertainty analysis, publication-quality results
- **Command**: `--method mcmc`
- **Additional Requirements**: `pip install pymc arviz pytensor`

### Combined Analysis
- **Recommended Workflow**: Use classical optimization for initial parameter estimates, then refine with MCMC for full uncertainty analysis
- **Command**: `--method all` (runs both methods sequentially)

**Note**: Scaling optimization (g₂ = offset + contrast × g₁) is always enabled in all methods for consistent and scientifically accurate chi-squared calculations.

## Analysis Workflows

### Recommended Analysis Pipeline

#### 1. **Data Validation Workflow**
```bash
# Step 1: Validate experimental data quality
homodyne --plot-experimental-data --config my_config.json
# Output: ./homodyne_results/exp_data/ with validation plots and statistics
```

#### 2. **Exploratory Analysis Workflow** 
```bash
# Step 2: Fast parameter estimation with classical optimization
homodyne --method classical --config my_config.json
# Output: ./homodyne_results/classical/ with point estimates and C2 heatmaps

# Optional: Generate C2 heatmaps for visual validation
homodyne --method classical --plot-c2-heatmaps --config my_config.json
```

#### 3. **Comprehensive Analysis Workflow**
```bash
# Step 3: Full uncertainty quantification with MCMC
homodyne --method mcmc --config my_config.json
# Output: ./homodyne_results/mcmc/ with posterior distributions, trace data, and diagnostics

# Optional: Generate C2 heatmaps using posterior means
homodyne --method mcmc --plot-c2-heatmaps --config my_config.json
```

#### 4. **Complete Pipeline Workflow**
```bash
# Step 4: Run all methods in sequence (recommended for publication)
homodyne --method all --config my_config.json
# Output: Both ./homodyne_results/classical/ and ./homodyne_results/mcmc/ directories
```

### Method-Specific Outputs

#### Classical Method Results (`./homodyne_results/classical/`)
- **Speed**: Fast execution (~minutes)
- **Best for**: Parameter screening, model validation, computational efficiency
- **Output files**:
  - `experimental_data.npz`: Original correlation data
  - `fitted_data.npz`: Optimally scaled theoretical predictions  
  - `residuals_data.npz`: Fit residuals for quality assessment
  - `c2_heatmaps_phi_*.png`: Visual comparison plots (if requested)

#### MCMC Method Results (`./homodyne_results/mcmc/`)
- **Speed**: Comprehensive but slower (~hours)
- **Best for**: Uncertainty quantification, publication-quality analysis, parameter correlations
- **Output files**:
  - `experimental_data.npz`: Original correlation data
  - `fitted_data.npz`: Scaled predictions using posterior means
  - `residuals_data.npz`: Residuals from posterior mean fit
  - `mcmc_summary.json`: Convergence diagnostics and posterior statistics
  - `mcmc_trace.nc`: Full trace data in NetCDF format (ArviZ compatible)
  - `c2_heatmaps_phi_*.png`: Heatmaps using posterior means (if requested)
  - `3d_surface_phi_*.png`: 3D surface plots with 95% confidence intervals
  - `3d_surface_residuals_phi_*.png`: 3D residuals plots for quality assessment
  - `trace_plot.png`: MCMC chain diagnostics
  - `corner_plot.png`: Parameter posterior distributions

### Data Analysis Best Practices

#### Quality Assessment
```bash
# Check experimental data quality first
homodyne --plot-experimental-data --config my_config.json

# Review output in ./homodyne_results/exp_data/summary_statistics.txt
# Ensure reasonable g2 values (typically 1.0 + small contrast)
```

#### Parameter Initialization
```bash
# Use classical results to initialize MCMC (automatic when using --method all)
homodyne --method all --config my_config.json

# Or run sequentially for more control:
homodyne --method classical --config my_config.json
homodyne --method mcmc --config my_config.json  # Uses classical results automatically
```

#### Results Interpretation
- **Classical χ² values**: Lower values indicate better fit quality (χ²_red < 2.0 excellent, < 5.0 acceptable)
- **MCMC convergence**: Check R̂ < 1.1 and ESS > 100 for reliable results
- **Visual validation**: Use C2 heatmaps to assess systematic deviations between experimental and fitted data
- **3D visualization**: MCMC method automatically generates 3D surface plots with confidence intervals for publication-quality figures
- **Residuals analysis**: Check `residuals_data.npz` for systematic patterns indicating model inadequacies

## Output Directory Structure

The analysis results are now organized into method-specific subdirectories for better organization:

### Default Output Structure
```
./homodyne_results/
├── homodyne_analysis_results.json    # Main results file (moved from root)
├── per_angle_chi_squared_classical.json
├── run.log                           # Analysis log file
├── exp_data/                         # Experimental data plots (--plot-experimental-data)
│   ├── data_validation_phi_*.png
│   └── summary_statistics.txt
├── classical/                       # Classical method outputs (--method classical)
│   ├── experimental_data.npz         # Original experimental correlation data
│   ├── fitted_data.npz              # Fitted data (contrast * theory + offset)
│   ├── residuals_data.npz           # Residuals (experimental - fitted)
│   └── c2_heatmaps_phi_*.png        # C2 correlation heatmaps (--plot-c2-heatmaps)
└── mcmc/                            # MCMC method outputs (--method mcmc)
    ├── experimental_data.npz         # Original experimental correlation data
    ├── fitted_data.npz              # Fitted data (contrast * posterior_means + offset)
    ├── residuals_data.npz           # Residuals (experimental - fitted)
    ├── mcmc_summary.json            # MCMC convergence diagnostics and posterior statistics
    ├── mcmc_trace.nc                # NetCDF trace data (ArviZ format)
    ├── c2_heatmaps_phi_*.png        # C2 correlation heatmaps using posterior means
    ├── 3d_surface_phi_*.png         # 3D surface plots with 95% confidence intervals
    ├── 3d_surface_residuals_phi_*.png # 3D residuals plots for quality assessment
    ├── trace_plot.png               # MCMC trace plots
    └── corner_plot.png              # Parameter posterior distributions
```

### Key Changes
- **Main results file**: `homodyne_analysis_results.json` now saved in output directory instead of current directory
- **Classical method**: Results organized in `./homodyne_results/classical/` subdirectory
- **MCMC method**: Results organized in `./homodyne_results/mcmc/` subdirectory
- **Experimental data plots**: Saved to `./homodyne_results/exp_data/` when using `--plot-experimental-data`
- **Data files**: Both classical and MCMC methods save experimental, fitted, and residuals data as `.npz` files
- **Method-specific files**:
  - **Classical**: C2 heatmaps only (diagnostic plots skipped)
  - **MCMC**: C2 heatmaps, 3D surface plots with confidence intervals, trace data (NetCDF), convergence diagnostics, trace plots, corner plots
- **Fitted data calculation**: Both methods use least squares scaling optimization (`fitted = contrast * theory + offset`)
- **Directory separation**: Each method maintains its own isolated output directory for clear organization

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

📚 **Complete Documentation**: https://homodyne.readthedocs.io/

The documentation includes:
- **User Guide**: Installation, quickstart, configuration, and examples
- **API Reference**: Complete API documentation with auto-generated reference
- **Developer Guide**: Architecture, contributing guidelines, and troubleshooting
