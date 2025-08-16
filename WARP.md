# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Repository Overview

This repository implements a comprehensive Python package for analyzing homodyne scattering in X-ray Photon Correlation Spectroscopy (XPCS) under nonequilibrium conditions. The package implements the theoretical framework described in He et al. PNAS 2024 for characterizing nonequilibrium dynamics in soft matter systems through detailed transport coefficient analysis.

**Physical Context**: The package analyzes time-dependent intensity correlation functions g₂(φ,t₁,t₂) for complex fluids under nonequilibrium laminar flow conditions. It captures the interplay between Brownian diffusion and advective shear flow, enabling quantitative characterization of transport properties in flowing soft matter systems.

**Core Capabilities**:
- **Dual Analysis Modes**: Static analysis (3 parameters) for quiescent systems and Laminar Flow analysis (7 parameters) for systems under shear
- **Multiple Optimization Approaches**: Fast classical optimization (Nelder-Mead) for point estimates and robust Bayesian MCMC (NUTS) for full posterior distributions with uncertainty quantification
- **Performance Optimizations**: Numba JIT compilation for computational kernels, smart angle filtering, and memory-efficient data handling
- **Comprehensive Configuration System**: JSON-based configuration with templates, validation, and runtime parameter override capabilities
- **Integrated Visualization**: Experimental data validation plots, parameter evolution tracking, MCMC convergence diagnostics, and corner plots for uncertainty visualization
- **Quality Assurance**: Extensive test coverage with pytest framework and performance benchmarking tools

## Package Architecture

### Core Components

- **`homodyne/core/`**: Central infrastructure including configuration management (`ConfigManager`), optimized computational kernels, and flexible I/O utilities with JSON serialization support
- **`homodyne/analysis/`**: Main analysis engine (`HomodyneAnalysisCore`) handling experimental data loading, correlation function calculations, and chi-squared fitting
- **`homodyne/optimization/`**: Dual optimization framework with classical methods (`ClassicalOptimizer`) and Bayesian MCMC sampling (`MCMCSampler`)
- **`homodyne/plotting.py`**: Comprehensive visualization system for data validation, parameter analysis, and diagnostic plotting

### Key Classes and Functions

- **`ConfigManager`**: Robust JSON configuration handling with template-based creation, validation, and runtime parameter override capabilities
- **`HomodyneAnalysisCore`**: Primary analysis engine managing experimental data loading, preprocessing, and chi-squared objective function calculations
- **`ClassicalOptimizer`**: Scipy-based optimization with intelligent angle filtering and performance monitoring
- **`MCMCSampler`**: PyMC-based Bayesian parameter estimation using NUTS sampling with convergence diagnostics
- **Analysis Plotting Functions**: Data validation plots, parameter evolution visualization, MCMC trace and corner plots for uncertainty analysis

### Dependencies

**Core Requirements**: `numpy`, `scipy`, `matplotlib`
**Performance Enhancement**: `numba` (provides 3-5x speedup via JIT compilation)
**Bayesian Analysis**: `pymc`, `arviz`, `pytensor` (for MCMC sampling and diagnostics)
**Optional**: `pyxpcsviewer` (specialized XPCS data loading)

## Installation and Setup

### Quick Start Installation

```bash
# Set up Python environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install core dependencies
pip install numpy scipy matplotlib

# For enhanced performance (recommended)
pip install numba

# For Bayesian MCMC analysis
pip install pymc arviz pytensor
```

### Configuration Creation

Generate analysis configurations from templates:

```bash
# Create a new configuration from template
python homodyne/create_config.py --output my_experiment.json --sample protein_01 --author "Your Name"

# Create with experiment description
python homodyne/create_config.py --experiment "Protein dynamics under shear" --output my_config.json
```

### Testing and Validation

```bash
# Run test suite
python homodyne/run_tests.py

# Quick test (exclude slow integration tests)
python homodyne/run_tests.py --fast

# Test specific functionality
python homodyne/run_tests.py -k "config" -v
```

## Command Reference

### Main Analysis Runner

```bash
# Basic classical optimization
python run_homodyne.py

# Run all methods with verbose logging
python run_homodyne.py --method all --verbose

# Use custom configuration
python run_homodyne.py --config my_experiment.json --output-dir ./results

# Force static mode analysis (3 parameters)
python run_homodyne.py --static --method classical

# Force laminar flow mode with MCMC
python run_homodyne.py --laminar-flow --method mcmc

# Run all methods in static mode
python run_homodyne.py --static --method all --verbose

# Custom output directory
python run_homodyne.py --output-dir ./my_results --verbose

# Generate experimental data validation plots for quality checking
python run_homodyne.py --plot-experimental-data --verbose

# Combine with other flags for comprehensive analysis
python run_homodyne.py --plot-experimental-data --method all --verbose
```

### Configuration Management

```bash
# View configuration template structure
cat homodyne/config_template.json

# Create minimal config with defaults
python homodyne/create_config.py

# Create config with custom sample name
python homodyne/create_config.py --sample my_sample --output my_sample_config.json
```

### Test Execution

```bash
# Quick test run (exclude slow tests)
python homodyne/run_tests.py --fast

# Verbose test output
python homodyne/run_tests.py --verbose

# Test specific functionality
python homodyne/run_tests.py -k "mcmc"
python homodyne/run_tests.py -k "config"
python homodyne/run_tests.py -m "integration"
```

## Analysis Modes

### Static Mode (3 parameters)
- **Physical Context**: Analysis of systems at equilibrium with zero shear rate
- **Parameters**: 
  - D₀: Effective diffusion coefficient
  - α: Time exponent characterizing dynamic scaling
  - D_offset: Baseline diffusion component
- **When to Use**: Quiescent systems, equilibrium measurements, validation runs
- **Command**: `--static`

### Laminar Flow Mode (7 parameters) 
- **Physical Context**: Analysis of systems under controlled shear flow conditions
- **Parameters**: 
  - D₀, α, D_offset: Same as static mode
  - γ̇₀: Characteristic shear rate
  - β: Shear rate exponent for flow scaling
  - γ̇_offset: Baseline shear component
  - φ₀: Angular offset parameter for flow geometry
- **When to Use**: Systems under shear, nonequilibrium conditions, transport coefficient analysis
- **Command**: `--laminar-flow`

## Data Validation and Quality Control

### Experimental Data Validation Plots

To ensure data quality before analysis, you can generate comprehensive validation plots of the experimental C2 correlation data:

```bash
# Enable experimental data plotting with the --plot-experimental-data flag
python run_homodyne.py --plot-experimental-data --verbose
```

**What the validation plots show:**
- **Heatmaps**: Full 2D correlation function g₂(t₁,t₂) for each angle
- **Diagonal slices**: g₂(t,t) values along the diagonal
- **Cross-sections**: Correlation function profiles at different time points
- **Statistics**: Data quality metrics including mean values, contrast, and validation checks

**Quality indicators to look for:**
- Mean values around 1.0 (expected for g₂ correlation functions)
- Enhanced diagonal values (should be higher than off-diagonal)
- Sufficient contrast (> 0.001) indicating dynamic signal
- Consistent structure across different angles

**Plots are saved to:** `<output_dir>/plots/experimental_data_validation/`

### Configuration Settings

You can also enable experimental data plotting in your configuration file:

```json
{
  "workflow_integration": {
    "analysis_workflow": {
      "plot_experimental_data_on_load": true
    }
  }
}
```

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

## Troubleshooting

### Common Issues

**Missing Dependencies**:
```bash
# For classical optimization
pip install scipy numpy

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

### Performance Tips

- **First Run**: Allow extra time for Numba JIT compilation warmup
- **Large Datasets**: Enable angle filtering in optimization config
- **Memory Constraints**: Use NPZ format instead of HDF5 for caching
- **Parallel Processing**: Set appropriate `num_threads` in configuration

### Log Files

Analysis logs are saved to `<output_dir>/run.log` with timestamps and performance metrics. Check logs for:
- Configuration validation errors
- Data loading issues  
- Optimization convergence details
- Performance timing information

## Testing and Quality Assurance

The package includes a comprehensive test suite using pytest:

- **Test Categories**: Core functionality, I/O operations, plotting, integration workflows
- **Quality Metrics**: Extensive coverage of critical code paths
- **Performance Tests**: Benchmarking and regression detection
- **Data Validation**: Ensuring numerical accuracy and consistency

## File Structure

```
homodyne/
├── run_homodyne.py              # Main CLI entry point
├── benchmark_performance.py    # Performance benchmarking
├── homodyne/
│   ├── __init__.py             # Package exports and version
│   ├── create_config.py        # Configuration file generator
│   ├── run_tests.py           # Test runner with options
│   ├── plotting.py            # Visualization utilities
│   ├── core/                  # Core functionality
│   │   ├── config.py         # Configuration management
│   │   ├── kernels.py        # Computational kernels
│   │   └── io_utils.py       # Data I/O utilities
│   ├── analysis/             # Analysis engines
│   │   └── core.py          # Main analysis class
│   ├── optimization/         # Optimization methods
│   │   ├── classical.py     # Scipy-based optimization
│   │   └── mcmc.py          # PyMC Bayesian sampling
│   └── tests/               # Test suite
│       ├── conftest.py      # Pytest configuration
│       ├── fixtures.py      # Test fixtures
│       └── test_*.py        # Individual test files
└── my_config_simon.json        # Example configuration file
```
