# Homodyne Scattering Analysis Package

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![Numba](https://img.shields.io/badge/Numba-JIT%20Accelerated-green)](https://numba.pydata.org/)
[![Performance](https://img.shields.io/badge/Performance-Optimized%20%26%20Monitored-brightgreen)](PERFORMANCE_OPTIMIZATIONS.md)

A high-performance Python package for analyzing homodyne scattering in X-ray Photon Correlation Spectroscopy (XPCS) under nonequilibrium conditions. Implements the theoretical framework from [He et al. PNAS 2024](https://doi.org/10.1073/pnas.2401162121) for characterizing transport properties in flowing soft matter systems.

## Overview

Analyzes time-dependent intensity correlation functions $c_2(\phi,t_1,t_2)$ in complex fluids under nonequilibrium conditions, capturing the interplay between Brownian diffusion and advective shear flow.

**Key Features:**
- **Three analysis modes**: Static Isotropic (3 params), Static Anisotropic (3 params), Laminar Flow (7 params)
- **Multiple optimization methods**: Classical (Nelder-Mead, Gurobi), Bayesian MCMC (NUTS)
- **High performance**: Numba JIT compilation with 3-5x speedup, vectorized operations, and optimized memory usage
- **Performance monitoring**: Comprehensive regression testing and automated benchmarking
- **Scientific accuracy**: Automatic $g_2 = \text{offset} + \text{contrast} \times g_1$ fitting for proper $\chi^2$ calculations


## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Analysis Modes](#analysis-modes)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Performance & Testing](#performance--testing)
- [Theoretical Background](#theoretical-background)
- [Citation](#citation)
- [Documentation](#documentation)

## Installation

### PyPI Installation (Recommended)

```bash
pip install homodyne-analysis[all]
```

### Development Installation

```bash
git clone https://github.com/imewei/homodyne.git
cd homodyne
pip install -e .[all]
```

### Dependencies

- **Core**: `numpy`, `scipy`, `matplotlib`
- **Performance**: `numba` (3-5x speedup via JIT compilation)
- **Bayesian Analysis**: `pymc`, `arviz`, `pytensor` (for MCMC sampling)
- **Optimization**: `gurobipy` (optional, requires license for Gurobi solver)
- **Optional**: `pytest`, `sphinx` (testing and documentation)

## Quick Start

```bash
# Install
pip install homodyne-analysis[all]

# Create configuration
homodyne-config --mode laminar_flow --sample my_sample

# Run analysis
homodyne --config my_config.json --method all
```

**Python API:**

```python
from homodyne import HomodyneAnalysisCore, ConfigManager

config = ConfigManager("config.json")
analysis = HomodyneAnalysisCore(config)
results = analysis.optimize_classical()  # Fast
results = analysis.optimize_all()        # Classical + MCMC
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
  - $D_0$: Effective diffusion coefficient
  - $\alpha$: Time exponent characterizing dynamic scaling
  - $D_{\text{offset}}$: Baseline diffusion component
- **Key Features**:
  - No angle filtering (automatically disabled)
  - No phi_angles_file loading (uses single dummy angle)
  - Fastest analysis mode
- **When to Use**: Isotropic samples, quick validation runs, preliminary analysis
- **Model**: $g_1(t_1,t_2) = \exp(-q^2 \int_{t_1}^{t_2} D(t)dt)$ with no angular dependence

### Static Anisotropic Mode (3 parameters)
- **Physical Context**: Analysis of systems at equilibrium with angular dependence but no flow effects
- **Parameters**: $D_0$, $\alpha$, $D_{\text{offset}}$ (same as isotropic mode)
- **Key Features**:
  - Angle filtering enabled for optimization efficiency
  - phi_angles_file loaded for angle information
  - Per-angle scaling optimization
- **When to Use**: Static samples with measurable angular variations, moderate computational resources
- **Model**: Same as isotropic mode but with angle filtering to focus optimization on specific angular ranges

### Laminar Flow Mode (7 parameters) 
- **Physical Context**: Analysis of systems under controlled shear flow conditions with full physics model
- **Parameters**: 
  - $D_0$, $\alpha$, $D_{\text{offset}}$: Same as static modes
  - $\dot{\gamma}_0$: Characteristic shear rate
  - $\beta$: Shear rate exponent for flow scaling
  - $\dot{\gamma}_{\text{offset}}$: Baseline shear component
  - $\phi_0$: Angular offset parameter for flow geometry
- **Key Features**:
  - All flow and diffusion effects included
  - phi_angles_file required for angle-dependent flow effects
  - Complex parameter space with potential correlations
- **When to Use**: Systems under shear, nonequilibrium conditions, transport coefficient analysis
- **Model**: $g_1(t_1,t_2) = g_{1,\text{diff}}(t_1,t_2) \times g_{1,\text{shear}}(t_1,t_2)$ where shear effects are $\text{sinc}^2(\Phi)$

## Usage Examples

### Command Line Interface

```bash
# Basic analysis
homodyne --static-isotropic --method classical
homodyne --static-anisotropic --method mcmc
homodyne --laminar-flow --method all

# Data validation only
homodyne --plot-experimental-data --config my_config.json

# Custom configuration and output
homodyne --config my_experiment.json --output-dir ./results

# Logging control options
homodyne --verbose                              # Debug logging to console and file
homodyne --quiet                               # File logging only, no console output
homodyne --config my_config.json --quiet       # Quiet mode with custom config

# Generate C2 heatmaps
homodyne --method classical --plot-c2-heatmaps
```

### Data Validation

Generate validation plots without fitting:

```bash
homodyne --plot-experimental-data --config my_config.json --verbose
homodyne --plot-experimental-data --config my_config.json --quiet  # Quiet mode
```

**Output**: Creates plots in `./homodyne_results/exp_data/`:
- 2D correlation function heatmaps $g_2(t_1,t_2)$
- Diagonal slices $g_2(t,t)$ showing decay
- Statistical summaries and quality metrics

## Configuration

### Creating Configurations

```bash
# Generate configuration templates
homodyne-config --mode static_isotropic --sample protein_01
homodyne-config --mode laminar_flow --sample microgel
```

### Mode Selection

Configuration files specify analysis mode:

```json
{
  "analysis_settings": {
    "static_mode": true/false,
    "static_submode": "isotropic" | "anisotropic" | null
  }
}
```

**Rules**:
- `static_mode: false` → Laminar Flow Mode (7 params)
- `static_mode: true, static_submode: "isotropic"` → Static Isotropic (3 params)
- `static_mode: true, static_submode: "anisotropic"` → Static Anisotropic (3 params)

### Quality Control

Check data quality before fitting:

```bash
homodyne --plot-experimental-data --verbose
```

**Look for**:
- Mean values around 1.0 ($g_2$ correlation functions)
- Enhanced diagonal values
- Sufficient contrast (> 0.001)

### Logging Control

The package provides flexible logging control for different use cases:

| Option | Console Output | File Output | Use Case |
|--------|---------------|-------------|----------|
| **Default** | INFO level | INFO level | Normal interactive analysis |
| **`--verbose`** | DEBUG level | DEBUG level | Detailed troubleshooting and debugging |
| **`--quiet`** | None | INFO level | Batch processing, scripting, clean output |

```bash
# Detailed debugging information
homodyne --verbose --method all

# Quiet execution (logs only to file)
homodyne --quiet --method classical --output-dir ./batch_results

# Cannot combine conflicting options
homodyne --verbose --quiet  # ERROR: conflicting options
```

**File Logging**: All modes save detailed logs to `output_dir/run.log` for analysis tracking and debugging, regardless of console settings.

## Performance and Stability

The homodyne package includes enterprise-grade performance optimization and monitoring features:

### Performance Stability Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **JIT Warmup** | Automatic Numba kernel pre-compilation | Eliminates JIT compilation overhead |
| **Adaptive Benchmarking** | Smart benchmarking with target stability | 95%+ improvement in performance consistency |
| **Memory Management** | Automatic memory monitoring and cleanup | Prevents memory bloat in long-running analyses |
| **Smart Caching** | Memory-aware LRU caching system | Optimizes memory usage while improving speed |
| **Environment Optimization** | Conservative threading and JIT settings | Balanced performance and numerical stability |
| **Performance Rebalancing** | Optimized chi-squared and kernel functions | 97% reduction in performance variability |

### Performance Monitoring

The package includes comprehensive performance monitoring tools:

```python
from homodyne.core.profiler import performance_monitor, get_performance_summary, get_performance_cache

# Monitor function performance
@performance_monitor(monitor_memory=True, log_threshold_seconds=0.5)
def my_analysis_function(data):
    return process_data(data)

# Get performance statistics
summary = get_performance_summary()
print(f"Function called {summary['my_analysis_function']['calls']} times")
print(f"Average time: {summary['my_analysis_function']['avg_time']:.3f}s")

# Access smart caching system
cache = get_performance_cache()
cache_stats = cache.stats()
print(f"Cache utilization: {cache_stats['utilization']:.1%}")
print(f"Memory usage: {cache_stats['memory_mb']:.1f}MB")
```

### JIT Compilation Warmup

Eliminate JIT compilation overhead with automatic kernel pre-compilation:

```python
from homodyne.core.kernels import warmup_numba_kernels

# Warmup all computational kernels
warmup_results = warmup_numba_kernels()
print(f"Kernels warmed up in {warmup_results['total_warmup_time']:.3f}s")
print(f"Warmed kernels: {list(warmup_results['warmup_results'].keys())}")
```

### Benchmarking Utilities

For developers and researchers who need reliable performance measurements:

```python
from homodyne.core.profiler import stable_benchmark, adaptive_stable_benchmark

# Standard stable benchmarking
results = stable_benchmark(my_function, warmup_runs=5, measurement_runs=15)
print(f"Mean time: {results['mean']:.4f}s, CV: {results['std']/results['mean']:.3f}")

# Adaptive benchmarking (finds optimal measurement count)
results = adaptive_stable_benchmark(my_function, target_cv=0.10)
print(f"Achieved {results['cv']:.3f} CV in {results['total_runs']} runs")
```

### Performance Configuration

Key environment variables for optimization:

```bash
# Conservative threading for stability (automatically set)
export NUMBA_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Balanced JIT optimization (automatically configured)
export NUMBA_FASTMATH=0  # Disabled for numerical stability
export NUMBA_LOOP_VECTORIZE=1
export NUMBA_OPT=2  # Moderate optimization level

# Memory optimization
export NUMBA_CACHE_DIR=~/.numba_cache
```

### Performance Baselines

The package maintains performance baselines with excellent stability:

**Stability Achievements:**
- **Chi-squared calculations**: CV < 0.31 across all array sizes
- **97% reduction** in performance variability 
- **Balanced optimization**: Performance and numerical stability
- **Production ready**: Consistent benchmarking results

Performance baselines and regression detection:

- **Chi-squared calculation**: ~0.8-1.2ms (CV ≤ 0.09)
- **Correlation calculation**: ~0.26-0.28ms (CV ≤ 0.16)
- **Memory efficiency**: Automatic cleanup prevents >50MB accumulation
- **Stability**: 95%+ improvement in coefficient of variation

## Performance & Testing

### Optimization Methods

**Classical Optimization (Fast)**
- **Nelder-Mead**: Derivative-free simplex algorithm, robust for noisy functions
- **Gurobi**: Quadratic programming solver (requires license), excellent for smooth functions with bounds constraints
- Speed: ~minutes (optimized with lazy imports and memory-efficient operations)
- Use: Exploratory analysis, parameter screening
- Command: `--method classical`

**Bayesian MCMC (Comprehensive)**
- Algorithm: NUTS sampler via PyMC (lazy-loaded for fast startup)
- Speed: ~hours (with Numba JIT acceleration and optional thinning)
- Features: Uncertainty quantification, thinning support, convergence diagnostics
- Use: Uncertainty quantification, publication results
- Command: `--method mcmc`

**Combined**
- Workflow: Classical → MCMC refinement
- Command: `--method all`

**Note**: Gurobi is automatically detected if installed and licensed. Both classical methods are attempted if available, with the best result selected based on chi-squared value. All optimization methods (Nelder-Mead, Gurobi, MCMC) use the same parameter bounds defined in the configuration for consistency.

### Performance Optimizations

The package includes comprehensive performance optimizations:

**🚀 Computational Optimizations:**
- **Numba JIT compilation**: 3-5x speedup for core kernels with comprehensive warmup
- **Vectorized operations**: NumPy-optimized angle filtering and array operations
- **Memory-efficient processing**: Lazy allocation and memory-mapped file loading
- **Enhanced caching**: Fast cache key generation for NumPy arrays
- **Stable benchmarking**: Outlier filtering and variance reduction for reliable performance testing

**⚡ Import Optimizations:**
- **Lazy loading**: Heavy dependencies loaded only when needed
- **Fast startup**: >99% reduction in import time for optional components
- **Modular imports**: Core functionality available without heavy dependencies

**🎯 MCMC Optimizations:**
- **Thinning support**: Configurable sample thinning to reduce autocorrelation and memory usage
- **Smart defaults**: Mode-aware thinning settings (thin=1 for laminar flow, thin=2 for static modes)
- **Convergence diagnostics**: R-hat, ESS, and mixing assessment with thinning recommendations

**📊 Memory Optimizations:**
- **Memory-mapped I/O**: Efficient loading of large experimental datasets
- **Lazy array allocation**: Reduced peak memory usage
- **Garbage collection optimization**: Automatic cleanup of temporary objects

**🔧 Recent Performance Enhancements (v2024.2):**
- **Enhanced JIT warmup**: Comprehensive function-level warmup reduces timing variance by 60%
- **Stable benchmarking**: Statistical outlier filtering for reliable performance measurement
- **Consolidated performance utilities**: Unified performance testing infrastructure
- **Improved type safety**: Enhanced type annotations and consistency checks

## Physical Constraints and Parameter Ranges

### Parameter Distributions and Constraints

The homodyne package implements comprehensive physical constraints to ensure scientifically meaningful results:

#### **Core Model Parameters**

| Parameter | Range | Distribution | Physical Constraint |
|-----------|-------|--------------|-------------------|
| `D0` | [1.0, 1000000.0] Å²/s | TruncatedNormal(μ=10000.0, σ=1000.0) | positive |
| `alpha` | [-2.0, 2.0] dimensionless | Normal(μ=-1.5, σ=0.1) | none |
| `D_offset` | [-100, 100] Å²/s | Normal(μ=0.0, σ=10.0) | none |
| `gamma_dot_t0` | [1e-06, 1.0] s⁻¹ | TruncatedNormal(μ=0.001, σ=0.01) | positive |
| `beta` | [-2.0, 2.0] dimensionless | Normal(μ=0.0, σ=0.1) | none |
| `gamma_dot_t_offset` | [-0.01, 0.01] s⁻¹ | Normal(μ=0.0, σ=0.001) | none |
| `phi0` | [-10, 10] degrees | Normal(μ=0.0, σ=5.0) | angular |

#### **Physical Function Constraints**

The package **automatically enforces positivity** for time-dependent functions:

- **D(t) = D₀(t)^α + D_offset** → **max(D(t), 1×10⁻¹⁰)**
  - Prevents negative diffusion coefficients
  - Maintains numerical stability with minimal threshold

- **γ̇(t) = γ̇₀(t)^β + γ̇_offset** → **max(γ̇(t), 1×10⁻¹⁰)**
  - Prevents negative shear rates
  - Ensures physical validity in all optimization scenarios

#### **Scaling Parameters for Correlation Functions**

The relationship **c2_fitted = c2_theory × contrast + offset** uses bounded parameters:

| Parameter | Range | Distribution | Physical Meaning |
|-----------|-------|--------------|------------------|
| `contrast` | (0.05, 0.5] | TruncatedNormal(μ=0.3, σ=0.1) | Correlation strength scaling |
| `offset` | (0.05, 1.95] | TruncatedNormal(μ=1.0, σ=0.2) | Baseline correlation level |
| `c2_fitted` | [1.0, 2.0] | *derived* | Final correlation function range |
| `c2_theory` | [0.0, 1.0] | *derived* | Theoretical correlation bounds |


### MCMC Configuration

**Optimized MCMC Settings:**
- **target_accept = 0.95**: High acceptance rate for constrained sampling
- **Distribution-aware priors**: TruncatedNormal for positive parameters, Normal otherwise  
- **Configuration-driven**: All parameters read from JSON files for consistency
#### Static Isotropic (3 parameters)
{
  "draws": 8000,
  "thin": 2,        # Effective samples: 4000
  "chains": 4
}

#### Static Anisotropic (3 parameters)  
{
  "draws": 8000,
  "thin": 2,        # Good convergence expected
  "chains": 4
}

#### Laminar Flow (7 parameters)
{
  "draws": 10000,
  "thin": 1,        # All samples needed for complex posterior
  "chains": 6
}

#### Memory-Constrained Systems
{
  "draws": 5000,
  "thin": 5,        # Effective samples: 1000
  "chains": 2
}
```

**Thinning Benefits:**
- ✅ Reduces autocorrelation between samples
- ✅ Lower memory usage (fewer stored samples)
- ✅ Faster post-processing and plotting
- ✅ Better effective sample size per stored sample
- ⚠️ Trades total samples for independence

### Performance Monitoring

**Automated Performance Testing:**
```bash
# Quick performance validation
python run_performance_tests.py --quick

# Full performance test suite
python run_performance_tests.py --full

# Memory usage tests
python run_performance_tests.py --memory

# Update performance baselines after optimizations
python run_performance_tests.py --update --quick
```

**Pytest Integration:**
```bash
# Performance tests with pytest
pytest -m performance                    # All performance tests
pytest -m "performance and not slow"     # Quick tests (CI-friendly)
pytest -m benchmark --benchmark-only     # Benchmarking tests
pytest -m memory                         # Memory usage tests
```

**Performance Benchmarking:**
```bash
# Comprehensive benchmark
python performance_benchmark_optimized.py --detailed

# Quick benchmark validation
python performance_benchmark_optimized.py
```

### Scaling Optimization

Always enabled for scientific accuracy:

$$g_2 = \text{offset} + \text{contrast} \times g_1$$

Accounts for instrumental effects, background, and normalization differences.

### Environment Optimization

```bash
# Threading optimization for reproducible performance
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMBA_DISABLE_INTEL_SVML=1

# Memory optimization
export NUMBA_CACHE_DIR=/tmp/numba_cache

# Performance monitoring mode
export HOMODYNE_PERFORMANCE_MODE=1
```

### Testing Framework

**Standard Testing:**
```bash
python homodyne/run_tests.py              # Standard tests
python homodyne/run_tests.py --fast       # Quick tests  
python homodyne/run_tests.py --coverage   # With coverage
pytest                                     # Pytest runner
```

**Performance Testing:**
```bash
# Performance validation
python run_performance_tests.py --quick   

# Performance test suite
pytest -m performance                     

# Regression detection
pytest -m regression                      

# Stable benchmarking with statistical analysis
pytest -m benchmark --benchmark-only
```

**CI/CD Integration:**
- **Automated testing**: Performance tests run on every PR
- **Regression detection**: Automatic alerts for performance degradation
- **Multi-platform**: Tests across Python 3.12, 3.13
- **Baseline tracking**: Performance history and trend monitoring

### Performance Documentation

📊 **Detailed Performance Guides:**
- [`docs/performance.rst`](docs/performance.rst) - Comprehensive performance optimization and monitoring guide

### Output Organization

```
./homodyne_results/
├── classical/                    # Classical method outputs
│   ├── experimental_data.npz
│   ├── fitted_data.npz
│   └── residuals_data.npz
├── mcmc/                         # MCMC method outputs  
│   ├── mcmc_summary.json
│   ├── mcmc_trace.nc
│   ├── trace_plot.png
│   └── corner_plot.png
└── exp_data/                     # Data validation plots
```

## Theoretical Background

The package implements three key equations describing correlation functions in nonequilibrium laminar flow systems:

**Equation 13 - Full Nonequilibrium Laminar Flow:**

$$c_2(\vec{q}, t_1, t_2) = 1 + \beta\left[e^{-q^2\int J(t)dt}\right] \times \text{sinc}^2\left[\frac{1}{2\pi} qh \int\dot{\gamma}(t)\cos(\phi(t))dt\right]$$

**Equation S-75 - Equilibrium Under Constant Shear:**

$$c_2(\vec{q}, t_1, t_2) = 1 + \beta\left[e^{-6q^2D(t_2-t_1)}\right] \text{sinc}^2\left[\frac{1}{2\pi} qh \cos(\phi)\dot{\gamma}(t_2-t_1)\right]$$

**Equation S-76 - One-time Correlation (Siegert Relation):**

$$g_2(\vec{q}, \tau) = 1 + \beta\left[e^{-6q^2D\tau}\right] \text{sinc}^2\left[\frac{1}{2\pi} qh \cos(\phi)\dot{\gamma}\tau\right]$$

**Key Parameters:**
- $\vec{q}$: scattering wavevector [Å⁻¹]  
- $h$: gap between stator and rotor [Å]
- $\phi(t)$: angle between shear/flow direction and $\vec{q}$ [degrees]
- $\dot{\gamma}(t)$: time-dependent shear rate [s⁻¹]
- $D(t)$: time-dependent diffusion coefficient [Å²/s]
- $\beta$: contrast parameter [dimensionless]

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

## Documentation

📚 **Complete Documentation**: https://homodyne.readthedocs.io/

Includes user guides, API reference, and developer documentation.

## Contributing

We welcome contributions! Please submit issues and pull requests.

**Development setup:**
```bash
git clone https://github.com/imewei/homodyne.git
cd homodyne
pip install -e .[all]
python homodyne/run_tests.py
```

**Authors:** Wei Chen, Hongrui He (Argonne National Laboratory)

**License:** MIT
