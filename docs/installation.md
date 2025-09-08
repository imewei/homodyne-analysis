# Installation Guide - Homodyne Analysis

## Quick Start

### Basic Installation (Core Features Only)
```bash
pip install homodyne-analysis
```
This installs only the essential dependencies: numpy, scipy, matplotlib

### Installation Options

#### Performance Optimization
```bash
pip install homodyne-analysis[performance]
```
Adds Numba for JIT compilation and significant speedup

#### MCMC Sampling (CPU Backend)
```bash
pip install homodyne-analysis[mcmc]
```
Adds PyMC, ArviZ, and Corner for Bayesian analysis

#### GPU Acceleration
```bash
pip install homodyne-analysis[gpu]
```
Adds JAX and NumPyro for GPU-accelerated sampling (Linux only for CUDA)

#### Robust Optimization
```bash
pip install homodyne-analysis[robust]
```
Adds CVXPY and scikit-learn for robust optimization methods

#### Data Handling
```bash
pip install homodyne-analysis[data]
```
Adds xpcs-viewer for XPCS data file support

#### Commercial Solver
```bash
pip install homodyne-analysis[gurobi]
```
Adds Gurobi solver support (requires license)

### Combined Installations

#### All Scientific Features
```bash
pip install homodyne-analysis[all]
```
Includes: performance, mcmc, gpu, robust, data

#### Development Environment
```bash
pip install -e ".[dev]"
```
Includes testing tools, code formatters, and development utilities

#### Full Development with All Features
```bash
pip install -e ".[dev-all]"
```
Everything including all scientific features and development tools

## From Source

### Clone and Install
```bash
git clone https://github.com/imewei/homodyne.git
cd homodyne
pip install -e .  # Basic installation
# or
pip install -e ".[all]"  # With all features
```

### For Contributors
```bash
pip install -e ".[dev-all]"
pre-commit install  # Set up code quality checks
```

## Requirements Files

For environments that need requirements.txt files:

```bash
# Core only
pip install -r requirements.txt

# Development
pip install -r requirements-dev.txt
```

## Platform Notes

- **Linux**: Full GPU support with CUDA 12.6+
- **macOS/Windows**: CPU-only for all backends (JAX runs in CPU mode)
- **Python**: Requires Python 3.12+

## Troubleshooting

If you encounter conflicts between PyMC and JAX:
- Use `[mcmc]` for CPU-only Bayesian analysis
- Use `[gpu]` for GPU-accelerated analysis
- They are designed to be used separately

For Numba threading issues:
```bash
export NUMBA_NUM_THREADS=1
```

## Minimal Docker Example

```dockerfile
FROM python:3.12-slim
RUN pip install homodyne-analysis[performance,robust]
```