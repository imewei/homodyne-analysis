# Homodyne Scattering Analysis Package

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![Numba](https://img.shields.io/badge/Numba-JIT%20Accelerated-green)](https://numba.pydata.org/)

A high-performance Python package for analyzing homodyne scattering in X-ray Photon Correlation Spectroscopy (XPCS) under nonequilibrium conditions. Implements the theoretical framework from [He et al. PNAS 2024](https://doi.org/10.1073/pnas.2401162121) for characterizing transport properties in flowing soft matter systems.

## Overview

Analyzes time-dependent intensity correlation functions $c_2(\phi,t_1,t_2)$ in complex fluids under nonequilibrium conditions, capturing the interplay between Brownian diffusion and advective shear flow.

**Key Features:**
- **Three analysis modes**: Static Isotropic (3 params), Static Anisotropic (3 params), Laminar Flow (7 params)
- **Dual optimization**: Fast classical (Nelder-Mead) and robust Bayesian MCMC (NUTS)
- **High performance**: Numba JIT compilation with 3-5x speedup and smart angle filtering
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

# Generate C2 heatmaps
homodyne --method classical --plot-c2-heatmaps
```

### Data Validation

Generate validation plots without fitting:

```bash
homodyne --plot-experimental-data --config my_config.json --verbose
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

## Performance & Testing

### Optimization Methods

**Classical (Fast)**
- Algorithm: Nelder-Mead simplex
- Speed: ~minutes
- Use: Exploratory analysis, parameter screening
- Command: `--method classical`

**Bayesian MCMC (Comprehensive)**
- Algorithm: NUTS sampler via PyMC
- Speed: ~hours
- Use: Uncertainty quantification, publication results
- Command: `--method mcmc`

**Combined**
- Workflow: Classical → MCMC refinement
- Command: `--method all`

### Scaling Optimization

Always enabled for scientific accuracy:

$$g_2 = \text{offset} + \text{contrast} \times g_1$$

Accounts for instrumental effects, background, and normalization differences.

### Performance Tips

```bash
# Environment optimization
export OMP_NUM_THREADS=8
export NUMBA_DISABLE_INTEL_SVML=1

# Performance benchmark
python benchmark_performance.py --fast
```

### Testing

```bash
python homodyne/run_tests.py              # Standard tests
python homodyne/run_tests.py --fast       # Quick tests
python homodyne/run_tests.py --coverage   # With coverage
```

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
