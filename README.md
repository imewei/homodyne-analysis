# Homodyne Scattering Analysis Package

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![PyPI version](https://badge.fury.io/py/homodyne-analysis.svg)](https://badge.fury.io/py/homodyne-analysis)

Python package for analyzing homodyne scattering in X-ray Photon Correlation
Spectroscopy (XPCS) under nonequilibrium conditions. Implements theoretical framework
from [He et al. PNAS 2024](https://doi.org/10.1073/pnas.2401162121).

## Features

- **Three analysis modes**: Static Isotropic (3 params), Static Anisotropic (3 params),
  Laminar Flow (7 params)
- **Multiple optimization methods**: Classical (Nelder-Mead, Gurobi), Robust
  (Wasserstein DRO, Scenario-based, Ellipsoidal)
- **High performance**: Numba JIT compilation, vectorized operations
- **Noise resistance**: Robust optimization for measurement uncertainty

## Installation

### Standard Installation

```bash
pip install homodyne-analysis[all]
```

### Development Installation

```bash
git clone https://github.com/imewei/homodyne.git
cd homodyne
pip install -e .[all]
```

### Optional Dependencies

- **Performance**: `pip install homodyne-analysis[performance]` (numba, jax)
- **Robust optimization**: `pip install homodyne-analysis[robust]` (cvxpy)
- **Gurobi solver**: `pip install homodyne-analysis[gurobi]` (requires license)
- **Development**: `pip install homodyne-analysis[dev]` (all tools)

## Quick Start

```bash
# Install
pip install homodyne-analysis[all]

# Create configuration
homodyne-config --mode laminar_flow --sample my_sample

# Run analysis
homodyne --config my_config.json --method all

# Run only robust optimization
homodyne --config my_config.json --method robust
```

## Commands

### Main Analysis Command

```bash
homodyne [OPTIONS]
```

**Key Options:**

- `--method {classical,robust,all}` - Analysis method (default: classical)
- `--config CONFIG` - Configuration file (default: ./homodyne_config.json)
- `--output-dir DIR` - Output directory (default: ./homodyne_results)
- `--verbose` - Debug logging
- `--quiet` - File logging only
- `--static-isotropic` - Force 3-parameter isotropic mode
- `--static-anisotropic` - Force 3-parameter anisotropic mode
- `--laminar-flow` - Force 7-parameter flow mode
- `--plot-experimental-data` - Generate data validation plots
- `--plot-simulated-data` - Plot theoretical correlations

**Examples:**

```bash
# Basic analysis
homodyne --method classical
homodyne --method robust --verbose
homodyne --method all

# Force analysis modes
homodyne --static-isotropic --method classical
homodyne --laminar-flow --method all

# Data validation
homodyne --plot-experimental-data
homodyne --plot-simulated-data --contrast 1.5 --offset 0.1

# Custom configuration
homodyne --config experiment.json --output-dir ./results
```

### Configuration Generator

```bash
homodyne-config [OPTIONS]
```

**Options:**

- `--mode {static_isotropic,static_anisotropic,laminar_flow}` - Analysis mode
- `--output OUTPUT` - Output file (default: my_config.json)
- `--sample SAMPLE` - Sample name
- `--author AUTHOR` - Author name
- `--experiment EXPERIMENT` - Experiment description

**Examples:**

```bash
# Default laminar flow config
homodyne-config

# Static isotropic (fastest)
homodyne-config --mode static_isotropic --output fast_config.json

# With metadata
homodyne-config --sample protein --author "Your Name"
```

## Shell Completion

Install and enable shell completion:

```bash
# Install completion support
pip install homodyne-analysis[completion]

# Enable for your shell
homodyne --install-completion bash  # or zsh, fish, powershell

# Restart shell or reload config
source ~/.bashrc
```

**Shortcuts (always available):**

```bash
hc          # homodyne --method classical
hr          # homodyne --method robust
ha          # homodyne --method all
hconfig     # homodyne --config
hplot       # homodyne --plot-experimental-data
```

## Analysis Modes

| Mode | Parameters | Use Case | Speed | |------|------------|----------|-------| |
**Static Isotropic** | 3 | Isotropic systems, fastest analysis | ⭐⭐⭐ | | **Static
Anisotropic** | 3 | Static with angular dependence | ⭐⭐ | | **Laminar Flow** | 7 | Flow
and shear analysis | ⭐ |

### Static Isotropic (3 parameters)

- Parameters: D₀, α, D_offset
- No angle filtering
- Fastest mode for isotropic samples

### Static Anisotropic (3 parameters)

- Parameters: D₀, α, D_offset
- Angle filtering enabled
- For static samples with angular variations

### Laminar Flow (7 parameters)

- Parameters: D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀
- Full physics model with flow effects
- For systems under shear

## Python API

```python
from homodyne.analysis.core import HomodyneAnalysisCore
from homodyne.optimization.classical import ClassicalOptimizer
from homodyne.optimization.robust import RobustHomodyneOptimizer

# Load configuration file
config_file = "config_static_isotropic.json"  # Use actual config file
core = HomodyneAnalysisCore(config_file)

# Load experimental data
phi_angles = np.array([0, 36, 72, 108, 144])  # Example angles
c2_data = core.load_experimental_data(phi_angles, len(phi_angles))

# Classical optimization
classical = ClassicalOptimizer(core, config)
optimal_params, results = classical.run_classical_optimization_optimized(
    phi_angles=phi_angles,
    c2_experimental=c2_data
)

print(f"Optimal parameters: {optimal_params}")
print(f"Results: {results}")
```

## Optimization Methods

### Classical Methods

- **Nelder-Mead**: Derivative-free simplex algorithm
- **Gurobi**: Iterative trust region optimization (requires license)

### Robust Methods

- **Robust-Wasserstein**: Distributionally robust with Wasserstein uncertainty
- **Robust-Scenario**: Bootstrap scenario-based optimization
- **Robust-Ellipsoidal**: Ellipsoidal uncertainty sets

Use `--method robust` for noisy data with outliers. Use `--method classical` for clean,
low-noise data.

## Configuration

### Creating Configurations

```bash
# Generate templates
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

**Rules:**

- `static_mode: false` → Laminar Flow Mode (7 params)
- `static_mode: true, static_submode: "isotropic"` → Static Isotropic (3 params)
- `static_mode: true, static_submode: "anisotropic"` → Static Anisotropic (3 params)

## Output Structure

```
homodyne_results/
├── homodyne_analysis_results.json    # Main results
├── run.log                           # Execution log
├── classical/                        # Classical results
│   ├── all_classical_methods_summary.json
│   ├── nelder_mead/
│   │   ├── analysis_results_nelder_mead.json
│   │   ├── parameters.json
│   │   ├── fitted_data.npz
│   │   └── c2_heatmaps_*.png
│   └── gurobi/                       # If available
├── robust/                           # Robust results
│   ├── all_robust_methods_summary.json
│   ├── wasserstein/
│   ├── scenario/
│   └── ellipsoidal/
└── exp_data/                         # Data validation plots
```

## Performance

### Environment Optimization

```bash
export OMP_NUM_THREADS=8
export NUMBA_DISABLE_INTEL_SVML=1
export HOMODYNE_PERFORMANCE_MODE=1
```

### Optimizations

- **Numba JIT**: 3-5x speedup for core calculations
- **Vectorized operations**: Optimized array processing
- **Memory efficiency**: Smart caching and allocation
- **Batch processing**: Vectorized chi-squared calculation

## Physical Model

The package implements correlation functions in nonequilibrium laminar flow:

**Full Nonequilibrium Model:** $$c_2(\\vec{q}, t_1, t_2) = 1 +
\\beta\\left\[e^{-q^2\\int J(t)dt}\\right\] \\times
\\text{sinc}^2\\left\[\\frac{1}{2\\pi} qh
\\int\\dot{\\gamma}(t)\\cos(\\phi(t))dt\\right\]$$

**Key Parameters:**

- $\\vec{q}$: scattering wavevector [Å⁻¹]
- $h$: gap between stator and rotor [Å]
- $\\phi(t)$: angle between shear/flow direction and $\\vec{q}$ [degrees]
- $\\dot{\\gamma}(t)$: time-dependent shear rate [s⁻¹]
- $D(t)$: time-dependent diffusion coefficient [Å²/s]

## Citation

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

## Development

Development setup:

```bash
git clone https://github.com/imewei/homodyne.git
cd homodyne
pip install -e .[all]

# Run tests
python homodyne/run_tests.py

# Code quality
black homodyne/
isort homodyne/
flake8 homodyne/
```

**Authors:** Wei Chen, Hongrui He (Argonne National Laboratory) **License:** MIT
