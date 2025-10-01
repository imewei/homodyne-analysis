# Homodyne Scattering Analysis Package

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![PyPI version](https://badge.fury.io/py/homodyne-analysis.svg)](https://badge.fury.io/py/homodyne-analysis)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-green.svg)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy-1.9+-green.svg)](https://scipy.org)
[![Numba](https://img.shields.io/badge/Numba-JIT-orange.svg)](https://numba.pydata.org)
[![DOI](https://img.shields.io/badge/DOI-10.1073/pnas.2401162121-blue.svg)](https://doi.org/10.1073/pnas.2401162121)
[![Research](https://img.shields.io/badge/Research-XPCS%20Nonequilibrium-purple.svg)](https://github.com/imewei/homodyne)

## Overview

**homodyne-analysis** is a high-performance Python package for analyzing homodyne
scattering in X-ray Photon Correlation Spectroscopy (XPCS) under nonequilibrium
conditions. This research-grade software implements the theoretical framework from
[He et al. PNAS 2024](https://doi.org/10.1073/pnas.2401162121) for characterizing
transport properties in flowing soft matter systems through time-dependent intensity
correlation functions.

The package enables comprehensive analysis of nonequilibrium dynamics by capturing the
interplay between Brownian diffusion and advective shear flow in complex fluids, with
applications to biological systems, colloids, and active matter under flow conditions.

## Features

- **Three analysis modes**: Static Isotropic (3 params), Static Anisotropic (3 params),
  Laminar Flow (7 params)
- **Multiple optimization methods**: Classical (Nelder-Mead, Gurobi), Robust
  (Wasserstein DRO, Scenario-based, Ellipsoidal)
- **High performance**: Numba JIT compilation, vectorized operations
- **Noise resistance**: Robust optimization for measurement uncertainty
- **Research-grade validation**: Experimental validation with synchrotron facilities

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

## Installation

### Standard Installation

```bash
pip install homodyne-analysis[all]
```

### Research Environment Setup

```bash
# Create isolated research environment
conda create -n homodyne-research python=3.12
conda activate homodyne-research

# Install with all scientific dependencies
pip install homodyne-analysis[all]

# For development and method extension
git clone https://github.com/imewei/homodyne.git
cd homodyne
pip install -e .[dev]
```

### Optional Dependencies

- **Performance**: `pip install homodyne-analysis[performance]` (numba, jax)
- **Robust optimization**: `pip install homodyne-analysis[robust]` (cvxpy)
- **Gurobi solver**: `pip install homodyne-analysis[gurobi]` (requires license)
- **Development**: `pip install homodyne-analysis[dev]` (all tools)

### High-Performance Configuration

```bash
# Optimize for computational performance
export OMP_NUM_THREADS=8
export NUMBA_DISABLE_INTEL_SVML=1
export HOMODYNE_PERFORMANCE_MODE=1

# Enable advanced optimization (requires license)
pip install homodyne-analysis[gurobi]
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

## Scientific Background

### Physical Model

The package analyzes time-dependent intensity correlation functions in the presence of
laminar flow:

$$c_2(\\vec{q}, t_1, t_2) = 1 + \\beta\\left[e^{-q^2\\int J(t)dt}\\right] \\times
\\text{sinc}^2\\left\[\\frac{1}{2\\pi} qh
\\int\\dot{\\gamma}(t)\\cos(\\phi(t))dt\\right\]$$

where:

- $\\vec{q}$: scattering wavevector [Å⁻¹]
- $h$: gap between stator and rotor [Å]
- $\\phi(t)$: angle between shear/flow direction and $\\vec{q}$ [degrees]
- $\\dot{\\gamma}(t)$: time-dependent shear rate [s⁻¹]
- $D(t)$: time-dependent diffusion coefficient [Å²/s]
- $\\beta$: instrumental contrast parameter

### Analysis Modes

| Mode | Parameters | Physical Description | Computational Complexity | Speed |
|------|------------|---------------------|--------------------------|-------| |
**Static Isotropic** | 3 | Brownian motion only, isotropic systems | O(N) | ⭐⭐⭐ | |
**Static Anisotropic** | 3 | Static systems with angular dependence | O(N log N) | ⭐⭐ |
| **Laminar Flow** | 7 | Full nonequilibrium with flow and shear | O(N²) | ⭐ |

#### Model Parameters

**Static Parameters (All Modes):**

- $D_0$: baseline diffusion coefficient [Å²/s]
- $\\alpha$: diffusion scaling exponent
- $D\_{\\text{offset}}$: additive diffusion offset [Å²/s]

**Flow Parameters (Laminar Flow Mode):**

- $\\dot{\\gamma}\_0$: baseline shear rate [s⁻¹]
- $\\beta$: shear rate scaling exponent
- $\\dot{\\gamma}\_{\\text{offset}}$: additive shear rate offset [s⁻¹]
- $\\phi_0$: flow direction angle [degrees]

## Optimization Methods

### Classical Methods

1. **Nelder-Mead Simplex**: Derivative-free optimization for robust convergence
1. **Gurobi Quadratic Programming**: High-performance commercial solver with trust
   region methods

### Robust Optimization Framework

Advanced uncertainty-aware optimization for noisy experimental data:

1. **Distributionally Robust Optimization (DRO)**:

   - Wasserstein uncertainty sets for data distribution robustness
   - Optimal transport-based uncertainty quantification

1. **Scenario-Based Optimization**:

   - Bootstrap resampling for statistical robustness
   - Monte Carlo uncertainty propagation

1. **Ellipsoidal Uncertainty Sets**:

   - Bounded uncertainty with confidence ellipsoids
   - Analytical uncertainty bounds

Use `--method robust` for noisy data with outliers. Use `--method classical` for clean,
low-noise data.

## Python API

### Basic Usage

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

### Research Workflow

```python
import numpy as np
from homodyne.analysis.core import HomodyneAnalysisCore
from homodyne.optimization.classical import ClassicalOptimizer
from homodyne.optimization.robust import RobustHomodyneOptimizer

# Load experimental configuration
config_file = "config_laminar_flow.json"
core = HomodyneAnalysisCore(config_file)

# Load XPCS correlation data
phi_angles = np.array([0, 36, 72, 108, 144])  # Experimental angles
c2_data = core.load_experimental_data(phi_angles, len(phi_angles))

# Classical analysis for clean data
classical = ClassicalOptimizer(core, config)
classical_params, classical_results = classical.run_classical_optimization_optimized(
    phi_angles=phi_angles,
    c2_experimental=c2_data
)

# Robust analysis for noisy data
robust = RobustHomodyneOptimizer(core, config)
robust_params, robust_results = robust.optimize_wasserstein_robust(
    phi_angles=phi_angles,
    c2_experimental=c2_data,
    epsilon=0.1  # Uncertainty radius
)

print(f"Classical D₀: {classical_params[0]:.3e} Å²/s")
print(f"Robust D₀: {robust_params[0]:.3e} ± {robust_results['uncertainty'][0]:.3e} Å²/s")
```

### Performance Benchmarking

```python
from homodyne.performance_monitoring import benchmark_analysis

# Benchmark different optimization methods
results = benchmark_analysis(
    config_file="config_laminar_flow.json",
    methods=['classical', 'robust'],
    data_sizes=[100, 500, 1000, 5000],
    repeats=10
)

# Generate performance plots
results.plot_performance_comparison(save_path="performance_analysis.png")
```

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

### Data Formats and Standards

**XPCS Correlation Data Format:**

- Time correlation functions: `c2(q, φ, t1, t2)` as HDF5 or NumPy arrays
- Scattering angles: φ values in degrees \[0°, 360°)
- Time delays: τ = t2 - t1 in seconds
- Wavevector magnitude: q in Å⁻¹

**Configuration Schema:**

```json
{
  "analysis_settings": {
    "static_mode": false,
    "static_submode": null,
    "angle_filtering": true,
    "optimization_method": "all"
  },
  "experimental_parameters": {
    "q_magnitude": 0.0045,
    "gap_height": 50000.0,
    "temperature": 293.15,
    "viscosity": 1.0e-3
  },
  "optimization_bounds": {
    "D0": [1e-15, 1e-10],
    "alpha": [0.1, 2.0],
    "D_offset": [-1e-12, 1e-12]
  }
}
```

## Output Structure

When running `homodyne --method all`, the complete analysis produces a comprehensive results directory with all optimization methods:

```
homodyne_results/
├── homodyne_analysis_results.json    # Summary with all methods
├── run.log                           # Detailed execution log
│
├── classical/                        # Classical optimization results
│   ├── nelder_mead/                  # Nelder-Mead simplex method
│   │   ├── parameters.json           # Optimal parameters with metadata
│   │   ├── fitted_data.npz          # Fitted correlation functions + experimental metadata
│   │   ├── analysis_results_nelder_mead.json  # Complete results + chi-squared
│   │   ├── convergence_metrics.json  # Iterations, function evaluations, diagnostics
│   │   └── c2_heatmaps_phi_*.png    # Experimental vs fitted comparison plots
│   └── gurobi/                       # Gurobi quadratic programming (if available)
│       ├── parameters.json
│       ├── fitted_data.npz
│       ├── analysis_results_gurobi.json
│       ├── convergence_metrics.json
│       └── c2_heatmaps_phi_*.png
│
├── robust/                           # Robust optimization results
│   ├── wasserstein/                  # Distributionally Robust Optimization (DRO)
│   │   ├── parameters.json           # Robust optimal parameters
│   │   ├── fitted_data.npz          # Fitted correlations with uncertainty bounds
│   │   ├── analysis_results_wasserstein.json  # DRO results + uncertainty radius
│   │   ├── convergence_metrics.json  # Optimization convergence info
│   │   └── c2_heatmaps_phi_*.png    # Robust fit comparison plots
│   ├── scenario/                     # Scenario-based bootstrap optimization
│   │   ├── parameters.json
│   │   ├── fitted_data.npz
│   │   ├── analysis_results_scenario.json
│   │   ├── convergence_metrics.json
│   │   └── c2_heatmaps_phi_*.png
│   └── ellipsoidal/                  # Ellipsoidal uncertainty sets
│       ├── parameters.json
│       ├── fitted_data.npz
│       ├── analysis_results_ellipsoidal.json
│       ├── convergence_metrics.json
│       └── c2_heatmaps_phi_*.png
│
└── comparison_plots/                 # Method comparison visualizations
    ├── method_comparison_phi_*.png   # Classical vs Robust comparison
    └── parameter_comparison.png      # Parameter values across methods
```

### Key Output Files

**homodyne_analysis_results.json**: Main summary containing:
- Analysis timestamp and methods run
- Experimental parameters (q, dt, gap size, frames)
- Optimization results for all methods:
  - `classical_nelder_mead`, `classical_gurobi`, `classical_best`
  - `robust_wasserstein`, `robust_scenario`, `robust_ellipsoidal`, `robust_best`

**fitted_data.npz**: NumPy compressed archive with:
- Experimental metadata: `wavevector_q`, `dt`, `stator_rotor_gap`, `start_frame`, `end_frame`
- Correlation data: `c2_experimental`, `c2_theoretical_raw`, `c2_theoretical_scaled`
- Scaling parameters: `contrast_params`, `offset_params`
- Quality metrics: `residuals`

**analysis_results_{method}.json**: Method-specific detailed results:
- Optimized parameters with names
- Chi-squared and reduced chi-squared values
- Experimental metadata
- Scaling parameters for each angle
- Success status and timestamp

**convergence_metrics.json**: Optimization diagnostics:
- Number of iterations
- Function evaluations
- Convergence message
- Final chi-squared value

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

### Benchmarking Results

**Performance Comparison (Intel Xeon, 8 cores):**

| Data Size | Pure Python | Numba JIT | Speedup |
|-----------|-------------|-----------|---------| | 100 points | 2.3 s | 0.7 s | 3.3× |
| 500 points | 12.1 s | 3.2 s | 3.8× | | 1000 points | 45.2 s | 8.9 s | 5.1× | | 5000
points | 892 s | 178 s | 5.0× |

## Research Applications

### Soft Matter Physics

- **Colloidal Dynamics**: Particle diffusion in crowded environments
- **Active Matter**: Self-propelled particle systems under flow
- **Biological Systems**: Protein dynamics in cellular environments

### Flow Rheology

- **Shear Thinning/Thickening**: Non-Newtonian fluid characterization
- **Microfluidics**: Flow behavior in confined geometries
- **Complex Fluids**: Polymer solutions and suspensions

### Materials Science

- **Phase Transitions**: Dynamic behavior near critical points
- **Glass Transition**: Aging and relaxation dynamics
- **Crystallization**: Nucleation and growth processes

## Research Validation

### Experimental Validation

- **Synchrotron Facilities**: Advanced Photon Source (APS) Sector 8-ID-I
- **Sample Systems**: Colloidal suspensions, protein solutions, active matter
- **Flow Conditions**: Laminar shear, pressure-driven flow, microfluidic geometries

### Statistical Validation

- **Cross-validation**: K-fold validation for parameter estimation reliability
- **Bootstrap Analysis**: Statistical uncertainty quantification
- **Residual Analysis**: Goodness-of-fit assessment and outlier detection

## Citation

If you use this software in your research, please cite the original paper:

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

**For the software package:**

```bibtex
@software{homodyne_analysis,
  title={homodyne-analysis: High-performance XPCS analysis with robust optimization},
  author={Chen, Wei and He, Hongrui},
  year={2024},
  url={https://github.com/imewei/homodyne},
  version={0.7.1},
  institution={Argonne National Laboratory}
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

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed development guidelines.

## Acknowledgments

### Funding Support

- U.S. Department of Energy, Office of Science, Basic Energy Sciences
- National Science Foundation (NSF) Division of Materials Research
- Advanced Photon Source User Facility

### Collaborating Institutions

- **Argonne National Laboratory** - X-ray Science Division
- **University of Chicago** - Institute for Molecular Engineering
- **Northwestern University** - Department of Materials Science

### Technical Contributors

- **Wei Chen** (Argonne National Laboratory) - Principal Investigator
- **Hongrui He** (Argonne National Laboratory) - Lead Developer
- **Advanced Photon Source** - Experimental facility and user support

## License

This research software is distributed under the MIT License, enabling open collaboration
while maintaining attribution requirements for academic use.

**Research Use**: Freely available for academic research with proper citation
**Commercial Use**: Permitted under MIT License terms **Modification**: Encouraged with
contribution back to the community

______________________________________________________________________

**Contact Information:**

- **Primary Investigator**: Wei Chen ([wchen@anl.gov](mailto:wchen@anl.gov))
- **Technical Support**: [GitHub Issues](https://github.com/imewei/homodyne/issues)
- **Research Collaboration**: Argonne National Laboratory, X-ray Science Division

**Authors:** Wei Chen, Hongrui He (Argonne National Laboratory) **License:** MIT
