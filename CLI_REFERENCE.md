# Homodyne CLI - Actual Implementation

**Version 0.6.6+** | **Python 3.12+ Required** | **Trust Region Gurobi ✅** | **Code
Quality: Black ✅ isort ✅**

## Overview

The Homodyne project provides three command-line tools for analyzing X-ray Photon
Correlation Spectroscopy (XPCS) data:

1. **`homodyne`** - Main analysis command with enhanced Gurobi trust region optimization
1. **`homodyne-gpu`** - GPU-accelerated analysis with automatic JAX/CUDA activation
1. **`homodyne-config`** - Configuration file generator

**Recent Improvements (v0.6.6+):**

- **Enhanced Gurobi optimization** with iterative trust region SQP approach
- **Improved code quality** with comprehensive formatting and linting (Black, isort)
- **Shell completion support** for enhanced CLI experience
- **Removed interactive mode** - use shell completion instead
- **308 lines of code cleanup** from unused implementations

## Available Commands

### 1. `homodyne` - Main Analysis Command

Runs the complete homodyne scattering analysis for XPCS under nonequilibrium conditions.

#### Usage

```bash
homodyne [OPTIONS]
```

#### Options

```bash
  -h, --help                    Show help message and exit
  --method {classical,mcmc,robust,all}
                                Analysis method to use (default: classical)
  --config CONFIG               Path to configuration file (default: ./homodyne_config.json)
  --output-dir OUTPUT_DIR       Output directory for results (default: ./homodyne_results)
  --verbose                     Enable verbose DEBUG logging
  --quiet                       Disable console logging (file logging remains enabled)
  --static-isotropic            Force static isotropic mode analysis (3 parameters, no angle selection)
  --static-anisotropic          Force static anisotropic mode analysis (3 parameters, with angle selection)
  --laminar-flow                Force laminar flow mode analysis (7 parameters: all diffusion and shear parameters)
  --plot-experimental-data      Generate validation plots of experimental data after loading for quality checking
  --plot-simulated-data         Plot theoretical C2 heatmaps using initial parameters from config without experimental data
  --contrast CONTRAST           Contrast parameter for scaling: fitted = contrast * theory + offset (default: 1.0)
  --offset OFFSET              Offset parameter for scaling: fitted = contrast * theory + offset (default: 0.0)
  --phi-angles PHI_ANGLES       Comma-separated list of phi angles in degrees (e.g., '0,45,90,135'). Default: '0,36,72,108,144'
  --install-completion {bash,zsh,fish,powershell}
                                Install shell completion for the specified shell
  --uninstall-completion {bash,zsh,fish,powershell}
                                Uninstall shell completion for the specified shell
```

#### Methods

- **classical**: Traditional optimization (Nelder-Mead, **Enhanced Gurobi with Trust
  Regions**)
  - **Gurobi**: Iterative trust region SQP optimization for robust convergence (requires
    license)
  - **Nelder-Mead**: Derivative-free simplex algorithm, robust for noisy functions
- **robust**: Robust optimization with uncertainty quantification (requires CVXPY)
- **mcmc**: Bayesian MCMC sampling using PyMC and NUTS
- **all**: Run all available methods

**Gurobi Trust Region Features (v0.6.5+):**

- **Iterative optimization**: Up to 50 outer iterations with progressive χ² improvement
- **Adaptive trust region**: Radius adapts from 0.1 → 1e-8 to 1.0 based on step quality
- **Parameter-scaled finite differences**: Enhanced numerical stability
- **Progress logging**: Debug messages show iteration progress and convergence metrics
- **Expected convergence**: 10-30 iterations for typical XPCS problems

#### Examples

```bash
# Run with default classical method
homodyne

# Run only robust optimization methods
homodyne --method robust

# Run all methods with verbose logging
homodyne --method all --verbose

# Use custom config file
homodyne --config my_config.json

# Custom output directory with verbose logging
homodyne --output-dir ./homodyne_results --verbose

# Run MCMC analysis with custom config
homodyne --method mcmc --config mcmc_config.json

# Force static isotropic mode
homodyne --static-isotropic --method classical

# Generate experimental data validation plots
homodyne --plot-experimental-data

# Plot simulated data with default scaling
homodyne --plot-simulated-data

# Plot simulated data with custom scaling
homodyne --plot-simulated-data --contrast 1.5 --offset 0.1

# Custom phi angles for simulated data
homodyne --plot-simulated-data --phi-angles "0,45,90,135"

# Quiet mode (only file logging)
homodyne --quiet --method all
```

#### Output Structure

```
homodyne_results/
├── homodyne_analysis_results.json    # Main results with config and metadata
├── run.log                          # Execution log file
├── classical/                      # Classical optimization results (if run)
│   ├── all_classical_methods_summary.json
│   ├── nelder_mead/                # Method-specific directory
│   │   ├── analysis_results_nelder_mead.json
│   │   ├── parameters.json
│   │   ├── fitted_data.npz         # Experimental, fitted, residuals data
│   │   ├── c2_heatmaps_nelder_mead_phi_*.png
│   │   └── nelder_mead_diagnostic_summary.png
│   ├── gurobi/                     # Gurobi method directory (if available)
│   │   ├── analysis_results_gurobi.json
│   │   ├── parameters.json
│   │   ├── fitted_data.npz
│   │   └── c2_heatmaps_gurobi_phi_*.png
│   └── ...                         # Other classical methods
├── robust/                         # Robust optimization results (if run)
│   ├── all_robust_methods_summary.json
│   ├── wasserstein/               # Robust method directories
│   │   ├── analysis_results_wasserstein.json
│   │   ├── parameters.json
│   │   ├── fitted_data.npz
│   │   └── c2_heatmaps_wasserstein_phi_*.png
│   ├── scenario/
│   ├── ellipsoidal/
│   └── ...
├── mcmc/                          # MCMC results (if run)
│   ├── mcmc_summary.json          # MCMC summary statistics
│   ├── mcmc_trace.nc              # NetCDF trace file
│   ├── experimental_data.npz      # Original experimental data
│   ├── fitted_data.npz            # MCMC fitted data
│   ├── residuals_data.npz         # Residuals
│   ├── c2_heatmaps_phi_*.png      # Heatmap plots per angle
│   ├── 3d_surface_phi_*.png       # 3D surface plots
│   ├── 3d_surface_residuals_phi_*.png
│   ├── trace_plot.png             # MCMC trace plots
│   └── corner_plot.png            # Parameter posterior distributions
├── exp_data/                      # Experimental data plots (if --plot-experimental-data)
│   ├── data_validation_phi_*.png  # Per-angle validation plots
│   └── summary_statistics.txt     # Data summary
└── simulated_data/               # Simulated data plots (if --plot-simulated-data)
    ├── simulated_c2_fitted_phi_*.png    # Simulated fitted data plots
    ├── theoretical_c2_phi_*.png         # Theoretical correlation plots
    ├── fitted_c2_data.npz              # Fitted data arrays
    └── theoretical_c2_data.npz         # Theoretical data arrays
```

______________________________________________________________________

### 2. `homodyne-gpu` - GPU-Accelerated Analysis Command

Identical to `homodyne` but automatically activates GPU support by sourcing
`activate_gpu.sh` before running analysis. Provides seamless GPU acceleration with
JAX/CUDA without manual environment setup.

#### Usage

```bash
homodyne-gpu [OPTIONS]
```

#### Options

All options are identical to `homodyne` command. See Section 1 above for complete option
list.

#### GPU Requirements

- **CUDA-capable GPU** with compute capability 5.2+
- **JAX with CUDA support** installed via:
  ```bash
  pip install homodyne-analysis[jax]  # Installs JAX with CUDA 12 support on Linux
  ```
- **GPU activation script**: `activate_gpu.sh` must be present in:
  - Current working directory, OR
  - Package installation directory, OR
  - Site-packages directory

#### Examples

```bash
# GPU-accelerated classical analysis
homodyne-gpu --method classical

# GPU-accelerated MCMC with all methods
homodyne-gpu --method all --verbose

# GPU analysis with custom config
homodyne-gpu --config gpu_config.json --method mcmc

# Force static mode with GPU acceleration
homodyne-gpu --static-isotropic --method classical
```

#### Performance Benefits

- **5-10x speedup** for MCMC sampling with NumPyro/JAX
- **3-5x speedup** for classical optimization with JAX-compiled kernels
- **Automatic fallback** to CPU if GPU activation fails
- **Seamless integration** - same interface as `homodyne`

#### GPU Activation Process

1. **Automatic detection**: Searches for `activate_gpu.sh` in common locations
1. **Environment setup**: Sources script to configure CUDA paths and environment
   variables
1. **Fallback handling**: Continues with CPU if GPU activation fails
1. **Transparent operation**: Same analysis pipeline with GPU acceleration

______________________________________________________________________

### 3. `homodyne-config` - Configuration Generator

Creates homodyne analysis configuration files from mode-specific templates.

#### Usage

```bash
homodyne-config [OPTIONS]
```

#### Options

```bash
  -h, --help            Show help message and exit
  --mode, -m {static_isotropic,static_anisotropic,laminar_flow}
                        Analysis mode (default: laminar_flow)
  --output, -o OUTPUT   Output configuration file name (default: my_config.json)
  --sample, -s SAMPLE   Sample name (used in data paths)
  --experiment, -e EXPERIMENT
                        Experiment description
  --author, -a AUTHOR   Author name
```

#### Analysis Modes

1. **static_isotropic** - Fastest: Single dummy angle, no angle filtering, 3 parameters
1. **static_anisotropic** - Static with angle filtering optimization, 3 parameters
1. **laminar_flow** - Full flow analysis with 7 parameters (default)

#### Examples

```bash
# Create laminar flow configuration (default)
homodyne-config

# Create static isotropic config (fastest mode)
homodyne-config --mode static_isotropic --output static_config.json

# Create config with sample name and metadata
homodyne-config --sample protein_sample --author "Your Name" --experiment "Protein dynamics study"

# Create static anisotropic config with all metadata
homodyne-config --mode static_anisotropic --sample collagen --author "Your Name" --experiment "Collagen static analysis"
```

#### Generated Configuration Structure

```json
{
    "metadata": {
        "config_version": "1.0.0",
        "analysis_mode": "laminar_flow"
    },
    "experimental_data": {
        "data_folder_path": "./path/to/data/"
    },
    "analyzer_parameters": {
        "temporal": {
            "dt": 0.1,
            "start_frame": 1,
            "end_frame": 1000
        },
        "scattering": {
            "wavevector_q": 0.01
        }
    },
    "initial_parameters": {
        "values": [100.0, -0.5, 10.0, ...],
        "parameter_names": ["D0", "alpha", "D_offset", ...]
    },
    "optimization_config": {
        "classical_optimization": {
            "methods": ["Nelder-Mead"]
        },
        "mcmc_sampling": {
            "enabled": false,
            "chains": 4,
            "draws": 2000,
            "tune": 1000
        }
    }
}
```

## Environment Variables

Both commands respect these environment variables:

```bash
# Set default configuration file
export HOMODYNE_CONFIG=/path/to/config.json

# Set default output directory
export HOMODYNE_OUTPUT_DIR=/path/to/output

# Set logging level (DEBUG, INFO, WARNING, ERROR)
export HOMODYNE_LOG_LEVEL=INFO

# Enable profiling (development)
export HOMODYNE_PROFILE=1
```

## Exit Codes

| Code | Meaning | |------|---------| | 0 | Success | | 1 | General error or Python
version mismatch | | 2 | Configuration error | | 3 | Data format/loading error | | 4 |
Optimization failed | | 5 | Missing dependencies |

## Python Module Usage

Both commands can also be used programmatically:

```python
# Using the main analysis
from homodyne.run_homodyne import main as run_analysis
run_analysis()  # Uses sys.argv

# Creating configuration programmatically
from homodyne.create_config import create_config_from_template
create_config_from_template(
    output_file="my_config.json",
    mode="static_isotropic",
    sample_name="test_sample",
    author="Your Name"
)
```

## Workflow Example

### Complete Analysis Workflow

```bash
# Step 1: Generate configuration
homodyne-config --mode laminar_flow --sample experimental_data --output exp_config.json

# Step 2: Review/edit configuration
# (Edit exp_config.json as needed)

# Step 3: Run analysis
homodyne --config exp_config.json --method all --output-dir ./analysis_results --verbose

# Step 4: Review results
cd analysis_results
cat optimization_results.json  # Numerical results
ls plots/                       # Generated plots
```

### Quick Start for Testing

```bash
# Generate default config
homodyne-config

# Run with default settings
homodyne

# Results will be in ./homodyne_results/
```

## Limitations and Notes

1. **No subcommands**: Unlike the idealized CLI structure, the current implementation
   doesn't have subcommands like `homodyne analyze`, `homodyne mcmc`, etc.

1. **Method selection**: Different optimization methods are selected via the `--method`
   flag, not separate commands.

1. **Data preprocessing**: No separate preprocessing command; preprocessing happens
   automatically during analysis.

1. **Validation**: No separate validation command; validation is integrated into the
   analysis pipeline.

1. **Python 3.12+ required**: Both commands check for Python 3.12 or higher.

1. **Optional dependencies**:

   - MCMC requires PyMC installation
   - Robust optimization requires CVXPY
   - Some features require numba for performance
   - Shell completion requires `argcomplete`
   - **Note**: Interactive CLI mode has been **removed** in v0.6.5+ - use shell
     completion for enhanced CLI experience

## Shell Completion

The homodyne CLI supports multiple completion methods for enhanced user experience
across shells.

### Installation

Install completion support with optional dependencies:

```bash
pip install homodyne-analysis[completion]  # Updated in v0.6.5+
```

**Note**: The `interactive` dependency group has been **removed** in v0.6.5+. Use
`completion` group for shell tab completion only.

Enable shell completion for your shell (one-time setup):

```bash
# For bash
homodyne --install-completion bash

# For zsh  
homodyne --install-completion zsh

# For fish
homodyne --install-completion fish

# For PowerShell
homodyne --install-completion powershell
```

To remove shell completion later:

```bash
# For bash
homodyne --uninstall-completion bash

# For zsh  
homodyne --uninstall-completion zsh

# For fish
homodyne --uninstall-completion fish

# For PowerShell
homodyne --uninstall-completion powershell
```

After installation, restart your shell or source the configuration file:

```bash
# Bash/Zsh
source ~/.bashrc  # or ~/.zshrc

# Fish
source ~/.config/fish/config.fish
```

### Completion Features

The completion system provides multiple tiers of functionality:

#### 1. **Tab Completion** (Primary Method)

- **Method completion**: Tab complete `--method` options (classical, mcmc, robust, all)
- **File completion**: Tab complete `--config` with available .json files
- **Directory completion**: Tab complete `--output-dir` with available directories
- **Context-aware**: Completion adapts based on current command context
- **Cross-platform**: Works on Linux, macOS, and Windows

#### 2. **Command Shortcuts** (Always Available)

Fast shortcuts for common operations:

```bash
# homodyne shortcuts
hc          # homodyne --method classical
hm          # homodyne --method mcmc
hr          # homodyne --method robust
ha          # homodyne --method all
hconfig     # homodyne --config
hplot       # homodyne --plot-experimental-data

# homodyne-gpu shortcuts  
hgc         # homodyne-gpu --method classical
hgm         # homodyne-gpu --method mcmc
hgr         # homodyne-gpu --method robust
hga         # homodyne-gpu --method all
hgconfig    # homodyne-gpu --config
```

#### 3. **Completion Help System**

```bash
homodyne_help    # Show all available options and current config files
```

### Usage Examples

#### Tab Completion (if working)

```bash
# Tab completion examples (press TAB at cursor position)
homodyne --method <TAB>          # Shows: classical, mcmc, robust, all
homodyne --config <TAB>          # Shows available .json files
homodyne --output-dir <TAB>      # Shows available directories
homodyne-gpu --method <TAB>      # Shows: classical, mcmc, robust, all (same as homodyne)
homodyne-config --mode <TAB>     # Shows: static_isotropic, static_anisotropic, laminar_flow
```

#### Command Shortcuts (always work)

```bash
# Quick method selection - CPU
hc                               # homodyne --method classical
hm --verbose                     # homodyne --method mcmc --verbose
hr --config my_config.json       # homodyne --method robust --config my_config.json
ha                               # homodyne --method all

# Quick method selection - GPU
hgc                              # homodyne-gpu --method classical
hgm --verbose                    # homodyne-gpu --method mcmc --verbose
hgr --config gpu_config.json     # homodyne-gpu --method robust --config gpu_config.json
hga                              # homodyne-gpu --method all

# Quick config and plotting
hconfig my_config.json           # homodyne --config my_config.json
hgconfig gpu_config.json         # homodyne-gpu --config gpu_config.json
hplot                            # homodyne --plot-experimental-data
```

#### Help and Reference

```bash
homodyne_help                    # Show completion help and available options
```

### Manual Completion (Bypass Method)

If standard tab completion fails, use the bypass completion system with manual key
bindings:

#### Loading Bypass Completion

```bash
# Load bypass completion in current session
source /path/to/homodyne_completion_bypass.zsh

# Add to .zshrc for permanent use
echo "source /path/to/homodyne_completion_bypass.zsh" >> ~/.zshrc
```

#### Manual Key Bindings

When tab completion doesn't work, use these manual key combinations:

```bash
# Manual completion key bindings (after loading bypass script)
Ctrl-X h                         # Manual homodyne completion
Ctrl-X g                         # Manual homodyne-gpu completion  
Ctrl-X c                         # Manual homodyne-config completion
```

#### Bypass Script Features

The bypass script provides multiple fallback mechanisms:

1. **Manual key bindings** for direct completion when compdef fails
1. **Command shortcuts** that always work regardless of completion status
1. **Help system** with `homodyne_help` function
1. **Automatic compdef** registration with fallback handling

### Troubleshooting Completion

The completion system is designed to be robust with multiple fallback mechanisms:

#### If Tab Completion Doesn't Work

1. **Load bypass script**: `source homodyne_completion_bypass.zsh`
1. **Use manual keys**: `Ctrl-X h` (homodyne), `Ctrl-X g` (homodyne-gpu), `Ctrl-X c`
   (config)
1. **Use command shortcuts**: `hc`, `hm`, `hr`, `ha`, `hgc`, `hgm`, `hgr`, `hga` always
   work
1. **Check installation**: Run `homodyne --install-completion zsh` again
1. **Reload shell**: `source ~/.zshrc` or restart terminal
1. **Use help reference**: `homodyne_help` shows all options

#### Common Issues

```bash
# Issue: Tab completion not working after installation  
# Solution: Restart shell or source config
exec zsh                         # Restart shell
# OR
source ~/.zshrc                  # Reload config

# Issue: compdef errors in zsh
# Solution: Use bypass script and manual keys or shortcuts
source homodyne_completion_bypass.zsh  # Load bypass script
# Then use: Ctrl-X h for homodyne, Ctrl-X g for homodyne-gpu
# OR use shortcuts which always work:
hc --verbose                     # Instead of homodyne --method classical --verbose
hgc --verbose                    # Instead of homodyne-gpu --method classical --verbose

# Issue: Forgot available options
# Solution: Use help system
homodyne_help                    # Shows all methods, config files, and flags
```

## Code Quality and Maintenance (v0.6.5+)

The homodyne CLI is built with high code quality standards:

### Code Quality Status

| Tool | Status | Issues | Notes | |------|--------|---------|-------| | **Black** | ✅
100% | 0 | 88-character line length | | **isort** | ✅ 100% | 0 | Import sorting and
optimization | | **flake8** | ⚠️ ~400 | E501, F401 | Mostly line length and data scripts
| | **mypy** | ⚠️ ~285 | Various | Missing library stubs, annotations |

### Recent Code Quality Improvements

**Major Cleanup (v0.6.5):**

- **Removed 308 lines** of unused fallback implementations from `kernels.py`
- **Fixed critical issues**: Comparison operators (`== False` → `is False`)
- **Enhanced imports**: Resolved redefinition warnings and optimized import patterns
- **Added missing functions**: `_solve_least_squares_batch_fallback`,
  `_compute_chi_squared_batch_fallback`

**Enhanced Algorithms:**

- **Gurobi optimization**: Complete rewrite with iterative trust region SQP approach
- **Performance optimization**: 3-5x speedup with JIT compilation and optimized kernels
- **Numerical stability**: Parameter-scaled finite differences and improved convergence

### Development Standards

**Formatting and Style:**

```bash
# All code formatted with Black (88-character line length)
black homodyne --line-length 88

# Import sorting with isort  
isort homodyne --profile black

# Linting with flake8
flake8 homodyne --max-line-length 88

# Type checking with mypy
mypy homodyne --ignore-missing-imports
```

**Testing and Validation:**

- **Comprehensive test suite**: Unit, integration, and performance tests
- **Performance regression detection**: Automated benchmarking
- **95%+ test coverage**: Maintained across all critical components
- **Continuous integration**: Code quality checks on all changes

## Troubleshooting

### Common Issues and Solutions

**Gurobi Optimization Issues:**

```bash
# Issue: Gurobi license not found
# Solution: Install and activate Gurobi license
export GUROBI_HOME=/path/to/gurobi
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"

# Issue: Gurobi optimization not converging  
# Solution: Adjust trust region parameters in config
{
    "optimization_config": {
        "classical_optimization": {
            "method_options": {
                "Gurobi": {
                    "max_iterations": 100,        # Increase iterations
                    "trust_region_initial": 0.05,  # Smaller initial radius
                    "tolerance": 1e-8             # Stricter tolerance
                }
            }
        }
    }
}
```

**Shell Completion Issues:**

```bash
# Issue: Tab completion not working after installation
# Solution: Source the shell configuration or use shortcuts
source ~/.bashrc  # For bash
source ~/.zshrc   # For zsh
# OR use shortcuts that always work:
hc                # homodyne --method classical
hm                # homodyne --method mcmc

# Issue: argcomplete not found or compdef errors
# Solution: Install completion dependencies and use fallback
pip install homodyne-analysis[completion]
# Shortcuts work even when tab completion fails:
hr --verbose      # homodyne --method robust --verbose

# Issue: Forgot command options
# Solution: Use built-in help system
homodyne_help     # Shows all methods, config files, and flags
```

**Performance Issues:**

```bash
# Issue: Slow classical optimization
# Solution: Enable angle filtering and use Gurobi
homodyne --method classical --verbose  # Check which method is being used

# Issue: High memory usage
# Solution: Use static isotropic mode for large datasets
homodyne --static-isotropic --method classical
```

## Future Enhancement Possibilities

The CLI could be extended with additional commands:

- `homodyne-validate` - Validate data format
- `homodyne-benchmark` - Run performance benchmarks
- `homodyne-plot` - Generate plots from existing results
- `homodyne-convert` - Convert between data formats

However, these would require additional development and are not currently implemented.
