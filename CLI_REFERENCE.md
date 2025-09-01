# Homodyne CLI - Actual Implementation

**Latest Version** | **Python 3.12+ Required** | **JAX Backend GPU Acceleration ✅** | **Unified Shell Completion ✅** | **Smart GPU Optimization ✅** | **Code Quality: Black ✅ Ruff ✅**

*Updated: 2024-08-31 - Reflects unified completion system and streamlined CLI tools*

## Overview

The Homodyne project provides comprehensive command-line tools for analyzing X-ray Photon
Correlation Spectroscopy (XPCS) data:

### Core Analysis Commands
1. **`homodyne`** - Main analysis command with enhanced Gurobi trust region optimization
2. **`homodyne-gpu`** - JAX backend GPU-accelerated analysis with smart CUDA detection
3. **`homodyne-config`** - Configuration file generator

### Setup and Management Commands  
4. **`homodyne-post-install`** - Unified setup for shell completion and GPU acceleration
5. **`homodyne-cleanup`** - Environment cleanup utility

### Advanced Tools (with --advanced flag)
6. **`homodyne-gpu-optimize`** - GPU optimization and benchmarking
7. **`homodyne-validate`** - Comprehensive system validation

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

### 2. `homodyne-gpu` - JAX Backend GPU-Accelerated Analysis

**GPU-accelerated homodyne analysis using JAX backend with PyTensor environment variable auto-configuration**. Automatically configures JAX for GPU operations while PyTensor runs on CPU to avoid C compilation issues. System CUDA 12.6+ and cuDNN 9.12+ support for optimal performance on Linux systems.

#### Usage

```bash
homodyne-gpu [OPTIONS]
```

#### Options

All options are identical to `homodyne` command. See Section 1 above for complete option list.

#### System Requirements

- **Linux OS** (GPU acceleration only)
- **System CUDA 12.6+** installed at `/usr/local/cuda`
- **cuDNN 9.12+** installed in system libraries (`/usr/lib/x86_64-linux-gnu`)
- **JAX with local CUDA**: `pip install homodyne-analysis[jax]` (auto-installs `jax[cuda12-local]`)
- **NVIDIA GPU** with driver version 560.28+
- **PyTensor environment variables**: Auto-configured during installation

#### JAX Backend Integration

- **JAX handles GPU operations**: All GPU-accelerated computations use JAX backend
- **PyTensor CPU mode**: Configured with `PYTENSOR_FLAGS="device=cpu,cxx="` to avoid C compilation issues
- **No C compilation problems**: PyTensor C++ compiler disabled, preventing linking errors
- **Automatic environment setup**: PyTensor environment variables configured during `pip install`
- **Virtual environment support**: Automatic activation scripts for conda/mamba environments

#### Examples

```bash
# GPU-accelerated MCMC analysis with JAX backend (recommended)
homodyne-gpu --method mcmc --config config.json

# GPU-accelerated all methods comparison with JAX backend
homodyne-gpu --method all --verbose

# JAX backend GPU analysis with automatic PyTensor configuration
homodyne-gpu --config analysis_config.json --method mcmc

# Check GPU and PyTensor status (conda/mamba environments)
homodyne_gpu_status

# Note: Classical/robust methods show efficiency warning
homodyne-gpu --method classical  # Recommends CPU-only homodyne command
```

#### Performance Benefits with System CUDA

- **5-10x speedup** for MCMC sampling with JAX GPU acceleration
- **Reliable performance**: Consistent system library integration
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

### 4. `homodyne-post-install` - Unified Setup System

Sets up shell completion, GPU acceleration, and advanced features in a unified, streamlined process.

#### Usage

```bash
homodyne-post-install [OPTIONS]
```

#### Options

```bash
  -h, --help                    Show help message and exit
  -i, --interactive             Interactive setup - choose what to install
  --shell {bash,zsh,fish}       Specify shell type for completion
  --gpu                         Install GPU acceleration (Linux only)
  --advanced                    Install advanced features (GPU optimization, validation)
  -f, --force                   Force setup even outside virtual environment
```

#### Examples

```bash
# Interactive setup (recommended)
homodyne-post-install --interactive

# Install shell completion only
homodyne-post-install --shell zsh

# Install GPU acceleration
homodyne-post-install --gpu

# Install everything
homodyne-post-install --shell zsh --gpu --advanced

# Makefile integration
make setup-all
```

---

### 5. `homodyne-cleanup` - Environment Cleanup Utility

Removes homodyne-related files from conda/virtual environments. Uses the unified cleanup system.

#### Usage

```bash
homodyne-cleanup [OPTIONS]
```

#### Options

```bash
  -h, --help         Show help message and exit
  -i, --interactive  Interactive cleanup - choose what to remove
  -n, --dry-run      Show what would be removed without removing
```

#### What it removes

- Shell completion files and activation scripts
- GPU acceleration setup files
- Advanced CLI tools (homodyne-gpu-optimize, homodyne-validate)
- Empty directories created during setup

#### Complete Uninstallation Process

```bash
# Step 1: Clean up environment scripts (while package is still installed)
homodyne-cleanup

# Step 2: Uninstall the package
pip uninstall homodyne-analysis
```

#### Example Output

```bash
$ homodyne-cleanup
============================================================
🧹 Homodyne Script Cleanup
============================================================
🧹 Cleaning up Homodyne scripts in: /home/user/miniforge3/envs/myenv

✓ Removed: /home/user/miniforge3/envs/myenv/etc/conda/activate.d/homodyne-gpu-activate.sh
✓ Removed: /home/user/miniforge3/envs/myenv/etc/conda/deactivate.d/homodyne-gpu-deactivate.sh
✓ Removed: /home/user/miniforge3/envs/myenv/etc/homodyne/gpu_activation.sh
✓ Removed: /home/user/miniforge3/envs/myenv/etc/homodyne/homodyne_completion_bypass.zsh
✓ Removed: /home/user/miniforge3/envs/myenv/etc/homodyne/homodyne_config.sh
✓ Removed empty directory: /home/user/miniforge3/envs/myenv/etc/homodyne

✅ Successfully cleaned up 6 files/directories
🔄 Restart your shell or reactivate the conda environment to complete cleanup

✅ Cleanup completed successfully
```

#### Alternative Usage

Can also be run as a Python module:

```bash
python -m homodyne.uninstall_scripts
```

## homodyne-gpu-optimize

Advanced GPU optimization tool for hardware-specific performance tuning.

```bash
homodyne-gpu-optimize [OPTIONS]
```

### Options

```bash
--benchmark         Run GPU benchmarks to determine optimal settings
--apply            Apply the recommended optimization settings
--report           Generate detailed hardware and performance report
--force            Force hardware re-detection and reconfiguration
--profile NAME     Use specific optimization profile (conservative/aggressive)
--memory-fraction  Set GPU memory allocation fraction (0.1-0.95)
--batch-size       Set optimal batch size for computations
--help, -h         Show help message
```

### Examples

```bash
# Quick optimization with auto-apply
homodyne-gpu-optimize --benchmark --apply

# Generate detailed system report
homodyne-gpu-optimize --report

# Use conservative optimization profile
homodyne-gpu-optimize --profile conservative --apply

# Force hardware re-detection
homodyne-gpu-optimize --force --benchmark
```

### Features

- **Hardware Detection**: Automatic GPU memory and CUDA capability detection
- **Performance Benchmarking**: Matrix multiplication benchmarks across different sizes
- **XLA Optimization**: Hardware-specific XLA flags configuration
- **Memory Management**: Optimal GPU memory fraction calculation
- **Profile System**: Pre-configured optimization profiles

### Output Example

```
🚀 GPU OPTIMIZATION RESULTS
========================================
🖥️  Hardware: NVIDIA RTX 4090 (24GB)
⚡ Optimal Settings:
   Memory Fraction: 0.90
   Batch Size: 4000
   XLA Flags: Advanced Triton optimizations
📊 Benchmark: 2.3x faster than default
✅ Settings applied successfully
```

## homodyne-validate

Comprehensive system validation tool for testing homodyne installation and configuration.

```bash
homodyne-validate [OPTIONS]
```

### Options

```bash
--verbose, -v      Show detailed validation output
--test TYPE        Run specific test category (environment/installation/completion/gpu/integration)
--json             Output results in JSON format for automation
--quick            Run only essential validation tests
--fix              Attempt to fix common issues automatically
--help, -h         Show help message
```

### Examples

```bash
# Full system validation
homodyne-validate

# Verbose output with details
homodyne-validate --verbose

# Test specific components
homodyne-validate --test gpu
homodyne-validate --test completion

# JSON output for CI/CD
homodyne-validate --json > validation_report.json

# Quick validation (essential tests only)
homodyne-validate --quick
```

### Test Categories

1. **Environment Detection**
   - Platform identification (Linux/macOS/Windows)
   - Python version compatibility
   - Virtual environment detection
   - Shell type identification

2. **Homodyne Installation**
   - Command availability verification
   - Help output validation
   - Core module imports
   - Dependencies check

3. **Shell Completion**
   - Completion file presence
   - Activation script functionality
   - Alias availability testing
   - Cross-shell compatibility

4. **GPU Setup**
   - GPU hardware detection
   - JAX device availability
   - CUDA installation verification
   - Driver compatibility check

5. **Integration Testing**
   - Component interaction validation
   - Cross-module imports
   - Script execution verification
   - End-to-end workflow testing

### Sample Output

```
🔍 HOMODYNE SYSTEM VALIDATION REPORT
================================================================================
📊 Summary: 5/5 tests passed ✅
🎉 All systems operational!

🖥️  Environment: Linux, Python 3.12.0, Shell: zsh, Conda: xpcs

📋 Test Results:
✅ PASS Environment Detection (0.003s)
✅ PASS Homodyne Installation (0.152s) - Found 5/5 commands
✅ PASS Shell Completion (0.089s) - 2 completion files, aliases working
✅ PASS GPU Setup (0.234s) - 1 GPU with JAX support
✅ PASS Integration (0.067s) - All module imports successful

💡 Recommendations:
🚀 Your homodyne installation is ready!
📖 Check documentation for usage examples
```

### JSON Output Format

```json
{
  "summary": {
    "total_tests": 5,
    "passed": 5,
    "failed": 0,
    "status": "success"
  },
  "environment": {
    "platform": "Linux",
    "python_version": "3.12.0",
    "shell": "zsh",
    "conda_env": "xpcs"
  },
  "test_results": [
    {
      "category": "Environment Detection",
      "status": "pass",
      "message": "Detected: Linux, Python 3.12.0, Shell: zsh",
      "duration": 0.003
    }
  ]
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

The homodyne CLI provides optional shell completion for bash, zsh, and fish shells. The completion system is safe and doesn't interfere with system commands.

### Optional Installation

Shell completion can be installed optionally using the post-install script:

```bash
# Interactive setup (recommended)
homodyne-post-install --interactive

# Install completion automatically for current shell
homodyne-post-install

# Install for specific shell
homodyne-post-install --shell zsh

# Install GPU acceleration
homodyne-post-install --gpu

# Install advanced features (Phases 4-6)
homodyne-post-install --advanced

# Install everything
homodyne-post-install --shell zsh --gpu --advanced
```

**Safe Completion System**: The new completion system:
- **Compatible**: Works with bash, zsh, and fish shells
- **Safe**: Doesn't interfere with system commands or Oh My Zsh
- **Optional**: Only installs when explicitly requested
- **Clean**: Easy removal with homodyne-cleanup

### Shell Activation

After installation, restart your shell or reactivate your environment:

```bash
# For conda/mamba environments
conda deactivate && conda activate your_env_name

# For other virtual environments
deactivate && source venv/bin/activate  # or your activation command

# Or restart your terminal/shell session
```

### Removal

To remove shell completion:

```bash
# Interactive removal (choose what to remove)
homodyne-cleanup --interactive

# Remove all homodyne files
homodyne-cleanup

# Then uninstall the package if desired
pip uninstall homodyne-analysis
```

### Completion Features

The completion system provides:

#### 1. **Tab Completion**

- **Method completion**: Tab complete `--method` options (classical, mcmc, robust, all)  
- **File completion**: Tab complete `--config` with available .json files
- **Directory completion**: Tab complete `--output-dir` with available directories
- **Shell specific**: Optimized for bash, zsh, and fish shells
- **Safe**: Won't interfere with system commands or other completions

#### 2. **Command Shortcuts**

Convenient aliases for common operations:

```bash
# Main analysis shortcuts  
hm          # homodyne --method mcmc
hc          # homodyne --method classical  
hr          # homodyne --method robust
ha          # homodyne --method all

# Plotting shortcuts
hexp        # homodyne --plot-experimental-data
hsim        # homodyne --plot-simulated-data

# Configuration shortcuts
hconfig     # homodyne-config

# GPU shortcuts (Linux only, if GPU setup installed)
hgm         # homodyne-gpu --method mcmc
hgc         # homodyne-gpu --method classical
hgr         # homodyne-gpu --method robust
hga         # homodyne-gpu --method all
```

#### 3. **Help Functions**

```bash
homodyne_help         # Show all available shortcuts
homodyne_gpu_status   # Check GPU status (Linux only)
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

# Quick method selection - GPU (Linux only)
hgm --verbose                    # GPU-activated MCMC
hga --config gpu_config.json     # GPU-activated all methods
hgconfig gpu_config.json         # GPU-activated config analysis

# Configuration shortcuts
hc-iso                           # homodyne-config --mode static_isotropic
hc-aniso                         # homodyne-config --mode static_anisotropic
hc-flow                          # homodyne-config --mode laminar_flow
hconfig my_config.json           # homodyne --config my_config.json
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

1. **Reactivate environment**: `conda deactivate && conda activate your_env_name` (reloads completion scripts)
1. **Use command shortcuts**: `hc`, `hm`, `hr`, `ha`, `hgm`, `hga`, `hc-iso`, `hc-aniso`, `hc-flow` always work
1. **Clean reinstall**: `homodyne-cleanup && pip install --upgrade homodyne-analysis[completion]`
1. **Reload shell**: `source ~/.zshrc` or restart terminal  
1. **Use help reference**: `homodyne_help` shows all options
1. **Manual cleanup**: `homodyne-cleanup` if scripts are outdated

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
