# Homodyne CLI - Actual Implementation

## Overview
The Homodyne project currently provides two command-line tools for analyzing X-ray Photon Correlation Spectroscopy (XPCS) data:

1. **`homodyne`** - Main analysis command
2. **`homodyne-config`** - Configuration file generator

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
  {interactive}                 Special commands: 'interactive' starts interactive CLI mode
```

#### Methods
- **classical**: Traditional optimization (Nelder-Mead, Gurobi if available)
- **robust**: Robust optimization with uncertainty quantification (requires CVXPY)
- **mcmc**: Bayesian MCMC sampling using PyMC and NUTS
- **all**: Run all available methods

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

---

### 2. `homodyne-config` - Configuration Generator

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
2. **static_anisotropic** - Static with angle filtering optimization, 3 parameters  
3. **laminar_flow** - Full flow analysis with 7 parameters (default)

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

| Code | Meaning |
|------|---------|
| 0    | Success |
| 1    | General error or Python version mismatch |
| 2    | Configuration error |
| 3    | Data format/loading error |
| 4    | Optimization failed |
| 5    | Missing dependencies |

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

1. **No subcommands**: Unlike the idealized CLI structure, the current implementation doesn't have subcommands like `homodyne analyze`, `homodyne mcmc`, etc.

2. **Method selection**: Different optimization methods are selected via the `--method` flag, not separate commands.

3. **Data preprocessing**: No separate preprocessing command; preprocessing happens automatically during analysis.

4. **Validation**: No separate validation command; validation is integrated into the analysis pipeline.

5. **Python 3.12+ required**: Both commands check for Python 3.12 or higher.

6. **Optional dependencies**: 
   - MCMC requires PyMC installation
   - Robust optimization requires CVXPY
   - Some features require numba for performance
   - Shell completion and interactive mode require `argcomplete` and `cmd2`

## Shell Completion

The homodyne CLI supports tab completion for enhanced user experience across multiple shells.

### Installation

Install completion support with optional dependencies:
```bash
pip install homodyne-analysis[interactive]
```

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

After installation, restart your shell or source the configuration file:
```bash
# Bash/Zsh
source ~/.bashrc  # or ~/.zshrc

# Fish
source ~/.config/fish/config.fish
```

### Features

- **Method completion**: Tab complete `--method` options (classical, mcmc, robust, all)
- **File completion**: Tab complete `--config` with available .json files
- **Directory completion**: Tab complete `--output-dir` with available directories
- **Context-aware**: Completion adapts based on current command context
- **Cross-platform**: Works on Linux, macOS, and Windows

### Usage Examples

```bash
# Tab completion examples (press TAB at cursor position)
homodyne --method <TAB>          # Shows: classical, mcmc, robust, all
homodyne --config <TAB>          # Shows available .json files
homodyne --output-dir <TAB>      # Shows available directories
homodyne-config --mode <TAB>     # Shows: static_isotropic, static_anisotropic, laminar_flow
```

## Interactive Mode

The homodyne CLI provides a powerful interactive mode for streamlined analysis workflows.

### Starting Interactive Mode

```bash
homodyne interactive
```

### Interactive Commands

Once in interactive mode, you have access to specialized commands:

#### Core Analysis Commands

```bash
# Set analysis method
method classical        # Switch to classical optimization
method mcmc            # Switch to MCMC sampling
method robust          # Switch to robust optimization  
method all             # Use all methods
method                 # Show current method and available options

# Configure settings
config set my_config.json    # Load configuration file
config show                  # Display current config contents
config validate              # Validate configuration file
config                       # Show current config file

# Execute analysis
run                          # Run with current settings
run --method mcmc            # Run MCMC with current config  
run --verbose                # Run with debug logging
run --static-isotropic       # Run in static isotropic mode
```

#### Utility Commands

```bash
# File management
ls                          # List current directory contents
ls data/                    # List specific directory

# Session management
status                      # Show current session settings
help_guide                  # Comprehensive usage guide
quit                        # Exit interactive mode (or Ctrl+C)

# Configuration creation
create_config --mode laminar_flow --sample protein_01
create_config --mode static_isotropic --output my_config.json
```

### Interactive Features

- **Tab completion**: All commands and arguments support tab completion
- **Command history**: Use up/down arrows to navigate command history
- **Real-time validation**: Immediate feedback on configuration errors
- **Context awareness**: Commands adapt based on current session state
- **Integrated help**: Built-in help system with examples
- **File discovery**: Automatic detection of configuration files

### Example Interactive Session

```bash
$ homodyne interactive

╭─────────────────────────────────────────────────────────╮
│            Homodyne Analysis Interactive CLI            │
│                                                         │
│  Tab completion, command history, and real-time help   │
│  Type 'help' for commands or 'help <command>' for info │
│  Use Ctrl+C to exit or type 'quit'                     │
╰─────────────────────────────────────────────────────────╯

homodyne> ls
Config files:
  📄 homodyne_config.json
  📄 mcmc_config.json
  
homodyne> config set homodyne_config.json
✓ Config set to: homodyne_config.json

homodyne> method mcmc
✓ Method set to: mcmc

homodyne> status
╭─────────────────────────────────────────╮
│           Current Settings             │
├─────────────────────────────────────────┤
│ Method     : mcmc                      │
│ Config     : homodyne_config.json      │
│ Output Dir : ./homodyne_results        │
│ Verbose    : False                     │
╰─────────────────────────────────────────╯

homodyne> run --verbose
Executing: python -m homodyne.run_homodyne --method mcmc --config homodyne_config.json --output-dir ./homodyne_results --verbose
✓ Analysis completed successfully!

homodyne> quit
```

### Benefits of Interactive Mode

- **Faster iteration**: No need to type full commands repeatedly
- **Session persistence**: Settings maintained throughout session
- **Reduced errors**: Tab completion and validation prevent typos
- **Enhanced productivity**: Quick access to files and configurations
- **Learning friendly**: Built-in help and guidance

## Future Enhancement Possibilities

The CLI could be extended with additional commands:
- `homodyne-validate` - Validate data format
- `homodyne-benchmark` - Run performance benchmarks  
- `homodyne-plot` - Generate plots from existing results
- `homodyne-convert` - Convert between data formats

However, these would require additional development and are not currently implemented.