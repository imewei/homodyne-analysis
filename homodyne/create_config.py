"""
Configuration Creator for Homodyne Analysis
==========================================

Interactive configuration file generator for XPCS homodyne analysis workflows.
Creates customized JSON configuration files from specialized templates,
enabling quick setup of analysis parameters for different experimental scenarios.

Key Features:
- Three-mode template system (static_isotropic, static_anisotropic, laminar_flow)
- Dataset-specific optimization templates (small, standard, large)
- Isolated MCMC backend architecture (completely separated PyMC CPU + NumPyro GPU)
- Mode-specific optimized configurations
- Customizable sample and experiment metadata
- Automatic path structure generation
- Validation and guidance for next steps
- Support for different analysis modes and optimization methods

Analysis Modes:
- static_isotropic: Fastest analysis, single dummy angle, no angle filtering
- static_anisotropic: Static analysis with angle filtering for optimization
- laminar_flow: Full 7-parameter analysis with flow effects

Usage Scenarios:
- New experiment setup with appropriate mode selection
- Batch analysis preparation with consistent naming
- Quick configuration generation for different analysis modes
- Template customization for specific experimental conditions

Generated Configuration Includes:
- Mode-specific physics parameters and optimizations
- Isolated MCMC backend configuration (completely separated CPU and GPU implementations)
- Data loading paths and file specifications
- Optimization method settings and hyperparameters
- Analysis mode selection with automatic behavior
- Output formatting and result organization

Isolated MCMC Backend Architecture:
- CPU Backend: Pure PyMC implementation (homodyne/optimization/mcmc_cpu_backend.py)
- GPU Backend: Pure NumPyro/JAX implementation (homodyne/optimization/mcmc_gpu_backend.py)
- Conflict Prevention: Eliminates PyTensor/JAX namespace conflicts through complete separation
- Command Usage: homodyne --method mcmc (CPU) or homodyne-gpu --method mcmc (GPU)
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

# Shell completion is handled by unified post-install system
COMPLETION_AVAILABLE = False


def apply_dataset_optimizations(config, mode, dataset_size):
    """
    Apply dataset-specific optimizations to configuration.

    Parameters
    ----------
    config : dict
        Base configuration to modify
    mode : str
        Analysis mode (static_isotropic, static_anisotropic, laminar_flow)
    dataset_size : str
        Dataset size category (small, large)

    Returns
    -------
    dict
        Optimized configuration
    """
    # Create a copy to avoid modifying the original
    config = config.copy()

    # Ensure required sections exist
    if "optimization_config" not in config:
        config["optimization_config"] = {}
    if "classical_optimization" not in config["optimization_config"]:
        config["optimization_config"]["classical_optimization"] = {}
    if "mcmc_sampling" not in config["optimization_config"]:
        config["optimization_config"]["mcmc_sampling"] = {}
    if "robust_optimization" not in config["optimization_config"]:
        config["optimization_config"]["robust_optimization"] = {}

    # Get optimization sections
    classical = config["optimization_config"]["classical_optimization"]
    mcmc = config["optimization_config"]["mcmc_sampling"]
    robust = config["optimization_config"]["robust_optimization"]

    if dataset_size == "small":
        # Small dataset optimizations (high precision, conservative settings)
        apply_small_dataset_optimizations(classical, mcmc, robust, mode)
    elif dataset_size == "large":
        # Large dataset optimizations (aggressive performance settings)
        apply_large_dataset_optimizations(classical, mcmc, robust, mode)

    # Add dataset size metadata
    if "metadata" not in config:
        config["metadata"] = {}
    config["metadata"]["dataset_optimization"] = dataset_size

    return config


def apply_small_dataset_optimizations(classical, mcmc, robust, mode):
    """Apply settings optimized for small datasets (<50K points)."""

    # Conservative classical settings for higher precision
    mode_settings = {
        "static_isotropic": {
            "nelder_mead_maxiter": 5000,  # 250% increase from standard 1500
            "trust_region_maxiter": 600,  # 300% increase from standard 150
            "tolerance": 1e-8,  # 100x tighter than large dataset
        },
        "static_anisotropic": {
            "nelder_mead_maxiter": 6000,  # 300% increase from standard 2000
            "trust_region_maxiter": 750,  # 275% increase from standard 200
            "tolerance": 1e-8,  # Maintained precision with angles
        },
        "laminar_flow": {
            "nelder_mead_maxiter": 10000,  # 150% increase from standard 4000
            "trust_region_maxiter": 1200,  # 200% increase from standard 600
            "tolerance": 1e-9,  # 10x tighter than standard 1e-8
        },
    }

    settings = mode_settings[mode]

    # Apply classical optimization settings
    if "method_options" not in classical:
        classical["method_options"] = {}
    if "Nelder-Mead" not in classical["method_options"]:
        classical["method_options"]["Nelder-Mead"] = {}
    if "Gurobi" not in classical["method_options"]:
        classical["method_options"]["Gurobi"] = {}

    classical["method_options"]["Nelder-Mead"].update(
        {
            "maxiter": settings["nelder_mead_maxiter"],
            "xatol": settings["tolerance"],
            "fatol": settings["tolerance"],
        }
    )

    classical["method_options"]["Gurobi"].update(
        {
            "max_iterations": settings["trust_region_maxiter"],
            "tolerance": settings["tolerance"],
        }
    )

    # Conservative MCMC settings
    mcmc_settings = {
        "static_isotropic": {
            "draws": 6000,  # 140% increase from standard 2500
            "tune": 2000,  # 185% increase from standard 700
            "target_accept": 0.93,  # Increased from 0.85
            "max_treedepth": 11,  # Increased from 8
        },
        "static_anisotropic": {
            "draws": 7000,  # 133% increase from standard 3000
            "tune": 2200,  # 175% increase from standard 800
            "target_accept": 0.91,  # Increased from 0.83
            "max_treedepth": 11,  # Increased from 9
        },
        "laminar_flow": {
            "draws": 12000,  # 200% increase from standard 4000
            "tune": 4000,  # 233% increase from standard 1200
            "target_accept": 0.88,  # Increased from 0.78
            "max_treedepth": 13,  # Increased from 10
        },
    }

    mcmc_mode_settings = mcmc_settings[mode]
    mcmc.update(
        {
            "draws": mcmc_mode_settings["draws"],
            "tune": mcmc_mode_settings["tune"],
            "target_accept": mcmc_mode_settings["target_accept"],
            "max_treedepth": mcmc_mode_settings["max_treedepth"],
            "thin": 1,  # No thinning for small datasets
        }
    )

    # Enhanced robust optimization for small datasets
    robust.update(
        {
            "enabled": True,
            "n_scenarios": {
                "static_isotropic": 30,  # 50% increase from standard 20
                "static_anisotropic": 32,  # 60% increase from standard 20
                "laminar_flow": 45,  # 125% increase from standard 20
            }[mode],
        }
    )


def apply_large_dataset_optimizations(classical, mcmc, robust, mode):
    """Apply settings optimized for large datasets (>1M points)."""

    # Aggressive settings for performance
    mode_settings = {
        "static_isotropic": {
            "nelder_mead_maxiter": 800,  # 60% reduction from standard 2000
            "trust_region_maxiter": 150,  # 75% reduction from standard 600
            "tolerance": 1e-5,  # Relaxed from standard 1e-6
        },
        "static_anisotropic": {
            "nelder_mead_maxiter": 1200,  # 50% reduction from standard 2400
            "trust_region_maxiter": 200,  # 67% reduction from standard 600
            "tolerance": 1e-5,  # Relaxed from standard 1e-6
        },
        "laminar_flow": {
            "nelder_mead_maxiter": 2400,  # 40% reduction from standard 4000
            "trust_region_maxiter": 400,  # 33% reduction from standard 600
            "tolerance": 1e-5,  # Maintained for 7-parameter complexity
        },
    }

    settings = mode_settings[mode]

    # Apply classical optimization settings
    if "method_options" not in classical:
        classical["method_options"] = {}
    if "Nelder-Mead" not in classical["method_options"]:
        classical["method_options"]["Nelder-Mead"] = {}
    if "Gurobi" not in classical["method_options"]:
        classical["method_options"]["Gurobi"] = {}

    classical["method_options"]["Nelder-Mead"].update(
        {
            "maxiter": settings["nelder_mead_maxiter"],
            "xatol": settings["tolerance"],
            "fatol": settings["tolerance"],
        }
    )

    classical["method_options"]["Gurobi"].update(
        {
            "max_iterations": settings["trust_region_maxiter"],
            "tolerance": settings["tolerance"],
        }
    )

    # Aggressive MCMC settings
    mcmc_settings = {
        "static_isotropic": {
            "draws": 1500,  # 40% reduction from standard 2500
            "tune": 400,  # 43% reduction from standard 700
            "target_accept": 0.75,  # Reduced from 0.85
            "thin": 3,  # Aggressive memory optimization
        },
        "static_anisotropic": {
            "draws": 2000,  # 33% reduction from standard 3000
            "tune": 500,  # 38% reduction from standard 800
            "target_accept": 0.78,  # Reduced from 0.83
            "thin": 3,  # Memory optimization
        },
        "laminar_flow": {
            "draws": 2800,  # 30% reduction from standard 4000
            "tune": 850,  # 29% reduction from standard 1200
            "target_accept": 0.70,  # Reduced from 0.78
            "thin": 2,  # Moderate optimization for complex space
        },
    }

    mcmc_mode_settings = mcmc_settings[mode]
    mcmc.update(
        {
            "draws": mcmc_mode_settings["draws"],
            "tune": mcmc_mode_settings["tune"],
            "target_accept": mcmc_mode_settings["target_accept"],
            "thin": mcmc_mode_settings["thin"],
        }
    )

    # Optimized robust settings
    robust.update(
        {
            "enabled": True,
            "n_scenarios": {
                "static_isotropic": 10,  # 50% reduction from standard 20
                "static_anisotropic": 12,  # 40% reduction from standard 20
                "laminar_flow": 15,  # 25% reduction from standard 20
            }[mode],
        }
    )


def create_config_from_template(
    output_file="my_config.json",
    sample_name=None,
    experiment_name=None,
    author=None,
    mode="laminar_flow",
    dataset_size="standard",
):
    """
    Generate customized configuration file from mode-specific template.

    Creates a complete configuration file by loading the appropriate mode template,
    applying user customizations, and generating appropriate file paths
    and metadata. Removes template-specific fields to create a clean
    production configuration.

    Customization Process:
    - Select appropriate template based on analysis mode
    - Apply user-specified metadata (author, experiment, sample)
    - Generate appropriate data paths based on sample name
    - Set creation/update timestamps
    - Remove template metadata for clean output
    - Validate structure and provide usage guidance

    Parameters
    ----------
    output_file : str
        Output configuration filename (default: "my_config.json")
    sample_name : str, optional
        Sample identifier for automatic path generation
    experiment_name : str, optional
        Descriptive experiment name for metadata
    author : str, optional
        Author name for configuration attribution
    mode : str
        Analysis mode: "static_isotropic", "static_anisotropic", or "laminar_flow"
    dataset_size : str
        Dataset size category: "small", "standard", or "large"

    Raises
    ------
    FileNotFoundError
        Template file not found in expected location
    JSONDecodeError
        Template file contains invalid JSON
    OSError
        File system errors during creation
    ValueError
        Invalid analysis mode specified
    """

    # Validate inputs
    valid_modes = ["static_isotropic", "static_anisotropic", "laminar_flow"]
    valid_sizes = ["small", "standard", "large"]

    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Valid modes: {valid_modes}")

    if dataset_size not in valid_sizes:
        raise ValueError(
            f"Invalid dataset_size '{dataset_size}'. Valid sizes: {valid_sizes}"
        )

    # Select appropriate template based on mode and dataset size
    def get_template_name(mode, size):
        """Generate template filename based on mode and dataset size."""
        if size == "standard":
            # Standard templates (existing ones)
            return {
                "static_isotropic": "config_default_static_isotropic.json",
                "static_anisotropic": "config_default_static_anisotropic.json",
                "laminar_flow": "config_default_laminar_flow.json",
            }[mode]
        else:
            # Dataset-specific templates
            return {
                "static_isotropic": f"config_{size}_dataset_static_isotropic.json",
                "static_anisotropic": f"config_{size}_dataset_static_anisotropic.json",
                "laminar_flow": f"config_{size}_dataset_laminar_flow.json",
            }[mode]

    template_filename = get_template_name(mode, dataset_size)

    # Get template path (now that we're inside the homodyne package)
    template_dir = Path(__file__).parent
    template_file = template_dir / template_filename

    # Fallback logic for missing templates
    if not template_file.exists():
        print(f"Warning: Specific template not found: {template_file}")

        if dataset_size != "standard":
            # Try falling back to standard template for the mode
            fallback_template = get_template_name(mode, "standard")
            fallback_file = template_dir / fallback_template

            if fallback_file.exists():
                print(f"Falling back to standard template: {fallback_template}")
                template_file = fallback_file
                # We'll apply dataset-specific optimizations later
            else:
                print("Falling back to master template...")
                template_file = template_dir / "config_default_template.json"
        else:
            print("Falling back to master template...")
            template_file = template_dir / "config_default_template.json"

    if not template_file.exists():
        raise FileNotFoundError(f"Template not found: {template_file}")

    # Load template
    with open(template_file, encoding="utf-8") as f:
        config = json.load(f)

    # Apply dataset-specific optimizations if using fallback template
    if dataset_size != "standard" and template_file.name != template_filename:
        config = apply_dataset_optimizations(config, mode, dataset_size)

    # Remove template-specific fields from final config
    if "_template_info" in config:
        del config["_template_info"]

    # Apply customizations
    current_date = datetime.now().strftime("%Y-%m-%d")

    if "metadata" in config:
        config["metadata"]["created_date"] = current_date
        config["metadata"]["updated_date"] = current_date

        # Update analysis mode in metadata
        config["metadata"]["analysis_mode"] = mode

        if experiment_name:
            config["metadata"]["description"] = experiment_name
        elif "description" in config["metadata"]:
            # Customize description based on mode
            mode_descriptions = {
                "static_isotropic": "Static Isotropic Scattering Analysis - No flow, no angular dependence",
                "static_anisotropic": "Static Anisotropic Scattering Analysis - No flow, with angular dependence",
                "laminar_flow": "Laminar Flow Scattering Analysis - Full flow and diffusion model",
            }
            if mode in mode_descriptions:
                config["metadata"]["description"] = mode_descriptions[mode]

        if author:
            config["metadata"]["authors"] = [author]

    # Apply sample-specific customizations
    if sample_name and "experimental_data" in config:
        config["experimental_data"]["data_folder_path"] = f"./data/{sample_name}/"
        if "cache_file_path" in config["experimental_data"]:
            config["experimental_data"]["cache_file_path"] = f"./data/{sample_name}/"

        # Update cache filename template based on mode
        cache_templates = {
            "static_isotropic": f"cached_c2_isotropic_{sample_name}_{{start_frame}}_{{end_frame}}.npz",
            "static_anisotropic": f"cached_c2_anisotropic_{sample_name}_{{start_frame}}_{{end_frame}}.npz",
            "laminar_flow": f"cached_c2_flow_{sample_name}_{{start_frame}}_{{end_frame}}.npz",
        }
        if mode in cache_templates:
            config["experimental_data"]["cache_filename_template"] = cache_templates[
                mode
            ]

    # Save configuration
    output_path = Path(output_file)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✓ Configuration created: {output_path.absolute()}")
    print(f"✓ Analysis mode: {mode}")
    print(f"✓ Dataset optimization: {dataset_size}")
    print(
        "✓ Isolated MCMC backends: PyMC (CPU) + NumPyro (GPU) with complete separation configured"
    )

    # Print mode and dataset-specific information
    mode_info = {
        "static_isotropic": {
            "description": "Fastest analysis with single dummy angle",
            "parameters": "3 active parameters (D0, alpha, D_offset)",
            "features": "No angle filtering, no phi_angles_file loading",
        },
        "static_anisotropic": {
            "description": "Static analysis with angle filtering optimization",
            "parameters": "3 active parameters (D0, alpha, D_offset)",
            "features": "Angle filtering enabled, phi_angles_file required",
        },
        "laminar_flow": {
            "description": "Full physics model with flow effects",
            "parameters": "7 active parameters (all parameters)",
            "features": "Full flow analysis, phi_angles_file required",
        },
    }

    dataset_info = {
        "small": {
            "range": "<50K data points",
            "focus": "Maximum precision & validation",
            "trade_off": "2-3x longer runtime for highest reliability",
            "features": "Enhanced validation, bootstrap methods, comprehensive diagnostics",
        },
        "standard": {
            "range": "50K-1M data points",
            "focus": "Balanced accuracy & efficiency",
            "trade_off": "Optimal for typical XPCS datasets",
            "features": "Standard validation, robust convergence",
        },
        "large": {
            "range": ">1M data points",
            "focus": "Computational efficiency",
            "trade_off": "35-65% faster with maintained accuracy",
            "features": "Aggressive optimization, memory efficiency, enhanced caching",
        },
    }

    if mode in mode_info:
        info = mode_info[mode]
        print(f"  • {info['description']}")
        print(f"  • {info['parameters']}")
        print(f"  • {info['features']}")

    if dataset_size in dataset_info:
        ds_info = dataset_info[dataset_size]
        print(f"\n📊 Dataset Optimization ({dataset_size}):")
        print(f"  • Target range: {ds_info['range']}")
        print(f"  • Optimization focus: {ds_info['focus']}")
        print(f"  • Trade-off: {ds_info['trade_off']}")
        print(f"  • Features: {ds_info['features']}")

    # Provide next steps
    print("\nNext steps:")
    print(f"1. Edit {output_path} and customize the parameters for your experiment")
    print("2. Replace placeholder values (YOUR_*) with actual values")
    print("3. Adjust initial_parameters.values based on your system")
    if mode == "static_isotropic":
        print(
            "4. Note: phi_angles_file will be automatically skipped in isotropic mode"
        )
        print(f"5. Run analysis with: homodyne --config {output_path}")
    elif mode == "static_anisotropic":
        print("4. Ensure phi_angles_file exists and contains your scattering angles")
        print(f"5. Run analysis with: homodyne --config {output_path}")
    else:  # laminar_flow
        print("4. Ensure phi_angles_file exists and contains your scattering angles")
        print("5. Verify initial parameter estimates for all 7 parameters")
        print(f"6. Run analysis with: homodyne --config {output_path}")

    # Provide isolated MCMC backend guidance
    print("\n📊 Isolated MCMC Backend Architecture:")
    print("  • CPU Backend (PyMC):    homodyne --config [config] --method mcmc")
    print("    └─ Pure PyMC implementation, no JAX dependencies")
    print("  • GPU Backend (NumPyro):  homodyne-gpu --config [config] --method mcmc")
    print("    └─ Pure NumPyro/JAX implementation, no PyMC dependencies")
    print(
        "  • Conflict Prevention:   Complete separation eliminates PyTensor/JAX conflicts"
    )
    print(
        "  • Environment Control:   HOMODYNE_GPU_INTENT=true/false for explicit backend selection"
    )

    print("\n⚡ Performance Recommendations:")
    if mode in ["static_isotropic", "static_anisotropic"]:
        print("  • 3-parameter analysis: Both isolated backends work well")
        print("  • CPU backend recommended for cross-platform compatibility")
        print(
            "  • Isolated architecture ensures reliable execution without dependency conflicts"
        )
    else:  # laminar_flow
        print(
            "  • 7-parameter analysis: GPU backend recommended for optimal performance"
        )
        print(
            "  • GPU acceleration provides significant speedup for complex flow models"
        )
        print("  • Isolated architecture eliminates GPU/CPU backend conflicts")

    print("\n📚 Documentation:")
    print(
        "  • Configuration guide: https://homodyne.readthedocs.io/en/latest/user-guide/configuration.html"
    )
    print(
        "  • Analysis modes: https://homodyne.readthedocs.io/en/latest/user-guide/analysis-modes.html"
    )
    print("  • Isolated MCMC backends: homodyne/runtime/README.md")
    print(
        "  • Runtime system: https://homodyne.readthedocs.io/en/latest/api/runtime.html"
    )
    print("  • GPU setup: homodyne-post-install --gpu --help")
    print(f"  • Templates available: {', '.join(valid_modes)}")


def main():
    """Command-line interface for config creation."""
    # Check Python version requirement
    import sys

    if sys.version_info < (3, 12):  # noqa: UP036
        print(
            "Error: Python 3.12 or higher is required for homodyne-analysis.",
            file=sys.stderr,
        )
        print(
            f"Current Python version: {sys.version_info[0]}.{sys.version_info[1]}",
            file=sys.stderr,
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Create homodyne analysis configuration from mode-specific templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Analysis Modes:
  static_isotropic   - Fastest: Single dummy angle, no angle filtering, 3 parameters
  static_anisotropic - Static with angle filtering optimization, 3 parameters
  laminar_flow       - Full flow analysis with 7 parameters (default)

Dataset Size Optimizations:
  small    - <50K points: Maximum precision, enhanced validation (2-3x slower)
  standard - 50K-1M points: Balanced accuracy & efficiency (default)
  large    - >1M points: Aggressive optimization, 35-65% faster

Isolated MCMC Backends (configured automatically in all modes):
  PyMC CPU Backend   - Isolated pure PyMC implementation (homodyne --method mcmc)
  NumPyro GPU Backend- Isolated pure NumPyro/JAX implementation (homodyne-gpu --method mcmc)
  Complete Separation- Eliminates PyTensor/JAX conflicts through architectural isolation

Examples:
  # Create standard laminar flow configuration (default)
  homodyne-config --output my_flow_config.json

  # Create small dataset configuration for high precision
  homodyne-config --mode static_isotropic --dataset-size small --sample protein_01

  # Create large dataset configuration for performance
  homodyne-config --mode laminar_flow --dataset-size large --sample microgel \
                          --experiment "High-throughput microgel analysis"

  # Create standard configuration with metadata
  homodyne-config --mode static_anisotropic --sample collagen \
                          --author "Your Name" --experiment "Collagen static analysis"

Isolated MCMC Backend Usage:
  # CPU backend (isolated PyMC implementation)
  homodyne --config my_config.json --method mcmc

  # GPU backend (isolated NumPyro/JAX implementation)
  homodyne-gpu --config my_config.json --method mcmc
  # Environment variable control
  HOMODYNE_GPU_INTENT=false homodyne --method mcmc  # Force CPU backend
  HOMODYNE_GPU_INTENT=true homodyne --method mcmc   # Force GPU backend
        """,
    )

    parser.add_argument(
        "--mode",
        "-m",
        choices=["static_isotropic", "static_anisotropic", "laminar_flow"],
        default="laminar_flow",
        help="Analysis mode (default: laminar_flow)",
    )

    parser.add_argument(
        "--dataset-size",
        "-d",
        choices=["small", "standard", "large"],
        default="standard",
        help="Dataset size optimization: small (<50K points), standard (50K-1M), large (>1M) (default: standard)",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="my_config.json",
        help="Output configuration file name (default: my_config.json)",
    )

    parser.add_argument("--sample", "-s", help="Sample name (used in data paths)")

    parser.add_argument("--experiment", "-e", help="Experiment description")

    parser.add_argument("--author", "-a", help="Author name")

    # Shell completion is handled by post-install system, not at runtime

    args = parser.parse_args()

    try:
        create_config_from_template(
            output_file=args.output,
            sample_name=args.sample,
            experiment_name=args.experiment,
            author=args.author,
            mode=args.mode,
            dataset_size=args.dataset_size,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
