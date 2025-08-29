"""
Configuration Creator for Homodyne Analysis
==========================================

Interactive configuration file generator for XPCS homodyne analysis workflows.
Creates customized JSON configuration files from specialized templates,
enabling quick setup of analysis parameters for different experimental scenarios.

Key Features:
- Three-mode template system (static_isotropic, static_anisotropic, laminar_flow)
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
- Data loading paths and file specifications
- Optimization method settings and hyperparameters
- Analysis mode selection with automatic behavior
- Output formatting and result organization
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Import shell completion functionality
try:
    from homodyne.cli_completion import setup_shell_completion

    COMPLETION_AVAILABLE = True
except ImportError:
    COMPLETION_AVAILABLE = False

    def setup_shell_completion(parser: "argparse.ArgumentParser") -> None:
        """Fallback when completion is not available."""
        pass


def create_config_from_template(
    output_file="my_config.json",
    sample_name=None,
    experiment_name=None,
    author=None,
    mode="laminar_flow",
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

    # Validate and select appropriate template
    valid_modes = {
        "static_isotropic": "config_static_isotropic.json",
        "static_anisotropic": "config_static_anisotropic.json",
        "laminar_flow": "config_laminar_flow.json",
        "template": "config_template.json",  # Fallback to master template
    }

    if mode not in valid_modes:
        raise ValueError(
            f"Invalid mode '{mode}'. Valid modes: {list(valid_modes.keys())[:-1]}"
        )

    # Get template path (now that we're inside the homodyne package)
    template_dir = Path(__file__).parent
    template_file = template_dir / valid_modes[mode]

    # Fallback to master template if mode-specific template doesn't exist
    if not template_file.exists() and mode != "template":
        print(f"Warning: Mode-specific template not found: {template_file}")
        print("Falling back to master template...")
        template_file = template_dir / "config_template.json"

    if not template_file.exists():
        raise FileNotFoundError(f"Template not found: {template_file}")

    # Load template
    with open(template_file, encoding="utf-8") as f:
        config = json.load(f)

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

    # Print mode-specific information
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

    if mode in mode_info:
        info = mode_info[mode]
        print(f"  • {info['description']}")
        print(f"  • {info['parameters']}")
        print(f"  • {info['features']}")

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

    print("\nDocumentation: CONFIGURATION_MODES.md")
    print(f"Templates available: {', '.join(list(valid_modes.keys())[:-1])}")


def main():
    """Command-line interface for config creation."""
    # Check Python version requirement
    if sys.version_info < (3, 12):
        print("Error: This package requires Python 3.12 or higher.", file=sys.stderr)
        version_str = f"{sys.version_info[0]}.{sys.version_info[1]}"
        if len(sys.version_info) > 2:
            version_str += f".{sys.version_info[2]}"
        print(f"You are using Python {version_str}", file=sys.stderr)
        print("Please upgrade your Python version.", file=sys.stderr)
        sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description="Create homodyne analysis configuration from mode-specific templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Analysis Modes:
  static_isotropic   - Fastest: Single dummy angle, no angle filtering, 3 parameters
  static_anisotropic - Static with angle filtering optimization, 3 parameters
  laminar_flow       - Full flow analysis with 7 parameters (default)

Examples:
  # Create laminar flow configuration (default)
  homodyne-config --output my_flow_config.json

  # Create isotropic static configuration
  homodyne-config --mode static_isotropic --sample protein_01

  # Create anisotropic static configuration with metadata
  homodyne-config --mode static_anisotropic --sample collagen \
                          --author "Your Name" --experiment "Collagen static analysis"

  # Create flow analysis configuration
  homodyne-config --mode laminar_flow --sample microgel \
                          --experiment "Microgel dynamics under shear"
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
        "--output",
        "-o",
        default="my_config.json",
        help="Output configuration file name (default: my_config.json)",
    )

    parser.add_argument("--sample", "-s", help="Sample name (used in data paths)")

    parser.add_argument("--experiment", "-e", help="Experiment description")

    parser.add_argument("--author", "-a", help="Author name")

    # Setup shell completion if available
    if COMPLETION_AVAILABLE:
        setup_shell_completion(parser)

    args = parser.parse_args()

    try:
        create_config_from_template(
            output_file=args.output,
            sample_name=args.sample,
            experiment_name=args.experiment,
            author=args.author,
            mode=args.mode,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
