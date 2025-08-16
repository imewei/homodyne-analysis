#!/usr/bin/env python3
"""
Experimental Data Plotting Script for Homodyne XPCS Analysis
============================================================

This script creates comprehensive validation plots of experimental C2 correlation data
for X-ray Photon Correlation Spectroscopy (XPCS) analysis. It loads data according to
the configuration file and generates quality assessment plots.

Features:
- Load experimental data using homodyne package configuration
- Generate comprehensive validation plots including heatmaps, diagonal slices, and statistics
- Support for both HDF5 and NPZ data formats
- Automatic data quality assessment with validation metrics
- Configurable plotting parameters and output formats

Usage:
    python plot_experimental_data.py [--config CONFIG_FILE] [--output-dir OUTPUT_DIR] [--verbose]

Example:
    python plot_experimental_data.py --config my_config_simon.json --output-dir ./validation_plots --verbose

Author: Generated for Homodyne XPCS Analysis Package
Based on: He et al. PNAS 2024 - Transport coefficient approach
"""

import sys
import argparse
import logging
from pathlib import Path
import json


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def validate_config_file(config_path: Path) -> dict:
    """
    Validate and load the configuration file.

    Parameters
    ----------
    config_path : Path
        Path to the JSON configuration file

    Returns
    -------
    dict
        Loaded configuration dictionary

    Raises
    ------
    FileNotFoundError
        If configuration file doesn't exist
    json.JSONDecodeError
        If configuration file contains invalid JSON
    ValueError
        If required configuration sections are missing
    """
    logger = logging.getLogger(__name__)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path.absolute()}"
        )

    if not config_path.is_file():
        raise ValueError(f"Configuration path is not a file: {config_path.absolute()}")

    logger.info(f"Loading configuration from: {config_path.absolute()}")

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in configuration file: {e}") # type: ignore

    # Validate required sections
    required_sections = ["experimental_data", "analyzer_parameters"]
    missing_sections = [
        section for section in required_sections if section not in config
    ]
    if missing_sections:
        raise ValueError(f"Missing required configuration sections: {missing_sections}")

    logger.info("✓ Configuration file validated successfully")
    return config


def main():
    """Main function for experimental data plotting."""
    parser = argparse.ArgumentParser(
        description="Create experimental data validation plots using homodyne package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Use default config file
  %(prog)s --config my_config_simon.json     # Use specific config file
  %(prog)s --output-dir ./validation_plots   # Custom output directory
  %(prog)s --verbose                          # Enable debug logging
  %(prog)s --config my_config_simon.json --output-dir ./plots --verbose
        """,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default="my_config_simon.json",
        help="Path to configuration file (default: %(default)s)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./experimental_data_plots",
        help="Output directory for plots (default: %(default)s)",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose DEBUG logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("🔬 Homodyne XPCS Experimental Data Plotter")
    logger.info("=" * 50)

    try:
        # Validate configuration file
        config = validate_config_file(args.config)

        # Extract key parameters from config for plotting
        metadata = config.get("metadata", {})
        sample_description = metadata.get("sample_description", "Unknown Sample")
        experiment_name = metadata.get("experiment_name", "XPCS Experiment")

        # Get time step from configuration
        dt = config.get("analyzer_parameters", {}).get("temporal", {}).get("dt", 1.0)

        logger.info(f"Sample: {sample_description}")
        logger.info(f"Experiment: {experiment_name}")
        logger.info(f"Time step: {dt} seconds")
        logger.info(f"Output directory: {args.output_dir.absolute()}")

        # Import homodyne modules
        try:
            logger.info("Loading homodyne analysis modules...")
            from homodyne.analysis.core import HomodyneAnalysisCore
            from homodyne.plotting import plot_experimental_c2_data

            logger.info("✓ Homodyne modules loaded successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import homodyne modules: {e}")
            logger.error("Please ensure the homodyne package is properly installed")
            logger.error("Try: pip install numpy scipy matplotlib numba")
            sys.exit(1)

        # Initialize analysis core with configuration
        try:
            logger.info("Initializing homodyne analysis core...")
            analyzer = HomodyneAnalysisCore(config_file=str(args.config))
            logger.info("✓ Analysis core initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize analysis core: {e}")
            logger.error("Please check your configuration file and data paths")
            sys.exit(1)

        # Load experimental data
        try:
            logger.info("Loading experimental data...")
            logger.info("This may take a few moments depending on data size...")

            c2_experimental, time_length, phi_angles, num_angles = (
                analyzer.load_experimental_data()
            )

            logger.info(f"✓ Data loaded successfully:")
            logger.info(f"  • Data shape: {c2_experimental.shape}")
            logger.info(f"  • Number of angles: {num_angles}")
            logger.info(
                f"  • Angle range: {phi_angles.min():.1f}° to {phi_angles.max():.1f}°"
            )
            logger.info(f"  • Time frames: {time_length}")
            logger.info(f"  • Total measurement time: {time_length * dt:.1f} seconds")

        except Exception as e:
            logger.error(f"❌ Failed to load experimental data: {e}")
            logger.error("Please check your data file paths in the configuration")
            logger.error("Ensure data files exist and are accessible")
            sys.exit(1)

        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Output directory created: {args.output_dir.absolute()}")

        # Generate experimental data plots
        try:
            logger.info("Creating experimental data validation plots...")
            logger.info(
                "Generating heatmaps, diagonal slices, cross-sections, and statistics..."
            )

            success = plot_experimental_c2_data(
                c2_experimental=c2_experimental,
                phi_angles=phi_angles,
                outdir=args.output_dir,
                config=config,
                dt=dt,
                sample_description=f"{sample_description} - {experiment_name}",
            )

            if success:
                logger.info(
                    "✓ Experimental data validation plots created successfully!"
                )
                logger.info(
                    f"📊 Plots saved to: {args.output_dir / 'experimental_data_validation'}"
                )

                # Provide information about what was plotted
                logger.info("")
                logger.info("📋 Generated plots include:")
                logger.info("  • Full correlation function heatmaps g₂(t₁,t₂)")
                logger.info("  • Diagonal slices g₂(t,t) showing temporal decay")
                logger.info("  • Cross-sectional profiles at different time points")
                logger.info("  • Statistical summaries with data quality metrics")
                logger.info("")
                logger.info("🔍 Quality indicators to check:")
                logger.info("  • Mean values should be around 1.0 (expected for g₂)")
                logger.info("  • Diagonal values should be enhanced (> off-diagonal)")
                logger.info("  • Contrast should be > 0.001 (indicates dynamic signal)")
                logger.info("  • Structure should be consistent across angles")

            else:
                logger.error("❌ Failed to create experimental data plots")
                logger.error("Check the log messages above for specific error details")
                sys.exit(1)

        except Exception as e:
            logger.error(f"❌ Error during plot generation: {e}")
            logger.error("This may indicate issues with:")
            logger.error("  • Data format or structure")
            logger.error("  • Missing plotting dependencies (matplotlib)")
            logger.error("  • Insufficient memory for large datasets")
            sys.exit(1)

        # Summary and next steps
        logger.info("")
        logger.info("🎉 Experimental data plotting completed successfully!")
        logger.info("=" * 50)
        logger.info("📁 Results location:")
        logger.info(f"   {args.output_dir.absolute()}")
        logger.info("")
        logger.info("🚀 Next steps:")
        logger.info("1. Review the validation plots to assess data quality")
        logger.info("2. Check for any quality warnings in the statistics panels")
        logger.info("3. If data looks good, proceed with homodyne analysis:")
        logger.info(f"   python run_homodyne.py --config {args.config}")
        logger.info("4. For comprehensive analysis with validation plots:")
        logger.info(
            f"   python run_homodyne.py --config {args.config} --plot-experimental-data --method all --verbose"
        )

    except KeyboardInterrupt:
        logger.info("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"❌ Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
