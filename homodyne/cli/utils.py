"""
CLI Utilities and Helper Functions
==================================

Shared utility functions for the homodyne CLI interface including logging setup,
configuration validation, and common helper functions.

This module contains utilities that are used across multiple CLI modules to avoid
duplication and maintain consistent behavior.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Module-level logger
logger = logging.getLogger(__name__)


class MockResult:
    """Mock result class for robust optimization compatibility."""

    def __init__(
        self,
        method_results=None,
        best_method=None,
        x=None,
        fun=None,
        success=None,
    ):
        self.method_results = method_results or {}
        self.best_method = best_method
        self.x = x
        self.fun = fun
        self.success = success


def print_method_documentation():
    """
    Print the method flags documentation.

    This function extracts and displays the comprehensive documentation
    for all method flags (--classical, --robust, --all).
    """
    from . import run_homodyne

    doc = run_homodyne.__doc__
    if not doc:
        print("No documentation available")
        return

    lines = doc.split("\n")
    in_method_docs = False

    for line in lines:
        if "Method Flags Documentation" in line:
            in_method_docs = True
        elif line.startswith('"""'):
            break
        if in_method_docs:
            print(line)


def setup_logging(verbose: bool, quiet: bool, output_dir: Path) -> None:
    """
    Configure comprehensive logging for the analysis session.

    Sets up both console and file logging with appropriate formatting.
    Debug level provides detailed execution information for troubleshooting.

    Parameters
    ----------
    verbose : bool
        Enable DEBUG level logging for detailed output
    quiet : bool
        Disable console logging (file logging remains enabled)
    output_dir : Path
        Directory where log file will be created
    """
    # Ensure output directory exists for log file
    os.makedirs(output_dir, exist_ok=True)

    # Set logging level based on verbosity preference
    level = logging.DEBUG if verbose else logging.INFO

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Add console handler only if not in quiet mode
    if not quiet:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Add file handler that writes to output_dir/run.log
    log_file_path = output_dir / "run.log"
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def print_banner(args: argparse.Namespace) -> None:
    """
    Display analysis configuration and session information.

    Provides a clear overview of the selected analysis parameters,
    methods, and output settings before starting the computation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing analysis configuration
    """
    print("=" * 60)
    print("            HOMODYNE ANALYSIS RUNNER")
    print("=" * 60)
    print()
    print(f"Method:           {args.method}")
    print(f"Config file:      {args.config}")
    print(f"Output directory: {args.output_dir}")
    if args.quiet:
        print(
            f"Logging:          File only ({'DEBUG' if args.verbose else 'INFO'} level)"
        )
    else:
        print(
            f"Verbose logging:  {
                'Enabled (DEBUG)' if args.verbose else 'Disabled (INFO)'
            }"
        )

    # Show analysis mode
    if args.static_isotropic:
        print("Analysis mode:    Static isotropic (3 parameters, no angle selection)")
    elif args.static_anisotropic:
        print(
            "Analysis mode:    Static anisotropic (3 parameters, with angle selection)"
        )
    elif args.laminar_flow:
        print("Analysis mode:    Laminar flow (7 parameters)")
    else:
        print("Analysis mode:    From configuration file")

    print()
    print("Starting analysis...")
    print("-" * 60)


def validate_and_load_config(args: argparse.Namespace) -> Path:
    """
    Validate configuration file existence and accessibility.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments with config path

    Returns
    -------
    Path
        Validated configuration file path
    """
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"❌ Configuration file not found: {config_path.absolute()}")
        logger.error(
            "Please check the file path and ensure the configuration file exists."
        )
        sys.exit(1)

    if not config_path.is_file():
        logger.error(f"❌ Configuration path is not a file: {config_path.absolute()}")
        sys.exit(1)

    logger.info(f"✓ Configuration file found: {config_path.absolute()}")
    return config_path


def create_config_override(args: argparse.Namespace) -> dict[str, Any] | None:
    """
    Create configuration overrides based on command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments

    Returns
    -------
    dict[str, Any] | None
        Configuration override dictionary or None
    """
    config_override: dict[str, Any] | None = None

    # Handle analysis mode overrides
    if args.static_isotropic:
        config_override = {
            "analysis_settings": {
                "static_mode": True,
                "static_submode": "isotropic",
            }
        }
        logger.info(
            "Using command-line override: static isotropic mode (3 parameters, no angle selection)"
        )
    elif args.static_anisotropic:
        config_override = {
            "analysis_settings": {
                "static_mode": True,
                "static_submode": "anisotropic",
            }
        }
        logger.info(
            "Using command-line override: static anisotropic mode (3 parameters, with angle selection)"
        )
    elif args.laminar_flow:
        config_override = {"analysis_settings": {"static_mode": False}}
        logger.info("Using command-line override: laminar flow mode (7 parameters)")

    # Handle experimental data plotting override
    if hasattr(args, "plot_experimental_data") and args.plot_experimental_data:
        if config_override is None:
            config_override = {}
        if "workflow_integration" not in config_override:
            config_override["workflow_integration"] = {}
        if "analysis_workflow" not in config_override["workflow_integration"]:
            config_override["workflow_integration"]["analysis_workflow"] = {}
        config_override["workflow_integration"]["analysis_workflow"][
            "plot_experimental_data_on_load"
        ] = True
        logger.info(
            "Using command-line override: plot experimental data on load enabled"
        )

    return config_override


def validate_advanced_optimization_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """
    Validate advanced optimization arguments for consistency.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    parser : argparse.ArgumentParser
        Parser instance for error reporting
    """
    # Validate advanced optimization arguments
    if (
        hasattr(args, "parameter_sweep")
        and args.parameter_sweep
        and not args.distributed
    ):
        parser.error("--parameter-sweep requires --distributed to be enabled")

    if (
        hasattr(args, "parameter_ranges")
        and args.parameter_ranges
        and not args.parameter_sweep
    ):
        parser.error("--parameter-ranges can only be used with --parameter-sweep")

    if (
        hasattr(args, "benchmark")
        and args.benchmark
        and (args.distributed or args.ml_accelerated)
    ):
        parser.error(
            "--benchmark cannot be used with --distributed or --ml-accelerated (benchmarks compare these methods)"
        )


def validate_scaling_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """
    Validate scaling and plotting arguments for consistency.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    parser : argparse.ArgumentParser
        Parser instance for error reporting
    """
    # Check for consistent scaling parameters
    if (args.contrast != 1.0 or args.offset != 0.0) and not args.plot_simulated_data:
        parser.error(
            "--contrast and --offset can only be used with --plot-simulated-data"
        )

    # Check for consistent phi angles parameter
    if args.phi_angles is not None:
        if not args.plot_simulated_data:
            parser.error("--phi-angles can only be used with --plot-simulated-data")
