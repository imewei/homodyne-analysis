"""
CLI Core Module
===============

Main command-line interface core functionality for homodyne analysis.

This module contains the main entry point, workflow orchestration, and
primary analysis execution logic for the homodyne CLI interface.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .optimization import run_all_methods
from .optimization import run_classical_optimization
from .optimization import run_robust_optimization
from .parser import create_argument_parser
from .simulation import plot_simulated_data
from .utils import create_config_override
from .utils import print_banner
from .utils import setup_logging
from .utils import validate_advanced_optimization_args
from .utils import validate_and_load_config
from .utils import validate_scaling_args
from .visualization import generate_comparison_plots
from .visualization import save_individual_method_results
from .visualization import save_main_summary

# Module-level logger
logger = logging.getLogger(__name__)

# Import core analysis components with graceful error handling
try:
    from ..analysis.core import HomodyneAnalysisCore
    from ..optimization.classical import ClassicalOptimizer
    from ..optimization.robust import create_robust_optimizer

    CORE_ANALYSIS_AVAILABLE = True
except ImportError as e:
    import logging

    logger = logging.getLogger(__name__)
    logger.error(f"❌ Core analysis components not available: {e}")
    HomodyneAnalysisCore = None
    ClassicalOptimizer = None
    create_robust_optimizer = None
    CORE_ANALYSIS_AVAILABLE = False

# Import advanced optimization features with graceful degradation
try:
    from ..optimization.distributed import (
        integrate_with_classical_optimizer as integrate_distributed_classical,
        integrate_with_robust_optimizer as integrate_distributed_robust,
        get_available_backends,
    )

    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False
    integrate_distributed_classical = None
    integrate_distributed_robust = None
    get_available_backends = None

try:
    from ..optimization.ml_acceleration import (
        enhance_classical_optimizer_with_ml,
        enhance_robust_optimizer_with_ml,
        get_ml_backend_info,
    )

    ML_ACCELERATION_AVAILABLE = True
except ImportError:
    ML_ACCELERATION_AVAILABLE = False
    enhance_classical_optimizer_with_ml = None
    enhance_robust_optimizer_with_ml = None
    get_ml_backend_info = None

# Check for advanced optimization utilities
try:
    from ..optimization.utils import IntegrationHelper
    from ..optimization.utils import OptimizationBenchmark
    from ..optimization.utils import OptimizationConfig
    from ..optimization.utils import SystemResourceDetector
    from ..optimization.utils import quick_setup_distributed_optimization
    from ..optimization.utils import quick_setup_ml_acceleration
    from ..optimization.utils import setup_logging_for_optimization

    OPTIMIZATION_UTILS_AVAILABLE = True
except ImportError:
    OPTIMIZATION_UTILS_AVAILABLE = False
    # Create dummy functions to avoid errors
    OptimizationConfig = None
    SystemResourceDetector = None
    IntegrationHelper = None
    OptimizationBenchmark = None
    quick_setup_distributed_optimization = None
    quick_setup_ml_acceleration = None
    setup_logging_for_optimization = None

ADVANCED_OPTIMIZATION_AVAILABLE = (
    DISTRIBUTED_AVAILABLE or ML_ACCELERATION_AVAILABLE or OPTIMIZATION_UTILS_AVAILABLE
)


def initialize_analysis_engine(
    config_path: Path, config_override: dict[str, Any] | None
):
    """
    Initialize the analysis engine with configuration.

    Parameters
    ----------
    config_path : Path
        Path to configuration file
    config_override : Optional[Dict[str, Any]]
        Configuration overrides from command line

    Returns
    -------
    HomodyneAnalysisCore
        Initialized analysis engine
    """
    if not CORE_ANALYSIS_AVAILABLE:
        logger.error("❌ Core analysis components are not available")
        logger.error("Please ensure the homodyne package is properly installed")
        sys.exit(1)

    try:
        if config_override:
            analyzer = HomodyneAnalysisCore(str(config_path), config_override)
            logger.info("✓ Analysis engine initialized with configuration overrides")
        else:
            analyzer = HomodyneAnalysisCore(str(config_path))
            logger.info("✓ Analysis engine initialized")

        return analyzer

    except Exception as e:
        logger.error(f"❌ Failed to initialize analysis engine: {e}")
        logger.error(
            "Please check your configuration file and ensure all dependencies are installed"
        )
        sys.exit(1)


def load_and_validate_data(analyzer, args: argparse.Namespace):
    """
    Load and validate experimental data.

    Parameters
    ----------
    analyzer : HomodyneAnalysisCore
        Analysis engine
    args : argparse.Namespace
        Command-line arguments

    Returns
    -------
    tuple
        (initial_params, phi_angles, c2_exp)
    """
    try:
        logger.info("Loading experimental data...")

        # Load data using analyzer
        c2_exp, time_length, phi_angles, num_angles = analyzer.load_experimental_data()

        # Get initial parameters
        initial_params = np.array(
            analyzer.config.get("initial_parameters", {}).get("values", [])
        )

        if len(initial_params) == 0:
            logger.error("❌ No initial parameters found in configuration")
            sys.exit(1)

        logger.info("✓ Data loaded successfully:")
        logger.info(f"  Phi angles: {len(phi_angles)} angles")
        logger.info(f"  Data shape: {c2_exp.shape}")
        logger.info(f"  Initial parameters: {len(initial_params)} parameters")

        return initial_params, phi_angles, c2_exp

    except Exception as e:
        logger.error(f"❌ Failed to load experimental data: {e}")
        logger.error("Please check your data files and configuration")
        sys.exit(1)


def configure_optimization_enhancements(
    args: argparse.Namespace, initial_params, phi_angles, analyzer
):
    """
    Configure optimization enhancements based on command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    initial_params : np.ndarray
        Initial parameter values
    phi_angles : np.ndarray
        Array of phi angles
    analyzer : HomodyneAnalysisCore
        Analysis engine

    Returns
    -------
    dict
        Enhanced configuration dictionary
    """
    enhanced_config = {}

    # Configure distributed optimization
    if hasattr(args, "distributed") and args.distributed:
        if not ADVANCED_OPTIMIZATION_AVAILABLE:
            logger.warning("⚠️  Advanced optimization features not available")
            logger.warning("Continuing with standard optimization...")
        else:
            logger.info("Configuring distributed optimization...")
            try:
                # Create config dictionary for integration function
                import psutil
                num_processes = args.workers if hasattr(args, "workers") and args.workers else min(psutil.cpu_count() or 4, 8)
                distributed_config = {
                    "multiprocessing_config": {"num_processes": num_processes}
                }
                enhanced_config["distributed"] = distributed_config
                logger.info(
                    f"✓ Distributed optimization configured: {args.backend} backend with {num_processes} processes"
                )
            except Exception as e:
                logger.warning(f"⚠️  Failed to configure distributed optimization: {e}")

    # Configure ML acceleration
    if hasattr(args, "ml_accelerated") and args.ml_accelerated:
        if not ADVANCED_OPTIMIZATION_AVAILABLE:
            logger.warning("⚠️  ML acceleration features not available")
            logger.warning("Continuing with standard optimization...")
        else:
            logger.info("Configuring ML acceleration...")
            try:
                # Create config dictionary for integration function
                ml_config = {
                    "enable_transfer_learning": getattr(args, "enable_transfer_learning", False),
                    "data_storage_path": args.ml_data_path if hasattr(args, "ml_data_path") and args.ml_data_path else "./ml_optimization_data"
                }
                enhanced_config["ml_acceleration"] = ml_config
                logger.info("✓ ML acceleration configured")
            except Exception as e:
                logger.warning(f"⚠️  Failed to configure ML acceleration: {e}")

    return enhanced_config


def execute_optimization_methods(
    args: argparse.Namespace,
    analyzer,
    initial_params,
    phi_angles,
    c2_exp,
    enhanced_config,
):
    """
    Execute optimization methods based on command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    analyzer : HomodyneAnalysisCore
        Analysis engine
    initial_params : np.ndarray
        Initial parameter values
    phi_angles : np.ndarray
        Array of phi angles
    c2_exp : np.ndarray
        Experimental correlation data
    enhanced_config : dict
        Enhanced optimization configuration

    Returns
    -------
    dict
        Optimization results
    """
    logger.info(f"Executing optimization method: {args.method}")

    # Create enhanced optimizers if needed
    if enhanced_config:
        create_enhanced_optimizers(args, analyzer, enhanced_config)

    # Execute optimization based on method choice
    if args.method == "classical":
        result = run_classical_optimization(
            analyzer, initial_params, phi_angles, c2_exp, args.output_dir
        )
        return {"classical": result} if result else {}

    if args.method == "robust":
        result = run_robust_optimization(
            analyzer, initial_params, phi_angles, c2_exp, args.output_dir
        )
        return {"robust": result} if result else {}

    if args.method == "all":
        results = run_all_methods(
            analyzer, initial_params, phi_angles, c2_exp, args.output_dir
        )
        return results

    logger.error(f"❌ Unknown optimization method: {args.method}")
    return {}


def create_enhanced_optimizers(args: argparse.Namespace, analyzer, enhanced_config):
    """
    Create enhanced optimizers with distributed and ML capabilities.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    analyzer : HomodyneAnalysisCore
        Analysis engine
    enhanced_config : dict
        Enhanced optimization configuration
    """
    if not ADVANCED_OPTIMIZATION_AVAILABLE:
        return

    try:
        # Create enhanced classical optimizer
        if "distributed" in enhanced_config or "ml_acceleration" in enhanced_config:
            logger.info("Creating enhanced classical optimizer...")
            enhanced_classical = ClassicalOptimizer(analyzer, analyzer.config)

            if "distributed" in enhanced_config:
                enhanced_classical = integrate_distributed_classical(
                    enhanced_classical, enhanced_config["distributed"]
                )

            if "ml_acceleration" in enhanced_config:
                enhanced_classical = enhance_classical_optimizer_with_ml(
                    enhanced_classical, enhanced_config["ml_acceleration"]
                )

            analyzer._enhanced_classical_optimizer = enhanced_classical

        # Create enhanced robust optimizer
        if "distributed" in enhanced_config or "ml_acceleration" in enhanced_config:
            logger.info("Creating enhanced robust optimizer...")
            enhanced_robust = create_robust_optimizer(analyzer, analyzer.config)

            if "distributed" in enhanced_config:
                enhanced_robust = integrate_distributed_robust(
                    enhanced_robust, enhanced_config["distributed"]
                )

            if "ml_acceleration" in enhanced_config:
                enhanced_robust = enhance_robust_optimizer_with_ml(
                    enhanced_robust, enhanced_config["ml_acceleration"]
                )

            analyzer._enhanced_robust_optimizer = enhanced_robust

    except Exception as e:
        logger.warning(f"⚠️  Failed to create enhanced optimizers: {e}")


def process_and_save_results(
    results: dict[str, Any], args: argparse.Namespace, analyzer
):
    """
    Process and save optimization results.

    Parameters
    ----------
    results : Dict[str, Any]
        Optimization results
    args : argparse.Namespace
        Command-line arguments
    analyzer : HomodyneAnalysisCore
        Analysis engine
    """
    if not results:
        logger.error("❌ No results to save")
        return

    try:
        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data for visualization (returns tuple, not dict)
        c2_exp, time_length, phi_angles, num_angles = analyzer.load_experimental_data()

        # Save individual method results
        for method_name, result in results.items():
            if result:
                # Check if this is a result with multiple methods (robust or classical)
                if method_name == "robust" and "all_robust_results" in result:
                    # Save each robust method separately
                    for specific_method, method_result in result[
                        "all_robust_results"
                    ].items():
                        save_individual_method_results(
                            method_result,
                            "robust",
                            analyzer,
                            phi_angles,
                            c2_exp,
                            args.output_dir,
                        )
                elif method_name == "classical" and "all_classical_results" in result:
                    # Save each classical method separately
                    for specific_method, method_result in result[
                        "all_classical_results"
                    ].items():
                        save_individual_method_results(
                            method_result,
                            "classical",
                            analyzer,
                            phi_angles,
                            c2_exp,
                            args.output_dir,
                        )
                else:
                    # Save single method result
                    save_individual_method_results(
                        result,
                        method_name,
                        analyzer,
                        phi_angles,
                        c2_exp,
                        args.output_dir,
                    )

        # Generate comparison plots if both methods succeeded
        if len(results) > 1 and all(results.values()):
            generate_comparison_plots(
                analyzer,
                results.get("classical"),
                results.get("robust"),
                phi_angles,
                c2_exp,
                args.output_dir,
            )

        # Save main summary file
        save_main_summary(results, analyzer, args.output_dir)

        logger.info(f"✓ All results saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"❌ Failed to process and save results: {e}")


def run_analysis(args: argparse.Namespace) -> None:
    """
    Main analysis workflow execution.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    """
    logger.info("=" * 50)
    logger.info("Starting homodyne analysis workflow...")
    logger.info("=" * 50)

    # Validate and load configuration
    config_path = validate_and_load_config(args)
    config_override = create_config_override(args)

    # Initialize analysis engine
    analyzer = initialize_analysis_engine(config_path, config_override)

    # Load and validate data
    initial_params, phi_angles, c2_exp = load_and_validate_data(analyzer, args)

    # Configure optimization enhancements
    enhanced_config = configure_optimization_enhancements(
        args, initial_params, phi_angles, analyzer
    )

    # Execute optimization methods
    results = execute_optimization_methods(
        args, analyzer, initial_params, phi_angles, c2_exp, enhanced_config
    )

    # Process and save results
    process_and_save_results(results, args, analyzer)

    logger.info("=" * 50)
    logger.info("Analysis workflow completed successfully!")
    logger.info("=" * 50)


def main():
    """
    Command-line entry point for homodyne scattering analysis.

    Provides a complete interface for XPCS analysis under nonequilibrium
    conditions, supporting both static and laminar flow analysis modes
    with classical and robust optimization approaches.
    """
    # Create argument parser
    parser = create_argument_parser()
    args = parser.parse_args()

    # Handle special commands first
    if hasattr(args, "install_completion") and args.install_completion:
        try:
            from ..ui.completion.adapter import install_shell_completion

            return install_shell_completion(args.install_completion)
        except ImportError:
            print("Error: Shell completion requires additional packages.")
            print("Install with: pip install argcomplete")
            return 1

    if hasattr(args, "uninstall_completion") and args.uninstall_completion:
        try:
            from ..ui.completion.adapter import uninstall_shell_completion

            return uninstall_shell_completion(args.uninstall_completion)
        except ImportError:
            return 1

    # Check for conflicting logging options
    if args.verbose and args.quiet:
        parser.error("Cannot use --verbose and --quiet together")

    # Validate arguments
    validate_scaling_args(args, parser)
    validate_advanced_optimization_args(args, parser)

    # Setup logging and prepare output directory
    setup_logging(args.verbose, args.quiet, args.output_dir)

    # Print informative banner
    print_banner(args)

    # Log the configuration
    logger.info(f"Homodyne analysis starting with method: {args.method}")
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Log file: {args.output_dir / 'run.log'}")

    # Log advanced optimization features
    if hasattr(args, "distributed") and args.distributed:
        logger.info(f"Distributed optimization enabled with backend: {args.backend}")
        if hasattr(args, "workers") and args.workers:
            logger.info(f"Number of workers: {args.workers}")

    if hasattr(args, "ml_accelerated") and args.ml_accelerated:
        logger.info("ML-accelerated optimization enabled")
        logger.info(f"ML data path: {args.ml_data_path}")
        if hasattr(args, "enable_transfer_learning") and args.enable_transfer_learning:
            logger.info("Transfer learning enabled")

    # Log analysis mode selection
    if args.static_isotropic:
        logger.info(
            "Command-line mode: static isotropic (3 parameters, no angle selection)"
        )
    elif hasattr(args, "static_anisotropic") and args.static_anisotropic:
        logger.info(
            "Command-line mode: static anisotropic (3 parameters, with angle selection)"
        )
    elif args.laminar_flow:
        logger.info("Command-line mode: laminar flow (7 parameters)")
    else:
        logger.info("Analysis mode: from configuration file")

    # Handle special plotting modes
    if args.plot_simulated_data:
        try:
            plot_simulated_data(args)
            print()
            print("✓ Simulated data plotting completed successfully!")
            print(f"Results saved to: {args.output_dir}")
            sys.exit(0)
        except Exception as e:
            logger.error(f"❌ Simulated data plotting failed: {e}")
            sys.exit(1)

    if args.plot_experimental_data:
        try:
            # Load and plot experimental data only
            logger.info("==================================================")
            logger.info("Plotting experimental data only (no optimization)...")
            logger.info("==================================================")

            # Load configuration and create analyzer
            # Note: Don't use create_config_override here as it would trigger
            # automatic plotting during load_experimental_data, and we want
            # to control the save location explicitly
            config_path = validate_and_load_config(args)
            analyzer = initialize_analysis_engine(config_path, config_override=None)

            # Load experimental data
            logger.info("Loading experimental data...")
            c2_experimental, time_length, phi_angles, num_angles = (
                analyzer.load_experimental_data()
            )
            logger.info("✓ Data loaded successfully:")
            logger.info(f"  Phi angles: {num_angles} angles")
            logger.info(f"  Data shape: {c2_experimental.shape}")
            logger.info(f"  Time length: {time_length}")

            # Plot experimental data
            logger.info("Generating experimental data plots...")
            plot_path = args.output_dir / "exp_data" / "experimental_data_validation.png"
            analyzer._plot_experimental_data_validation(
                c2_experimental, phi_angles, save_path=str(plot_path)
            )

            print()
            print("✓ Experimental data plotting completed successfully!")
            print(f"Plot saved to: {plot_path}")
            sys.exit(0)
        except Exception as e:
            logger.error(f"❌ Experimental data plotting failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            sys.exit(1)

    # Run the analysis
    try:
        run_analysis(args)
        print()
        print("✓ Analysis completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        sys.exit(0)
    except SystemExit:
        # Re-raise SystemExit to preserve exit code
        raise
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        logger.error(
            "Please check your configuration and ensure all dependencies are installed"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
