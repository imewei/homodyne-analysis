"""
CLI Argument Parser Module
==========================

Command-line argument parser configuration for the homodyne CLI interface.

This module contains the argument parser setup with comprehensive command-line
options for analysis configuration, optimization methods, and advanced features.
"""

import argparse
from pathlib import Path

# Import completion setup if available
try:
    from .ui.completion.adapter import setup_shell_completion
    COMPLETION_AVAILABLE = True
except ImportError:
    COMPLETION_AVAILABLE = False
    def setup_shell_completion(parser: argparse.ArgumentParser) -> None:
        pass


def create_argument_parser():
    """
    Create and configure the argument parser for homodyne analysis.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Run homodyne scattering analysis for XPCS under nonequilibrium conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic Usage:
  %(prog)s                                    # Run with default classical method
  %(prog)s --method robust                    # Run only robust optimization methods
  %(prog)s --method all --verbose             # Run all methods with debug logging
  %(prog)s --config my_config.json            # Use custom config file
  %(prog)s --output-dir ./homodyne_results --verbose   # Custom output directory with verbose logging
  %(prog)s --quiet                            # Run with file logging only (no console output)

  Analysis Modes:
  %(prog)s --static-isotropic                 # Force static mode (zero shear, 3 parameters)
  %(prog)s --laminar-flow --method robust     # Force laminar flow mode with robust methods
  %(prog)s --static-isotropic --method robust # Run robust methods in static mode
  %(prog)s --static-isotropic --method all    # Run all methods in static mode

  Visualization:
  %(prog)s --plot-simulated-data                  # Plot with default scaling: fitted = 1.0 * theory + 0.0
  %(prog)s --plot-simulated-data --static-isotropic   # Plot simulated data in static mode
  %(prog)s --plot-simulated-data --contrast 1.5 --offset 0.1  # Plot scaled data: fitted = 1.5 * theory + 0.1
  %(prog)s --plot-simulated-data --phi-angles "0,45,90,135"  # Plot with custom phi angles
  %(prog)s --plot-simulated-data --phi-angles "30,60,90" --contrast 1.2 --offset 0.05  # Custom angles with scaling

  Distributed Computing (2-5x speedup):
  %(prog)s --distributed                      # Auto-detect backend, use all cores
  %(prog)s --distributed --backend ray        # Use Ray for cluster computing
  %(prog)s --distributed --backend mpi --workers 8  # Use MPI with 8 processes
  %(prog)s --distributed --workers 4          # Limit to 4 workers (any backend)
  %(prog)s --distributed --laminar-flow       # Distributed optimization for complex 7-parameter analysis

  ML Acceleration (2-5x faster convergence):
  %(prog)s --ml-accelerated                   # Enable ML with auto training data collection
  %(prog)s --ml-accelerated --train-ml-model  # Train models before analysis
  %(prog)s --ml-accelerated --enable-transfer-learning  # Use transfer learning for similar conditions

  Combined High-Performance Computing:
  %(prog)s --distributed --ml-accelerated     # Maximum speedup: distributed + ML acceleration
  %(prog)s --distributed --ml-accelerated --laminar-flow --method robust  # Full feature analysis
  %(prog)s --distributed --backend ray --ml-accelerated --workers 16      # Cluster + ML acceleration

Method Quality Assessment:
  Classical: Uses chi-squared goodness-of-fit (lower is better)
  Robust:    Uses chi-squared with uncertainty resistance (robust to measurement noise)

  Note: When running --method all, both methods provide chi-squared metrics for comparison.
        Robust methods provide noise resistance at computational cost.
        Classical methods offer faster execution with reliable parameter estimates.
        """,
    )

    # Basic options
    parser.add_argument(
        "--method",
        choices=["classical", "robust", "all"],
        default="classical",
        help="Analysis method to use (default: %(default)s)",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default="./homodyne_config.json",
        help="Path to configuration file (default: %(default)s)",
    )

    parser.add_argument(
        "--data",
        type=Path,
        help="Path to input data file (optional, can be specified in config)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Path to output file (optional, defaults to auto-generated filename)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default="./homodyne_results",
        help="Output directory for results (default: %(default)s)",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose DEBUG logging"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable console logging (file logging remains enabled)",
    )

    # Analysis mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--static-isotropic",
        action="store_true",
        help="Force static isotropic mode analysis (3 parameters, no angle selection)",
    )
    mode_group.add_argument(
        "--static-anisotropic",
        action="store_true",
        help="Force static anisotropic mode analysis (3 parameters, with angle selection)",
    )
    mode_group.add_argument(
        "--laminar-flow",
        action="store_true",
        help="Force laminar flow mode analysis (7 parameters: all diffusion and shear parameters)",
    )

    # Plotting options
    parser.add_argument(
        "--plot-experimental-data",
        action="store_true",
        help="Generate validation plots of experimental data after loading for quality checking",
    )

    parser.add_argument(
        "--plot-simulated-data",
        action="store_true",
        help="Plot theoretical C2 heatmaps using initial parameters from config without experimental data",
    )

    parser.add_argument(
        "--contrast",
        type=float,
        default=1.0,
        help="Contrast parameter for scaling: fitted = contrast * theory + offset (default: 1.0)",
    )

    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Offset parameter for scaling: fitted = contrast * theory + offset (default: 0.0)",
    )

    parser.add_argument(
        "--phi-angles",
        type=str,
        help="Comma-separated list of phi angles in degrees (e.g., '0,45,90,135'). Default: '0,36,72,108,144'",
    )

    # Shell completion
    parser.add_argument(
        "--install-completion",
        choices=["bash", "zsh", "fish", "powershell"],
        help="Install shell completion for the specified shell",
    )
    parser.add_argument(
        "--uninstall-completion",
        choices=["bash", "zsh", "fish", "powershell"],
        help="Uninstall shell completion for the specified shell",
    )

    # Distributed computing options
    distributed_group = parser.add_argument_group(
        "Distributed Computing Options",
        description="""
Enable distributed optimization across multiple nodes/processes for significantly faster analysis.
Supports Ray (scalable clusters), MPI (HPC environments), Dask (data science workflows),
and multiprocessing (local parallelization). Provides 2-5x speedup depending on problem size.

Examples:
  --distributed                           # Auto-detect best backend, use all CPU cores
  --distributed --backend ray             # Use Ray for cluster computing
  --distributed --backend mpi --workers 8 # Use MPI with 8 processes
  --distributed --backend dask            # Use Dask distributed computing
  --distributed --workers 4               # Limit to 4 worker processes
        """,
    )
    distributed_group.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed optimization across multiple nodes/processes. "
        "Automatically detects available backends and scales to available resources. "
        "Can provide 2-5x speedup for large parameter sweeps and complex optimizations.",
    )
    distributed_group.add_argument(
        "--backend",
        choices=["auto", "ray", "mpi", "dask", "multiprocessing"],
        default="auto",
        help="Distributed computing backend selection (default: %(default)s). "
        "'auto': detect best available backend; "
        "'ray': scalable clusters, best for cloud/multiple machines; "
        "'mpi': HPC environments, requires mpiexec; "
        "'dask': data science workflows, good for mixed compute; "
        "'multiprocessing': local parallelization, always available.",
    )
    distributed_group.add_argument(
        "--workers",
        type=int,
        help="Number of worker processes/nodes to use (default: auto-detect based on CPU cores). "
        "For multiprocessing: limited by CPU cores. "
        "For Ray/Dask: can exceed local cores if using cluster. "
        "For MPI: should match mpiexec -n parameter.",
    )
    distributed_group.add_argument(
        "--distributed-config",
        type=Path,
        help="Path to distributed computing configuration file. "
        "JSON file with backend-specific settings like cluster addresses, "
        "memory limits, and scaling parameters.",
    )

    # ML acceleration options
    ml_group = parser.add_argument_group(
        "ML Acceleration Options",
        description="""
Enable machine learning acceleration for faster optimization convergence.
Uses neural networks to provide better initial guesses and guide optimization,
potentially reducing analysis time by 2-5x while maintaining accuracy.

Examples:
  --ml-accelerated                        # Enable ML acceleration with defaults
  --ml-accelerated --train-ml-model       # Train new models on current data
  --ml-accelerated --enable-transfer-learning  # Use pre-trained models
  --ml-data-path ./ml_training_data       # Custom ML training data location
        """,
    )
    ml_group.add_argument(
        "--ml-accelerated",
        action="store_true",
        help="Enable machine learning acceleration for optimization. "
        "Uses neural networks to provide better initial parameter guesses "
        "and guide optimization convergence, potentially reducing analysis time by 2-5x.",
    )
    ml_group.add_argument(
        "--train-ml-model",
        action="store_true",
        help="Train new ML models on current analysis data. "
        "Useful when analyzing new experimental conditions or parameter ranges. "
        "Training data will be saved for future use.",
    )
    ml_group.add_argument(
        "--enable-transfer-learning",
        action="store_true",
        help="Enable transfer learning from pre-trained models. "
        "Uses existing ML models trained on similar conditions to accelerate convergence. "
        "Particularly effective for related experimental setups.",
    )
    ml_group.add_argument(
        "--ml-data-path",
        type=Path,
        default="./ml_training_data",
        help="Path to ML training data directory (default: %(default)s). "
        "Used for storing and loading ML model training datasets.",
    )

    # Advanced optimization options
    advanced_group = parser.add_argument_group(
        "Advanced Optimization Options",
        description="""
Advanced features for specialized optimization workflows including parameter sweeps,
benchmarking, and performance analysis. These options provide detailed control
over optimization behavior and performance characteristics.

Examples:
  --parameter-sweep --parameter-ranges "D0:10-100,alpha:-1-1"  # Parameter space exploration
  --benchmark                           # Compare method performance
  --auto-optimize                       # Automatic method and parameter selection
        """,
    )
    advanced_group.add_argument(
        "--parameter-sweep",
        action="store_true",
        help="Enable parameter sweep analysis across specified ranges. "
        "Requires --distributed to be enabled. Systematically explores "
        "parameter space to understand sensitivity and optimal regions.",
    )
    advanced_group.add_argument(
        "--parameter-ranges",
        type=str,
        help="Parameter ranges for sweep analysis in format 'param1:min-max,param2:min-max'. "
        "Example: 'D0:10-100,alpha:-1-1'. Only used with --parameter-sweep.",
    )
    advanced_group.add_argument(
        "--benchmark",
        action="store_true",
        help="Run comprehensive benchmark comparing all available optimization methods. "
        "Provides detailed performance analysis including timing, accuracy, and convergence. "
        "Cannot be combined with --distributed or --ml-accelerated.",
    )
    advanced_group.add_argument(
        "--auto-optimize",
        action="store_true",
        help="Enable automatic optimization strategy selection. "
        "Analyzes data characteristics and system capabilities to choose "
        "optimal combination of methods, backends, and parameters.",
    )

    # Setup shell completion if available
    if COMPLETION_AVAILABLE:
        setup_shell_completion(parser)

    return parser
