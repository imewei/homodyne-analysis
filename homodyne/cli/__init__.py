"""
Command-line interface and runner modules for homodyne analysis.

This module provides backward compatibility for CLI tools moved from the root directory.
"""

# Import main CLI functions when this module is imported
# This enables both new-style and old-style imports to work

try:
    from .core import initialize_analysis_engine
    from .core import load_and_validate_data
    from .core import main as core_main
    from .core import run_analysis
    from .create_config import main as create_config_main
    from .enhanced_runner import main as enhanced_runner_main

    # Import key functions from modular structure
    from .optimization import run_all_methods
    from .optimization import run_classical_optimization
    from .optimization import run_robust_optimization
    from .parser import create_argument_parser
    from .run_homodyne import main as run_homodyne_main
    from .simulation import plot_simulated_data
    from .utils import MockResult
    from .utils import print_banner
    from .utils import print_method_documentation
    from .utils import setup_logging
    from .visualization import generate_c2_heatmap_plots
    from .visualization import generate_classical_plots
    from .visualization import generate_comparison_plots
    from .visualization import generate_robust_plots
    from .visualization import save_individual_method_results

except ImportError as e:
    # Graceful degradation if files haven't been moved yet
    import warnings

    warnings.warn(f"Could not import CLI modules: {e}", ImportWarning, stacklevel=2)

    run_homodyne_main = None
    create_config_main = None
    enhanced_runner_main = None
    core_main = None
    run_analysis = None
    initialize_analysis_engine = None
    load_and_validate_data = None

    # Set other imports to None for graceful degradation
    run_classical_optimization = None
    run_robust_optimization = None
    run_all_methods = None
    plot_simulated_data = None
    generate_classical_plots = None
    generate_robust_plots = None
    generate_comparison_plots = None
    save_individual_method_results = None
    generate_c2_heatmap_plots = None
    setup_logging = None
    print_banner = None
    MockResult = None
    print_method_documentation = None
    create_argument_parser = None

__all__ = [
    "MockResult",
    "core_main",
    "create_argument_parser",
    "create_config_main",
    "enhanced_runner_main",
    "generate_classical_plots",
    "generate_comparison_plots",
    "generate_robust_plots",
    "plot_simulated_data",
    "print_banner",
    "print_method_documentation",
    "run_all_methods",
    "run_classical_optimization",
    "run_homodyne_main",
    "run_robust_optimization",
    "save_individual_method_results",
    "setup_logging",
]
