"""
Command-line interface and runner modules for homodyne analysis.

This module provides backward compatibility for CLI tools moved from the root directory.
"""

# Import main CLI functions when this module is imported
# This enables both new-style and old-style imports to work

try:
    from .create_config import main as create_config_main
    from .enhanced_runner import main as enhanced_runner_main
    from .run_homodyne import main as run_homodyne_main
    from .core import (
        main as core_main,
        run_analysis,
        initialize_analysis_engine,
        load_and_validate_data
    )

    # Import key functions from modular structure
    from .optimization import run_classical_optimization, run_robust_optimization, run_all_methods
    from .simulation import plot_simulated_data
    from .visualization import (
        generate_classical_plots,
        generate_robust_plots,
        generate_comparison_plots,
        save_individual_method_results,
        generate_c2_heatmap_plots
    )
    from .utils import setup_logging, print_banner, MockResult, print_method_documentation
    from .parser import create_argument_parser

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
    "create_config_main",
    "enhanced_runner_main",
    "run_homodyne_main",
    "core_main",
    "run_classical_optimization",
    "run_robust_optimization",
    "run_all_methods",
    "plot_simulated_data",
    "generate_classical_plots",
    "generate_robust_plots",
    "generate_comparison_plots",
    "save_individual_method_results",
    "setup_logging",
    "print_banner",
    "MockResult",
    "print_method_documentation",
    "create_argument_parser",
]
