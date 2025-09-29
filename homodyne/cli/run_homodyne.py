"""
Homodyne Analysis Runner
========================

Command-line interface for running homodyne scattering analysis in X-ray Photon
Correlation Spectroscopy (XPCS) under nonequilibrium conditions.

This script provides a unified interface for:
- Classical optimization (Nelder-Mead, Gurobi) for fast parameter estimation
- Robust optimization (Wasserstein DRO, Scenario-based, Ellipsoidal) for noise resistance
- Dual analysis modes: Static (3 params) and Laminar Flow (7 params)
- Comprehensive data validation and quality control
- Automated result saving and visualization

Method Flags Documentation
==========================

| Flag               | Methods Run                                               | Description                       |
|--------------------|-----------------------------------------------------------|-----------------------------------|
| --method classical | Nelder-Mead + Gurobi                                     | Traditional classical methods     |
|                    |                                                           | only (2 methods)                 |
| --method robust    | Robust-Wasserstein + Robust-Scenario + Robust-Ellipsoidal| Robust methods only (3 methods)  |
| --method all       | Classical + Robust                                        | All available methods (5 total)  |

Method Execution Logic for --method all
=======================================

The --method all flag runs both Classical and Robust optimization methods:

1. Classical Optimization:
   - Nelder-Mead: Derivative-free simplex algorithm
   - Gurobi: Quadratic programming with trust region (if available)
   - Fast execution with reliable parameter estimates

2. Robust Optimization:
   - Wasserstein DRO: Distributionally robust with uncertainty sets
   - Scenario-based: Bootstrap resampling for outlier resistance
   - Ellipsoidal: Bounded uncertainty quantification
   - Noise-resistant analysis for experimental data

3. Result Comparison:
   - Both methods provide independent parameter estimates
   - Includes chi-squared goodness-of-fit metrics for comparison
   - Users can evaluate robustness across optimization approaches
"""

__author__ = "Wei Chen, Hongrui He"
__credits__ = "Argonne National Laboratory"

import os
import sys


# Call fast completion handler immediately - before any heavy imports
try:
    from ..ui.completion.fast_handler import handle_fast_completion
    handle_fast_completion()
except ImportError:
    # Fallback to minimal fast completion if advanced system not available
    if os.environ.get("_ARGCOMPLETE") == "1":
        # Basic fallback completion
        comp_line = os.environ.get("COMP_LINE", "")
        if "--method" in comp_line:
            print("classical")
            print("robust")
            print("all")
        elif "--config" in comp_line:
            print("config.json")
        sys.exit(0)

# Import the main function from the modularized core
from .core import main

# Re-export key functions and classes for backward compatibility
try:
    from .optimization import run_classical_optimization, run_robust_optimization, run_all_methods
    from .simulation import plot_simulated_data
    from .visualization import generate_classical_plots, generate_robust_plots, generate_comparison_plots
    from .utils import setup_logging, print_banner, MockResult, print_method_documentation
    from .parser import create_argument_parser

    # For backward compatibility, also expose some internal functions
    from .core import run_analysis, initialize_analysis_engine, load_and_validate_data

    # Add perform_analysis as an alias to run_analysis for test compatibility
    perform_analysis = run_analysis

except ImportError as e:
    # Handle cases where modular imports fail
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Some modular imports failed: {e}")
    logger.warning("Falling back to basic functionality")

# Main entry point
if __name__ == "__main__":
    main()