"""
Optimization methods for homodyne scattering analysis.

This subpackage provides various optimization approaches for fitting
theoretical models to experimental data:

- **Classical optimization**: Multiple methods including Nelder-Mead simplex
  and Gurobi quadratic programming (with automatic detection)
- **Robust optimization**: Distributionally robust methods for handling
  measurement noise and experimental uncertainties

All optimization methods use consistent parameter bounds and physical constraints
for reliable and comparable results across different optimization approaches.
"""

# Import with error handling for optional dependencies
from typing import Any

# Track available exports
_available_exports: list[str] = []

# Always try to import ClassicalOptimizer
try:
    from .classical import ClassicalOptimizer
    from .classical import run_classical_optimization_optimized

    _available_exports.extend(
        ["ClassicalOptimizer", "run_classical_optimization_optimized"]
    )
except ImportError as e:
    ClassicalOptimizer: type[Any] | None = None  # type: ignore[misc,no-redef]
    run_classical_optimization_optimized = None  # type: ignore[misc,assignment]
    import warnings

    warnings.warn(f"ClassicalOptimizer not available: {e}", ImportWarning, stacklevel=2)

# Try to import RobustHomodyneOptimizer
try:
    from .robust import RobustHomodyneOptimizer
    from .robust import run_robust_optimization

    _available_exports.extend(["RobustHomodyneOptimizer", "run_robust_optimization"])
except ImportError as e:
    RobustHomodyneOptimizer: type[Any] | None = None  # type: ignore[misc,no-redef]
    run_robust_optimization = None  # type: ignore[misc,assignment]
    import warnings

    warnings.warn(
        f"RobustHomodyneOptimizer not available: {e}", ImportWarning, stacklevel=2
    )


# Dynamic __all__ - suppress Pylance warning as this is intentional
__all__ = _available_exports  # type: ignore[misc]
