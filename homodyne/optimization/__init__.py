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
from typing import TYPE_CHECKING, Any, List, Optional, Type

if TYPE_CHECKING:
    from .classical import ClassicalOptimizer

# Track available exports
_available_exports: list[str] = []

# Always try to import ClassicalOptimizer
try:
    from .classical import ClassicalOptimizer

    _available_exports.append("ClassicalOptimizer")
except ImportError as e:
    ClassicalOptimizer: type[Any] | None = None  # type: ignore[misc,no-redef]
    import warnings

    warnings.warn(f"ClassicalOptimizer not available: {e}", ImportWarning, stacklevel=2)


# Dynamic __all__ - suppress Pylance warning as this is intentional
__all__ = _available_exports  # type: ignore[misc]
