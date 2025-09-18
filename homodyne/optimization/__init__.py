"""
Optimization methods for homodyne scattering analysis.

This subpackage provides various optimization approaches for fitting
theoretical models to experimental data:

- **Classical optimization**: Multiple methods including Nelder-Mead simplex
  and Gurobi quadratic programming (with automatic detection)
- **MCMC sampling**: Bayesian uncertainty quantification using NUTS sampler

All optimization methods use consistent parameter bounds and physical constraints
for reliable and comparable results across different optimization approaches.
"""

# Import with error handling for optional dependencies
from typing import TYPE_CHECKING, Any, List, Optional, Type

if TYPE_CHECKING:
    from .classical import ClassicalOptimizer
    from .mcmc import MCMCSampler

# Track available exports
_available_exports: List[str] = []

# Always try to import ClassicalOptimizer
try:
    from .classical import ClassicalOptimizer

    _available_exports.append("ClassicalOptimizer")
except ImportError as e:
    ClassicalOptimizer: Optional[Type[Any]] = None  # type: ignore[misc]
    import warnings

    warnings.warn(f"ClassicalOptimizer not available: {e}", ImportWarning)

# Conditionally import MCMC components
try:
    from .mcmc import MCMCSampler, create_mcmc_sampler

    _available_exports.extend(["MCMCSampler", "create_mcmc_sampler"])
except ImportError as e:
    MCMCSampler: Optional[Type[Any]] = None  # type: ignore[misc]
    create_mcmc_sampler: Optional[Any] = None  # type: ignore[misc]
    import warnings

    warnings.warn(
        f"MCMC functionality not available (PyMC required): {e}", ImportWarning
    )

# Dynamic __all__ - suppress Pylance warning as this is intentional
__all__ = _available_exports  # type: ignore[misc]
