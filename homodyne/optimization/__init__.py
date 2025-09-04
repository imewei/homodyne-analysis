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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from homodyne.optimization.classical import (
        ClassicalOptimizer as _ClassicalOptimizer,
    )
    from homodyne.optimization.mcmc import MCMCSampler as _MCMCSampler

# Track available exports
_available_exports: list[str] = []

# Always try to import ClassicalOptimizer
try:
    from homodyne.optimization.classical import ClassicalOptimizer

    _available_exports.append("ClassicalOptimizer")
except ImportError as e:
    ClassicalOptimizer = None  # type: ignore[assignment,misc]
    import warnings

    warnings.warn(f"ClassicalOptimizer not available: {e}", ImportWarning, stacklevel=2)

# Conditionally import MCMC components
try:
    from homodyne.optimization.mcmc import MCMCSampler, create_mcmc_sampler

    _available_exports.extend(["MCMCSampler", "create_mcmc_sampler"])
except ImportError as e:
    MCMCSampler = None  # type: ignore[assignment,misc]
    create_mcmc_sampler = None  # type: ignore[assignment]
    import warnings

    warnings.warn(
        f"MCMC functionality not available (PyMC required): {e}",
        ImportWarning,
        stacklevel=2,
    )

# Dynamic __all__ - suppress Pylance warning as this is intentional
__all__ = _available_exports
