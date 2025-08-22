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
try:
    from .classical import ClassicalOptimizer
except ImportError as e:
    ClassicalOptimizer = None
    import warnings

    warnings.warn(f"ClassicalOptimizer not available: {e}", ImportWarning)


try:
    from .mcmc import MCMCSampler, create_mcmc_sampler
except ImportError as e:
    MCMCSampler = None
    create_mcmc_sampler = None
    import warnings

    warnings.warn(
        f"MCMC functionality not available (PyMC required): {e}", ImportWarning
    )

__all__ = [
    "ClassicalOptimizer",
    "MCMCSampler",
    "create_mcmc_sampler",
]
