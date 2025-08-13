"""
Optimization methods for homodyne scattering analysis.

This subpackage provides various optimization approaches for fitting
theoretical models to experimental data:
- Classical optimization using scipy methods
- Bayesian optimization using Gaussian processes
- MCMC sampling for uncertainty quantification
"""

# Import with error handling for optional dependencies
try:
    from .classical import ClassicalOptimizer
except ImportError as e:
    ClassicalOptimizer = None
    import warnings

    warnings.warn(f"ClassicalOptimizer not available: {e}", ImportWarning)

try:
    from .bayesian import BayesianOptimizer
except ImportError as e:
    BayesianOptimizer = None
    import warnings

    warnings.warn(
        f"BayesianOptimizer not available (scikit-optimize required): {e}",
        ImportWarning,
    )

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
    "BayesianOptimizer",
    "MCMCSampler",
    "create_mcmc_sampler",
]
