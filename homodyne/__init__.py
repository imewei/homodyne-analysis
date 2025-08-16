"""
Homodyne Scattering Analysis Package
===================================

A comprehensive package for analyzing homodyne scattering in X-ray Photon
Correlation Spectroscopy (XPCS) under nonequilibrium conditions.

This package implements the theoretical framework described in:
H. He, H. Liang, M. Chu, Z. Jiang, J.J. de Pablo, M.V. Tirrell, S. Narayanan,
& W. Chen, "Transport coefficient approach for characterizing nonequilibrium
dynamics in soft matter", Proc. Natl. Acad. Sci. U.S.A. 121 (31) e2401162121 (2024).

Modules:
--------
- core: Core functionality (configuration, computational kernels)
- analysis: Analysis engines and data processing
- optimization: Optimization methods (classical, MCMC)
- io_utils: Input/output utilities
- plotting: Visualization functions

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory & University of Chicago
"""

from .core.config import ConfigManager, configure_logging
from .core.kernels import (
    create_time_integral_matrix_numba,
    calculate_diffusion_coefficient_numba,
    calculate_shear_rate_numba,
    compute_g1_correlation_numba,
    compute_sinc_squared_numba,
    memory_efficient_cache,
)
from .analysis.core import HomodyneAnalysisCore

# Optional optimization imports with error handling
try:
    from .optimization.classical import ClassicalOptimizer
except ImportError as e:
    ClassicalOptimizer = None
    import logging

    logging.getLogger(__name__).warning(
        f"ClassicalOptimizer not available: {e}"
    )


try:
    from .optimization.mcmc import MCMCSampler, create_mcmc_sampler
except ImportError as e:
    MCMCSampler = None
    create_mcmc_sampler = None
    import logging

    logging.getLogger(__name__).warning(
        f"MCMC functionality not available: {e}"
    )

# Core exports that should always be available
__all__ = [
    # Core functionality
    "ConfigManager",
    "configure_logging",
    "HomodyneAnalysisCore",
    # Computational kernels
    "create_time_integral_matrix_numba",
    "calculate_diffusion_coefficient_numba",
    "calculate_shear_rate_numba",
    "compute_g1_correlation_numba",
    "compute_sinc_squared_numba",
    "memory_efficient_cache",
    # Optimization (conditionally available)
    "ClassicalOptimizer",
    "MCMCSampler",
    "create_mcmc_sampler",
]

# Version information
__version__ = "5.1"
__author__ = "Wei Chen, Hongrui He"
__email__ = "wchen@anl.gov"
__institution__ = "Argonne National Laboratory & University of Chicago"
