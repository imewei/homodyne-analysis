"""
Homodyne Scattering Analysis Package
===================================

A comprehensive Python package for analyzing homodyne scattering in X-ray Photon
Correlation Spectroscopy (XPCS) under nonequilibrium conditions, specifically
designed for characterizing transport properties in flowing soft matter systems.

Physical Framework:
------------------
This package implements the theoretical framework for analyzing time-dependent
intensity correlation functions g₂(φ,t₁,t₂) that capture the interplay between
Brownian diffusion and advective shear flow in complex fluids.

Reference:
H. He, H. Liang, M. Chu, Z. Jiang, J.J. de Pablo, M.V. Tirrell, S. Narayanan,
& W. Chen, "Transport coefficient approach for characterizing nonequilibrium
dynamics in soft matter", Proc. Natl. Acad. Sci. U.S.A. 121 (31) e2401162121 (2024).

Key Capabilities:
----------------
- Dual Analysis Modes: Static (3 parameters) and Laminar Flow (7 parameters)
- Classical Optimization: Fast Nelder-Mead for point estimates
- Bayesian MCMC: Full posterior distributions with uncertainty quantification
- Performance Optimization: Numba JIT compilation and smart angle filtering
- Data Validation: Comprehensive quality control and visualization
- Result Management: JSON serialization with custom NumPy array handling

Core Modules:
------------
- core.config: Configuration management with template system
- core.kernels: Optimized computational kernels for correlation functions
- core.io_utils: Data I/O with experimental data loading and result saving
- analysis.core: Main analysis engine and chi-squared fitting
- optimization.classical: Scipy-based optimization with angle filtering
- optimization.mcmc: PyMC-based Bayesian parameter estimation
- plotting: Comprehensive visualization for data validation and diagnostics

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory & University of Chicago
"""

from .core.config import ConfigManager, configure_logging, performance_monitor
from .core.kernels import (
    create_time_integral_matrix_numba,
    calculate_diffusion_coefficient_numba,
    calculate_shear_rate_numba,
    compute_g1_correlation_numba,
    compute_sinc_squared_numba,
    memory_efficient_cache,
    # New optimized kernels
    create_symmetric_matrix_optimized,
    matrix_vector_multiply_optimized,
    apply_scaling_vectorized,
    compute_chi_squared_fast,
    exp_negative_vectorized,
)
from .analysis.core import HomodyneAnalysisCore

# Import optimization modules with graceful degradation
# Classical optimization requires only scipy (typically available)
try:
    from .optimization.classical import ClassicalOptimizer
except ImportError as e:
    ClassicalOptimizer = None
    import logging

    logging.getLogger(__name__).warning(
        f"Classical optimization not available - missing scipy: {e}"
    )

# MCMC optimization requires PyMC ecosystem (optional advanced feature)
try:
    from .optimization.mcmc import MCMCSampler, create_mcmc_sampler
except ImportError as e:
    MCMCSampler = None
    create_mcmc_sampler = None
    import logging

    logging.getLogger(__name__).warning(
        f"MCMC Bayesian analysis not available - missing PyMC/ArviZ: {e}"
    )

# Core exports that should always be available
__all__ = [
    # Core functionality
    "ConfigManager",
    "configure_logging",
    "performance_monitor",
    "HomodyneAnalysisCore",
    # Computational kernels
    "create_time_integral_matrix_numba",
    "calculate_diffusion_coefficient_numba",
    "calculate_shear_rate_numba",
    "compute_g1_correlation_numba",
    "compute_sinc_squared_numba",
    "memory_efficient_cache",
    # Optimized kernels
    "create_symmetric_matrix_optimized",
    "matrix_vector_multiply_optimized",
    "apply_scaling_vectorized",
    "compute_chi_squared_fast",
    "exp_negative_vectorized",
    # Optimization (conditionally available)
    "ClassicalOptimizer",
    "MCMCSampler",
    "create_mcmc_sampler",
]

# Version information
__version__ = "6.0"
__author__ = "Wei Chen, Hongrui He"
__email__ = "wchen@anl.gov"
__institution__ = "Argonne National Laboratory & University of Chicago"
