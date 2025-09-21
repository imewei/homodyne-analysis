"""
Homodyne Scattering Analysis Package
====================================

High-performance Python package for analyzing homodyne scattering in X-ray Photon
Correlation Spectroscopy (XPCS) under nonequilibrium conditions. Implements the
theoretical framework from He et al. PNAS 2024 for characterizing transport
properties in flowing soft matter systems.

Analyzes time-dependent intensity correlation functions c₂(φ,t₁,t₂) capturing
the interplay between Brownian diffusion and advective shear flow.

Reference:
H. He, H. Liang, M. Chu, Z. Jiang, J.J. de Pablo, M.V. Tirrell, S. Narayanan,
& W. Chen, "Transport coefficient approach for characterizing nonequilibrium
dynamics in soft matter", Proc. Natl. Acad. Sci. U.S.A. 121 (31) e2401162121 (2024).

Key Features:
- Three analysis modes: Static Isotropic (3 params), Static Anisotropic (3 params),
  Laminar Flow (7 params)
- Multiple optimization methods: Classical (Nelder-Mead, Gurobi), Robust
  (Wasserstein DRO, Scenario-based, Ellipsoidal)
- Noise-resistant analysis: Robust optimization for measurement uncertainty and outliers
- High performance: Numba JIT compilation with 3-5x speedup and smart angle filtering
- Scientific accuracy: Automatic g2 = offset + contrast * g1 fitting
- Consistent bounds: All optimization methods use identical parameter constraints

Core Modules:
- core.config: Configuration management with template system
- core.kernels: Optimized computational kernels for correlation functions
- core.io_utils: Data I/O with experimental data loading and result saving
- analysis.core: Main analysis engine and chi-squared fitting
- optimization.classical: Multiple methods (Nelder-Mead, Gurobi QP) with angle filtering
- optimization.robust: Robust optimization (Wasserstein DRO, Scenario-based,
  Ellipsoidal)
- plotting: Comprehensive visualization for data validation and diagnostics

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

# Check Python version requirement early
import sys

# Performance profiling utilities removed - functionality available via
# core.config.performance_monitor
from .analysis.core import HomodyneAnalysisCore
from .core.config import ConfigManager, configure_logging, performance_monitor
from .core.kernels import (
    calculate_diffusion_coefficient_numba,
    calculate_shear_rate_numba,
    compute_g1_correlation_numba,
    compute_sinc_squared_numba,
    create_time_integral_matrix_numba,
    memory_efficient_cache,
)

# Optional optimization modules with graceful degradation
try:
    from .optimization.classical import ClassicalOptimizer
except ImportError as e:
    ClassicalOptimizer = None  # type: ignore[assignment]
    import logging

    logging.getLogger(__name__).warning(
        f"Classical optimization not available - missing scipy: {e}"
    )

try:
    from .optimization.robust import RobustHomodyneOptimizer, create_robust_optimizer
except ImportError as e:
    RobustHomodyneOptimizer = None  # type: ignore[assignment]
    create_robust_optimizer = None  # type: ignore[assignment]
    import logging

    logging.getLogger(__name__).warning(
        f"Robust optimization not available - missing CVXPY: {e}"
    )


__all__ = [
    "ClassicalOptimizer",
    "ConfigManager",
    "HomodyneAnalysisCore",
    "RobustHomodyneOptimizer",
    "calculate_diffusion_coefficient_numba",
    "calculate_shear_rate_numba",
    "compute_g1_correlation_numba",
    "compute_sinc_squared_numba",
    "configure_logging",
    "create_robust_optimizer",
    "create_time_integral_matrix_numba",
    "memory_efficient_cache",
    "performance_monitor",
]

# Version information
__version__ = "0.7.1"
__author__ = "Wei Chen, Hongrui He"
__email__ = "wchen@anl.gov"
__institution__ = "Argonne National Laboratory"

# Recent improvements (v0.6.6)
# - Added robust optimization framework with CVXPY + Gurobi
# - Distributionally Robust Optimization (DRO) with Wasserstein uncertainty sets
# - Scenario-based robust optimization with bootstrap resampling
# - Ellipsoidal uncertainty sets for bounded data uncertainty
# - Seamless integration with existing classical optimization workflow
# - Comprehensive configuration support for robust methods
# - Enhanced error handling and graceful degradation for optional dependencies
#
# Previous improvements (v0.6.2):
# - Major performance optimizations: Chi-squared calculation 38% faster
# - Memory access optimizations with vectorized operations
# - Configuration caching to reduce overhead
# - Optimized least squares solving for parameter scaling
# - Memory pooling for reduced allocation overhead
# - Enhanced performance regression testing
