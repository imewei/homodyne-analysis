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
    ClassicalOptimizer = None  # type: ignore[assignment,misc]
    import logging

    logging.getLogger(__name__).warning(
        f"Classical optimization not available - missing scipy: {e}"
    )

try:
    from .optimization.robust import RobustHomodyneOptimizer, create_robust_optimizer
except ImportError as e:
    RobustHomodyneOptimizer = None  # type: ignore[assignment,misc]
    create_robust_optimizer = None  # type: ignore[assignment]
    import logging

    logging.getLogger(__name__).warning(
        f"Robust optimization not available - missing CVXPY: {e}"
    )

# Backward compatibility imports for CLI functions
try:
    from .cli.create_config import main as create_config_main
    from .cli.enhanced_runner import main as enhanced_runner_main
    from .cli.run_homodyne import main as run_homodyne_main
except ImportError:
    run_homodyne_main = None
    create_config_main = None
    enhanced_runner_main = None

# Backward compatibility imports for plotting functions
try:
    from .visualization.enhanced_plotting import EnhancedPlottingManager
    from .visualization.plotting import get_plot_config, plot_c2_heatmaps
except ImportError:
    plot_c2_heatmaps = None
    get_plot_config = None
    EnhancedPlottingManager = None

# Backward compatibility imports for performance monitoring
try:
    from .performance import PerformanceMonitor
except ImportError:
    PerformanceMonitor = None

# Backward compatibility imports for configuration utilities
try:
    from .config import TEMPLATE_FILES, get_config_dir, get_template_path
except ImportError:
    get_template_path = None
    get_config_dir = None
    TEMPLATE_FILES = None


__all__ = [
    "TEMPLATE_FILES",
    "ClassicalOptimizer",
    "ConfigManager",
    "EnhancedPlottingManager",
    "HomodyneAnalysisCore",
    "PerformanceMonitor",
    "RobustHomodyneOptimizer",
    "calculate_diffusion_coefficient_numba",
    "calculate_shear_rate_numba",
    "compute_g1_correlation_numba",
    "compute_sinc_squared_numba",
    "configure_logging",
    "create_config_main",
    "create_robust_optimizer",
    "create_time_integral_matrix_numba",
    "enhanced_runner_main",
    "get_config_dir",
    "get_plot_config",
    "get_template_path",
    "memory_efficient_cache",
    "performance_monitor",
    "plot_c2_heatmaps",
    "run_homodyne_main",
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
