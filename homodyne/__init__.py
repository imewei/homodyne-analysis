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

import importlib

# Enhanced lazy loading implementation for performance optimization
from typing import TYPE_CHECKING
from typing import Any

# Import initialization optimizer for startup performance
from .core.initialization_optimizer import get_initialization_optimizer

# Import the advanced lazy loading system
from .core.lazy_imports import HeavyDependencyLoader
from .core.lazy_imports import scientific_deps

# Type checking imports (no runtime cost)
if TYPE_CHECKING:
    from .analysis.core import HomodyneAnalysisCore
    from .config import TEMPLATE_FILES
    from .config import get_config_dir
    from .config import get_template_path
    from .core.config import ConfigManager
    from .core.config import performance_monitor
    from .optimization.classical import ClassicalOptimizer
    from .optimization.robust import RobustHomodyneOptimizer
    from .optimization.robust import create_robust_optimizer
    from .performance import PerformanceMonitor
    from .visualization.enhanced_plotting import EnhancedPlottingManager
    from .visualization.plotting import get_plot_config
    from .visualization.plotting import plot_c2_heatmaps

# Essential imports only (fast loading)


# Enhanced lazy loading class for deferred imports
class _LazyLoader:
    """Enhanced lazy loader for expensive imports with performance monitoring."""

    def __init__(self, module_name: str, class_name: str | None = None):
        self.module_name = module_name
        self.class_name = class_name
        self._heavy_loader = HeavyDependencyLoader(
            module_name=module_name,
            attribute=class_name,
            required=True  # Most package components are required
        )

    def __call__(self, *args, **kwargs):
        return self._heavy_loader(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._heavy_loader, name)

    @property
    def is_available(self) -> bool:
        """Check if the module is available."""
        return self._heavy_loader.is_available


# Core functionality - lazy loaded
HomodyneAnalysisCore = _LazyLoader(".analysis.core", "HomodyneAnalysisCore")
ConfigManager = _LazyLoader(".core.config", "ConfigManager")
performance_monitor = _LazyLoader(".core.config", "performance_monitor")

# Kernels - lazy loaded for Numba compilation overhead
_kernels_module = _LazyLoader(".core.kernels")
calculate_diffusion_coefficient_numba = (
    lambda *args, **kwargs: _kernels_module.calculate_diffusion_coefficient_numba(
        *args, **kwargs
    )
)
calculate_shear_rate_numba = (
    lambda *args, **kwargs: _kernels_module.calculate_shear_rate_numba(*args, **kwargs)
)
compute_g1_correlation_numba = (
    lambda *args, **kwargs: _kernels_module.compute_g1_correlation_numba(
        *args, **kwargs
    )
)
compute_sinc_squared_numba = (
    lambda *args, **kwargs: _kernels_module.compute_sinc_squared_numba(*args, **kwargs)
)
create_time_integral_matrix_numba = (
    lambda *args, **kwargs: _kernels_module.create_time_integral_matrix_numba(
        *args, **kwargs
    )
)
memory_efficient_cache = lambda *args, **kwargs: _kernels_module.memory_efficient_cache(
    *args, **kwargs
)

# Optimization modules - expensive imports, lazy loaded
ClassicalOptimizer = _LazyLoader(".optimization.classical", "ClassicalOptimizer")
RobustHomodyneOptimizer = _LazyLoader(".optimization.robust", "RobustHomodyneOptimizer")
create_robust_optimizer = _LazyLoader(".optimization.robust", "create_robust_optimizer")

# CLI functions - lazy loaded
run_homodyne_main = _LazyLoader(".cli.run_homodyne", "main")
create_config_main = _LazyLoader(".cli.create_config", "main")
enhanced_runner_main = _LazyLoader(".cli.enhanced_runner", "main")

# Plotting functions - lazy loaded to avoid matplotlib import cost
plot_c2_heatmaps = _LazyLoader(".visualization.plotting", "plot_c2_heatmaps")
get_plot_config = _LazyLoader(".visualization.plotting", "get_plot_config")
EnhancedPlottingManager = _LazyLoader(
    ".visualization.enhanced_plotting", "EnhancedPlottingManager"
)

# Performance monitoring - lazy loaded
PerformanceMonitor = _LazyLoader(".performance", "PerformanceMonitor")

# Configuration utilities - lazy loaded
get_template_path = _LazyLoader(".config", "get_template_path")
get_config_dir = _LazyLoader(".config", "get_config_dir")
TEMPLATE_FILES = _LazyLoader(".config", "TEMPLATE_FILES")

# Performance and monitoring utilities
def get_import_performance_report() -> dict[str, Any]:
    """
    Get comprehensive performance report for package imports.

    Returns
    -------
    dict[str, Any]
        Performance metrics including load times and success rates
    """
    from .core.lazy_imports import get_import_performance_report
    return get_import_performance_report()

def preload_scientific_dependencies() -> None:
    """
    Preload critical scientific computing dependencies.

    This can reduce latency for first-time usage of heavy computational modules.
    """
    from .core.lazy_imports import preload_critical_dependencies
    preload_critical_dependencies()

def configure_logging() -> None:
    """Configure logging for the homodyne package."""
    import logging
    import os

    # Set logging level from environment or default to INFO
    log_level = os.environ.get("HOMODYNE_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def optimize_initialization() -> dict[str, Any]:
    """
    Optimize package initialization for better startup performance.

    Returns
    -------
    dict[str, Any]
        Optimization strategy and performance metrics
    """
    from .core.initialization_optimizer import optimize_package_initialization
    strategy = optimize_package_initialization()
    return {
        "strategy": {
            "core_modules": strategy.core_modules,
            "lazy_modules": strategy.lazy_modules,
            "deferred_modules": strategy.deferred_modules,
            "preload_modules": strategy.preload_modules,
            "optimization_level": strategy.optimization_level,
        }
    }

def get_startup_performance_report() -> dict[str, Any]:
    """
    Get comprehensive startup performance report.

    Returns
    -------
    dict[str, Any]
        Startup performance metrics and analysis
    """
    from .core.initialization_optimizer import profile_startup_performance
    return profile_startup_performance()

def establish_performance_baseline(name: str,
                                 target_import_time: float = 2.0) -> dict[str, Any]:
    """
    Establish a performance baseline for startup monitoring.

    Parameters
    ----------
    name : str
        Baseline name
    target_import_time : float
        Target import time in seconds

    Returns
    -------
    dict[str, Any]
        Created baseline information
    """
    from .performance.simple_monitoring import create_performance_baseline
    return create_performance_baseline(name=name, target_time=target_import_time)

def check_performance_health() -> dict[str, Any]:
    """
    Quick performance health check.

    Returns
    -------
    dict[str, Any]
        Performance health status
    """
    from .performance.simple_monitoring import quick_startup_check
    return quick_startup_check()

def monitor_startup_performance(iterations: int = 5) -> dict[str, Any]:
    """
    Monitor current startup performance.

    Parameters
    ----------
    iterations : int
        Number of measurement iterations

    Returns
    -------
    dict[str, Any]
        Startup performance metrics
    """
    from .performance.simple_monitoring import measure_current_startup_performance
    return measure_current_startup_performance(iterations=iterations)

def get_performance_trend_report(days: int = 30) -> dict[str, Any]:
    """
    Get performance trend analysis.

    Parameters
    ----------
    days : int
        Number of days to analyze

    Returns
    -------
    dict[str, Any]
        Performance trend report
    """
    from .performance.startup_monitoring import get_startup_monitor
    monitor = get_startup_monitor()
    return monitor.get_performance_trend(days=days)

# Apply initialization optimizations automatically if enabled
def _apply_startup_optimizations() -> None:
    """Apply startup optimizations if enabled by environment variable."""
    import os
    if os.environ.get("HOMODYNE_OPTIMIZE_STARTUP", "true").lower() in ("true", "1", "yes"):
        try:
            optimizer = get_initialization_optimizer()
            optimizer.optimize_initialization_order()
            optimizer.apply_optimizations()
        except Exception:
            # Silently ignore optimization failures to prevent startup issues
            pass

# Apply optimizations during package import
_apply_startup_optimizations()


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
    "check_performance_health",
    "compute_g1_correlation_numba",
    "compute_sinc_squared_numba",
    "configure_logging",
    "create_config_main",
    "create_robust_optimizer",
    "create_time_integral_matrix_numba",
    "enhanced_runner_main",
    "establish_performance_baseline",
    "get_config_dir",
    "get_import_performance_report",
    "get_performance_trend_report",
    "get_plot_config",
    "get_startup_performance_report",
    "get_template_path",
    "memory_efficient_cache",
    "monitor_startup_performance",
    "optimize_initialization",
    "performance_monitor",
    "plot_c2_heatmaps",
    "preload_scientific_dependencies",
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
