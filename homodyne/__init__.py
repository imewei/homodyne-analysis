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

# Lazy loading implementation for performance optimization
import sys
from typing import TYPE_CHECKING, Any, Optional

# Type checking imports (no runtime cost)
if TYPE_CHECKING:
    from .analysis.core import HomodyneAnalysisCore
    from .config import TEMPLATE_FILES, get_config_dir, get_template_path
    from .core.config import ConfigManager, performance_monitor
    from .optimization.classical import ClassicalOptimizer
    from .optimization.robust import RobustHomodyneOptimizer, create_robust_optimizer
    from .performance import PerformanceMonitor
    from .visualization.enhanced_plotting import EnhancedPlottingManager
    from .visualization.plotting import get_plot_config, plot_c2_heatmaps

# Essential imports only (fast loading)
from .core.config import configure_logging


# Lazy loading class for deferred imports
class _LazyLoader:
    """Lazy loader for expensive imports."""

    def __init__(self, module_name: str, class_name: str | None = None):
        self.module_name = module_name
        self.class_name = class_name
        self._cached_object = None

    def __call__(self, *args, **kwargs):
        if self._cached_object is None:
            self._load()
        if self.class_name:
            return self._cached_object(*args, **kwargs)
        return self._cached_object

    def __getattr__(self, name):
        if self._cached_object is None:
            self._load()
        return getattr(self._cached_object, name)

    def _load(self):
        try:
            # Handle relative imports properly
            if self.module_name.startswith("."):
                module = importlib.import_module(self.module_name, package="homodyne")
            else:
                module = importlib.import_module(self.module_name)

            if self.class_name:
                self._cached_object = getattr(module, self.class_name)
            else:
                self._cached_object = module
        except ImportError as e:
            import logging

            logging.getLogger(__name__).warning(
                f"Failed to load {self.module_name}: {e}"
            )
            self._cached_object = None


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
