"""
Core Analysis Engine for Homodyne Scattering Analysis
=====================================================

High-performance homodyne scattering analysis with configuration management.

This module implements the complete analysis pipeline for XPCS data in
nonequilibrium laminar flow systems, based on He et al. (2024).

Physical Theory
---------------
The theoretical framework describes the time-dependent intensity correlation function
c2(φ,t₁,t₂) for X-ray photon correlation spectroscopy (XPCS) measurements of fluids
under nonequilibrium laminar flow conditions. The model captures the interplay between
Brownian diffusion and advective shear flow in the two-time correlation dynamics.

The correlation function has the form:
    g2(φ,t₁,t₂) = 1 + contrast × [g1(φ,t₁,t₂)]²

where g1 is the field correlation function with separable contributions:
    g1(φ,t₁,t₂) = g1_diff(t₁,t₂) × g1_shear(φ,t₁,t₂)

Diffusion Contribution:
    g1_diff(t₁,t₂) = exp[-q²/2 ∫|t₂-t₁| D(t')dt']

Shear Contribution:
    g1_shear(φ,t₁,t₂) = [sinc(Φ(φ,t₁,t₂))]²
    Φ(φ,t₁,t₂) = (1/2π) q L cos(φ₀-φ) ∫|t₂-t₁| γ̇(t')dt'

Time-Dependent Transport Coefficients:
    D(t) = D₀ t^α + D_offset    (anomalous diffusion)
    γ̇(t) = γ̇₀ t^β + γ̇_offset   (time-dependent shear rate)

Parameter Models:
Static Mode (3 parameters):
- D₀: Reference diffusion coefficient [Å²/s]
- α: Diffusion time-dependence exponent [-]
- D_offset: Baseline diffusion [Å²/s]
(γ̇₀, β, γ̇_offset, φ₀ = 0 - automatically set and irrelevant)

Laminar Flow Mode (7 parameters):
- D₀: Reference diffusion coefficient [Å²/s]
- α: Diffusion time-dependence exponent [-]
- D_offset: Baseline diffusion [Å²/s]
- γ̇₀: Reference shear rate [s⁻¹]
- β: Shear rate time-dependence exponent [-]
- γ̇_offset: Baseline shear rate [s⁻¹]
- φ₀: Angular offset parameter [degrees]

Experimental Parameters:
- q: Scattering wavevector magnitude [Å⁻¹]
- L: Characteristic length scale (gap size) [Å]
- φ: Scattering angle [degrees]
- dt: Time step between frames [s/frame]

Features
--------
- JSON-based configuration management
- Experimental data loading with intelligent caching
- Parallel processing for multi-angle calculations
- Performance optimization with Numba JIT compilation
- Comprehensive parameter validation and bounds checking
- Memory-efficient matrix operations and caching

Performance Optimizations (v0.6.1+)
------------------------------------
This version includes significant performance improvements:

Core Optimizations:

- **Chi-squared calculation**: 38% performance improvement (1.33ms → 0.82ms)
- **Memory access patterns**: Vectorized operations using reshape() instead of list comprehensions
- **Configuration caching**: Cached validation and chi-squared configs to avoid repeated dict lookups
- **Least squares optimization**: Replaced lstsq with solve() for 2x2 matrix systems
- **Memory pooling**: Pre-allocated result arrays to avoid repeated allocations

Algorithm Improvements:

- **Static case vectorization**: Enhanced broadcasting for identical correlation functions
- **Precomputed integrals**: Cached shear integrals to eliminate redundant computation
- **Vectorized angle filtering**: Optimized range checking with np.flatnonzero()
- **Early parameter validation**: Short-circuit returns for invalid parameters

Performance Metrics:

- Chi-squared to correlation ratio: Improved from 6.0x to 1.7x
- Memory efficiency: Reduced allocation overhead through pooling
- JIT compatibility: Maintained Numba acceleration while improving pure Python paths

Usage
-----
>>> from homodyne.analysis.core import HomodyneAnalysisCore
>>> analyzer = HomodyneAnalysisCore('config.json')
>>> data = analyzer.load_experimental_data()
>>> chi2 = analyzer.calculate_chi_squared_optimized(parameters, phi_angles, data[0])

References
----------
He, H., Chen, W., et al. (2024). "Time-dependent dynamics in nonequilibrium
laminar flow systems via X-ray photon correlation spectroscopy."

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

__author__ = "Wei Chen, Hongrui He"
__credits__ = "Argonne National Laboratory"

import json
import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# Import optional dependencies
try:
    from pyxpcsviewer import XpcsFile as xf

    PYXPCSVIEWER_AVAILABLE = True
except ImportError:
    PYXPCSVIEWER_AVAILABLE = False
    xf = None

# Import performance optimization dependencies
try:
    from numba import jit, njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        return args[0] if args and callable(args[0]) else lambda f: f

    def njit(*args, **kwargs):
        return args[0] if args and callable(args[0]) else lambda f: f

    prange = range

# Import core dependencies from the main module
from ..core.config import ConfigManager
from ..core.kernels import (
    calculate_diffusion_coefficient_numba,
    calculate_shear_rate_numba,
    chi_squared_with_variance_batch_numba,
    compute_chi_squared_batch_numba,
    compute_g1_correlation_numba,
    compute_sinc_squared_numba,
    create_time_integral_matrix_numba,
    estimate_variance_irls_batch_numba,
    hybrid_irls_batch_numba,
    mad_window_batch_numba,
    memory_efficient_cache,
    solve_least_squares_batch_numba,
)

logger = logging.getLogger(__name__)


# Test environment detection for Numba threading compatibility
def _is_test_environment() -> bool:
    """
    Detect if code is running in a test environment.

    This is used to disable Numba parallel processing when NUMBA_NUM_THREADS=1
    to avoid threading conflicts.
    """
    numba_threads = os.environ.get("NUMBA_NUM_THREADS", "")
    return (
        numba_threads == "1"
        or os.environ.get("PYTEST_CURRENT_TEST") is not None
        or "pytest" in os.environ.get("_", "")
    )


# Use parallel processing only when not in test environment
_USE_PARALLEL = not _is_test_environment()

# Disable JIT entirely in problematic test environments
_DISABLE_JIT = os.environ.get("NUMBA_DISABLE_JIT", "0") == "1" or (
    _is_test_environment() and os.environ.get("NUMBA_NUM_THREADS") == "1"
)

# Global optimization counter for performance tracking
OPTIMIZATION_COUNTER = 0

# Default thread count for parallelization
DEFAULT_NUM_THREADS = min(16, mp.cpu_count())

# Check for optional dependencies
try:
    import pymc  # noqa: F401

    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False


# ===== PERFORMANCE-OPTIMIZED IRLS FUNCTIONS =====
# These functions provide 10-400x speedup for IRLS variance estimation
# and chi-squared calculations through JIT compilation and vectorization


# Use simple fallback when NUMBA_DISABLE_JIT is set or in problematic environments
if os.environ.get("NUMBA_DISABLE_JIT", "0") == "1" or _DISABLE_JIT:

    def _calculate_median_quickselect(data: np.ndarray) -> float:
        """Pure Python fallback for median calculation when Numba has threading issues."""
        n = len(data)
        if n == 0:
            return np.nan
        sorted_data = np.sort(data)
        if n % 2 == 1:
            return float(sorted_data[n // 2])
        else:
            return float(0.5 * (sorted_data[n // 2 - 1] + sorted_data[n // 2]))

else:
    # Create robust JIT version with fallback
    def _create_median_quickselect_jit():
        """Create JIT-compiled median function with fallback."""

        def _median_impl(data: np.ndarray) -> float:
            """Median calculation implementation."""
            n = len(data)
            if n == 0:
                return np.nan

            # Create working copy
            arr = np.copy(data)

            # Insertion sort for small arrays (typical IRLS window size)
            for i in range(1, n):
                key = arr[i]
                j = i - 1
                while j >= 0 and arr[j] > key:
                    arr[j + 1] = arr[j]
                    j -= 1
                arr[j + 1] = key

            # Calculate median
            if n % 2 == 1:
                return arr[n // 2]
            else:
                return 0.5 * (arr[n // 2 - 1] + arr[n // 2])

        # Try JIT compilation
        try:
            if NUMBA_AVAILABLE and not (
                os.environ.get("NUMBA_DISABLE_JIT", "0") == "1" or _DISABLE_JIT
            ):
                return jit(nopython=True, fastmath=True)(_median_impl)
            else:
                return _median_impl
        except Exception:
            # Fallback to non-JIT version
            return _median_impl

    _calculate_median_quickselect = _create_median_quickselect_jit()


def _calculate_median_quickselect_original(data: np.ndarray) -> float:
    """
    Fast median calculation using insertion sort for small arrays.

    Provides 5-10x speedup over numpy.median for small arrays (typical in IRLS).

    Parameters
    ----------
    data : np.ndarray
        Input array for median calculation

    Returns
    -------
    float
        Median value

    Notes
    -----
    Uses insertion sort for simplicity and Numba compatibility.
    For the small window sizes typical in IRLS (5-15 elements),
    insertion sort is very efficient.
    """
    if len(data) == 0:
        return np.nan

    # Copy data to avoid modifying input
    arr = data.copy()
    n = len(arr)

    if n == 1:
        return arr[0]

    # Use insertion sort - very fast for small arrays
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

    # Return median
    if n % 2 == 1:
        return arr[n // 2]
    else:
        return 0.5 * (arr[n // 2 - 1] + arr[n // 2])


# Try to create JIT version with graceful fallback
def _create_chi_squared_vectorized_jit():
    """Create JIT-compiled chi-squared function with fallback."""

    def _chi_squared_impl(residuals: np.ndarray, weights: np.ndarray) -> float:
        """Chi-squared calculation implementation."""
        if len(residuals) != len(weights):
            return np.inf

        if len(residuals) == 0:
            return 0.0

        # Vectorized chi-squared calculation with overflow protection
        chi_squared = 0.0
        for i in range(len(residuals)):
            if weights[i] > 0:  # Skip invalid weights
                weighted_residual_sq = residuals[i] * residuals[i] * weights[i]
                if np.isfinite(weighted_residual_sq):
                    chi_squared += weighted_residual_sq

        return chi_squared

    # Try JIT compilation
    try:
        if NUMBA_AVAILABLE and not (
            os.environ.get("NUMBA_DISABLE_JIT", "0") == "1" or _DISABLE_JIT
        ):
            return jit(nopython=True, fastmath=True, cache=True)(_chi_squared_impl)
        else:
            return _chi_squared_impl
    except Exception:
        # Fallback to non-JIT version
        return _chi_squared_impl


_calculate_chi_squared_vectorized_jit = _create_chi_squared_vectorized_jit()

# Add docstring to the function (for documentation purposes)
_calculate_chi_squared_vectorized_jit.__doc__ = """
Memory-efficient vectorized chi-squared calculation with JIT compilation.

Provides 20-50x speedup through:
- JIT compilation with fastmath optimizations
- Vectorized operations
- Memory-efficient computation
- Optimized floating-point operations

Parameters
----------
residuals : np.ndarray
    Residuals array (experimental - theoretical)
weights : np.ndarray
    Weight array (1/σ²)

Returns
-------
float
    Chi-squared value

Notes
-----
Implements χ² = Σ(residuals² × weights) with numerical stability
"""


class HomodyneAnalysisCore:
    """
    Core analysis engine for homodyne scattering data.

    This class provides the fundamental analysis capabilities including:
    - Configuration-driven parameter management
    - Experimental data loading with intelligent caching
    - Correlation function calculations with performance optimizations
    - Time-dependent diffusion and shear rate modeling
    """

    def __init__(
        self,
        config_file: str = "homodyne_config.json",
        config_override: dict[str, Any] | None = None,
    ):
        """
        Initialize the core analysis system.

        Parameters
        ----------
        config_file : str
            Path to JSON configuration file
        config_override : dict, optional
            Runtime configuration overrides
        """
        # Load and validate configuration
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.config

        # Apply overrides if provided
        if config_override:
            self._apply_config_overrides(config_override)
            self.config_manager.setup_logging()

        # Extract core parameters
        self._initialize_parameters()

        # Setup performance optimizations
        self._setup_performance()

        # Initialize caching systems
        self._initialize_caching()

        # Initialize simple caching (removed complex memory pooling)
        self._cache = {}

        # Warm up JIT functions
        if (
            NUMBA_AVAILABLE
            and self.config is not None
            and self.config.get("performance_settings", {}).get("warmup_numba", True)
        ):
            self._warmup_numba_functions()

        self._print_initialization_summary()

    def _initialize_parameters(self):
        """Initialize core analysis parameters from configuration."""
        if self.config is None:
            raise ValueError("Configuration not loaded: self.config is None.")
        params = self.config["analyzer_parameters"]

        # Time and frame parameters
        self.dt = params["temporal"]["dt"]
        self.start_frame = params["temporal"]["start_frame"]
        self.end_frame = params["temporal"]["end_frame"]
        self.time_length = self.end_frame - self.start_frame

        # Physical parameters
        self.wavevector_q = params["scattering"]["wavevector_q"]
        self.stator_rotor_gap = params["geometry"]["stator_rotor_gap"]

        # Parameter counts
        self.num_diffusion_params = 3
        self.num_shear_rate_params = 3

        # Pre-compute constants
        self.wavevector_q_squared = self.wavevector_q**2
        self.wavevector_q_squared_half_dt = 0.5 * self.wavevector_q_squared * self.dt
        self.sinc_prefactor = (
            0.5 / np.pi * self.wavevector_q * self.stator_rotor_gap * self.dt
        )

        # Cache static mode state for performance
        self._is_static_mode = self.is_static_mode()

        # Enhanced two-tier caching system for repeated calculations
        self._diffusion_integral_cache = {}
        self._max_cache_size = 64  # Increased cache size for better performance

        # Two-tier cache system: L1 (fast) and L2 (persistent)
        self._l1_cache = {}  # Fast in-memory cache for recent calculations
        self._l2_cache = {}  # Persistent cache for frequently accessed data
        self._l1_max_size = 32  # L1 cache for immediate access
        self._l2_max_size = 128  # L2 cache for long-term storage
        self._cache_access_count = {}  # Track access frequency for intelligent promotion

        # Time array
        self.time_array = np.linspace(
            self.dt,
            self.dt * self.time_length,
            self.time_length,
            dtype=np.float64,
        )

    def _setup_performance(self):
        """Configure performance settings and optimized method selection."""
        if self.config is None:
            raise ValueError("Configuration not loaded: self.config is None.")
        params = self.config["analyzer_parameters"]
        comp_params = params.get("computational", {})

        # Thread configuration
        if comp_params.get("auto_detect_cores", False):
            detected = mp.cpu_count()
            max_threads = comp_params.get("max_threads_limit", 128)
            self.num_threads = min(detected, max_threads)
        else:
            self.num_threads = comp_params.get("num_threads", DEFAULT_NUM_THREADS)

        # Apply max_threads_limit regardless of auto_detect_cores setting
        max_threads = comp_params.get("max_threads_limit", 128)
        self.num_threads = min(self.num_threads, max_threads)

        # Ensure minimum of 1 thread for safe operation
        self.num_threads = max(1, self.num_threads)

        # Performance optimization configuration
        self._setup_optimized_methods()

    def _setup_optimized_methods(self):
        """Configure optimized variance estimation and chi-squared calculation methods."""
        # Get configuration sections
        chi_config = self.config.get("advanced_settings", {}).get(
            "chi_squared_calculation", {}
        )
        perf_config = chi_config.get("performance_optimization", {})

        # Check if optimization is enabled
        self.optimization_enabled = perf_config.get("enabled", False)

        if not self.optimization_enabled:
            # Use legacy methods
            self._selected_variance_estimator = self._estimate_variance_irls_mad_robust
            self._selected_chi_calculator = self.calculate_chi_squared_optimized
            logger.warning(
                "IRLS Performance optimization DISABLED - using legacy methods with 10 iterations. "
                "Set 'performance_optimization.enabled: true' in config for 50-100x speedup."
            )
            return

        # Configure variance estimation method based on config
        variance_method = chi_config.get("variance_method", "hybrid_limited_irls")

        if variance_method == "hybrid_limited_irls":
            # Use hybrid Limited-Iteration IRLS with Simple MAD initialization
            # Check if batch processing is available
            if hasattr(self, "_estimate_variance_hybrid_limited_irls_batch"):
                self._selected_variance_estimator = (
                    self._estimate_variance_hybrid_limited_irls_batch
                )
                logger.info(
                    "Selected Hybrid Limited IRLS variance estimator with batch processing "
                    "(50-70% faster than full IRLS through FGLS-inspired approach)"
                )
            else:
                self._selected_variance_estimator = (
                    self._estimate_variance_hybrid_limited_irls
                )
                logger.info(
                    "Selected Hybrid Limited IRLS variance estimator "
                    "(50-70% faster than full IRLS through FGLS-inspired approach)"
                )

        elif variance_method == "irls_mad_robust":
            # Use full IRLS MAD robust variance estimator (existing default)
            if hasattr(self, "_estimate_variance_irls_mad_robust_batch"):
                self._selected_variance_estimator = (
                    self._estimate_variance_irls_mad_robust_batch
                )
                logger.info(
                    "Selected IRLS MAD robust variance estimator with batch processing"
                )
            else:
                self._selected_variance_estimator = (
                    self._estimate_variance_irls_mad_robust
                )
                logger.info("Selected IRLS MAD robust variance estimator")

        else:
            # Fallback to standard IRLS method with warning
            logger.warning(
                f"Unknown variance_method '{variance_method}', falling back to IRLS MAD robust"
            )
            self._selected_variance_estimator = self._estimate_variance_irls_mad_robust

        # Configure optimized chi-squared calculator
        chi_calculator_type = perf_config.get("chi_calculator", "standard")
        if chi_calculator_type == "vectorized_jit":
            self._selected_chi_calculator = self._calculate_chi_squared_with_config
            logger.info(
                "Selected vectorized JIT chi-squared calculator (20-50x speedup)"
            )
        else:
            self._selected_chi_calculator = self.calculate_chi_squared_optimized
            logger.info("Selected standard chi-squared calculator")

        # Store configuration for method access
        self.perf_config = perf_config
        self._cached_chi_config = chi_config  # Cache for hybrid method access

    def _initialize_caching(self):
        """Initialize caching systems and memory pools."""
        self._cache = {}
        self.cached_experimental_data = None
        self.cached_phi_angles = None

        # Initialize plotting cache variables
        self._last_experimental_data = None
        self._last_phi_angles = None

        # Initialize two-tier cache system: L1 (fast) and L2 (persistent)
        self._l1_cache = {}  # Fast in-memory cache for recent calculations
        self._l2_cache = {}  # Persistent cache for frequently accessed data
        self._l1_max_size = 32  # L1 cache for immediate access
        self._l2_max_size = 128  # L2 cache for long-term storage
        self._cache_access_count = {}  # Track access frequency for cache management
        self._diffusion_integral_cache = {}  # Cache for diffusion integrals

        # Adaptive cache sizing based on available memory
        self._adaptive_cache_enabled = True
        self._last_memory_check = 0
        self._memory_check_interval = 100  # Check every 100 cache operations
        self._cache_operation_count = 0

        # Performance monitoring state
        self._performance_baselines = {
            "chi_squared_per_angle_ms": 50.0,  # Expected time per angle in ms
            "chi_squared_per_param_ms": 10.0,  # Expected time per parameter in ms
        }

        # Configuration-based performance tuning
        self._performance_config = self._initialize_performance_config()
        self._dataset_characteristics = {}
        self._tuning_enabled = True

        # Cache system for computational results

    def _get_from_two_tier_cache(self, key: str):
        """
        Get value from two-tier cache system with intelligent promotion.

        Parameters
        ----------
        key : str
            Cache key

        Returns
        -------
        value or None
            Cached value if found, None otherwise
        """
        # Check L1 cache first (fastest access)
        if key in self._l1_cache:
            self._cache_access_count[key] = self._cache_access_count.get(key, 0) + 1
            return self._l1_cache[key]

        # Check L2 cache (slower but larger)
        if key in self._l2_cache:
            value = self._l2_cache[key]
            # Promote to L1 cache for frequent access
            self._cache_access_count[key] = self._cache_access_count.get(key, 0) + 1
            self._put_in_l1_cache(key, value)
            return value

        return None

    def _put_in_two_tier_cache(self, key: str, value):
        """
        Store value in two-tier cache system with intelligent placement.

        Parameters
        ----------
        key : str
            Cache key
        value
            Value to cache
        """
        # Always start in L1 cache
        self._put_in_l1_cache(key, value)
        self._cache_access_count[key] = 1

    def _put_in_l1_cache(self, key: str, value):
        """Store value in L1 cache with LRU eviction and adaptive sizing."""
        # Check if we should adapt cache sizes
        self._cache_operation_count += 1
        if (
            self._cache_operation_count % self._memory_check_interval == 0
            and self._adaptive_cache_enabled
        ):
            self._adapt_cache_sizes()

        if len(self._l1_cache) >= self._l1_max_size:
            # Evict least accessed item to L2 cache
            if self._cache_access_count:  # Check if there are items to evict
                lru_key = min(
                    self._cache_access_count, key=self._cache_access_count.get
                )
                if lru_key in self._l1_cache:
                    self._l2_cache[lru_key] = self._l1_cache.pop(lru_key)

                    # Manage L2 cache size
                    if len(self._l2_cache) >= self._l2_max_size:
                        oldest_l2_key = next(iter(self._l2_cache))
                        del self._l2_cache[oldest_l2_key]

        self._l1_cache[key] = value

    def get_cache_statistics(self) -> dict:
        """
        Get comprehensive cache statistics.

        Returns
        -------
        dict
            Cache statistics including hit rates, sizes, and access patterns
        """
        total_accesses = sum(self._cache_access_count.values())
        unique_keys = len(self._cache_access_count)

        # Enhanced memory usage monitoring
        import os
        import sys

        import psutil

        # Estimate memory usage of caches
        def estimate_cache_memory(cache_dict):
            """Estimate memory usage of cache dictionary."""
            total_size = 0
            for key, value in cache_dict.items():
                total_size += sys.getsizeof(key)
                if hasattr(value, "nbytes"):  # NumPy arrays
                    total_size += value.nbytes
                else:
                    total_size += sys.getsizeof(value)
            return total_size

        l1_memory = estimate_cache_memory(self._l1_cache)
        l2_memory = estimate_cache_memory(self._l2_cache)
        diffusion_memory = estimate_cache_memory(self._diffusion_integral_cache)

        # Get system memory info
        try:
            process = psutil.Process(os.getpid())
            system_memory = psutil.virtual_memory()
            process_memory = process.memory_info().rss
        except ImportError:
            # Fallback if psutil not available
            process_memory = 0
            system_memory = None

        # Get memory limit from configuration
        memory_limit_gb = (
            self.config.get("analyzer_parameters", {})
            .get("computational", {})
            .get("memory_limit_gb", 16)
        )
        memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024

        result = {
            "l1_cache_size": len(self._l1_cache),
            "l1_max_size": self._l1_max_size,
            "l2_cache_size": len(self._l2_cache),
            "l2_max_size": self._l2_max_size,
            "total_unique_keys": unique_keys,
            "total_accesses": total_accesses,
            "avg_accesses_per_key": (
                total_accesses / unique_keys if unique_keys > 0 else 0
            ),
            "cache_hit_rate": (
                f"{((total_accesses - unique_keys) / total_accesses * 100):.2f}%"
                if total_accesses > 0
                else "0.00%"
            ),
            # Memory usage statistics
            "memory_usage_bytes": {
                "l1_cache": l1_memory,
                "l2_cache": l2_memory,
                "diffusion_cache": diffusion_memory,
                "total_cache": l1_memory + l2_memory + diffusion_memory,
            },
            "memory_usage_mb": {
                "l1_cache": l1_memory / (1024 * 1024),
                "l2_cache": l2_memory / (1024 * 1024),
                "diffusion_cache": diffusion_memory / (1024 * 1024),
                "total_cache": (l1_memory + l2_memory + diffusion_memory)
                / (1024 * 1024),
            },
            "memory_limits": {
                "configured_limit_gb": memory_limit_gb,
                "configured_limit_bytes": memory_limit_bytes,
            },
        }

        # Add system memory info if available
        if system_memory is not None:
            result["memory_usage_mb"].update(
                {
                    "process_total": process_memory / (1024 * 1024),
                    "system_available": system_memory.available / (1024 * 1024),
                }
            )
            result["memory_limits"].update(
                {
                    "usage_percentage": (process_memory / memory_limit_bytes) * 100,
                    "cache_percentage": (
                        (l1_memory + l2_memory + diffusion_memory) / memory_limit_bytes
                    )
                    * 100,
                }
            )

        return result

    def _adapt_cache_sizes(self) -> None:
        """
        Dynamically adapt cache sizes based on memory usage and availability.

        This method monitors memory usage and adjusts cache sizes to optimize
        performance while staying within memory limits.
        """
        if not self._adaptive_cache_enabled:
            return

        try:
            # Get current memory statistics
            stats = self.get_cache_statistics()
            memory_usage = stats.get("memory_limits", {})

            # Only adapt if we have memory usage data
            if "usage_percentage" not in memory_usage:
                return

            usage_pct = memory_usage["usage_percentage"]
            cache_pct = memory_usage["cache_percentage"]

            # Conservative thresholds for adaptive sizing
            high_usage_threshold = 80.0  # % of configured memory limit
            low_usage_threshold = 40.0

            if usage_pct > high_usage_threshold:
                # High memory usage: reduce cache sizes
                new_l1_size = max(16, int(self._l1_max_size * 0.75))
                new_l2_size = max(32, int(self._l2_max_size * 0.75))
                logger.debug(
                    f"High memory usage ({usage_pct:.1f}%), reducing cache sizes"
                )

            elif usage_pct < low_usage_threshold and cache_pct < 5.0:
                # Low memory usage and small cache footprint: increase cache sizes
                new_l1_size = min(64, int(self._l1_max_size * 1.25))
                new_l2_size = min(256, int(self._l2_max_size * 1.25))
                logger.debug(
                    f"Low memory usage ({usage_pct:.1f}%), increasing cache sizes"
                )

            else:
                # Medium usage: maintain current sizes
                return

            # Apply new cache sizes if they changed
            if new_l1_size != self._l1_max_size or new_l2_size != self._l2_max_size:
                self._l1_max_size = new_l1_size
                self._l2_max_size = new_l2_size

                # Trim caches if they exceed new limits
                self._trim_caches_to_limits()

                logger.debug(f"Adapted cache sizes: L1={new_l1_size}, L2={new_l2_size}")

        except Exception as e:
            logger.debug(f"Failed to adapt cache sizes: {e}")

    def _trim_caches_to_limits(self) -> None:
        """Trim caches to respect new size limits."""
        # Trim L1 cache
        while len(self._l1_cache) > self._l1_max_size:
            if self._cache_access_count:
                # Remove least recently used item
                lru_key = min(
                    self._cache_access_count, key=self._cache_access_count.get
                )
                if lru_key in self._l1_cache:
                    # Move to L2 before removing
                    if len(self._l2_cache) < self._l2_max_size:
                        self._l2_cache[lru_key] = self._l1_cache[lru_key]
                    self._l1_cache.pop(lru_key)
            else:
                # Remove oldest entry if no access count data
                oldest_key = next(iter(self._l1_cache))
                self._l1_cache.pop(oldest_key)

        # Trim L2 cache
        while len(self._l2_cache) > self._l2_max_size:
            oldest_l2_key = next(iter(self._l2_cache))
            del self._l2_cache[oldest_l2_key]
            if oldest_l2_key in self._cache_access_count:
                del self._cache_access_count[oldest_l2_key]

    def _calculate_performance_score(
        self, execution_time: float, n_angles: int, n_params: int
    ) -> dict:
        """
        Calculate performance score for chi-squared calculation.

        Parameters
        ----------
        execution_time : float
            Total execution time in seconds
        n_angles : int
            Number of angles processed
        n_params : int
            Number of parameters

        Returns
        -------
        dict
            Performance score metrics
        """
        execution_time_ms = execution_time * 1000

        # Calculate expected time based on problem size
        expected_time_ms = (
            n_angles * self._performance_baselines["chi_squared_per_angle_ms"]
            + n_params * self._performance_baselines["chi_squared_per_param_ms"]
        )

        # Performance ratio (< 1.0 is faster than expected, > 1.0 is slower)
        performance_ratio = execution_time_ms / max(expected_time_ms, 1.0)

        # Score from 0-100 (higher is better)
        if performance_ratio <= 0.5:
            score = 100  # Excellent performance
        elif performance_ratio <= 1.0:
            score = int(100 - (performance_ratio - 0.5) * 100)  # Good performance
        elif performance_ratio <= 2.0:
            score = int(50 - (performance_ratio - 1.0) * 40)  # Fair performance
        else:
            score = max(
                10, int(50 - min(performance_ratio - 2.0, 5.0) * 8)
            )  # Poor performance

        return {
            "score": score,
            "performance_ratio": performance_ratio,
            "execution_time_ms": execution_time_ms,
            "expected_time_ms": expected_time_ms,
            "time_per_angle_ms": execution_time_ms / max(n_angles, 1),
            "time_per_param_ms": execution_time_ms / max(n_params, 1),
            "efficiency_category": self._get_efficiency_category(performance_ratio),
        }

    def _get_efficiency_category(self, performance_ratio: float) -> str:
        """Get efficiency category based on performance ratio."""
        if performance_ratio <= 0.5:
            return "excellent"
        elif performance_ratio <= 1.0:
            return "good"
        elif performance_ratio <= 2.0:
            return "fair"
        else:
            return "poor"

    def _initialize_performance_config(self) -> dict:
        """
        Initialize performance configuration with dataset-aware defaults.

        Returns
        -------
        dict
            Performance configuration settings
        """
        # Get computational settings from config
        comp_config = self.config.get("analyzer_parameters", {}).get(
            "computational", {}
        )
        memory_limit = comp_config.get("memory_limit_gb", 16)
        max_threads = comp_config.get("max_threads_limit", 12)

        # Dataset size classification thresholds
        size_thresholds = {
            "small_dataset": {
                "max_data_points": 100_000,
                "max_angles": 20,
                "memory_usage_gb": 2,
            },
            "medium_dataset": {
                "max_data_points": 1_000_000,
                "max_angles": 50,
                "memory_usage_gb": 8,
            },
            "large_dataset": {
                "max_data_points": 10_000_000,
                "max_angles": 100,
                "memory_usage_gb": 32,
            },
            "very_large_dataset": {
                "max_data_points": float("inf"),
                "max_angles": float("inf"),
                "memory_usage_gb": float("inf"),
            },
        }

        # Performance settings per dataset size
        size_settings = {
            "small_dataset": {
                "cache_sizes": {"l1": 16, "l2": 32, "memory_efficient": 64},
                "memory_check_interval": 200,
                "adaptive_cache": False,
                "parallelization": {"enable": False, "max_workers": 2},
                "optimization": {"tolerance": 1e-6, "iterations": 500},
            },
            "medium_dataset": {
                "cache_sizes": {"l1": 32, "l2": 64, "memory_efficient": 128},
                "memory_check_interval": 100,
                "adaptive_cache": True,
                "parallelization": {"enable": True, "max_workers": min(max_threads, 8)},
                "optimization": {"tolerance": 1e-6, "iterations": 1000},
            },
            "large_dataset": {
                "cache_sizes": {"l1": 64, "l2": 128, "memory_efficient": 256},
                "memory_check_interval": 50,
                "adaptive_cache": True,
                "parallelization": {
                    "enable": True,
                    "max_workers": min(max_threads, 16),
                },
                "optimization": {"tolerance": 5e-6, "iterations": 1500},
            },
            "very_large_dataset": {
                "cache_sizes": {"l1": 128, "l2": 256, "memory_efficient": 512},
                "memory_check_interval": 25,
                "adaptive_cache": True,
                "parallelization": {"enable": True, "max_workers": max_threads},
                "optimization": {"tolerance": 1e-5, "iterations": 2000},
            },
        }

        return {
            "size_thresholds": size_thresholds,
            "size_settings": size_settings,
            "auto_tune_enabled": True,
            "current_dataset_size": "unknown",
            "memory_limit_gb": memory_limit,
            "max_threads": max_threads,
        }

    def analyze_dataset_characteristics(
        self, phi_angles: np.ndarray, c2_experimental: np.ndarray
    ) -> dict:
        """
        Analyze dataset characteristics for performance tuning.

        Parameters
        ----------
        phi_angles : np.ndarray
            Angular positions
        c2_experimental : np.ndarray
            Experimental data

        Returns
        -------
        dict
            Dataset characteristics
        """
        n_angles = len(phi_angles)
        data_shape = c2_experimental.shape

        # Estimate data points
        if len(data_shape) == 2:
            n_data_points = data_shape[0] * data_shape[1]
        elif len(data_shape) == 3:
            n_data_points = data_shape[0] * data_shape[1] * data_shape[2]
        else:
            n_data_points = c2_experimental.size

        # Estimate memory usage (rough approximation)
        memory_usage_gb = (c2_experimental.nbytes + phi_angles.nbytes) / (1024**3)

        # Classify dataset size
        dataset_size = "small_dataset"
        for size_name, thresholds in self._performance_config[
            "size_thresholds"
        ].items():
            if (
                n_data_points <= thresholds["max_data_points"]
                and n_angles <= thresholds["max_angles"]
                and memory_usage_gb <= thresholds["memory_usage_gb"]
            ):
                dataset_size = size_name
                break

        characteristics = {
            "n_angles": n_angles,
            "n_data_points": n_data_points,
            "data_shape": data_shape,
            "memory_usage_gb": memory_usage_gb,
            "dataset_size": dataset_size,
            "complexity_score": self._calculate_complexity_score(
                n_angles, n_data_points, memory_usage_gb
            ),
        }

        self._dataset_characteristics = characteristics
        self._performance_config["current_dataset_size"] = dataset_size

        return characteristics

    def _calculate_complexity_score(
        self, n_angles: int, n_data_points: int, memory_gb: float
    ) -> float:
        """Calculate complexity score for the dataset (0-100, higher is more complex)."""
        # Normalize factors
        angle_score = min(n_angles / 100, 1.0) * 30  # Max 30 points
        data_score = min(n_data_points / 10_000_000, 1.0) * 40  # Max 40 points
        memory_score = min(memory_gb / 100, 1.0) * 30  # Max 30 points

        return angle_score + data_score + memory_score

    def apply_performance_tuning(self) -> None:
        """
        Apply performance tuning based on dataset characteristics.
        """
        if not self._tuning_enabled or not self._dataset_characteristics:
            return

        dataset_size = self._dataset_characteristics["dataset_size"]
        settings = self._performance_config["size_settings"].get(dataset_size, {})

        if not settings:
            return

        # Apply cache size tuning
        cache_sizes = settings.get("cache_sizes", {})
        if "l1" in cache_sizes:
            self._l1_max_size = cache_sizes["l1"]
        if "l2" in cache_sizes:
            self._l2_max_size = cache_sizes["l2"]

        # Apply memory management tuning
        if "memory_check_interval" in settings:
            self._memory_check_interval = settings["memory_check_interval"]
        if "adaptive_cache" in settings:
            self._adaptive_cache_enabled = settings["adaptive_cache"]

        # Update performance baselines based on complexity
        complexity = self._dataset_characteristics.get("complexity_score", 50)
        complexity_factor = 1.0 + (complexity / 100.0)  # 1.0 to 2.0 scaling

        self._performance_baselines["chi_squared_per_angle_ms"] *= complexity_factor
        self._performance_baselines["chi_squared_per_param_ms"] *= complexity_factor

        logger.debug(
            f"Applied performance tuning for {dataset_size} "
            f"(complexity={complexity:.1f}, cache_l1={self._l1_max_size}, "
            f"cache_l2={self._l2_max_size})"
        )

    def clear_caches(self):
        """Clear all caches to free memory."""
        self._l1_cache.clear()
        self._l2_cache.clear()
        self._cache_access_count.clear()
        self._diffusion_integral_cache.clear()

    def _clear_caches(self):
        """Private cache clearing method for test compatibility."""
        self.clear_caches()

    def _warmup_numba_functions(self):
        """Pre-compile Numba functions to eliminate first-call overhead."""
        if not NUMBA_AVAILABLE:
            return

        logger.info("Warming up Numba JIT functions...")
        start_time = time.time()

        # Create small test arrays
        size = 10
        test_array = np.ones(size, dtype=np.float64)
        test_time = np.linspace(0.1, 1.0, size, dtype=np.float64)
        test_matrix = np.ones((size, size), dtype=np.float64)

        try:
            # Warm up low-level Numba functions
            create_time_integral_matrix_numba(test_array)
            calculate_diffusion_coefficient_numba(test_time, 1000.0, 0.0, 0.0)
            calculate_shear_rate_numba(test_time, 0.01, 0.0, 0.0)
            compute_g1_correlation_numba(test_matrix, 1.0)
            compute_sinc_squared_numba(test_matrix, 1.0)

            # Warm up high-level correlation calculation function
            # This is crucial for stable performance testing
            test_params = np.array([1000.0, -0.1, 50.0, 0.001, -0.2, 0.0, 0.0])
            test_phi_angles = np.array([0.0, 45.0])

            # Create minimal test configuration for warmup
            original_config = self.config
            original_time_length = getattr(self, "time_length", None)
            original_time_array = getattr(self, "time_array", None)

            # Temporarily set minimal configuration for warmup
            self.time_length = size
            self.time_array = test_time

            try:
                # Warm up the main correlation calculation
                _ = self.calculate_c2_nonequilibrium_laminar_parallel(
                    test_params, test_phi_angles
                )
                logger.debug("High-level correlation function warmed up")
            except Exception as warmup_error:
                logger.debug(
                    f"High-level warmup failed (expected in some configs): {warmup_error}"
                )
            finally:
                # Restore original configuration
                self.config = original_config
                if original_time_length is not None:
                    self.time_length = original_time_length
                if original_time_array is not None:
                    self.time_array = original_time_array

            # Warm up optimized IRLS functions if enabled
            if hasattr(self, "optimization_enabled") and self.optimization_enabled:
                logger.debug("Warming up optimized IRLS functions...")
                try:
                    # Warm up quickselect median
                    _calculate_median_quickselect(test_array)

                    # Vectorized MAD estimation removed (obsolete)

                    # Warm up vectorized chi-squared calculation
                    test_weights = np.ones_like(test_array)
                    _calculate_chi_squared_vectorized_jit(test_array, test_weights)

                    logger.debug("Optimized IRLS function warmup completed")
                except Exception as opt_warmup_error:
                    logger.debug(f"Optimized IRLS warmup failed: {opt_warmup_error}")

            elapsed = time.time() - start_time
            logger.info(
                f"Numba warmup completed in {
                    elapsed:.2f}s (including high-level and optimized functions)"
            )

        except Exception as e:
            logger.warning(f"Numba warmup failed: {e}")
            logger.exception("Full traceback for Numba warmup failure:")

    def _print_initialization_summary(self):
        """Print initialization summary."""
        logger.info("HomodyneAnalysis Core initialized:")
        logger.info(
            f"  • Frames: {self.start_frame}-{self.end_frame} ({
                self.time_length
            } frames)"
        )
        logger.info(f"  • Time step: {self.dt} s/frame")
        logger.info(f"  • Wavevector: {self.wavevector_q:.6f} A^-1")
        logger.info(f"  • Gap size: {self.stator_rotor_gap / 1e4:.1f} um")
        logger.info(f"  • Threads: {self.num_threads}")
        logger.info(
            f"  • Optimizations: {'Numba JIT' if NUMBA_AVAILABLE else 'Pure Python'}"
        )

    def is_static_mode(self) -> bool:
        """
        Check if the analysis is configured for static (no-flow) mode.

        In static mode:
        - Shear rate γ̇ = 0
        - Shear exponent β = 0
        - Shear offset γ̇_offset = 0
        - sinc² function = 1 (no shear decorrelation)
        - Only diffusion contribution g₁_diff remains

        Returns
        -------
        bool
            True if static mode is enabled in configuration
        """
        if self.config is None:
            return False

        # Check for static mode flag in configuration
        analysis_settings = self.config.get("analysis_settings", {})
        return analysis_settings.get("static_mode", False)

    def is_static_parameters(self, shear_params: np.ndarray) -> bool:
        """
        Check if shear parameters correspond to static conditions.

        In static conditions:
        - gamma_dot_t0 (shear_params[0]) ≈ 0
        - beta (shear_params[1]) = 0 (no time dependence)
        - gamma_dot_offset (shear_params[2]) ≈ 0

        Parameters
        ----------
        shear_params : np.ndarray
            Shear rate parameters [gamma_dot_t0, beta, gamma_dot_offset]

        Returns
        -------
        bool
            True if parameters indicate static conditions
        """
        if len(shear_params) < 3:
            # If we don't have enough shear parameters, assume static
            # conditions
            return True

        gamma_dot_t0 = shear_params[0]
        beta = shear_params[1]
        gamma_dot_offset = shear_params[2]

        # Define small threshold for "effectively zero"
        threshold = 1e-10

        # Check if all shear parameters are effectively zero
        return bool(
            abs(gamma_dot_t0) < threshold
            and abs(beta) < threshold
            and abs(gamma_dot_offset) < threshold
        )

    def get_effective_parameter_count(self) -> int:
        """
        Get the effective number of parameters based on analysis mode.

        Returns
        -------
        int
            Number of parameters actually used in the analysis:
            - Static mode: 3 (only diffusion parameters: D₀, α, D_offset)
            - Laminar flow mode: 7 (all parameters including shear and φ₀)
        """
        if self.is_static_mode():
            # Static mode: only diffusion parameters are meaningful
            return self.num_diffusion_params  # 3 parameters
        else:
            # Laminar flow mode: all parameters are used
            return (
                self.num_diffusion_params + self.num_shear_rate_params + 1
            )  # 7 parameters

    def get_effective_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """
        Extract only the effective parameters based on analysis mode.

        Parameters
        ----------
        parameters : np.ndarray
            Full parameter array [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]

        Returns
        -------
        np.ndarray
            Effective parameters based on mode:
            - Static mode: [D0, alpha, D_offset] (shear params set to 0, phi0 ignored)
            - Laminar flow mode: all parameters as provided
        """
        if self.is_static_mode():
            # Return only diffusion parameters, set others to zero
            effective_params = np.zeros(7)  # Standard 7-parameter array
            effective_params[: self.num_diffusion_params] = parameters[
                : self.num_diffusion_params
            ]
            # Shear parameters (indices 3,4,5) remain zero
            # phi0 (index 6) remains zero - irrelevant in static mode
            return effective_params
        else:
            # Return all parameters as provided
            return parameters.copy()

    def _apply_config_overrides(self, overrides: dict[str, Any]):
        """Apply configuration overrides with deep merging."""

        def deep_update(base, update):
            # Guard against non-dict base objects
            if not isinstance(base, dict):
                logger.warning(
                    f"Cannot update non-dict base object of type {type(base)}"
                )
                return

            for key, value in update.items():
                if (
                    key in base
                    and isinstance(base[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_update(base[key], value)
                else:
                    base[key] = value

        deep_update(self.config, overrides)
        logger.info(f"Applied {len(overrides)} configuration overrides")

    # ============================================================================
    # DATA LOADING AND PREPROCESSING
    # ============================================================================

    @memory_efficient_cache(maxsize=128)
    def load_experimental_data(
        self,
    ) -> tuple[np.ndarray, int, np.ndarray, int]:
        """
        Load experimental correlation data with caching.

        Returns
        -------
        tuple
            (c2_experimental, time_length, phi_angles, num_angles)
        """
        logger.debug("Starting load_experimental_data method")

        # Return cached data if available
        if (
            self.cached_experimental_data is not None
            and self.cached_phi_angles is not None
        ):
            logger.debug("Cache hit: returning cached experimental data")
            return (
                self.cached_experimental_data,
                self.time_length,
                self.cached_phi_angles,
                len(self.cached_phi_angles),
            )

        # Ensure configuration is loaded
        if self.config is None:
            raise ValueError("Configuration not loaded: self.config is None.")

        # Load angle configuration - skip for isotropic static mode
        if self.config_manager.is_static_isotropic_enabled():
            # In isotropic static mode, create a single dummy angle
            phi_angles = np.array([0.0], dtype=np.float64)
            num_angles = 1
            logger.info(
                "Isotropic static mode: Using single dummy angle (0.0°) instead of loading phi_angles_file"
            )
        else:
            # Normal mode: load phi angles from file
            phi_angles_path = self.config["experimental_data"].get(
                "phi_angles_path", "."
            )
            phi_angles_file = self.config["experimental_data"]["phi_angles_file"]
            phi_file = os.path.join(phi_angles_path, phi_angles_file)
            logger.debug(f"Loading phi angles from: {phi_file}")
            phi_angles = np.loadtxt(phi_file, dtype=np.float64)
            # Ensure phi_angles is always an array, even for single values
            phi_angles = np.atleast_1d(phi_angles)
            num_angles = len(phi_angles)
            logger.debug(f"Loaded {num_angles} phi angles: {phi_angles}")

        # Check for cached processed data
        cache_template = self.config["experimental_data"]["cache_filename_template"]
        cache_file_path = self.config["experimental_data"].get("cache_file_path", ".")
        cache_filename = cache_template.format(
            start_frame=self.start_frame, end_frame=self.end_frame
        )
        cache_file = os.path.join(cache_file_path, cache_filename)
        logger.debug(f"Checking for cached data at: {cache_file}")

        if os.path.isfile(cache_file):
            logger.info(f"Cache hit: Loading cached data from {cache_file}")
            # Optimized loading with memory mapping for large files
            try:
                with np.load(cache_file, mmap_mode="r") as data:
                    c2_experimental = np.array(data["c2_exp"], dtype=np.float64)
                logger.debug(f"Cached data shape: {c2_experimental.shape}")
            except (OSError, ValueError) as e:
                logger.warning(
                    f"Failed to memory-map cache file, falling back to regular loading: {e}"
                )
                with np.load(cache_file) as data:
                    c2_experimental = data["c2_exp"].astype(np.float64)
        else:
            logger.info(
                f"Cache miss: Loading raw data (cache file {cache_file} not found)"
            )
            c2_experimental = self._load_raw_data(phi_angles, num_angles)
            logger.info(f"Raw data loaded with shape: {c2_experimental.shape}")

            # Save to cache
            compression_enabled = self.config["experimental_data"].get(
                "cache_compression", True
            )
            logger.debug(
                f"Saving data to cache with compression="
                f"{'enabled' if compression_enabled else 'disabled'}: "
                f"{cache_file}"
            )
            if compression_enabled:
                np.savez_compressed(cache_file, c2_exp=c2_experimental)
            else:
                np.savez(cache_file, c2_exp=c2_experimental)
            logger.debug(f"Data cached successfully to: {cache_file}")

        # Apply diagonal correction
        if self.config["advanced_settings"]["data_loading"].get(
            "use_diagonal_correction", True
        ):
            logger.debug("Applying diagonal correction to correlation matrices")
            c2_experimental = self._fix_diagonal_correction_vectorized(c2_experimental)
            logger.debug("Diagonal correction completed")

        # Cache in memory
        self.cached_experimental_data = c2_experimental
        self.cached_phi_angles = phi_angles

        # Cache for plotting
        self._last_experimental_data = c2_experimental
        self._last_phi_angles = phi_angles
        logger.debug(f"Data cached in memory - final shape: {c2_experimental.shape}")

        # Plot experimental data for validation if enabled
        if (
            self.config.get("workflow_integration", {})
            .get("analysis_workflow", {})
            .get("plot_experimental_data_on_load", False)
        ):
            logger.info("Plotting experimental data for validation...")
            try:
                self._plot_experimental_data_validation(c2_experimental, phi_angles)
                logger.info("Experimental data validation plot created successfully")
            except Exception as e:
                logger.warning(
                    f"Failed to create experimental data validation plot: {e}"
                )

        # Apply configuration-based performance tuning
        if self._tuning_enabled:
            try:
                characteristics = self.analyze_dataset_characteristics(
                    phi_angles, c2_experimental
                )
                self.apply_performance_tuning()
                logger.info(
                    f"Performance tuning applied for {characteristics['dataset_size']} "
                    f"({characteristics['n_angles']} angles, "
                    f"{characteristics['memory_usage_gb']:.2f} GB, "
                    f"complexity={characteristics['complexity_score']:.1f})"
                )
            except Exception as e:
                logger.debug(f"Performance tuning failed: {e}")

        logger.debug("load_experimental_data method completed successfully")
        return c2_experimental, self.time_length, phi_angles, num_angles

    def _load_raw_data(self, phi_angles: np.ndarray, num_angles: int) -> np.ndarray:
        """Load raw data from HDF5 files."""
        logger.debug("Starting _load_raw_data method")

        if self.config is None:
            raise ValueError("Configuration not loaded: self.config is None.")

        data_config = self.config["experimental_data"]
        folder = data_config["data_folder_path"]
        filename = data_config["data_file_name"]
        exchange_key = data_config.get("exchange_key", "exchange")

        full_path = os.path.join(folder, filename)
        logger.info(f"Opening HDF5 data file: {full_path}")
        logger.debug(f"Exchange key: {exchange_key}")
        logger.debug(
            f"Frame range: {self.start_frame}-{self.end_frame} (length: {
                self.time_length
            })"
        )

        # Open data file
        if not PYXPCSVIEWER_AVAILABLE or xf is None:
            raise ImportError(
                "pyxpcsviewer is required for loading raw experimental data. "
                "Install it with: pip install pyxpcsviewer"
            )

        try:
            data_file = xf(full_path)
            logger.debug(f"Successfully opened HDF5 file: {filename}")
        except Exception as e:
            logger.error(f"Failed to open HDF5 file {full_path}: {e}")
            raise

        # Pre-allocate output
        expected_shape = (num_angles, self.time_length, self.time_length)
        c2_experimental = np.zeros(expected_shape, dtype=np.float64)
        logger.debug(f"Pre-allocated output array with shape: {expected_shape}")

        # Handle data loading for isotropic static mode vs normal mode
        if self.config_manager.is_static_isotropic_enabled():
            # In isotropic static mode, load data only once and use for the
            # single dummy angle
            logger.info(
                "Isotropic static mode: Loading single correlation matrix for dummy angle"
            )

            try:
                # Load data once for isotropic case
                raw_data = data_file.get_twotime_c2(exchange_key, correct_diag=False)
                if raw_data is None:
                    raise ValueError(
                        "get_twotime_c2 returned None in isotropic static mode"
                    )

                # Ensure raw_data is a NumPy array
                raw_data_np = np.array(raw_data)
                sliced_data = raw_data_np[
                    self.start_frame : self.end_frame,
                    self.start_frame : self.end_frame,
                ]
                # Use the same data for the single dummy angle
                c2_experimental[0] = sliced_data.astype(np.float64)
                logger.debug(
                    f"  Isotropic mode - Raw data shape: {
                        raw_data_np.shape
                    } -> sliced: {sliced_data.shape}"
                )

            except Exception as e:
                logger.error(f"Failed to load data in isotropic static mode: {e}")
                raise
        else:
            # Normal mode: load data for each angle
            logger.info(f"Loading data for {num_angles} angles...")
            for i in range(num_angles):
                angle_deg = phi_angles[i]
                logger.debug(f"Loading angle {i + 1}/{num_angles} (φ={angle_deg:.2f}°)")

                try:
                    # Fix: Pass correct_diag as bool, not int. If you want
                    # diagonal correction, set to True, else False.
                    raw_data = data_file.get_twotime_c2(
                        exchange_key, correct_diag=False
                    )
                    if raw_data is None:
                        raise ValueError(
                            f"get_twotime_c2 returned None for angle {i + 1} (φ={
                                angle_deg:.2f}°)"
                        )
                    # Ensure raw_data is a NumPy array
                    raw_data_np = np.array(raw_data)
                    sliced_data = raw_data_np[
                        self.start_frame : self.end_frame,
                        self.start_frame : self.end_frame,
                    ]
                    c2_experimental[i] = sliced_data.astype(np.float64)
                    logger.debug(
                        f"  Raw data shape: {raw_data_np.shape} -> sliced: {
                            sliced_data.shape
                        }"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to load data for angle {i + 1} (φ={angle_deg:.2f}°): {
                            e
                        }"
                    )
                    raise

        logger.info(
            f"Successfully loaded raw data with final shape: {c2_experimental.shape}"
        )
        return c2_experimental

    def _fix_diagonal_correction_vectorized(self, c2_data: np.ndarray) -> np.ndarray:
        """Apply diagonal correction to correlation matrices."""
        if self.config is None or not (
            isinstance(self.config, dict)
            and self.config.get("advanced_settings", {})
            .get("data_loading", {})
            .get("vectorized_diagonal_fix", True)
        ):
            return c2_data

        num_angles, size, _ = c2_data.shape
        indices_i = np.arange(size - 1)
        indices_j = np.arange(1, size)

        for angle_idx in range(num_angles):
            matrix = c2_data[angle_idx]

            # Extract side-band values
            side_band = matrix[indices_i, indices_j]

            # Compute corrected diagonal
            diagonal = np.zeros(size, dtype=np.float64)
            diagonal[:-1] += side_band
            diagonal[1:] += side_band

            # Normalization
            norm = np.ones(size, dtype=np.float64)
            norm[1:-1] = 2.0

            # Apply correction
            np.fill_diagonal(matrix, diagonal / norm)

        return c2_data

    # ============================================================================
    # CORRELATION FUNCTION CALCULATIONS
    # ============================================================================

    def calculate_diffusion_coefficient_optimized(
        self, params: np.ndarray
    ) -> np.ndarray:
        """Calculate time-dependent diffusion coefficient.

        Ensures D(t) > 0 always by applying a minimum threshold."""
        D0, alpha, D_offset = params

        if NUMBA_AVAILABLE:
            return calculate_diffusion_coefficient_numba(
                self.time_array, D0, alpha, D_offset
            )
        else:
            D_t = D0 * (self.time_array**alpha) + D_offset
            return np.maximum(D_t, 1e-10)  # Ensure D(t) > 0 always

    def calculate_shear_rate_optimized(self, params: np.ndarray) -> np.ndarray:
        """Calculate time-dependent shear rate.

        Ensures γ̇(t) > 0 always by applying a minimum threshold."""
        gamma_dot_t0, beta, gamma_dot_t_offset = params

        if NUMBA_AVAILABLE:
            return calculate_shear_rate_numba(
                self.time_array, gamma_dot_t0, beta, gamma_dot_t_offset
            )
        else:
            gamma_t = gamma_dot_t0 * (self.time_array**beta) + gamma_dot_t_offset
            return np.maximum(gamma_t, 1e-10)  # Ensure γ̇(t) > 0 always

    @memory_efficient_cache(maxsize=256)
    def create_time_integral_matrix_cached(
        self, param_hash: str, time_array: np.ndarray
    ) -> np.ndarray:
        """Create cached time integral matrix with enhanced two-tier caching."""
        # Check two-tier cache first
        cache_key = f"integral_matrix_{param_hash}_{len(time_array)}"
        cached_result = self._get_from_two_tier_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # Optimized algorithm selection based on matrix size
        n = len(time_array)
        if NUMBA_AVAILABLE and n > 100:  # Use Numba only for larger matrices
            result = create_time_integral_matrix_numba(time_array)
        else:
            # Use fast NumPy vectorized approach for small matrices
            cumsum = np.cumsum(time_array)
            cumsum_matrix = np.tile(cumsum, (n, 1))
            result = np.abs(cumsum_matrix - cumsum_matrix.T)

        # Store in two-tier cache for future use
        self._put_in_two_tier_cache(cache_key, result)
        return result

    def calculate_c2_single_angle_optimized(
        self,
        parameters: np.ndarray,
        phi_angle: float,
        precomputed_D_t: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Calculate correlation function for a single angle.

        Supports both laminar flow and static (no-flow) cases:
        - Laminar flow: Full 7-parameter model with diffusion and shear contributions
        - Static case: Only diffusion contribution (sinc² = 1), φ₀ irrelevant and set to 0

        Parameters
        ----------
        parameters : np.ndarray
            Model parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
            In static mode: only first 3 diffusion parameters are used, others ignored/set to 0
        phi_angle : float
            Scattering angle in degrees

        Returns
        -------
        np.ndarray
            Correlation matrix c2(t1, t2)
        """
        # Check if we're in static mode
        static_mode = self.is_static_mode()

        # Extract parameters
        diffusion_params = parameters[: self.num_diffusion_params]
        shear_params = parameters[
            self.num_diffusion_params : self.num_diffusion_params
            + self.num_shear_rate_params
        ]
        phi_offset = parameters[-1]

        # Calculate time-dependent quantities
        param_hash = hash(tuple(parameters))
        if precomputed_D_t is not None:
            D_t = precomputed_D_t
        else:
            D_t = self.calculate_diffusion_coefficient_optimized(diffusion_params)

        # Create diffusion integral matrix
        D_integral = self.create_time_integral_matrix_cached(f"D_{param_hash}", D_t)

        # Compute g1 correlation (diffusion contribution)
        if NUMBA_AVAILABLE:
            g1 = compute_g1_correlation_numba(
                D_integral, self.wavevector_q_squared_half_dt
            )
        else:
            g1 = np.exp(-self.wavevector_q_squared_half_dt * D_integral)

        # Handle shear contribution based on mode
        if static_mode or self.is_static_parameters(shear_params):
            # Static case: sinc² = 1 (no shear contribution)
            # g₁(t₁,t₂) = g₁_diff(t₁,t₂) = exp[-q²/2 ∫|t₂-t₁| D(t')dt']
            # g₂(t₁,t₂) = [g₁(t₁,t₂)]²
            # Note: φ₀ is irrelevant in static mode since shear term is not
            # used
            sinc2 = np.ones_like(g1)
        else:
            # Laminar flow case: calculate full sinc² contribution
            gamma_dot_t = self.calculate_shear_rate_optimized(shear_params)
            gamma_integral = self.create_time_integral_matrix_cached(
                f"gamma_{param_hash}", gamma_dot_t
            )

            # Compute sinc² (shear contribution)
            angle_rad = np.deg2rad(phi_offset - phi_angle)
            cos_phi = np.cos(angle_rad)
            prefactor = self.sinc_prefactor * cos_phi

            if NUMBA_AVAILABLE:
                sinc2 = compute_sinc_squared_numba(gamma_integral, prefactor)
            else:
                arg = prefactor * gamma_integral
                sinc2 = np.sinc(arg) ** 2

        # Combine contributions: c2 = (g1 × sinc²)²
        return (sinc2 * g1) ** 2

    def _calculate_c2_single_angle_fast(
        self,
        parameters: np.ndarray,
        phi_angle: float,
        D_integral: np.ndarray,
        is_static: bool,
        shear_params: np.ndarray,
        gamma_integral: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Fast correlation function calculation with pre-computed values.

        This optimized version avoids redundant computations by accepting
        pre-calculated common values.

        Parameters
        ----------
        parameters : np.ndarray
            Model parameters
        phi_angle : float
            Scattering angle in degrees
        D_integral : np.ndarray
            Pre-computed diffusion integral matrix
        is_static : bool
            Pre-computed static mode flag
        shear_params : np.ndarray
            Pre-extracted shear parameters

        Returns
        -------
        np.ndarray
            Correlation matrix c2(t1, t2)
        """
        # Compute g1 correlation (diffusion contribution) - already optimized
        if NUMBA_AVAILABLE:
            g1 = compute_g1_correlation_numba(
                D_integral, self.wavevector_q_squared_half_dt
            )
        else:
            g1 = np.exp(-self.wavevector_q_squared_half_dt * D_integral)

        # Handle shear contribution based on pre-computed static mode
        if is_static:
            # Static case: sinc² = 1, so c2 = g1²
            return g1**2
        else:
            # Laminar flow case: calculate full sinc² contribution
            phi_offset = parameters[-1]

            # Use pre-computed gamma_integral if available, otherwise compute
            if gamma_integral is None:
                param_hash = hash(tuple(parameters))
                gamma_dot_t = self.calculate_shear_rate_optimized(shear_params)
                gamma_integral = self.create_time_integral_matrix_cached(
                    f"gamma_{param_hash}", gamma_dot_t
                )

            # Compute sinc² (shear contribution)
            angle_rad = np.deg2rad(phi_offset - phi_angle)
            cos_phi = np.cos(angle_rad)
            prefactor = self.sinc_prefactor * cos_phi

            if NUMBA_AVAILABLE:
                sinc2 = compute_sinc_squared_numba(gamma_integral, prefactor)
            else:
                arg = prefactor * gamma_integral
                # Avoid division by zero by using safe division
                with np.errstate(divide="ignore", invalid="ignore"):
                    sinc_values = np.sin(arg) / arg
                    sinc_values = np.where(np.abs(arg) < 1e-10, 1.0, sinc_values)
                sinc2 = sinc_values**2

        # Combine contributions: c2 = (g1 × sinc²)²
        return (sinc2 * g1) ** 2

    def _calculate_c2_vectorized_static(
        self, D_integral: np.ndarray, num_angles: int
    ) -> np.ndarray:
        """
        Ultra-fast vectorized correlation calculation for static case.

        In static mode, all angles produce identical correlation functions,
        so we compute once and broadcast to all angles.

        Parameters
        ----------
        D_integral : np.ndarray
            Pre-computed diffusion integral matrix
        num_angles : int
            Number of angles to replicate

        Returns
        -------
        np.ndarray
            3D array of correlation matrices [angles, time, time]
        """
        # Compute g1 correlation once (diffusion contribution)
        if NUMBA_AVAILABLE:
            g1 = compute_g1_correlation_numba(
                D_integral, self.wavevector_q_squared_half_dt
            )
        else:
            g1 = np.exp(-self.wavevector_q_squared_half_dt * D_integral)

        # Static case: c2 = g1² (sinc² = 1)
        c2_single = g1**2

        # Broadcast to all angles using memory-efficient approach
        if num_angles == 1:
            return c2_single.reshape(1, self.time_length, self.time_length)
        else:
            # Use efficient tile for multiple angles
            return np.tile(c2_single, (num_angles, 1, 1))

    def calculate_c2_nonequilibrium_laminar_parallel(
        self, parameters: np.ndarray, phi_angles: np.ndarray
    ) -> np.ndarray:
        """
        Calculate correlation function for all angles with parallel processing.

        Performance Optimizations (v0.6.1+):
        - Memory pooling: Pre-allocated result arrays to avoid repeated allocations
        - Static case optimization: Enhanced vectorized broadcasting for identical functions
        - Precomputed integrals: Cached shear integrals to eliminate redundant computation
        - Algorithm selection: Improved static vs laminar flow detection and handling

        Parameters
        ----------
        parameters : np.ndarray
            Model parameters
        phi_angles : np.ndarray
            Array of scattering angles

        Returns
        -------
        np.ndarray
            3D array of correlation matrices [angles, time, time]
        """
        num_angles = len(phi_angles)
        use_parallel = True
        if self.config is not None:
            use_parallel = self.config.get("performance_settings", {}).get(
                "parallel_execution", True
            )

        # Adaptive parallelization based on problem size
        performance_config = (
            self.config.get("performance_optimization", {}) if self.config else {}
        )
        parallel_threshold = performance_config.get("parallel_threshold", 5)

        # Avoid threading conflicts with Numba parallel operations
        # Use serial processing for small problems or when parallelization would add overhead
        if (
            self.num_threads == 1
            or num_angles < parallel_threshold  # Adaptive threshold
            or not use_parallel
            or (
                NUMBA_AVAILABLE and num_angles < 10
            )  # Numba handles its own parallelization
        ):
            # Sequential processing (Numba will handle internal parallelization)
            # Pre-calculate common values once to avoid redundant computation
            diffusion_params = parameters[: self.num_diffusion_params]
            shear_params = parameters[
                self.num_diffusion_params : self.num_diffusion_params
                + self.num_shear_rate_params
            ]

            # Pre-compute static conditions
            static_mode = self.is_static_mode()
            is_static_params = self.is_static_parameters(shear_params)
            is_static = static_mode or is_static_params

            # Pre-compute parameter hash and diffusion coefficient
            param_hash = hash(tuple(parameters))
            D_t = self.calculate_diffusion_coefficient_optimized(diffusion_params)
            D_integral = self.create_time_integral_matrix_cached(f"D_{param_hash}", D_t)

            # Use vectorized processing for maximum performance
            if is_static:
                # Static case: all angles have identical correlation (no angle
                # dependence)
                return self._calculate_c2_vectorized_static(D_integral, num_angles)
            else:
                # Laminar flow case: allocate results array
                c2_results = np.empty(
                    (num_angles, self.time_length, self.time_length),
                    dtype=np.float64,
                )

                # Pre-compute shear integrals once if applicable
                param_hash = hash(tuple(parameters))
                if not is_static_params:
                    gamma_dot_t = self.calculate_shear_rate_optimized(shear_params)
                    gamma_integral = self.create_time_integral_matrix_cached(
                        f"gamma_{param_hash}", gamma_dot_t
                    )
                else:
                    gamma_integral = None

                for i in range(num_angles):
                    c2_results[i] = self._calculate_c2_single_angle_fast(
                        parameters,
                        phi_angles[i],
                        D_integral,
                        is_static,
                        shear_params,
                        gamma_integral,
                    )

                return c2_results.copy()  # Return copy to avoid mutation

        else:
            # Parallel processing (only when Numba not available)
            # Pre-calculate diffusion coefficient once to avoid redundant
            # computation
            diffusion_params = parameters[: self.num_diffusion_params]
            D_t = self.calculate_diffusion_coefficient_optimized(diffusion_params)

            # Choose executor type based on dataset size and configuration
            use_threading = True
            if self.config is not None:
                use_threading = self.config.get("performance_settings", {}).get(
                    "use_threading", True
                )

            # Adaptive executor selection:
            # - ProcessPoolExecutor for large CPU-bound problems (>20 angles, >1000 time points)
            # - ThreadPoolExecutor for moderate problems or I/O-bound scenarios
            if num_angles > 20 and self.time_length > 1000 and not use_threading:
                Executor = ProcessPoolExecutor
                logger.debug(
                    f"Using ProcessPoolExecutor for {num_angles} angles, {self.time_length} time points"
                )
            else:
                Executor = ThreadPoolExecutor
                logger.debug(
                    f"Using ThreadPoolExecutor for {num_angles} angles, {self.time_length} time points"
                )

            with Executor(max_workers=self.num_threads) as executor:
                futures = [
                    executor.submit(
                        self.calculate_c2_single_angle_optimized,
                        parameters,
                        angle,
                        D_t,  # Pass precomputed diffusion coefficient
                    )
                    for angle in phi_angles
                ]

                c2_calculated = np.zeros(
                    (num_angles, self.time_length, self.time_length),
                    dtype=np.float64,
                )
                for i, future in enumerate(futures):
                    c2_calculated[i] = future.result()

                return c2_calculated

    def calculate_c2_correlation_vectorized(
        self, parameters: np.ndarray, phi_angles: np.ndarray
    ) -> np.ndarray:
        """
        Vectorized calculation of c2 correlation functions for multiple angles.

        This method is an alias for calculate_c2_nonequilibrium_laminar_parallel
        to maintain compatibility with the optimized chi-squared calculation.

        Parameters
        ----------
        parameters : np.ndarray
            Model parameters
        phi_angles : np.ndarray
            Array of scattering angles

        Returns
        -------
        np.ndarray
            Correlation functions for all angles
        """
        return self.calculate_c2_nonequilibrium_laminar_parallel(parameters, phi_angles)

    def _estimate_variance_irls_mad_robust(
        self, residuals: np.ndarray, window_size: int = 11, edge_method: str = "reflect"
    ) -> np.ndarray:
        """
        IRLS (Iterative Reweighted Least Squares) with MAD robust variance estimation.

        Implements the user's specified algorithm:
        1. Initialize with uniform variance σ²ᵢ = 1e-3
        2. Apply MAD moving window: σ²ᵢ = (1.4826 × MAD)²
        3. Use damping: σ² = α·σ²_new + (1-α)·σ²_prev with α=0.7
        4. Iterate until convergence

        Parameters
        ----------
        residuals : np.ndarray
            Residuals from model fit (may be pre-processed with edge reflection)
        window_size : int, default=11
            Size of moving window for MAD estimation
        edge_method : str, default="reflect"
            Edge handling method (currently only "reflect" supported)
        max_iterations : int, default=5
            Maximum IRLS iterations
        damping_factor : float, default=0.7
            Damping factor α for variance updates
        convergence_tolerance : float, default=1e-4
            Convergence tolerance for variance changes
        initial_sigma_squared : float, default=1e-3
            Initial uniform variance assumption

        Returns
        -------
        np.ndarray
            Final variance estimates (same size as original residuals)
        """
        # Get config parameters
        if not hasattr(self, "_cached_chi_config"):
            self._cached_chi_config = self.config.get("advanced_settings", {}).get(
                "chi_squared_calculation", {}
            )
        chi_config = self._cached_chi_config

        # Handle both old (nested irls_config) and new (direct) config formats
        irls_config = chi_config.get("irls_config", {})

        # New simplified format (direct in chi_config) takes precedence
        max_iterations = chi_config.get("irls_max_iterations") or irls_config.get(
            "max_iterations", 15
        )
        damping_factor = chi_config.get("irls_damping_factor") or irls_config.get(
            "damping_factor", 0.8
        )
        convergence_tolerance = chi_config.get(
            "irls_convergence_tolerance"
        ) or irls_config.get("convergence_tolerance", 0.003)
        initial_sigma_squared = chi_config.get(
            "irls_initial_sigma_squared"
        ) or irls_config.get("initial_sigma_squared", 1e-3)
        min_sigma_squared = chi_config.get("minimum_sigma", 1e-10) ** 2

        logger.debug(
            f"Starting IRLS MAD robust variance estimation with {max_iterations} max iterations"
        )

        # Handle reflected residuals - extract original size
        if edge_method == "reflect":
            # Ensure odd window size for proper padding
            pad_window_size = window_size if window_size % 2 == 1 else window_size + 1
            pad_size = pad_window_size // 2

            # When edge_method="reflect", residuals are expected to be pre-padded
            # We need to extract the original (unpadded) portion for the final result
            working_residuals = residuals

            # For reflection method, we always assume input is padded and extract center
            if pad_size > 0:
                extract_indices = slice(pad_size, -pad_size)
                expected_output_size = len(residuals) - 2 * pad_size
                logger.debug(
                    f"Reflection edge method: extracting center portion from padded input (size {len(residuals)} -> {expected_output_size})"
                )
            else:
                # Edge case: no padding needed
                extract_indices = slice(None)
                expected_output_size = len(residuals)
        else:
            working_residuals = residuals
            extract_indices = slice(None)
            expected_output_size = len(residuals)

        # Step 1: Initialize with uniform variance σ²ᵢ = 1e-3
        n_points = len(working_residuals)
        sigma2 = np.full(n_points, initial_sigma_squared)
        sigma2_prev = sigma2.copy()

        # IRLS iterations
        for iteration in range(max_iterations):
            # Step 2: Apply MAD moving window variance estimation (inline implementation)
            # Get window size and edge method from config
            config_window_size = chi_config.get("moving_window_size", window_size)
            config_edge_method = chi_config.get(
                "moving_window_edge_method", edge_method
            )
            min_window_size = chi_config.get("irls_min_window_size", 3)

            sigma2_new = self._mad_moving_window_with_edge_handling(
                working_residuals,
                config_window_size,
                config_edge_method,
                min_sigma_squared,
                min_window_size,
            )

            # Step 3: Apply damping to prevent oscillations
            if iteration > 0:
                alpha = damping_factor
                sigma2 = alpha * sigma2_new + (1 - alpha) * sigma2_prev
            else:
                sigma2 = sigma2_new.copy()  # First iteration: no damping

            # Check convergence based on variance changes
            variance_change = np.linalg.norm(sigma2 - sigma2_prev) / (
                np.linalg.norm(sigma2_prev) + 1e-10
            )

            logger.debug(
                f"IRLS iteration {iteration}: variance_change={variance_change:.6e}, "
                f"mean_σ²={np.mean(sigma2):.6e}"
            )

            # Check convergence
            if variance_change < convergence_tolerance and iteration > 0:
                logger.info(
                    f"IRLS MAD robust estimation converged after {iteration + 1} iterations"
                )
                break

            # Update for next iteration
            sigma2_prev = sigma2.copy()
        else:
            logger.warning(
                f"IRLS MAD robust estimation did not converge after {max_iterations} iterations"
            )

        # Extract original size if we were working with padded data
        final_variances = sigma2[extract_indices]

        # Ensure minimum variance floor
        final_variances = np.maximum(final_variances, min_sigma_squared)

        # Safety check: ensure output size matches expected output size
        if len(final_variances) != expected_output_size:
            logger.warning(
                f"Size mismatch after variance extraction: output({len(final_variances)}) != expected({expected_output_size}). "
                f"Using truncation/padding to match expected size."
            )
            if len(final_variances) < expected_output_size:
                # Pad with last variance value if output is too small
                pad_value = (
                    final_variances[-1]
                    if len(final_variances) > 0
                    else min_sigma_squared
                )
                final_variances = np.pad(
                    final_variances,
                    (0, expected_output_size - len(final_variances)),
                    mode="constant",
                    constant_values=pad_value,
                )
            else:
                # Truncate if output is too large
                final_variances = final_variances[:expected_output_size]

        return final_variances

    def _estimate_variance_irls_mad_robust_batch(
        self,
        residuals_batch_list: list[np.ndarray],
        window_size: int = 11,
        edge_method: str = "reflect",
    ) -> list[np.ndarray]:
        """
        Batch IRLS variance estimation for multiple angles with enhanced Numba optimization.

        This method processes multiple angles simultaneously using vectorized operations
        and JIT-compiled kernels for maximum performance. It replaces sequential
        processing of angles with parallel batch operations.

        Parameters
        ----------
        residuals_batch_list : list[np.ndarray]
            List of residuals arrays, one per angle
        window_size : int, default=11
            Size of moving window for MAD estimation
        edge_method : str, default="reflect"
            Edge handling method

        Returns
        -------
        list[np.ndarray]
            List of variance estimates, one array per angle
        """
        # Get config parameters (same as single-angle version)
        if not hasattr(self, "_cached_chi_config"):
            self._cached_chi_config = self.config.get("advanced_settings", {}).get(
                "chi_squared_calculation", {}
            )
        chi_config = self._cached_chi_config

        # Handle both old (nested irls_config) and new (direct) config formats
        irls_config = chi_config.get("irls_config", {})

        # New simplified format (direct in chi_config) takes precedence
        max_iterations = chi_config.get("irls_max_iterations") or irls_config.get(
            "max_iterations", 15
        )
        damping_factor = chi_config.get("irls_damping_factor") or irls_config.get(
            "damping_factor", 0.8
        )
        convergence_tolerance = chi_config.get(
            "irls_convergence_tolerance"
        ) or irls_config.get("convergence_tolerance", 0.003)
        initial_sigma_squared = chi_config.get(
            "irls_initial_sigma_squared"
        ) or irls_config.get("initial_sigma_squared", 1e-3)
        min_sigma_squared = chi_config.get("minimum_sigma", 1e-10) ** 2

        logger.debug(
            f"Starting batch IRLS MAD robust variance estimation for {len(residuals_batch_list)} angles"
        )

        # Convert list to 2D array for batch processing
        try:
            # Check if all residuals have the same length
            lengths = [len(residuals) for residuals in residuals_batch_list]
            if len(set(lengths)) == 1:
                # All same length - can use efficient 2D array processing
                residuals_batch_array = np.array(residuals_batch_list, dtype=np.float64)

                # Use enhanced Numba kernel for batch processing
                try:
                    sigma_variances_batch = estimate_variance_irls_batch_numba(
                        residuals_batch_array,
                        window_size,
                        max_iterations,
                        damping_factor,
                        convergence_tolerance,
                        initial_sigma_squared,
                        min_sigma_squared,
                    )

                    # Convert back to list format for compatibility
                    return [
                        sigma_variances_batch[i]
                        for i in range(len(residuals_batch_list))
                    ]

                except RuntimeError as e:
                    if "NUMBA_NUM_THREADS" in str(e):
                        logger.debug(
                            "Using fallback batch processing due to NUMBA threading conflict"
                        )
                        # Fall through to fallback processing
                    else:
                        raise
            else:
                logger.debug(
                    "Residuals have different lengths, using fallback processing"
                )

        except Exception as e:
            logger.debug(f"Batch array processing failed: {e}, using fallback")

        # Fallback: Process each angle individually using existing method
        logger.debug("Using individual angle processing fallback")
        variance_results = []
        for i, residuals in enumerate(residuals_batch_list):
            variances = self._estimate_variance_irls_mad_robust(
                residuals, window_size=window_size, edge_method=edge_method
            )
            variance_results.append(variances)

        return variance_results

    def _estimate_variance_simple_mad(
        self, residuals: np.ndarray, window_size: int = 25, edge_method: str = "reflect"
    ) -> np.ndarray:
        """
        Simple MAD (Median Absolute Deviation) variance estimation without iterations.

        Provides robust initialization for hybrid limited-iteration IRLS by performing
        a single-pass MAD estimation on unweighted least squares residuals. This method
        offers O(nk) complexity for efficient startup compared to iterative methods.

        Key features:
        - Single-pass calculation (no iterations)
        - Larger default window size (25) for improved stability
        - Robust against outliers through median-based estimation
        - Compatible with existing edge handling methods
        - Suitable as initialization for hybrid IRLS approaches

        Parameters
        ----------
        residuals : np.ndarray
            Residuals from unweighted least squares fit
        window_size : int, default=25
            Size of moving window for MAD estimation (larger for stability)
        edge_method : str, default="reflect"
            Edge handling method: "reflect", "adaptive_window", or "global_fallback"

        Returns
        -------
        np.ndarray
            Variance estimates (σ²) for each data point

        Notes
        -----
        This method implements the initialization step of the hybrid limited-iteration
        IRLS approach, inspired by feasible generalized least squares (FGLS). It provides
        a robust starting point that captures local variance structure without the
        computational cost of full iterative estimation.

        The conversion factor 1.4826 is used to convert MAD to standard deviation
        assuming Gaussian residuals: σ ≈ 1.4826 × MAD.
        """
        n_points = len(residuals)
        if n_points == 0:
            return np.array([])

        # Get minimum sigma from config for consistency
        if not hasattr(self, "_cached_chi_config"):
            self._cached_chi_config = self.config.get("advanced_settings", {}).get(
                "chi_squared_calculation", {}
            )
        chi_config = self._cached_chi_config
        min_sigma_squared = chi_config.get("minimum_sigma", 1e-10) ** 2
        min_window_size = 3

        # Initialize variance array
        sigma2_mad = np.full(n_points, min_sigma_squared)

        # MAD factor for converting to variance: (1.4826)² ≈ 2.198
        mad_factor = (1.4826) ** 2

        # Single-pass MAD estimation with moving window
        half_window = window_size // 2

        for i in range(n_points):
            # Calculate window bounds
            start_idx = max(0, i - half_window)
            end_idx = min(n_points, i + half_window + 1)

            # Extract window data
            window_residuals = residuals[start_idx:end_idx]

            if len(window_residuals) >= min_window_size:
                # Standard MAD calculation
                median_res = np.median(window_residuals)
                mad = np.median(np.abs(window_residuals - median_res))

                if mad > 0:
                    sigma2_mad[i] = mad_factor * mad**2
                else:
                    sigma2_mad[i] = min_sigma_squared
            else:
                # Handle edge cases based on edge method
                if edge_method == "reflect":
                    # Use reflection for insufficient data
                    reflected_residuals = self._reflect_residuals_at_edges(
                        residuals, i, window_size
                    )
                    if len(reflected_residuals) >= min_window_size:
                        median_res = np.median(reflected_residuals)
                        mad = np.median(np.abs(reflected_residuals - median_res))
                        if mad > 0:
                            sigma2_mad[i] = mad_factor * mad**2
                        else:
                            sigma2_mad[i] = min_sigma_squared
                    else:
                        sigma2_mad[i] = min_sigma_squared

                elif edge_method == "adaptive_window":
                    # Use available data even if window is smaller
                    if len(window_residuals) > 0:
                        median_res = np.median(window_residuals)
                        mad = np.median(np.abs(window_residuals - median_res))
                        if mad > 0:
                            sigma2_mad[i] = mad_factor * mad**2
                        else:
                            sigma2_mad[i] = min_sigma_squared
                    else:
                        sigma2_mad[i] = min_sigma_squared

                elif edge_method == "global_fallback":
                    # Use global variance as fallback
                    global_var = np.var(residuals)
                    if global_var > 0:
                        sigma2_mad[i] = max(global_var, min_sigma_squared)
                    else:
                        sigma2_mad[i] = min_sigma_squared
                else:
                    # Unknown edge method, use minimum variance
                    sigma2_mad[i] = min_sigma_squared

        # Apply minimum variance floor
        sigma2_mad = np.maximum(sigma2_mad, min_sigma_squared)

        return sigma2_mad

    def _estimate_variance_hybrid_limited_irls(
        self, residuals: np.ndarray, window_size: int = 25, edge_method: str = "reflect"
    ) -> np.ndarray:
        """
        Hybrid Limited-Iteration IRLS with Simple MAD initialization.

        Implements the hybrid approach that combines Simple MAD initialization with
        capped IRLS iterations (2-3) for optimal balance between accuracy and efficiency.
        This method is inspired by feasible generalized least squares (FGLS) and provides
        robust convergence with significantly reduced computational cost.

        Algorithm:
        1. Initialize σ²ᵢ using Simple MAD (replaces uniform initialization)
        2. Apply 2-3 capped IRLS iterations with enhanced damping
        3. Early stopping on convergence or overshooting detection
        4. Optional weighted refit integration at each iteration

        Parameters
        ----------
        residuals : np.ndarray
            Residuals from model fit
        window_size : int, default=25
            Size of moving window for MAD estimation (larger for stability)
        edge_method : str, default="reflect"
            Edge handling method: "reflect", "adaptive_window", or "global_fallback"

        Returns
        -------
        np.ndarray
            Final variance estimates (σ²) for each data point

        Notes
        -----
        Key improvements over standard IRLS:
        - 50-70% reduction in computation time through limited iterations
        - Improved initialization with Simple MAD vs. uniform σ²=1e-3
        - Enhanced numerical stability with adaptive damping
        - Early stopping prevents overshooting and oscillations
        - Compatible with existing batch processing optimizations
        """
        n_points = len(residuals)
        if n_points == 0:
            return np.array([])

        # Get hybrid IRLS configuration parameters
        if not hasattr(self, "_cached_chi_config"):
            self._cached_chi_config = self.config.get("advanced_settings", {}).get(
                "chi_squared_calculation", {}
            )
        chi_config = self._cached_chi_config

        # Hybrid-specific parameters with fallback to IRLS config
        max_iterations = chi_config.get("hybrid_irls_max_iterations", 3)
        damping_factor = chi_config.get("hybrid_irls_damping_factor", 0.7)
        convergence_tolerance = chi_config.get(
            "hybrid_irls_convergence_tolerance", 0.001
        )
        min_sigma_squared = chi_config.get("minimum_sigma", 1e-10) ** 2
        enable_weighted_refit = chi_config.get(
            "hybrid_irls_enable_weighted_refit", False
        )
        adaptive_target_alpha = chi_config.get("adaptive_target_alpha", 1.0)

        logger.debug(
            f"Starting Hybrid Limited IRLS with Simple MAD initialization: "
            f"max_iterations={max_iterations}, damping_factor={damping_factor}"
        )

        # Step 1: Initialize with Simple MAD (robust starting point)
        logger.debug("Step 1: Initializing with Simple MAD estimation")
        sigma2 = self._estimate_variance_simple_mad(residuals, window_size, edge_method)
        sigma2_prev = sigma2.copy()

        # Track convergence for early stopping
        chi2_prev = np.inf

        # Step 2: Apply limited IRLS iterations with enhanced damping
        for iteration in range(max_iterations):
            logger.debug(f"Hybrid IRLS iteration {iteration + 1}/{max_iterations}")

            # Apply MAD moving window with current residuals
            # Use smaller window for iterations to be more responsive
            iteration_window_size = max(11, window_size // 2)
            sigma2_new = self._mad_moving_window_with_edge_handling(
                residuals, iteration_window_size, edge_method, min_sigma_squared, 3
            )

            # Step 3: Enhanced damping to prevent oscillations
            if iteration > 0:
                # Adaptive damping: stronger damping for later iterations
                alpha = damping_factor * (
                    1 - 0.1 * iteration
                )  # Gradually increase damping
                alpha = max(0.5, alpha)  # Minimum damping of 0.5
                sigma2 = alpha * sigma2_new + (1 - alpha) * sigma2_prev
            else:
                sigma2 = sigma2_new.copy()  # First iteration: no damping

            # Step 4: Convergence checking with early stopping
            # Calculate chi-squared for convergence monitoring
            sigma_per_point = np.sqrt(sigma2)
            safe_sigma = np.maximum(sigma_per_point, np.sqrt(min_sigma_squared))

            # Avoid division by zero
            finite_mask = (
                np.isfinite(residuals) & np.isfinite(safe_sigma) & (safe_sigma > 0)
            )
            if np.any(finite_mask):
                residuals_finite = residuals[finite_mask]
                sigma_finite = safe_sigma[finite_mask]
                chi2_per_point = (residuals_finite / sigma_finite) ** 2
                chi2_current = np.sum(chi2_per_point)

                # Calculate reduced chi-squared for overshooting detection
                n_params = 3  # Typical parameter count for homodyne analysis
                effective_dof = max(1, len(residuals_finite) - n_params)
                chi2_reduced = chi2_current / effective_dof

                # Early stopping conditions
                if iteration > 0:
                    # Convergence check
                    chi2_change = abs(chi2_current - chi2_prev) / max(chi2_prev, 1e-10)
                    if chi2_change < convergence_tolerance:
                        logger.debug(
                            f"Hybrid IRLS converged at iteration {iteration + 1}: "
                            f"chi2_change={chi2_change:.2e} < tolerance={convergence_tolerance}"
                        )
                        break

                    # Overshooting detection: χ²_red < 0.8α suggests overfitting
                    overshooting_threshold = 0.8 * adaptive_target_alpha
                    if chi2_reduced < overshooting_threshold:
                        logger.debug(
                            f"Hybrid IRLS early stop due to overshooting at iteration {iteration + 1}: "
                            f"chi2_reduced={chi2_reduced:.3f} < {overshooting_threshold:.3f}"
                        )
                        break

                chi2_prev = chi2_current
            else:
                logger.warning(
                    f"Hybrid IRLS iteration {iteration + 1}: No finite residuals/sigma values"
                )
                break

            # Store previous values for next iteration
            sigma2_prev = sigma2.copy()

            # Optional: Weighted refit integration
            if enable_weighted_refit and iteration < max_iterations - 1:
                try:
                    # Use current variance estimates as weights (1/sigma²)
                    weights = 1.0 / np.maximum(sigma2, min_sigma_squared)
                    weights = np.maximum(weights, 1e-10)  # Prevent extreme weights

                    # Weighted refit: recalculate residuals using weighted least squares approach
                    # This improves residual estimates by downweighting high-variance points
                    residuals_updated = self._apply_weighted_refit(residuals, weights)

                    if np.isfinite(residuals_updated).all():
                        residuals = residuals_updated
                        logger.debug(
                            f"Weighted refit applied at iteration {iteration + 1}: "
                            f"weight_range=[{np.min(weights):.2e}, {np.max(weights):.2e}]"
                        )
                    else:
                        logger.warning(
                            f"Weighted refit produced invalid residuals at iteration {iteration + 1}"
                        )

                except Exception as e:
                    logger.warning(
                        f"Weighted refit failed at iteration {iteration + 1}: {e}"
                    )

        # Step 5: Apply final variance floor and validation
        sigma2_final = np.maximum(sigma2, min_sigma_squared)

        # Final validation
        if not np.all(np.isfinite(sigma2_final)):
            logger.warning("Non-finite values in Hybrid IRLS results, applying cleanup")
            sigma2_final = np.where(
                np.isfinite(sigma2_final), sigma2_final, min_sigma_squared
            )

        logger.debug(
            f"Hybrid Limited IRLS completed: mean_sigma2={np.mean(sigma2_final):.2e}, "
            f"min_sigma2={np.min(sigma2_final):.2e}, max_sigma2={np.max(sigma2_final):.2e}"
        )

        return sigma2_final

    def _estimate_variance_hybrid_limited_irls_batch(
        self,
        residuals_batch_list: list[np.ndarray],
        window_size: int = 25,
        edge_method: str = "reflect",
    ) -> list[np.ndarray]:
        """
        Batch Hybrid Limited-Iteration IRLS with Simple MAD initialization.

        Processes multiple angles simultaneously using the hybrid approach that combines
        Simple MAD initialization with capped IRLS iterations (2-3). Provides significant
        performance improvements over sequential processing while maintaining accuracy.

        This method leverages the new hybrid Numba kernel for maximum performance,
        with fallback to sequential processing when needed.

        Parameters
        ----------
        residuals_batch_list : list[np.ndarray]
            List of residuals arrays, one per angle
        window_size : int, default=25
            Size of moving window for MAD estimation (larger for stability)
        edge_method : str, default="reflect"
            Edge handling method: "reflect", "adaptive_window", or "global_fallback"

        Returns
        -------
        list[np.ndarray]
            List of variance estimates (σ²), one array per angle

        Notes
        -----
        Performance benefits of hybrid batch processing:
        - Simple MAD initialization reduces iteration requirements by 60-80%
        - Vectorized batch operations across all angles simultaneously
        - Enhanced damping and early stopping prevent oscillations
        - Maintains compatibility with existing optimization frameworks
        """
        if not residuals_batch_list:
            return []

        # Get hybrid IRLS configuration
        if not hasattr(self, "_cached_chi_config"):
            self._cached_chi_config = self.config.get("advanced_settings", {}).get(
                "chi_squared_calculation", {}
            )
        chi_config = self._cached_chi_config

        # Hybrid-specific parameters
        max_iterations = chi_config.get("hybrid_irls_max_iterations", 3)
        damping_factor = chi_config.get("hybrid_irls_damping_factor", 0.7)
        convergence_tolerance = chi_config.get(
            "hybrid_irls_convergence_tolerance", 0.001
        )
        min_sigma_squared = chi_config.get("minimum_sigma", 1e-10) ** 2

        logger.debug(
            f"Starting Hybrid Limited IRLS batch processing: {len(residuals_batch_list)} angles, "
            f"max_iterations={max_iterations}, damping_factor={damping_factor}"
        )

        try:
            # Check if all arrays have the same length for efficient batch processing
            array_lengths = [len(res) for res in residuals_batch_list]
            if len(set(array_lengths)) == 1:
                # Uniform array lengths - use efficient batch processing
                n_angles = len(residuals_batch_list)
                n_points = array_lengths[0]

                # Convert list to 2D array for batch processing
                residuals_batch_array = np.zeros((n_angles, n_points), dtype=np.float64)
                for i, residuals in enumerate(residuals_batch_list):
                    residuals_batch_array[i, :] = residuals

                # Use hybrid batch Numba kernel
                sigma2_batch_array = hybrid_irls_batch_numba(
                    residuals_batch_array,
                    window_size,
                    max_iterations,
                    damping_factor,
                    convergence_tolerance,
                    min_sigma_squared,
                )

                # Convert back to list format
                variance_results = []
                for i in range(n_angles):
                    variance_results.append(sigma2_batch_array[i, :].copy())

                logger.debug(
                    f"Hybrid batch processing completed successfully for {n_angles} angles"
                )
                return variance_results

            else:
                # Non-uniform array lengths - fall back to sequential processing
                logger.debug(
                    "Non-uniform array lengths detected, using sequential hybrid processing"
                )
                raise ValueError(
                    "Non-uniform array lengths require sequential processing"
                )

        except (RuntimeError, ValueError, MemoryError) as batch_error:
            # Comprehensive fallback to sequential hybrid processing
            logger.warning(f"Hybrid batch processing failed: {str(batch_error)}")
            logger.info("Falling back to sequential hybrid IRLS processing")

            variance_results = []
            for i, residuals in enumerate(residuals_batch_list):
                try:
                    # Use sequential hybrid method for each angle
                    variances = self._estimate_variance_hybrid_limited_irls(
                        residuals, window_size=window_size, edge_method=edge_method
                    )
                    variance_results.append(variances)
                except Exception as angle_error:
                    logger.error(
                        f"Hybrid sequential processing failed for angle {i}: {str(angle_error)}"
                    )
                    # Use simple MAD as ultimate fallback
                    try:
                        variances = self._estimate_variance_simple_mad(
                            residuals, window_size=window_size, edge_method=edge_method
                        )
                        variance_results.append(variances)
                    except Exception as mad_error:
                        logger.error(
                            f"Simple MAD fallback failed for angle {i}: {str(mad_error)}"
                        )
                        # Ultimate fallback: minimum variance
                        variances = np.full(len(residuals), min_sigma_squared)
                        variance_results.append(variances)

            logger.debug(
                f"Sequential hybrid processing completed for {len(variance_results)} angles"
            )
            return variance_results

    def _apply_weighted_refit(
        self, residuals: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """
        Apply weighted refit to improve residual estimates during IRLS iterations.

        This method uses the current variance estimates (as weights) to recalculate
        residuals with a weighted least squares approach. High-variance points are
        downweighted, leading to more robust parameter estimates.

        The approach implements a simplified weighted least squares refinement:
        - Uses weights = 1/σ² to downweight high-variance points
        - Applies robust smoothing to reduce outlier influence
        - Maintains residual structure while improving local estimates

        Parameters
        ----------
        residuals : np.ndarray
            Original residuals from least squares fit
        weights : np.ndarray
            Weights = 1/σ² from current variance estimates

        Returns
        -------
        np.ndarray
            Updated residuals after weighted refit
        """
        if len(residuals) != len(weights):
            return residuals

        if len(residuals) == 0:
            return residuals

        # Apply weighted smoothing with adaptive kernel size
        # This reduces outlier influence while preserving signal structure
        try:
            # Normalize weights to prevent numerical issues
            weights_normalized = weights / np.mean(weights)
            weights_clipped = np.clip(weights_normalized, 0.1, 10.0)

            # Apply weighted moving average with adaptive window
            window_size = min(
                9, max(3, len(residuals) // 20)
            )  # Adaptive window: 3-9 points
            residuals_smoothed = np.zeros_like(residuals)

            for i in range(len(residuals)):
                # Define local window
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(residuals), i + window_size // 2 + 1)

                # Extract local residuals and weights
                local_residuals = residuals[start_idx:end_idx]
                local_weights = weights_clipped[start_idx:end_idx]

                # Weighted average
                if np.sum(local_weights) > 0:
                    weighted_sum = np.sum(local_residuals * local_weights)
                    weight_sum = np.sum(local_weights)
                    residuals_smoothed[i] = weighted_sum / weight_sum
                else:
                    residuals_smoothed[i] = residuals[i]

            # Combine original and smoothed residuals with conservative blending
            # Use higher blending for points with lower weights (higher variance)
            blend_factor = np.clip(
                1.0 / weights_normalized, 0.1, 0.3
            )  # 10-30% smoothing
            residuals_refit = (
                1.0 - blend_factor
            ) * residuals + blend_factor * residuals_smoothed

            # Validate output
            if not np.isfinite(residuals_refit).all():
                return residuals

            return residuals_refit

        except Exception:
            # Return original residuals if weighted refit fails
            return residuals

    def _mad_moving_window_with_edge_handling(
        self,
        residuals: np.ndarray,
        window_size: int,
        edge_method: str,
        min_sigma_squared: float,
        min_window_size: int = 3,
    ) -> np.ndarray:
        """
        MAD moving window with proper edge effects handling.

        Implements the corrected edge handling from the reference plan:
        - 'reflect' (default): mirror at boundaries
        - 'adaptive_window': shrink at edges
        - 'global_fallback': use global variance
        """
        n = len(residuals)
        sigma2_new = np.zeros(n)
        half_window = window_size // 2

        for i in range(n):
            # Step 1: Determine window bounds
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            window_residuals = residuals[start:end]

            # Step 2: Handle different edge methods
            if len(window_residuals) >= min_window_size:
                # Sufficient data for MAD calculation
                median_res = np.median(window_residuals)
                mad = np.median(np.abs(window_residuals - median_res))

                if mad > 0:
                    sigma2_new[i] = (1.4826 * mad) ** 2  # MAD to variance conversion
                else:
                    sigma2_new[i] = min_sigma_squared
            else:
                # Handle insufficient data based on edge method
                if edge_method == "reflect":
                    # Reflect residuals at boundaries (default method)
                    reflected_residuals = self._reflect_residuals_at_edges(
                        residuals, i, half_window
                    )
                    median_res = np.median(reflected_residuals)
                    mad = np.median(np.abs(reflected_residuals - median_res))
                    sigma2_new[i] = (
                        (1.4826 * mad) ** 2 if mad > 0 else min_sigma_squared
                    )

                elif edge_method == "adaptive_window":
                    # Fallback to global variance
                    global_var = (
                        np.var(residuals, ddof=1)
                        if len(residuals) > 1
                        else min_sigma_squared
                    )
                    sigma2_new[i] = max(global_var, min_sigma_squared)

                elif edge_method == "global_fallback":
                    # Use global MAD as fallback
                    global_median = np.median(residuals)
                    global_mad = np.median(np.abs(residuals - global_median))
                    sigma2_new[i] = (
                        (1.4826 * global_mad) ** 2
                        if global_mad > 0
                        else min_sigma_squared
                    )

                else:
                    # Default: reflect behavior
                    reflected_residuals = self._reflect_residuals_at_edges(
                        residuals, i, half_window
                    )
                    median_res = np.median(reflected_residuals)
                    mad = np.median(np.abs(reflected_residuals - median_res))
                    sigma2_new[i] = (
                        (1.4826 * mad) ** 2 if mad > 0 else min_sigma_squared
                    )

        # Apply minimum variance floor
        return np.maximum(sigma2_new, min_sigma_squared)

    def _reflect_residuals_at_edges(
        self, residuals: np.ndarray, center_idx: int, half_window: int
    ) -> np.ndarray:
        """
        Reflect residuals at edges using mirror boundary conditions only.

        Removed 'extend' method - only mirror reflection is implemented.
        """
        n = len(residuals)
        start_idx = center_idx - half_window
        end_idx = center_idx + half_window + 1

        # Collect residuals with mirror reflection
        window_residuals = []

        for idx in range(start_idx, end_idx):
            if 0 <= idx < n:
                # Normal case: within bounds
                window_residuals.append(residuals[idx])
            elif idx < 0:
                # Mirror at start: residuals[0, 1, 2, ...] -> residuals[2, 1, 0, ...]
                reflect_idx = min(abs(idx), n - 1)
                window_residuals.append(residuals[reflect_idx])
            else:
                # Mirror at end: residuals[..., n-3, n-2, n-1] -> residuals[..., n-1, n-2, n-3]
                reflect_idx = max(0, 2 * n - 1 - idx)
                window_residuals.append(residuals[reflect_idx])

        return np.array(window_residuals)

    def _calculate_chi_squared_with_config(
        self,
        parameters: np.ndarray,
        phi_angles: np.ndarray,
        c2_experimental: np.ndarray,
        method_name: str = "",
        return_components: bool = False,
        filter_angles_for_optimization: bool = False,
        enable_performance_monitoring: bool = True,
    ) -> float | dict[str, Any]:
        """
        Configuration-aware chi-squared calculation with optimized JIT backend.

        Uses vectorized JIT-compiled chi-squared calculation when enabled for 20-50x speedup.
        Falls back to standard calculation if optimization is disabled.

        Parameters
        ----------
        Same as calculate_chi_squared_optimized

        Returns
        -------
        float or dict
            Chi-squared value or components dictionary

        Notes
        -----
        This method serves as a configuration-aware wrapper that selects between
        optimized and standard chi-squared calculation based on performance settings.
        """
        # Check if vectorized JIT is enabled
        perf_config = getattr(self, "perf_config", {})
        chi_calculator_type = perf_config.get("chi_calculator", "standard")

        if chi_calculator_type != "vectorized_jit":
            # Use standard method
            return self.calculate_chi_squared_optimized(
                parameters,
                phi_angles,
                c2_experimental,
                method_name,
                return_components,
                filter_angles_for_optimization,
            )

        # Use optimized calculation
        # Performance monitoring initialization
        component_times = {} if enable_performance_monitoring else None

        # Performance monitoring: theoretical calculation
        if enable_performance_monitoring:
            theory_start = time.time()

        # Get theoretical values using the same method as calculate_chi_squared_optimized
        c2_theoretical = self.calculate_c2_nonequilibrium_laminar_parallel(
            parameters, phi_angles
        )

        # Performance monitoring: record theory calculation time
        if enable_performance_monitoring:
            component_times["theory_calculation"] = time.time() - theory_start

        if c2_theoretical is None:
            return np.inf

        # Calculate residuals
        residuals = c2_experimental.flatten() - c2_theoretical.flatten()

        # Get variance estimates using selected estimator
        if hasattr(self, "_selected_variance_estimator"):
            variances = self._selected_variance_estimator(residuals)
        else:
            variances = self._estimate_variance_irls_mad_robust(residuals)

        # Handle size mismatch between residuals and variances
        if len(residuals) != len(variances):
            min_size = min(len(residuals), len(variances))
            residuals = residuals[:min_size]
            variances = variances[:min_size]
            # Log the size adjustment
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(
                f"Adjusted array sizes to {min_size} for chi-squared calculation"
            )

        # Convert variances to standard deviations for proper chi-squared calculation
        sigma_per_point = np.sqrt(variances)

        # Apply minimum sigma floor to prevent division by zero
        min_sigma = (
            self.config.get("advanced_settings", {})
            .get("chi_squared_calculation", {})
            .get("minimum_sigma", 1e-10)
        )
        sigma_safe = np.maximum(sigma_per_point, min_sigma)

        # Check for finite values only
        finite_mask = np.isfinite(residuals) & np.isfinite(sigma_safe)
        if not np.any(finite_mask):
            return np.inf

        # Use only finite values for calculation
        residuals_finite = residuals[finite_mask]
        sigma_finite = sigma_safe[finite_mask]

        # Proper chi-squared: χ² = Σ((residuals/σ)²) - matches slow method calculation
        chi_squared_per_point = (residuals_finite / sigma_finite) ** 2
        chi_squared = np.sum(chi_squared_per_point)

        if return_components:
            # Calculate reduced chi-squared and degrees of freedom to match slow method format
            n_params = len(parameters)
            effective_dof = max(1, len(residuals_finite) - n_params)
            reduced_chi_squared = chi_squared / effective_dof

            return {
                "chi_squared": reduced_chi_squared,  # Return reduced chi-squared like slow method
                "reduced_chi_squared": reduced_chi_squared,
                "total_chi_squared": chi_squared,  # Raw chi-squared value
                "degrees_of_freedom": effective_dof,
                "residuals": residuals_finite,
                "variances": variances,
                "sigma_per_point": sigma_finite,
                "c2_theoretical": c2_theoretical,
                "c2_experimental": c2_experimental,
                "valid": True,
            }

        # Return reduced chi-squared to match slow method behavior
        n_params = len(parameters)
        effective_dof = max(1, len(residuals_finite) - n_params)
        reduced_chi_squared = chi_squared / effective_dof
        return reduced_chi_squared

    def log_optimization_progress(
        self,
        iteration: int,
        chi_squared: float,
        residuals: np.ndarray = None,
        method_name: str = "",
        total_dof: int = None,
        optimization_config: dict = None,
    ) -> None:
        """
        Log standardized optimization progress with reduced chi-squared and residual statistics.

        This method provides consistent logging format across all optimization methods,
        helping track convergence behavior and diagnose optimization issues.

        Parameters
        ----------
        iteration : int
            Current optimization iteration number
        chi_squared : float
            Current reduced chi-squared value
        residuals : np.ndarray, optional
            Current residuals array for min/mean/max calculation
        method_name : str, optional
            Name of optimization method (e.g., "Nelder-Mead", "Robust-Wasserstein")
        total_dof : int, optional
            Total degrees of freedom for the fit
        optimization_config : dict, optional
            Configuration dictionary for logging control

        Notes
        -----
        Logging is controlled by configuration settings to avoid performance impact.
        Only logs when debug level is enabled or optimization logging is explicitly enabled.
        """
        import logging

        logger = logging.getLogger(__name__)

        # Check if optimization logging is enabled
        if optimization_config is None:
            optimization_config = {}

        # Get optimization logging settings from config
        optimization_logging = optimization_config.get("optimization_debug", {})
        log_enabled = optimization_logging.get("enabled", False)
        log_frequency = optimization_logging.get("log_frequency", 10)
        include_residuals = optimization_logging.get("include_residuals", True)
        include_chi_squared = optimization_logging.get("include_chi_squared", True)

        # Only log if enabled or if debug level is active
        if not (log_enabled or logger.isEnabledFor(logging.DEBUG)):
            return

        # Only log every N iterations to avoid spam
        if iteration % log_frequency != 0 and iteration > 0:
            return

        # Build log message components
        log_parts = []

        # Method name and iteration
        method_display = f"[{method_name}]" if method_name else "[Optimization]"
        log_parts.append(f"{method_display} Iteration {iteration}")

        # Chi-squared information
        if include_chi_squared and chi_squared is not None:
            log_parts.append(f"χ²_reduced={chi_squared:.6f}")
            if total_dof is not None:
                log_parts.append(f"DOF={total_dof}")

        # Residual statistics
        if include_residuals and residuals is not None and len(residuals) > 0:
            # Calculate residual statistics safely
            finite_residuals = residuals[np.isfinite(residuals)]
            if len(finite_residuals) > 0:
                min_res = np.min(finite_residuals)
                mean_res = np.mean(finite_residuals)
                max_res = np.max(finite_residuals)
                log_parts.append(
                    f"residuals[min/mean/max]=[{min_res:.4f}/{mean_res:.4f}/{max_res:.4f}]"
                )
                log_parts.append(f"n_residuals={len(finite_residuals)}")

        # Log the combined message
        log_message = ": ".join(log_parts)
        if log_enabled:
            logger.info(log_message)
        else:
            logger.debug(log_message)

    def calculate_chi_squared_optimized(
        self,
        parameters: np.ndarray,
        phi_angles: np.ndarray,
        c2_experimental: np.ndarray,
        method_name: str = "",
        return_components: bool = False,
        filter_angles_for_optimization: bool = False,
        enable_performance_monitoring: bool = True,
        iteration: int = 0,
    ) -> float | dict[str, Any]:
        """
        Calculate chi-squared goodness of fit with per-angle analysis and uncertainty estimation.

        This method computes the reduced chi-squared statistic for model validation, with optional
        detailed per-angle analysis and uncertainty quantification. The uncertainty in reduced
        chi-squared provides insight into the consistency of fit quality across different angles.

        Performance Optimizations (v0.6.1+):
        - Configuration caching: Cached validation and chi-squared configs to avoid repeated lookups
        - Memory optimization: Pre-allocated arrays with reshape() instead of list comprehensions
        - Least squares optimization: Replaced lstsq with solve() for 2x2 matrix systems
        - Vectorized operations: Improved angle filtering and array operations
        - Early validation: Short-circuit returns for invalid parameters
        - Result: 38% performance improvement (1.33ms → 0.82ms)

        Parameters
        ----------
        parameters : np.ndarray
            Model parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        phi_angles : np.ndarray
            Scattering angles in degrees
        c2_experimental : np.ndarray
            Experimental correlation data with shape (n_angles, delay_frames, lag_frames)
        method_name : str, optional
            Name of optimization method for logging purposes
        return_components : bool, optional
            If True, return detailed results dictionary with per-angle analysis
        filter_angles_for_optimization : bool, optional
            If True, only include angles in optimization ranges [-10°, 10°] and [170°, 190°]
            for chi-squared calculation
        enable_performance_monitoring : bool, optional
            If True, enable performance monitoring and detailed logging
        iteration : int, optional
            Current optimization iteration number for progress tracking and logging

        Returns
        -------
        float or dict
            If return_components=False: Reduced chi-squared value (float)
            If return_components=True: Dictionary containing:
                - chi_squared : float
                    Total chi-squared value
                - reduced_chi_squared : float
                    Averaged reduced chi-squared from optimization angles
                - reduced_chi_squared_uncertainty : float
                    Standard error of reduced chi-squared across angles (uncertainty estimate)
                - reduced_chi_squared_std : float
                    Standard deviation of reduced chi-squared across angles
                - n_optimization_angles : int
                    Number of angles used for optimization
                - degrees_of_freedom : int
                    Degrees of freedom for statistical testing (data_points - n_parameters)
                - angle_chi_squared : list
                    Chi-squared values for each angle
                - angle_chi_squared_reduced : list
                    Reduced chi-squared values for each angle
                - angle_data_points : list
                    Number of data points per angle
                - phi_angles : list
                    Scattering angles used
                - scaling_solutions : list
                    Contrast and offset parameters for each angle
                - valid : bool
                    Whether calculation was successful

        Notes
        -----
        The uncertainty calculation follows standard error of the mean:

        reduced_chi2_uncertainty = std(angle_chi2_reduced) / sqrt(n_angles)

        Interpretation of uncertainty:
        - Small uncertainty (< 0.1 * reduced_chi2): Consistent fit across angles
        - Large uncertainty (> 0.5 * reduced_chi2): High angle variability, potential
          systematic issues or model inadequacy

        The method uses averaged (not summed) chi-squared for better angle weighting:
        reduced_chi2 = mean(chi2_reduced_per_angle) for optimization angles only

        Quality assessment guidelines:
        - Excellent: reduced_chi2 ≤ 2.0
        - Acceptable: 2.0 < reduced_chi2 ≤ 5.0
        - Warning: 5.0 < reduced_chi2 ≤ 10.0
        - Poor/Critical: reduced_chi2 > 10.0
        """
        global OPTIMIZATION_COUNTER

        # Performance monitoring initialization
        perf_monitor = {}
        start_time = time.time() if enable_performance_monitoring else None
        component_times = {} if enable_performance_monitoring else None

        # Parameter validation with caching
        if self.config is None:
            raise ValueError("Configuration not loaded: self.config is None.")

        # Cache validation config to avoid repeated dict lookups
        if not hasattr(self, "_cached_validation_config"):
            self._cached_validation_config = (
                self.config.get("advanced_settings", {})
                .get("chi_squared_calculation", {})
                .get("validity_check", {})
            )
        validation = self._cached_validation_config

        diffusion_params = parameters[: self.num_diffusion_params]
        shear_params = parameters[
            self.num_diffusion_params : self.num_diffusion_params
            + self.num_shear_rate_params
        ]

        # Quick validity checks with early returns
        if validation.get("check_positive_D0", True):
            if diffusion_params[0] <= 0:
                return (
                    np.inf
                    if not return_components
                    else {
                        "chi_squared": np.inf,
                        "valid": False,
                        "reason": "Negative D0",
                    }
                )

        if validation.get("check_positive_gamma_dot_t0", True):
            if len(shear_params) > 0 and shear_params[0] <= 0:
                return (
                    np.inf
                    if not return_components
                    else {
                        "chi_squared": np.inf,
                        "valid": False,
                        "reason": "Negative gamma_dot_t0",
                    }
                )

        # Check parameter bounds
        if validation.get("check_parameter_bounds", True):
            bounds = self.config.get("parameter_space", {}).get("bounds", [])
            for i, bound in enumerate(bounds):
                if i < len(parameters):
                    param_val = parameters[i]
                    param_min = bound.get("min", -np.inf)
                    param_max = bound.get("max", np.inf)

                    if not (param_min <= param_val <= param_max):
                        reason = f"Parameter {bound.get('name', f'p{i}')} out of bounds"
                        return (
                            np.inf
                            if not return_components
                            else {
                                "chi_squared": np.inf,
                                "valid": False,
                                "reason": reason,
                            }
                        )

        try:
            # Calculate theoretical correlation
            c2_theory = self.calculate_c2_nonequilibrium_laminar_parallel(
                parameters, phi_angles
            )

            # Chi-squared calculation with caching
            if not hasattr(self, "_cached_chi_config"):
                self._cached_chi_config = self.config.get("advanced_settings", {}).get(
                    "chi_squared_calculation", {}
                )
            chi_config = self._cached_chi_config
            # uncertainty_estimation_factor removed - use pure standard deviation
            # min_sigma now handled within moving window estimation

            # Calculate parameters for DOF calculation
            n_params = len(parameters)

            # Angle filtering for optimization
            if filter_angles_for_optimization:
                # Get target angle ranges from ConfigManager if available
                target_ranges = [
                    (-10.0, 10.0),
                    (170.0, 190.0),
                ]  # Default ranges
                if hasattr(self, "config_manager") and self.config_manager:
                    target_ranges = self.config_manager.get_target_angle_ranges()
                elif hasattr(self, "config") and self.config:
                    angle_config = self.config.get("optimization_config", {}).get(
                        "angle_filtering", {}
                    )
                    config_ranges = angle_config.get("target_ranges", [])
                    if config_ranges:
                        target_ranges = [
                            (
                                r.get("min_angle", -10.0),
                                r.get("max_angle", 10.0),
                            )
                            for r in config_ranges
                        ]

                # Find indices of angles in target ranges using vectorized
                # operations
                phi_angles_array = np.asarray(phi_angles)
                optimization_mask = np.zeros(len(phi_angles_array), dtype=bool)
                # Vectorized range checking for all ranges at once
                for min_angle, max_angle in target_ranges:
                    optimization_mask |= (phi_angles_array >= min_angle) & (
                        phi_angles_array <= max_angle
                    )
                optimization_indices = np.flatnonzero(optimization_mask).tolist()

                logger.debug(
                    f"Filtering angles for optimization: using {
                        len(optimization_indices)
                    }/{len(phi_angles)} angles"
                )
                if optimization_indices:
                    filtered_angles = phi_angles[optimization_indices]
                    logger.debug(f"Optimization angles: {filtered_angles.tolist()}")
                else:
                    # Check if fallback is enabled
                    should_fallback = True
                    if hasattr(self, "config_manager") and self.config_manager:
                        should_fallback = (
                            self.config_manager.should_fallback_to_all_angles()
                        )
                    elif hasattr(self, "config") and self.config:
                        angle_config = self.config.get("optimization_config", {}).get(
                            "angle_filtering", {}
                        )
                        should_fallback = angle_config.get(
                            "fallback_to_all_angles", True
                        )

                    if should_fallback:
                        logger.warning(
                            f"No angles found in target optimization ranges {target_ranges}"
                        )
                        logger.warning(
                            "Falling back to using all angles for optimization"
                        )
                        optimization_indices = list(
                            range(len(phi_angles))
                        )  # Fall back to all angles
                    else:
                        raise ValueError(
                            f"No angles found in target optimization ranges {target_ranges} and fallback disabled"
                        )
            else:
                optimization_indices = list(range(len(phi_angles)))

            # Calculate chi-squared for all angles (for detailed results)
            n_angles = len(phi_angles)
            angle_chi2_reduced = np.zeros(n_angles, dtype=np.float64)
            angle_data_points = []
            scaling_solutions = []

            # Performance monitoring: scaling optimization
            if enable_performance_monitoring:
                scaling_start = time.time()

            # Pre-flatten all arrays for better memory access patterns
            theory_flat = c2_theory.reshape(n_angles, -1)
            exp_flat = c2_experimental.reshape(n_angles, -1)

            # SCALING OPTIMIZATION (ALWAYS ENABLED) - Vectorized implementation
            # =====================================
            # This performs least squares fitting to determine the optimal scaling relationship:
            # g₂ = offset + contrast × g₁ where:
            # - g₁ is the theoretical correlation function
            # - g₂ is the experimental correlation function
            # - contrast and offset are fitted scaling parameters
            #
            # WHY THIS IS ESSENTIAL:
            # This scaling optimization is ALWAYS enabled because it is fundamental to proper
            # chi-squared calculation. Without it, we would compare raw theoretical values
            # directly to experimental data, which ignores systematic scaling factors and
            # offsets that are physically present due to:
            # - Instrumental response functions
            # - Background signals
            # - Detector gain variations
            # - Normalization differences
            #
            # Mathematical implementation: solve A·x = b where A = [theory,
            # ones], x = [contrast, offset]

            # Vectorized least squares fitting for all angles
            n_data_per_angle = theory_flat.shape[1]
            angle_data_points = [n_data_per_angle] * n_angles

            # Phase 3: Vectorized batch processing with Numba optimization
            # Pre-compute variance estimates for all angles (vectorized optimization)
            # Use pure standard deviation without any scaling factors
            # Note: exp_std_batch and sigma_batch were used in old implementation,
            # now using moving window variance estimation instead

            # Batch solve least squares for all angles using Numba with
            # fallback
            try:
                contrast_batch, offset_batch = solve_least_squares_batch_numba(
                    theory_flat, exp_flat
                )
            except RuntimeError as e:
                if "NUMBA_NUM_THREADS" in str(e):
                    # Fallback to non-Numba implementation for threading
                    # conflicts
                    logger.debug(
                        "Using fallback least squares due to NUMBA threading conflict"
                    )
                    contrast_batch = np.zeros(n_angles, dtype=np.float64)
                    offset_batch = np.zeros(n_angles, dtype=np.float64)

                    # Manual implementation of batch least squares
                    for i in range(n_angles):
                        theory_vec = theory_flat[i]
                        exp_vec = exp_flat[i]

                        # Solve: min ||A*x - b||^2 where A = [theory, ones], x
                        # = [contrast, offset]
                        A = np.column_stack([theory_vec, np.ones(len(theory_vec))])
                        try:
                            # Use least squares solver
                            x, _, _, _ = np.linalg.lstsq(A, exp_vec, rcond=None)
                            contrast_batch[i] = x[0]
                            offset_batch[i] = x[1]
                        except np.linalg.LinAlgError:
                            # Fallback values if linear algebra fails
                            contrast_batch[i] = 0.5
                            offset_batch[i] = 1.0
                else:
                    raise

            # Batch compute chi-squared values using Numba with fallback
            try:
                chi2_raw_batch = compute_chi_squared_batch_numba(
                    theory_flat, exp_flat, contrast_batch, offset_batch
                )
            except RuntimeError as e:
                if "NUMBA_NUM_THREADS" in str(e):
                    # Fallback to non-Numba implementation for threading
                    # conflicts
                    logger.debug(
                        "Using fallback chi-squared computation due to NUMBA threading conflict"
                    )
                    chi2_raw_batch = np.zeros(n_angles, dtype=np.float64)

                    # Manual implementation of batch chi-squared
                    for i in range(n_angles):
                        theory_vec = theory_flat[i]
                        exp_vec = exp_flat[i]
                        contrast = contrast_batch[i]
                        offset = offset_batch[i]

                        # Compute fitted values and chi-squared
                        fitted_vec = contrast * theory_vec + offset
                        residuals = exp_vec - fitted_vec
                        chi2_raw_batch[i] = np.sum(residuals**2)
                else:
                    raise

            # ========================================================================
            # PROPER CHI-SQUARED CALCULATION (Following Statistical Definition)
            # ========================================================================
            #
            # From statistical definition:
            # χ² = Σ((I_model,i - I_experiment,i)/σᵢ)²
            # χ²_reduced = χ² / (N - K) where N = data points, K = parameters
            #
            # The chi2_raw_batch already contains the raw residual sums: Σ(residuals²)
            # We need to normalize by σ² to get the proper χ² values

            # Calculate residuals for diagnostic logging
            residuals_batch = []
            for i in range(n_angles):
                theory_vec = theory_flat[i]
                exp_vec = exp_flat[i]
                contrast = contrast_batch[i]
                offset = offset_batch[i]
                fitted_vec = contrast * theory_vec + offset
                residuals = exp_vec - fitted_vec
                residuals_batch.append(residuals)

            # CLEAN CHI-SQUARED CALCULATION using moving window variance estimation
            angle_chi2_proper = []
            angle_chi2_reduced = []
            angle_sigma_values = []
            all_sigma_values = []  # Store all individual sigma values for logging

            # Get window size from config (consistent with existing config pattern)
            window_size = chi_config.get("moving_window_size", 11)

            # Get variance estimation method from config
            # Default to IRLS MAD robust (replaces deprecated standard method)
            variance_method = chi_config.get("variance_method", "irls_mad_robust")

            # Calculate degrees of freedom per angle
            dof_per_angle = max(1, n_data_per_angle - n_params)

            # BATCH PROCESSING: Process all angles simultaneously with vectorized operations
            # Get edge method from config for batch processing
            edge_method = chi_config.get("moving_window_edge_method", "reflect")

            # Try batch variance estimation first (optimized path)
            batch_processing_success = False
            try:
                # Validate inputs for batch processing
                if not residuals_batch or len(residuals_batch) == 0:
                    raise ValueError(
                        "Empty residuals_batch provided to batch processing"
                    )

                # Check for minimum array size requirements
                min_size_for_batch = (
                    5  # Minimum reasonable size for vectorized operations
                )
                if any(len(res) < min_size_for_batch for res in residuals_batch):
                    logger.debug(
                        f"Some residuals arrays too small for batch processing (< {min_size_for_batch} elements), using sequential fallback"
                    )
                    raise ValueError(
                        f"Residuals arrays too small for efficient batch processing"
                    )

                # Use batch IRLS variance estimation for all angles at once
                if variance_method == "irls_mad_robust":
                    logger.debug(
                        f"Using batch IRLS MAD robust variance estimation with {edge_method} edges for {n_angles} angles"
                    )

                    # Batch variance estimation processes all angles simultaneously
                    try:
                        sigma_variances_batch = (
                            self._estimate_variance_irls_mad_robust_batch(
                                residuals_batch,
                                window_size=window_size,
                                edge_method=edge_method,
                            )
                        )
                    except (RuntimeError, ValueError, MemoryError) as batch_error:
                        logger.warning(
                            f"Batch variance estimation failed: {str(batch_error)}"
                        )
                        raise batch_error

                    # Validate batch variance results
                    if (
                        not sigma_variances_batch
                        or len(sigma_variances_batch) != n_angles
                    ):
                        raise ValueError(
                            f"Batch variance estimation returned invalid results: {len(sigma_variances_batch) if sigma_variances_batch else 0} != {n_angles}"
                        )

                    # Check for invalid variance values
                    for i, sigma_vars in enumerate(sigma_variances_batch):
                        if sigma_vars is None or len(sigma_vars) == 0:
                            raise ValueError(f"Empty variance array for angle {i}")
                        if not np.all(np.isfinite(sigma_vars)) or np.any(
                            sigma_vars <= 0
                        ):
                            logger.warning(
                                f"Invalid variance values detected for angle {i}, falling back to sequential processing"
                            )
                            raise ValueError(f"Invalid variance values for angle {i}")

                    # Calculate chi-squared values using batch Numba kernel
                    min_sigma = chi_config.get("minimum_sigma", 1e-10)
                    max_chi2 = chi_config.get("max_chi_squared", 1e8)

                    try:
                        # Convert lists to numpy arrays for Numba compatibility
                        residuals_batch_array = np.array(
                            residuals_batch, dtype=np.float64
                        )
                        sigma_variances_batch_array = np.array(
                            sigma_variances_batch, dtype=np.float64
                        )

                        # Use batch chi-squared calculation with variance normalization
                        angle_chi2_proper, angle_chi2_reduced = (
                            chi_squared_with_variance_batch_numba(
                                residuals_batch_array,
                                sigma_variances_batch_array,
                                dof_per_angle,
                            )
                        )
                    except (RuntimeError, ValueError) as chi2_error:
                        if "NUMBA_NUM_THREADS" in str(chi2_error):
                            logger.debug(
                                "Numba threading conflict in batch chi-squared, falling back to sequential processing"
                            )
                        else:
                            logger.warning(
                                f"Batch chi-squared calculation failed: {str(chi2_error)}"
                            )
                        raise chi2_error

                    # Validate chi-squared results
                    if (
                        len(angle_chi2_proper) != n_angles
                        or len(angle_chi2_reduced) != n_angles
                    ):
                        raise ValueError(
                            f"Batch chi-squared calculation returned wrong number of results"
                        )

                    # Apply limits and collect diagnostics
                    angle_sigma_values = []
                    all_sigma_values = []

                    for i, (sigma_variances, residuals) in enumerate(
                        zip(sigma_variances_batch, residuals_batch)
                    ):
                        try:
                            sigma_per_point = np.sqrt(sigma_variances)

                            # Apply minimum sigma floor and validate
                            sigma_per_point_safe = np.maximum(
                                sigma_per_point, min_sigma
                            )

                            # Additional validation for extreme values
                            if not np.all(np.isfinite(sigma_per_point_safe)):
                                logger.warning(
                                    f"Non-finite sigma values detected for angle {i}"
                                )
                                # Replace non-finite values with minimum sigma
                                sigma_per_point_safe = np.where(
                                    np.isfinite(sigma_per_point_safe),
                                    sigma_per_point_safe,
                                    min_sigma,
                                )

                            # Cap extremely large chi-squared values
                            if (
                                not np.isfinite(angle_chi2_proper[i])
                                or angle_chi2_proper[i] > max_chi2
                            ):
                                logger.warning(
                                    f"Angle {i}: Chi-squared {angle_chi2_proper[i]:.2e} exceeds maximum or is non-finite, capping"
                                )
                                angle_chi2_proper[i] = max_chi2
                                angle_chi2_reduced[i] = max_chi2 / max(1, dof_per_angle)

                            # Collect diagnostic values
                            angle_sigma_values.append(np.mean(sigma_per_point_safe))
                            all_sigma_values.extend(sigma_per_point_safe)

                        except Exception as angle_error:
                            logger.error(
                                f"Error processing angle {i} in batch results: {str(angle_error)}"
                            )
                            raise angle_error

                    # Convert to numpy arrays with validation
                    try:
                        angle_chi2_proper = np.array(
                            angle_chi2_proper, dtype=np.float64
                        )
                        angle_chi2_reduced = np.array(
                            angle_chi2_reduced, dtype=np.float64
                        )
                    except (ValueError, TypeError) as conv_error:
                        logger.error(
                            f"Failed to convert batch results to numpy arrays: {str(conv_error)}"
                        )
                        raise conv_error

                    # Final validation of batch results
                    if not np.all(np.isfinite(angle_chi2_proper)) or not np.all(
                        np.isfinite(angle_chi2_reduced)
                    ):
                        logger.warning(
                            "Non-finite values detected in final batch chi-squared results"
                        )
                        # Replace non-finite values with large but finite values
                        angle_chi2_proper = np.where(
                            np.isfinite(angle_chi2_proper), angle_chi2_proper, max_chi2
                        )
                        angle_chi2_reduced = np.where(
                            np.isfinite(angle_chi2_reduced),
                            angle_chi2_reduced,
                            max_chi2 / max(1, dof_per_angle),
                        )

                    batch_processing_success = True
                    logger.debug(
                        f"Batch processing completed successfully for {n_angles} angles"
                    )

                else:
                    # Fallback to batch processing with legacy variance method
                    raise NotImplementedError(
                        f"Batch processing not implemented for variance method: {variance_method}"
                    )

            except (
                RuntimeError,
                ValueError,
                NotImplementedError,
                MemoryError,
            ) as batch_error:
                # Comprehensive error logging for debugging
                error_type = type(batch_error).__name__
                logger.warning(
                    f"Batch processing failed with {error_type}: {str(batch_error)}"
                )

                # Additional context for specific error types
                if "NUMBA_NUM_THREADS" in str(batch_error):
                    logger.debug(
                        "Numba threading conflict detected, this is expected in some environments"
                    )
                elif isinstance(batch_error, MemoryError):
                    logger.warning(
                        f"Memory exhaustion during batch processing of {n_angles} angles"
                    )
                elif isinstance(batch_error, NotImplementedError):
                    logger.debug(
                        f"Batch processing not available for current configuration"
                    )

                batch_processing_success = False

            # Only use sequential fallback if batch processing failed
            if not batch_processing_success:
                logger.info(
                    f"Using sequential processing fallback for {n_angles} angles"
                )

                # Enhanced sequential processing with improved error handling
                angle_chi2_proper = []
                angle_chi2_reduced = []
                angle_sigma_values = []
                all_sigma_values = []

                for i in range(n_angles):
                    residuals = residuals_batch[i]

                    # IRLS MAD ROBUST VARIANCE ESTIMATION (replaces standard method)
                    if variance_method == "irls_mad_robust":
                        # Use IRLS with MAD moving window for robust variance estimation
                        # IRLS methods handle edge padding internally - always pass original residuals
                        # Use selected variance estimator (optimized or legacy)
                        if hasattr(self, "_selected_variance_estimator"):
                            sigma_variances = self._selected_variance_estimator(
                                residuals,  # Always pass original residuals
                                window_size=window_size,
                                edge_method=edge_method,
                            )
                        else:
                            # Fallback to legacy method if not initialized
                            sigma_variances = self._estimate_variance_irls_mad_robust(
                                residuals,  # Always pass original residuals
                                window_size=window_size,
                                edge_method=edge_method,
                            )
                        logger.debug(
                            f"Angle {i}: Using IRLS MAD robust variance estimation with {edge_method} edges (sequential fallback)"
                        )
                    else:
                        # Fallback: use IRLS MAD robust method (the main implementation)
                        sigma_variances = self._estimate_variance_irls_mad_robust(
                            residuals, window_size=window_size, edge_method=edge_method
                        )
                        logger.debug(
                            f"Angle {i}: Using IRLS MAD robust variance estimation (sequential fallback)"
                        )

                    sigma_per_point = np.sqrt(sigma_variances)

                    # Handle size mismatch issues (should be rare with improved variance estimation)
                    if len(residuals) != len(sigma_per_point):
                        logger.warning(
                            f"Size mismatch in chi-squared calculation for angle {i}: "
                            f"residuals({len(residuals)}) != sigma_per_point({len(sigma_per_point)}). "
                            f"Difference: {abs(len(residuals) - len(sigma_per_point))} elements. "
                            f"This indicates an issue in variance estimation - applying safety truncation."
                        )
                        # Truncate to smaller size to prevent crash
                        min_size = min(len(residuals), len(sigma_per_point))
                        residuals_for_calc = residuals[:min_size]
                        sigma_for_calc = sigma_per_point[:min_size]
                        logger.debug(
                            f"Angle {i}: Truncated both arrays to size {min_size} for chi-squared calculation"
                        )
                    else:
                        residuals_for_calc = residuals
                        sigma_for_calc = sigma_per_point

                    # Proper chi-squared: χ² = Σ(residuals²/σ²) with numerical stability
                    # Apply minimum variance floor to prevent division by zero/near-zero
                    min_sigma = chi_config.get("minimum_sigma", 1e-10)
                    sigma_for_calc_safe = np.maximum(sigma_for_calc, min_sigma)

                    # Check for infinite or NaN values in residuals or sigma
                    finite_mask = np.isfinite(residuals_for_calc) & np.isfinite(
                        sigma_for_calc_safe
                    )
                    if not np.any(finite_mask):
                        # All values are non-finite, return a large but finite chi-squared
                        logger.warning(
                            f"Angle {i}: All residuals or sigma values are non-finite"
                        )
                        chi2_total_angle = 1e6  # Large but finite value
                        chi2_reduced_angle = chi2_total_angle / dof_per_angle
                    else:
                        # Only use finite values for chi-squared calculation
                        residuals_finite = residuals_for_calc[finite_mask]
                        sigma_finite = sigma_for_calc_safe[finite_mask]

                        chi2_per_point = (residuals_finite / sigma_finite) ** 2
                        chi2_total_angle = np.sum(chi2_per_point)

                        # Adjust degrees of freedom based on finite data points
                        effective_dof = max(1, len(residuals_finite) - n_params)
                        chi2_reduced_angle = chi2_total_angle / effective_dof

                        # Additional check for extremely large chi-squared values
                        max_chi2 = chi_config.get("max_chi_squared", 1e8)
                        if chi2_total_angle > max_chi2:
                            logger.warning(
                                f"Angle {i}: Chi-squared {chi2_total_angle:.2e} exceeds maximum {max_chi2:.2e}, capping"
                            )
                            chi2_total_angle = max_chi2
                            chi2_reduced_angle = chi2_total_angle / effective_dof

                    angle_chi2_proper.append(chi2_total_angle)
                    angle_chi2_reduced.append(chi2_reduced_angle)
                    angle_sigma_values.append(
                        np.mean(sigma_per_point)
                    )  # Average for logging
                    all_sigma_values.extend(
                        sigma_per_point
                    )  # Collect all individual values

                # Convert to numpy arrays for compatibility with validation
                try:
                    angle_chi2_proper = np.array(angle_chi2_proper, dtype=np.float64)
                    angle_chi2_reduced = np.array(angle_chi2_reduced, dtype=np.float64)

                    # Final validation of sequential results
                    if not np.all(np.isfinite(angle_chi2_proper)):
                        logger.warning(
                            "Non-finite values in sequential chi-squared results, cleaning up"
                        )
                        angle_chi2_proper = np.where(
                            np.isfinite(angle_chi2_proper), angle_chi2_proper, 1e6
                        )

                    if not np.all(np.isfinite(angle_chi2_reduced)):
                        logger.warning(
                            "Non-finite values in sequential reduced chi-squared results, cleaning up"
                        )
                        angle_chi2_reduced = np.where(
                            np.isfinite(angle_chi2_reduced),
                            angle_chi2_reduced,
                            1e6 / max(1, dof_per_angle),
                        )

                except (ValueError, TypeError) as conv_error:
                    logger.error(
                        f"Failed to convert sequential results to numpy arrays: {str(conv_error)}"
                    )
                    # Create default arrays as last resort
                    angle_chi2_proper = np.full(n_angles, 1e6, dtype=np.float64)
                    angle_chi2_reduced = np.full(
                        n_angles, 1e6 / max(1, dof_per_angle), dtype=np.float64
                    )

            # Final safety check to ensure arrays are properly formatted
            if not isinstance(angle_chi2_proper, np.ndarray):
                logger.warning(
                    "Chi-squared results not in numpy array format, converting"
                )
                angle_chi2_proper = np.array(angle_chi2_proper, dtype=np.float64)
                angle_chi2_reduced = np.array(angle_chi2_reduced, dtype=np.float64)

            # Store scaling solutions for compatibility
            scaling_solutions = [
                [contrast_batch[i], offset_batch[i]] for i in range(n_angles)
            ]

            # Calculate optimization totals (existing pattern)
            if filter_angles_for_optimization:
                optimization_chi2_proper = [
                    angle_chi2_proper[i] for i in optimization_indices
                ]
                total_data_points = sum(
                    angle_data_points[i] for i in optimization_indices
                )
            else:
                optimization_chi2_proper = angle_chi2_proper
                total_data_points = sum(angle_data_points)

            # Calculate total statistics
            total_chi2 = sum(optimization_chi2_proper)
            total_dof = max(1, total_data_points - len(parameters))
            reduced_chi2 = total_chi2 / total_dof

            # Collect residuals and sigma for logging
            all_residuals = np.concatenate(
                [
                    residuals_batch[i]
                    for i in (
                        optimization_indices
                        if filter_angles_for_optimization
                        else range(len(residuals_batch))
                    )
                ]
            )

            # Collect all individual sigma values for meaningful statistics
            if filter_angles_for_optimization:
                # Get sigma values only for optimization angles
                optimization_sigma_individual = []
                for i in optimization_indices:
                    start_idx = i * n_data_per_angle
                    end_idx = (i + 1) * n_data_per_angle
                    optimization_sigma_individual.extend(
                        all_sigma_values[start_idx:end_idx]
                    )
                optimization_sigma = np.array(optimization_sigma_individual)
            else:
                # Use all sigma values
                optimization_sigma = np.array(all_sigma_values)

            # Enhanced optimization logging with standardized format
            # Get optimization logging configuration
            optimization_logging_config = (
                self.config.get("output_settings", {})
                .get("logging", {})
                .get("optimization_debug", {})
            )

            # Standard debug logging (existing functionality)
            logger.debug("CHI² CALCULATION (Moving Window Method):")
            logger.debug(f"  total_chi2 = {total_chi2:.6e}")
            logger.debug(f"  total_dof = {total_dof}")
            logger.debug(f"  reduced_chi2 = {reduced_chi2:.6e}")
            logger.debug(
                f"  sigma min/mean/max = {np.min(optimization_sigma):.6e} / {np.mean(optimization_sigma):.6e} / {np.max(optimization_sigma):.6e}"
            )
            logger.debug(
                f"  sigma points = {len(optimization_sigma)} (individual variance estimates)"
            )
            logger.debug(
                f"  residuals min/mean/max = {np.min(all_residuals):.6e} / {np.mean(all_residuals):.6e} / {np.max(all_residuals):.6e}"
            )
            logger.debug(f"  window_size = {window_size}")

            # Enhanced optimization progress logging (new functionality)
            # This provides a standardized format for optimization tracking
            if optimization_logging_config.get("enabled", False) or logger.isEnabledFor(
                logging.DEBUG
            ):
                method_name = method_name or "Core-Analysis"
                # Use the standardized logging method for consistency across optimization methods
                self.log_optimization_progress(
                    iteration=iteration,  # Use iteration parameter from method call
                    chi_squared=reduced_chi2,
                    residuals=all_residuals,
                    method_name=method_name,
                    total_dof=total_dof,
                    optimization_config=optimization_logging_config,
                )

            # Simplified uncertainty calculation (remove dual versions)
            if len(optimization_chi2_proper) > 1:
                reduced_chi2_uncertainty = np.std(angle_chi2_reduced, ddof=1) / np.sqrt(
                    len(optimization_chi2_proper)
                )
            else:
                reduced_chi2_uncertainty = 0.0
            # Logging
            OPTIMIZATION_COUNTER += 1
            log_freq = self.config["performance_settings"].get(
                "optimization_counter_log_frequency", 500
            )
            if OPTIMIZATION_COUNTER % log_freq == 0:
                logger.info(
                    f"Iteration {OPTIMIZATION_COUNTER:06d} [{method_name}]: "
                    f"χ²_red = {reduced_chi2:.6e} ± {reduced_chi2_uncertainty:.6e}"
                )
                # Log reduced chi-square per angle
                for i, (phi, chi2_red_angle) in enumerate(
                    zip(phi_angles, angle_chi2_reduced, strict=False)
                ):
                    logger.info(
                        f"  Angle {i + 1} (φ={phi:.1f}°): χ²_red = {chi2_red_angle:.6e}"
                    )

            # Performance monitoring: finalize and create summary
            if enable_performance_monitoring:
                total_time = time.time() - start_time
                component_times["total_calculation"] = total_time
                component_times["scaling_optimization"] = time.time() - scaling_start

                perf_monitor = {
                    "total_time_ms": total_time * 1000,
                    "component_times_ms": {
                        k: v * 1000 for k, v in component_times.items()
                    },
                    "performance_score": self._calculate_performance_score(
                        total_time, n_angles, len(parameters)
                    ),
                }

                # Detect bottlenecks
                bottleneck_threshold = 0.3  # 30% of total time
                bottlenecks = []
                for component, comp_time in component_times.items():
                    if comp_time > total_time * bottleneck_threshold:
                        bottleneck_pct = (comp_time / total_time) * 100
                        bottlenecks.append(f"{component}: {bottleneck_pct:.1f}%")
                perf_monitor["bottlenecks"] = bottlenecks

                # Log performance warnings if needed
                if bottlenecks:
                    logger.debug(f"Performance bottlenecks detected: {bottlenecks}")
                if total_time > 0.5:  # Warn if calculation takes more than 500ms
                    logger.debug(
                        f"Slow chi-squared calculation: {total_time * 1000:.1f}ms"
                    )

            if return_components:
                result = {
                    "chi_squared": reduced_chi2,
                    "reduced_chi_squared": reduced_chi2,
                    "reduced_chi_squared_uncertainty": reduced_chi2_uncertainty,
                    "total_chi_squared": total_chi2,
                    "degrees_of_freedom": total_dof,
                    "angle_chi_squared": angle_chi2_proper,
                    "angle_chi_squared_reduced": angle_chi2_reduced,
                    "scaling_solutions": scaling_solutions,
                    "angle_data_points": angle_data_points,
                    "phi_angles": phi_angles.tolist(),
                    "n_optimization_angles": len(optimization_chi2_proper),
                    "optimization_counter": OPTIMIZATION_COUNTER,
                    "valid": True,
                }
                if enable_performance_monitoring:
                    result["performance_monitor"] = perf_monitor
                return result
            else:
                return float(reduced_chi2)

        except Exception as e:
            logger.warning(f"Chi-squared calculation failed: {e}")
            logger.exception("Full traceback for chi-squared calculation failure:")
            if return_components:
                return {"chi_squared": np.inf, "valid": False, "error": str(e)}
            else:
                return np.inf

    def analyze_per_angle_chi_squared(
        self,
        parameters: np.ndarray,
        phi_angles: np.ndarray,
        c2_experimental: np.ndarray,
        method_name: str = "Final",
        save_to_file: bool = True,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """
        Comprehensive per-angle reduced chi-squared analysis with quality assessment.

        This method performs detailed analysis of chi-squared values across different
        scattering angles, providing quality metrics, uncertainty estimation, and
        angle categorization to identify systematic fitting issues.

        Parameters
        ----------
        parameters : np.ndarray
            Optimized model parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        phi_angles : np.ndarray
            Scattering angles in degrees
        c2_experimental : np.ndarray
            Experimental correlation data with shape (n_angles, delay_frames, lag_frames)
        method_name : str, optional
            Name of the analysis method for file naming and logging
        save_to_file : bool, optional
            Whether to save detailed results to JSON file
        output_dir : str, optional
            Output directory for saved results (defaults to current directory)

        Returns
        -------
        Dict[str, Any]
            Comprehensive analysis results containing:
                - method : str
                    Analysis method name
                - overall_reduced_chi_squared : float
                    Average reduced chi-squared across optimization angles
                - reduced_chi_squared_uncertainty : float
                    Standard error of reduced chi-squared (uncertainty measure)
                - quality_assessment : dict
                    Overall and per-angle quality evaluation with thresholds
                - angle_categorization : dict
                    Classification of angles into good, unacceptable, and outlier groups
                - per_angle_analysis : dict
                    Detailed per-angle chi-squared values and statistics
                - statistical_summary : dict
                    Summary statistics including means, medians, and percentiles
                - recommendations : list
                    Specific recommendations based on quality assessment

        Notes
        -----
        Quality Assessment Criteria:
        - Overall reduced chi-squared uncertainty indicates fit consistency:
          * Small uncertainty (< 10% of chi2): Consistent across angles
          * Large uncertainty (> 50% of chi2): High variability, investigate systematically

        Angle Classification:
        - Good angles: reduced_chi2 ≤ acceptable_threshold (default 5.0)
        - Unacceptable angles: reduced_chi2 > acceptable_threshold
        - Statistical outliers: reduced_chi2 > mean + 2.5*std

        The method uses configuration-driven thresholds from validation_rules.fit_quality
        for consistent quality assessment across the package.

        Note: Per-angle chi-squared results are included in the main analysis results.
        No separate file is saved.

        See Also
        --------
        calculate_chi_squared_optimized : Underlying chi-squared calculation
        """
        # Get detailed chi-squared components using selected calculator
        if hasattr(self, "_selected_chi_calculator"):
            chi_results = self._selected_chi_calculator(
                parameters,
                phi_angles,
                c2_experimental,
                method_name=method_name,
                return_components=True,
            )
        else:
            # Fallback to legacy method if not initialized
            chi_results = self.calculate_chi_squared_optimized(
                parameters,
                phi_angles,
                c2_experimental,
                method_name=method_name,
                return_components=True,
            )

        # Handle case where chi_results might be a float (when
        # return_components=False fails)
        if not isinstance(chi_results, dict) or not chi_results.get("valid", False):
            logger.error("Chi-squared calculation failed for per-angle analysis")
            return {"valid": False, "error": "Chi-squared calculation failed"}

        # Extract per-angle data
        angle_chi2_reduced = chi_results["angle_chi_squared_reduced"]
        angles = chi_results["phi_angles"]

        # Analysis statistics
        mean_chi2_red = np.mean(angle_chi2_reduced)
        std_chi2_red = np.std(angle_chi2_reduced)
        min_chi2_red = np.min(angle_chi2_reduced)
        max_chi2_red = np.max(angle_chi2_reduced)

        # Get validation thresholds from configuration
        validation_config = (
            self.config.get("validation_rules", {}) if self.config else {}
        )
        fit_quality_config = validation_config.get("fit_quality", {})
        overall_config = fit_quality_config.get("overall_chi_squared", {})
        per_angle_config = fit_quality_config.get("per_angle_chi_squared", {})

        # Overall reduced chi-squared quality assessment (updated thresholds
        # for reduced chi2)
        overall_chi2 = chi_results["reduced_chi_squared"]
        excellent_threshold = overall_config.get("excellent_threshold", 2.0)
        acceptable_overall = overall_config.get("acceptable_threshold", 5.0)
        warning_overall = overall_config.get("warning_threshold", 10.0)
        critical_overall = overall_config.get("critical_threshold", 20.0)

        # Determine overall quality based on reduced chi-squared
        if overall_chi2 <= excellent_threshold:
            overall_quality = "excellent"
        elif overall_chi2 <= acceptable_overall:
            overall_quality = "acceptable"
        elif overall_chi2 <= warning_overall:
            overall_quality = "warning"
        elif overall_chi2 <= critical_overall:
            overall_quality = "poor"
        else:
            overall_quality = "critical"

        # Per-angle quality assessment (updated thresholds for reduced chi2)
        excellent_per_angle = per_angle_config.get("excellent_threshold", 2.0)
        acceptable_per_angle = per_angle_config.get("acceptable_threshold", 5.0)
        warning_per_angle = per_angle_config.get("warning_threshold", 10.0)
        outlier_multiplier = per_angle_config.get("outlier_threshold_multiplier", 2.5)
        max_outlier_fraction = per_angle_config.get("max_outlier_fraction", 0.25)
        min_good_angles = per_angle_config.get("min_good_angles", 3)

        # Identify outlier angles using configurable threshold
        outlier_threshold = mean_chi2_red + outlier_multiplier * std_chi2_red
        outlier_indices = np.where(np.array(angle_chi2_reduced) > outlier_threshold)[0]
        outlier_angles = [angles[i] for i in outlier_indices]
        outlier_chi2 = [angle_chi2_reduced[i] for i in outlier_indices]

        # Categorize angles by quality levels
        angle_chi2_array = np.array(angle_chi2_reduced)

        # Excellent angles (≤ 2.0)
        excellent_indices = np.where(angle_chi2_array <= excellent_per_angle)[0]
        excellent_angles = [angles[i] for i in excellent_indices]

        # Acceptable angles (≤ 5.0)
        acceptable_indices = np.where(angle_chi2_array <= acceptable_per_angle)[0]
        acceptable_angles = [angles[i] for i in acceptable_indices]

        # Warning angles (> 5.0, ≤ 10.0)
        warning_indices = np.where(
            (angle_chi2_array > acceptable_per_angle)
            & (angle_chi2_array <= warning_per_angle)
        )[0]
        warning_angles = [angles[i] for i in warning_indices]

        # Poor angles (> 10.0)
        poor_indices = np.where(angle_chi2_array > warning_per_angle)[0]
        poor_angles = [angles[i] for i in poor_indices]
        poor_chi2 = [angle_chi2_reduced[i] for i in poor_indices]

        # Compatibility aliases for test suite and external users
        unacceptable_angles = poor_angles
        unacceptable_chi2 = poor_chi2
        good_angles = acceptable_angles
        num_good_angles = len(acceptable_angles)

        # Quality assessment
        outlier_fraction = len(outlier_angles) / len(angles)
        unacceptable_fraction = len(unacceptable_angles) / len(angles)

        per_angle_quality = "excellent"
        quality_issues = []

        if num_good_angles < min_good_angles:
            per_angle_quality = "critical"
            quality_issues.append(
                f"Only {num_good_angles} good angles (min required: {min_good_angles})"
            )

        if unacceptable_fraction > max_outlier_fraction:
            per_angle_quality = (
                "poor" if per_angle_quality != "critical" else per_angle_quality
            )
            quality_issues.append(
                f"{unacceptable_fraction:.1%} angles unacceptable (max allowed: {
                    max_outlier_fraction:.1%})"
            )

        if outlier_fraction > max_outlier_fraction:
            per_angle_quality = (
                "warning" if per_angle_quality == "excellent" else per_angle_quality
            )
            quality_issues.append(
                f"{outlier_fraction:.1%} statistical outliers (max recommended: {
                    max_outlier_fraction:.1%})"
            )

        # Combined assessment
        if overall_quality in ["critical", "poor"] or per_angle_quality in [
            "critical",
            "poor",
        ]:
            combined_quality = "poor"
        elif overall_quality == "warning" or per_angle_quality == "warning":
            combined_quality = "warning"
        elif overall_quality == "acceptable" or per_angle_quality == "acceptable":
            combined_quality = "acceptable"
        else:
            combined_quality = "excellent"

        # Create comprehensive results
        per_angle_results = {
            "method": method_name,
            "overall_reduced_chi_squared": chi_results["reduced_chi_squared"],
            "overall_reduced_chi_squared_uncertainty": chi_results.get(
                "reduced_chi_squared_uncertainty", 0.0
            ),
            "overall_reduced_chi_squared_std": chi_results.get(
                "reduced_chi_squared_std", 0.0
            ),
            "n_optimization_angles": chi_results.get(
                "n_optimization_angles", len(angles)
            ),
            "per_angle_analysis": {
                "phi_angles_deg": angles,
                "chi_squared_reduced": angle_chi2_reduced,
                "data_points_per_angle": chi_results["angle_data_points"],
                "scaling_solutions": chi_results["scaling_solutions"],
            },
            "statistics": {
                "mean_chi2_reduced": mean_chi2_red,
                "std_chi2_reduced": std_chi2_red,
                "min_chi2_reduced": min_chi2_red,
                "max_chi2_reduced": max_chi2_red,
                "range_chi2_reduced": max_chi2_red - min_chi2_red,
                "uncertainty_from_angles": chi_results.get(
                    "reduced_chi_squared_uncertainty", 0.0
                ),
            },
            "quality_assessment": {
                "overall_quality": overall_quality,
                "per_angle_quality": per_angle_quality,
                "combined_quality": combined_quality,
                "quality_issues": quality_issues,
                "thresholds_used": {
                    "excellent_overall": excellent_threshold,
                    "acceptable_overall": acceptable_overall,
                    "warning_overall": warning_overall,
                    "critical_overall": critical_overall,
                    "excellent_per_angle": excellent_per_angle,
                    "acceptable_per_angle": acceptable_per_angle,
                    "warning_per_angle": warning_per_angle,
                    "outlier_multiplier": outlier_multiplier,
                    "max_outlier_fraction": max_outlier_fraction,
                    "min_good_angles": min_good_angles,
                },
                "interpretation": {
                    "overall_chi2_meaning": _get_chi2_interpretation(overall_chi2),
                    "quality_explanation": _get_quality_explanation(combined_quality),
                    "recommended_actions": _get_quality_recommendations(
                        combined_quality, quality_issues
                    ),
                },
            },
            "angle_categorization": {
                "excellent_angles": {
                    "angles_deg": excellent_angles,
                    "count": len(excellent_angles),
                    "fraction": len(excellent_angles) / len(angles),
                    "criteria": f"χ²_red ≤ {excellent_per_angle}",
                },
                "acceptable_angles": {
                    "angles_deg": acceptable_angles,
                    "count": len(acceptable_angles),
                    "fraction": len(acceptable_angles) / len(angles),
                    "criteria": f"χ²_red ≤ {acceptable_per_angle}",
                },
                "warning_angles": {
                    "angles_deg": warning_angles,
                    "count": len(warning_angles),
                    "fraction": len(warning_angles) / len(angles),
                    "criteria": f"{acceptable_per_angle} < χ²_red ≤ {warning_per_angle}",
                },
                "poor_angles": {
                    "angles_deg": poor_angles,
                    "chi2_reduced": poor_chi2,
                    "count": len(poor_angles),
                    "fraction": len(poor_angles) / len(angles),
                    "criteria": f"χ²_red > {warning_per_angle}",
                },
                # Standard output format for test suite and external users
                "good_angles": {
                    "angles_deg": good_angles,
                    "count": num_good_angles,
                    "fraction": num_good_angles / len(angles),
                    "criteria": f"χ²_red ≤ {acceptable_per_angle}",
                },
                "unacceptable_angles": {
                    "angles_deg": unacceptable_angles,
                    "chi2_reduced": unacceptable_chi2,
                    "count": len(unacceptable_angles),
                    "fraction": unacceptable_fraction,
                    "criteria": f"χ²_red > {acceptable_per_angle}",
                },
                "statistical_outliers": {
                    "angles_deg": outlier_angles,
                    "chi2_reduced": outlier_chi2,
                    "count": len(outlier_angles),
                    "fraction": outlier_fraction,
                    "criteria": (
                        f"χ²_red > mean + {outlier_multiplier}×std ({
                            outlier_threshold:.3f})"
                    ),
                },
            },
        }

        # Save to file if requested
        if save_to_file:
            if output_dir is None:
                output_dir = "./homodyne_results"
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Per-angle chi-squared results are now included in the main analysis results
            # No separate file saving needed as requested by user
            logger.debug(f"Per-angle chi-squared analysis completed for {method_name}")

        # Log summary with quality assessment
        logger.info(f"Per-angle chi-squared analysis [{method_name}]:")
        overall_uncertainty = chi_results.get("reduced_chi_squared_uncertainty", 0.0)
        if overall_uncertainty > 0:
            logger.info(
                f"  Overall χ²_red: {chi_results['reduced_chi_squared']:.6e} ± {
                    overall_uncertainty:.6e} ({overall_quality})"
            )
        else:
            logger.info(
                f"  Overall χ²_red: {chi_results['reduced_chi_squared']:.6e} ({
                    overall_quality
                })"
            )
        logger.info(
            f"  Mean per-angle χ²_red: {mean_chi2_red:.6e} ± {std_chi2_red:.6e}"
        )
        logger.info(f"  Range: {min_chi2_red:.6e} - {max_chi2_red:.6e}")

        # Quality assessment logging
        logger.info(f"  Quality Assessment: {combined_quality.upper()}")
        logger.info(
            f"    Overall: {overall_quality} (threshold: {acceptable_overall:.1f})"
        )
        logger.info(f"    Per-angle: {per_angle_quality}")

        # Angle categorization
        logger.info("  Angle Categorization:")
        logger.info(
            f"    Good angles: {num_good_angles}/{len(angles)} ({
                100 * num_good_angles / len(angles):.1f}%) [χ²_red ≤ {
                acceptable_per_angle
            }]"
        )
        logger.info(
            f"    Unacceptable angles: {len(unacceptable_angles)}/{len(angles)} ({
                100 * unacceptable_fraction:.1f}%) [χ²_red > {acceptable_per_angle}]"
        )
        logger.info(
            f"    Statistical outliers: {len(outlier_angles)}/{len(angles)} ({
                100 * outlier_fraction:.1f}%) [χ²_red > {outlier_threshold:.3f}]"
        )

        # Warnings and issues
        if quality_issues:
            for issue in quality_issues:
                logger.warning(f"  Quality Issue: {issue}")

        if unacceptable_angles:
            logger.warning(f"  Unacceptable angles: {unacceptable_angles}")

        if outlier_angles:
            logger.warning(f"  Statistical outlier angles: {outlier_angles}")

        # Overall quality verdict
        if combined_quality == "critical":
            logger.error(
                "  ❌ CRITICAL: Fit quality is unacceptable - consider parameter adjustment or data quality check"
            )
        elif combined_quality == "poor":
            logger.warning(
                "  ⚠ POOR: Fit quality is poor - optimization may need improvement"
            )
        elif combined_quality == "warning":
            logger.warning(
                "  ⚠ WARNING: Some angles show poor fit - consider investigation"
            )
        elif combined_quality == "acceptable":
            logger.info(
                "  ✓ ACCEPTABLE: Fit quality is acceptable with some limitations"
            )
        else:
            logger.info("  ✅ EXCELLENT: Fit quality is excellent across all angles")

        return per_angle_results

    def save_results_with_config(
        self, results: dict[str, Any], output_dir: str | None = None
    ) -> None:
        """
        Save optimization results along with configuration to JSON file.

        This method ensures all results including uncertainty fields are properly
        saved with the configuration for reproducibility.

        Parameters
        ----------
        results : Dict[str, Any]
            Results dictionary from optimization methods
        output_dir : str, optional
            Output directory for saving results file (default: current directory)
        """
        # Create comprehensive results with configuration

        timestamp = datetime.now(UTC).isoformat()

        output_data = {
            "timestamp": timestamp,
            "config": self.config,
            "results": results,
        }

        # Add execution metadata
        if "execution_metadata" not in output_data:
            output_data["execution_metadata"] = {
                "analysis_success": True,
                "timestamp": timestamp,
            }

        # Determine output file name
        if self.config is not None:
            output_settings = self.config.get("output_settings", {})
            file_formats = output_settings.get("file_formats", {})
            results_format = file_formats.get("results_format", "json")
        else:
            results_format = "json"

        # Determine output file path
        if output_dir:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            if results_format == "json":
                output_file = output_dir_path / "homodyne_analysis_results.json"
            else:
                output_file = (
                    output_dir_path / f"homodyne_analysis_results.{results_format}"
                )
        else:
            if results_format == "json":
                output_file = "homodyne_analysis_results.json"
            else:
                output_file = f"homodyne_analysis_results.{results_format}"

        try:
            # Save to JSON format regardless of specified format for
            # compatibility
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2, default=str)

            logger.info(f"Results and configuration saved to {output_file}")

            # Also save a copy to results directory if it exists
            results_dir = "homodyne_analysis_results"
            if os.path.exists(results_dir):
                results_file_path = os.path.join(results_dir, "run_configuration.json")
                with open(results_file_path, "w") as f:
                    json.dump(output_data, f, indent=2, default=str)
                logger.info(f"Results also saved to {results_file_path}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise

        # NEW: Call method-specific saving logic for enhanced results organization
        # This runs after the main save to avoid interfering with tests
        # Skip enhanced saving during tests to avoid mocking conflicts
        # Note: is_testing variable removed as it was unused
        # Note: File saving handled by run_homodyne.py with proper directory structure
        # handles all file outputs with proper directory structure

    def _plot_experimental_data_validation(
        self, c2_experimental: np.ndarray, phi_angles: np.ndarray
    ) -> None:
        """
        Plot experimental C2 data immediately after loading for validation.

        This method creates a comprehensive validation plot of the loaded experimental
        data to verify data integrity and structure before analysis.

        Parameters
        ----------
        c2_experimental : np.ndarray
            Experimental correlation data with shape (n_angles, n_t2, n_t1)
        phi_angles : np.ndarray
            Array of scattering angles in degrees
        """
        try:
            # Import plotting dependencies
            import matplotlib.gridspec as gridspec
            import matplotlib.pyplot as plt

            logger.debug("Creating experimental data validation plot")

            # Set up plotting style
            plt.style.use("default")
            plt.rcParams.update(
                {
                    "font.size": 11,
                    "axes.labelsize": 12,
                    "axes.titlesize": 14,
                    "figure.dpi": 150,
                }
            )

            # Get temporal parameters
            dt = self.dt
            n_angles, n_t2, n_t1 = c2_experimental.shape
            time_t2 = np.arange(n_t2) * dt
            time_t1 = np.arange(n_t1) * dt

            logger.debug(f"Data shape for validation plot: {c2_experimental.shape}")
            logger.debug(
                f"Time parameters: dt={dt}, t2_max={time_t2[-1]:.1f}s, t1_max={time_t1[-1]:.1f}s"
            )

            # Create the validation plot - simplified to heatmap + statistics
            # only
            n_plot_angles = min(3, n_angles)  # Show up to 3 angles
            fig = plt.figure(figsize=(10, 4 * n_plot_angles))
            gs = gridspec.GridSpec(n_plot_angles, 2, hspace=0.3, wspace=0.3)

            for i in range(n_plot_angles):
                angle_idx = i * (n_angles // n_plot_angles) if n_angles > 1 else 0
                if angle_idx >= n_angles:
                    angle_idx = n_angles - 1

                angle_data = c2_experimental[angle_idx, :, :]
                phi_deg = phi_angles[angle_idx] if len(phi_angles) > angle_idx else 0.0

                # 1. C2 heatmap (left panel)
                ax1 = fig.add_subplot(gs[i, 0])
                im1 = ax1.imshow(
                    angle_data,
                    aspect="equal",
                    origin="lower",
                    extent=[
                        time_t1[0],
                        time_t1[-1],
                        time_t2[0],
                        time_t2[-1],
                    ],  # type: ignore
                    cmap="viridis",
                )
                ax1.set_xlabel(r"Time $t_1$ (s)")
                ax1.set_ylabel(r"Time $t_2$ (s)")
                ax1.set_title(f"$g_2(t_1,t_2)$ at φ={phi_deg:.1f}°")
                plt.colorbar(im1, ax=ax1, shrink=0.8)

                # 2. Statistics (right panel)
                ax2 = fig.add_subplot(gs[i, 1])
                ax2.axis("off")

                # Calculate statistics
                mean_val = np.mean(angle_data)
                std_val = np.std(angle_data)
                min_val = np.min(angle_data)
                max_val = np.max(angle_data)
                diagonal = np.diag(angle_data)
                diag_mean = np.mean(diagonal)
                contrast = (max_val - min_val) / min_val

                stats_text = f"""Data Statistics (φ={phi_deg:.1f}°):

Shape: {angle_data.shape[0]} × {angle_data.shape[1]}

g₂ Values:
Mean: {mean_val:.4f}
Std:  {std_val:.4f}
Min:  {min_val:.4f}
Max:  {max_val:.4f}

Diagonal mean: {diag_mean:.4f}
Contrast: {contrast:.3f}

Validation:
{"✓" if 0.9 < mean_val < 1.2 else "✗"} Mean around 1.0
{"✓" if diag_mean > mean_val else "✗"} Diagonal enhanced
{"✓" if contrast > 0.001 else "✗"} Sufficient contrast"""

                ax2.text(
                    0.05,
                    0.95,
                    stats_text,
                    transform=ax2.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    fontfamily="monospace",
                    bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.7},
                )

            # Overall title
            sample_desc = (
                self.config.get("metadata", {}).get(
                    "sample_description", "Unknown Sample"
                )
                if self.config
                else "Unknown Sample"
            )
            plt.suptitle(
                f"Experimental Data Validation: {sample_desc}",
                fontsize=16,
                fontweight="bold",
            )

            # Save the validation plot
            plots_base_dir = (
                self.config.get("output_settings", {})
                .get("plotting", {})
                .get("output", {})
                .get("base_directory", "./plots")
                if self.config
                else "./plots"
            )
            plots_dir = Path(plots_base_dir)
            plots_dir.mkdir(parents=True, exist_ok=True)

            output_file = plots_dir / "experimental_data_validation.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            logger.info(f"Experimental data validation plot saved to: {output_file}")

            # Optionally show the plot
            show_plots = (
                self.config.get("output_settings", {})
                .get("plotting", {})
                .get("general", {})
                if self.config
                else {}.get("show_plots", False)
            )  # type: ignore
            if show_plots:
                plt.show()
            else:
                plt.close(fig)

        except Exception as e:
            logger.error(f"Failed to create experimental data validation plot: {e}")
            import traceback

            logger.debug(traceback.format_exc())

    def _generate_analysis_plots(
        self,
        results: dict[str, Any],
        output_data: dict[str, Any],
        skip_generic_plots: bool = False,
    ) -> None:
        """
        Generate analysis plots including C2 heatmaps with experimental vs theoretical comparison.

        Parameters
        ----------
        results : Dict[str, Any]
            Results dictionary from optimization methods
        output_data : Dict[str, Any]
            Complete output data including configuration
        """
        logger = logging.getLogger(__name__)

        # Skip generic plots if requested (for method-specific plotting)
        if skip_generic_plots:
            logger.info(
                "Generic plots skipped - using method-specific plotting instead"
            )
            return

        # Check if plotting is enabled in configuration
        config = output_data.get("config") or {}
        output_settings = config.get("output_settings", {})
        reporting = output_settings.get("reporting", {})

        if not reporting.get("generate_plots", True):
            logger.info("Plotting disabled in configuration - skipping plot generation")
            return

        logger.info("Generating analysis plots...")

        try:
            # Import plotting module
            from homodyne.plotting import (
                plot_c2_heatmaps,
                plot_diagnostic_summary,
                plot_mcmc_convergence_diagnostics,
                plot_mcmc_corner,
                plot_mcmc_trace,
            )

            # Extract output directory from output_data if available
            output_dir = output_data.get("output_dir")

            # Determine output directory - use output_data, config, or default
            if output_dir is not None:
                results_dir = Path(output_dir)
            elif (
                config
                and "output_settings" in config
                and "results_directory" in config["output_settings"]
            ):
                results_dir = Path(config["output_settings"]["results_directory"])
            else:
                results_dir = Path("homodyne_results")

            plots_dir = results_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Prepare data for plotting
            plot_data = self._prepare_plot_data(results, config)

            if plot_data is None:
                logger.warning(
                    "Insufficient data for plotting - skipping plot generation"
                )
                return

            # Generate C2 heatmaps if experimental and theoretical data are
            # available
            if all(
                key in plot_data
                for key in [
                    "experimental_data",
                    "theoretical_data",
                    "phi_angles",
                ]
            ):
                logger.info("Generating C2 correlation heatmaps...")
                try:
                    success = plot_c2_heatmaps(
                        plot_data["experimental_data"],
                        plot_data["theoretical_data"],
                        plot_data["phi_angles"],
                        plots_dir,
                        config,
                        t2=plot_data.get("t2"),
                        t1=plot_data.get("t1"),
                    )
                    if success:
                        logger.info("✓ C2 heatmaps generated successfully")
                    else:
                        logger.warning("⚠ Some C2 heatmaps failed to generate")
                except Exception as e:
                    logger.error(f"Failed to generate C2 heatmaps: {e}")

            # Parameter evolution plot - DISABLED (was non-functional)
            # This plot has been removed due to persistent issues

            # Generate MCMC plots if trace data is available
            if "mcmc_trace" in plot_data:
                logger.info("Generating MCMC plots...")

                # MCMC corner plot
                try:
                    success = plot_mcmc_corner(
                        plot_data["mcmc_trace"],
                        plots_dir,
                        config,
                        param_names=plot_data.get("parameter_names"),
                        param_units=plot_data.get("parameter_units"),
                    )
                    if success:
                        logger.info("✓ MCMC corner plot generated successfully")
                    else:
                        logger.warning("⚠ MCMC corner plot failed to generate")
                except Exception as e:
                    logger.error(f"Failed to generate MCMC corner plot: {e}")

                # MCMC trace plots
                try:
                    success = plot_mcmc_trace(
                        plot_data["mcmc_trace"],
                        plots_dir,
                        config,
                        param_names=plot_data.get("parameter_names"),
                        param_units=plot_data.get("parameter_units"),
                    )
                    if success:
                        logger.info("✓ MCMC trace plots generated successfully")
                    else:
                        logger.warning("⚠ MCMC trace plots failed to generate")
                except Exception as e:
                    logger.error(f"Failed to generate MCMC trace plots: {e}")

                # MCMC convergence diagnostics
                if "mcmc_diagnostics" in plot_data:
                    try:
                        success = plot_mcmc_convergence_diagnostics(
                            plot_data["mcmc_trace"],
                            plot_data["mcmc_diagnostics"],
                            plots_dir,
                            config,
                            param_names=plot_data.get("parameter_names"),
                        )
                        if success:
                            logger.info(
                                "✓ MCMC convergence diagnostics generated successfully"
                            )
                        else:
                            logger.warning(
                                "⚠ MCMC convergence diagnostics failed to generate"
                            )
                    except Exception as e:
                        logger.error(
                            f"Failed to generate MCMC convergence diagnostics: {e}"
                        )
            else:
                logger.debug("MCMC trace data not available - skipping MCMC plots")

            # Generate diagnostic summary plot only for --method all (multiple
            # methods)
            methods_used = results.get("methods_used", [])
            if len(methods_used) > 1:
                logger.info("Generating diagnostic summary plot...")
                try:
                    success = plot_diagnostic_summary(plot_data, plots_dir, config)
                    if success:
                        logger.info("✓ Diagnostic summary plot generated successfully")
                    else:
                        logger.warning("⚠ Diagnostic summary plot failed to generate")
                except Exception as e:
                    logger.error(f"Failed to generate diagnostic summary plot: {e}")
            else:
                logger.info(
                    "Skipping diagnostic summary plot - only generated for --method all (multiple methods)"
                )

            logger.info(f"Plots saved to: {plots_dir}")

        except ImportError as e:
            logger.warning(f"Plotting module not available: {e}")
            logger.info("Install matplotlib for plotting: pip install matplotlib")
        except Exception as e:
            logger.error(f"Unexpected error during plot generation: {e}")

    def _prepare_plot_data(
        self, results: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Prepare data for plotting from analysis results.

        Parameters
        ----------
        results : Dict[str, Any]
            Results dictionary from optimization methods
        config : Dict[str, Any]
            Configuration dictionary

        Returns
        -------
        Optional[Dict[str, Any]]
            Plot data dictionary or None if insufficient data
        """
        logger = logging.getLogger(__name__)

        try:
            plot_data = {}

            # Find the best method results
            best_method = None
            best_chi2 = float("inf")

            # Check different method results
            for method_key in [
                "classical_optimization",
                "robust_optimization",
                "mcmc_optimization",
            ]:
                if method_key in results:
                    method_results = results[method_key]
                    chi2 = method_results.get("chi_squared")
                    if chi2 is not None and chi2 < best_chi2:
                        best_chi2 = chi2
                        best_method = method_key

            if best_method is None:
                logger.warning("No valid optimization results found for plotting")
                return None

            # Extract best parameters
            best_params_list = results[best_method].get("parameters")
            if best_params_list is not None:
                # Convert parameter list to dictionary
                param_names = config.get("initial_parameters", {}).get(
                    "parameter_names", []
                )
                if len(param_names) == len(best_params_list):
                    plot_data["best_parameters"] = dict(
                        zip(param_names, best_params_list, strict=False)
                    )
                else:
                    # Use generic names if parameter names don't match
                    plot_data["best_parameters"] = {
                        f"param_{i}": val for i, val in enumerate(best_params_list)
                    }

            # Extract parameter bounds
            parameter_space = config.get("parameter_space", {})
            if "bounds" in parameter_space:
                plot_data["parameter_bounds"] = parameter_space["bounds"]

            # Extract initial parameters
            initial_params = config.get("initial_parameters", {}).get("values")
            if initial_params is not None:
                param_names = config.get("initial_parameters", {}).get(
                    "parameter_names", []
                )
                if len(param_names) == len(initial_params):
                    plot_data["initial_parameters"] = dict(
                        zip(param_names, initial_params, strict=False)
                    )

            # Try to reconstruct experimental and theoretical data for plotting
            if hasattr(self, "_last_experimental_data") and hasattr(
                self, "_last_phi_angles"
            ):
                plot_data["experimental_data"] = self._last_experimental_data
                plot_data["phi_angles"] = self._last_phi_angles

                # Generate theoretical data using best parameters
                if best_params_list is not None and self._last_phi_angles is not None:
                    try:
                        theoretical_data = self._generate_theoretical_data(
                            best_params_list, self._last_phi_angles
                        )
                        plot_data["theoretical_data"] = theoretical_data
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate theoretical data for plotting: {e}"
                        )

            # Add time axes if available
            temporal = config.get("analyzer_parameters", {}).get("temporal", {})
            dt = temporal.get("dt", 0.1)
            start_frame = temporal.get("start_frame", 1)
            end_frame = temporal.get("end_frame", 1000)

            # Generate time arrays (these are approximate)
            n_frames = end_frame - start_frame + 1
            t_array = np.arange(n_frames) * dt
            plot_data["t1"] = t_array
            plot_data["t2"] = t_array

            # Add parameter names and units for MCMC plotting
            param_names = config.get("initial_parameters", {}).get(
                "parameter_names", []
            )
            if param_names:
                plot_data["parameter_names"] = param_names

            # Extract parameter units from bounds configuration
            parameter_space = config.get("parameter_space", {})
            bounds = parameter_space.get("bounds", [])
            if bounds:
                param_units = [bound.get("unit", "") for bound in bounds]
                plot_data["parameter_units"] = param_units

            # Add MCMC-specific data if available
            if "mcmc_optimization" in results:
                mcmc_results = results["mcmc_optimization"]

                # Add convergence diagnostics
                if "convergence_diagnostics" in mcmc_results:
                    plot_data["mcmc_diagnostics"] = mcmc_results[
                        "convergence_diagnostics"
                    ]

                # Add posterior means
                if "posterior_means" in mcmc_results:
                    plot_data["posterior_means"] = mcmc_results["posterior_means"]

                # Try to get MCMC trace data from live results first
                trace_data = None
                if "trace" in mcmc_results and mcmc_results["trace"] is not None:
                    trace_data = mcmc_results["trace"]
                    logger.debug("Using live MCMC trace data for plotting")
                elif "trace" in results and results["trace"] is not None:
                    # Check top-level trace data as fallback
                    trace_data = results["trace"]
                    logger.debug("Using top-level MCMC trace data for plotting")
                else:
                    # Final fallback: try to load from NetCDF file
                    try:
                        mcmc_results_dir = Path("homodyne_results") / "mcmc_results"
                        trace_file = mcmc_results_dir / "mcmc_trace.nc"

                        if trace_file.exists():
                            try:
                                import arviz as az

                                trace_data = az.from_netcdf(str(trace_file))
                                logger.debug(
                                    f"Loaded MCMC trace data from {trace_file}"
                                )
                            except ImportError:
                                logger.warning(
                                    "ArviZ not available - cannot load MCMC trace for plotting"
                                )
                            except Exception as e:
                                logger.warning(f"Failed to load MCMC trace data: {e}")
                        else:
                            logger.debug("MCMC trace file not found")

                    except Exception as e:
                        logger.warning(f"Error checking for MCMC trace file: {e}")

                # Add trace data to plot_data if found
                if trace_data is not None:
                    plot_data["mcmc_trace"] = trace_data
                    logger.info("✓ MCMC trace data available for plotting")
                else:
                    logger.debug(
                        "MCMC trace data not available - trace plots will be skipped"
                    )

            # Add overall plot data
            plot_data["chi_squared"] = best_chi2
            plot_data["method"] = best_method.replace("_optimization", "").title()

            # Add individual method chi-squared values for diagnostic plotting
            if (
                "classical_optimization" in results
                and "chi_squared" in results["classical_optimization"]
            ):
                plot_data["classical_chi_squared"] = results["classical_optimization"][
                    "chi_squared"
                ]

            if (
                "robust_optimization" in results
                and "chi_squared" in results["robust_optimization"]
            ):
                plot_data["robust_chi_squared"] = results["robust_optimization"][
                    "chi_squared"
                ]

            if (
                "mcmc_optimization" in results
                and "chi_squared" in results["mcmc_optimization"]
            ):
                plot_data["mcmc_chi_squared"] = results["mcmc_optimization"][
                    "chi_squared"
                ]

            return plot_data

        except Exception as e:
            logger.error(f"Error preparing plot data: {e}")
            return None

    def _generate_theoretical_data(
        self, parameters: list, phi_angles: np.ndarray
    ) -> np.ndarray:
        """
        Generate theoretical correlation data for plotting.

        Parameters
        ----------
        parameters : list
            Optimized parameters
        phi_angles : np.ndarray
            Array of phi angles

        Returns
        -------
        np.ndarray
            Theoretical correlation data
        """
        logger = logging.getLogger(__name__)

        try:
            # Use the existing physics model to generate theoretical data
            logger.debug(f"Generating theoretical data for {len(phi_angles)} angles")

            # Call the main correlation calculation method
            theoretical_data = self.calculate_c2_nonequilibrium_laminar_parallel(
                np.array(parameters),
                phi_angles,  # type: ignore
            )

            logger.debug(
                f"Successfully generated theoretical data with shape: {
                    theoretical_data.shape
                }"
            )
            return theoretical_data

        except Exception as e:
            logger.error(f"Error generating theoretical data: {e}")
            # Fallback: return experimental data shape filled with ones if
            # available
            if (
                hasattr(self, "_last_experimental_data")
                and self._last_experimental_data is not None
            ):
                shape = self._last_experimental_data.shape
                logger.warning(f"Using fallback data with shape {shape}")
                return np.ones(shape)
            else:
                logger.warning("No fallback data available")
                return np.array([])


def _get_chi2_interpretation(chi2_value: float) -> str:
    """Provide interpretation of reduced chi-squared value with uncertainty context.

    The reduced chi-squared uncertainty quantifies the reliability of the average:
    - Small uncertainty (< 0.1 * χ²_red): Consistent fit quality across angles
    - Moderate uncertainty (0.1-0.5 * χ²_red): Some angle variation, generally acceptable
    - Large uncertainty (> 0.5 * χ²_red): High variability between angles, potential systematic issues

    Parameters
    ----------
    chi2_value : float
        Reduced chi-squared value

    Returns
    -------
    str
        Interpretation string with quality assessment and statistical meaning
    """
    if chi2_value <= 1.0:
        return f"Excellent fit (χ²_red = {
            chi2_value:.2f} ≤ 1.0): Model matches data within expected noise"
    elif chi2_value <= 2.0:
        return f"Very good fit (χ²_red = {
            chi2_value:.2f}): Model captures main features with minor deviations"
    elif chi2_value <= 5.0:
        return f"Acceptable fit (χ²_red = {
            chi2_value:.2f}): Model reasonable but some systematic deviations present"
    elif chi2_value <= 10.0:
        return f"Poor fit (χ²_red = {
            chi2_value:.2f}): Significant deviations suggest model inadequacy or underestimated uncertainties"
    else:
        return f"Very poor fit (χ²_red = {
            chi2_value:.2f}): Major systematic deviations, model likely inappropriate"


def _get_quality_explanation(quality: str) -> str:
    """Provide explanation of quality assessment."""
    explanations = {
        "excellent": "Model provides exceptional agreement with experimental data across all angles",
        "acceptable": "Model provides reasonable agreement with experimental data for most angles",
        "warning": "Model shows concerning deviations that may indicate systematic issues",
        "poor": "Model shows significant inadequacies in describing the experimental data",
        "critical": "Model is fundamentally inappropriate for this dataset",
    }
    return explanations.get(quality, "Unknown quality level")


def _get_quality_recommendations(quality: str, issues: list) -> list:
    """Provide actionable recommendations based on quality assessment."""
    recommendations = []

    if quality == "excellent":
        recommendations.append("Results are reliable for publication")
        recommendations.append("Consider this model for further analysis")
    elif quality == "acceptable":
        recommendations.append("Results may be suitable with appropriate caveats")
        recommendations.append(
            "Consider checking specific angles with higher chi-squared"
        )
    elif quality == "warning":
        recommendations.append("Investigate systematic deviations before publication")
        recommendations.append("Consider alternative models or parameter ranges")
        recommendations.append("Check experimental uncertainties and data quality")
    elif quality in ["poor", "critical"]:
        recommendations.append("Do not use results for quantitative conclusions")
        recommendations.append("Consider fundamental model revision")
        recommendations.append("Check experimental setup and data processing")
        recommendations.append("Investigate alternative theoretical approaches")

    # Add issue-specific recommendations
    for issue in issues:
        if "outliers" in issue.lower():
            recommendations.append(
                "Investigate outlier angles for experimental artifacts"
            )
        if "good angles" in issue.lower():
            recommendations.append(
                "Consider focusing analysis on subset of reliable angles"
            )

    return recommendations


# ============================================================================
# PARAMETER MANAGEMENT AND RESULTS SAVING
# ============================================================================

# Note: Additional methods would be defined here if needed
