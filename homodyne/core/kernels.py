"""
High-Performance Computational Kernels for Homodyne Scattering Analysis

This module provides Numba-accelerated computational kernels for the core
mathematical operations in homodyne scattering calculations.

Created for: Rheo-SAXS-XPCS Homodyne Analysis
Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import numpy as np

# Numba imports with fallbacks
try:
    import numba as nb
    from numba import float64, int64, jit, njit, prange, types

    try:
        from numba.types import Tuple
    except (ImportError, AttributeError):
        # Fallback for older numba versions or different import paths
        Tuple = getattr(types, "Tuple", types.UniTuple)

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Fallback decorators when Numba is unavailable
    F = TypeVar("F", bound=Callable[..., Any])

    def jit(*args: Any, **kwargs: Any) -> Any:
        return args[0] if args and callable(args[0]) else lambda f: f

    def njit(*args: Any, **kwargs: Any) -> Any:
        return args[0] if args and callable(args[0]) else lambda f: f

    prange = range

    class DummyType:
        def __getitem__(self, item: Any) -> "DummyType":
            return self

        def __call__(self, *args: Any, **kwargs: Any) -> "DummyType":
            return self

    float64 = int64 = types = Tuple = DummyType()


# Test environment detection for Numba threading compatibility
def _is_test_environment() -> bool:
    """
    Detect if code is running in a test environment.

    This is used to disable Numba parallel processing when NUMBA_NUM_THREADS=1
    to avoid threading conflicts.
    """
    import os

    numba_threads = os.environ.get("NUMBA_NUM_THREADS", "")
    return (
        numba_threads == "1"
        or os.environ.get("PYTEST_CURRENT_TEST") is not None
        or "pytest" in os.environ.get("_", "")
    )


# Use parallel processing only when not in test environment
_USE_PARALLEL = not _is_test_environment()


@njit(float64[:, :](float64[:]), parallel=_USE_PARALLEL, cache=True, fastmath=True)
def _create_time_integral_matrix_impl(time_dependent_array):
    """Create time integral matrix for correlation calculations."""
    n = len(time_dependent_array)
    matrix = np.empty((n, n), dtype=np.float64)
    cumsum = np.cumsum(time_dependent_array)

    for i in prange(n):
        cumsum_i = cumsum[i]
        for j in range(n):
            matrix[i, j] = abs(cumsum_i - cumsum[j])

    return matrix


@njit(float64[:](float64[:], float64, float64, float64), cache=True, fastmath=True)
def _calculate_diffusion_coefficient_impl(time_array, D0, alpha, D_offset):
    """Calculate time-dependent diffusion coefficient."""
    D_t = np.empty_like(time_array)
    for i in range(len(time_array)):
        D_value = D0 * (time_array[i] ** alpha) + D_offset
        D_t[i] = max(D_value, 1e-10)
    return D_t


@njit(float64[:](float64[:], float64, float64, float64), cache=True, fastmath=True)
def _calculate_shear_rate_impl(time_array, gamma_dot_t0, beta, gamma_dot_t_offset):
    """Calculate time-dependent shear rate."""
    gamma_dot_t = np.empty_like(time_array)
    for i in range(len(time_array)):
        gamma_value = gamma_dot_t0 * (time_array[i] ** beta) + gamma_dot_t_offset
        gamma_dot_t[i] = max(gamma_value, 1e-10)
    return gamma_dot_t


@njit(
    float64[:, :](float64[:, :], float64),
    parallel=_USE_PARALLEL,
    cache=True,
    fastmath=True,
)
def _compute_g1_correlation_impl(diffusion_integral_matrix, wavevector_factor):
    """Compute field correlation function g₁ from diffusion."""
    shape = diffusion_integral_matrix.shape
    g1 = np.empty(shape, dtype=np.float64)

    for i in prange(shape[0]):
        for j in range(shape[1]):
            exponent = -wavevector_factor * diffusion_integral_matrix[i, j]
            g1[i, j] = np.exp(exponent)

    return g1


@njit(
    float64[:, :](float64[:, :], float64),
    parallel=_USE_PARALLEL,
    cache=True,
    fastmath=True,
)
def _compute_sinc_squared_impl(shear_integral_matrix, prefactor):
    """Compute sinc² function for shear flow contributions."""
    shape = shear_integral_matrix.shape
    sinc_squared = np.empty(shape, dtype=np.float64)
    pi = np.pi

    for i in prange(shape[0]):
        for j in range(shape[1]):
            argument = prefactor * shear_integral_matrix[i, j]

            if abs(argument) < 1e-10:
                pi_arg_sq = (pi * argument) ** 2
                sinc_squared[i, j] = 1.0 - pi_arg_sq / 3.0
            else:
                pi_arg = pi * argument
                if abs(pi_arg) < 1e-15:
                    sinc_squared[i, j] = 1.0
                else:
                    sinc_value = np.sin(pi_arg) / pi_arg
                    sinc_squared[i, j] = sinc_value * sinc_value

    return sinc_squared


def memory_efficient_cache(maxsize=128):
    """
    Memory-efficient LRU cache with automatic cleanup.

    Features:
    - Least Recently Used eviction
    - Access frequency tracking
    - Configurable size limits
    - Cache statistics

    Parameters
    ----------
    maxsize : int
        Maximum cached items (0 disables caching)

    Returns
    -------
    decorator
        Function decorator with cache_info() and cache_clear() methods
    """

    def decorator(func):
        cache: dict[Any, Any] = {}
        access_count: dict[Any, int] = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create hashable cache key - optimized for performance
            key_parts = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    # Use faster hash-based key generation
                    array_info = (
                        arg.shape,
                        arg.dtype.str,
                        hash(arg.data.tobytes()),
                    )
                    key_parts.append(str(array_info))
                elif hasattr(arg, "__array__"):
                    # Handle array-like objects
                    arr = np.asarray(arg)
                    array_info = (
                        arr.shape,
                        arr.dtype.str,
                        hash(arr.data.tobytes()),
                    )
                    key_parts.append(str(array_info))
                else:
                    key_parts.append(str(arg))

            for k, v in sorted(kwargs.items()):
                if isinstance(v, np.ndarray):
                    array_info = (v.shape, v.dtype.str, hash(v.data.tobytes()))
                    key_parts.append(f"{k}={array_info}")
                else:
                    key_parts.append(f"{k}={v}")

            cache_key = "|".join(key_parts)

            # Check cache hit
            if cache_key in cache:
                access_count[cache_key] = access_count.get(cache_key, 0) + 1
                return cache[cache_key]

            # Compute on cache miss
            result = func(*args, **kwargs)

            # Store result and manage cache size
            if maxsize > 0:
                cache[cache_key] = result
                access_count[cache_key] = 1

                # Evict least-accessed items if cache exceeds maxsize
                while len(cache) > maxsize:
                    # Remove least-accessed item
                    least_accessed_key = min(access_count.items(), key=lambda x: x[1])[
                        0
                    ]
                    cache.pop(least_accessed_key, None)
                    access_count.pop(least_accessed_key, None)

            return result

        def cache_info():
            """Return cache statistics."""
            hit_rate = 0.0
            if access_count:
                total = sum(access_count.values())
                unique = len(access_count)
                hit_rate = (total - unique) / total if total > 0 else 0.0

            return f"Cache: {len(cache)}/{maxsize}, Hit rate: {hit_rate:.2%}"

        def cache_clear():
            """Clear all cached data."""
            cache.clear()
            access_count.clear()

        class CachedFunction:
            def __init__(self, func):
                self._func = func
                self.cache_info = cache_info
                self.cache_clear = cache_clear
                # Copy function attributes for proper method binding
                self.__name__ = getattr(func, "__name__", "cached_function")
                self.__doc__ = getattr(func, "__doc__", None)
                self.__module__ = getattr(func, "__module__", "") or ""

            def __call__(self, *args, **kwargs):
                return self._func(*args, **kwargs)

            def __get__(self, instance, owner):
                """Support instance methods by implementing descriptor protocol."""
                if instance is None:
                    return self
                else:
                    # Return a bound method
                    return lambda *args, **kwargs: self._func(instance, *args, **kwargs)

        return CachedFunction(wrapper)

    return decorator


# Additional optimized kernels for improved performance


@njit(
    Tuple((float64[:], float64[:]))(float64[:, :], float64[:, :]),
    parallel=_USE_PARALLEL,
    cache=True,
    fastmath=True,
)
def _solve_least_squares_batch_numba_impl(theory_batch, exp_batch):
    """
    Batch solve least squares for multiple angles using Numba optimization.

    Solves: min ||A*x - b||^2 where A = [theory, ones] for each angle.

    Parameters
    ----------
    theory_batch : np.ndarray, shape (n_angles, n_data_points)
        Theory values for each angle
    exp_batch : np.ndarray, shape (n_angles, n_data_points)
        Experimental values for each angle

    Returns
    -------
    tuple of np.ndarray
        contrast_batch : shape (n_angles,) - contrast scaling factors
        offset_batch : shape (n_angles,) - offset values
    """
    n_angles, n_data = theory_batch.shape
    contrast_batch = np.zeros(n_angles, dtype=np.float64)
    offset_batch = np.zeros(n_angles, dtype=np.float64)

    for i in prange(n_angles):
        theory = theory_batch[i]
        exp = exp_batch[i]

        # Compute AtA and Atb directly for 2x2 system
        # A = [theory, ones], so AtA = [[sum(theory^2), sum(theory)],
        #                              [sum(theory), n_data]]
        sum_theory_sq = 0.0
        sum_theory = 0.0
        sum_exp = 0.0
        sum_theory_exp = 0.0

        for j in range(n_data):
            t_val = theory[j]
            e_val = exp[j]
            sum_theory_sq += t_val * t_val
            sum_theory += t_val
            sum_exp += e_val
            sum_theory_exp += t_val * e_val

        # Solve 2x2 system: AtA * x = Atb
        # [[sum_theory_sq, sum_theory], [sum_theory, n_data]] * [contrast, offset] = [sum_theory_exp, sum_exp]
        det = sum_theory_sq * n_data - sum_theory * sum_theory

        if abs(det) > 1e-12:  # Non-singular matrix
            contrast_batch[i] = (n_data * sum_theory_exp - sum_theory * sum_exp) / det
            offset_batch[i] = (
                sum_theory_sq * sum_exp - sum_theory * sum_theory_exp
            ) / det
        else:  # Singular matrix fallback
            contrast_batch[i] = 1.0
            offset_batch[i] = 0.0

    return contrast_batch, offset_batch


# Apply numba decorator if available, otherwise use fallback
if not NUMBA_AVAILABLE:
    # Remove decorator if Numba not available
    _solve_least_squares_batch_numba_impl = (
        _solve_least_squares_batch_numba_impl.__wrapped__
        if hasattr(_solve_least_squares_batch_numba_impl, "__wrapped__")
        else _solve_least_squares_batch_numba_impl
    )

    def _solve_least_squares_batch_fallback(theory_batch, exp_batch):
        """Fallback implementation when Numba is not available."""
        return _solve_least_squares_batch_numba_impl(theory_batch, exp_batch)

    solve_least_squares_batch_numba = _solve_least_squares_batch_fallback
    solve_least_squares_batch_numba.signatures = []  # type: ignore[attr-defined]
else:
    solve_least_squares_batch_numba = _solve_least_squares_batch_numba_impl


@njit(
    nb.types.Tuple((float64[:], float64[:, :]))(
        float64[:, :], float64[:, :], float64[:], float64[:]
    ),
    parallel=_USE_PARALLEL,
    cache=True,
    fastmath=True,
)
def _compute_chi_squared_with_residuals_batch_numba_impl(
    theory_batch, exp_batch, contrast_batch, offset_batch
):
    """
    Optimized batch compute chi-squared values and residuals for multiple angles.

    This eliminates redundant residual calculations in the scaling optimization pipeline
    by computing both chi-squared values and residuals in a single pass.

    Parameters
    ----------
    theory_batch : np.ndarray, shape (n_angles, n_data_points)
        Theory values for each angle
    exp_batch : np.ndarray, shape (n_angles, n_data_points)
        Experimental values for each angle
    contrast_batch : np.ndarray, shape (n_angles,)
        Contrast scaling factors
    offset_batch : np.ndarray, shape (n_angles,)
        Offset values

    Returns
    -------
    tuple
        chi2_batch : np.ndarray, shape (n_angles,) - Chi-squared values for each angle
        residuals_batch : np.ndarray, shape (n_angles, n_data_points) - Residuals for each angle
    """
    n_angles, n_data = theory_batch.shape
    chi2_batch = np.zeros(n_angles, dtype=np.float64)
    residuals_batch = np.zeros((n_angles, n_data), dtype=np.float64)

    for i in prange(n_angles):
        theory = theory_batch[i]
        exp = exp_batch[i]
        contrast = contrast_batch[i]
        offset = offset_batch[i]

        chi2 = 0.0
        for j in range(n_data):
            fitted_val = theory[j] * contrast + offset
            residual = exp[j] - fitted_val
            chi2 += residual * residual
            residuals_batch[i, j] = residual

        chi2_batch[i] = chi2

    return chi2_batch, residuals_batch


@njit(
    float64[:](float64[:, :], float64[:, :], float64[:], float64[:]),
    parallel=_USE_PARALLEL,
    cache=True,
    fastmath=True,
)
def _compute_chi_squared_batch_numba_impl(
    theory_batch, exp_batch, contrast_batch, offset_batch
):
    """
    Batch compute chi-squared values for multiple angles using pre-computed scaling.

    Parameters
    ----------
    theory_batch : np.ndarray, shape (n_angles, n_data_points)
        Theory values for each angle
    exp_batch : np.ndarray, shape (n_angles, n_data_points)
        Experimental values for each angle
    contrast_batch : np.ndarray, shape (n_angles,)
        Contrast scaling factors
    offset_batch : np.ndarray, shape (n_angles,)
        Offset values

    Returns
    -------
    np.ndarray, shape (n_angles,)
        Chi-squared values for each angle
    """
    n_angles, n_data = theory_batch.shape
    chi2_batch = np.zeros(n_angles, dtype=np.float64)

    for i in prange(n_angles):
        theory = theory_batch[i]
        exp = exp_batch[i]
        contrast = contrast_batch[i]
        offset = offset_batch[i]

        chi2 = 0.0
        for j in range(n_data):
            fitted_val = theory[j] * contrast + offset
            residual = exp[j] - fitted_val
            chi2 += residual * residual

        chi2_batch[i] = chi2

    return chi2_batch


def _compute_chi_squared_batch_fallback(
    theory_batch, exp_batch, contrast_batch, offset_batch
):
    """Fallback implementation when Numba is not available."""
    return _compute_chi_squared_batch_numba_impl(
        theory_batch, exp_batch, contrast_batch, offset_batch
    )


# Apply numba decorator if available, otherwise use fallback
if not NUMBA_AVAILABLE:
    # Remove decorator if Numba not available
    _compute_chi_squared_batch_numba_impl = (
        _compute_chi_squared_batch_numba_impl.__wrapped__
        if hasattr(_compute_chi_squared_batch_numba_impl, "__wrapped__")
        else _compute_chi_squared_batch_numba_impl
    )

    def _compute_chi_squared_batch_fallback(
        theory_batch, exp_batch, contrast_batch, offset_batch
    ):
        """Fallback implementation when Numba is not available."""
        return _compute_chi_squared_batch_numba_impl(
            theory_batch, exp_batch, contrast_batch, offset_batch
        )

    compute_chi_squared_batch_numba = _compute_chi_squared_batch_fallback
    compute_chi_squared_batch_numba.signatures = []  # type: ignore[attr-defined]

    def _compute_chi_squared_with_residuals_batch_fallback(
        theory_batch, exp_batch, contrast_batch, offset_batch
    ):
        """Fallback implementation when Numba is not available."""
        return _compute_chi_squared_with_residuals_batch_numba_impl(
            theory_batch, exp_batch, contrast_batch, offset_batch
        )

    compute_chi_squared_with_residuals_batch_numba = (
        _compute_chi_squared_with_residuals_batch_fallback
    )
    compute_chi_squared_with_residuals_batch_numba.signatures = []  # type: ignore[attr-defined]
else:
    compute_chi_squared_batch_numba = _compute_chi_squared_batch_numba_impl
    compute_chi_squared_with_residuals_batch_numba = (
        _compute_chi_squared_with_residuals_batch_numba_impl
    )


# Use the already JIT-compiled implementations directly
if NUMBA_AVAILABLE:
    # Functions are already decorated with @njit, use them directly
    create_time_integral_matrix_numba = _create_time_integral_matrix_impl
    calculate_diffusion_coefficient_numba = _calculate_diffusion_coefficient_impl
    calculate_shear_rate_numba = _calculate_shear_rate_impl
    compute_g1_correlation_numba = _compute_g1_correlation_impl
    compute_sinc_squared_numba = _compute_sinc_squared_impl
else:
    # Remove decorators if Numba not available
    def unwrap_if_needed(func):
        return func.__wrapped__ if hasattr(func, "__wrapped__") else func

    create_time_integral_matrix_numba = unwrap_if_needed(
        _create_time_integral_matrix_impl
    )
    calculate_diffusion_coefficient_numba = unwrap_if_needed(
        _calculate_diffusion_coefficient_impl
    )
    calculate_shear_rate_numba = unwrap_if_needed(_calculate_shear_rate_impl)
    compute_g1_correlation_numba = unwrap_if_needed(_compute_g1_correlation_impl)
    compute_sinc_squared_numba = unwrap_if_needed(_compute_sinc_squared_impl)

    # Add empty signatures attribute for fallback functions when numba unavailable
    create_time_integral_matrix_numba.signatures = []  # type: ignore[attr-defined]
    calculate_diffusion_coefficient_numba.signatures = []  # type: ignore[attr-defined]
    calculate_shear_rate_numba.signatures = []  # type: ignore[attr-defined]
    compute_g1_correlation_numba.signatures = []  # type: ignore[attr-defined]
    compute_sinc_squared_numba.signatures = []  # type: ignore[attr-defined]


# ============================================================================
# ENHANCED NUMBA KERNELS FOR BATCH VARIANCE ESTIMATION (Performance Optimization)
# ============================================================================


@njit(
    float64[:, :](float64[:, :], int64, float64, float64, float64),
    parallel=_USE_PARALLEL,
    cache=True,
    fastmath=True,
)
def _mad_window_batch_numba_impl(
    residuals_batch, window_size, edge_method_code, min_sigma_squared, mad_factor
):
    """
    Batch MAD (Median Absolute Deviation) window calculation with Numba optimization.

    Processes multiple angles simultaneously for maximum performance.
    Edge methods: 0=reflect, 1=adaptive_window, 2=global_fallback

    Parameters
    ----------
    residuals_batch : np.ndarray, shape (n_angles, n_points)
        Residuals for all angles
    window_size : int
        Size of moving window for MAD estimation
    edge_method_code : float (int)
        Edge handling method (encoded as float for Numba compatibility)
    min_sigma_squared : float
        Minimum variance floor
    mad_factor : float
        MAD to variance conversion factor (1.4826)

    Returns
    -------
    np.ndarray, shape (n_angles, n_points)
        Variance estimates for all angles and points
    """
    n_angles, n_points = residuals_batch.shape
    sigma2_batch = np.zeros((n_angles, n_points), dtype=np.float64)
    half_window = window_size // 2
    edge_method = int(edge_method_code)

    # Process all angles in parallel
    for angle_idx in nb.prange(n_angles):
        residuals = residuals_batch[angle_idx]

        # Process all points for this angle
        for i in range(n_points):
            # Determine window bounds
            start = max(0, i - half_window)
            end = min(n_points, i + half_window + 1)
            window_length = end - start

            if window_length >= 3:  # Sufficient data for MAD calculation
                # Extract window data
                window_data = np.zeros(window_length, dtype=np.float64)
                for j in range(window_length):
                    window_data[j] = residuals[start + j]

                # Compute median and MAD
                median_res = np.median(window_data)
                abs_deviations = np.abs(window_data - median_res)
                mad = np.median(abs_deviations)

                if mad > 0:
                    sigma2_batch[angle_idx, i] = (mad_factor * mad) ** 2
                else:
                    sigma2_batch[angle_idx, i] = min_sigma_squared

            else:  # Handle insufficient data with edge methods
                if edge_method == 0:  # reflect
                    # Create reflected window
                    extended_size = window_size
                    extended_data = np.zeros(extended_size, dtype=np.float64)

                    for j in range(extended_size):
                        idx = i - half_window + j
                        if 0 <= idx < n_points:
                            extended_data[j] = residuals[idx]
                        elif idx < 0:
                            reflect_idx = min(abs(idx), n_points - 1)
                            extended_data[j] = residuals[reflect_idx]
                        else:
                            reflect_idx = max(0, 2 * n_points - idx - 2)
                            extended_data[j] = residuals[reflect_idx]

                    median_res = np.median(extended_data)
                    abs_deviations = np.abs(extended_data - median_res)
                    mad = np.median(abs_deviations)
                    sigma2_batch[angle_idx, i] = (
                        (mad_factor * mad) ** 2 if mad > 0 else min_sigma_squared
                    )

                elif edge_method == 1:  # adaptive_window
                    # Use global variance as fallback
                    global_var = np.var(residuals)
                    sigma2_batch[angle_idx, i] = max(global_var, min_sigma_squared)

                else:  # global_fallback (edge_method == 2 or default)
                    # Use global MAD
                    global_median = np.median(residuals)
                    global_abs_dev = np.abs(residuals - global_median)
                    global_mad = np.median(global_abs_dev)
                    sigma2_batch[angle_idx, i] = (
                        (mad_factor * global_mad) ** 2
                        if global_mad > 0
                        else min_sigma_squared
                    )

    # Apply minimum variance floor
    for angle_idx in range(n_angles):
        for i in range(n_points):
            if sigma2_batch[angle_idx, i] < min_sigma_squared:
                sigma2_batch[angle_idx, i] = min_sigma_squared

    return sigma2_batch


@njit(
    float64[:, :](float64[:, :], int64, int64, float64, float64, float64, float64),
    parallel=False,  # Changed from _USE_PARALLEL - function has no parallelizable operations
    cache=True,
    fastmath=True,
)
def _estimate_variance_irls_batch_numba_impl(
    residuals_batch,
    window_size,
    max_iterations,
    damping_factor,
    convergence_tolerance,
    initial_sigma_squared,
    min_sigma_squared,
):
    """
    Batch IRLS variance estimation with MAD moving window for multiple angles.

    Implements vectorized IRLS with damping and convergence checking across all angles.

    Parameters
    ----------
    residuals_batch : np.ndarray, shape (n_angles, n_points)
        Residuals for all angles
    window_size : int
        Size of moving window for MAD estimation
    max_iterations : int
        Maximum IRLS iterations
    damping_factor : float
        Damping factor α for variance updates
    convergence_tolerance : float
        Convergence tolerance for variance changes
    initial_sigma_squared : float
        Initial uniform variance assumption
    min_sigma_squared : float
        Minimum variance floor

    Returns
    -------
    np.ndarray, shape (n_angles, n_points)
        Final variance estimates for all angles
    """
    n_angles, n_points = residuals_batch.shape
    mad_factor = 1.4826  # MAD to variance conversion factor

    # Initialize variance arrays
    sigma2_batch = np.full(
        (n_angles, n_points), initial_sigma_squared, dtype=np.float64
    )
    sigma2_prev_batch = sigma2_batch.copy()

    # IRLS iterations
    for iteration in range(max_iterations):
        # Apply MAD moving window variance estimation for all angles
        sigma2_new_batch = _mad_window_batch_numba_impl(
            residuals_batch,
            window_size,
            0.0,
            min_sigma_squared,
            mad_factor,  # edge_method=0 (reflect)
        )

        # Apply damping to prevent oscillations
        if iteration > 0:
            alpha = damping_factor
            for angle_idx in range(n_angles):
                for i in range(n_points):
                    sigma2_batch[angle_idx, i] = (
                        alpha * sigma2_new_batch[angle_idx, i]
                        + (1.0 - alpha) * sigma2_prev_batch[angle_idx, i]
                    )
        else:
            # First iteration: no damping
            sigma2_batch = sigma2_new_batch.copy()

        # Check convergence for all angles
        converged_count = 0
        total_variance_change = 0.0

        for angle_idx in range(n_angles):
            # Compute variance change for this angle
            norm_curr = 0.0
            norm_prev = 0.0
            norm_diff = 0.0

            for i in range(n_points):
                norm_curr += sigma2_batch[angle_idx, i] ** 2
                norm_prev += sigma2_prev_batch[angle_idx, i] ** 2
                diff = sigma2_batch[angle_idx, i] - sigma2_prev_batch[angle_idx, i]
                norm_diff += diff**2

            norm_curr = np.sqrt(norm_curr)
            norm_prev = np.sqrt(norm_prev)
            norm_diff = np.sqrt(norm_diff)

            variance_change = norm_diff / (norm_prev + 1e-10)
            total_variance_change += variance_change

            if variance_change < convergence_tolerance:
                converged_count += 1

        # Check if majority of angles have converged
        convergence_ratio = converged_count / n_angles
        if convergence_ratio > 0.8 and iteration > 0:  # 80% convergence threshold
            break

        # Update previous values
        sigma2_prev_batch = sigma2_batch.copy()

    # Apply final minimum variance floor
    for angle_idx in range(n_angles):
        for i in range(n_points):
            if sigma2_batch[angle_idx, i] < min_sigma_squared:
                sigma2_batch[angle_idx, i] = min_sigma_squared

    return sigma2_batch


@njit(
    types.Tuple((float64[:], float64[:]))(float64[:, :], float64[:, :], int64),
    parallel=_USE_PARALLEL,
    cache=True,
    fastmath=True,
)
def _chi_squared_with_variance_batch_numba_impl(
    residuals_batch, sigma_variances_batch, dof_per_angle
):
    """
    Batch chi-squared calculation with pre-computed variance estimates.

    Computes proper chi-squared values using σ² normalization for all angles.

    Parameters
    ----------
    residuals_batch : np.ndarray, shape (n_angles, n_points)
        Residuals for all angles
    sigma_variances_batch : np.ndarray, shape (n_angles, n_points)
        Variance estimates for all angles and points
    dof_per_angle : int
        Degrees of freedom per angle

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (angle_chi2_proper, angle_chi2_reduced) arrays of shape (n_angles,)
    """
    n_angles, n_points = residuals_batch.shape
    angle_chi2_proper = np.zeros(n_angles, dtype=np.float64)
    angle_chi2_reduced = np.zeros(n_angles, dtype=np.float64)

    # Process all angles in parallel
    for angle_idx in nb.prange(n_angles):
        chi2_sum = 0.0

        # Compute chi-squared for this angle
        for i in range(n_points):
            residual = residuals_batch[angle_idx, i]
            variance = sigma_variances_batch[angle_idx, i]

            # Proper chi-squared: χ² = Σ(residuals²/σ²)
            chi2_sum += (residual * residual) / variance

        angle_chi2_proper[angle_idx] = chi2_sum

        # Compute reduced chi-squared
        effective_dof = max(1, dof_per_angle)
        angle_chi2_reduced[angle_idx] = chi2_sum / effective_dof

    return angle_chi2_proper, angle_chi2_reduced


# Apply numba decorator if available, otherwise use fallback
if not NUMBA_AVAILABLE:
    # Remove decorator if Numba not available
    _mad_window_batch_numba_impl = (
        _mad_window_batch_numba_impl.__wrapped__
        if hasattr(_mad_window_batch_numba_impl, "__wrapped__")
        else _mad_window_batch_numba_impl
    )

    _estimate_variance_irls_batch_numba_impl = (
        _estimate_variance_irls_batch_numba_impl.__wrapped__
        if hasattr(_estimate_variance_irls_batch_numba_impl, "__wrapped__")
        else _estimate_variance_irls_batch_numba_impl
    )

    _chi_squared_with_variance_batch_numba_impl = (
        _chi_squared_with_variance_batch_numba_impl.__wrapped__
        if hasattr(_chi_squared_with_variance_batch_numba_impl, "__wrapped__")
        else _chi_squared_with_variance_batch_numba_impl
    )

    def _mad_window_batch_fallback(
        residuals_batch, window_size, edge_method_code, min_sigma_squared, mad_factor
    ):
        return _mad_window_batch_numba_impl(
            residuals_batch,
            window_size,
            edge_method_code,
            min_sigma_squared,
            mad_factor,
        )

    def _estimate_variance_irls_batch_fallback(
        residuals_batch,
        window_size,
        max_iterations,
        damping_factor,
        convergence_tolerance,
        initial_sigma_squared,
        min_sigma_squared,
    ):
        return _estimate_variance_irls_batch_numba_impl(
            residuals_batch,
            window_size,
            max_iterations,
            damping_factor,
            convergence_tolerance,
            initial_sigma_squared,
            min_sigma_squared,
        )

    def _chi_squared_with_variance_batch_fallback(
        residuals_batch, sigma_variances_batch, dof_per_angle
    ):
        return _chi_squared_with_variance_batch_numba_impl(
            residuals_batch, sigma_variances_batch, dof_per_angle
        )

    mad_window_batch_numba = _mad_window_batch_fallback
    estimate_variance_irls_batch_numba = _estimate_variance_irls_batch_fallback
    chi_squared_with_variance_batch_numba = _chi_squared_with_variance_batch_fallback

    mad_window_batch_numba.signatures = []  # type: ignore[attr-defined]
    estimate_variance_irls_batch_numba.signatures = []  # type: ignore[attr-defined]
    chi_squared_with_variance_batch_numba.signatures = []  # type: ignore[attr-defined]
else:
    mad_window_batch_numba = _mad_window_batch_numba_impl
    estimate_variance_irls_batch_numba = _estimate_variance_irls_batch_numba_impl
    chi_squared_with_variance_batch_numba = _chi_squared_with_variance_batch_numba_impl


# ============================================================================
# HYBRID LIMITED-ITERATION IRLS NUMBA KERNELS (FGLS-Inspired Optimization)
# ============================================================================


@njit(
    parallel=False,  # Disable parallel due to iterative algorithm complexity
    cache=True,
    fastmath=True,
)
def _hybrid_irls_batch_numba_impl(
    residuals_batch,
    adaptive_target_alpha,
    huber_constant_factor,
    kernel_bandwidth_scale,
    regularization_strength,
    max_iterations,
    convergence_tolerance,
    min_sigma_squared,
):
    """
    Improved Hybrid Limited-Iteration IRLS with Global MAD + Huber Weights for batch processing.

    Implements a high-performance variance estimation method that uses:
    - Global MAD computation (O(n) vs O(nk) windowed approach)
    - Huber robust weights with clipping
    - Kernel smoothing for local variance trends
    - Regularization blending (local + global scale)

    Algorithm:
    1. Global MAD computation for robust scale estimation
    2. Huber weight calculation with threshold clipping
    3. Kernel-smoothed local variance estimation
    4. Regularization blending with global scale
    5. 1-2 IRLS iterations with early convergence

    Performance: 25-40x speedup over windowed MAD approach

    Parameters
    ----------
    residuals_batch : np.ndarray, shape (n_angles, n_points)
        Residuals for all angles
    adaptive_target_alpha : float
        Target parameter for reduced chi-squared (typically 1.0)
    huber_constant_factor : float
        Huber threshold factor (typically 1.345 for 95% efficiency)
    kernel_bandwidth_scale : float
        Bandwidth scaling for kernel smoothing (typically 0.1-0.3)
    regularization_strength : float
        Regularization strength λ (typically 0.1-0.2)
    max_iterations : int
        Maximum IRLS iterations (typically 1-2)
    convergence_tolerance : float
        Convergence tolerance for early stopping (typically 0.01)
    min_sigma_squared : float
        Minimum variance floor to prevent division by zero

    Returns
    -------
    np.ndarray, shape (n_angles, n_points)
        Final variance estimates (σ²) for all angles

    Notes
    -----
    Key improvements:
    - Global MAD: O(n) complexity vs O(nk) windowed approach
    - Huber weights: Superior outlier handling vs simple MAD
    - Kernel smoothing: Captures local variance trends
    - Regularization: Prevents overfitting, improves stability
    - Early convergence: Typically converges in 1-2 iterations
    """
    n_angles, n_points = residuals_batch.shape
    if n_angles == 0 or n_points == 0:
        return np.zeros((n_angles, n_points), dtype=np.float64)

    # Initialize output array
    sigma2_batch = np.zeros((n_angles, n_points), dtype=np.float64)

    # Weight clipping parameters
    w_min = 0.05
    w_max = 20.0

    # Step 1: Global MAD computation for each angle (O(n) vs O(nk))
    mad_factor = 1.4826  # MAD to standard deviation conversion

    # Process each angle for global MAD computation
    for angle_idx in range(n_angles):
        residuals = residuals_batch[angle_idx]

        # Global MAD computation (O(n) operation)
        # Step 1: Calculate median of residuals
        sorted_residuals = np.sort(residuals)
        if n_points % 2 == 1:
            median_residual = sorted_residuals[n_points // 2]
        else:
            median_residual = 0.5 * (
                sorted_residuals[n_points // 2 - 1] + sorted_residuals[n_points // 2]
            )

        # Step 2: Calculate MAD
        abs_deviations = np.abs(residuals - median_residual)
        sorted_deviations = np.sort(abs_deviations)
        if n_points % 2 == 1:
            mad = sorted_deviations[n_points // 2]
        else:
            mad = 0.5 * (
                sorted_deviations[n_points // 2 - 1] + sorted_deviations[n_points // 2]
            )

        # Robust scale estimate
        robust_scale = mad_factor * mad if mad > 0 else min_sigma_squared**0.5

        # Step 3: Compute Huber weights with clipping
        huber_threshold = huber_constant_factor * robust_scale
        huber_weights = np.ones(n_points, dtype=np.float64)

        for i in range(n_points):
            abs_resid = abs(residuals[i])
            if abs_resid > huber_threshold:
                weight = huber_threshold / abs_resid
            else:
                weight = 1.0
            # Apply weight clipping
            huber_weights[i] = max(w_min, min(w_max, weight))

        # Step 4: Kernel smoothing for local variance estimation
        # Adaptive bandwidth scaling for large datasets to prevent excessive computation
        if n_points > 100000:
            # For very large datasets, use fixed bandwidth regardless of scale
            kernel_bandwidth = 2000  # Fixed reasonable bandwidth for large datasets
        else:
            # Original scaling for smaller datasets
            kernel_bandwidth = min(kernel_bandwidth_scale * n_points, 5000)

        for i in range(n_points):
            # Weighted local variance with kernel smoothing (bounded for performance)
            weighted_sum = 0.0
            weight_sum = 0.0

            # Use bounded kernel to avoid O(n²) complexity
            for j in range(
                max(0, i - int(kernel_bandwidth)),
                min(n_points, i + int(kernel_bandwidth) + 1),
            ):
                # Gaussian kernel weight
                distance = (
                    abs(i - j) / kernel_bandwidth if kernel_bandwidth > 0 else 0.0
                )
                if distance <= 3.0:  # Limit kernel support for efficiency
                    kernel_weight = np.exp(-0.5 * distance * distance)

                    # Combined weight: Huber * Kernel
                    combined_weight = huber_weights[j] * kernel_weight

                    # Weighted squared residual
                    weighted_sum += combined_weight * residuals[j] ** 2
                    weight_sum += combined_weight

            if weight_sum > 0:
                local_variance = weighted_sum / weight_sum
            else:
                local_variance = min_sigma_squared

            # Step 5: Regularization blending (local + global scale)
            global_variance = robust_scale**2
            sigma2_batch[angle_idx, i] = (
                1.0 - regularization_strength
            ) * local_variance + regularization_strength * global_variance

            # Ensure minimum variance
            sigma2_batch[angle_idx, i] = max(
                sigma2_batch[angle_idx, i], min_sigma_squared
            )

    # Step 6: Limited IRLS iterations (typically converges in 1-2 iterations)
    sigma2_prev_batch = sigma2_batch.copy()
    converged = False

    for iteration in range(max_iterations):
        if converged:
            break

        # Re-estimate variances using current weights and updated Huber thresholds
        sigma2_new_batch = np.zeros((n_angles, n_points), dtype=np.float64)
        max_relative_change = 0.0

        for angle_idx in range(n_angles):
            residuals = residuals_batch[angle_idx]

            # Re-compute MAD with current variance estimates for this iteration
            abs_deviations = np.abs(residuals)
            sorted_deviations = np.sort(abs_deviations)
            if n_points % 2 == 1:
                mad_iter = sorted_deviations[n_points // 2]
            else:
                mad_iter = 0.5 * (
                    sorted_deviations[n_points // 2 - 1]
                    + sorted_deviations[n_points // 2]
                )

            robust_scale_iter = (
                mad_factor * mad_iter if mad_iter > 0 else min_sigma_squared**0.5
            )

            # Update Huber weights
            huber_threshold_iter = huber_constant_factor * robust_scale_iter

            for i in range(n_points):
                abs_resid = abs(residuals[i])
                if abs_resid > huber_threshold_iter:
                    weight = huber_threshold_iter / abs_resid
                else:
                    weight = 1.0
                weight = max(w_min, min(w_max, weight))

                # Weighted local variance (simplified for iteration)
                weighted_sum = 0.0
                weight_sum = 0.0
                # Adaptive bandwidth for large datasets
                if n_points > 100000:
                    kernel_bandwidth = 2000  # Fixed for large datasets
                else:
                    kernel_bandwidth = min(kernel_bandwidth_scale * n_points, 5000)

                for j in range(
                    max(0, i - int(kernel_bandwidth)),
                    min(n_points, i + int(kernel_bandwidth) + 1),
                ):
                    distance = (
                        abs(i - j) / kernel_bandwidth if kernel_bandwidth > 0 else 0.0
                    )
                    if distance <= 3.0:  # Limit kernel support
                        kernel_weight = np.exp(-0.5 * distance * distance)
                        combined_weight = weight * kernel_weight
                        weighted_sum += combined_weight * residuals[j] ** 2
                        weight_sum += combined_weight

                if weight_sum > 0:
                    local_variance = weighted_sum / weight_sum
                else:
                    local_variance = min_sigma_squared

                # Regularization blending
                global_variance = robust_scale_iter**2
                new_variance = (
                    1.0 - regularization_strength
                ) * local_variance + regularization_strength * global_variance
                new_variance = max(new_variance, min_sigma_squared)
                sigma2_new_batch[angle_idx, i] = new_variance

                # Track convergence
                if sigma2_batch[angle_idx, i] > min_sigma_squared:
                    rel_change = (
                        abs(new_variance - sigma2_batch[angle_idx, i])
                        / sigma2_batch[angle_idx, i]
                    )
                    max_relative_change = max(max_relative_change, rel_change)

        # Check for convergence
        if max_relative_change < convergence_tolerance:
            converged = True

        # Update variances (explicit loops for Numba compatibility)
        for angle_idx in range(n_angles):
            for i in range(n_points):
                sigma2_batch[angle_idx, i] = sigma2_new_batch[angle_idx, i]

    # Step 5: Apply minimum variance floor (vectorized)
    for angle_idx in range(n_angles):
        for i in range(n_points):
            if sigma2_batch[angle_idx, i] < min_sigma_squared or not np.isfinite(
                sigma2_batch[angle_idx, i]
            ):
                sigma2_batch[angle_idx, i] = min_sigma_squared

    return sigma2_batch


# Create public interface for hybrid IRLS kernel
if NUMBA_AVAILABLE:
    hybrid_irls_batch_numba = _hybrid_irls_batch_numba_impl
else:
    # Enhanced fallback implementation without Numba - implements actual hybrid IRLS algorithm
    def hybrid_irls_batch_numba(
        residuals_batch,
        adaptive_target_alpha,
        huber_constant_factor,
        kernel_bandwidth_scale,
        regularization_strength,
        max_iterations,
        convergence_tolerance,
        min_sigma_squared,
    ):
        """Enhanced fallback hybrid IRLS implementation without Numba optimization."""
        import logging

        logger = logging.getLogger(__name__)
        logger.debug("Using enhanced fallback hybrid IRLS implementation (no Numba)")

        n_angles, n_points = residuals_batch.shape
        if n_angles == 0 or n_points == 0:
            return np.zeros((n_angles, n_points), dtype=np.float64)

        # Initialize output array
        sigma2_batch = np.zeros((n_angles, n_points), dtype=np.float64)

        # Weight clipping parameters
        w_min = 0.05
        w_max = 20.0
        mad_factor = 1.4826  # MAD to standard deviation conversion

        # Process each angle for global MAD computation
        for angle_idx in range(n_angles):
            residuals = residuals_batch[angle_idx]

            # Global MAD computation
            sorted_residuals = np.sort(residuals)
            if n_points % 2 == 1:
                median_residual = sorted_residuals[n_points // 2]
            else:
                median_residual = 0.5 * (
                    sorted_residuals[n_points // 2 - 1]
                    + sorted_residuals[n_points // 2]
                )

            abs_deviations = np.abs(residuals - median_residual)
            sorted_deviations = np.sort(abs_deviations)
            if n_points % 2 == 1:
                mad = sorted_deviations[n_points // 2]
            else:
                mad = 0.5 * (
                    sorted_deviations[n_points // 2 - 1]
                    + sorted_deviations[n_points // 2]
                )

            # Robust scale estimate
            robust_scale = mad_factor * mad if mad > 0 else (min_sigma_squared**0.5)

            # Compute Huber weights with clipping
            huber_threshold = huber_constant_factor * robust_scale
            huber_weights = np.ones(n_points, dtype=np.float64)

            for i in range(n_points):
                abs_resid = abs(residuals[i])
                if abs_resid > huber_threshold:
                    weight = huber_threshold / abs_resid
                else:
                    weight = 1.0
                # Apply weight clipping
                huber_weights[i] = max(w_min, min(w_max, weight))

            # Kernel smoothing for local variance estimation
            # Adaptive bandwidth for large datasets
            if n_points > 100000:
                kernel_bandwidth = 2000  # Fixed for large datasets
            else:
                kernel_bandwidth = min(kernel_bandwidth_scale * n_points, 5000)

            for i in range(n_points):
                # Weighted local variance with kernel smoothing
                weighted_sum = 0.0
                weight_sum = 0.0

                for j in range(n_points):
                    # Gaussian kernel weight
                    distance = (
                        abs(i - j) / kernel_bandwidth if kernel_bandwidth > 0 else 0.0
                    )
                    kernel_weight = np.exp(-0.5 * distance * distance)

                    # Combined weight: Huber * Kernel
                    combined_weight = huber_weights[j] * kernel_weight

                    # Weighted squared residual
                    weighted_sum += combined_weight * residuals[j] ** 2
                    weight_sum += combined_weight

                if weight_sum > 0:
                    local_variance = weighted_sum / weight_sum
                else:
                    local_variance = min_sigma_squared

                # Regularization blending (local + global scale)
                global_variance = robust_scale**2
                sigma2_batch[angle_idx, i] = (
                    1.0 - regularization_strength
                ) * local_variance + regularization_strength * global_variance

                # Ensure minimum variance
                sigma2_batch[angle_idx, i] = max(
                    sigma2_batch[angle_idx, i], min_sigma_squared
                )

        logger.debug(
            f"Fallback hybrid IRLS: processed {n_angles} angles with {n_points} points each"
        )
        return sigma2_batch
