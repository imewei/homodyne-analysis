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
    from numba import float64, int64, jit, njit, prange, types

    try:
        from numba.types import Tuple  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        # Fallback for older numba versions or different import paths
        Tuple = getattr(types, "Tuple", types.UniTuple)  # type: ignore[union-attr]

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


def _create_time_integral_matrix_impl(time_dependent_array):
    """Create time integral matrix for correlation calculations."""
    n = len(time_dependent_array)
    matrix = np.empty((n, n), dtype=np.float64)
    cumsum = np.cumsum(time_dependent_array)

    for i in range(n):
        cumsum_i = cumsum[i]
        for j in range(n):
            matrix[i, j] = abs(cumsum_i - cumsum[j])

    return matrix


def _calculate_diffusion_coefficient_impl(time_array, D0, alpha, D_offset):
    """Calculate time-dependent diffusion coefficient."""
    D_t = np.empty_like(time_array)
    for i in range(len(time_array)):
        D_value = D0 * (time_array[i] ** alpha) + D_offset
        D_t[i] = max(D_value, 1e-10)
    return D_t


def _calculate_shear_rate_impl(time_array, gamma_dot_t0, beta, gamma_dot_t_offset):
    """Calculate time-dependent shear rate."""
    gamma_dot_t = np.empty_like(time_array)
    for i in range(len(time_array)):
        gamma_value = gamma_dot_t0 * (time_array[i] ** beta) + gamma_dot_t_offset
        gamma_dot_t[i] = max(gamma_value, 1e-10)
    return gamma_dot_t


def _compute_g1_correlation_impl(diffusion_integral_matrix, wavevector_factor):
    """Compute field correlation function g₁ from diffusion."""
    shape = diffusion_integral_matrix.shape
    g1 = np.empty(shape, dtype=np.float64)

    for i in range(shape[0]):
        for j in range(shape[1]):
            exponent = -wavevector_factor * diffusion_integral_matrix[i, j]
            g1[i, j] = np.exp(exponent)

    return g1


def _compute_sinc_squared_impl(shear_integral_matrix, prefactor):
    """Compute sinc² function for shear flow contributions."""
    shape = shear_integral_matrix.shape
    sinc_squared = np.empty(shape, dtype=np.float64)
    pi = np.pi

    for i in range(shape[0]):
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

            # Manage cache size
            if len(cache) >= maxsize and maxsize > 0:
                # Remove 25% of least-accessed items
                items_to_remove = maxsize // 4
                sorted_items = sorted(access_count.items(), key=lambda x: x[1])

                for key, _ in sorted_items[:items_to_remove]:
                    cache.pop(key, None)
                    access_count.pop(key, None)

            # Store result
            if maxsize > 0:
                cache[cache_key] = result
                access_count[cache_key] = 1

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

    for i in range(n_angles):
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
def _solve_least_squares_batch_fallback(theory_batch, exp_batch):
    """Fallback implementation when Numba is not available."""
    return _solve_least_squares_batch_numba_impl(theory_batch, exp_batch)


if NUMBA_AVAILABLE:
    solve_least_squares_batch_numba = njit(
        cache=True,
        fastmath=True,
        nogil=True,
    )(_solve_least_squares_batch_numba_impl)
else:
    solve_least_squares_batch_numba = _solve_least_squares_batch_fallback
    # Add signatures attribute for compatibility with numba compiled functions
    solve_least_squares_batch_numba.signatures = []


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

    for i in range(n_angles):
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
if NUMBA_AVAILABLE:
    compute_chi_squared_batch_numba = njit(
        float64[:](float64[:, :], float64[:, :], float64[:], float64[:]),
        parallel=False,
        cache=True,
        fastmath=True,
        nogil=True,
    )(_compute_chi_squared_batch_numba_impl)
else:
    compute_chi_squared_batch_numba = _compute_chi_squared_batch_fallback
    # Add signatures attribute for compatibility with numba compiled functions
    compute_chi_squared_batch_numba.signatures = []


# Apply numba decorator to all other functions if available, otherwise use
# implementations directly
if NUMBA_AVAILABLE:
    create_time_integral_matrix_numba = njit(
        float64[:, :](float64[:]),
        parallel=False,
        cache=True,
        fastmath=True,
        nogil=True,
    )(_create_time_integral_matrix_impl)

    calculate_diffusion_coefficient_numba = njit(
        float64[:](float64[:], float64, float64, float64),
        cache=True,
        fastmath=True,
        parallel=False,
        nogil=True,
    )(_calculate_diffusion_coefficient_impl)

    calculate_shear_rate_numba = njit(
        float64[:](float64[:], float64, float64, float64),
        cache=True,
        fastmath=True,
        parallel=False,
    )(_calculate_shear_rate_impl)

    compute_g1_correlation_numba = njit(
        float64[:, :](float64[:, :], float64),
        parallel=False,
        cache=True,
        fastmath=True,
    )(_compute_g1_correlation_impl)

    compute_sinc_squared_numba = njit(
        float64[:, :](float64[:, :], float64),
        parallel=False,
        cache=True,
        fastmath=True,
    )(_compute_sinc_squared_impl)
else:
    create_time_integral_matrix_numba = _create_time_integral_matrix_impl
    calculate_diffusion_coefficient_numba = _calculate_diffusion_coefficient_impl
    calculate_shear_rate_numba = _calculate_shear_rate_impl
    compute_g1_correlation_numba = _compute_g1_correlation_impl
    compute_sinc_squared_numba = _compute_sinc_squared_impl

    # Add empty signatures attribute for fallback functions when numba
    # unavailable
    create_time_integral_matrix_numba.signatures = []
    calculate_diffusion_coefficient_numba.signatures = []
    calculate_shear_rate_numba.signatures = []
    compute_g1_correlation_numba.signatures = []
    compute_sinc_squared_numba.signatures = []
