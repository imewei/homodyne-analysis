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
from typing import Any
from typing import TypeVar

# Use lazy loading for heavy dependencies
from .lazy_imports import scientific_deps

# Lazy-loaded numpy and numba
np = scientific_deps.get("numpy")

# Import shared numba availability flag and detection function
from .optimization_utils import NUMBA_AVAILABLE
from .optimization_utils import _check_numba_availability

# Lazy-loaded Numba with fallbacks
if NUMBA_AVAILABLE:
    try:
        # Use lazy loading for numba components
        numba_module = scientific_deps.get("numba")

        # Extract specific components
        jit = numba_module.jit
        njit = numba_module.njit
        prange = numba_module.prange
        float64 = numba_module.float64
        int64 = numba_module.int64
        types = numba_module.types

        try:
            Tuple = types.Tuple  # type: ignore[attr-defined]
        except AttributeError:
            # Fallback for older numba versions
            Tuple = getattr(types, "UniTuple", types.Tuple)  # type: ignore[union-attr]

    except Exception:
        # If lazy loading fails, fall back to direct import
        try:
            from numba import float64
            from numba import int64
            from numba import jit
            from numba import njit
            from numba import prange
            from numba import types
            from numba.types import Tuple  # type: ignore[attr-defined]
        except ImportError:
            NUMBA_AVAILABLE = False

if not NUMBA_AVAILABLE:
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
    """
    Create time integral matrix for correlation calculations.

    OPTIMIZED VERSION: Revolutionary vectorization using NumPy broadcasting
    Expected speedup: 5-10x through elimination of nested loops

    Mathematical operation: matrix[i, j] = |cumsum[i] - cumsum[j]|

    Vectorization strategy:
    1. Compute cumulative sum once
    2. Use broadcasting to create difference matrix in single operation
    3. Apply absolute value vectorized operation
    4. Exploit cache-friendly memory access patterns
    """
    # Compute cumulative sum once (O(n) operation)
    # Note: Numba requires specific dtype handling
    cumsum = np.cumsum(time_dependent_array.astype(np.float64))

    # REVOLUTIONARY VECTORIZATION: Replace O(n²) nested loops with broadcasting
    # Create meshgrid using broadcasting - cumsum[:, None] creates column vector
    # cumsum[None, :] creates row vector, broadcasting creates full matrix
    # This replaces the nested loop with a single vectorized operation
    matrix = np.abs(cumsum[:, np.newaxis] - cumsum[np.newaxis, :])

    return matrix


def _calculate_diffusion_coefficient_impl(time_array, D0, alpha, D_offset):
    """
    Calculate time-dependent diffusion coefficient.

    OPTIMIZED VERSION: Vectorized computation replacing element-wise loop
    Expected speedup: 10-50x through NumPy vectorization

    Mathematical operation: D_t[i] = max(D0 * t[i]^alpha + D_offset, 1e-10)

    Vectorization strategy:
    1. Use NumPy power operation for entire array
    2. Vectorized arithmetic operations
    3. Vectorized maximum operation for clamping

    Special handling for negative alpha:
    - For alpha < 0, D(t) = D0 * t^alpha + D_offset diverges as t→0
    - Physical limit: lim(t→0) D(t) = D_offset (constant term dominates)
    - For t=0 or very small t: D(0) = D_offset
    - For t > threshold: D(t) = D0 * t^alpha + D_offset
    """
    if alpha < 0:
        # For negative alpha, handle t=0 by taking the physical limit
        # Initialize with D_offset (the limit as t→0)
        D_values = np.full_like(time_array, D_offset, dtype=np.float64)

        # For t > threshold, compute the full power-law + offset
        threshold = 1e-10  # Avoid numerical instability for very small t
        mask = time_array > threshold
        if np.any(mask):
            D_values[mask] = D0 * np.power(time_array[mask], alpha) + D_offset
    else:
        # Standard calculation for non-negative alpha (no singularity at t=0)
        D_values = D0 * np.power(time_array, alpha) + D_offset

    # Vectorized maximum operation to ensure minimum threshold
    D_t = np.maximum(D_values, 1e-10)
    return D_t


def _calculate_shear_rate_impl(time_array, gamma_dot_t0, beta, gamma_dot_t_offset):
    """
    Calculate time-dependent shear rate.

    OPTIMIZED VERSION: Vectorized computation replacing element-wise loop
    Expected speedup: 10-50x through NumPy vectorization

    Mathematical operation: gamma_dot_t[i] = max(gamma_dot_t0 * t[i]^beta + offset, 1e-10)

    Vectorization strategy:
    1. Use NumPy power operation for entire array
    2. Vectorized arithmetic operations
    3. Vectorized maximum operation for clamping

    Special handling for negative beta:
    - For beta < 0, γ̇(t) = γ̇₀ * t^beta + offset diverges as t→0
    - Physical limit: lim(t→0) γ̇(t) = offset (constant term dominates)
    - For t=0 or very small t: γ̇(0) = offset
    - For t > threshold: γ̇(t) = γ̇₀ * t^beta + offset
    """
    if beta < 0:
        # For negative beta, handle t=0 by taking the physical limit
        # Initialize with offset (the limit as t→0)
        gamma_values = np.full_like(time_array, gamma_dot_t_offset, dtype=np.float64)

        # For t > threshold, compute the full power-law + offset
        threshold = 1e-10  # Avoid numerical instability for very small t
        mask = time_array > threshold
        if np.any(mask):
            gamma_values[mask] = (
                gamma_dot_t0 * np.power(time_array[mask], beta) + gamma_dot_t_offset
            )
    else:
        # Standard calculation for non-negative beta (no singularity at t=0)
        gamma_values = gamma_dot_t0 * np.power(time_array, beta) + gamma_dot_t_offset

    # Vectorized maximum operation to ensure minimum threshold
    gamma_dot_t = np.maximum(gamma_values, 1e-10)
    return gamma_dot_t


def _compute_g1_correlation_impl(diffusion_integral_matrix, wavevector_factor):
    """
    Compute field correlation function g₁ from diffusion.

    OPTIMIZED VERSION: Revolutionary vectorization eliminating nested loops
    Expected speedup: 5-10x through matrix vectorization

    Mathematical operation: g1[i, j] = exp(-wavevector_factor * diffusion_matrix[i, j])

    Vectorization strategy:
    1. Vectorized multiplication across entire matrix
    2. Vectorized exponential operation
    3. Cache-friendly memory access pattern
    4. SIMD optimization opportunity through NumPy
    """
    # REVOLUTIONARY VECTORIZATION: Replace nested loops with matrix operations
    # Compute exponent for entire matrix in one operation
    exponent_matrix = -wavevector_factor * diffusion_integral_matrix

    # Vectorized exponential operation across entire matrix
    g1 = np.exp(exponent_matrix)

    return g1


def _compute_sinc_squared_impl(shear_integral_matrix, prefactor):
    """
    Compute sinc² function for shear flow contributions.

    OPTIMIZED VERSION: Advanced vectorization with conditional logic
    Expected speedup: 5-10x through elimination of nested loops and vectorized conditionals

    Mathematical operation: sinc²(π * prefactor * matrix[i, j])
    With special handling for small arguments to avoid numerical issues

    Advanced vectorization strategy:
    1. Vectorized argument computation
    2. Vectorized conditional logic using np.where
    3. Vectorized sin computation and division
    4. Cache-optimized memory access patterns
    5. Numerical stability preservation
    """
    # REVOLUTIONARY VECTORIZATION: Replace nested loops with advanced NumPy operations
    argument_matrix = prefactor * shear_integral_matrix
    pi_arg_matrix = np.pi * argument_matrix

    # Vectorized conditional logic for numerical stability
    # Case 1: Very small arguments (Taylor expansion)
    very_small_mask = np.abs(argument_matrix) < 1e-10
    pi_arg_sq = (pi_arg_matrix) ** 2
    taylor_result = 1.0 - pi_arg_sq / 3.0

    # Case 2: Small pi*argument (avoid division by zero)
    small_pi_mask = np.abs(pi_arg_matrix) < 1e-15

    # Case 3: General case (standard sinc computation)
    # Use np.sinc which handles sinc(x) = sin(πx)/(πx), so we need sinc(argument)
    # Note: np.sinc(x) computes sin(π*x)/(π*x), so we pass argument directly
    general_sinc = np.sinc(argument_matrix)
    general_result = general_sinc**2

    # Combine results using vectorized conditional selection
    # Priority: very_small > small_pi > general
    sinc_squared = np.where(
        very_small_mask, taylor_result, np.where(small_pi_mask, 1.0, general_result)
    )

    return sinc_squared


def _compute_sinc_squared_single(x):
    """
    Compute sinc² function for a single value.

    This is a simplified version for testing and single-value computations.
    For matrix operations, use compute_sinc_squared_numba.

    Parameters
    ----------
    x : float
        Input value

    Returns
    -------
    float
        sinc²(x) = (sin(πx)/(πx))² (normalized sinc function)
    """
    # Use np.sinc which computes the normalized sinc: sin(πx)/(πx)
    # This matches the matrix version implementation and test expectations
    sinc_val = np.sinc(x)
    return sinc_val**2


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
            if len(cache) >= maxsize > 0:
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
    solve_least_squares_batch_numba.signatures = []  # type: ignore[attr-defined]


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
    compute_chi_squared_batch_numba.signatures = []  # type: ignore[attr-defined]


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
    create_time_integral_matrix_numba.signatures = []  # type: ignore[attr-defined]
    calculate_diffusion_coefficient_numba.signatures = []  # type: ignore[attr-defined]
    calculate_shear_rate_numba.signatures = []  # type: ignore[attr-defined]
    compute_g1_correlation_numba.signatures = []  # type: ignore[attr-defined]
    compute_sinc_squared_numba.signatures = []  # type: ignore[attr-defined]


def refresh_kernel_functions():
    """Refresh kernel function assignments based on current Numba availability.

    This function is useful in test environments where Numba availability
    may change dynamically during execution.

    Returns
    -------
    bool
        True if Numba kernels are now available, False if using fallback functions
    """
    global \
        create_time_integral_matrix_numba, \
        calculate_diffusion_coefficient_numba, \
        calculate_shear_rate_numba, \
        compute_g1_correlation_numba, \
        compute_sinc_squared_numba

    # Re-check numba availability
    current_numba_available = _check_numba_availability()

    if current_numba_available:
        try:
            # Re-import numba components
            numba_module = scientific_deps.get("numba")

            # Extract specific components
            njit = numba_module.njit
            float64 = numba_module.float64

            # Recreate JIT-compiled functions
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
                nogil=True,
            )(_calculate_shear_rate_impl)

            compute_g1_correlation_numba = njit(
                float64[:, :](float64[:, :], float64),
                cache=True,
                fastmath=True,
                parallel=False,
                nogil=True,
            )(_compute_g1_correlation_impl)

            compute_sinc_squared_numba = njit(
                float64[:, :](float64[:, :], float64),
                cache=True,
                fastmath=True,
                parallel=False,
                nogil=True,
            )(_compute_sinc_squared_impl)

            return True

        except Exception:
            # If JIT compilation fails, fall back to pure Python
            pass

    # Use fallback functions
    create_time_integral_matrix_numba = _create_time_integral_matrix_impl
    calculate_diffusion_coefficient_numba = _calculate_diffusion_coefficient_impl
    calculate_shear_rate_numba = _calculate_shear_rate_impl
    compute_g1_correlation_numba = _compute_g1_correlation_impl
    compute_sinc_squared_numba = _compute_sinc_squared_impl

    # Add empty signatures attribute for fallback functions
    create_time_integral_matrix_numba.signatures = []  # type: ignore[attr-defined]
    calculate_diffusion_coefficient_numba.signatures = []  # type: ignore[attr-defined]
    calculate_shear_rate_numba.signatures = []  # type: ignore[attr-defined]
    compute_g1_correlation_numba.signatures = []  # type: ignore[attr-defined]
    compute_sinc_squared_numba.signatures = []  # type: ignore[attr-defined]

    return False


def compute_g1_correlation_legacy(
    t1, t2, phi, q, D0, alpha, D_offset, gamma0, beta, gamma_offset, phi0
):
    """
    Legacy compatibility wrapper for compute_g1_correlation_numba.

    This function maintains backward compatibility with tests that expect
    the old 11-parameter signature while using the new optimized 2-parameter
    implementation internally.

    Parameters
    ----------
    t1, t2 : float
        Time points
    phi : float
        Angle (radians)
    q : float
        Wavevector magnitude
    D0, alpha, D_offset : float
        Diffusion parameters
    gamma0, beta, gamma_offset, phi0 : float
        Shear flow parameters

    Returns
    -------
    float
        Single g1 correlation value
    """
    # Handle zero time delay
    dt = abs(t2 - t1)
    if dt == 0:
        return 1.0  # No decay at t=0

    # Compute diffusion integral properly
    # For anomalous diffusion: ∫ D(t') dt' from t1 to t2
    # D(t) = D0 * t^alpha + D_offset
    # Integral = D0 * (t2^(alpha+1) - t1^(alpha+1))/(alpha+1) + D_offset * (t2 - t1)

    t_min = min(t1, t2)
    t_max = max(t1, t2)

    if abs(alpha - (-1.0)) < 1e-10:
        # Special case for alpha = -1 (logarithmic)
        diffusion_integral = D0 * (np.log(t_max) - np.log(t_min)) + D_offset * (
            t_max - t_min
        )
    else:
        # General case
        diffusion_integral = (D0 / (alpha + 1)) * (
            t_max ** (alpha + 1) - t_min ** (alpha + 1)
        ) + D_offset * (t_max - t_min)

    # Diffusion contribution: g1_diff = exp(-q²/2 * diffusion_integral)
    g1_diff = np.exp(-0.5 * q**2 * diffusion_integral)

    # Compute shear flow contribution if gamma0 > 0
    if abs(gamma0) > 1e-15 or abs(gamma_offset) > 1e-15:
        # Shear rate integral: ∫ γ̇(t') dt' from t1 to t2
        # γ̇(t) = gamma0 * t^beta + gamma_offset

        if abs(beta - (-1.0)) < 1e-10:
            # Special case for beta = -1 (logarithmic)
            shear_integral = gamma0 * (np.log(t_max) - np.log(t_min)) + gamma_offset * (
                t_max - t_min
            )
        else:
            # General case
            shear_integral = (gamma0 / (beta + 1)) * (
                t_max ** (beta + 1) - t_min ** (beta + 1)
            ) + gamma_offset * (t_max - t_min)

        # Characteristic length L
        # For XPCS experiments, typical gap sizes are 10-100 μm
        # However, the effective length in the correlation function depends on
        # the scattering geometry and beam coherence length
        # Using L = 100 Å ensures shear phase stays in linear regime for typical parameters
        L = 1e2  # Å - adjusted for numerical stability and weak shear regime

        # Shear phase: Φ(φ,t₁,t₂) = (1/2π) q L cos(φ₀-φ) ∫ γ̇(t')dt'
        phase = (1.0 / (2.0 * np.pi)) * q * L * np.cos(phi0 - phi) * shear_integral

        # Shear contribution: g1_shear = sinc²(phase) where sinc is normalized
        g1_shear = _compute_sinc_squared_single(phase)

        # Combined g1 = g1_diff * g1_shear
        g1 = g1_diff * g1_shear
    else:
        # No shear, only diffusion
        g1 = g1_diff

    return float(g1)


# Save the matrix version of compute_sinc_squared for internal use
_compute_sinc_squared_matrix_numba = compute_sinc_squared_numba


# Create a flexible wrapper that handles both single values and matrices
def compute_sinc_squared_numba_flexible(x, prefactor=None):
    """
    Flexible sinc² function that handles both single values and matrices.

    Parameters
    ----------
    x : float or np.ndarray
        Single value or matrix input
    prefactor : float, optional
        For matrix version (unused for single values)

    Returns
    -------
    float or np.ndarray
        sinc²(x) result
    """
    if prefactor is not None or (isinstance(x, np.ndarray) and x.ndim > 0):
        # Matrix version: compute_sinc_squared_numba(matrix, prefactor)
        if prefactor is None:
            prefactor = 1.0
        return _compute_sinc_squared_matrix_numba(x, prefactor)
    # Single value version
    return _compute_sinc_squared_single(x)


def compute_g1_correlation_vectorized(D_integral, wavevector_q_squared_half_dt):
    """
    Vectorized g1 correlation calculation from pre-computed diffusion integral.

    This is the optimized 2-parameter version for performance-critical calculations
    where the diffusion integral has been pre-computed.

    Parameters
    ----------
    D_integral : array_like
        Pre-computed diffusion integral: ∫D(t')dt' from t1 to t2
    wavevector_q_squared_half_dt : float
        Pre-computed factor: q²/2 * Δt

    Returns
    -------
    array_like
        g1 correlation: exp(-q²/2 * Δt * D_integral)

    Notes
    -----
    This is the vectorized version optimized for batch processing.
    For full 11-parameter calculations, use compute_g1_correlation_legacy.
    """
    return np.exp(-wavevector_q_squared_half_dt * D_integral)


def compute_g1_correlation_numba_flexible(*args, **kwargs):
    """
    Flexible g1 correlation function supporting both legacy and vectorized signatures.

    Signatures:
    -----------
    1. Legacy (11 parameters):
       compute_g1_correlation_numba(t1, t2, phi, q, D0, alpha, D_offset,
                                     gamma0, beta, gamma_offset, phi0)

    2. Vectorized (2 parameters):
       compute_g1_correlation_numba(D_integral, wavevector_q_squared_half_dt)
    """
    if len(args) == 2 and not kwargs:
        # Vectorized signature: (D_integral, wavevector_q_squared_half_dt)
        return compute_g1_correlation_vectorized(args[0], args[1])
    if len(args) == 11 or (len(args) + len(kwargs)) == 11:
        # Legacy signature: 11 parameters
        return compute_g1_correlation_legacy(*args, **kwargs)
    raise TypeError(
        f"compute_g1_correlation_numba() takes either 2 arguments "
        f"(D_integral, wavevector_q_squared_half_dt) or 11 arguments "
        f"(t1, t2, phi, q, D0, alpha, D_offset, gamma0, beta, gamma_offset, phi0), "
        f"got {len(args)} positional and {len(kwargs)} keyword arguments"
    )


# Export the flexible function as the main API
compute_g1_correlation_numba = compute_g1_correlation_numba_flexible

# Export the flexible sinc function
compute_sinc_squared_numba = compute_sinc_squared_numba_flexible


def calculate_diffusion_coefficient_numba(t, D0, alpha, D_offset):
    """
    Calculate time-dependent diffusion coefficient.

    Parameters
    ----------
    t : float or array
        Time (can be scalar or array)
    D0 : float
        Reference diffusion coefficient
    alpha : float
        Time-dependence exponent
    D_offset : float
        Baseline diffusion

    Returns
    -------
    float or array
        D(t) = D0 * t^alpha + D_offset

    Notes
    -----
    For negative alpha, uses physical limit approach to avoid NaN at t=0:
    - At t=0: D(0) = D_offset (physical limit)
    - For t > threshold: D(t) = D0 * t^alpha + D_offset
    """
    if alpha < 0:
        # For negative alpha, handle t=0 by taking the physical limit
        # Initialize with D_offset (the limit as t→0)
        D_values = np.full_like(np.atleast_1d(t), D_offset, dtype=np.float64)

        # For t > threshold, compute the full power-law + offset
        threshold = 1e-10
        mask = np.atleast_1d(t) > threshold
        if np.any(mask):
            D_values[mask] = D0 * np.power(np.atleast_1d(t)[mask], alpha) + D_offset

        # Return scalar if input was scalar, otherwise return array
        if np.isscalar(t):
            D_t = D_values[0]
        else:
            D_t = D_values
    else:
        # Standard calculation for non-negative alpha
        D_t = D0 * (t**alpha) + D_offset

    return np.maximum(D_t, 1e-10)  # Ensure minimum value for physical validity


def calculate_shear_rate_numba(t, gamma0, beta, gamma_offset):
    """
    Calculate time-dependent shear rate.

    Parameters
    ----------
    t : float or array
        Time (can be scalar or array)
    gamma0 : float
        Reference shear rate
    beta : float
        Time-dependence exponent
    gamma_offset : float
        Baseline shear rate

    Returns
    -------
    float or array
        gamma_dot(t) = gamma0 * t^beta + gamma_offset

    Notes
    -----
    For negative beta, uses physical limit approach to avoid NaN at t=0:
    - At t=0: gamma(0) = gamma_offset (physical limit)
    - For t > threshold: gamma(t) = gamma0 * t^beta + gamma_offset
    """
    if beta < 0:
        # For negative beta, handle t=0 by taking the physical limit
        # Initialize with gamma_offset (the limit as t→0)
        gamma_values = np.full_like(np.atleast_1d(t), gamma_offset, dtype=np.float64)

        # For t > threshold, compute the full power-law + offset
        threshold = 1e-10
        mask = np.atleast_1d(t) > threshold
        if np.any(mask):
            gamma_values[mask] = (
                gamma0 * np.power(np.atleast_1d(t)[mask], beta) + gamma_offset
            )

        # Return scalar if input was scalar, otherwise return array
        if np.isscalar(t):
            gamma_t = gamma_values[0]
        else:
            gamma_t = gamma_values
    else:
        # Standard calculation for non-negative beta
        gamma_t = gamma0 * (t**beta) + gamma_offset

    return np.maximum(gamma_t, 0.0)  # Shear rate must be non-negative


def get_kernel_info() -> dict[str, Any]:
    """Get information about the current kernel configuration.

    Returns
    -------
    dict[str, Any]
        Information about kernel availability and configuration
    """
    current_numba_available = _check_numba_availability()

    info = {
        "numba_available": current_numba_available,
        "functions_compiled": current_numba_available,
        "kernel_functions": [
            "create_time_integral_matrix_numba",
            "calculate_diffusion_coefficient_numba",
            "calculate_shear_rate_numba",
            "compute_g1_correlation_numba",
            "compute_sinc_squared_numba",
        ],
    }

    # Add signature information if available
    if hasattr(create_time_integral_matrix_numba, "signatures"):
        info["function_signatures"] = {
            "create_time_integral_matrix_numba": len(
                create_time_integral_matrix_numba.signatures
            ),
            "calculate_diffusion_coefficient_numba": len(
                calculate_diffusion_coefficient_numba.signatures
            ),
            "calculate_shear_rate_numba": len(calculate_shear_rate_numba.signatures),
            "compute_g1_correlation_numba": len(
                compute_g1_correlation_numba.signatures
            ),
            "compute_sinc_squared_numba": len(compute_sinc_squared_numba.signatures),
        }

    return info
