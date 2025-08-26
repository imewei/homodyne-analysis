"""
High-Performance Computational Kernels for Homodyne Scattering Analysis

This module provides Numba-accelerated computational kernels for the core
mathematical operations in homodyne scattering calculations.

Created for: Rheo-SAXS-XPCS Homodyne Analysis
Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

from functools import wraps
from typing import Any, Callable
from typing import Tuple as TypingTuple
from typing import TypeVar, Union

import numpy as np

# Numba imports with fallbacks
try:
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

    def jit(*args: Any, **kwargs: Any) -> Union[F, Callable[[F], F]]:
        return args[0] if args and callable(args[0]) else lambda f: f

    def njit(*args: Any, **kwargs: Any) -> Union[F, Callable[[F], F]]:
        return args[0] if args and callable(args[0]) else lambda f: f

    prange = range

    class DummyType:
        def __getitem__(self, item: Any) -> "DummyType":
            return self

        def __call__(self, *args: Any, **kwargs: Any) -> "DummyType":
            return self

    float64 = int64 = types = Tuple = DummyType()


# Define fallback implementations for when numba is not available
def _compute_chi_squared_batch_fallback(
    theory_batch: np.ndarray,
    exp_batch: np.ndarray,
    contrast_batch: np.ndarray,
    offset_batch: np.ndarray,
) -> np.ndarray:
    """Pure numpy fallback for compute_chi_squared_batch_numba when numba is unavailable."""
    n_angles, n_data = theory_batch.shape
    chi2_batch = np.zeros(n_angles, dtype=np.float64)

    for i in range(n_angles):
        theory = theory_batch[i]
        exp = exp_batch[i]
        contrast = contrast_batch[i]
        offset = offset_batch[i]

        fitted_vals = theory * contrast + offset
        residuals = exp - fitted_vals
        chi2_batch[i] = np.sum(residuals**2)

    return chi2_batch


def _solve_least_squares_batch_fallback(
    theory_batch: np.ndarray, exp_batch: np.ndarray
) -> TypingTuple[np.ndarray, np.ndarray]:
    """Pure numpy fallback for solve_least_squares_batch_numba when numba is unavailable."""
    n_angles, n_data = theory_batch.shape
    contrast_batch = np.zeros(n_angles, dtype=np.float64)
    offset_batch = np.zeros(n_angles, dtype=np.float64)

    for i in range(n_angles):
        theory = theory_batch[i]
        exp = exp_batch[i]

        # Compute sums for 2x2 least squares system
        sum_theory_sq = np.sum(theory**2)
        sum_theory = np.sum(theory)
        sum_exp = np.sum(exp)
        sum_theory_exp = np.sum(theory * exp)

        # Solve 2x2 system: AtA * x = Atb
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


def _create_time_integral_matrix_impl(
    time_dependent_array: np.ndarray,
) -> np.ndarray:
    """
    Create time integral matrix for correlation calculations.

    Computes matrix of time integrals I[i,j] = |integral_{t_i}^{t_j} f(t)dt|
    using optimized algorithm that mimics numpy's vectorized approach.

    This implementation is equivalent to:
    cumsum_matrix = np.tile(cumsum, (n, 1))
    return np.abs(cumsum_matrix - cumsum_matrix.T)

    But optimized for Numba with parallel execution.

    Parameters
    ----------
    time_dependent_array : np.ndarray
        Time-dependent values f(t) at discrete points

    Returns
    -------
    np.ndarray
        Matrix where element [i,j] = integral from time i to j
    """
    n = len(time_dependent_array)
    matrix = np.empty((n, n), dtype=np.float64)

    # Compute cumulative sum once - O(n) operation
    cumsum = np.cumsum(time_dependent_array)

    # Use parallel loops to fill the matrix efficiently
    # This approach avoids the complex indexing and simply computes the
    # difference
    for i in range(n):
        cumsum_i = cumsum[i]
        for j in range(n):
            cumsum_j = cumsum[j]
            # Absolute difference of cumulative sums
            matrix[i, j] = abs(cumsum_i - cumsum_j)

    return matrix


def _calculate_diffusion_coefficient_impl(time_array, D0, alpha, D_offset):
    """
    Calculate time-dependent diffusion coefficient.

    Implements power-law model: D(t) = D₀(t/t₀)^α + D_offset

    This model captures various physical scenarios:
    1. Aging systems: D decreases with time (α < 0) as structure develops
    2. Swelling/dissolution: D increases with time (α > 0) as structure breaks down
    3. Stress relaxation: D changes as internal stresses redistribute
    4. Temperature effects: Time-dependent heating/cooling during measurement

    Parameters
    ----------
    time_array : np.ndarray
        Time points [seconds]
    D0 : float
        Reference diffusion coefficient [Å²/s]
    alpha : float
        Power-law exponent [dimensionless]
        α = 0: constant diffusion (equilibrium systems)
        α > 0: enhanced diffusion (aging, structural relaxation)
        α < 0: subdiffusion (crowded environments, viscoelastic media)
    D_offset : float
        Baseline diffusion [Å²/s]
        Accounts for residual thermal motion or measurement artifacts

    Returns
    -------
    np.ndarray
        D(t) at each time point [Å²/s], guaranteed to be > 0
    """
    D_t = np.empty_like(time_array)
    # Vectorized computation with positivity constraint
    for i in range(len(time_array)):
        D_value = D0 * (time_array[i] ** alpha) + D_offset
        # Ensure D(t) > 0 always using conditional (Numba-compatible)
        if D_value > 1e-10:
            D_t[i] = D_value
        else:
            D_t[i] = 1e-10
    return D_t


def _calculate_shear_rate_impl(time_array, gamma_dot_t0, beta, gamma_dot_t_offset):
    """
    Calculate time-dependent shear rate.

    Implements power-law model: γ̇(t) = γ̇₀(t/t₀)^β + γ̇_offset

    Common experimental scenarios:

    1. Creep tests: Applied constant stress → time-dependent strain rate

       - Initially: γ̇(t) ~ t^(-n) as material yields
       - Later: γ̇(t) ~ constant for steady-state flow
       - Recovery: γ̇(t) → 0 with power-law or exponential decay

    2. Startup flows: Sudden application of shear

       - Acceleration phase: γ̇(t) increases toward steady state
       - Overshoot possible in viscoelastic materials
       - Eventually: γ̇(t) → constant

    3. Oscillatory flows: Time-varying shear rates

       - Can be captured by appropriate choice of β
       - More complex forms may require Fourier series

    4. Stress relaxation: Removal of applied stress

       - Exponential or power-law decay: γ̇(t) ~ t^(-β)
       - Long-time tail behavior depends on material properties

    Parameters
    ----------
    time_array : np.ndarray
        Time points [seconds]
    gamma_dot_t0 : float
        Reference shear rate [s⁻¹]
    beta : float
        Power-law exponent [dimensionless]
        β = 0: constant shear rate (steady flow)
        β > 0: accelerating flow (creep acceleration, startup transients)
        β < 0: decelerating flow (stress relaxation, recovery)
    gamma_dot_t_offset : float
        Baseline shear rate [s⁻¹]

    Returns
    -------
    np.ndarray
        γ̇(t) at each time point [s⁻¹], guaranteed to be > 0
    """
    gamma_dot_t = np.empty_like(time_array)
    # Vectorized computation with positivity constraint
    for i in range(len(time_array)):
        gamma_value = gamma_dot_t0 * (time_array[i] ** beta) + gamma_dot_t_offset
        # Ensure γ̇(t) > 0 always using conditional (Numba-compatible)
        if gamma_value > 1e-10:
            gamma_dot_t[i] = gamma_value
        else:
            gamma_dot_t[i] = 1e-10
    return gamma_dot_t


def _compute_g1_correlation_impl(diffusion_integral_matrix, wavevector_factor):
    """
    Compute field correlation function g₁ from diffusion.

    Calculates the field correlation function arising from translational diffusion
    of scattering particles in the sample. This implements the core diffusion term
    from the homodyne scattering equations:

        g₁(q⃗, t₁, t₂) = exp(-q² ∫ᵗ²ᵗ¹ D(t)dt)

    Parameters
    ----------
    diffusion_integral_matrix : np.ndarray
        Matrix of diffusion integrals [Å²]
    wavevector_factor : float
        Pre-computed 0.5×q²×Δt [Å⁻²]

    Returns
    -------
    np.ndarray
        Field correlation g₁(t₁,t₂), values in [0,1]
    """
    shape = diffusion_integral_matrix.shape
    g1 = np.empty(shape, dtype=np.float64)

    # Optimized computation with fastmath and cache-friendly access
    for i in range(shape[0]):
        for j in range(shape[1]):
            exponent = -wavevector_factor * diffusion_integral_matrix[i, j]
            g1[i, j] = np.exp(exponent)

    return g1


def _compute_sinc_squared_impl(shear_integral_matrix, prefactor):
    """
    Compute sinc² function for shear flow contributions (Numba-optimized).

    Calculates the shear-induced decorrelation term appearing in homodyne
    scattering under flow conditions. This implements the sinc² term from
    the theoretical equations:

        sinc²(x) = [sin(πx)/(πx)]²

    where x = (1/2π) × q⃗·v̄ × ∫γ̇(t)dt

    Physical interpretation:

    - The sinc function arises from the Fourier transform of a rectangular
      velocity profile in laminar shear flow
    - Shear flow creates a systematic velocity gradient perpendicular to the flow
    - Particles at different heights have different velocities
    - This creates a phase modulation of the scattered light
    - The sinc² envelope reflects the coherent averaging over the velocity distribution

    Mathematical origin:

    In laminar flow between parallel plates with gap h::

        v(y) = γ̇ × y  (linear velocity profile)

    The scattering from particles at height y picks up phase φ = q⃗·v⃗(y)t
    Coherent averaging over y gives::

        ⟨exp(iq⃗·v⃗(y)t)⟩ = ∫₀ʰ exp(iqy̲γ̇t)dy/h = sinc(qy̲γ̇th/2)

    Where qy̲ is the component of q⃗ perpendicular to flow direction.

    Flow geometry considerations:

    - For q⃗ parallel to flow: cos(φ) = ±1, maximum shear effect
    - For q⃗ perpendicular to flow: cos(φ) = 0, no shear contribution
    - Intermediate angles: cos(φ) determines the effective shear sensitivity

    Limiting behaviors:

    - Small argument (weak shear): sinc²(x) ≈ 1 - (πx)²/3
    - Large argument (strong shear): sinc²(x) ≈ 0 with oscillations
    - First zero at x = 1: complete destructive interference

    Parameters
    ----------
    shear_integral_matrix : np.ndarray
        Matrix of shear rate integrals (strain)
    prefactor : float
        Geometric factor (1/2π)×q×h×cos(φ)×Δt

    Returns
    -------
    np.ndarray
        Sinc² values for shear contribution, in [0,1] with sinc²(0) = 1

    Special cases
    -------------
    - Zero shear (γ̇ = 0): sinc²(0) = 1 (no shear decorrelation)
    - Perpendicular scattering (cos φ = 0): sinc²(0) = 1 (no sensitivity)
    - Very small arguments: use Taylor expansion to avoid 0/0 indeterminacy
    - Very large arguments: sinc² oscillates rapidly around small values
    """
    shape = shear_integral_matrix.shape
    sinc_squared = np.empty(shape, dtype=np.float64)

    # Pre-compute pi for efficiency
    pi = np.pi

    for i in range(shape[0]):
        for j in range(shape[1]):
            argument = prefactor * shear_integral_matrix[i, j]

            # Optimized small argument handling with Taylor expansion
            if abs(argument) < 1e-10:  # Use tighter threshold to catch more edge cases
                # Taylor expansion: sinc²(x) ≈ 1 - (πx)²/3 + O(x⁴)
                pi_arg_sq = (pi * argument) ** 2
                sinc_squared[i, j] = 1.0 - pi_arg_sq / 3.0
            else:
                pi_arg = pi * argument
                # Additional safety check to prevent division by exactly zero
                if abs(pi_arg) < 1e-15:
                    sinc_squared[i, j] = 1.0  # sinc(0) = 1, so sinc²(0) = 1
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
if NUMBA_AVAILABLE:
    solve_least_squares_batch_numba = njit(
        cache=True,
        fastmath=True,
        nogil=True,
    )(_solve_least_squares_batch_numba_impl)
else:
    solve_least_squares_batch_numba = _solve_least_squares_batch_fallback
    # Add signatures attribute for compatibility with numba compiled functions
    setattr(solve_least_squares_batch_numba, "signatures", [])


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
    setattr(compute_chi_squared_batch_numba, "signatures", [])


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
    setattr(create_time_integral_matrix_numba, "signatures", [])
    setattr(calculate_diffusion_coefficient_numba, "signatures", [])
    setattr(calculate_shear_rate_numba, "signatures", [])
    setattr(compute_g1_correlation_numba, "signatures", [])
    setattr(compute_sinc_squared_numba, "signatures", [])
