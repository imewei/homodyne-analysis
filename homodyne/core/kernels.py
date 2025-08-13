"""
High-Performance Computational Kernels for Homodyne Scattering Analysis

This module provides Numba-accelerated computational kernels for the core
mathematical operations in homodyne scattering calculations.

Created for: Rheo-SAXS-XPCS Homodyne Analysis
Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory & University of Chicago
"""

import numpy as np
from functools import wraps

# Numba imports with fallbacks
try:
    from numba import jit, njit, prange, float64, int64

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Fallback decorators when Numba is unavailable
    def jit(*args, **kwargs):
        return args[0] if args and callable(args[0]) else lambda f: f

    def njit(*args, **kwargs):
        return args[0] if args and callable(args[0]) else lambda f: f

    prange = range

    class DummyType:
        def __getitem__(self, item):
            return self

        def __call__(self, *args, **kwargs):
            return self

    float64 = int64 = DummyType()


@njit(float64[:, :](float64[:]), parallel=True, cache=True)
def create_time_integral_matrix_numba(time_dependent_array):
    """
    Create time integral matrix for correlation calculations.

    Computes matrix of time integrals I[i,j] = |∫ᵗʲₜᵢ f(t)dt|
    using cumulative sum for O(n) complexity instead of O(n²).

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
    matrix = np.zeros((n, n), dtype=np.float64)

    # Compute cumulative sum for efficient integration
    cumsum = np.zeros(n, dtype=np.float64)
    cumsum[0] = time_dependent_array[0]
    for i in range(1, n):
        cumsum[i] = cumsum[i - 1] + time_dependent_array[i]

    # Fill integral matrix using parallel execution
    for i in prange(n):
        for j in range(n):
            matrix[i, j] = abs(cumsum[j] - cumsum[i])

    return matrix


@njit(float64[:](float64[:], float64, float64, float64), cache=True)
def calculate_diffusion_coefficient_numba(time_array, D0, alpha, D_offset):
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
        D(t) at each time point [Å²/s]
    """
    D_t = np.zeros_like(time_array)
    for i in range(len(time_array)):
        D_t[i] = D0 * (time_array[i] ** alpha) + D_offset
    return D_t


@njit(float64[:](float64[:], float64, float64, float64), cache=True)
def calculate_shear_rate_numba(
    time_array, gamma_dot_t0, beta, gamma_dot_t_offset
):
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
        γ̇(t) at each time point [s⁻¹]
    """
    gamma_dot_t = np.zeros_like(time_array)
    for i in range(len(time_array)):
        gamma_dot_t[i] = (
            gamma_dot_t0 * (time_array[i] ** beta) + gamma_dot_t_offset
        )
    return gamma_dot_t


@njit(float64[:, :](float64[:, :], float64), parallel=True, cache=True)
def compute_g1_correlation_numba(diffusion_integral_matrix, wavevector_factor):
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
    g1 = np.zeros(shape, dtype=np.float64)

    for i in prange(shape[0]):
        for j in range(shape[1]):
            exponent = -wavevector_factor * diffusion_integral_matrix[i, j]
            g1[i, j] = np.exp(exponent)

    return g1


@njit(float64[:, :](float64[:, :], float64), parallel=True, cache=True)
def compute_sinc_squared_numba(shear_integral_matrix, prefactor):
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
    In laminar flow between parallel plates with gap h:
        v(y) = γ̇ × y  (linear velocity profile)

    The scattering from particles at height y picks up phase φ = q⃗·v⃗(y)t
    Coherent averaging over y gives:
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

    Special cases:
    --------------
    - Zero shear (γ̇ = 0): sinc²(0) = 1 (no shear decorrelation)
    - Perpendicular scattering (cos φ = 0): sinc²(0) = 1 (no sensitivity)
    - Very small arguments: use Taylor expansion to avoid 0/0 indeterminacy
    - Very large arguments: sinc² oscillates rapidly around small values
    """
    shape = shear_integral_matrix.shape
    sinc_squared = np.zeros(shape, dtype=np.float64)

    for i in prange(shape[0]):
        for j in range(shape[1]):
            argument = prefactor * shear_integral_matrix[i, j]

            # Handle small arguments to avoid numerical instability
            if abs(argument) < 1e-10:
                sinc_squared[i, j] = 1.0
            else:
                pi_arg = np.pi * argument
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
        cache = {}
        access_count = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create hashable cache key
            key_parts = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    # Convert bytes to hex string to avoid joining bytes with strings
                    key_parts.append(arg.tobytes().hex())
                else:
                    key_parts.append(str(arg))

            for k, v in sorted(kwargs.items()):
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
                    return lambda *args, **kwargs: self._func(
                        instance, *args, **kwargs
                    )

        return CachedFunction(wrapper)

    return decorator
