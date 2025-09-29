"""
Optimization Utilities
======================

Shared utilities for optimization tracking and monitoring across the homodyne
analysis package.

This module provides global optimization counters and tracking functionality
used by both classical and robust optimization methods, as well as common
numba detection utilities.
"""

from typing import Any


# Numba availability detection
def _check_numba_availability() -> bool:
    """Check if Numba is truly available and functional.

    This function handles edge cases like test environments where
    sys.modules['numba'] is set to None to disable numba.

    Returns
    -------
    bool
        True if Numba is available and functional, False otherwise
    """
    try:
        import sys

        # First check if numba is disabled in test environment
        # (common pattern: sys.modules['numba'] = None)
        if 'numba' in sys.modules and sys.modules['numba'] is None:
            return False

        # Try to import numba normally
        import numba

        # Check if the imported module is actually None (test environment case)
        if numba is None:
            return False

        # Check if numba has the required JIT functionality
        if not hasattr(numba, 'jit'):
            return False

        # Test that Numba actually works by trying to compile a simple function
        try:
            @numba.jit(nopython=True, cache=True, nogil=True)
            def _test_numba_function(x):
                return x + 1.0

            # Try to compile and call the function
            result = _test_numba_function(1.0)
            return abs(result - 2.0) < 1e-10

        except Exception:
            # If Numba compilation fails, treat as unavailable
            return False

    except (ImportError, AttributeError, ModuleNotFoundError):
        return False

# Initialize the global flag
NUMBA_AVAILABLE = _check_numba_availability()


def refresh_numba_availability() -> bool:
    """Refresh the global NUMBA_AVAILABLE flag by re-checking availability.

    This is useful in test environments where numba availability may change
    dynamically during test execution.

    Returns
    -------
    bool
        Updated numba availability status
    """
    global NUMBA_AVAILABLE
    NUMBA_AVAILABLE = _check_numba_availability()
    return NUMBA_AVAILABLE

# Global optimization counter for tracking iterations across all methods
OPTIMIZATION_COUNTER = 0


def reset_optimization_counter() -> None:
    """Reset the global optimization counter to zero."""
    global OPTIMIZATION_COUNTER
    OPTIMIZATION_COUNTER = 0


def get_optimization_counter() -> int:
    """Get the current optimization counter value.

    Returns
    -------
    int
        Current optimization counter value
    """
    return OPTIMIZATION_COUNTER


def increment_optimization_counter() -> int:
    """Increment the optimization counter and return the new value.

    Returns
    -------
    int
        New optimization counter value after incrementing
    """
    global OPTIMIZATION_COUNTER
    OPTIMIZATION_COUNTER += 1
    return OPTIMIZATION_COUNTER


# CPU Optimization Integration - Using late imports to avoid circular dependencies
# This avoids circular import issues while still providing optimization capabilities
CPU_OPTIMIZATION_AVAILABLE = None  # Will be determined on first access


def _check_cpu_optimization_availability() -> bool:
    """Check if CPU optimization modules are available (late import)."""
    global CPU_OPTIMIZATION_AVAILABLE
    if CPU_OPTIMIZATION_AVAILABLE is None:
        try:
            from ..performance.cpu_profiling import CPUProfiler  # noqa: F401
            from ..performance.cpu_profiling import profile_homodyne_function
            from .cpu_optimization import CPUOptimizer  # noqa: F401
            from .cpu_optimization import get_cpu_optimization_info
            CPU_OPTIMIZATION_AVAILABLE = True
        except ImportError:
            CPU_OPTIMIZATION_AVAILABLE = False
    return CPU_OPTIMIZATION_AVAILABLE


def get_optimization_capabilities() -> dict[str, bool]:
    """
    Get available optimization capabilities.

    This function dynamically checks the current state of optimization
    features, including re-checking Numba availability in case the
    environment changed (e.g., during tests).

    Returns
    -------
    dict[str, bool]
        Available optimization features
    """
    # Re-check numba availability dynamically for robustness
    current_numba_available = _check_numba_availability()
    cpu_opt_available = _check_cpu_optimization_availability()

    return {
        "numba_jit": current_numba_available,
        "cpu_optimization": cpu_opt_available,
        "openmp_threading": current_numba_available,  # Numba provides OpenMP support
        "vectorization": True,  # NumPy always available
        "multiprocessing": True,  # Built-in Python feature
        "cache_optimization": cpu_opt_available,
        "performance_profiling": cpu_opt_available,
    }


def create_optimized_configuration() -> dict[str, Any]:
    """
    Create optimized configuration based on available capabilities.

    Returns
    -------
    dict[str, Any]
        Optimized configuration for current system
    """
    capabilities = get_optimization_capabilities()
    config = {
        "optimization": {
            "use_numba": capabilities["numba_jit"],
            "use_cpu_optimization": capabilities["cpu_optimization"],
            "enable_profiling": capabilities["performance_profiling"],
        }
    }

    if _check_cpu_optimization_availability():
        try:
            from .cpu_optimization import get_cpu_optimization_info
            cpu_info = get_cpu_optimization_info()
            config["cpu_specific"] = {
                "max_threads": cpu_info.get("recommended_threads", 1),
                "cache_optimization": True,
                "simd_support": cpu_info.get("simd_support", {}),
            }
        except ImportError:
            pass  # CPU optimization not available

    return config


def get_performance_recommendations() -> list[str]:
    """
    Get performance optimization recommendations for current system.

    Returns
    -------
    list[str]
        Performance optimization recommendations
    """
    capabilities = get_optimization_capabilities()
    recommendations = []

    if not capabilities["numba_jit"]:
        recommendations.append(
            "Install Numba for 3-5x speedup with JIT compilation: pip install numba"
        )

    if not capabilities["cpu_optimization"]:
        recommendations.append(
            "CPU optimization utilities not available - check installation"
        )

    if capabilities["cpu_optimization"] and _check_cpu_optimization_availability():
        try:
            from .cpu_optimization import get_cpu_optimization_info
            cpu_info = get_cpu_optimization_info()
            if not any(cpu_info.get("simd_support", {}).values()):
                recommendations.append(
                    "Limited SIMD support detected - consider upgrading NumPy/SciPy"
                )

            if cpu_info.get("cpu_count", 1) > 4:
                recommendations.append(
                    f"System has {cpu_info['cpu_count']} CPUs - enable parallel processing"
                )
        except ImportError:
            pass  # CPU optimization not available

    if not recommendations:
        recommendations.append("System is optimally configured for CPU performance")

    return recommendations


# Late import utility functions to provide access to CPU optimization components
# These functions use lazy loading to avoid circular import issues

def get_cpu_optimizer():
    """Get CPUOptimizer class (late import to avoid circular dependency)."""
    if not _check_cpu_optimization_availability():
        return None
    try:
        from .cpu_optimization import CPUOptimizer
        return CPUOptimizer
    except ImportError:
        return None


def get_cpu_profiler():
    """Get CPUProfiler class (late import to avoid circular dependency)."""
    if not _check_cpu_optimization_availability():
        return None
    try:
        from ..performance.cpu_profiling import CPUProfiler
        return CPUProfiler
    except ImportError:
        return None


def get_profile_homodyne_function():
    """Get profile_homodyne_function decorator (late import to avoid circular dependency)."""
    if not _check_cpu_optimization_availability():
        return None
    try:
        from ..performance.cpu_profiling import profile_homodyne_function
        return profile_homodyne_function
    except ImportError:
        return None


def get_cpu_optimization_info_func():
    """Get get_cpu_optimization_info function (late import to avoid circular dependency)."""
    if not _check_cpu_optimization_availability():
        return None
    try:
        from .cpu_optimization import get_cpu_optimization_info
        return get_cpu_optimization_info
    except ImportError:
        return None
