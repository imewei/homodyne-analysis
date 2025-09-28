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
try:
    import numba  # noqa: F401

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

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
            from .cpu_optimization import CPUOptimizer, get_cpu_optimization_info  # noqa: F401
            from ..performance.cpu_profiling import CPUProfiler, profile_homodyne_function  # noqa: F401
            CPU_OPTIMIZATION_AVAILABLE = True
        except ImportError:
            CPU_OPTIMIZATION_AVAILABLE = False
    return CPU_OPTIMIZATION_AVAILABLE


def get_optimization_capabilities() -> dict[str, bool]:
    """
    Get available optimization capabilities.

    Returns
    -------
    dict[str, bool]
        Available optimization features
    """
    cpu_opt_available = _check_cpu_optimization_availability()
    return {
        "numba_jit": NUMBA_AVAILABLE,
        "cpu_optimization": cpu_opt_available,
        "openmp_threading": NUMBA_AVAILABLE,  # Numba provides OpenMP support
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
