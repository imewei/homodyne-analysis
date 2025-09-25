"""
Optimization Utilities
======================

Shared utilities for optimization tracking and monitoring across the homodyne
analysis package.

This module provides global optimization counters and tracking functionality
used by both classical and robust optimization methods, as well as common
numba detection utilities.
"""

# Numba availability detection
try:
    import numba
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