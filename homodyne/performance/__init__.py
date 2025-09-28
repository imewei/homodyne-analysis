"""
Performance monitoring and baseline modules for homodyne analysis.

This module provides backward compatibility for performance modules moved from the root directory.
"""

# Import performance functions when this module is imported
# This enables both new-style and old-style imports to work

# Performance module imports with explicit fallback handling
try:
    from . import baseline, monitoring
except ImportError as e:
    import warnings

    warnings.warn(
        f"Could not import performance modules: {e}", ImportWarning, stacklevel=2
    )
    baseline = None
    monitoring = None

# For specific backward compatibility with performance_monitoring
try:
    from .monitoring import PerformanceMonitor
except ImportError as e:
    import warnings

    warnings.warn(
        f"Could not import PerformanceMonitor: {e}", ImportWarning, stacklevel=2
    )
    PerformanceMonitor = None

__all__ = [
    "PerformanceMonitor",
    "baseline",
    "monitoring",
]
