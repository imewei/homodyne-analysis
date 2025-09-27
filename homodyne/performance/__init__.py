"""
Performance monitoring and baseline modules for homodyne analysis.

This module provides backward compatibility for performance modules moved from the root directory.
"""

# Import performance functions when this module is imported
# This enables both new-style and old-style imports to work

# Performance module imports
try:
    from . import baseline, monitoring
except ImportError:
    baseline = None
    monitoring = None

# For specific backward compatibility with performance_monitoring
try:
    from .monitoring import PerformanceMonitor
except ImportError:
    PerformanceMonitor = None

__all__ = [
    "PerformanceMonitor",
    "baseline",
    "monitoring",
]
