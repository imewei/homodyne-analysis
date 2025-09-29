"""
Plotting and visualization modules for homodyne analysis.

This module provides backward compatibility for plotting modules moved from the root directory.
"""

# Import plotting functions when this module is imported
# This enables both new-style and old-style imports to work

# Visualization module imports with explicit fallback handling
try:
    from . import enhanced_plotting
    from . import plotting
except ImportError as e:
    import warnings

    warnings.warn(
        f"Could not import visualization modules: {e}", ImportWarning, stacklevel=2
    )
    # Create placeholder modules for compatibility
    import types
    enhanced_plotting = types.ModuleType('enhanced_plotting')
    plotting = types.ModuleType('plotting')

# Specific backward compatibility functions with explicit error handling
try:
    from .enhanced_plotting import EnhancedPlottingManager
    from .plotting import get_plot_config
    from .plotting import plot_c2_heatmaps
except ImportError as e:
    import warnings

    warnings.warn(
        f"Could not import visualization functions: {e}", ImportWarning, stacklevel=2
    )
    plot_c2_heatmaps = None
    get_plot_config = None
    EnhancedPlottingManager = None

__all__ = [
    "EnhancedPlottingManager",
    "enhanced_plotting",
    "get_plot_config",
    "plot_c2_heatmaps",
    "plotting",
]
