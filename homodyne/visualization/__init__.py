"""
Plotting and visualization modules for homodyne analysis.

This module provides backward compatibility for plotting modules moved from the root directory.
"""

# Import plotting functions when this module is imported
# This enables both new-style and old-style imports to work

# Visualization module imports
try:
    from . import enhanced_plotting, plotting
except ImportError:
    enhanced_plotting = None
    plotting = None

# Specific backward compatibility functions
try:
    from .enhanced_plotting import EnhancedPlottingManager
    from .plotting import get_plot_config, plot_c2_heatmaps
except ImportError:
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
