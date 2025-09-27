"""
Command-line interface and runner modules for homodyne analysis.

This module provides backward compatibility for CLI tools moved from the root directory.
"""

# Import main CLI functions when this module is imported
# This enables both new-style and old-style imports to work

try:
    from .create_config import main as create_config_main
    from .enhanced_runner import main as enhanced_runner_main
    from .run_homodyne import main as run_homodyne_main
except ImportError:
    # Graceful degradation if files haven't been moved yet
    run_homodyne_main = None
    create_config_main = None
    enhanced_runner_main = None

__all__ = [
    "create_config_main",
    "enhanced_runner_main",
    "run_homodyne_main",
]
