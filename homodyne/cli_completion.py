"""
Shell Completion for Homodyne Analysis with Conda Environment Integration
=========================================================================
Ultra-fast shell completion with conda environment auto-loading and persistent
caching for instant response times. Supports both CPU-only homodyne and
GPU-accelerated homodyne-gpu commands with automatic conda integration.

Commands supported:
- homodyne: CPU analysis (all platforms)
- homodyne-gpu: GPU analysis with system CUDA (Linux only)
- homodyne-config: Configuration file generator

Conda Environment Integration:
    Shell completion is AUTOMATICALLY installed when you install homodyne-analysis
    in a conda/virtual environment on Linux systems.

    Available shortcuts after conda environment activation:
    hm --help                            # homodyne --method mcmc
    hga --config config.json             # homodyne-gpu --method all
    homodyne_help                        # Show all available shortcuts

    Shell completion in regular commands:
    homodyne --method <TAB>         # Shows: classical, mcmc, robust, all
    homodyne-gpu --method <TAB>     # Shows: mcmc, all (GPU-optimized)
    homodyne --config <TAB>         # Shows available .json files
    homodyne-config --mode <TAB>    # Shows: static_isotropic, etc.
"""

import argparse
from typing import Any

try:
    import argcomplete

    ARGCOMPLETE_AVAILABLE = True
except ImportError:
    ARGCOMPLETE_AVAILABLE = False
    argcomplete: Any | None = None

# Import completion functions from the dedicated fast completion module
try:
    from .completion_fast import (
        METHODS,
        MODES,
        complete_config,
        complete_method,
        complete_mode,
        complete_output_dir,
    )

    FAST_COMPLETION_AVAILABLE = True
except ImportError:
    FAST_COMPLETION_AVAILABLE = False
    # Fallback static data
    METHODS = ["classical", "mcmc", "robust", "all"]
    MODES = ["static_isotropic", "static_anisotropic", "laminar_flow"]


class HomodyneCompleter:
    """Ultra-fast shell completion using the dedicated completion_fast module."""

    @staticmethod
    def method_completer(prefix: str, parsed_args: Any, **kwargs: Any) -> list[str]:
        """Suggest method names - instant static lookup."""
        if FAST_COMPLETION_AVAILABLE:
            return complete_method(prefix)
        # Fallback for when completion_fast is not available
        methods = METHODS
        if not prefix:
            return methods
        prefix_lower = prefix.lower()
        return [m for m in methods if m.startswith(prefix_lower)]

    @staticmethod
    def config_files_completer(
        prefix: str, parsed_args: Any, **kwargs: Any
    ) -> list[str]:
        """Suggest JSON config files - instant cached lookup."""
        if FAST_COMPLETION_AVAILABLE:
            return complete_config(prefix)
        # Fallback - basic file completion
        return []

    @staticmethod
    def output_dir_completer(prefix: str, parsed_args: Any, **kwargs: Any) -> list[str]:
        """Suggest directories - instant cached lookup."""
        if FAST_COMPLETION_AVAILABLE:
            return complete_output_dir(prefix)
        # Fallback - basic directory completion
        return []

    @staticmethod
    def analysis_mode_completer(prefix: str, parsed_args, **kwargs) -> list[str]:
        """Suggest analysis modes - instant static lookup."""
        if FAST_COMPLETION_AVAILABLE:
            return complete_mode(prefix)
        # Fallback for when completion_fast is not available
        modes = MODES
        if not prefix:
            return modes
        prefix_lower = prefix.lower()
        return [m for m in modes if m.startswith(prefix_lower)]

    @staticmethod
    def clear_cache():
        """Clear cache for testing."""
        # Import here to avoid circular imports during testing
        if FAST_COMPLETION_AVAILABLE:
            from .completion_fast import _cache

            _cache._data = {}
            _cache._scan_current_dir()


def setup_shell_completion(parser: argparse.ArgumentParser) -> None:
    """Add shell completion support to argument parser."""
    if not ARGCOMPLETE_AVAILABLE or argcomplete is None:
        return

    # Create completion methods for specific argument types
    def _create_completer(completion_type: str):
        """Create a completer for the specified completion type."""

        def completer(prefix, parsed_args, **kwargs):
            # Use built-in completion methods directly
            if completion_type == "method":
                return HomodyneCompleter.method_completer(prefix, parsed_args, **kwargs)
            elif completion_type == "config":
                return HomodyneCompleter.config_files_completer(
                    prefix, parsed_args, **kwargs
                )
            elif completion_type == "output_dir":
                return HomodyneCompleter.output_dir_completer(
                    prefix, parsed_args, **kwargs
                )
            elif completion_type == "mode":
                return HomodyneCompleter.analysis_mode_completer(
                    prefix, parsed_args, **kwargs
                )
            return []

        return completer

    # Add completers to specific arguments
    for action in parser._actions:
        if action.dest == "method":
            action.completer = _create_completer("method")
        elif action.dest == "config":
            action.completer = _create_completer("config")
        elif action.dest == "output_dir":
            action.completer = _create_completer("output_dir")
        elif action.dest == "mode":
            action.completer = _create_completer("mode")
        elif action.dest == "output":
            action.completer = _create_completer("config")
    # Enable argcomplete with error handling for zsh compdef issues
    try:
        argcomplete.autocomplete(parser)
    except SystemExit:
        # Re-raise SystemExit - this is how argcomplete exits after completion
        raise
    except (ImportError, AttributeError):
        # Only catch import/attribute errors, not completion functionality
        import os

        # Not in completion mode or argcomplete unavailable, continue silently
        if os.environ.get("_ARGCOMPLETE") != "1":
            pass
        else:
            # In completion mode but argcomplete failed - re-raise
            raise


# Export public functions
__all__ = [
    "HomodyneCompleter",
    "setup_shell_completion",
]
