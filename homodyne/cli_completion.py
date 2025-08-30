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
import json
import os
import time
from pathlib import Path
from typing import Any

try:
    import argcomplete

    ARGCOMPLETE_AVAILABLE = True
except ImportError:
    ARGCOMPLETE_AVAILABLE = False
    argcomplete: Any | None = None


class FastCompletionCache:
    """Ultra-fast completion cache with persistent storage and instant lookups."""

    def __init__(self):
        # Pre-computed static completions (never change)
        self.METHODS = ["classical", "mcmc", "robust", "all"]
        self.MODES = ["static_isotropic", "static_anisotropic", "laminar_flow"]
        self.COMMON_CONFIGS = ["config.json", "homodyne_config.json", "my_config.json"]
        self.COMMON_DIRS = ["output", "results", "data", "plots", "analysis"]
        # Cache file path
        self.cache_dir = Path.home() / ".cache" / "homodyne"
        self.cache_file = self.cache_dir / "completion_cache.json"
        self.cache_ttl = 5.0  # Cache valid for 5 seconds
        # In-memory cache
        self._file_cache: dict[str, list[str]] = {}
        self._dir_cache: dict[str, list[str]] = {}
        self._last_update = 0.0
        # Initialize cache
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk if valid, otherwise create new cache."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file) as f:
                    data = json.load(f)
                # Check if cache is still valid
                cache_time = data.get("timestamp", 0)
                if time.time() - cache_time < self.cache_ttl:
                    self._file_cache = data.get("files", {})
                    self._dir_cache = data.get("dirs", {})
                    self._last_update = cache_time
                    return
        except (json.JSONDecodeError, OSError):
            pass
        # Cache is invalid or doesn't exist - create new one
        self._update_cache()

    def _update_cache(self) -> None:
        """Update cache with current directory contents."""
        current_time = time.time()
        # Only update if cache is old enough
        if current_time - self._last_update < 1.0:
            return
        # Scan current directory for files and dirs
        self._file_cache = {}
        self._dir_cache = {}
        try:
            cwd = Path.cwd()
            # Get JSON files in current directory
            json_files = [
                f.name for f in cwd.iterdir() if f.is_file() and f.suffix == ".json"
            ]
            self._file_cache["."] = json_files
            # Get directories in current directory
            dirs = [d.name for d in cwd.iterdir() if d.is_dir()]
            self._dir_cache["."] = dirs
            # Also cache common subdirectories if they exist
            for subdir in ["config", "configs", "data", "output", "results"]:
                sub_path = cwd / subdir
                if sub_path.exists() and sub_path.is_dir():
                    try:
                        sub_files = [
                            f.name
                            for f in sub_path.iterdir()
                            if f.is_file() and f.suffix == ".json"
                        ]
                        if sub_files:
                            self._file_cache[subdir] = sub_files
                        sub_dirs = [d.name for d in sub_path.iterdir() if d.is_dir()]
                        if sub_dirs:
                            self._dir_cache[subdir] = sub_dirs
                    except (OSError, PermissionError):
                        pass
        except (OSError, PermissionError):
            # Fallback to empty cache
            self._file_cache["."] = []
            self._dir_cache["."] = []
        self._last_update = current_time
        self._save_cache()

    def _save_cache(self) -> None:
        """Save cache to disk for next startup."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_data = {
                "timestamp": self._last_update,
                "files": self._file_cache,
                "dirs": self._dir_cache,
            }
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
        except (OSError, PermissionError):
            pass  # Fail silently if we can't write cache

    def get_json_files(self, directory: str = ".") -> list[str]:
        """Get cached JSON files for directory."""
        # Update cache if needed
        if time.time() - self._last_update > self.cache_ttl:
            self._update_cache()
        files = self._file_cache.get(directory, [])
        # For current directory, prioritize common config files
        if directory == "." and files:
            common_found = [f for f in self.COMMON_CONFIGS if f in files]
            other_files = [f for f in files if f not in self.COMMON_CONFIGS]
            return common_found + sorted(other_files)[:12]  # Limit for speed
        return sorted(files)[:15]  # Limit for speed

    def get_directories(self, directory: str = ".") -> list[str]:
        """Get cached directories for directory."""
        # Update cache if needed
        if time.time() - self._last_update > self.cache_ttl:
            self._update_cache()
        dirs = self._dir_cache.get(directory, [])
        # For current directory, prioritize common output directories
        if directory == "." and dirs:
            common_found = [d for d in self.COMMON_DIRS if d in dirs]
            other_dirs = [d for d in dirs if d not in self.COMMON_DIRS]
            return common_found + sorted(other_dirs)[:8]  # Limit for speed
        return sorted(dirs)[:12]  # Limit for speed


# Global cache instance
_completion_cache = FastCompletionCache()


class HomodyneCompleter:
    """Ultra-fast shell completion using pre-cached data."""

    @staticmethod
    def method_completer(prefix: str, parsed_args: Any, **kwargs: Any) -> list[str]:
        """Suggest method names - instant static lookup."""
        methods = _completion_cache.METHODS
        if not prefix:
            return methods
        prefix_lower = prefix.lower()
        return [m for m in methods if m.startswith(prefix_lower)]

    @staticmethod
    def config_files_completer(
        prefix: str, parsed_args: Any, **kwargs: Any
    ) -> list[str]:
        """Suggest JSON config files - instant cached lookup."""
        # Handle path with directory (cross-platform)
        if os.sep in prefix or "/" in prefix:
            dir_path, file_prefix = os.path.split(prefix)
            if not dir_path:
                dir_path = "."
        else:
            dir_path = "."
            file_prefix = prefix
        # Get cached files (instant lookup)
        json_files = _completion_cache.get_json_files(dir_path)
        if not file_prefix:
            # No prefix - return prioritized list
            if dir_path == ".":
                return json_files  # Already prioritized in get_json_files
            else:
                return [os.path.join(dir_path, f) for f in json_files]
        # Filter by prefix (case-insensitive)
        file_prefix_lower = file_prefix.lower()
        matches = []
        for f in json_files:
            if f.lower().startswith(file_prefix_lower):
                if dir_path == ".":
                    matches.append(f)
                else:
                    matches.append(os.path.join(dir_path, f))
        return matches

    @staticmethod
    def output_dir_completer(prefix: str, parsed_args: Any, **kwargs: Any) -> list[str]:
        """Suggest directories - instant cached lookup."""
        # Handle path with directory (cross-platform)
        if os.sep in prefix or "/" in prefix:
            parent_dir, dir_prefix = os.path.split(prefix)
            if not parent_dir:
                parent_dir = "."
        else:
            parent_dir = "."
            dir_prefix = prefix
        # Get cached directories (instant lookup)
        dirs = _completion_cache.get_directories(parent_dir)
        if not dir_prefix:
            # No prefix - return prioritized list with trailing slash
            results = []
            for d in dirs:
                if parent_dir == ".":
                    results.append(d + os.sep)
                else:
                    results.append(os.path.join(parent_dir, d) + os.sep)
            return results
        # Filter by prefix (case-insensitive)
        dir_prefix_lower = dir_prefix.lower()
        matches = []
        for d in dirs:
            if d.lower().startswith(dir_prefix_lower):
                if parent_dir == ".":
                    matches.append(d + os.sep)
                else:
                    matches.append(os.path.join(parent_dir, d) + os.sep)
        return matches

    @staticmethod
    def analysis_mode_completer(prefix: str, parsed_args, **kwargs) -> list[str]:
        """Suggest analysis modes - instant static lookup."""
        modes = _completion_cache.MODES
        if not prefix:
            return modes
        prefix_lower = prefix.lower()
        return [m for m in modes if m.startswith(prefix_lower)]

    @staticmethod
    def clear_cache():
        """Clear cache for testing."""
        global _completion_cache
        _completion_cache = FastCompletionCache()
        # Force immediate cache update for testing
        _completion_cache._last_update = 0.0
        _completion_cache._update_cache()


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
    except Exception:
        # Fallback for zsh compdef issues - use simplified completion
        import os

        # Not in completion mode, just continue silently
        pass








# Export public functions
__all__ = [
    "HomodyneCompleter",
    "setup_shell_completion",
]
