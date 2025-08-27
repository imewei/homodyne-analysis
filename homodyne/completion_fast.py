#!/usr/bin/env python3
"""
Ultra-lightweight shell completion for homodyne - ZERO heavy imports!

This module is specifically designed for shell completion performance.
It imports NO scientific libraries and has minimal dependencies.

Usage in shell completion:
    python -m homodyne.completion_fast method c
    python -m homodyne.completion_fast config my
    python -m homodyne.completion_fast output_dir out
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List


class FastCache:
    """Ultra-lightweight cache with zero heavy imports."""

    def __init__(self):
        self.cache_file = Path.home() / ".cache" / "homodyne" / "completion_cache.json"
        self.cache_ttl = 5.0
        self._data = {}
        self._load_cache()

    def _load_cache(self):
        """Load cache or create minimal fallback."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file) as f:
                    cache_data = json.load(f)

                if time.time() - cache_data.get("timestamp", 0) < self.cache_ttl:
                    self._data = cache_data
                    return
        except:
            pass

        # Minimal fallback - scan current directory only
        self._scan_current_dir()

    def _scan_current_dir(self):
        """Quick scan of current directory - no recursion."""
        try:
            cwd = Path.cwd()
            files = [
                f.name for f in cwd.iterdir() if f.is_file() and f.suffix == ".json"
            ]
            dirs = [d.name for d in cwd.iterdir() if d.is_dir()]

            self._data = {
                "timestamp": time.time(),
                "files": {".": files[:20]},  # Limit for speed
                "dirs": {".": dirs[:20]},
            }
        except:
            self._data = {
                "timestamp": time.time(),
                "files": {".": []},
                "dirs": {".": []},
            }

    def get_files(self, directory="."):
        """Get cached JSON files."""
        files = self._data.get("files", {}).get(directory, [])

        # Prioritize common configs
        common = ["config.json", "homodyne_config.json", "my_config.json"]
        priority = [f for f in common if f in files]
        others = [f for f in files if f not in common]

        return priority + others[:12]  # Limit results

    def get_dirs(self, directory="."):
        """Get cached directories."""
        dirs = self._data.get("dirs", {}).get(directory, [])

        # Prioritize common output dirs
        common = ["output", "results", "data", "plots"]
        priority = [d for d in common if d in dirs]
        others = [d for d in dirs if d not in common]

        return priority + others[:8]  # Limit results


# Global cache instance
_cache = FastCache()

# Static completions - zero file system operations
METHODS = ["classical", "mcmc", "robust", "all"]
MODES = ["static_isotropic", "static_anisotropic", "laminar_flow"]


def complete_method(prefix: str) -> List[str]:
    """Complete method names - instant."""
    if not prefix:
        return METHODS

    prefix_lower = prefix.lower()
    return [m for m in METHODS if m.startswith(prefix_lower)]


def complete_mode(prefix: str) -> List[str]:
    """Complete analysis modes - instant."""
    if not prefix:
        return MODES

    prefix_lower = prefix.lower()
    return [m for m in MODES if m.startswith(prefix_lower)]


def complete_config(prefix: str) -> List[str]:
    """Complete config files - cached lookup."""
    # Handle directory prefix
    if "/" in prefix:
        dir_path, file_prefix = os.path.split(prefix)
        if not dir_path:
            dir_path = "."
    else:
        dir_path = "."
        file_prefix = prefix

    # Get cached files
    files = _cache.get_files(dir_path)

    if not file_prefix:
        # Return all files (already prioritized)
        if dir_path == ".":
            return files
        else:
            return [os.path.join(dir_path, f) for f in files]

    # Filter by prefix
    file_prefix_lower = file_prefix.lower()
    matches = []
    for f in files:
        if f.lower().startswith(file_prefix_lower):
            if dir_path == ".":
                matches.append(f)
            else:
                matches.append(os.path.join(dir_path, f))

    return matches


def complete_output_dir(prefix: str) -> List[str]:
    """Complete output directories - cached lookup."""
    # Handle directory prefix
    if "/" in prefix:
        parent_dir, dir_prefix = os.path.split(prefix)
        if not parent_dir:
            parent_dir = "."
    else:
        parent_dir = "."
        dir_prefix = prefix

    # Get cached directories
    dirs = _cache.get_dirs(parent_dir)

    if not dir_prefix:
        # Return all dirs with trailing slash
        results = []
        for d in dirs:
            if parent_dir == ".":
                results.append(d + "/")
            else:
                results.append(os.path.join(parent_dir, d) + "/")
        return results

    # Filter by prefix
    dir_prefix_lower = dir_prefix.lower()
    matches = []
    for d in dirs:
        if d.lower().startswith(dir_prefix_lower):
            if parent_dir == ".":
                matches.append(d + "/")
            else:
                matches.append(os.path.join(parent_dir, d) + "/")

    return matches


def main():
    """Main completion function - called by shell."""
    if len(sys.argv) < 2:
        return

    completion_type = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else ""

    # Route to appropriate completer
    if completion_type == "method":
        results = complete_method(prefix)
    elif completion_type == "mode":
        results = complete_mode(prefix)
    elif completion_type == "config":
        results = complete_config(prefix)
    elif completion_type == "output_dir":
        results = complete_output_dir(prefix)
    else:
        results = []

    # Output results for shell
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
