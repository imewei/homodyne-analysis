"""
Shell Completion for Homodyne Analysis
=======================================

Ultra-fast shell completion with persistent caching for instant response times.

Usage:
    # Enable shell completion (one-time setup)
    homodyne --install-completion bash
    homodyne --install-completion zsh
    homodyne --install-completion fish

    # Shell completion in regular commands
    homodyne --method <TAB>     # Shows: classical, mcmc, robust, all
    homodyne --config <TAB>     # Shows available .json files
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    import argcomplete

    ARGCOMPLETE_AVAILABLE = True
except ImportError:
    ARGCOMPLETE_AVAILABLE = False
    argcomplete = None


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
        self._file_cache: Dict[str, List[str]] = {}
        self._dir_cache: Dict[str, List[str]] = {}
        self._last_update = 0.0

        # Initialize cache
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk if valid, otherwise create new cache."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, "r") as f:
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

    def get_json_files(self, directory: str = ".") -> List[str]:
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

    def get_directories(self, directory: str = ".") -> List[str]:
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
    def method_completer(prefix: str, parsed_args, **kwargs) -> List[str]:
        """Suggest method names - instant static lookup."""
        methods = _completion_cache.METHODS
        if not prefix:
            return methods

        prefix_lower = prefix.lower()
        return [m for m in methods if m.startswith(prefix_lower)]

    @staticmethod
    def config_files_completer(prefix: str, parsed_args, **kwargs) -> List[str]:
        """Suggest JSON config files - instant cached lookup."""
        # Handle path with directory
        if "/" in prefix:
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
    def output_dir_completer(prefix: str, parsed_args, **kwargs) -> List[str]:
        """Suggest directories - instant cached lookup."""
        # Handle path with directory
        if "/" in prefix:
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
                    results.append(d + "/")
                else:
                    results.append(os.path.join(parent_dir, d) + "/")
            return results

        # Filter by prefix (case-insensitive)
        dir_prefix_lower = dir_prefix.lower()
        matches = []
        for d in dirs:
            if d.lower().startswith(dir_prefix_lower):
                if parent_dir == ".":
                    matches.append(d + "/")
                else:
                    matches.append(os.path.join(parent_dir, d) + "/")

        return matches

    @staticmethod
    def analysis_mode_completer(prefix: str, parsed_args, **kwargs) -> List[str]:
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


def setup_shell_completion(parser: argparse.ArgumentParser) -> None:
    """Add shell completion support to argument parser."""
    if not ARGCOMPLETE_AVAILABLE or argcomplete is None:
        return

    # Use fast external completion script to avoid heavy imports
    def _fast_completer(completion_type: str):
        """Create a fast completer using external script."""

        def completer(prefix, parsed_args, **kwargs):
            import os
            import subprocess
            from pathlib import Path

            try:
                # Find the fast completion script
                script_path = Path(__file__).parent.parent / "homodyne_complete"
                if not script_path.exists():
                    # Fallback to slow method if script not found
                    if completion_type == "method":
                        return HomodyneCompleter.method_completer(
                            prefix, parsed_args, **kwargs
                        )
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

                # Call fast completion script
                result = subprocess.run(
                    [str(script_path), completion_type, prefix or ""],
                    capture_output=True,
                    text=True,
                    timeout=1.0,
                )

                if result.returncode == 0:
                    return [
                        line.strip()
                        for line in result.stdout.strip().split("\n")
                        if line.strip()
                    ]

            except:
                pass

            # Fallback to built-in completion
            return []

        return completer

    # Add fast completers to specific arguments
    for action in parser._actions:
        if action.dest == "method":
            setattr(action, "completer", _fast_completer("method"))
        elif action.dest == "config":
            setattr(action, "completer", _fast_completer("config"))
        elif action.dest == "output_dir":
            setattr(action, "completer", _fast_completer("output_dir"))
        elif action.dest == "mode":
            setattr(action, "completer", _fast_completer("mode"))

    # Enable argcomplete with error handling for zsh compdef issues
    try:
        argcomplete.autocomplete(parser)
    except Exception as e:
        # Fallback for zsh compdef issues - use simplified completion
        import os

        if "ZSH_VERSION" in os.environ or os.environ.get("_ARGCOMPLETE") == "1":
            # Still in completion mode, try to provide basic completions
            _handle_zsh_fallback_completion()
        else:
            # Not in completion mode, just continue silently
            pass


def _handle_zsh_fallback_completion():
    """Handle completion when argcomplete fails in zsh."""
    import os
    import sys

    # Get completion context
    comp_line = os.environ.get("COMP_LINE", "")
    comp_point = int(os.environ.get("COMP_POINT", len(comp_line)))

    # Simple parsing
    words = comp_line[:comp_point].split()

    if len(words) >= 2:
        prev_word = (
            words[-2]
            if comp_line[comp_point - 1 : comp_point].isspace()
            else words[-2] if len(words) >= 2 else ""
        )
        current_word = (
            "" if comp_line[comp_point - 1 : comp_point].isspace() else words[-1]
        )

        # Provide completions based on previous word
        if prev_word == "--method":
            completions = ["classical", "mcmc", "robust", "all"]
        elif prev_word == "--config":
            completions = HomodyneCompleter.config_files_completer(current_word, None)
        elif prev_word == "--output-dir":
            completions = HomodyneCompleter.output_dir_completer(current_word, None)
        elif prev_word == "--install-completion":
            completions = ["bash", "zsh", "fish", "powershell"]
        else:
            completions = []

        # Filter by current word if any
        if current_word:
            completions = [c for c in completions if c.startswith(current_word)]

        # Output completions
        for completion in completions:
            print(completion)

        sys.exit(0)


def install_shell_completion(shell: str) -> int:
    """Install shell completion for the specified shell."""
    if not ARGCOMPLETE_AVAILABLE or argcomplete is None:
        print("Error: argcomplete package is required for shell completion.")
        print("Install with: pip install argcomplete")
        return 1

    completion_scripts = {
        "bash": """# Homodyne completion for bash
eval "$(register-python-argcomplete homodyne)"
eval "$(register-python-argcomplete homodyne-config)"
""",
        "zsh": """# Homodyne completion for zsh with fallback for compdef issues
# Try standard argcomplete first
if eval "$(register-python-argcomplete homodyne 2>/dev/null)"; then
    eval "$(register-python-argcomplete homodyne-config 2>/dev/null)"
else
    # Fallback completion for zsh compdef issues
    _homodyne_simple() {
        local cur prev
        cur="${words[CURRENT]}"
        prev="${words[CURRENT-1]}"
        
        case "${prev}" in
            --method)
                compadd classical mcmc robust all
                return 0
                ;;
            --config)
                compadd *.json(N) config/*.json(N) configs/*.json(N)
                return 0
                ;;
            --output-dir)
                compadd -S / -q */
                return 0
                ;;
        esac
        
        if [[ "${cur}" == --* ]]; then
            compadd --method --config --output-dir --verbose --quiet \\
                    --static-isotropic --static-anisotropic --laminar-flow \\
                    --plot-experimental-data --plot-simulated-data
            return 0
        fi
        
        return 1
    }
    
    # Try to register with compdef, fall back to aliases if it fails
    if ! compdef _homodyne_simple homodyne 2>/dev/null; then
        # Create convenience aliases as ultimate fallback
        alias homodyne-classical='homodyne --method classical'
        alias homodyne-mcmc='homodyne --method mcmc' 
        alias homodyne-robust='homodyne --method robust'
        alias homodyne-all='homodyne --method all'
        echo "Note: Tab completion may not work. Use aliases: homodyne-classical, homodyne-mcmc, homodyne-robust, homodyne-all"
    fi
fi
""",
        "fish": """# Homodyne completion for fish
register-python-argcomplete --shell fish homodyne | source
register-python-argcomplete --shell fish homodyne-config | source
""",
        "powershell": """# Homodyne completion for PowerShell
Register-ArgumentCompleter -Native -CommandName homodyne -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)
    $Env:_ARGCOMPLETE_COMP_WORDBREAKS = ' \t\n'
    $Env:_ARGCOMPLETE = 1
    $Env:_ARGCOMPLETE_SUPPRESS_SPACE = 1
    $Env:COMP_LINE = $commandAst
    $Env:COMP_POINT = $cursorPosition
    homodyne 2>&1 | Where-Object { $_ -like "$wordToComplete*" }
}
""",
    }

    if shell not in completion_scripts:
        supported_shells = ", ".join(completion_scripts.keys())
        print(f"Error: Shell '{shell}' not supported. Choose from: {supported_shells}")
        return 1

    script = completion_scripts[shell]

    # Determine the appropriate config file
    home = Path.home()
    config_files = {
        "bash": home / ".bashrc",
        "zsh": home / ".zshrc",
        "fish": home / ".config" / "fish" / "config.fish",
        "powershell": home
        / "Documents"
        / "PowerShell"
        / "Microsoft.PowerShell_profile.ps1",
    }

    config_file = config_files[shell]

    print(f"Installing {shell} completion for homodyne...")

    try:
        # Create directory if needed
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # Check if completion is already installed
        if config_file.exists():
            content = config_file.read_text()
            if "homodyne" in content and "argcomplete" in content:
                print(f"✓ Completion already installed in {config_file}")
                return 0

        # Append completion script
        with open(config_file, "a", encoding="utf-8") as f:
            f.write(f"\n{script}\n")

        print(f"✓ Completion installed in {config_file}")
        print(f"✓ Restart your {shell} session or run: source {config_file}")
        print("\nAfter reloading, you can use Tab completion:")
        print("  homodyne --method <TAB>")
        print("  homodyne --config <TAB>")
        print("  homodyne --output-dir <TAB>")

        # Install fast completion script
        try:
            script_source = Path(__file__).parent.parent / "homodyne_complete"

            if script_source.exists():
                # Make sure it's executable
                import stat

                script_source.chmod(script_source.stat().st_mode | stat.S_IEXEC)
                print("✓ Fast completion script configured for optimal performance")

            # Pre-populate cache for faster first use
            _completion_cache._update_cache()
            print("✓ Completion cache pre-populated for instant response")
        except Exception as e:
            print(f"⚠ Warning: Could not configure fast completion: {e}")
            print("  Completion will still work but may be slower")

        return 0

    except Exception as e:
        print(f"Error installing completion: {e}")
        return 1


# Export public functions
__all__ = [
    "setup_shell_completion",
    "install_shell_completion",
    "HomodyneCompleter",
]
