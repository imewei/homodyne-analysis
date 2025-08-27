"""
Shell Completion for Homodyne Analysis
=======================================

Provides shell completion features for the homodyne CLI:
- Shell completion for bash, zsh, fish, and PowerShell
- Context-aware suggestions for method names, config files, and directories

Usage:
    # Enable shell completion (one-time setup)
    homodyne --install-completion bash    # For bash
    homodyne --install-completion zsh     # For zsh
    homodyne --install-completion fish    # For fish

    # Shell completion in regular commands
    homodyne --method <TAB>     # Shows: classical, mcmc, robust, all
    homodyne --config <TAB>     # Shows available .json files
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import List

try:
    import argcomplete

    ARGCOMPLETE_AVAILABLE = True
except ImportError:
    ARGCOMPLETE_AVAILABLE = False
    argcomplete = None


# Readline is not directly used but checked for availability
try:
    import readline  # noqa: F401

    READLINE_AVAILABLE = True
except ImportError:
    READLINE_AVAILABLE = False


class HomodyneCompleter:
    """Provides tab completion suggestions for homodyne CLI arguments."""

    @staticmethod
    def method_completer(prefix: str, parsed_args, **kwargs) -> List[str]:
        """Suggest method names for --method argument."""
        methods = ["classical", "mcmc", "robust", "all"]
        return [m for m in methods if m.startswith(prefix)]

    @staticmethod
    def config_files_completer(prefix: str, parsed_args, **kwargs) -> List[str]:
        """Suggest JSON config files for --config argument."""
        # Look for JSON files in current directory
        files = glob.glob(f"{prefix}*.json")
        # Also check common config patterns
        if not prefix:
            files.extend(glob.glob("config*.json"))
            files.extend(glob.glob("homodyne*.json"))
        return sorted(set(files))

    @staticmethod
    def output_dir_completer(prefix: str, parsed_args, **kwargs) -> List[str]:
        """Suggest directories for --output-dir argument."""
        # Get directories matching prefix
        dirs = []
        path_prefix = prefix if prefix else "."
        parent_dir = os.path.dirname(path_prefix) if "/" in path_prefix else "."
        
        try:
            items = os.listdir(parent_dir)
            for item in items:
                full_path = os.path.join(parent_dir, item)
                if os.path.isdir(full_path) and item.startswith(
                    os.path.basename(path_prefix)
                ):
                    dirs.append(full_path + "/")
        except (OSError, PermissionError):
            pass
        
        return dirs

    @staticmethod
    def analysis_mode_completer(prefix: str, parsed_args, **kwargs) -> List[str]:
        """Suggest analysis modes for mode-related arguments."""
        modes = ["static_isotropic", "static_anisotropic", "laminar_flow"]
        return [m for m in modes if m.startswith(prefix)]


def setup_shell_completion(parser: argparse.ArgumentParser) -> None:
    """Add shell completion support to argument parser."""
    if not ARGCOMPLETE_AVAILABLE or argcomplete is None:
        return

    # Add completers to specific arguments
    for action in parser._actions:  # noqa: W291
        if action.dest == "method":
            # Use setattr to avoid type checker issues with dynamic attribute
            setattr(action, "completer", HomodyneCompleter.method_completer)
        elif action.dest == "config":
            setattr(action, "completer", HomodyneCompleter.config_files_completer)
        elif action.dest == "output_dir":
            setattr(action, "completer", HomodyneCompleter.output_dir_completer)

    # Enable argcomplete
    argcomplete.autocomplete(parser)


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
        "zsh": """# Homodyne completion for zsh
eval "$(register-python-argcomplete homodyne)"
eval "$(register-python-argcomplete homodyne-config)"
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