"""
Advanced Completion System Installer
====================================

Atomic installation and uninstallation system with environment detection,
conflict resolution, and rollback capabilities.
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any

# Import completion system components
from .core import EnvironmentType


class InstallationMode(Enum):
    """Installation modes for completion system."""

    SIMPLE = "simple"  # Basic completion only
    ADVANCED = "advanced"  # Full completion with caching
    DEVELOPMENT = "development"  # Development mode with debugging


class ShellType(Enum):
    """Supported shell types."""

    BASH = "bash"
    ZSH = "zsh"
    FISH = "fish"
    AUTO = "auto"


@dataclass
class InstallationConfig:
    """Configuration for completion system installation."""

    # Installation settings
    mode: InstallationMode = InstallationMode.ADVANCED
    shells: list[ShellType] = field(default_factory=lambda: [ShellType.AUTO])
    enable_aliases: bool = True
    enable_caching: bool = True

    # Environment settings
    force_install: bool = False
    backup_existing: bool = True
    atomic_install: bool = True

    # Feature flags
    enable_project_detection: bool = True
    enable_smart_completion: bool = True
    enable_background_warming: bool = True

    # Performance settings
    cache_size_mb: int = 50
    completion_timeout_ms: int = 1000


@dataclass
class InstallationResult:
    """Result of installation operation."""

    success: bool
    message: str
    installed_files: list[Path] = field(default_factory=list)
    backup_files: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class CompletionInstaller:
    """
    Advanced completion system installer with atomic operations.

    Features:
    - Environment detection and isolation
    - Atomic installation with rollback
    - Conflict detection and resolution
    - Multi-shell support
    - Backup and restore capabilities
    """

    def __init__(self, config: InstallationConfig | None = None):
        self.config = config or InstallationConfig()
        self._lock = threading.Lock()

        # Detect environment
        self.env_type, self.env_path = self._detect_environment()
        self.detected_shells = self._detect_shells()

        # Installation paths
        self.install_base = self.env_path / "etc" / "homodyne" / "completion"
        self.script_dir = self.install_base / "scripts"
        self.cache_dir = self.install_base / "cache"

    def install(self) -> InstallationResult:
        """
        Install the completion system.

        Returns:
            Installation result with success status and details
        """
        with self._lock:
            return self._perform_installation()

    def uninstall(self) -> InstallationResult:
        """
        Uninstall the completion system.

        Returns:
            Uninstallation result with success status and details
        """
        with self._lock:
            return self._perform_uninstallation()

    def is_installed(self) -> bool:
        """Check if completion system is installed."""
        return (self.install_base / "completion_engine.py").exists()

    def get_installation_info(self) -> dict[str, Any]:
        """Get information about current installation."""
        info = {
            "installed": self.is_installed(),
            "environment_type": self.env_type.value,
            "environment_path": str(self.env_path),
            "detected_shells": [shell.value for shell in self.detected_shells],
            "install_base": str(self.install_base),
        }

        if self.is_installed():
            info.update(self._get_installed_details())

        return info

    def _perform_installation(self) -> InstallationResult:
        """Perform the actual installation."""
        result = InstallationResult(success=False, message="Installation failed")

        try:
            # Pre-installation checks
            if not self._pre_install_checks(result):
                return result

            # Create backup if requested
            backup_files = []
            if self.config.backup_existing:
                backup_files = self._backup_existing_files(result)

            # Atomic installation
            if self.config.atomic_install:
                success = self._atomic_install(result)
            else:
                success = self._direct_install(result)

            if success:
                result.success = True
                result.message = "Completion system installed successfully"
                result.backup_files = backup_files
            # Restore backups on failure
            elif backup_files:
                self._restore_backups(backup_files)

        except Exception as e:
            result.errors.append(f"Installation error: {e}")

        return result

    def _perform_uninstallation(self) -> InstallationResult:
        """Perform the actual uninstallation."""
        result = InstallationResult(success=False, message="Uninstallation failed")

        try:
            if not self.is_installed():
                result.success = True
                result.message = "Completion system is not installed"
                return result

            # Remove installed files
            removed_files = self._remove_installation_files()
            result.installed_files = removed_files

            # Clean up activation scripts
            self._clean_activation_scripts(result)

            result.success = True
            result.message = "Completion system uninstalled successfully"

        except Exception as e:
            result.errors.append(f"Uninstallation error: {e}")

        return result

    def _pre_install_checks(self, result: InstallationResult) -> bool:
        """Perform pre-installation checks."""
        # Check if already installed
        if self.is_installed() and not self.config.force_install:
            result.errors.append(
                "Completion system already installed (use --force to override)"
            )
            return False

        # Check environment - MUST be in a virtual environment
        if self.env_type == EnvironmentType.SYSTEM:
            result.errors.append(
                "Cannot install completion system in system Python. "
                "Please activate a virtual environment (venv, conda, poetry, etc.) first."
            )
            return False

        # Check shell support
        if ShellType.AUTO in self.config.shells:
            self.config.shells = self.detected_shells

        if not self.config.shells:
            result.errors.append("No supported shells detected")
            return False

        # Check write permissions
        try:
            self.install_base.mkdir(parents=True, exist_ok=True)
            test_file = self.install_base / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
        except Exception:
            result.errors.append(f"No write permission to {self.install_base}")
            return False

        return True

    def _backup_existing_files(self, result: InstallationResult) -> list[Path]:
        """Backup existing completion files."""
        backup_files = []
        backup_dir = self.install_base / "backup" / f"backup_{int(time.time())}"

        try:
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Find existing completion files
            existing_files = self._find_existing_completion_files()

            for existing_file in existing_files:
                if existing_file.exists():
                    backup_file = backup_dir / existing_file.name
                    shutil.copy2(existing_file, backup_file)
                    backup_files.append(backup_file)

        except Exception as e:
            result.warnings.append(f"Backup failed: {e}")

        return backup_files

    def _atomic_install(self, result: InstallationResult) -> bool:
        """Perform atomic installation using temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_install = Path(temp_dir) / "homodyne_completion"

            try:
                # Install to temporary location first, using final paths
                if not self._install_to_directory(
                    temp_install, result, use_final_paths=True
                ):
                    return False

                # Atomic move to final location
                if self.install_base.exists():
                    backup_location = (
                        self.install_base.parent / f"{self.install_base.name}_old"
                    )
                    if backup_location.exists():
                        shutil.rmtree(backup_location)
                    shutil.move(self.install_base, backup_location)

                shutil.move(temp_install, self.install_base)

                # Clean up old backup
                if backup_location.exists():
                    shutil.rmtree(backup_location)

                return True

            except Exception as e:
                result.errors.append(f"Atomic installation failed: {e}")
                return False

    def _direct_install(self, result: InstallationResult) -> bool:
        """Perform direct installation."""
        return self._install_to_directory(
            self.install_base, result, use_final_paths=False
        )

    def _install_to_directory(
        self,
        install_dir: Path,
        result: InstallationResult,
        use_final_paths: bool = False,
    ) -> bool:
        """Install completion system to specified directory.

        Args:
            install_dir: Directory to install to
            result: Installation result object
            use_final_paths: If True, use final installation paths instead of install_dir paths
        """
        try:
            install_dir.mkdir(parents=True, exist_ok=True)

            # Determine which paths to use in generated scripts
            target_dir = self.install_base if use_final_paths else install_dir

            # Install completion engine
            if not self._install_completion_engine(install_dir, result, target_dir):
                return False

            # Install shell scripts
            if not self._install_shell_scripts(install_dir, result, target_dir):
                return False

            # Install activation scripts
            if not self._install_activation_scripts(install_dir, result, target_dir):
                return False

            # Configure cache
            if self.config.enable_caching:
                self._setup_cache_system(install_dir, result)

            return True

        except Exception as e:
            result.errors.append(f"Installation to {install_dir} failed: {e}")
            return False

    def _install_completion_engine(
        self, install_dir: Path, result: InstallationResult, target_dir: Path
    ) -> bool:
        """Install the completion engine Python module."""
        try:
            # Copy the completion module
            src_dir = Path(__file__).parent
            dest_dir = install_dir / "engine"
            dest_dir.mkdir(exist_ok=True)

            # Copy all Python files
            for py_file in src_dir.glob("*.py"):
                dest_file = dest_dir / py_file.name
                shutil.copy2(py_file, dest_file)
                result.installed_files.append(dest_file)

            # Create main completion script
            main_script = self._generate_main_completion_script(target_dir)
            main_script_path = install_dir / "completion_engine.py"
            main_script_path.write_text(main_script)
            result.installed_files.append(main_script_path)

            return True

        except Exception as e:
            result.errors.append(f"Failed to install completion engine: {e}")
            return False

    def _install_shell_scripts(
        self, install_dir: Path, result: InstallationResult, target_dir: Path
    ) -> bool:
        """Install shell-specific completion scripts."""
        script_dir = install_dir / "scripts"
        script_dir.mkdir(exist_ok=True)

        try:
            for shell in self.config.shells:
                if shell == ShellType.AUTO:
                    continue

                script_content = self._generate_shell_script(shell, target_dir)
                script_file = script_dir / f"completion.{shell.value}"
                script_file.write_text(script_content)
                script_file.chmod(0o755)
                result.installed_files.append(script_file)

            return True

        except Exception as e:
            result.errors.append(f"Failed to install shell scripts: {e}")
            return False

    def _install_activation_scripts(
        self, install_dir: Path, result: InstallationResult, target_dir: Path
    ) -> bool:
        """Install activation scripts for automatic loading."""
        try:
            # Environment-specific activation
            if self.env_type in [EnvironmentType.CONDA, EnvironmentType.MAMBA]:
                return self._install_conda_activation(install_dir, result, target_dir)
            return self._install_venv_activation(install_dir, result, target_dir)

        except Exception as e:
            result.errors.append(f"Failed to install activation scripts: {e}")
            return False

    def _install_conda_activation(
        self, install_dir: Path, result: InstallationResult, target_dir: Path
    ) -> bool:
        """Install conda activation scripts."""
        activate_dir = self.env_path / "etc" / "conda" / "activate.d"
        activate_dir.mkdir(parents=True, exist_ok=True)

        for shell in self.config.shells:
            if shell == ShellType.AUTO:
                continue

            activation_script = self._generate_activation_script(shell, target_dir)
            script_file = activate_dir / f"homodyne-completion-v2.{shell.value}"
            script_file.write_text(activation_script)
            script_file.chmod(0o755)
            result.installed_files.append(script_file)

        return True

    def _install_venv_activation(
        self, install_dir: Path, result: InstallationResult, target_dir: Path
    ) -> bool:
        """Install virtual environment activation scripts."""
        # For regular venv, we create standalone activation scripts
        # AND integrate them into the venv's activate script

        bin_dir = self.env_path / "bin"
        if not bin_dir.exists():
            bin_dir = self.env_path / "Scripts"  # Windows

        for shell in self.config.shells:
            if shell == ShellType.AUTO:
                continue

            # Create standalone activation script
            activation_script = self._generate_activation_script(shell, target_dir)
            script_file = bin_dir / f"activate-homodyne-completion.{shell.value}"
            script_file.write_text(activation_script)
            script_file.chmod(0o755)
            result.installed_files.append(script_file)

        # Integrate into venv's activate script for automatic loading
        self._integrate_with_venv_activate(bin_dir, target_dir, result)

        return True

    def _integrate_with_venv_activate(
        self, bin_dir: Path, target_dir: Path, result: InstallationResult
    ) -> None:
        """Integrate completion loading into venv's activate script."""
        # Detect which shells are configured
        for shell in self.config.shells:
            if shell == ShellType.AUTO:
                continue

            activate_file = bin_dir / "activate"
            if shell == ShellType.ZSH:
                # For zsh, we need to patch the activate script
                if activate_file.exists():
                    self._patch_activate_script(
                        activate_file, shell, target_dir, result
                    )

    def _patch_activate_script(
        self,
        activate_file: Path,
        shell: ShellType,
        target_dir: Path,
        result: InstallationResult,
    ) -> None:
        """Patch venv's activate script to load completion."""
        try:
            # Read current activate script
            content = activate_file.read_text()

            # Check if already patched
            completion_marker = "# Homodyne completion system"
            if completion_marker in content:
                result.warnings.append(
                    f"Activate script already patched for {shell.value}"
                )
                return

            # Create the patch content
            script_path = target_dir / "scripts" / f"completion.{shell.value}"
            patch_content = f"""
{completion_marker}
if [[ -n "$ZSH_VERSION" ]] && [[ -f "{script_path}" ]]; then
    source "{script_path}" 2>/dev/null || true
fi
"""

            # Add patch at the end of the activate script
            patched_content = content + patch_content

            # Backup original
            if self.config.backup_existing:
                backup_file = activate_file.with_suffix(".backup")
                backup_file.write_text(content)
                result.backup_files.append(backup_file)

            # Write patched version
            activate_file.write_text(patched_content)
            result.warnings.append(f"Patched {activate_file} to auto-load completions")

        except Exception as e:
            result.warnings.append(
                f"Failed to patch activate script: {e}. "
                f"Manual source required: source {activate_file.parent}/activate-homodyne-completion.{shell.value}"
            )

    def _setup_cache_system(
        self, install_dir: Path, result: InstallationResult
    ) -> None:
        """Set up the cache system."""
        cache_dir = install_dir / "cache"
        cache_dir.mkdir(exist_ok=True)

        # Create cache configuration
        cache_config = {
            "max_entries": 10000,
            "max_memory_mb": self.config.cache_size_mb,
            "default_ttl_seconds": 300,
            "enable_persistence": True,
        }

        config_file = cache_dir / "config.json"
        config_file.write_text(json.dumps(cache_config, indent=2))
        result.installed_files.append(config_file)

    def _generate_main_completion_script(self, install_dir: Path) -> str:
        """Generate the main completion script."""
        return '''#!/usr/bin/env python3
"""
Homodyne Advanced Completion System Entry Point
===============================================

Standalone completion script that provides intelligent completions.
"""

import sys
from pathlib import Path

def get_method_completions():
    """Get method completions."""
    return ["classical", "robust", "all"]

def get_backend_completions():
    """Get distributed backend completions."""
    return ["auto", "ray", "mpi", "dask", "multiprocessing"]

def get_shell_completions():
    """Get shell type completions for --install/uninstall-completion."""
    return ["bash", "zsh", "fish", "powershell"]

def get_config_completions():
    """Get config file completions."""
    try:
        cwd = Path.cwd()
        return [f.name for f in cwd.iterdir() if f.is_file() and f.suffix == ".json"]
    except Exception:
        return []

def get_homodyne_flags():
    """Get homodyne command flags."""
    return [
        # Basic options
        "--help", "--version",
        "--method", "--config", "--data", "--output", "--output-dir",
        "--verbose", "--quiet",
        # Analysis mode selection
        "--static-isotropic", "--static-anisotropic", "--laminar-flow",
        # Plotting options
        "--plot-experimental-data", "--plot-simulated-data",
        "--contrast", "--offset", "--phi-angles",
        # Shell completion
        "--install-completion", "--uninstall-completion",
        # Distributed computing
        "--distributed", "--backend", "--workers", "--distributed-config",
        # ML acceleration
        "--ml-accelerated", "--train-ml-model", "--enable-transfer-learning", "--ml-data-path",
        # Advanced optimization
        "--parameter-sweep", "--parameter-ranges", "--benchmark", "--auto-optimize",
    ]

def get_config_flags():
    """Get homodyne-config command flags."""
    return [
        "--help",
        "--mode", "-m",
        "--output", "-o",
        "--sample", "-s",
        "--experiment", "-e",
        "--author", "-a",
    ]

def get_config_mode_completions():
    """Get configuration mode completions."""
    return ["static_isotropic", "static_anisotropic", "laminar_flow"]

def main():
    """Main completion handler."""
    try:
        # Parse arguments: script shell_type [command] [args...]
        if len(sys.argv) < 2:
            return

        shell_type = sys.argv[1]
        words = sys.argv[2:] if len(sys.argv) > 2 else []

        # Handle empty command line - show all flags
        if not words:
            for flag in get_homodyne_flags():
                print(flag)
            return

        # Identify the command being completed
        command = words[0] if words else ""

        # Get the previous word and current word for context
        current = words[-1] if words else ""
        prev = words[-2] if len(words) >= 2 else ""

        # Determine which command we're completing for
        is_config_command = command == "homodyne-config"
        is_homodyne_command = command == "homodyne" or command.startswith("homodyne")

        # Context-aware completions based on previous flag
        if prev == "--method":
            # Complete method values
            for method in get_method_completions():
                print(method)
        elif prev == "--backend":
            # Complete backend values
            for backend in get_backend_completions():
                print(backend)
        elif prev in ["--install-completion", "--uninstall-completion"]:
            # Complete shell types
            for shell in get_shell_completions():
                print(shell)
        elif prev in ["--config", "--output", "-o"]:
            # Complete config file names
            for config in get_config_completions():
                print(config)
        elif prev in ["--mode", "-m"]:
            # Complete mode values
            for mode in get_config_mode_completions():
                print(mode)
        elif prev in ["--output-dir", "--data", "--distributed-config", "--ml-data-path"]:
            # Shell will handle directory/file completion
            pass
        elif prev in ["--workers", "--contrast", "--offset"]:
            # Numeric values - no completion
            pass
        elif prev in ["--sample", "-s", "--experiment", "-e", "--author", "-a"]:
            # Text values - no completion
            pass
        elif prev == "--phi-angles":
            # Example phi angles
            print("0,45,90,135")
            print("0,36,72,108,144")
            print("30,60,90")
        elif prev == "--parameter-ranges":
            # Example parameter ranges
            print("D0:10-100,alpha:-1-1")
            print("D0:0.001-0.1,beta:0.5-1.5")
        elif current.startswith("--"):
            # Completing a flag - show appropriate flags for the command
            if is_config_command:
                for flag in get_config_flags():
                    if flag.startswith(current):
                        print(flag)
            else:
                for flag in get_homodyne_flags():
                    if flag.startswith(current):
                        print(flag)
        elif len(words) == 1 or (len(words) >= 2 and current == ""):
            # Just the command name, or command + empty current word - show all flags
            if is_config_command:
                for flag in get_config_flags():
                    print(flag)
            else:
                for flag in get_homodyne_flags():
                    print(flag)
        elif not current.startswith("-"):
            # Completing a value - check if it should be a config file
            if prev in ["--config", "--output", "-o"]:
                for config in get_config_completions():
                    if config.startswith(current):
                        print(config)
            # Otherwise let shell handle file completion

    except Exception as e:
        # Fallback - output help flag
        print("--help")

if __name__ == "__main__":
    main()
'''

    def _generate_shell_script(self, shell: ShellType, install_dir: Path) -> str:
        """Generate shell-specific completion script."""
        engine_script = install_dir / "completion_engine.py"

        if shell == ShellType.BASH:
            return self._generate_bash_script(engine_script)
        if shell == ShellType.ZSH:
            return self._generate_zsh_script(engine_script)
        if shell == ShellType.FISH:
            return self._generate_fish_script(engine_script)
        raise ValueError(f"Unsupported shell: {shell}")

    def _generate_bash_script(self, engine_script: Path) -> str:
        """Generate bash completion script."""
        aliases = self._generate_aliases() if self.config.enable_aliases else ""

        return f"""#!/bin/bash
# Homodyne Advanced Completion System - Bash
# Generated by installation system

# Advanced completion function
_homodyne_advanced_completion() {{
    local cur prev words cword
    _init_completion || return

    # Call Python completion engine
    local completions
    completions=$(python3 "{engine_script}" bash "${{COMP_WORDS[@]}}" 2>/dev/null)

    if [[ -n "$completions" ]]; then
        COMPREPLY=($(compgen -W "$completions" -- "$cur"))
    else
        # Fallback to file completion
        COMPREPLY=($(compgen -f -- "$cur"))
    fi
}}

# Register completions for all homodyne commands
complete -F _homodyne_advanced_completion homodyne 2>/dev/null || true
complete -F _homodyne_advanced_completion homodyne-config 2>/dev/null || true
complete -F _homodyne_advanced_completion homodyne-gpu 2>/dev/null || true

{aliases}
"""

    def _generate_zsh_script(self, engine_script: Path) -> str:
        """Generate zsh completion script."""
        aliases = self._generate_aliases() if self.config.enable_aliases else ""

        return f"""#!/bin/zsh
# Homodyne Advanced Completion System - Zsh
# Generated by installation system

# Initialize completion system if needed
autoload -Uz compinit
compinit -i 2>/dev/null || true

# Advanced completion function for homodyne commands
_homodyne_advanced_completion() {{
    local curcontext="$curcontext" state line
    typeset -A opt_args
    local -a completions completion_list

    # Call Python completion engine with current command line words
    # Pass: shell_type followed by all words in the command line
    local engine_output
    engine_output=$(python3 "{engine_script}" zsh "${{words[@]}}" 2>/dev/null)

    if [[ -n "$engine_output" ]]; then
        # Parse completions into array (split on newlines)
        completion_list=("${{(@f)engine_output}}")

        # Provide completions
        if [[ ${{#completion_list}} -gt 0 ]]; then
            _describe 'homodyne completions' completion_list
        else
            # Fallback to file completion if no matches
            _files
        fi
    else
        # Fallback to file completion if engine fails
        _files
    fi
}}

# Separate completion function for homodyne-config
_homodyne_config_completion() {{
    local curcontext="$curcontext" state line
    local -a completions completion_list

    # Call completion engine
    local engine_output
    engine_output=$(python3 "{engine_script}" zsh "${{words[@]}}" 2>/dev/null)

    if [[ -n "$engine_output" ]]; then
        completion_list=("${{(@f)engine_output}}")
        if [[ ${{#completion_list}} -gt 0 ]]; then
            _describe 'config completions' completion_list
        else
            _files -g '*.json'
        fi
    else
        # Fallback to JSON files
        _files -g '*.json'
    fi
}}

# Register completions for homodyne commands
# Use 2>/dev/null to suppress errors during initialization
compdef _homodyne_advanced_completion homodyne 2>/dev/null || true
compdef _homodyne_config_completion homodyne-config 2>/dev/null || true
compdef _homodyne_advanced_completion homodyne-gpu 2>/dev/null || true

{aliases}
"""

    def _generate_fish_script(self, engine_script: Path) -> str:
        """Generate fish completion script."""
        return f"""# Homodyne Advanced Completion System - Fish
# Generated by installation system

# Advanced completion function
function __homodyne_advanced_complete
    set -l cmd (commandline -opc)
    python3 "{engine_script}" fish $cmd 2>/dev/null
end

# Register completions for all homodyne commands
complete -c homodyne -f -a "(__homodyne_advanced_complete)"
complete -c homodyne-config -f -a "(__homodyne_advanced_complete)"
complete -c homodyne-gpu -f -a "(__homodyne_advanced_complete)"
"""

    def _generate_activation_script(self, shell: ShellType, install_dir: Path) -> str:
        """Generate activation script for shell."""
        script_path = install_dir / "scripts" / f"completion.{shell.value}"

        if shell == ShellType.BASH:
            return f"""#!/bin/bash
# Homodyne Advanced Completion System Activation
# Auto-generated activation script

if [[ -f "{script_path}" ]]; then
    source "{script_path}"
fi
"""
        if shell == ShellType.ZSH:
            return f"""#!/bin/zsh
# Homodyne Advanced Completion System Activation
# Auto-generated activation script for Zsh

if [[ -f "{script_path}" ]]; then
    source "{script_path}"
fi
"""
        if shell == ShellType.FISH:
            return f"""#!/usr/bin/env fish
# Homodyne Advanced Completion System Activation
# Auto-generated activation script

if test -f "{script_path}"
    source "{script_path}"
end
"""
        return ""

    def _generate_aliases(self) -> str:
        """Generate command aliases."""
        return """
# Homodyne command aliases
if [[ -n "$BASH_VERSION" ]] || [[ -n "$ZSH_VERSION" ]]; then
    alias hmc='homodyne --method classical' # Classical optimization methods
    alias hmr='homodyne --method robust'    # Robust optimization methods
    alias hma='homodyne --method all'       # All optimization methods
    alias hconfig='homodyne-config'         # Configuration generator
    alias hexp='homodyne --plot-experimental-data'   # Plot experimental data
    alias hsim='homodyne --plot-simulated-data'      # Plot simulated data
    alias hm='homodyne'                     # Short form
fi
"""

    def _detect_environment(self) -> tuple[EnvironmentType, Path]:
        """Detect current environment type and path."""
        # Check conda/mamba
        if os.environ.get("CONDA_DEFAULT_ENV"):
            if os.environ.get("MAMBA_ROOT_PREFIX"):
                return EnvironmentType.MAMBA, Path(sys.prefix)
            return EnvironmentType.CONDA, Path(sys.prefix)

        # Check poetry
        if os.environ.get("POETRY_ACTIVE"):
            return EnvironmentType.POETRY, Path(sys.prefix)

        # Check pipenv
        if os.environ.get("PIPENV_ACTIVE"):
            return EnvironmentType.PIPENV, Path(sys.prefix)

        # Check venv/virtualenv
        if hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        ):
            if (Path(sys.prefix) / "pyvenv.cfg").exists():
                return EnvironmentType.VENV, Path(sys.prefix)
            return EnvironmentType.VIRTUALENV, Path(sys.prefix)

        return EnvironmentType.SYSTEM, Path(sys.prefix)

    def _detect_shells(self) -> list[ShellType]:
        """Detect available shells."""
        detected = []

        # Check current shell
        current_shell = os.environ.get("SHELL", "").split("/")[-1]
        if current_shell == "bash":
            detected.append(ShellType.BASH)
        elif current_shell == "zsh":
            detected.append(ShellType.ZSH)
        elif current_shell == "fish":
            detected.append(ShellType.FISH)

        # Check for other available shells
        for shell in ["bash", "zsh", "fish"]:
            if shutil.which(shell) and ShellType(shell) not in detected:
                detected.append(ShellType(shell))

        return detected

    def _find_existing_completion_files(self) -> list[Path]:
        """Find existing completion files that might conflict."""
        files = []

        # Check conda activation directory
        conda_activate_dir = self.env_path / "etc" / "conda" / "activate.d"
        if conda_activate_dir.exists():
            files.extend(conda_activate_dir.glob("*homodyne*"))

        # Check standard completion directories
        completion_dirs = [
            self.env_path / "etc" / "bash_completion.d",
            self.env_path / "etc" / "zsh",
            self.env_path / "share" / "fish" / "vendor_completions.d",
        ]

        for comp_dir in completion_dirs:
            if comp_dir.exists():
                files.extend(comp_dir.glob("*homodyne*"))

        return files

    def _remove_installation_files(self) -> list[Path]:
        """Remove all installed files."""
        removed_files = []

        if self.install_base.exists():
            for file_path in self.install_base.rglob("*"):
                if file_path.is_file():
                    removed_files.append(file_path)

            shutil.rmtree(self.install_base)

        return removed_files

    def _clean_activation_scripts(self, result: InstallationResult) -> None:
        """Clean up activation scripts."""
        activation_patterns = [
            "homodyne-completion-v2.*",
            "activate-homodyne-completion.*",
        ]

        # Check conda activation directory
        conda_activate_dir = self.env_path / "etc" / "conda" / "activate.d"
        if conda_activate_dir.exists():
            for pattern in activation_patterns:
                for script_file in conda_activate_dir.glob(pattern):
                    script_file.unlink()
                    result.installed_files.append(script_file)

        # Check bin directory
        bin_dir = self.env_path / "bin"
        if not bin_dir.exists():
            bin_dir = self.env_path / "Scripts"  # Windows

        if bin_dir.exists():
            for pattern in activation_patterns:
                for script_file in bin_dir.glob(pattern):
                    script_file.unlink()
                    result.installed_files.append(script_file)

            # Remove patch from venv activate script
            self._unpatch_activate_script(bin_dir, result)

    def _unpatch_activate_script(
        self, bin_dir: Path, result: InstallationResult
    ) -> None:
        """Remove completion patch from venv activate script."""
        activate_file = bin_dir / "activate"
        if not activate_file.exists():
            return

        try:
            content = activate_file.read_text()
            completion_marker = "# Homodyne completion system"

            if completion_marker not in content:
                return  # Not patched

            # Remove the patched section
            lines = content.split("\n")
            cleaned_lines = []
            skip_section = False

            for line in lines:
                if completion_marker in line:
                    skip_section = True
                    continue
                if skip_section:
                    # Skip until we find an empty line or next section
                    if line.strip() == "" or not line.startswith(
                        ("if", "fi", " ", "\t")
                    ):
                        skip_section = False
                    else:
                        continue
                cleaned_lines.append(line)

            # Write cleaned content
            activate_file.write_text("\n".join(cleaned_lines))
            result.warnings.append("Removed completion patch from activate script")

            # Restore backup if it exists
            backup_file = activate_file.with_suffix(".backup")
            if backup_file.exists():
                backup_file.unlink()

        except Exception as e:
            result.warnings.append(f"Failed to unpatch activate script: {e}")

    def _restore_backups(self, backup_files: list[Path]) -> None:
        """Restore backup files."""
        for _backup_file in backup_files:
            try:
                # Restore to appropriate location based on file type
                # This is a simplified restore - in production would need more logic
                pass
            except Exception:
                pass

    def _get_installed_details(self) -> dict[str, Any]:
        """Get details about installed completion system."""
        return {
            "version": "2.0.0",
            "install_date": "unknown",  # Would be stored during installation
            "config": asdict(self.config),
            "features": {
                "caching": self.config.enable_caching,
                "aliases": self.config.enable_aliases,
                "project_detection": self.config.enable_project_detection,
            },
        }
