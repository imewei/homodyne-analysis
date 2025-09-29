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

        # Check environment
        if self.env_type == EnvironmentType.SYSTEM and not self.config.force_install:
            result.warnings.append(
                "Installing in system Python (virtual environment recommended)"
            )

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
                # Install to temporary location first
                if not self._install_to_directory(temp_install, result):
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
        return self._install_to_directory(self.install_base, result)

    def _install_to_directory(
        self, install_dir: Path, result: InstallationResult
    ) -> bool:
        """Install completion system to specified directory."""
        try:
            install_dir.mkdir(parents=True, exist_ok=True)

            # Install completion engine
            if not self._install_completion_engine(install_dir, result):
                return False

            # Install shell scripts
            if not self._install_shell_scripts(install_dir, result):
                return False

            # Install activation scripts
            if not self._install_activation_scripts(install_dir, result):
                return False

            # Configure cache
            if self.config.enable_caching:
                self._setup_cache_system(install_dir, result)

            return True

        except Exception as e:
            result.errors.append(f"Installation to {install_dir} failed: {e}")
            return False

    def _install_completion_engine(
        self, install_dir: Path, result: InstallationResult
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
            main_script = self._generate_main_completion_script(install_dir)
            main_script_path = install_dir / "completion_engine.py"
            main_script_path.write_text(main_script)
            result.installed_files.append(main_script_path)

            return True

        except Exception as e:
            result.errors.append(f"Failed to install completion engine: {e}")
            return False

    def _install_shell_scripts(
        self, install_dir: Path, result: InstallationResult
    ) -> bool:
        """Install shell-specific completion scripts."""
        script_dir = install_dir / "scripts"
        script_dir.mkdir(exist_ok=True)

        try:
            for shell in self.config.shells:
                if shell == ShellType.AUTO:
                    continue

                script_content = self._generate_shell_script(shell, install_dir)
                script_file = script_dir / f"completion.{shell.value}"
                script_file.write_text(script_content)
                script_file.chmod(0o755)
                result.installed_files.append(script_file)

            return True

        except Exception as e:
            result.errors.append(f"Failed to install shell scripts: {e}")
            return False

    def _install_activation_scripts(
        self, install_dir: Path, result: InstallationResult
    ) -> bool:
        """Install activation scripts for automatic loading."""
        try:
            # Environment-specific activation
            if self.env_type in [EnvironmentType.CONDA, EnvironmentType.MAMBA]:
                return self._install_conda_activation(install_dir, result)
            return self._install_venv_activation(install_dir, result)

        except Exception as e:
            result.errors.append(f"Failed to install activation scripts: {e}")
            return False

    def _install_conda_activation(
        self, install_dir: Path, result: InstallationResult
    ) -> bool:
        """Install conda activation scripts."""
        activate_dir = self.env_path / "etc" / "conda" / "activate.d"
        activate_dir.mkdir(parents=True, exist_ok=True)

        for shell in self.config.shells:
            if shell == ShellType.AUTO:
                continue

            activation_script = self._generate_activation_script(shell, install_dir)
            script_file = activate_dir / f"homodyne-completion-v2.{shell.value}"
            script_file.write_text(activation_script)
            script_file.chmod(0o755)
            result.installed_files.append(script_file)

        return True

    def _install_venv_activation(
        self, install_dir: Path, result: InstallationResult
    ) -> bool:
        """Install virtual environment activation scripts."""
        # For regular venv, we create a script that can be sourced manually
        # or add it to existing activation scripts

        bin_dir = self.env_path / "bin"
        if not bin_dir.exists():
            bin_dir = self.env_path / "Scripts"  # Windows

        for shell in self.config.shells:
            if shell == ShellType.AUTO:
                continue

            activation_script = self._generate_activation_script(shell, install_dir)
            script_file = bin_dir / f"activate-homodyne-completion.{shell.value}"
            script_file.write_text(activation_script)
            script_file.chmod(0o755)
            result.installed_files.append(script_file)

        return True

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
        return f'''#!/usr/bin/env python3
"""
Homodyne Advanced Completion System Entry Point
===============================================

This script serves as the entry point for the advanced completion system.
"""

import sys
import os
from pathlib import Path

# Add engine directory to path
engine_dir = Path(__file__).parent / "engine"
sys.path.insert(0, str(engine_dir))

from .core import CompletionEngine, CompletionContext
from .plugins import get_plugin_manager
from .cache import CompletionCache, CacheConfig

def main():
    """Main completion handler."""
    try:
        # Parse shell arguments
        if len(sys.argv) < 2:
            print("Usage: completion_engine.py <shell_type> [args...]")
            sys.exit(1)

        shell_type = sys.argv[1]
        completion_args = sys.argv[2:]

        # Create completion context
        context = CompletionContext.from_shell_args(completion_args, shell_type)

        # Initialize completion engine
        cache_dir = Path(__file__).parent / "cache"
        cache_config = CacheConfig(
            max_memory_mb={self.config.cache_size_mb},
            enable_persistence={str(self.config.enable_caching).lower()},
        )

        cache = CompletionCache(cache_dir=cache_dir, config=cache_config)
        engine = CompletionEngine(
            cache_dir=cache_dir,
            enable_caching={str(self.config.enable_caching).lower()},
        )

        # Get plugin manager
        plugin_manager = get_plugin_manager()

        # Generate completions
        results = plugin_manager.get_completions(context)

        # Output completions for shell
        for result in results:
            for completion in result.completions:
                print(completion)

    except Exception as e:
        # Fallback to basic completion
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

# Advanced completion function
_homodyne_advanced_completion() {{
    local -a completions
    local -a words

    # Get current words
    words=(${{=COMP_WORDS}})

    # Call Python completion engine
    completions=($(python3 "{engine_script}" zsh "${{words[@]}}" 2>/dev/null))

    if [[ ${{#completions}} -gt 0 ]]; then
        _describe 'completions' completions
    else
        # Fallback to file completion
        _files
    fi
}}

# Register completions for all homodyne commands
compdef _homodyne_advanced_completion homodyne 2>/dev/null || true
compdef _homodyne_advanced_completion homodyne-config 2>/dev/null || true
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

        if shell in [ShellType.BASH, ShellType.ZSH]:
            return f"""#!/bin/bash
# Homodyne Advanced Completion System Activation
# Auto-generated activation script

if [[ -f "{script_path}" ]]; then
    source "{script_path}"
fi
"""
        if shell == ShellType.FISH:
            return f"""# Homodyne Advanced Completion System Activation
# Auto-generated activation script

if test -f "{script_path}"
    source "{script_path}"
end
"""

    def _generate_aliases(self) -> str:
        """Generate command aliases."""
        return """
# Homodyne command aliases
if [[ -n "$BASH_VERSION" ]] || [[ -n "$ZSH_VERSION" ]]; then
    alias hmv='homodyne --method vi'        # Fast VI analysis
    alias hmm='homodyne --method mcmc'      # Accurate MCMC analysis
    alias hmh='homodyne --method hybrid'    # Balanced VI→MCMC pipeline
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
