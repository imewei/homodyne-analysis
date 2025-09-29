"""
Adapter for Legacy Completion System Integration
===============================================

This module provides backward-compatible adapters that integrate the advanced
completion system with the existing CLI infrastructure, ensuring seamless
migration without breaking existing functionality.
"""

import argparse
from pathlib import Path
from typing import Any

try:
    from .cache import CacheConfig
    from .cache import CompletionCache
    from .core import CompletionContext
    from .core import CompletionEngine
    from .installer import CompletionInstaller
    from .installer import InstallationConfig
    from .installer import InstallationMode
except ImportError:
    from cache import CacheConfig
    from cache import CompletionCache
    from core import CompletionContext
    from core import CompletionEngine
    from installer import CompletionInstaller
    from installer import InstallationConfig
    from installer import InstallationMode


class LegacyCompletionAdapter:
    """
    Adapter that provides legacy completion interface using the new system.

    This adapter maintains backward compatibility with existing CLI code
    while leveraging the advanced completion engine underneath.
    """

    def __init__(self):
        self._engine: CompletionEngine | None = None
        self._installer: CompletionInstaller | None = None
        self._cache: CompletionCache | None = None

    def _get_engine(self) -> CompletionEngine:
        """Get or create the completion engine instance."""
        if self._engine is None:
            try:
                # Initialize cache
                cache_config = CacheConfig(
                    max_memory_mb=50,
                    default_ttl_seconds=300,
                    enable_persistence=True,
                )
                self._cache = CompletionCache(config=cache_config)

                # Initialize engine
                self._engine = CompletionEngine(
                    enable_caching=True,
                    max_cache_size=1000,
                    background_warming=True,
                )
            except Exception:
                # Fallback to minimal engine
                self._engine = CompletionEngine(
                    enable_caching=False,
                    background_warming=False,
                )

        return self._engine

    def _get_installer(self) -> CompletionInstaller:
        """Get or create the completion installer instance."""
        if self._installer is None:
            config = InstallationConfig(
                mode=InstallationMode.ADVANCED,
                enable_aliases=True,
                enable_caching=True,
                force_install=False,
                backup_existing=True,
            )
            self._installer = CompletionInstaller(config)

        return self._installer

    def get_method_completions(self, prefix: str = "") -> list[str]:
        """Get method completions (legacy interface)."""
        try:
            engine = self._get_engine()
            context = CompletionContext.from_shell_args(
                ["homodyne", "--method", prefix], "bash"
            )
            results = engine.complete(context)

            # Extract completions from results
            completions = []
            for result in results:
                completions.extend(result.completions)

            # Filter by prefix if provided
            if prefix:
                completions = [c for c in completions if c.startswith(prefix.lower())]

            return completions or ["classical", "robust", "all"]  # Fallback

        except Exception:
            # Fallback to static completions
            methods = ["classical", "robust", "all"]
            if prefix:
                return [m for m in methods if m.startswith(prefix.lower())]
            return methods

    def get_config_file_completions(self, prefix: str = "") -> list[str]:
        """Get config file completions (legacy interface)."""
        try:
            engine = self._get_engine()
            context = CompletionContext.from_shell_args(
                ["homodyne", "--config", prefix], "bash"
            )
            results = engine.complete(context)

            # Extract completions from results
            completions = []
            for result in results:
                completions.extend(result.completions)

            return completions

        except Exception:
            # Fallback to simple file scan
            try:
                cwd = Path.cwd()
                json_files = [
                    f.name for f in cwd.iterdir() if f.is_file() and f.suffix == ".json"
                ]

                if prefix:
                    json_files = [
                        f for f in json_files if f.lower().startswith(prefix.lower())
                    ]

                return json_files
            except Exception:
                return []

    def get_directory_completions(self, prefix: str = "") -> list[str]:
        """Get directory completions (legacy interface)."""
        try:
            engine = self._get_engine()
            context = CompletionContext.from_shell_args(
                ["homodyne", "--output-dir", prefix], "bash"
            )
            results = engine.complete(context)

            # Extract completions from results
            completions = []
            for result in results:
                completions.extend(result.completions)

            return completions

        except Exception:
            # Fallback to simple directory scan
            try:
                cwd = Path.cwd()
                dirs = [d.name + "/" for d in cwd.iterdir() if d.is_dir()]

                if prefix:
                    dirs = [d for d in dirs if d.lower().startswith(prefix.lower())]

                return dirs
            except Exception:
                return []

    def get_mode_completions(self, prefix: str = "") -> list[str]:
        """Get analysis mode completions (legacy interface)."""
        try:
            engine = self._get_engine()
            context = CompletionContext.from_shell_args(
                ["homodyne-config", "--mode", prefix], "bash"
            )
            results = engine.complete(context)

            # Extract completions from results
            completions = []
            for result in results:
                completions.extend(result.completions)

            return completions or [
                "static_isotropic",
                "static_anisotropic",
                "laminar_flow",
            ]

        except Exception:
            # Fallback to static modes
            modes = ["static_isotropic", "static_anisotropic", "laminar_flow"]
            if prefix:
                return [m for m in modes if m.startswith(prefix.lower())]
            return modes

    def setup_shell_completion(self, parser: argparse.ArgumentParser) -> None:
        """Set up shell completion for argparse parser (legacy interface)."""
        try:
            # Try to use argcomplete if available
            try:
                import argcomplete

                # Create custom completers using the new system
                def method_completer(prefix, parsed_args, **kwargs):
                    return self.get_method_completions(prefix)

                def config_completer(prefix, parsed_args, **kwargs):
                    return self.get_config_file_completions(prefix)

                def dir_completer(prefix, parsed_args, **kwargs):
                    return self.get_directory_completions(prefix)

                def mode_completer(prefix, parsed_args, **kwargs):
                    return self.get_mode_completions(prefix)

                # Attach completers to actions
                for action in parser._actions:
                    if action.dest == "method":
                        action.completer = method_completer
                    elif action.dest == "config":
                        action.completer = config_completer
                    elif action.dest == "output_dir":
                        action.completer = dir_completer
                    elif action.dest == "mode":
                        action.completer = mode_completer
                    elif action.dest == "output":
                        action.completer = config_completer

                # Enable argcomplete
                argcomplete.autocomplete(parser)

            except ImportError:
                # argcomplete not available, skip completion setup
                pass

        except Exception:
            # Completion setup failed, continue without completion
            pass

    def install_shell_completion(self, shell: str) -> int:
        """Install shell completion (legacy interface)."""
        try:
            installer = self._get_installer()

            # Update installer config for specified shell
            if shell == "bash":
                from .installer import ShellType

                installer.config.shells = [ShellType.BASH]
            elif shell == "zsh":
                from .installer import ShellType

                installer.config.shells = [ShellType.ZSH]
            elif shell == "fish":
                from .installer import ShellType

                installer.config.shells = [ShellType.FISH]
            else:
                print(f"Unsupported shell: {shell}")
                return 1

            result = installer.install()

            if result.success:
                print(f"✓ Advanced completion system installed for {shell}")
                print("✓ Restart your shell session to enable completions")
                return 0
            else:
                print(f"✗ Installation failed: {result.message}")
                for error in result.errors:
                    print(f"  Error: {error}")
                return 1

        except Exception as e:
            print(f"✗ Installation error: {e}")
            return 1

    def uninstall_shell_completion(self, shell: str) -> int:
        """Uninstall shell completion (legacy interface)."""
        try:
            installer = self._get_installer()
            result = installer.uninstall()

            if result.success:
                print("✓ Advanced completion system uninstalled")
                print("✓ Restart your shell session for changes to take effect")
                return 0
            else:
                print(f"✗ Uninstallation failed: {result.message}")
                for error in result.errors:
                    print(f"  Error: {error}")
                return 1

        except Exception as e:
            print(f"✗ Uninstallation error: {e}")
            return 1

    def is_installed(self) -> bool:
        """Check if advanced completion system is installed."""
        try:
            installer = self._get_installer()
            return installer.is_installed()
        except Exception:
            return False

    def get_installation_info(self) -> dict[str, Any]:
        """Get installation information."""
        try:
            installer = self._get_installer()
            return installer.get_installation_info()
        except Exception:
            return {"installed": False, "error": "Unable to get installation info"}


# Global adapter instance
_adapter: LegacyCompletionAdapter | None = None


def get_adapter() -> LegacyCompletionAdapter:
    """Get the global adapter instance."""
    global _adapter
    if _adapter is None:
        _adapter = LegacyCompletionAdapter()
    return _adapter


# Legacy interface functions for backward compatibility
def method_completer(prefix: str, parsed_args=None, **kwargs) -> list[str]:
    """Legacy method completer function."""
    return get_adapter().get_method_completions(prefix)


def config_files_completer(prefix: str, parsed_args=None, **kwargs) -> list[str]:
    """Legacy config file completer function."""
    return get_adapter().get_config_file_completions(prefix)


def output_dir_completer(prefix: str, parsed_args=None, **kwargs) -> list[str]:
    """Legacy output directory completer function."""
    return get_adapter().get_directory_completions(prefix)


def analysis_mode_completer(prefix: str, parsed_args=None, **kwargs) -> list[str]:
    """Legacy analysis mode completer function."""
    return get_adapter().get_mode_completions(prefix)


def setup_shell_completion(parser: argparse.ArgumentParser) -> None:
    """Legacy shell completion setup function."""
    get_adapter().setup_shell_completion(parser)


def install_shell_completion(shell: str) -> int:
    """Legacy shell completion installation function."""
    return get_adapter().install_shell_completion(shell)


def uninstall_shell_completion(shell: str) -> int:
    """Legacy shell completion uninstallation function."""
    return get_adapter().uninstall_shell_completion(shell)
