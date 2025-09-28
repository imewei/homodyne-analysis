"""
Homodyne Advanced Completion System
===================================

Modular, high-performance shell completion system with virtual environment isolation,
intelligent caching, and extensible plugin architecture.

Features:
- Plugin-based completion modules
- Environment-isolated caching with intelligent invalidation
- Project-aware context detection
- Cross-shell compatibility (bash, zsh, fish)
- Performance optimization with background cache warming
- Atomic installation/uninstallation
"""

from .adapter import (
    install_shell_completion,
    setup_shell_completion,
    uninstall_shell_completion,
)

__all__ = [
    "CacheConfig",
    "CompletionCache",
    "CompletionContext",
    "CompletionEngine",
    "CompletionInstaller",
    "CompletionPlugin",
    "InstallationConfig",
    "PluginManager",
    "install_shell_completion",
    "setup_shell_completion",
    "uninstall_shell_completion",
]

__version__ = "2.0.0"
