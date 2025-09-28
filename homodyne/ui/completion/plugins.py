"""
Completion Plugin System
========================

Extensible plugin architecture for custom completion modules.
Supports dynamic loading, plugin priorities, and context-aware completions.
"""

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

try:
    from .core import CompletionContext, CompletionResult, CompletionType
except ImportError:
    from core import CompletionContext, CompletionResult, CompletionType


@dataclass
class PluginInfo:
    """Information about a completion plugin."""

    name: str
    version: str
    description: str
    author: str
    priority: int = 50
    supports_shells: list[str] = None
    requires_packages: list[str] = None
    enabled: bool = True

    def __post_init__(self):
        if self.supports_shells is None:
            self.supports_shells = ["bash", "zsh", "fish"]
        if self.requires_packages is None:
            self.requires_packages = []


class CompletionPlugin(ABC):
    """
    Base class for completion plugins.

    Plugins extend the completion system with custom logic for specific
    completion scenarios, commands, or contexts.
    """

    def __init__(self):
        self._info: PluginInfo | None = None
        self._enabled = True

    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """Plugin information and metadata."""
        pass

    @abstractmethod
    def can_complete(self, context: CompletionContext) -> bool:
        """
        Check if this plugin can provide completions for the given context.

        Args:
            context: Completion context with command, words, environment info

        Returns:
            True if plugin can provide completions, False otherwise
        """
        pass

    @abstractmethod
    def complete(self, context: CompletionContext) -> list[CompletionResult]:
        """
        Generate completions for the given context.

        Args:
            context: Completion context

        Returns:
            List of completion results
        """
        pass

    def initialize(self) -> bool:
        """
        Initialize the plugin. Called when plugin is loaded.

        Returns:
            True if initialization successful, False otherwise
        """
        return True

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup plugin resources. Called when plugin is unloaded."""
        pass

    @property
    def enabled(self) -> bool:
        """Whether plugin is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable plugin."""
        self._enabled = value


class HomodyneCommandPlugin(CompletionPlugin):
    """Plugin for core homodyne command completions."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="homodyne-core",
            version="1.0.0",
            description="Core homodyne command completions",
            author="Homodyne Team",
            priority=100,
        )

    def can_complete(self, context: CompletionContext) -> bool:
        """Can complete homodyne commands."""
        return context.command in ["homodyne", "homodyne-config", "homodyne-gpu"]

    def complete(self, context: CompletionContext) -> list[CompletionResult]:
        """Complete homodyne commands."""
        if context.command == "homodyne":
            return self._complete_homodyne(context)
        elif context.command == "homodyne-config":
            return self._complete_homodyne_config(context)
        elif context.command == "homodyne-gpu":
            return self._complete_homodyne_gpu(context)
        return []

    def _complete_homodyne(self, context: CompletionContext) -> list[CompletionResult]:
        """Complete main homodyne command."""
        if not context.words or len(context.words) == 1:
            # Main options
            options = [
                "--help",
                "--method",
                "--config",
                "--output-dir",
                "--verbose",
                "--quiet",
                "--static-isotropic",
                "--static-anisotropic",
                "--laminar-flow",
                "--plot-experimental-data",
                "--plot-simulated-data",
                "--contrast",
                "--offset",
                "--phi-angles",
            ]
            matching = [opt for opt in options if opt.startswith(context.current_word)]
            if matching:
                return [
                    CompletionResult(
                        completions=matching,
                        completion_type=CompletionType.OPTION,
                        description="Homodyne analysis options",
                        priority=90,
                        source_plugin="homodyne-core",
                    )
                ]

        elif context.previous_word == "--method":
            # Smart method completion based on config
            methods = self._get_smart_methods(context)
            matching = [m for m in methods if m.startswith(context.current_word)]
            if matching:
                return [
                    CompletionResult(
                        completions=matching,
                        completion_type=CompletionType.VALUE,
                        description="Analysis methods",
                        priority=95,
                        source_plugin="homodyne-core",
                        metadata={"context_aware": True},
                    )
                ]

        elif context.previous_word == "--config":
            # Config file completion
            configs = self._get_config_files(context)
            matching = [c for c in configs if c.startswith(context.current_word)]
            if matching:
                return [
                    CompletionResult(
                        completions=matching,
                        completion_type=CompletionType.FILE,
                        description="Configuration files",
                        priority=95,
                        source_plugin="homodyne-core",
                    )
                ]

        return []

    def _complete_homodyne_config(
        self, context: CompletionContext
    ) -> list[CompletionResult]:
        """Complete homodyne-config command."""
        options = [
            "--help",
            "--mode",
            "--dataset-size",
            "--output",
            "--sample",
            "--experiment",
            "--author",
        ]

        if context.previous_word == "--mode":
            modes = ["static_isotropic", "static_anisotropic", "laminar_flow"]
            matching = [m for m in modes if m.startswith(context.current_word)]
            if matching:
                return [
                    CompletionResult(
                        completions=matching,
                        completion_type=CompletionType.VALUE,
                        description="Analysis modes",
                        priority=95,
                        source_plugin="homodyne-core",
                    )
                ]

        elif context.previous_word == "--dataset-size":
            sizes = ["small", "standard", "large"]
            matching = [s for s in sizes if s.startswith(context.current_word)]
            if matching:
                return [
                    CompletionResult(
                        completions=matching,
                        completion_type=CompletionType.VALUE,
                        description="Dataset sizes",
                        priority=95,
                        source_plugin="homodyne-core",
                    )
                ]

        # Default to options
        matching = [opt for opt in options if opt.startswith(context.current_word)]
        if matching:
            return [
                CompletionResult(
                    completions=matching,
                    completion_type=CompletionType.OPTION,
                    description="Configuration options",
                    priority=90,
                    source_plugin="homodyne-core",
                )
            ]

        return []

    def _complete_homodyne_gpu(
        self, context: CompletionContext
    ) -> list[CompletionResult]:
        """Complete homodyne-gpu command."""
        options = [
            "--help",
            "--test",
            "--optimize",
            "--status",
            "--install",
            "--uninstall",
        ]
        matching = [opt for opt in options if opt.startswith(context.current_word)]
        if matching:
            return [
                CompletionResult(
                    completions=matching,
                    completion_type=CompletionType.OPTION,
                    description="GPU acceleration options",
                    priority=90,
                    source_plugin="homodyne-core",
                )
            ]
        return []

    def _get_smart_methods(self, context: CompletionContext) -> list[str]:
        """Get smart method suggestions based on config context."""
        default_methods = ["vi", "mcmc", "hybrid"]

        if context.homodyne_config:
            mode = context.homodyne_config.get("mode", "").lower()
            if "static" in mode:
                return ["vi", "hybrid"]  # VI better for static cases
            elif "laminar" in mode:
                return ["mcmc", "hybrid"]  # MCMC better for dynamic cases

        return default_methods

    def _get_config_files(self, context: CompletionContext) -> list[str]:
        """Get available config files."""
        configs = []

        # Add found config files
        for config_file in context.config_files:
            if context.project_root:
                try:
                    relative_path = config_file.relative_to(context.project_root)
                    configs.append(str(relative_path))
                except ValueError:
                    configs.append(str(config_file))
            else:
                configs.append(config_file.name)

        # Add common config names
        common_configs = [
            "config.json",
            "homodyne_config.json",
            "analysis_config.json",
        ]
        configs.extend(common_configs)

        return list(set(configs))


class AliasPlugin(CompletionPlugin):
    """Plugin for completion of homodyne aliases (hmv, hmm, hmh, etc.)."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="homodyne-aliases",
            version="1.0.0",
            description="Completions for homodyne command aliases",
            author="Homodyne Team",
            priority=95,
        )

    def can_complete(self, context: CompletionContext) -> bool:
        """Can complete homodyne aliases."""
        aliases = ["hmv", "hmm", "hmh", "hconfig", "hexp", "hsim", "hm"]
        return context.command in aliases

    def complete(self, context: CompletionContext) -> list[CompletionResult]:
        """Complete alias commands."""
        # Map aliases to their full equivalents
        alias_map = {
            "hmv": "homodyne --method vi",
            "hmm": "homodyne --method mcmc",
            "hmh": "homodyne --method hybrid",
            "hconfig": "homodyne-config",
            "hexp": "homodyne --plot-experimental-data",
            "hsim": "homodyne --plot-simulated-data",
            "hm": "homodyne",
        }

        if context.command in alias_map:
            # Create modified context for the full command
            full_command = alias_map[context.command]
            full_words = full_command.split() + context.words[1:]

            # Create new context with expanded command
            expanded_context = CompletionContext(
                command=full_words[0],
                words=full_words,
                current_word=context.current_word,
                previous_word=context.previous_word,
                cursor_position=context.cursor_position,
                environment_type=context.environment_type,
                environment_path=context.environment_path,
                shell_type=context.shell_type,
                project_root=context.project_root,
                config_files=context.config_files,
                homodyne_config=context.homodyne_config,
            )

            # Use core plugin to complete
            core_plugin = HomodyneCommandPlugin()
            if core_plugin.can_complete(expanded_context):
                results = core_plugin.complete(expanded_context)
                # Mark as coming from alias plugin
                for result in results:
                    result.source_plugin = "homodyne-aliases"
                    result.metadata["alias"] = context.command
                return results

        return []


class ProjectPlugin(CompletionPlugin):
    """Plugin for project-specific completions."""

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="homodyne-project",
            version="1.0.0",
            description="Project-aware completions",
            author="Homodyne Team",
            priority=80,
        )

    def can_complete(self, context: CompletionContext) -> bool:
        """Can complete when in a homodyne project."""
        return context.project_root is not None

    def complete(self, context: CompletionContext) -> list[CompletionResult]:
        """Complete based on project context."""
        results = []

        if context.previous_word == "--output-dir":
            # Suggest project-specific output directories
            output_dirs = self._get_project_output_dirs(context)
            matching = [d for d in output_dirs if d.startswith(context.current_word)]
            if matching:
                results.append(
                    CompletionResult(
                        completions=matching,
                        completion_type=CompletionType.DIRECTORY,
                        description="Project output directories",
                        priority=85,
                        source_plugin="homodyne-project",
                    )
                )

        elif context.current_word.endswith(".json") or not context.current_word:
            # Smart config file suggestions
            smart_configs = self._get_smart_config_suggestions(context)
            matching = [c for c in smart_configs if c.startswith(context.current_word)]
            if matching:
                results.append(
                    CompletionResult(
                        completions=matching,
                        completion_type=CompletionType.CONFIG,
                        description="Smart config suggestions",
                        priority=85,
                        source_plugin="homodyne-project",
                        metadata={"smart": True},
                    )
                )

        return results

    def _get_project_output_dirs(self, context: CompletionContext) -> list[str]:
        """Get project-specific output directory suggestions."""
        if not context.project_root:
            return []

        dirs = ["results", "output", "analysis", "plots", "data"]
        existing_dirs = []

        try:
            for d in context.project_root.iterdir():
                if d.is_dir() and d.name.lower() in [
                    "results",
                    "output",
                    "analysis",
                    "plots",
                ]:
                    existing_dirs.append(f"./{d.name}")
        except Exception:
            pass

        # Prefer existing directories
        return existing_dirs + [f"./{d}" for d in dirs if f"./{d}" not in existing_dirs]

    def _get_smart_config_suggestions(self, context: CompletionContext) -> list[str]:
        """Get smart config file suggestions based on project structure."""
        suggestions = []

        if context.homodyne_config:
            # Current config suggests similar configs
            mode = context.homodyne_config.get("mode", "")
            if mode:
                suggestions.append(f"{mode}_config.json")

        # Add recent configs
        for config_file in context.config_files[:3]:  # Top 3 most relevant
            try:
                if context.project_root:
                    rel_path = config_file.relative_to(context.project_root)
                    suggestions.append(str(rel_path))
                else:
                    suggestions.append(config_file.name)
            except ValueError:
                suggestions.append(config_file.name)

        return list(set(suggestions))


class PluginManager:
    """
    Manages completion plugins with dynamic loading and priority-based execution.

    Features:
    - Plugin discovery and loading
    - Priority-based plugin execution
    - Plugin enabling/disabling
    - Performance monitoring
    """

    def __init__(self):
        self._plugins: dict[str, CompletionPlugin] = {}
        self._plugin_order: list[str] = []
        self._lock = threading.Lock()
        self._stats = {
            "plugin_calls": {},
            "plugin_errors": {},
            "plugin_timing": {},
        }

        # Load built-in plugins
        self._load_builtin_plugins()

    def _load_builtin_plugins(self) -> None:
        """Load built-in completion plugins."""
        builtin_plugins = [
            HomodyneCommandPlugin(),
            AliasPlugin(),
            ProjectPlugin(),
        ]

        for plugin in builtin_plugins:
            self.register_plugin(plugin)

    def register_plugin(self, plugin: CompletionPlugin) -> bool:
        """
        Register a completion plugin.

        Args:
            plugin: Plugin instance to register

        Returns:
            True if registration successful, False otherwise
        """
        try:
            if not plugin.initialize():
                return False

            with self._lock:
                plugin_name = plugin.info.name
                self._plugins[plugin_name] = plugin

                # Insert in priority order
                priority = plugin.info.priority
                inserted = False
                for i, existing_name in enumerate(self._plugin_order):
                    if self._plugins[existing_name].info.priority < priority:
                        self._plugin_order.insert(i, plugin_name)
                        inserted = True
                        break

                if not inserted:
                    self._plugin_order.append(plugin_name)

                # Initialize stats
                self._stats["plugin_calls"][plugin_name] = 0
                self._stats["plugin_errors"][plugin_name] = 0
                self._stats["plugin_timing"][plugin_name] = []

            return True

        except Exception:
            return False

    def unregister_plugin(self, plugin_name: str) -> bool:
        """
        Unregister a completion plugin.

        Args:
            plugin_name: Name of plugin to unregister

        Returns:
            True if unregistration successful, False otherwise
        """
        with self._lock:
            if plugin_name in self._plugins:
                plugin = self._plugins[plugin_name]
                try:
                    plugin.cleanup()
                except Exception:
                    pass

                del self._plugins[plugin_name]
                self._plugin_order.remove(plugin_name)

                # Clean up stats
                for stat_dict in self._stats.values():
                    stat_dict.pop(plugin_name, None)

                return True

        return False

    def get_completions(self, context: CompletionContext) -> list[CompletionResult]:
        """
        Get completions from all enabled plugins.

        Args:
            context: Completion context

        Returns:
            Aggregated completion results from all plugins
        """
        all_results = []

        with self._lock:
            plugins_to_use = [
                (name, self._plugins[name]) for name in self._plugin_order
            ]

        for plugin_name, plugin in plugins_to_use:
            if not plugin.enabled:
                continue

            try:
                start_time = time.perf_counter()

                if plugin.can_complete(context):
                    results = plugin.complete(context)
                    all_results.extend(results)

                    # Update stats
                    execution_time = (time.perf_counter() - start_time) * 1000
                    self._stats["plugin_calls"][plugin_name] += 1
                    self._stats["plugin_timing"][plugin_name].append(execution_time)

            except Exception:
                self._stats["plugin_errors"][plugin_name] += 1
                continue

        # Sort by priority and remove duplicates
        unique_results = {}
        for result in all_results:
            key = "|".join(result.completions)
            if (
                key not in unique_results
                or result.priority > unique_results[key].priority
            ):
                unique_results[key] = result

        return sorted(unique_results.values(), key=lambda r: -r.priority)

    def list_plugins(self) -> list[PluginInfo]:
        """Get list of registered plugins."""
        with self._lock:
            return [plugin.info for plugin in self._plugins.values()]

    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin."""
        with self._lock:
            if plugin_name in self._plugins:
                self._plugins[plugin_name].enabled = True
                return True
        return False

    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin."""
        with self._lock:
            if plugin_name in self._plugins:
                self._plugins[plugin_name].enabled = False
                return True
        return False

    def get_statistics(self) -> dict[str, Any]:
        """Get plugin manager statistics."""
        with self._lock:
            stats = {}
            for plugin_name in self._plugins:
                calls = self._stats["plugin_calls"][plugin_name]
                errors = self._stats["plugin_errors"][plugin_name]
                timings = self._stats["plugin_timing"][plugin_name]

                avg_time = sum(timings) / len(timings) if timings else 0
                error_rate = errors / calls if calls > 0 else 0

                stats[plugin_name] = {
                    "calls": calls,
                    "errors": errors,
                    "error_rate": error_rate,
                    "average_time_ms": avg_time,
                    "enabled": self._plugins[plugin_name].enabled,
                    "priority": self._plugins[plugin_name].info.priority,
                }

        return stats

    def reload_plugins(self) -> None:
        """Reload all plugins."""
        with self._lock:
            # Cleanup existing plugins
            for plugin in self._plugins.values():
                try:
                    plugin.cleanup()
                except Exception:
                    pass

            self._plugins.clear()
            self._plugin_order.clear()

            # Reset stats
            for stat_dict in self._stats.values():
                stat_dict.clear()

        # Reload built-in plugins
        self._load_builtin_plugins()


# Global plugin manager instance
_plugin_manager: PluginManager | None = None
_manager_lock = threading.Lock()


def get_plugin_manager() -> PluginManager:
    """Get global plugin manager instance (singleton pattern)."""
    global _plugin_manager

    with _manager_lock:
        if _plugin_manager is None:
            _plugin_manager = PluginManager()

    return _plugin_manager
