"""
Core Completion Engine
======================

Advanced completion engine with plugin architecture, intelligent caching,
and environment-aware context detection.
"""

import os
import sys
import threading
import time
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import json

    import yaml
except ImportError:
    yaml = None


class CompletionType(Enum):
    """Types of completions supported by the system."""

    COMMAND = "command"
    OPTION = "option"
    ARGUMENT = "argument"
    FILE = "file"
    DIRECTORY = "directory"
    CONFIG = "config"
    METHOD = "method"
    VALUE = "value"


class EnvironmentType(Enum):
    """Types of virtual environments detected."""

    CONDA = "conda"
    MAMBA = "mamba"
    VENV = "venv"
    VIRTUALENV = "virtualenv"
    POETRY = "poetry"
    PIPENV = "pipenv"
    SYSTEM = "system"


@dataclass
class CompletionResult:
    """Result of a completion operation."""

    completions: list[str]
    completion_type: CompletionType
    description: str | None = None
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    source_plugin: str | None = None
    cache_ttl: int = 300  # 5 minutes default


@dataclass
class CompletionContext:
    """Context information for completion generation."""

    # Shell command context
    command: str
    words: list[str]
    current_word: str
    previous_word: str
    cursor_position: int

    # Environment context
    environment_type: EnvironmentType
    environment_path: Path
    shell_type: str

    # Project context
    project_root: Path | None = None
    config_files: list[Path] = field(default_factory=list)
    homodyne_config: dict[str, Any] | None = None

    # Performance context
    max_completions: int = 50
    timeout_ms: int = 1000
    enable_caching: bool = True

    @classmethod
    def from_shell_args(
        cls, argv: list[str], shell_type: str = "bash"
    ) -> "CompletionContext":
        """Create context from shell completion arguments."""
        current_word = argv[-1] if argv else ""
        previous_word = argv[-2] if len(argv) > 1 else ""

        # Detect environment
        env_type, env_path = cls._detect_environment()

        # Detect project context
        project_root = cls._find_project_root()
        config_files = cls._find_config_files(project_root)
        homodyne_config = cls._load_homodyne_config(config_files)

        return cls(
            command=argv[0] if argv else "",
            words=argv,
            current_word=current_word,
            previous_word=previous_word,
            cursor_position=len(current_word),
            environment_type=env_type,
            environment_path=env_path,
            shell_type=shell_type,
            project_root=project_root,
            config_files=config_files,
            homodyne_config=homodyne_config,
        )

    @staticmethod
    def _detect_environment() -> tuple[EnvironmentType, Path]:
        """Detect current virtual environment type and path."""
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

    @staticmethod
    def _find_project_root(start_dir: Path | None = None) -> Path | None:
        """Find project root by looking for common project markers."""
        if start_dir is None:
            start_dir = Path.cwd()

        markers = [
            ".git",
            "pyproject.toml",
            "setup.py",
            "requirements.txt",
            "environment.yml",
            "homodyne_config.json",
            "config.json",
        ]

        current = start_dir.resolve()
        while current != current.parent:
            if any((current / marker).exists() for marker in markers):
                return current
            current = current.parent

        return None

    @staticmethod
    def _find_config_files(project_root: Path | None = None) -> list[Path]:
        """Find homodyne configuration files in project."""
        if project_root is None:
            search_dirs = [Path.cwd()]
        else:
            search_dirs = [
                project_root,
                project_root / "configs",
                project_root / "config",
            ]

        config_files = []
        config_patterns = [
            "*homodyne*.json",
            "*homodyne*.yaml",
            "*homodyne*.yml",
            "config.json",
            "analysis_config.json",
        ]

        for search_dir in search_dirs:
            if search_dir.exists():
                for pattern in config_patterns:
                    config_files.extend(search_dir.glob(pattern))

        return sorted(set(config_files))

    @staticmethod
    def _load_homodyne_config(config_files: list[Path]) -> dict[str, Any] | None:
        """Load and parse the first valid homodyne config file."""
        for config_file in config_files:
            try:
                with open(config_file) as f:
                    if config_file.suffix in [".yaml", ".yml"] and yaml:
                        return yaml.safe_load(f)
                    return json.load(f)
            except Exception:
                continue
        return None


class CompletionEngine:
    """
    Advanced completion engine with plugin architecture and intelligent caching.

    Features:
    - Plugin-based completion modules
    - Environment-isolated caching
    - Project-aware context detection
    - Performance optimization
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        enable_caching: bool = True,
        max_cache_size: int = 1000,
        background_warming: bool = True,
    ):
        self.enable_caching = enable_caching
        self.max_cache_size = max_cache_size
        self.background_warming = background_warming

        # Set up cache directory
        if cache_dir is None:
            cache_dir = self._get_default_cache_dir()
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Plugin system
        self.plugins: dict[str, Any] = {}  # Will be proper plugins when implemented
        self._plugin_lock = threading.Lock()

        # Cache system
        self._completion_cache: dict[str, tuple[CompletionResult, float]] = {}
        self._cache_lock = threading.Lock()

        # Performance tracking
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "completion_time_ms": [],
            "plugin_usage": {},
        }

    def _get_default_cache_dir(self) -> Path:
        """Get default cache directory based on environment."""
        if xdg_cache := os.environ.get("XDG_CACHE_HOME"):
            return Path(xdg_cache) / "homodyne" / "completion"

        home = Path.home()

        # Environment-specific cache directories
        if conda_env := os.environ.get("CONDA_DEFAULT_ENV"):
            return home / ".cache" / "homodyne" / "completion" / f"conda-{conda_env}"
        if virtual_env := os.environ.get("VIRTUAL_ENV"):
            env_name = Path(virtual_env).name
            return home / ".cache" / "homodyne" / "completion" / f"venv-{env_name}"

        return home / ".cache" / "homodyne" / "completion" / "system"

    def complete(
        self,
        context: CompletionContext,
        use_cache: bool = True,
    ) -> list[CompletionResult]:
        """
        Generate completions for the given context.

        Args:
            context: Completion context with command, words, environment info
            use_cache: Whether to use cached results

        Returns:
            List of completion results sorted by priority and relevance
        """
        start_time = time.perf_counter()

        try:
            # Generate cache key
            cache_key = self._generate_cache_key(context)

            # Check cache first
            if use_cache and self.enable_caching:
                if cached_result := self._get_cached_completion(cache_key):
                    self._stats["cache_hits"] += 1
                    return cached_result

            self._stats["cache_misses"] += 1

            # Generate completions
            results = self._generate_completions(context)

            # Cache results
            if self.enable_caching and results:
                self._cache_completion(cache_key, results)

            # Track performance
            completion_time = (time.perf_counter() - start_time) * 1000
            self._stats["completion_time_ms"].append(completion_time)

            return results

        except Exception:
            # Fallback to basic completion on error
            return self._fallback_completion(context)

    def _generate_completions(
        self, context: CompletionContext
    ) -> list[CompletionResult]:
        """Generate completions using registered plugins."""
        all_results = []

        # Basic command completion
        if not context.words or len(context.words) == 1:
            all_results.extend(self._complete_commands(context))
        elif context.previous_word.startswith("-"):
            all_results.extend(self._complete_options(context))
        else:
            all_results.extend(self._complete_arguments(context))

        # Sort by priority and relevance
        all_results.sort(
            key=lambda r: (-r.priority, r.completions[0] if r.completions else "")
        )

        # Limit results
        return all_results[: context.max_completions]

    def _complete_commands(self, context: CompletionContext) -> list[CompletionResult]:
        """Complete main commands."""
        commands = ["homodyne", "homodyne-config", "homodyne-gpu"]
        matching = [cmd for cmd in commands if cmd.startswith(context.current_word)]

        if matching:
            return [
                CompletionResult(
                    completions=matching,
                    completion_type=CompletionType.COMMAND,
                    description="Homodyne analysis commands",
                    priority=100,
                )
            ]
        return []

    def _complete_options(self, context: CompletionContext) -> list[CompletionResult]:
        """Complete command options based on previous option."""
        option_completions = {
            "--method": self._complete_methods(context),
            "--config": self._complete_config_files(context),
            "--output-dir": self._complete_directories(context),
            "--mode": ["static_isotropic", "static_anisotropic", "laminar_flow"],
            "--dataset-size": ["small", "standard", "large"],
        }

        if context.previous_word in option_completions:
            completions = option_completions[context.previous_word]
            if isinstance(completions, list):
                matching = [
                    c for c in completions if c.startswith(context.current_word)
                ]
                if matching:
                    return [
                        CompletionResult(
                            completions=matching,
                            completion_type=CompletionType.ARGUMENT,
                            description=f"Values for {context.previous_word}",
                            priority=90,
                        )
                    ]

        return []

    def _complete_methods(self, context: CompletionContext) -> list[str]:
        """Complete analysis methods based on config context."""
        default_methods = ["vi", "mcmc", "hybrid"]

        # Smart method suggestion based on config
        if context.homodyne_config:
            mode = context.homodyne_config.get("mode", "")
            if "static" in mode.lower():
                return ["vi", "hybrid"]
            if "laminar" in mode.lower():
                return ["mcmc", "hybrid"]

        return default_methods

    def _complete_config_files(self, context: CompletionContext) -> list[str]:
        """Complete configuration files from project context."""
        completions = []

        # Add found config files
        for config_file in context.config_files:
            if context.project_root:
                try:
                    relative_path = config_file.relative_to(context.project_root)
                    completions.append(str(relative_path))
                except ValueError:
                    completions.append(str(config_file))
            else:
                completions.append(config_file.name)

        # Add common config names
        common_configs = [
            "config.json",
            "homodyne_config.json",
            "analysis_config.json",
        ]
        completions.extend(common_configs)

        return list(set(completions))

    def _complete_directories(self, context: CompletionContext) -> list[str]:
        """Complete directory paths with smart suggestions."""
        suggestions = ["./results", "./output", "./analysis", "./data"]

        # Add existing directories
        try:
            current_dir = Path.cwd()
            existing_dirs = [d.name for d in current_dir.iterdir() if d.is_dir()]
            suggestions.extend(existing_dirs)
        except Exception:
            pass

        return list(set(suggestions))

    def _complete_arguments(self, context: CompletionContext) -> list[CompletionResult]:
        """Complete general arguments."""
        # Default to config file completion
        config_completions = self._complete_config_files(context)
        matching = [c for c in config_completions if c.startswith(context.current_word)]

        if matching:
            return [
                CompletionResult(
                    completions=matching,
                    completion_type=CompletionType.CONFIG,
                    description="Configuration files",
                    priority=70,
                )
            ]

        return []

    def _generate_cache_key(self, context: CompletionContext) -> str:
        """Generate cache key for completion context."""
        key_parts = [
            context.command,
            "|".join(context.words),
            context.current_word,
            context.previous_word,
            str(context.environment_path),
            str(context.project_root),
        ]
        return "|".join(key_parts)

    def _get_cached_completion(self, cache_key: str) -> list[CompletionResult] | None:
        """Get cached completion if valid."""
        with self._cache_lock:
            if cache_key in self._completion_cache:
                result, timestamp = self._completion_cache[cache_key]
                if time.time() - timestamp < result.cache_ttl:
                    return [result]
                del self._completion_cache[cache_key]
        return None

    def _cache_completion(
        self, cache_key: str, results: list[CompletionResult]
    ) -> None:
        """Cache completion results."""
        if not results:
            return

        with self._cache_lock:
            # Use first result for caching (most relevant)
            self._completion_cache[cache_key] = (results[0], time.time())

            # Maintain cache size
            if len(self._completion_cache) > self.max_cache_size:
                # Remove oldest entries
                oldest_keys = sorted(
                    self._completion_cache.keys(),
                    key=lambda k: self._completion_cache[k][1],
                )[: len(self._completion_cache) - self.max_cache_size + 1]

                for key in oldest_keys:
                    del self._completion_cache[key]

    def _fallback_completion(
        self, context: CompletionContext
    ) -> list[CompletionResult]:
        """Fallback completion when main system fails."""
        return [
            CompletionResult(
                completions=["--help"],
                completion_type=CompletionType.OPTION,
                description="Show help information",
                priority=0,
            )
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get completion engine performance statistics."""
        with self._cache_lock:
            cache_size = len(self._completion_cache)
            cache_hit_rate = (
                self._stats["cache_hits"]
                / (self._stats["cache_hits"] + self._stats["cache_misses"])
                if self._stats["cache_hits"] + self._stats["cache_misses"] > 0
                else 0
            )

        avg_completion_time = (
            sum(self._stats["completion_time_ms"])
            / len(self._stats["completion_time_ms"])
            if self._stats["completion_time_ms"]
            else 0
        )

        return {
            "cache_size": cache_size,
            "cache_hit_rate": cache_hit_rate,
            "average_completion_time_ms": avg_completion_time,
            "total_completions": self._stats["cache_hits"]
            + self._stats["cache_misses"],
            "plugin_count": len(self.plugins),
        }

    def clear_cache(self) -> None:
        """Clear completion cache."""
        with self._cache_lock:
            self._completion_cache.clear()

    def warm_cache(self, common_contexts: list[CompletionContext]) -> None:
        """Pre-warm cache with common completion contexts."""
        if not self.background_warming:
            return

        def _warm_worker():
            for context in common_contexts:
                try:
                    self.complete(context, use_cache=False)
                except Exception:
                    continue

        warming_thread = threading.Thread(target=_warm_worker, daemon=True)
        warming_thread.start()


# Export all main classes and types
__all__ = [
    "CompletionContext",
    "CompletionEngine",
    "CompletionResult",
    "CompletionType",
    "EnvironmentType",
]
