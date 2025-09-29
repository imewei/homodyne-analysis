"""
Intelligent Completion Cache System
===================================

High-performance caching system with environment isolation,
smart invalidation, and background optimization.
"""

import hashlib
import json
import pickle
import sqlite3
import threading
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .core import CompletionContext
    from .core import CompletionResult
except ImportError:
    from core import CompletionContext
    from core import CompletionResult


@dataclass
class CacheConfig:
    """Configuration for completion cache system."""

    # Cache limits
    max_entries: int = 10000
    max_memory_mb: int = 50
    default_ttl_seconds: int = 300  # 5 minutes

    # Performance settings
    cleanup_interval_seconds: int = 3600  # 1 hour
    max_key_length: int = 1000
    enable_persistence: bool = True
    enable_compression: bool = True

    # Environment isolation
    isolate_by_environment: bool = True
    isolate_by_project: bool = True
    share_system_completions: bool = True


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""

    key: str
    results: list[CompletionResult]
    timestamp: float
    ttl: int
    access_count: int = 0
    last_access: float = 0.0
    size_bytes: int = 0
    environment_path: str | None = None
    project_path: str | None = None


class CompletionCache:
    """
    High-performance completion cache with intelligent invalidation.

    Features:
    - Environment-isolated caching
    - Project-aware cache scoping
    - Smart cache invalidation
    - Background cleanup and optimization
    - Persistent storage with SQLite
    - Memory usage monitoring
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        config: CacheConfig | None = None,
    ):
        self.config = config or CacheConfig()

        # Set up cache directory
        if cache_dir is None:
            cache_dir = self._get_default_cache_dir()
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._memory_cache: dict[str, CacheEntry] = {}
        self._cache_lock = threading.RWLock()

        # Persistent storage
        self._db_path = self.cache_dir / "completions.db"
        self._db_lock = threading.Lock()
        self._init_database()

        # Background tasks
        self._cleanup_timer: threading.Timer | None = None
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "cleanup_runs": 0,
            "memory_usage_mb": 0,
        }

        # Start background cleanup
        self._schedule_cleanup()

    def _get_default_cache_dir(self) -> Path:
        """Get default cache directory."""
        import os

        if xdg_cache := os.environ.get("XDG_CACHE_HOME"):
            return Path(xdg_cache) / "homodyne" / "completion"

        home = Path.home()
        return home / ".cache" / "homodyne" / "completion"

    def _init_database(self) -> None:
        """Initialize SQLite database for persistent storage."""
        if not self.config.enable_persistence:
            return

        with self._db_lock:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS completion_cache (
                        key TEXT PRIMARY KEY,
                        results BLOB NOT NULL,
                        timestamp REAL NOT NULL,
                        ttl INTEGER NOT NULL,
                        access_count INTEGER DEFAULT 0,
                        last_access REAL DEFAULT 0,
                        size_bytes INTEGER DEFAULT 0,
                        environment_path TEXT,
                        project_path TEXT
                    )
                """
                )

                # Create indexes for performance
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_timestamp ON completion_cache(timestamp)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_environment ON completion_cache(environment_path)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_project ON completion_cache(project_path)"
                )

                conn.commit()
            finally:
                conn.close()

    def get(self, context: CompletionContext) -> list[CompletionResult] | None:
        """
        Get cached completion results for context.

        Args:
            context: Completion context

        Returns:
            Cached results if valid and available, None otherwise
        """
        cache_key = self._generate_cache_key(context)
        current_time = time.time()

        # Try memory cache first
        with self._cache_lock.read_lock():
            if cache_key in self._memory_cache:
                entry = self._memory_cache[cache_key]

                # Check if still valid
                if current_time - entry.timestamp <= entry.ttl:
                    # Update access stats
                    entry.access_count += 1
                    entry.last_access = current_time
                    self._stats["hits"] += 1
                    return entry.results
                # Expired, will be cleaned up later

        # Try persistent cache
        if self.config.enable_persistence:
            if cached_results := self._get_from_database(cache_key, current_time):
                # Add to memory cache for faster access
                entry = CacheEntry(
                    key=cache_key,
                    results=cached_results,
                    timestamp=current_time,
                    ttl=self.config.default_ttl_seconds,
                    access_count=1,
                    last_access=current_time,
                    environment_path=str(context.environment_path),
                    project_path=(
                        str(context.project_root) if context.project_root else None
                    ),
                )

                with self._cache_lock.write_lock():
                    self._memory_cache[cache_key] = entry

                self._stats["hits"] += 1
                return cached_results

        self._stats["misses"] += 1
        return None

    def put(
        self,
        context: CompletionContext,
        results: list[CompletionResult],
        ttl: int | None = None,
    ) -> None:
        """
        Cache completion results for context.

        Args:
            context: Completion context
            results: Completion results to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        if not results:
            return

        cache_key = self._generate_cache_key(context)
        ttl = ttl or self.config.default_ttl_seconds
        current_time = time.time()

        # Calculate entry size
        size_bytes = (
            len(pickle.dumps(results))
            if self.config.enable_compression
            else len(str(results))
        )

        entry = CacheEntry(
            key=cache_key,
            results=results,
            timestamp=current_time,
            ttl=ttl,
            access_count=1,
            last_access=current_time,
            size_bytes=size_bytes,
            environment_path=str(context.environment_path),
            project_path=str(context.project_root) if context.project_root else None,
        )

        # Add to memory cache
        with self._cache_lock.write_lock():
            self._memory_cache[cache_key] = entry

            # Check memory limits
            self._enforce_memory_limits()

        # Add to persistent cache
        if self.config.enable_persistence:
            self._save_to_database(entry)

    def invalidate_environment(self, environment_path: Path) -> int:
        """
        Invalidate all cache entries for a specific environment.

        Args:
            environment_path: Path to virtual environment

        Returns:
            Number of entries invalidated
        """
        env_str = str(environment_path)
        invalidated = 0

        # Invalidate memory cache
        with self._cache_lock.write_lock():
            keys_to_remove = [
                key
                for key, entry in self._memory_cache.items()
                if entry.environment_path == env_str
            ]
            for key in keys_to_remove:
                del self._memory_cache[key]
                invalidated += 1

        # Invalidate persistent cache
        if self.config.enable_persistence:
            with self._db_lock:
                conn = sqlite3.connect(self._db_path)
                try:
                    cursor = conn.execute(
                        "DELETE FROM completion_cache WHERE environment_path = ?",
                        (env_str,),
                    )
                    invalidated += cursor.rowcount
                    conn.commit()
                finally:
                    conn.close()

        return invalidated

    def invalidate_project(self, project_path: Path) -> int:
        """
        Invalidate all cache entries for a specific project.

        Args:
            project_path: Path to project root

        Returns:
            Number of entries invalidated
        """
        project_str = str(project_path)
        invalidated = 0

        # Invalidate memory cache
        with self._cache_lock.write_lock():
            keys_to_remove = [
                key
                for key, entry in self._memory_cache.items()
                if entry.project_path == project_str
            ]
            for key in keys_to_remove:
                del self._memory_cache[key]
                invalidated += 1

        # Invalidate persistent cache
        if self.config.enable_persistence:
            with self._db_lock:
                conn = sqlite3.connect(self._db_path)
                try:
                    cursor = conn.execute(
                        "DELETE FROM completion_cache WHERE project_path = ?",
                        (project_str,),
                    )
                    invalidated += cursor.rowcount
                    conn.commit()
                finally:
                    conn.close()

        return invalidated

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._cache_lock.write_lock():
            self._memory_cache.clear()

        if self.config.enable_persistence:
            with self._db_lock:
                conn = sqlite3.connect(self._db_path)
                try:
                    conn.execute("DELETE FROM completion_cache")
                    conn.commit()
                finally:
                    conn.close()

    def cleanup(self) -> int:
        """
        Clean up expired cache entries.

        Returns:
            Number of entries cleaned up
        """
        current_time = time.time()
        cleaned = 0

        # Clean memory cache
        with self._cache_lock.write_lock():
            expired_keys = [
                key
                for key, entry in self._memory_cache.items()
                if current_time - entry.timestamp > entry.ttl
            ]
            for key in expired_keys:
                del self._memory_cache[key]
                cleaned += 1

        # Clean persistent cache
        if self.config.enable_persistence:
            with self._db_lock:
                conn = sqlite3.connect(self._db_path)
                try:
                    cursor = conn.execute(
                        "DELETE FROM completion_cache WHERE timestamp + ttl < ?",
                        (current_time,),
                    )
                    cleaned += cursor.rowcount
                    conn.commit()
                finally:
                    conn.close()

        self._stats["cleanup_runs"] += 1
        return cleaned

    def get_statistics(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        with self._cache_lock.read_lock():
            memory_entries = len(self._memory_cache)
            memory_size_mb = (
                sum(entry.size_bytes for entry in self._memory_cache.values())
                / 1024
                / 1024
            )

        persistent_entries = 0
        if self.config.enable_persistence:
            with self._db_lock:
                conn = sqlite3.connect(self._db_path)
                try:
                    cursor = conn.execute("SELECT COUNT(*) FROM completion_cache")
                    persistent_entries = cursor.fetchone()[0]
                finally:
                    conn.close()

        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0

        return {
            "memory_entries": memory_entries,
            "persistent_entries": persistent_entries,
            "memory_size_mb": memory_size_mb,
            "hit_rate": hit_rate,
            "total_hits": self._stats["hits"],
            "total_misses": self._stats["misses"],
            "evictions": self._stats["evictions"],
            "cleanup_runs": self._stats["cleanup_runs"],
        }

    def _generate_cache_key(self, context: CompletionContext) -> str:
        """Generate cache key for completion context."""
        # Build key components based on isolation settings
        key_parts = [
            context.command,
            "|".join(context.words),
            context.current_word,
            context.previous_word,
        ]

        if self.config.isolate_by_environment:
            key_parts.append(f"env:{context.environment_path}")

        if self.config.isolate_by_project and context.project_root:
            key_parts.append(f"proj:{context.project_root}")

        # Add config file hash for context sensitivity
        if context.homodyne_config:
            config_hash = hashlib.md5(
                json.dumps(context.homodyne_config, sort_keys=True).encode()
            ).hexdigest()[:8]
            key_parts.append(f"cfg:{config_hash}")

        key = "|".join(key_parts)

        # Ensure key length limit
        if len(key) > self.config.max_key_length:
            key_hash = hashlib.sha256(key.encode()).hexdigest()
            key = f"hash:{key_hash}"

        return key

    def _get_from_database(
        self, cache_key: str, current_time: float
    ) -> list[CompletionResult] | None:
        """Get entry from persistent database."""
        if not self.config.enable_persistence:
            return None

        with self._db_lock:
            conn = sqlite3.connect(self._db_path)
            try:
                cursor = conn.execute(
                    """
                    SELECT results, timestamp, ttl, access_count
                    FROM completion_cache
                    WHERE key = ? AND timestamp + ttl >= ?
                """,
                    (cache_key, current_time),
                )

                row = cursor.fetchone()
                if row:
                    results_blob, _timestamp, _ttl, access_count = row

                    # Deserialize results
                    if self.config.enable_compression:
                        results = pickle.loads(results_blob)
                    else:
                        results = json.loads(results_blob.decode())

                    # Update access count
                    conn.execute(
                        """
                        UPDATE completion_cache
                        SET access_count = ?, last_access = ?
                        WHERE key = ?
                    """,
                        (access_count + 1, current_time, cache_key),
                    )
                    conn.commit()

                    return results

            except Exception:
                return None
            finally:
                conn.close()

        return None

    def _save_to_database(self, entry: CacheEntry) -> None:
        """Save entry to persistent database."""
        if not self.config.enable_persistence:
            return

        try:
            # Serialize results
            if self.config.enable_compression:
                results_blob = pickle.dumps(entry.results)
            else:
                results_blob = json.dumps([asdict(r) for r in entry.results]).encode()

            with self._db_lock:
                conn = sqlite3.connect(self._db_path)
                try:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO completion_cache
                        (key, results, timestamp, ttl, access_count, last_access,
                         size_bytes, environment_path, project_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            entry.key,
                            results_blob,
                            entry.timestamp,
                            entry.ttl,
                            entry.access_count,
                            entry.last_access,
                            entry.size_bytes,
                            entry.environment_path,
                            entry.project_path,
                        ),
                    )
                    conn.commit()
                finally:
                    conn.close()

        except Exception:
            pass  # Fail silently for cache operations

    def _enforce_memory_limits(self) -> None:
        """Enforce memory cache size limits."""
        current_size_mb = (
            sum(entry.size_bytes for entry in self._memory_cache.values()) / 1024 / 1024
        )

        if (
            len(self._memory_cache) > self.config.max_entries
            or current_size_mb > self.config.max_memory_mb
        ):
            # Sort by access frequency and age (LRU with frequency consideration)
            sorted_entries = sorted(
                self._memory_cache.items(),
                key=lambda x: (x[1].access_count, x[1].last_access),
            )

            # Remove least valuable entries
            entries_to_remove = max(1, len(sorted_entries) // 4)  # Remove 25%
            for key, _ in sorted_entries[:entries_to_remove]:
                del self._memory_cache[key]
                self._stats["evictions"] += 1

    def _schedule_cleanup(self) -> None:
        """Schedule background cleanup task."""

        def cleanup_task():
            try:
                self.cleanup()
            except Exception:
                pass
            finally:
                # Reschedule
                self._schedule_cleanup()

        self._cleanup_timer = threading.Timer(
            self.config.cleanup_interval_seconds, cleanup_task
        )
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()

    def __del__(self):
        """Cleanup when cache is destroyed."""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()


# Thread-safe read-write lock implementation
class RWLock:
    """Read-write lock implementation."""

    def __init__(self):
        self._read_ready = threading.Condition(threading.RLock())
        self._readers = 0

    def read_lock(self):
        return _RLockContext(self, "read")

    def write_lock(self):
        return _RLockContext(self, "write")

    def acquire_read(self):
        self._read_ready.acquire()
        try:
            self._readers += 1
        finally:
            self._read_ready.release()

    def release_read(self):
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notifyAll()
        finally:
            self._read_ready.release()

    def acquire_write(self):
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self):
        self._read_ready.release()


class _RLockContext:
    """Context manager for read-write locks."""

    def __init__(self, lock, mode):
        self.lock = lock
        self.mode = mode

    def __enter__(self):
        if self.mode == "read":
            self.lock.acquire_read()
        else:
            self.lock.acquire_write()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mode == "read":
            self.lock.release_read()
        else:
            self.lock.release_write()


# Monkey patch threading module
threading.RWLock = RWLock
