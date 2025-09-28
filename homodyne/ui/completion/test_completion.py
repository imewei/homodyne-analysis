#!/usr/bin/env python3
"""
Homodyne Advanced Completion System Test Suite
==============================================

Comprehensive test suite for validating the upgraded completion system
including unit tests, integration tests, and performance benchmarks.

Usage:
    python test_completion.py [options]
    homodyne-test-completion [options]

Examples:
    # Run all tests
    python test_completion.py

    # Run specific test categories
    python test_completion.py --unit --integration

    # Run performance benchmarks
    python test_completion.py --benchmark

    # Test with verbose output
    python test_completion.py --verbose
"""

import argparse
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

from .cache import CacheConfig, CompletionCache
from .core import CompletionContext, CompletionEngine, CompletionType, EnvironmentType
from .installer import CompletionInstaller, InstallationConfig
from .plugins import (
    AliasPlugin,
    HomodyneCommandPlugin,
    PluginManager,
    get_plugin_manager,
)


class TestCompletionCore(unittest.TestCase):
    """Test core completion engine functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = CompletionEngine(enable_caching=False)

    def test_completion_context_creation(self):
        """Test completion context creation from shell args."""
        args = ["homodyne", "--method", "vi", "config.json"]
        context = CompletionContext.from_shell_args(args, "bash")

        self.assertEqual(context.command, "homodyne")
        self.assertEqual(context.words, args)
        self.assertEqual(context.current_word, "config.json")
        self.assertEqual(context.previous_word, "vi")
        self.assertEqual(context.shell_type, "bash")

    def test_basic_completion(self):
        """Test basic completion functionality."""
        context = CompletionContext(
            command="homodyne",
            words=["homodyne"],
            current_word="",
            previous_word="homodyne",
            cursor_position=0,
            environment_type=EnvironmentType.CONDA,
            environment_path=Path("/test/env"),
            shell_type="bash",
        )

        results = self.engine.complete(context)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0].completions, list)

    def test_environment_detection(self):
        """Test environment type detection."""
        env_type, env_path = CompletionContext._detect_environment()
        self.assertIsInstance(env_type, EnvironmentType)
        self.assertIsInstance(env_path, Path)

    def test_project_detection(self):
        """Test project root detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a mock project structure
            (temp_path / ".git").mkdir()
            (temp_path / "pyproject.toml").touch()

            project_root = CompletionContext._find_project_root(temp_path)
            self.assertEqual(project_root, temp_path)


class TestCompletionPlugins(unittest.TestCase):
    """Test completion plugin system."""

    def setUp(self):
        """Set up test fixtures."""
        self.plugin_manager = PluginManager()

    def test_homodyne_command_plugin(self):
        """Test core homodyne command plugin."""
        plugin = HomodyneCommandPlugin()
        self.assertTrue(plugin.enabled)
        self.assertEqual(plugin.info.name, "homodyne-core")

        # Test can_complete
        context = CompletionContext(
            command="homodyne",
            words=["homodyne", "--method"],
            current_word="",
            previous_word="--method",
            cursor_position=0,
            environment_type=EnvironmentType.CONDA,
            environment_path=Path("/test/env"),
            shell_type="bash",
        )

        self.assertTrue(plugin.can_complete(context))
        results = plugin.complete(context)
        self.assertGreater(len(results), 0)

    def test_alias_plugin(self):
        """Test alias completion plugin."""
        plugin = AliasPlugin()
        self.assertTrue(plugin.enabled)

        # Test alias completion
        context = CompletionContext(
            command="hmv",
            words=["hmv"],
            current_word="",
            previous_word="hmv",
            cursor_position=0,
            environment_type=EnvironmentType.CONDA,
            environment_path=Path("/test/env"),
            shell_type="bash",
        )

        self.assertTrue(plugin.can_complete(context))

    def test_plugin_manager(self):
        """Test plugin manager functionality."""
        # Test plugin listing
        plugins = self.plugin_manager.list_plugins()
        self.assertGreater(len(plugins), 0)

        # Test plugin statistics
        stats = self.plugin_manager.get_statistics()
        self.assertIsInstance(stats, dict)

    def test_plugin_priority(self):
        """Test plugin priority ordering."""
        plugins = self.plugin_manager.list_plugins()
        priorities = [plugin.priority for plugin in plugins]

        # Check that plugins are ordered by priority (descending)
        self.assertEqual(priorities, sorted(priorities, reverse=True))


class TestCompletionCache(unittest.TestCase):
    """Test completion cache system."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "cache"

        config = CacheConfig(
            max_entries=100,
            enable_persistence=False,  # Disable for testing
        )
        self.cache = CompletionCache(self.cache_dir, config)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        context = CompletionContext(
            command="homodyne",
            words=["homodyne"],
            current_word="",
            previous_word="homodyne",
            cursor_position=0,
            environment_type=EnvironmentType.CONDA,
            environment_path=Path("/test/env"),
            shell_type="bash",
        )

        # Test cache miss
        result = self.cache.get(context)
        self.assertIsNone(result)

        # Test cache put and hit
        from .core import CompletionResult

        test_results = [
            CompletionResult(
                completions=["--help", "--method"],
                completion_type=CompletionType.OPTION,
            )
        ]

        self.cache.put(context, test_results)
        cached_result = self.cache.get(context)
        self.assertIsNotNone(cached_result)
        self.assertEqual(len(cached_result), 1)

    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        stats = self.cache.get_statistics()
        self.assertIn("hit_rate", stats)
        self.assertIn("memory_entries", stats)


class TestInstallationSystem(unittest.TestCase):
    """Test installation and uninstallation system."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = InstallationConfig(
            force_install=True,
            backup_existing=False,
            atomic_install=False,
        )

    def test_environment_detection(self):
        """Test environment detection in installer."""
        installer = CompletionInstaller(self.config)
        self.assertIsInstance(installer.env_type, EnvironmentType)
        self.assertIsInstance(installer.env_path, Path)

    def test_installation_info(self):
        """Test installation information gathering."""
        installer = CompletionInstaller(self.config)
        info = installer.get_installation_info()

        self.assertIn("installed", info)
        self.assertIn("environment_type", info)
        self.assertIn("environment_path", info)

    def test_shell_detection(self):
        """Test shell detection."""
        installer = CompletionInstaller(self.config)
        shells = installer.detected_shells
        self.assertIsInstance(shells, list)


class CompletionBenchmark:
    """Performance benchmark for completion system."""

    def __init__(self):
        self.engine = CompletionEngine(enable_caching=True)
        self.plugin_manager = get_plugin_manager()

    def benchmark_completion_speed(self, iterations: int = 1000) -> dict[str, float]:
        """Benchmark completion generation speed."""
        contexts = self._generate_test_contexts(10)
        results = {}

        # Warm up
        for context in contexts[:3]:
            self.engine.complete(context)

        # Benchmark without cache
        start_time = time.perf_counter()
        for _ in range(iterations):
            for context in contexts:
                self.engine.complete(context, use_cache=False)
        no_cache_time = (time.perf_counter() - start_time) / iterations

        # Benchmark with cache
        start_time = time.perf_counter()
        for _ in range(iterations):
            for context in contexts:
                self.engine.complete(context, use_cache=True)
        cache_time = (time.perf_counter() - start_time) / iterations

        results = {
            "avg_completion_time_no_cache_ms": no_cache_time * 1000,
            "avg_completion_time_with_cache_ms": cache_time * 1000,
            "cache_speedup_factor": no_cache_time / cache_time if cache_time > 0 else 0,
            "completions_per_second": 1 / cache_time if cache_time > 0 else 0,
        }

        return results

    def benchmark_plugin_performance(self) -> dict[str, Any]:
        """Benchmark plugin system performance."""
        contexts = self._generate_test_contexts(100)

        start_time = time.perf_counter()
        for context in contexts:
            self.plugin_manager.get_completions(context)
        total_time = time.perf_counter() - start_time

        plugin_stats = self.plugin_manager.get_statistics()

        return {
            "total_time_ms": total_time * 1000,
            "avg_time_per_completion_ms": (total_time / len(contexts)) * 1000,
            "plugin_stats": plugin_stats,
        }

    def _generate_test_contexts(self, count: int) -> list[CompletionContext]:
        """Generate test completion contexts."""
        contexts = []
        test_commands = [
            ["homodyne", "--method"],
            ["homodyne", "--config"],
            ["homodyne-config", "--mode"],
            ["hmv", "config.json"],
            ["hmm", "--output-dir"],
        ]

        for i in range(count):
            cmd_args = test_commands[i % len(test_commands)]
            context = CompletionContext(
                command=cmd_args[0],
                words=cmd_args,
                current_word="",
                previous_word=cmd_args[-1] if len(cmd_args) > 1 else cmd_args[0],
                cursor_position=0,
                environment_type=EnvironmentType.CONDA,
                environment_path=Path(f"/test/env/{i}"),
                shell_type="bash",
            )
            contexts.append(context)

        return contexts


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="homodyne-test-completion",
        description="Test Homodyne Advanced Completion System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Test selection
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests",
    )

    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests",
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmarks",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests and benchmarks",
    )

    # Output control
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress output except results",
    )

    return parser


def print_status(message: str, level: str = "info") -> None:
    """Print status message."""
    if level == "error":
        print(f"❌ {message}", file=sys.stderr)
    elif level == "success":
        print(f"✅ {message}")
    elif level == "info":
        print(f"i  {message}")
    else:
        print(message)


def run_unit_tests(verbose: bool = False) -> bool:
    """Run unit tests."""
    print_status("Running unit tests...")

    test_classes = [
        TestCompletionCore,
        TestCompletionPlugins,
        TestCompletionCache,
        TestInstallationSystem,
    ]

    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1, stream=sys.stdout)
    result = runner.run(suite)

    success = result.wasSuccessful()
    if success:
        print_status(f"Unit tests passed ({result.testsRun} tests)", "success")
    else:
        print_status(
            f"Unit tests failed ({len(result.failures)} failures, {len(result.errors)} errors)",
            "error",
        )

    return success


def run_benchmarks(verbose: bool = False) -> dict[str, Any]:
    """Run performance benchmarks."""
    print_status("Running performance benchmarks...")

    benchmark = CompletionBenchmark()

    # Completion speed benchmark
    speed_results = benchmark.benchmark_completion_speed(100)

    # Plugin performance benchmark
    plugin_results = benchmark.benchmark_plugin_performance()

    results = {
        "completion_speed": speed_results,
        "plugin_performance": plugin_results,
    }

    # Print results
    print("\n📊 Benchmark Results:")
    print("=" * 50)

    print("\n🚀 Completion Speed:")
    print(
        f"   Without cache: {speed_results['avg_completion_time_no_cache_ms']:.2f} ms"
    )
    print(
        f"   With cache:    {speed_results['avg_completion_time_with_cache_ms']:.2f} ms"
    )
    print(f"   Speedup:       {speed_results['cache_speedup_factor']:.1f}x")
    print(
        f"   Throughput:    {speed_results['completions_per_second']:.0f} completions/sec"
    )

    print("\n🔌 Plugin Performance:")
    print(f"   Average time:  {plugin_results['avg_time_per_completion_ms']:.2f} ms")

    if verbose:
        print("\n🔍 Detailed Plugin Stats:")
        for plugin_name, stats in plugin_results["plugin_stats"].items():
            print(f"   {plugin_name}:")
            print(f"     Calls: {stats['calls']}")
            print(f"     Errors: {stats['errors']}")
            print(f"     Avg time: {stats['average_time_ms']:.2f} ms")

    return results


def main() -> int:
    """Main test routine."""
    parser = create_parser()
    args = parser.parse_args()

    # Default to running all tests if none specified
    if not any([args.unit, args.integration, args.benchmark, args.all]):
        args.all = True

    success = True
    results = {}

    try:
        # Run unit tests
        if args.unit or args.all:
            success &= run_unit_tests(args.verbose)

        # Run benchmarks
        if args.benchmark or args.all:
            benchmark_results = run_benchmarks(args.verbose)
            results.update(benchmark_results)

        # Overall result
        if success:
            print_status("\nAll tests completed successfully! 🎉", "success")
            return 0
        else:
            print_status("\nSome tests failed ❌", "error")
            return 1

    except KeyboardInterrupt:
        print_status("Tests cancelled by user", "error")
        return 1

    except Exception as e:
        print_status(f"Test error: {e}", "error")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
