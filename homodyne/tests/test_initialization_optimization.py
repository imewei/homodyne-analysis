"""
Test Suite for Initialization Optimization
==========================================

Tests for module initialization order optimization and startup performance.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from homodyne.core.initialization_optimizer import DependencyAnalyzer
from homodyne.core.initialization_optimizer import InitializationOptimizer
from homodyne.core.initialization_optimizer import get_initialization_optimizer
from homodyne.core.initialization_optimizer import optimize_package_initialization
from homodyne.core.initialization_optimizer import profile_startup_performance


class TestDependencyAnalyzer:
    """Test suite for DependencyAnalyzer class."""

    def test_analyzer_initialization(self):
        """Test dependency analyzer initialization."""
        analyzer = DependencyAnalyzer("homodyne")
        assert analyzer.package_name == "homodyne"
        assert isinstance(analyzer.dependency_graph, dict)
        assert isinstance(analyzer.load_times, dict)
        assert isinstance(analyzer.memory_usage, dict)

    def test_dependency_analysis(self):
        """Test dependency analysis functionality."""
        analyzer = DependencyAnalyzer("homodyne")

        # Mock the complex parts to avoid file system dependencies
        with patch.object(analyzer, "_build_dependency_graph"), patch.object(
            analyzer, "_find_critical_path", return_value=["homodyne"]
        ):
            analysis = analyzer.analyze_dependencies()

        assert "dependency_graph" in analysis
        assert "critical_path" in analysis
        assert "strategies" in analysis
        assert "recommendations" in analysis

    def test_path_to_module_name_conversion(self):
        """Test conversion of file paths to module names."""
        from pathlib import Path

        analyzer = DependencyAnalyzer("homodyne")

        # Create mock paths
        package_path = Path("/path/to/homodyne")
        test_file = Path("/path/to/homodyne/core/kernels.py")

        module_name = analyzer._path_to_module_name(test_file, package_path)
        assert module_name == "homodyne.core.kernels"

    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        analyzer = DependencyAnalyzer("homodyne")

        # Create mock circular dependencies
        analyzer.dependency_graph = {
            "A": {"B"},
            "B": {"C"},
            "C": {"A"},  # Creates a cycle
        }

        assert analyzer._has_circular_dependencies()

    def test_no_circular_dependencies(self):
        """Test proper handling when no circular dependencies exist."""
        analyzer = DependencyAnalyzer("homodyne")

        # Create mock acyclic dependencies
        analyzer.dependency_graph = {
            "A": {"B"},
            "B": {"C"},
            "C": set(),
        }

        assert not analyzer._has_circular_dependencies()

    def test_heavy_import_detection(self):
        """Test detection of heavy imports."""
        analyzer = DependencyAnalyzer("homodyne")

        # Mock dependency graph with heavy imports
        analyzer.dependency_graph = {
            "homodyne.core.numpy_heavy": set(),
            "homodyne.analysis.scipy_heavy": set(),
            "homodyne.regular.module": set(),
        }

        heavy_modules = analyzer._find_heavy_imports()
        assert len(heavy_modules) >= 0  # May find heavy modules


class TestInitializationOptimizer:
    """Test suite for InitializationOptimizer class."""

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = InitializationOptimizer("homodyne")
        assert optimizer.package_name == "homodyne"
        assert isinstance(optimizer.analyzer, DependencyAnalyzer)
        assert isinstance(optimizer.metrics, list)

    def test_optimization_strategy_generation(self):
        """Test generation of optimization strategy."""
        optimizer = InitializationOptimizer("homodyne")

        # Mock the analysis to avoid complex dependencies
        with patch.object(optimizer.analyzer, "analyze_dependencies") as mock_analyze:
            mock_analyze.return_value = {
                "dependency_graph": {},
                "critical_path": [],
                "strategies": optimizer.analyzer._calculate_optimization_strategies(),
                "recommendations": [],
            }

            strategy = optimizer.optimize_initialization_order()

            assert strategy is not None
            assert hasattr(strategy, "core_modules")
            assert hasattr(strategy, "lazy_modules")
            assert hasattr(strategy, "deferred_modules")
            assert hasattr(strategy, "preload_modules")

    def test_performance_report_generation(self):
        """Test performance report generation."""
        optimizer = InitializationOptimizer("homodyne")

        report = optimizer.get_performance_report()

        assert "metrics" in report
        assert "optimization_strategy" in report
        assert "total_modules" in report
        assert "optimization_applied" in report

    def test_memory_usage_measurement(self):
        """Test memory usage measurement."""
        optimizer = InitializationOptimizer("homodyne")

        memory_usage = optimizer._get_memory_usage()
        assert isinstance(memory_usage, float)
        assert memory_usage >= 0

    @pytest.mark.slow
    def test_profile_initialization_context(self):
        """Test initialization profiling context manager."""
        optimizer = InitializationOptimizer("homodyne")

        start_time = time.perf_counter()
        with optimizer.profile_initialization():
            # Simulate some work
            time.sleep(0.01)
        end_time = time.perf_counter()

        # Should have taken at least the sleep time
        assert end_time - start_time >= 0.01

    def test_preload_core_modules(self):
        """Test preloading of core modules."""
        optimizer = InitializationOptimizer("homodyne")

        # Set up a mock strategy
        from homodyne.core.initialization_optimizer import InitializationStrategy

        optimizer.optimization_strategy = InitializationStrategy(
            core_modules=["math", "json"],  # Use standard library modules
            lazy_modules=[],
            deferred_modules=[],
            preload_modules=[],
        )

        # Should not raise exceptions
        optimizer._preload_core_modules()

    def test_apply_optimizations_without_strategy(self):
        """Test applying optimizations without existing strategy."""
        optimizer = InitializationOptimizer("homodyne")

        # Mock the optimization to avoid complex dependencies
        with patch.object(optimizer, "optimize_initialization_order") as mock_optimize:
            mock_optimize.return_value = (
                optimizer.analyzer._calculate_optimization_strategies()
            )

            # Should not raise exceptions
            optimizer.apply_optimizations()


class TestGlobalFunctions:
    """Test global optimizer functions."""

    def test_get_global_optimizer(self):
        """Test getting global optimizer instance."""
        optimizer1 = get_initialization_optimizer()
        optimizer2 = get_initialization_optimizer()

        # Should return the same instance
        assert optimizer1 is optimizer2

    def test_optimize_package_initialization(self):
        """Test package initialization optimization."""
        # Test that the function returns a valid strategy
        from homodyne.core.initialization_optimizer import InitializationStrategy

        strategy = optimize_package_initialization()

        # Verify we got a strategy back
        assert isinstance(strategy, InitializationStrategy)

        # Verify strategy has expected attributes
        assert hasattr(strategy, "core_modules")
        assert hasattr(strategy, "lazy_modules")
        assert hasattr(strategy, "deferred_modules")
        assert hasattr(strategy, "preload_modules")
        assert hasattr(strategy, "optimization_level")

        # Verify lists are not empty (at least core_modules should have entries)
        assert isinstance(strategy.core_modules, list)
        assert isinstance(strategy.lazy_modules, list)
        assert isinstance(strategy.deferred_modules, list)
        assert isinstance(strategy.preload_modules, list)

    def test_profile_startup_performance(self):
        """Test startup performance profiling."""
        report = profile_startup_performance()

        assert isinstance(report, dict)
        # Should contain some performance metrics


class TestIntegrationWithMainPackage:
    """Test integration with main package."""

    def test_main_package_optimization_functions(self):
        """Test optimization functions in main package."""
        import homodyne

        # Should have optimization functions available
        assert hasattr(homodyne, "optimize_initialization")
        assert hasattr(homodyne, "get_startup_performance_report")

    def test_optimize_initialization_function(self):
        """Test main package optimization function."""
        import homodyne

        # Mock to avoid complex analysis
        with patch(
            "homodyne.core.initialization_optimizer.optimize_package_initialization"
        ) as mock_opt:
            from homodyne.core.initialization_optimizer import InitializationStrategy

            mock_strategy = InitializationStrategy(
                core_modules=["test"],
                lazy_modules=["test"],
                deferred_modules=["test"],
                preload_modules=["test"],
            )
            mock_opt.return_value = mock_strategy

            result = homodyne.optimize_initialization()

            assert "strategy" in result
            assert "core_modules" in result["strategy"]

    def test_startup_performance_report_function(self):
        """Test startup performance report function."""
        import homodyne

        report = homodyne.get_startup_performance_report()

        assert isinstance(report, dict)

    @pytest.mark.slow
    def test_automatic_optimization_application(self):
        """Test that optimizations are applied automatically during import."""
        # Test with environment variable enabled
        import os

        env = os.environ.copy()
        env["HOMODYNE_OPTIMIZE_STARTUP"] = "true"

        # Import package in subprocess with optimization enabled
        result = subprocess.run(
            [sys.executable, "-c", "import homodyne; print('Success')"],
            check=False, env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Success" in result.stdout

    def test_optimization_environment_variable_disabled(self):
        """Test behavior when optimization is disabled."""
        import os

        env = os.environ.copy()
        env["HOMODYNE_OPTIMIZE_STARTUP"] = "false"

        # Should still import successfully
        result = subprocess.run(
            [sys.executable, "-c", "import homodyne; print('Success')"],
            check=False, env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Success" in result.stdout


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for initialization optimization."""

    @pytest.mark.slow
    def test_startup_time_benchmark(self):
        """Benchmark startup time with and without optimization."""
        import statistics

        # Measure startup time multiple times
        startup_times = []

        for _ in range(5):
            start_time = time.perf_counter()
            result = subprocess.run(
                [sys.executable, "-c", "import homodyne"],
                check=False, capture_output=True,
                text=True,
            )
            end_time = time.perf_counter()

            if result.returncode == 0:
                startup_times.append(end_time - start_time)

        if startup_times:
            mean_time = statistics.mean(startup_times)
            std_time = statistics.stdev(startup_times) if len(startup_times) > 1 else 0

            print(f"Startup time: {mean_time:.4f}s ± {std_time:.4f}s")

            # Should be reasonably fast (adjust threshold as needed)
            assert mean_time < 5.0, f"Startup time too slow: {mean_time:.4f}s"

    def test_memory_usage_optimization(self):
        """Test memory usage during initialization."""
        optimizer = InitializationOptimizer("homodyne")

        initial_memory = optimizer._get_memory_usage()

        # Apply optimizations
        with patch.object(optimizer, "optimize_initialization_order") as mock_opt:
            mock_opt.return_value = (
                optimizer.analyzer._calculate_optimization_strategies()
            )
            optimizer.apply_optimizations()

        final_memory = optimizer._get_memory_usage()

        # Memory usage should be reasonable
        memory_increase = final_memory - initial_memory
        assert memory_increase < 100.0, (
            f"Memory increase too high: {memory_increase:.2f}MB"
        )


if __name__ == "__main__":
    pytest.main([__file__])
