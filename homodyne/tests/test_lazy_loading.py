"""
Test Suite for Advanced Lazy Loading Implementation
==================================================

Comprehensive tests for the enhanced lazy loading system that handles
heavy scientific computing dependencies.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import pytest
import time
import threading

from homodyne.core.lazy_imports import (
    HeavyDependencyLoader,
    BatchDependencyLoader,
    scientific_deps,
    get_import_performance_report,
    clear_import_cache,
    preload_critical_dependencies,
    LazyImportError,
)


class TestHeavyDependencyLoader:
    """Test suite for HeavyDependencyLoader class."""

    def setup_method(self):
        """Reset import cache before each test."""
        clear_import_cache()

    def test_successful_module_loading(self):
        """Test successful loading of a real module."""
        loader = HeavyDependencyLoader("math", required=True)

        # Module should not be loaded initially
        assert loader._cached_object is None
        assert not loader._load_attempted

        # Access should trigger loading
        cos_func = loader.cos
        assert cos_func(0) == 1.0

        # Object should now be cached
        assert loader._cached_object is not None
        assert loader._load_attempted
        assert loader.is_available

    def test_module_attribute_loading(self):
        """Test loading specific attribute from module."""
        loader = HeavyDependencyLoader("math", "cos", required=True)

        # Should load cos function specifically
        result = loader(0)
        assert result == 1.0

    def test_failed_loading_required(self):
        """Test handling of failed loading for required modules."""
        loader = HeavyDependencyLoader("nonexistent_module", required=True)

        with pytest.raises(LazyImportError):
            _ = loader.some_attr

    def test_failed_loading_optional(self):
        """Test handling of failed loading for optional modules."""
        fallback_value = "fallback"
        loader = HeavyDependencyLoader(
            "nonexistent_module",
            required=False,
            fallback=fallback_value
        )

        # Should return fallback without raising
        result = loader._get_object()
        assert result == fallback_value

    def test_thread_safety(self):
        """Test thread-safe loading behavior."""
        loader = HeavyDependencyLoader("json", required=True)
        results = []
        errors = []

        def load_module():
            try:
                obj = loader._get_object()
                results.append(obj)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=load_module) for _ in range(10)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should have no errors and all results should be the same object
        assert len(errors) == 0
        assert len(results) == 10
        assert all(result is results[0] for result in results)

    def test_performance_monitoring(self):
        """Test performance monitoring features."""
        loader = HeavyDependencyLoader("time", required=True)

        # Load the module
        _ = loader._get_object()

        # Should have recorded load time
        assert loader.load_time > 0
        assert loader._load_time > 0


class TestBatchDependencyLoader:
    """Test suite for BatchDependencyLoader class."""

    def setup_method(self):
        """Reset import cache before each test."""
        clear_import_cache()

    def test_batch_loading(self):
        """Test batch loading of multiple dependencies."""
        dependencies = {
            "math_module": {"module_name": "math", "required": True},
            "json_module": {"module_name": "json", "required": True},
            "time_module": {"module_name": "time", "required": True},
        }

        batch_loader = BatchDependencyLoader(dependencies)

        # Load all dependencies
        results = batch_loader.load_all()

        assert len(results) == 3
        assert "math_module" in results
        assert "json_module" in results
        assert "time_module" in results

        # All should be successfully loaded
        assert all(result is not None for result in results.values())

    def test_availability_report(self):
        """Test availability reporting."""
        dependencies = {
            "existing": {"module_name": "math", "required": True},
            "nonexistent": {"module_name": "nonexistent_module", "required": False},
        }

        batch_loader = BatchDependencyLoader(dependencies)
        report = batch_loader.availability_report

        assert report["existing"] is True
        assert report["nonexistent"] is False

    def test_get_dependency_by_name(self):
        """Test getting individual dependencies by name."""
        dependencies = {
            "math_module": {"module_name": "math", "required": True},
        }

        batch_loader = BatchDependencyLoader(dependencies)
        math_module = batch_loader.get("math_module")

        assert math_module is not None
        assert hasattr(math_module, "cos")

    def test_nonexistent_dependency_name(self):
        """Test error handling for nonexistent dependency names."""
        batch_loader = BatchDependencyLoader({})

        with pytest.raises(KeyError):
            batch_loader.get("nonexistent")


class TestScientificDependencies:
    """Test suite for pre-configured scientific dependencies."""

    def setup_method(self):
        """Reset import cache before each test."""
        clear_import_cache()

    def test_numpy_availability(self):
        """Test numpy dependency availability."""
        try:
            expected_available = True
        except ImportError:
            expected_available = False

        numpy_loader = scientific_deps.loaders["numpy"]
        assert numpy_loader.is_available == expected_available

    def test_scipy_availability(self):
        """Test scipy dependency availability."""
        try:
            expected_available = True
        except ImportError:
            expected_available = False

        scipy_loader = scientific_deps.loaders["scipy"]
        assert scipy_loader.is_available == expected_available

    def test_numba_fallback(self):
        """Test numba fallback functionality."""
        numba_jit_loader = scientific_deps.loaders["numba_jit"]

        # Should either load numba.jit or return fallback function
        jit_func = numba_jit_loader._get_object()
        assert callable(jit_func)

        # Fallback should be a no-op decorator
        if not numba_jit_loader.is_available:
            @jit_func
            def test_func():
                return 42

            assert test_func() == 42


class TestPerformanceMonitoring:
    """Test suite for performance monitoring features."""

    def setup_method(self):
        """Reset import cache before each test."""
        clear_import_cache()

    def test_import_performance_report(self):
        """Test import performance reporting."""
        # Load some modules to generate data
        loader1 = HeavyDependencyLoader("math", required=True)
        loader2 = HeavyDependencyLoader("json", required=True)

        _ = loader1._get_object()
        _ = loader2._get_object()

        # Get performance report
        report = get_import_performance_report()

        assert "summary" in report
        assert "individual_imports" in report
        assert "slowest_imports" in report

        summary = report["summary"]
        assert summary["total_imports"] >= 2
        assert summary["successful_imports"] >= 2
        assert summary["total_load_time"] > 0
        assert summary["success_rate"] > 0

    def test_preload_critical_dependencies(self):
        """Test preloading of critical dependencies."""
        # This should not raise exceptions
        preload_critical_dependencies()

        # Check that critical dependencies are loaded
        report = get_import_performance_report()
        assert report["summary"]["total_imports"] > 0


class TestIntegrationWithMainPackage:
    """Test integration with main package __init__.py."""

    def test_lazy_loader_integration(self):
        """Test that main package lazy loaders work."""
        import homodyne

        # Should be able to access performance report
        report = homodyne.get_import_performance_report()
        assert isinstance(report, dict)
        assert "summary" in report

    def test_preload_function_integration(self):
        """Test preload function integration."""
        import homodyne

        # Should not raise exceptions
        homodyne.preload_scientific_dependencies()

    @pytest.mark.slow
    def test_startup_time_improvement(self):
        """Test that lazy loading improves startup time."""
        # This is more of a benchmark than a strict test
        import subprocess
        import sys

        # Measure import time
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, "-c", "import homodyne"],
            capture_output=True,
            text=True
        )
        end_time = time.time()

        # Should import successfully
        assert result.returncode == 0

        # Import should be reasonably fast (adjust threshold as needed)
        import_time = end_time - start_time
        assert import_time < 5.0  # 5 seconds should be generous


class TestErrorHandling:
    """Test error handling and edge cases."""

    def setup_method(self):
        """Reset import cache before each test."""
        clear_import_cache()

    def test_import_error_handling(self):
        """Test proper handling of import errors."""
        loader = HeavyDependencyLoader("definitely_nonexistent_module", required=False)

        # Should handle gracefully
        result = loader._get_object()
        assert result is None

    def test_attribute_error_handling(self):
        """Test handling of missing attributes."""
        loader = HeavyDependencyLoader("math", "nonexistent_function", required=False)

        # Should handle gracefully
        result = loader._get_object()
        assert result is None

    def test_cache_key_uniqueness(self):
        """Test that cache keys are unique for different loaders."""
        loader1 = HeavyDependencyLoader("math", "cos")
        loader2 = HeavyDependencyLoader("math", "sin")
        loader3 = HeavyDependencyLoader("math")

        assert loader1._cache_key != loader2._cache_key
        assert loader1._cache_key != loader3._cache_key
        assert loader2._cache_key != loader3._cache_key


if __name__ == "__main__":
    pytest.main([__file__])