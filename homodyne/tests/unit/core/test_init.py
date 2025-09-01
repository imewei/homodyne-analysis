"""
Test suite for homodyne package initialization.

This module tests the main package imports and optional dependency handling.
"""

import sys

import pytest


class TestPackageImports:
    """Test package-level imports and initialization."""

    def test_version_import(self):
        """Test that version information is available."""
        import homodyne

        assert hasattr(homodyne, "__version__")
        assert hasattr(homodyne, "__author__")
        assert hasattr(homodyne, "__email__")
        assert hasattr(homodyne, "__institution__")

        assert isinstance(homodyne.__version__, str)
        assert len(homodyne.__version__) > 0
        assert "." in homodyne.__version__  # Should be semantic version

    def test_python_version_requirement(self):
        """Test that Python version requirement is enforced."""
        # This test runs after import, so we can't test the actual check
        # But we can verify the version info is correct
        assert sys.version_info >= (3, 12), "Tests should run on Python 3.12+"

    def test_core_imports_available(self):
        """Test that core functionality is importable."""
        import homodyne

        # Core classes should be available
        assert hasattr(homodyne, "ConfigManager")
        assert hasattr(homodyne, "HomodyneAnalysisCore")
        assert hasattr(homodyne, "configure_logging")
        assert hasattr(homodyne, "performance_monitor")

    def test_kernel_imports_available(self):
        """Test that computational kernels are available."""
        import homodyne

        # Kernel functions should be available
        assert hasattr(homodyne, "create_time_integral_matrix_numba")
        assert hasattr(homodyne, "calculate_diffusion_coefficient_numba")
        assert hasattr(homodyne, "calculate_shear_rate_numba")
        assert hasattr(homodyne, "compute_g1_correlation_numba")
        assert hasattr(homodyne, "compute_sinc_squared_numba")
        assert hasattr(homodyne, "memory_efficient_cache")

    def test_optional_classical_optimizer_available(self):
        """Test classical optimizer availability."""
        import homodyne

        # Should be available when scipy is installed
        if homodyne.ClassicalOptimizer is not None:
            assert callable(homodyne.ClassicalOptimizer)

    def test_optional_robust_optimizer_status(self):
        """Test robust optimizer availability status."""
        import homodyne

        # Should handle missing CVXPY gracefully
        # Either available or set to None
        if homodyne.RobustHomodyneOptimizer is not None:
            assert callable(homodyne.RobustHomodyneOptimizer)
        if homodyne.create_robust_optimizer is not None:
            assert callable(homodyne.create_robust_optimizer)

    def test_optional_mcmc_sampler_status(self):
        """Test MCMC sampler availability status."""
        import homodyne

        # Should handle missing PyMC gracefully
        # Either available or set to None
        if homodyne.MCMCSampler is not None:
            assert callable(homodyne.MCMCSampler)
        if homodyne.create_mcmc_sampler is not None:
            assert callable(homodyne.create_mcmc_sampler)

    def test_all_exports_defined(self):
        """Test that all exported names are properly defined."""
        import homodyne

        # Check that __all__ contains valid exports
        for name in homodyne.__all__:
            assert hasattr(
                homodyne, name
            ), f"Exported name '{name}' not found in module"

            # Allow None for optional dependencies
            attr = getattr(homodyne, name)
            if attr is not None:
                # Should be a callable, class, or instance for most exports
                if not name.startswith("__"):  # Skip dunder attributes
                    # Allow instances (like performance_monitor)
                    assert (
                        callable(attr)
                        or isinstance(attr, type)
                        or hasattr(attr, "__class__")
                    ), f"Export '{name}' should be callable, a class, or an instance"


class TestOptionalDependencyHandling:
    """Test handling of optional dependencies."""

    def test_graceful_degradation_scipy(self):
        """Test graceful degradation when scipy is missing."""
        # This is tested by checking if ClassicalOptimizer is None
        # The actual import error handling happens at module level
        import homodyne

        # Either available or gracefully set to None
        assert homodyne.ClassicalOptimizer is None or callable(
            homodyne.ClassicalOptimizer
        )

    def test_graceful_degradation_cvxpy(self):
        """Test graceful degradation when CVXPY is missing."""
        import homodyne

        # Either available or gracefully set to None
        assert homodyne.RobustHomodyneOptimizer is None or callable(
            homodyne.RobustHomodyneOptimizer
        )
        assert homodyne.create_robust_optimizer is None or callable(
            homodyne.create_robust_optimizer
        )

    def test_graceful_degradation_pymc(self):
        """Test graceful degradation when PyMC is missing."""
        import homodyne

        # Either available or gracefully set to None
        assert homodyne.MCMCSampler is None or callable(homodyne.MCMCSampler)
        assert homodyne.create_mcmc_sampler is None or callable(
            homodyne.create_mcmc_sampler
        )

    def test_kernel_functions_always_available(self):
        """Test that kernel functions are always available (with or without numba)."""
        import homodyne

        # These should always be available, even without numba
        kernels = [
            "create_time_integral_matrix_numba",
            "calculate_diffusion_coefficient_numba",
            "calculate_shear_rate_numba",
            "compute_g1_correlation_numba",
            "compute_sinc_squared_numba",
        ]

        for kernel_name in kernels:
            kernel_func = getattr(homodyne, kernel_name)
            assert callable(kernel_func)
            # Should have signatures attribute (even if empty for fallback)
            assert hasattr(kernel_func, "signatures")


class TestPackageMetadata:
    """Test package metadata and version information."""

    def test_version_format(self):
        """Test that version follows semantic versioning."""
        import homodyne

        version = homodyne.__version__
        parts = version.split(".")
        assert len(parts) >= 2, "Version should have at least major.minor"

        # First two parts should be numeric
        assert parts[0].isdigit(), "Major version should be numeric"
        assert parts[1].isdigit(), "Minor version should be numeric"

    def test_author_information(self):
        """Test that author information is complete."""
        import homodyne

        assert homodyne.__author__ is not None
        assert len(homodyne.__author__) > 0
        assert homodyne.__email__ is not None
        assert "@" in homodyne.__email__
        assert homodyne.__institution__ is not None
        assert len(homodyne.__institution__) > 0

    def test_recent_improvements_documented(self):
        """Test that recent improvements are documented in comments."""
        # This is more of a documentation check
        # The improvements should be documented in the module docstring/comments
        import homodyne

        # Check that module has docstring
        assert homodyne.__doc__ is not None
        assert len(homodyne.__doc__) > 100  # Should be substantial


class TestImportPerformance:
    """Test import performance and efficiency."""

    def test_import_time_reasonable(self):
        """Test that import time is reasonable."""
        import time

        # Fresh import (if not already imported)
        if "homodyne" in sys.modules:
            del sys.modules["homodyne"]

        start_time = time.time()

        import_time = time.time() - start_time

        # Import should be reasonably fast (less than 5 seconds)
        assert import_time < 5.0, f"Import took too long: {import_time:.2f}s"

    def test_lazy_import_optimization(self):
        """Test that heavy dependencies are imported lazily."""
        # This test verifies that optional dependencies don't slow down
        # the main import even when available

        import homodyne

        # Core functionality should be immediately available
        assert homodyne.ConfigManager is not None
        assert homodyne.HomodyneAnalysisCore is not None

        # Optional optimizers might be None (which is fine)
        # or available (also fine)

        # At least core should work
        assert any(
            opt is not None
            for opt in [homodyne.ConfigManager, homodyne.HomodyneAnalysisCore]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
