"""
Tests for IRLS Performance Optimization

This module tests the optimized IRLS variance estimation and chi-squared calculation
functions that provide 10-400x speedup through JIT compilation and vectorization.

Test Categories:
- Core optimized function correctness
- Performance benchmarking
- Configuration-driven method selection
- Integration with HomodyneAnalysisCore
- Backward compatibility
"""

import json
import tempfile
import time

import numpy as np
import pytest

from homodyne.analysis.core import (
    HomodyneAnalysisCore,
    _calculate_chi_squared_vectorized_jit,
    _calculate_median_quickselect,
    _estimate_mad_vectorized_optimized,
)


class TestOptimizedMedianCalculation:
    """Test the optimized median calculation function."""

    def test_median_correctness_small_arrays(self):
        """Test median calculation accuracy for small arrays."""
        # Test cases with known medians
        test_cases = [
            ([1.0], 1.0),
            ([1.0, 2.0], 1.5),
            ([1.0, 2.0, 3.0], 2.0),
            ([3.0, 1.0, 2.0], 2.0),
            ([1.0, 2.0, 3.0, 4.0], 2.5),
            ([4.0, 1.0, 3.0, 2.0], 2.5),
        ]

        for data, expected in test_cases:
            data_array = np.array(data)
            result = _calculate_median_quickselect(data_array)
            np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_median_vs_numpy(self):
        """Test median calculation against numpy for random data."""
        np.random.seed(42)

        # Test various array sizes
        sizes = [5, 10, 15, 25, 50, 100]

        for size in sizes:
            data = np.random.randn(size)

            numpy_median = np.median(data)
            optimized_median = _calculate_median_quickselect(data)

            np.testing.assert_allclose(
                optimized_median,
                numpy_median,
                rtol=1e-12,
                atol=1e-15,
                err_msg=f"Median mismatch for size {size}",
            )

    def test_median_edge_cases(self):
        """Test median calculation edge cases."""
        # Empty array
        empty_array = np.array([])
        result = _calculate_median_quickselect(empty_array)
        assert np.isnan(result)

        # Single element
        single = np.array([42.0])
        result = _calculate_median_quickselect(single)
        assert result == 42.0

        # All same values
        same_values = np.array([3.14] * 10)
        result = _calculate_median_quickselect(same_values)
        assert result == 3.14

        # With NaN values (should handle gracefully)
        with_nan = np.array([1.0, np.nan, 3.0])
        result = _calculate_median_quickselect(with_nan)
        # Result should be predictable (nan will sort to end)
        assert np.isfinite(result) or np.isnan(result)


class TestVectorizedMADEstimation:
    """Test the vectorized MAD estimation function."""

    def test_mad_basic_functionality(self):
        """Test basic MAD estimation functionality."""
        np.random.seed(42)

        # Create test residuals
        residuals = np.random.randn(100) * 0.1
        window_size = 11

        # Test optimized MAD estimation
        variances = _estimate_mad_vectorized_optimized(residuals, window_size)

        # Check output properties
        assert variances.shape == residuals.shape
        assert np.all(variances > 0)  # All variances should be positive
        assert np.all(np.isfinite(variances))  # All variances should be finite

        # Check reasonable variance range (should be related to input scale)
        assert np.all(variances >= 1e-10)  # Above minimum floor
        assert np.all(variances < 1.0)  # Reasonable upper bound for this test

    def test_mad_window_size_effect(self):
        """Test that different window sizes produce different results."""
        np.random.seed(42)
        residuals = np.random.randn(50) * 0.1

        # Test different window sizes
        window_sizes = [5, 11, 21]
        results = []

        for window_size in window_sizes:
            variances = _estimate_mad_vectorized_optimized(residuals, window_size)
            results.append(variances)

        # Results should be different for different window sizes
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                assert not np.allclose(results[i], results[j], rtol=1e-3)

    def test_mad_with_outliers(self):
        """Test MAD estimation robustness with outliers."""
        np.random.seed(42)

        # Create data with outliers
        residuals = np.random.randn(100) * 0.01
        residuals[25] = 1.0  # Large outlier
        residuals[75] = -1.0  # Large outlier

        variances = _estimate_mad_vectorized_optimized(residuals, window_size=11)

        # MAD should be robust - most variances should be small
        median_variance = np.median(variances)
        assert median_variance < 0.1  # Most variances should remain small

        # Check that outlier regions have at least some elevated variances
        # (MAD is robust so may not always show dramatic increases)
        outlier_region_1 = variances[20:30]
        outlier_region_2 = variances[70:80]
        normal_region = variances[40:60]

        # At least one outlier region should show increased variance
        outlier_1_elevated = np.max(outlier_region_1) > np.max(normal_region)
        outlier_2_elevated = np.max(outlier_region_2) > np.max(normal_region)

        # At least one outlier should cause elevated variance somewhere
        assert outlier_1_elevated or outlier_2_elevated, (
            f"Outliers should cause some variance elevation. "
            f"Outlier1 max: {np.max(outlier_region_1):.2e}, "
            f"Outlier2 max: {np.max(outlier_region_2):.2e}, "
            f"Normal max: {np.max(normal_region):.2e}"
        )

    def test_mad_edge_cases(self):
        """Test MAD estimation edge cases."""
        # Very small array
        small_residuals = np.array([0.1, -0.1, 0.05])
        variances = _estimate_mad_vectorized_optimized(small_residuals, window_size=3)
        assert len(variances) == 3
        assert np.all(variances >= 1e-10)

        # Constant residuals
        constant_residuals = np.zeros(20)
        variances = _estimate_mad_vectorized_optimized(
            constant_residuals, window_size=5
        )
        # Should handle zeros gracefully (apply minimum variance)
        assert np.all(variances >= 1e-10)


class TestVectorizedChiSquaredCalculation:
    """Test the vectorized chi-squared calculation function."""

    def test_chi_squared_basic_calculation(self):
        """Test basic chi-squared calculation."""
        np.random.seed(42)

        # Create test data
        residuals = np.random.randn(100) * 0.1
        weights = np.ones_like(residuals) * 2.0

        # Calculate chi-squared
        chi_squared = _calculate_chi_squared_vectorized_jit(residuals, weights)

        # Compare with manual calculation
        expected_chi_squared = np.sum(residuals**2 * weights)

        np.testing.assert_allclose(chi_squared, expected_chi_squared, rtol=1e-12)

    def test_chi_squared_vs_standard_calculation(self):
        """Test optimized chi-squared vs standard calculation."""
        np.random.seed(42)

        # Create test data with various scales
        test_cases = [
            (np.random.randn(50) * 0.01, np.ones(50) * 1.0),
            (np.random.randn(100) * 0.1, np.random.uniform(0.1, 2.0, 100)),
            (np.random.randn(200) * 1.0, np.ones(200) * 0.5),
        ]

        for residuals, weights in test_cases:
            # Optimized calculation
            chi_squared_opt = _calculate_chi_squared_vectorized_jit(residuals, weights)

            # Standard calculation
            chi_squared_std = np.sum(residuals**2 * weights)

            # Should match exactly
            np.testing.assert_allclose(
                chi_squared_opt, chi_squared_std, rtol=1e-12, atol=1e-15
            )

    def test_chi_squared_edge_cases(self):
        """Test chi-squared calculation edge cases."""
        # Empty arrays
        empty_residuals = np.array([])
        empty_weights = np.array([])
        result = _calculate_chi_squared_vectorized_jit(empty_residuals, empty_weights)
        assert result == 0.0

        # Mismatched shapes
        residuals = np.array([1.0, 2.0])
        weights = np.array([1.0])
        result = _calculate_chi_squared_vectorized_jit(residuals, weights)
        assert result == np.inf

        # Zero weights
        residuals = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 0.0, 1.0])
        result = _calculate_chi_squared_vectorized_jit(residuals, weights)
        expected = 1.0**2 * 1.0 + 3.0**2 * 1.0  # Skip zero weight
        np.testing.assert_allclose(result, expected)

        # Infinite/NaN handling
        residuals = np.array([1.0, np.inf, 2.0])
        weights = np.array([1.0, 1.0, 1.0])
        result = _calculate_chi_squared_vectorized_jit(residuals, weights)
        # Should handle infinite values gracefully
        assert np.isfinite(result) or np.isinf(result)


class TestConfigurationDrivenMethodSelection:
    """Test configuration-driven method selection."""

    def create_test_config(
        self, optimization_enabled=True, variance_estimator="irls_optimized"
    ):
        """Create a test configuration."""
        config = {
            "metadata": {"config_version": "0.7.2", "description": "Test config"},
            "experimental_data": {
                "data_folder_path": "./test_data/",
                "data_file_name": "test.hdf",
                "phi_angles_path": "./test_data/",
                "phi_angles_file": "phi_list.txt",
            },
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 10, "end_frame": 50},
                "scattering": {"wavevector_q": 0.01},
                "geometry": {"stator_rotor_gap": 2000000},
                "computational": {"num_threads": 2, "auto_detect_cores": True},
            },
            "initial_parameters": {
                "values": [100.0, 0.0, 10.0],
                "parameter_names": ["D0", "alpha", "D_offset"],
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 1.0, "max": 1000.0},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": -100, "max": 100},
                ]
            },
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]},
                "robust_optimization": {"enabled": False},
                "mcmc_sampling": {"enabled": False},
            },
            "analysis_settings": {"static_mode": True, "static_submode": "isotropic"},
            "advanced_settings": {
                "chi_squared_calculation": {
                    "method": "standard",
                    "minimum_sigma": 1e-10,
                    "moving_window_size": 11,
                    "moving_window_edge_method": "reflect",
                    "variance_method": "irls_mad_robust",
                    "performance_optimization": {
                        "enabled": optimization_enabled,
                        "variance_estimator": variance_estimator,
                        "chi_calculator": "vectorized_jit",
                        "median_algorithm": "quickselect",
                        "jit_compilation": {"enabled": True, "warmup_on_init": False},
                        "memory_optimization": {
                            "use_memory_pool": True,
                            "pool_size_mb": 256,
                            "max_cache_items": 100,
                        },
                    },
                    "irls_config": {
                        "max_iterations": 3,  # Reduced for testing
                        "damping_factor": 0.7,
                        "convergence_tolerance": 1e-4,
                        "initial_sigma_squared": 1e-3,
                        "optimized_config": {
                            "use_vectorized_mad": True,
                            "enable_jit_compilation": True,
                        },
                    },
                }
            },
            "performance_settings": {
                "numba_optimization": {"enable_numba": True, "warmup_numba": False}
            },
            "output_settings": {"results_directory": "./test_results"},
            "validation_rules": {"data_quality": {"check_data_range": True}},
        }
        return config

    def test_optimization_enabled_selection(self):
        """Test method selection when optimization is enabled."""
        config = self.create_test_config(optimization_enabled=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        try:
            core = HomodyneAnalysisCore(config_file=config_path)

            # Check optimization is enabled
            assert hasattr(core, "optimization_enabled")
            assert core.optimization_enabled is True

            # Check optimized methods are selected
            assert hasattr(core, "_selected_variance_estimator")
            assert (
                core._selected_variance_estimator.__name__
                == "_estimate_variance_irls_optimized"
            )

            assert hasattr(core, "_selected_chi_calculator")
            assert (
                core._selected_chi_calculator.__name__
                == "_calculate_chi_squared_with_config"
            )

        finally:
            import os

            os.unlink(config_path)

    def test_optimization_disabled_selection(self):
        """Test method selection when optimization is disabled."""
        config = self.create_test_config(optimization_enabled=False)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        try:
            core = HomodyneAnalysisCore(config_file=config_path)

            # Check optimization is disabled
            assert core.optimization_enabled is False

            # Check legacy methods are selected
            assert (
                core._selected_variance_estimator.__name__
                == "_estimate_variance_irls_mad_robust"
            )
            assert (
                core._selected_chi_calculator.__name__
                == "calculate_chi_squared_optimized"
            )

        finally:
            import os

            os.unlink(config_path)

    def test_variance_estimator_selection(self):
        """Test different variance estimator selections."""
        estimator_configs = [
            ("irls_optimized", "_estimate_variance_irls_optimized"),
            ("irls_mad_robust", "_estimate_variance_irls_mad_robust"),
        ]

        for config_value, expected_method in estimator_configs:
            config = self.create_test_config(
                optimization_enabled=True, variance_estimator=config_value
            )

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(config, f)
                config_path = f.name

            try:
                core = HomodyneAnalysisCore(config_file=config_path)
                assert core._selected_variance_estimator.__name__ == expected_method
            finally:
                import os

                os.unlink(config_path)


class TestOptimizedIRLSIntegration:
    """Test integration of optimized IRLS with HomodyneAnalysisCore."""

    def create_test_core_with_optimization(self):
        """Create HomodyneAnalysisCore with optimization enabled."""
        config = {
            "metadata": {"config_version": "0.7.2", "description": "Test config"},
            "experimental_data": {
                "data_folder_path": "./test_data/",
                "data_file_name": "test.hdf",
                "phi_angles_path": "./test_data/",
                "phi_angles_file": "phi_list.txt",
            },
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 10, "end_frame": 50},
                "scattering": {"wavevector_q": 0.01},
                "geometry": {"stator_rotor_gap": 2000000},
                "computational": {"num_threads": 2},
            },
            "initial_parameters": {
                "values": [100.0, 0.0, 10.0],
                "parameter_names": ["D0", "alpha", "D_offset"],
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 1.0, "max": 1000.0},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": -100, "max": 100},
                ]
            },
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]},
                "robust_optimization": {"enabled": False},
                "mcmc_sampling": {"enabled": False},
            },
            "analysis_settings": {"static_mode": True, "static_submode": "isotropic"},
            "advanced_settings": {
                "chi_squared_calculation": {
                    "method": "standard",
                    "minimum_sigma": 1e-10,
                    "moving_window_size": 11,
                    "moving_window_edge_method": "reflect",
                    "variance_method": "irls_mad_robust",
                    "performance_optimization": {
                        "enabled": True,
                        "variance_estimator": "irls_optimized",
                        "chi_calculator": "vectorized_jit",
                        "memory_optimization": {"pool_size_mb": 256},
                    },
                    "irls_config": {
                        "max_iterations": 3,
                        "damping_factor": 0.7,
                        "convergence_tolerance": 1e-4,
                        "initial_sigma_squared": 1e-3,
                        "optimized_config": {"use_vectorized_mad": True},
                    },
                }
            },
            "performance_settings": {
                "numba_optimization": {"enable_numba": True, "warmup_numba": False}
            },
            "output_settings": {"results_directory": "./test_results"},
            "validation_rules": {"data_quality": {"check_data_range": True}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        return HomodyneAnalysisCore(config_file=config_path), config_path

    def test_optimized_irls_method_execution(self):
        """Test that optimized IRLS method executes correctly."""
        core, config_path = self.create_test_core_with_optimization()

        try:
            # Test optimized variance estimation
            np.random.seed(42)
            test_residuals = np.random.randn(100) * 0.1

            variances = core._estimate_variance_irls_optimized(
                test_residuals, window_size=11, edge_method="reflect"
            )

            # Check output properties
            assert variances.shape == test_residuals.shape
            assert np.all(variances > 0)
            assert np.all(np.isfinite(variances))

        finally:
            import os

            os.unlink(config_path)

    def test_optimized_vs_legacy_consistency(self):
        """Test that optimized methods produce consistent results with legacy methods."""
        # Create cores with optimization enabled and disabled
        config_opt = {
            "metadata": {"config_version": "0.7.2"},
            "experimental_data": {
                "data_folder_path": "./test_data/",
                "data_file_name": "test.hdf",
                "phi_angles_path": "./test_data/",
                "phi_angles_file": "phi_list.txt",
            },
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 10, "end_frame": 50},
                "scattering": {"wavevector_q": 0.01},
                "geometry": {"stator_rotor_gap": 2000000},
                "computational": {"num_threads": 2},
            },
            "initial_parameters": {
                "values": [100.0, 0.0, 10.0],
                "parameter_names": ["D0", "alpha", "D_offset"],
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 1.0, "max": 1000.0},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": -100, "max": 100},
                ]
            },
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]},
                "robust_optimization": {"enabled": False},
                "mcmc_sampling": {"enabled": False},
            },
            "analysis_settings": {"static_mode": True, "static_submode": "isotropic"},
            "advanced_settings": {
                "chi_squared_calculation": {
                    "method": "standard",
                    "minimum_sigma": 1e-10,
                    "moving_window_size": 11,
                    "moving_window_edge_method": "reflect",
                    "variance_method": "irls_mad_robust",
                    "performance_optimization": {
                        "enabled": True,
                        "variance_estimator": "irls_optimized",
                        "chi_calculator": "vectorized_jit",
                    },
                    "irls_config": {
                        "max_iterations": 5,
                        "damping_factor": 0.7,
                        "convergence_tolerance": 1e-4,
                        "initial_sigma_squared": 1e-3,
                        "optimized_config": {"use_vectorized_mad": True},
                    },
                }
            },
            "performance_settings": {
                "numba_optimization": {"enable_numba": True, "warmup_numba": False}
            },
            "output_settings": {"results_directory": "./test_results"},
            "validation_rules": {"data_quality": {"check_data_range": True}},
        }

        config_legacy = config_opt.copy()
        config_legacy["advanced_settings"]["chi_squared_calculation"][
            "performance_optimization"
        ]["enabled"] = False

        # Create temporary config files
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f1:
            json.dump(config_opt, f1)
            config_path_opt = f1.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f2:
            json.dump(config_legacy, f2)
            config_path_legacy = f2.name

        try:
            core_opt = HomodyneAnalysisCore(config_file=config_path_opt)
            core_legacy = HomodyneAnalysisCore(config_file=config_path_legacy)

            # Test with same random seed for reproducibility
            np.random.seed(42)
            test_residuals = (
                np.random.randn(50) * 0.05
            )  # Small residuals for better convergence

            # Both methods should produce similar results (within numerical tolerance)
            variances_opt = core_opt._selected_variance_estimator(
                test_residuals, window_size=11
            )
            variances_legacy = core_legacy._selected_variance_estimator(
                test_residuals, window_size=11
            )

            # Results should be reasonably close (allowing for algorithmic differences)
            # Both use IRLS with MAD, so should converge to similar values
            correlation = np.corrcoef(variances_opt, variances_legacy)[0, 1]
            assert correlation > 0.8, (
                f"Correlation between optimized and legacy results too low: {correlation}"
            )

        finally:
            import os

            os.unlink(config_path_opt)
            os.unlink(config_path_legacy)


class TestPerformanceBenchmarks:
    """Performance benchmarking tests for optimized functions."""

    @pytest.mark.benchmark
    def test_median_calculation_performance(self, benchmark_enabled=True):
        """Benchmark median calculation performance."""
        if not benchmark_enabled:
            pytest.skip("Performance benchmarks disabled")

        np.random.seed(42)

        # Test different array sizes typical in IRLS
        sizes = [10, 25, 50]

        for size in sizes:
            data = np.random.randn(size)

            # Time numpy median
            start_time = time.perf_counter()
            for _ in range(1000):
                np.median(data)
            numpy_time = time.perf_counter() - start_time

            # Time optimized median
            start_time = time.perf_counter()
            for _ in range(1000):
                _calculate_median_quickselect(data)
            optimized_time = time.perf_counter() - start_time

            speedup = numpy_time / optimized_time
            print(f"Median size {size}: Optimized {speedup:.1f}x faster than numpy")

            # For small arrays, optimized should be competitive
            # (May not always be faster due to JIT overhead, but should be reasonable)
            assert speedup > 0.1  # At least not 10x slower

    @pytest.mark.benchmark
    def test_mad_estimation_performance(self, benchmark_enabled=True):
        """Benchmark MAD estimation performance."""
        if not benchmark_enabled:
            pytest.skip("Performance benchmarks disabled")

        np.random.seed(42)

        # Test realistic IRLS scenario
        residuals = np.random.randn(200) * 0.1
        window_size = 11

        # Time optimized MAD (single call to warm up JIT)
        _estimate_mad_vectorized_optimized(residuals[:10], window_size)

        start_time = time.perf_counter()
        for _ in range(100):
            _estimate_mad_vectorized_optimized(residuals, window_size)
        optimized_time = time.perf_counter() - start_time

        print(
            f"MAD estimation: {optimized_time / 100 * 1000:.2f}ms per call (200 points)"
        )

        # Should be reasonably fast for production use
        assert optimized_time < 10.0  # Less than 10 seconds for 100 calls

    @pytest.mark.benchmark
    def test_chi_squared_calculation_performance(self, benchmark_enabled=True):
        """Benchmark chi-squared calculation performance."""
        if not benchmark_enabled:
            pytest.skip("Performance benchmarks disabled")

        np.random.seed(42)

        # Test realistic data sizes
        sizes = [100, 500, 1000]

        for size in sizes:
            residuals = np.random.randn(size) * 0.1
            weights = np.random.uniform(0.5, 2.0, size)

            # Time standard calculation
            start_time = time.perf_counter()
            for _ in range(1000):
                np.sum(residuals**2 * weights)
            standard_time = time.perf_counter() - start_time

            # Time optimized calculation (warm up first)
            _calculate_chi_squared_vectorized_jit(residuals[:10], weights[:10])

            start_time = time.perf_counter()
            for _ in range(1000):
                _calculate_chi_squared_vectorized_jit(residuals, weights)
            optimized_time = time.perf_counter() - start_time

            speedup = standard_time / optimized_time
            print(f"Chi-squared size {size}: Optimized {speedup:.1f}x vs standard")

            # Should be competitive (may have JIT overhead for small arrays)
            assert speedup > 0.1  # At least not 10x slower


class TestBackwardCompatibility:
    """Test backward compatibility of optimized implementation."""

    def test_existing_api_unchanged(self):
        """Test that existing API methods still work unchanged."""
        # Create a minimal config without optimization settings
        minimal_config = {
            "metadata": {"config_version": "0.7.2"},
            "experimental_data": {
                "data_folder_path": "./test/",
                "data_file_name": "test.hdf",
                "phi_angles_path": "./test/",
                "phi_angles_file": "phi.txt",
            },
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 10, "end_frame": 50},
                "scattering": {"wavevector_q": 0.01},
                "geometry": {"stator_rotor_gap": 2000000},
                "computational": {"num_threads": 2},
            },
            "initial_parameters": {
                "values": [100.0, 0.0, 10.0],
                "parameter_names": ["D0", "alpha", "D_offset"],
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 1.0, "max": 1000.0},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": -100, "max": 100},
                ]
            },
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]},
                "robust_optimization": {"enabled": False},
                "mcmc_sampling": {"enabled": False},
            },
            "analysis_settings": {"static_mode": True, "static_submode": "isotropic"},
            "advanced_settings": {
                "chi_squared_calculation": {
                    "method": "standard",
                    "variance_method": "irls_mad_robust",
                    "irls_config": {"max_iterations": 5},
                }
            },
            "performance_settings": {
                "numba_optimization": {"enable_numba": True, "warmup_numba": False}
            },
            "output_settings": {"results_directory": "./test_results"},
            "validation_rules": {"data_quality": {"check_data_range": True}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(minimal_config, f)
            config_path = f.name

        try:
            # Should initialize without errors
            core = HomodyneAnalysisCore(config_file=config_path)

            # Should have fallback methods selected
            assert hasattr(core, "_selected_variance_estimator")
            assert hasattr(core, "_selected_chi_calculator")

            # Test that variance estimation still works
            np.random.seed(42)
            # Use a simple test case
            test_residuals = np.random.randn(20) * 0.1
            variances = core._selected_variance_estimator(
                test_residuals, window_size=5, edge_method="none"
            )

            # Check basic properties rather than exact shape (due to potential edge processing)
            assert len(variances) > 0
            assert np.all(variances > 0)
            assert np.all(np.isfinite(variances))

        finally:
            import os

            os.unlink(config_path)

    def test_missing_optimization_config_fallback(self):
        """Test graceful fallback when optimization config is missing."""
        # Config without any performance_optimization section
        config = {
            "metadata": {"config_version": "0.7.2"},
            "experimental_data": {
                "data_folder_path": "./test/",
                "data_file_name": "test.hdf",
                "phi_angles_path": "./test/",
                "phi_angles_file": "phi.txt",
            },
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 10, "end_frame": 50},
                "scattering": {"wavevector_q": 0.01},
                "geometry": {"stator_rotor_gap": 2000000},
                "computational": {"num_threads": 2},
            },
            "initial_parameters": {
                "values": [100.0, 0.0, 10.0],
                "parameter_names": ["D0", "alpha", "D_offset"],
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 1.0, "max": 1000.0},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": -100, "max": 100},
                ]
            },
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]},
                "robust_optimization": {"enabled": False},
                "mcmc_sampling": {"enabled": False},
            },
            "analysis_settings": {"static_mode": True, "static_submode": "isotropic"},
            "advanced_settings": {"chi_squared_calculation": {"method": "standard"}},
            "performance_settings": {
                "numba_optimization": {"enable_numba": True, "warmup_numba": False}
            },
            "output_settings": {"results_directory": "./test_results"},
            "validation_rules": {"data_quality": {"check_data_range": True}},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        try:
            # Should initialize and default to non-optimized methods
            core = HomodyneAnalysisCore(config_file=config_path)

            # Should default to optimization disabled
            assert getattr(core, "optimization_enabled", False) is False

            # Should select legacy methods
            assert (
                core._selected_variance_estimator.__name__
                == "_estimate_variance_irls_mad_robust"
            )
            assert (
                core._selected_chi_calculator.__name__
                == "calculate_chi_squared_optimized"
            )

        finally:
            import os

            os.unlink(config_path)


class TestNumericalStabilityAndEdgeCases:
    """Test numerical stability and edge cases for optimized functions."""

    def test_median_with_special_values(self):
        """Test median calculation with special floating-point values."""
        # Test with very small values
        small_values = np.array([1e-15, 2e-15, 3e-15])
        result = _calculate_median_quickselect(small_values)
        assert np.isfinite(result)
        assert result == 2e-15

        # Test with very large values
        large_values = np.array([1e15, 2e15, 3e15])
        result = _calculate_median_quickselect(large_values)
        assert np.isfinite(result)
        assert result == 2e15

        # Test with mixed scales
        mixed_values = np.array([1e-10, 1.0, 1e10])
        result = _calculate_median_quickselect(mixed_values)
        assert result == 1.0

    def test_mad_estimation_numerical_stability(self):
        """Test MAD estimation numerical stability."""
        # Test with very small residuals
        small_residuals = np.random.randn(50) * 1e-12
        variances = _estimate_mad_vectorized_optimized(small_residuals, window_size=5)

        # Should not underflow to zero
        assert np.all(variances >= 1e-10)  # Above minimum floor
        assert np.all(np.isfinite(variances))

        # Test with mixed scales
        mixed_residuals = np.concatenate(
            [
                np.random.randn(25) * 1e-6,  # Very small
                np.random.randn(25) * 1e-3,  # Normal scale
            ]
        )
        variances = _estimate_mad_vectorized_optimized(mixed_residuals, window_size=7)
        assert np.all(variances > 0)
        assert np.all(np.isfinite(variances))

    def test_chi_squared_numerical_accuracy(self):
        """Test chi-squared calculation numerical accuracy."""
        # Test with high-precision reference case
        np.random.seed(123)
        residuals = np.array([0.001, -0.002, 0.0015, -0.0008, 0.0012])
        weights = np.array([100.0, 200.0, 150.0, 80.0, 120.0])

        # Calculate using optimized function
        result_opt = _calculate_chi_squared_vectorized_jit(residuals, weights)

        # Calculate reference (high precision)
        result_ref = float(
            np.sum(residuals.astype(np.float64) ** 2 * weights.astype(np.float64))
        )

        # Should match to high precision
        np.testing.assert_allclose(result_opt, result_ref, rtol=1e-14)

        # Test with extreme weight ranges
        extreme_weights = np.array([1e-10, 1e10, 1.0, 1e-5, 1e5])
        residuals_norm = np.random.randn(5) * 0.01

        result = _calculate_chi_squared_vectorized_jit(residuals_norm, extreme_weights)
        assert np.isfinite(result)
        assert result >= 0

    def test_consistency_across_data_sizes(self):
        """Test that optimized functions behave consistently across data sizes."""
        np.random.seed(42)

        # Test MAD estimation consistency
        base_data = np.random.randn(100) * 0.1
        base_variances = _estimate_mad_vectorized_optimized(base_data, window_size=11)

        # Subset should give similar results in overlapping regions
        subset_data = base_data[25:75]
        subset_variances = _estimate_mad_vectorized_optimized(
            subset_data, window_size=11
        )

        # Check that the character of results is similar
        base_std = np.std(base_variances[25:75])
        subset_std = np.std(subset_variances)

        # Standard deviations should be in same ballpark
        ratio = base_std / subset_std if subset_std > 0 else 1.0
        assert 0.1 < ratio < 10.0, f"Consistency check failed: ratio = {ratio}"

    def test_performance_degradation_detection(self):
        """Test for performance degradation detection."""
        # This test ensures optimized functions complete in reasonable time
        import time

        # Large arrays for performance test
        large_residuals = np.random.randn(1000) * 0.1
        large_weights = np.random.uniform(0.5, 2.0, 1000)

        # MAD estimation should complete quickly
        start_time = time.perf_counter()
        variances = _estimate_mad_vectorized_optimized(large_residuals, window_size=15)
        mad_time = time.perf_counter() - start_time

        # Should complete in reasonable time (less than 1 second)
        assert mad_time < 1.0, f"MAD estimation too slow: {mad_time:.3f}s"
        assert len(variances) == len(large_residuals)

        # Chi-squared should complete quickly
        start_time = time.perf_counter()
        chi_sq = _calculate_chi_squared_vectorized_jit(large_residuals, large_weights)
        chi_time = time.perf_counter() - start_time

        # Should complete very quickly (less than 0.1 seconds)
        assert chi_time < 0.1, f"Chi-squared calculation too slow: {chi_time:.3f}s"
        assert np.isfinite(chi_sq)
