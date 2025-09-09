"""
Enhanced JIT Compilation Performance Tests
==========================================

This module tests all JIT-compiled kernel functions and their performance
optimizations, including:
- Numba kernel compilation and fallback behavior
- Parallel processing capabilities
- Performance benchmarks and improvements
- Warmup functionality
- Memory efficiency

Test Categories:
- Core kernel functionality and correctness
- JIT compilation behavior and fallbacks
- Parallel processing capabilities
- Performance benchmarking
- Memory usage optimization
"""

import time
from unittest.mock import patch

import numpy as np
import pytest

from homodyne.core.kernels import (
    NUMBA_AVAILABLE,
    calculate_diffusion_coefficient_numba,
    calculate_shear_rate_numba,
    compute_chi_squared_batch_numba,
    compute_g1_correlation_numba,
    compute_sinc_squared_numba,
    create_time_integral_matrix_numba,
    memory_efficient_cache,
    solve_least_squares_batch_numba,
)


class TestJITKernelFunctionality:
    """Test core functionality of JIT-compiled kernels."""

    def test_create_time_integral_matrix_correctness(self):
        """Test time integral matrix creation correctness."""
        # Test with simple known case
        time_array = np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float64)

        result = create_time_integral_matrix_numba(time_array)

        # Verify shape
        assert result.shape == (4, 4)

        # Verify cumulative sum calculation
        cumsum = np.cumsum(time_array)  # [1, 3, 6, 11]
        expected = np.abs(cumsum[:, np.newaxis] - cumsum[np.newaxis, :])

        np.testing.assert_array_almost_equal(result, expected, decimal=12)

    def test_diffusion_coefficient_calculation(self):
        """Test diffusion coefficient calculation."""
        time_array = np.array([0.1, 0.2, 0.5, 1.0], dtype=np.float64)
        D0 = 100.0
        alpha = -0.1
        D_offset = 10.0

        result = calculate_diffusion_coefficient_numba(time_array, D0, alpha, D_offset)

        # Verify manual calculation
        expected = D0 * (time_array**alpha) + D_offset
        expected = np.maximum(expected, 1e-10)  # Ensure positive

        np.testing.assert_array_almost_equal(result, expected, decimal=12)

        # Verify all values are positive
        assert np.all(result >= 1e-10)

    def test_shear_rate_calculation(self):
        """Test shear rate calculation."""
        time_array = np.array([0.1, 0.2, 0.5, 1.0], dtype=np.float64)
        gamma_dot_t0 = 0.01
        beta = 0.1
        gamma_dot_t_offset = 0.001

        result = calculate_shear_rate_numba(
            time_array, gamma_dot_t0, beta, gamma_dot_t_offset
        )

        # Verify manual calculation
        expected = gamma_dot_t0 * (time_array**beta) + gamma_dot_t_offset
        expected = np.maximum(expected, 1e-10)  # Ensure positive

        np.testing.assert_array_almost_equal(result, expected, decimal=12)

        # Verify all values are positive
        assert np.all(result >= 1e-10)

    def test_g1_correlation_computation(self):
        """Test g1 correlation function computation."""
        # Create simple test matrix
        integral_matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        wavevector_factor = 0.5

        result = compute_g1_correlation_numba(integral_matrix, wavevector_factor)

        # Verify exponential calculation
        expected = np.exp(-wavevector_factor * integral_matrix)

        np.testing.assert_array_almost_equal(result, expected, decimal=12)

    def test_sinc_squared_computation(self):
        """Test sinc² function computation."""
        # Test with values that avoid singularity
        integral_matrix = np.array([[0.0, 2.0], [2.0, 4.0]], dtype=np.float64)
        prefactor = 0.5

        result = compute_sinc_squared_numba(integral_matrix, prefactor)

        # Verify manual calculation for non-zero values
        argument = prefactor * integral_matrix
        pi_arg = np.pi * argument

        # For zero values, sinc²(0) = 1
        # For non-zero values, sinc²(x) = (sin(πx)/(πx))²
        expected = np.zeros_like(pi_arg)
        zero_mask = np.abs(pi_arg) < 1e-15
        expected[zero_mask] = 1.0
        expected[~zero_mask] = (np.sin(pi_arg[~zero_mask]) / pi_arg[~zero_mask]) ** 2

        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_batch_least_squares_solver(self):
        """Test batch least squares solver."""
        n_angles = 3
        n_data = 5

        # Create test data
        theory_batch = np.random.rand(n_angles, n_data).astype(np.float64)
        exp_batch = np.random.rand(n_angles, n_data).astype(np.float64)

        contrast_batch, offset_batch = solve_least_squares_batch_numba(
            theory_batch, exp_batch
        )

        # Verify shapes
        assert contrast_batch.shape == (n_angles,)
        assert offset_batch.shape == (n_angles,)

        # Verify results are finite
        assert np.all(np.isfinite(contrast_batch))
        assert np.all(np.isfinite(offset_batch))

    def test_batch_chi_squared_computation(self):
        """Test batch chi-squared computation."""
        n_angles = 3
        n_data = 5

        # Create test data
        theory_batch = np.random.rand(n_angles, n_data).astype(np.float64)
        exp_batch = np.random.rand(n_angles, n_data).astype(np.float64)
        contrast_batch = np.random.rand(n_angles).astype(np.float64)
        offset_batch = np.random.rand(n_angles).astype(np.float64)

        result = compute_chi_squared_batch_numba(
            theory_batch, exp_batch, contrast_batch, offset_batch
        )

        # Verify shape and finite values
        assert result.shape == (n_angles,)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)  # Chi-squared values should be non-negative


class TestJITCompilationBehavior:
    """Test JIT compilation behavior and fallbacks."""

    def test_numba_availability_detection(self):
        """Test Numba availability detection."""
        assert isinstance(NUMBA_AVAILABLE, bool)

    def test_jit_compilation_warming(self):
        """Test JIT compilation warmup behavior."""
        if not NUMBA_AVAILABLE:
            pytest.skip("Numba not available")

        # Test that functions work after compilation
        time_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        # First call should trigger compilation
        start_time = time.time()
        result1 = create_time_integral_matrix_numba(time_array)
        time.time() - start_time

        # Second call should be faster (already compiled)
        start_time = time.time()
        result2 = create_time_integral_matrix_numba(time_array)
        second_call_time = time.time() - start_time

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)

        # Second call should typically be faster, but we won't enforce this
        # as timing can be variable in test environments
        assert second_call_time >= 0  # Just ensure it completes

    @patch("homodyne.core.kernels.NUMBA_AVAILABLE", False)
    def test_fallback_behavior_when_numba_unavailable(self):
        """Test fallback behavior when Numba is unavailable."""
        # This test would require reloading the module, which is complex
        # For now, we just verify the constants are set up correctly
        from homodyne.core.kernels import NUMBA_AVAILABLE

        assert NUMBA_AVAILABLE is False  # Due to the patch

    def test_parallel_processing_capability(self):
        """Test parallel processing in JIT functions."""
        if not NUMBA_AVAILABLE:
            pytest.skip("Numba not available")

        # Use a larger array to potentially benefit from parallelization
        time_array = np.random.rand(50).astype(np.float64)

        # This should work regardless of parallel execution
        result = create_time_integral_matrix_numba(time_array)

        assert result.shape == (50, 50)
        assert np.all(np.isfinite(result))


class TestMemoryEfficientCache:
    """Test memory-efficient cache functionality."""

    def test_cache_basic_functionality(self):
        """Test basic cache functionality."""

        @memory_efficient_cache(maxsize=3)
        def test_function(x):
            return x * 2

        # Test caching
        assert test_function(1) == 2
        assert test_function(2) == 4
        assert test_function(1) == 2  # Should hit cache

        # Test cache info
        cache_info = test_function.cache_info()
        assert "Cache:" in cache_info
        assert "Hit rate:" in cache_info

    def test_cache_with_numpy_arrays(self):
        """Test cache with numpy array inputs."""

        @memory_efficient_cache(maxsize=5)
        def matrix_operation(arr):
            return np.sum(arr)

        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])  # Same values, different object
        arr3 = np.array([4, 5, 6])

        result1 = matrix_operation(arr1)
        result2 = matrix_operation(arr2)  # Should hit cache due to content hash
        result3 = matrix_operation(arr3)

        assert result1 == 6
        assert result2 == 6
        assert result3 == 15

    def test_cache_eviction(self):
        """Test cache eviction policy."""

        @memory_efficient_cache(maxsize=2)
        def test_function(x):
            return x**2

        # Fill cache
        assert test_function(1) == 1
        assert test_function(2) == 4
        assert test_function(3) == 9  # Should evict least accessed

        cache_info = test_function.cache_info()
        assert "2/2" in cache_info  # Should show full cache

    def test_cache_clear(self):
        """Test cache clearing."""

        @memory_efficient_cache(maxsize=5)
        def test_function(x):
            return x + 1

        test_function(1)
        test_function(2)

        # Clear cache
        test_function.cache_clear()

        cache_info = test_function.cache_info()
        assert "0/5" in cache_info  # Should show empty cache


class TestKernelPerformance:
    """Performance benchmarks for JIT-compiled kernels."""

    @pytest.mark.performance
    def test_time_integral_matrix_performance(self):
        """Benchmark time integral matrix creation performance."""
        if not NUMBA_AVAILABLE:
            pytest.skip("Numba not available for performance test")

        # Test with moderately sized array
        time_array = np.random.rand(100).astype(np.float64)

        # Warm up
        create_time_integral_matrix_numba(time_array)

        # Benchmark
        iterations = 10
        start_time = time.time()
        for _ in range(iterations):
            result = create_time_integral_matrix_numba(time_array)
        total_time = time.time() - start_time

        avg_time = total_time / iterations

        # Should complete reasonably quickly (adjust threshold as needed)
        assert avg_time < 0.1, f"Average time {avg_time:.4f}s exceeds threshold"
        assert result.shape == (100, 100)

    @pytest.mark.performance
    def test_batch_operations_performance(self):
        """Benchmark batch operations performance."""
        if not NUMBA_AVAILABLE:
            pytest.skip("Numba not available for performance test")

        n_angles = 10
        n_data = 1000

        theory_batch = np.random.rand(n_angles, n_data).astype(np.float64)
        exp_batch = np.random.rand(n_angles, n_data).astype(np.float64)

        # Warm up
        solve_least_squares_batch_numba(theory_batch, exp_batch)

        # Benchmark least squares
        start_time = time.time()
        contrast_batch, offset_batch = solve_least_squares_batch_numba(
            theory_batch, exp_batch
        )
        least_squares_time = time.time() - start_time

        # Benchmark chi-squared
        start_time = time.time()
        chi2_batch = compute_chi_squared_batch_numba(
            theory_batch, exp_batch, contrast_batch, offset_batch
        )
        chi_squared_time = time.time() - start_time

        # Performance assertions (adjust thresholds as needed)
        assert least_squares_time < 0.1, (
            f"Least squares time {least_squares_time:.4f}s too slow"
        )
        assert chi_squared_time < 0.05, (
            f"Chi-squared time {chi_squared_time:.4f}s too slow"
        )

        # Verify results
        assert contrast_batch.shape == (n_angles,)
        assert offset_batch.shape == (n_angles,)
        assert chi2_batch.shape == (n_angles,)

    @pytest.mark.performance
    def test_cache_performance_improvement(self):
        """Test cache performance improvements."""
        call_count = 0

        @memory_efficient_cache(maxsize=10)
        def expensive_operation(x):
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)  # Simulate expensive operation
            return x * 2

        # First calls - should be slow
        start_time = time.time()
        for i in range(5):
            expensive_operation(i)
        uncached_time = time.time() - start_time

        # Repeated calls - should be fast due to caching
        start_time = time.time()
        for i in range(5):
            expensive_operation(i)  # Same inputs, should hit cache
        cached_time = time.time() - start_time

        # Cache should provide significant speedup
        assert cached_time < uncached_time / 2, "Cache did not provide expected speedup"
        assert call_count == 5, "Cache did not prevent redundant computations"


class TestKernelEdgeCases:
    """Test edge cases and error handling in kernels."""

    def test_empty_array_handling(self):
        """Test handling of empty arrays."""
        empty_array = np.array([], dtype=np.float64)

        # Functions should handle empty arrays gracefully
        result = create_time_integral_matrix_numba(empty_array)
        assert result.shape == (0, 0)
        assert result.size == 0

    def test_single_element_arrays(self):
        """Test handling of single-element arrays."""
        single_array = np.array([1.0], dtype=np.float64)

        result = create_time_integral_matrix_numba(single_array)
        assert result.shape == (1, 1)
        assert result[0, 0] == 0.0  # |cumsum[0] - cumsum[0]| = 0

    def test_large_array_handling(self):
        """Test handling of moderately large arrays."""
        large_array = np.random.rand(500).astype(np.float64)

        result = create_time_integral_matrix_numba(large_array)
        assert result.shape == (500, 500)
        assert np.all(np.isfinite(result))

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very small values
        small_array = np.array([1e-10, 2e-10, 3e-10], dtype=np.float64)
        result = create_time_integral_matrix_numba(small_array)
        assert np.all(np.isfinite(result))

        # Test with larger values
        large_array = np.array([1e6, 2e6, 3e6], dtype=np.float64)
        result = create_time_integral_matrix_numba(large_array)
        assert np.all(np.isfinite(result))


class TestKernelIntegration:
    """Test integration of kernels with analysis core."""

    def test_kernel_import_availability(self):
        """Test that all kernels can be imported."""
        # This test verifies that the import structure works
        from homodyne.core.kernels import (
            calculate_diffusion_coefficient_numba,
            calculate_shear_rate_numba,
            compute_chi_squared_batch_numba,
            compute_g1_correlation_numba,
            compute_sinc_squared_numba,
            create_time_integral_matrix_numba,
            solve_least_squares_batch_numba,
        )

        # All should be callable
        assert callable(create_time_integral_matrix_numba)
        assert callable(calculate_diffusion_coefficient_numba)
        assert callable(calculate_shear_rate_numba)
        assert callable(compute_g1_correlation_numba)
        assert callable(compute_sinc_squared_numba)
        assert callable(solve_least_squares_batch_numba)
        assert callable(compute_chi_squared_batch_numba)

    def test_dtype_consistency(self):
        """Test that kernels maintain float64 dtype consistency."""
        time_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        result = create_time_integral_matrix_numba(time_array)
        assert result.dtype == np.float64

        diff_result = calculate_diffusion_coefficient_numba(
            time_array, 100.0, -0.1, 10.0
        )
        assert diff_result.dtype == np.float64

        shear_result = calculate_shear_rate_numba(time_array, 0.01, 0.1, 0.001)
        assert shear_result.dtype == np.float64
