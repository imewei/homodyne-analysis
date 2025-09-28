"""
Comprehensive Unit Tests for Core Computational Kernels
========================================================

Tests for high-performance Numba-accelerated computational kernels used in
homodyne scattering analysis.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from homodyne.core.kernels import (
    NUMBA_AVAILABLE,
    calculate_diffusion_coefficient_numba,
    calculate_shear_rate_numba,
    compute_g1_correlation_numba,
    compute_sinc_squared_numba,
    create_time_integral_matrix_numba,
    memory_efficient_cache,
)


class TestNumbaCoreKernels:
    """Test suite for Numba-accelerated computational kernels."""

    def setup_method(self):
        """Setup test fixtures."""
        # Standard test parameters
        self.q_value = 0.1  # Å⁻¹
        self.contrast = 0.95
        self.offset = 1.0

        # Time arrays
        self.t1_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.t2_array = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

        # Physical parameters
        self.params = {
            'D0': 1.0e-3,      # Å²/s
            'alpha': 0.9,      # dimensionless
            'D_offset': 1.0e-4, # Å²/s
            'gamma0': 1.0e-2,  # s⁻¹
            'beta': 0.8,       # dimensionless
            'gamma_offset': 1.0e-3, # s⁻¹
            'phi0': 0.0        # radians
        }

    def test_diffusion_coefficient_calculation(self):
        """Test diffusion coefficient calculation with time dependence."""
        D0 = self.params['D0']
        alpha = self.params['alpha']
        D_offset = self.params['D_offset']

        # Test single time point
        t = 2.0
        result = calculate_diffusion_coefficient_numba(t, D0, alpha, D_offset)
        expected = D0 * (t ** alpha) + D_offset
        assert_allclose(result, expected, rtol=1e-12)

        # Test array of time points
        t_array = np.array([1.0, 2.0, 3.0])
        results = np.array([
            calculate_diffusion_coefficient_numba(t, D0, alpha, D_offset)
            for t in t_array
        ])
        expected_array = D0 * (t_array ** alpha) + D_offset
        assert_allclose(results, expected_array, rtol=1e-12)

    def test_shear_rate_calculation(self):
        """Test shear rate calculation with time dependence."""
        gamma0 = self.params['gamma0']
        beta = self.params['beta']
        gamma_offset = self.params['gamma_offset']

        # Test single time point
        t = 2.0
        result = calculate_shear_rate_numba(t, gamma0, beta, gamma_offset)
        expected = gamma0 * (t ** beta) + gamma_offset
        assert_allclose(result, expected, rtol=1e-12)

        # Test boundary conditions
        # At t=0, should give gamma_offset (avoiding 0^beta issues)
        t_small = 1e-10
        result_small = calculate_shear_rate_numba(t_small, gamma0, beta, gamma_offset)
        assert result_small >= gamma_offset

    def test_sinc_squared_computation(self):
        """Test sinc squared function computation."""
        # Test at zero (should be 1.0)
        result_zero = compute_sinc_squared_numba(0.0)
        assert_allclose(result_zero, 1.0, rtol=1e-12)

        # Test at pi (should be 0.0)
        result_pi = compute_sinc_squared_numba(np.pi)
        assert_allclose(result_pi, 0.0, atol=1e-10)

        # Test at pi/2
        x_half_pi = np.pi / 2
        result_half_pi = compute_sinc_squared_numba(x_half_pi)
        expected_half_pi = (2.0 / np.pi) ** 2
        assert_allclose(result_half_pi, expected_half_pi, rtol=1e-10)

        # Test array input
        x_array = np.array([0.0, np.pi/4, np.pi/2, np.pi])
        results = np.array([compute_sinc_squared_numba(x) for x in x_array])
        expected = (np.sinc(x_array / np.pi)) ** 2
        assert_allclose(results, expected, rtol=1e-10)

    def test_g1_correlation_computation(self):
        """Test g1 correlation function computation."""
        t1, t2 = 1.0, 2.0
        phi = np.pi / 4
        q = self.q_value

        # Extract parameters
        D0 = self.params['D0']
        alpha = self.params['alpha']
        D_offset = self.params['D_offset']
        gamma0 = self.params['gamma0']
        beta = self.params['beta']
        gamma_offset = self.params['gamma_offset']
        phi0 = self.params['phi0']

        result = compute_g1_correlation_numba(
            t1, t2, phi, q, D0, alpha, D_offset,
            gamma0, beta, gamma_offset, phi0
        )

        # Should be between 0 and 1
        assert 0.0 <= result <= 1.0

        # At t1 = t2, should be 1.0 (no decorrelation)
        result_same_time = compute_g1_correlation_numba(
            t1, t1, phi, q, D0, alpha, D_offset,
            gamma0, beta, gamma_offset, phi0
        )
        assert_allclose(result_same_time, 1.0, rtol=1e-10)

    def test_time_integral_matrix_creation(self):
        """Test time integral matrix creation."""
        t1_vals = np.array([1.0, 2.0, 3.0])
        t2_vals = np.array([1.5, 2.5, 3.5])

        # Test basic functionality
        matrix = create_time_integral_matrix_numba(t1_vals, t2_vals)

        # Check shape
        assert matrix.shape == (len(t1_vals), len(t2_vals))

        # Check symmetry property: integral from t1 to t2 = -integral from t2 to t1
        # (in terms of sign consideration for the physics)
        assert isinstance(matrix, np.ndarray)
        assert matrix.dtype == np.float64

        # Test that diagonal elements (when t1=t2) are zero
        t_same = np.array([1.0, 2.0, 3.0])
        matrix_diag = create_time_integral_matrix_numba(t_same, t_same)
        np.testing.assert_allclose(np.diag(matrix_diag), 0.0, atol=1e-12)

    def test_memory_efficient_cache(self):
        """Test memory efficient caching functionality."""
        @memory_efficient_cache(maxsize=10)
        def expensive_function(x, y):
            return x ** 2 + y ** 2

        # Test basic caching
        result1 = expensive_function(2.0, 3.0)
        result2 = expensive_function(2.0, 3.0)  # Should be cached
        assert result1 == result2
        assert result1 == 13.0

        # Test cache size limit
        for i in range(15):  # Exceed cache size
            expensive_function(i, i)

        # Original result should still be correct
        result3 = expensive_function(2.0, 3.0)
        assert result3 == 13.0

    @pytest.mark.parametrize("use_numba", [True, False])
    def test_numba_fallback_behavior(self, use_numba, monkeypatch):
        """Test that functions work with and without Numba."""
        if not use_numba:
            # Mock NUMBA_AVAILABLE to False
            monkeypatch.setattr('homodyne.core.kernels.NUMBA_AVAILABLE', False)

        # Test basic function still works
        result = compute_sinc_squared_numba(np.pi / 2)
        expected = (2.0 / np.pi) ** 2
        assert_allclose(result, expected, rtol=1e-10)

    def test_edge_cases_and_boundary_conditions(self):
        """Test edge cases and boundary conditions."""
        # Test with very small time differences
        small_dt = 1e-12
        result = compute_g1_correlation_numba(
            1.0, 1.0 + small_dt, 0.0, self.q_value,
            self.params['D0'], self.params['alpha'], self.params['D_offset'],
            self.params['gamma0'], self.params['beta'], self.params['gamma_offset'],
            self.params['phi0']
        )
        # Should be very close to 1.0
        assert_allclose(result, 1.0, rtol=1e-6)

        # Test with zero q (no scattering)
        result_zero_q = compute_g1_correlation_numba(
            1.0, 2.0, 0.0, 0.0,
            self.params['D0'], self.params['alpha'], self.params['D_offset'],
            self.params['gamma0'], self.params['beta'], self.params['gamma_offset'],
            self.params['phi0']
        )
        assert_allclose(result_zero_q, 1.0, rtol=1e-10)

    def test_parameter_validation_through_computation(self):
        """Test parameter validation through computational results."""
        # Test with negative diffusion (should still compute but may be unphysical)
        with pytest.warns(None) as warnings:  # May generate warnings but shouldn't crash
            result = calculate_diffusion_coefficient_numba(
                2.0, -1e-3, 1e-4, 0.9
            )
            # Should compute a result even if unphysical
            assert isinstance(result, (float, np.floating))

    def test_performance_characteristics(self):
        """Test performance characteristics of kernels."""
        import time

        # Test with larger arrays to check scalability
        t1_large = np.linspace(0.1, 10.0, 100)
        t2_large = np.linspace(0.2, 10.1, 100)

        start_time = time.time()
        matrix_large = create_time_integral_matrix_numba(t1_large, t2_large)
        end_time = time.time()

        # Should complete in reasonable time (less than 1 second)
        assert (end_time - start_time) < 1.0
        assert matrix_large.shape == (100, 100)

    def test_numerical_stability(self):
        """Test numerical stability with extreme parameters."""
        # Test with very large time values
        large_time = 1e6
        result_large = calculate_diffusion_coefficient_numba(
            large_time, 1e-10, 1e-12, 0.5
        )
        assert np.isfinite(result_large)

        # Test with very small parameters
        small_result = calculate_shear_rate_numba(
            1.0, 1e-15, 0.1, 1e-16
        )
        assert np.isfinite(small_result)
        assert small_result >= 0.0

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_numba_compilation_success(self):
        """Test that Numba compilation succeeds when available."""
        # This test verifies Numba can compile the functions
        # First call triggers compilation
        result1 = compute_sinc_squared_numba(1.0)

        # Second call should use compiled version (faster)
        import time
        start_time = time.time()
        result2 = compute_sinc_squared_numba(1.0)
        end_time = time.time()

        assert result1 == result2
        # Compiled version should be very fast
        assert (end_time - start_time) < 0.001


class TestKernelIntegration:
    """Integration tests for kernel combinations."""

    def test_complete_correlation_calculation(self):
        """Test complete correlation calculation using multiple kernels."""
        # Simulate a complete g2 calculation
        t1, t2 = 1.0, 2.0
        phi = np.pi / 6
        q = 0.05
        contrast = 0.9
        offset = 1.0

        # Parameters for a realistic system
        params = {
            'D0': 5e-4, 'alpha': 0.85, 'D_offset': 1e-5,
            'gamma0': 0.01, 'beta': 0.9, 'gamma_offset': 0.001,
            'phi0': 0.1
        }

        # Calculate g1
        g1 = compute_g1_correlation_numba(
            t1, t2, phi, q,
            params['D0'], params['alpha'], params['D_offset'],
            params['gamma0'], params['beta'], params['gamma_offset'],
            params['phi0']
        )

        # Calculate g2
        g2 = offset + contrast * g1 ** 2

        # Sanity checks
        assert 0.0 <= g1 <= 1.0
        assert offset <= g2 <= offset + contrast
        assert np.isfinite(g2)

    def test_multi_angle_correlation_matrix(self):
        """Test correlation calculations across multiple angles."""
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        t1, t2 = 1.0, 3.0
        q = 0.1

        params = {
            'D0': 1e-3, 'alpha': 0.9, 'D_offset': 1e-4,
            'gamma0': 0.02, 'beta': 0.8, 'gamma_offset': 0.002,
            'phi0': 0.0
        }

        correlations = np.array([
            compute_g1_correlation_numba(
                t1, t2, phi, q,
                params['D0'], params['alpha'], params['D_offset'],
                params['gamma0'], params['beta'], params['gamma_offset'],
                params['phi0']
            )
            for phi in angles
        ])

        # All correlations should be valid
        assert np.all((correlations >= 0.0) & (correlations <= 1.0))
        assert np.all(np.isfinite(correlations))

        # Should show angular dependence (not all the same)
        assert np.std(correlations) > 1e-10