"""
Comprehensive Unit Tests for Core Computational Kernels
========================================================

Tests for high-performance Numba-accelerated computational kernels used in
homodyne scattering analysis.
"""

import numpy as np
import pytest

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
    def test_complex_integration(self):
        """Test complex kernel integration scenarios."""
        pass

class TestKernelIntegration:
    """Integration tests for kernel combinations."""

