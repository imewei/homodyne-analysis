"""
Comprehensive Scientific Computing Validation Tests
===================================================

Scientific validation tests for numerical accuracy, physical consistency,
and mathematical correctness of homodyne scattering analysis algorithms.
"""

import numpy as np
import pytest

try:
    from homodyne.core.kernels import (
        compute_g1_correlation_numba,
        compute_sinc_squared_numba,
        calculate_diffusion_coefficient_numba,
        calculate_shear_rate_numba
    )
    SCIENTIFIC_MODULES_AVAILABLE = True
except ImportError:
    SCIENTIFIC_MODULES_AVAILABLE = False


class TestPhysicalConsistency:
    """Test physical consistency of computational results."""

    def setup_method(self):
        """Setup physical consistency tests."""
        # Physical constants and realistic parameters
        self.realistic_params = {
            'D0': 1e-3,      # Å²/s - typical for colloidal particles
            'alpha': 0.9,    # dimensionless - sub-diffusive behavior
            'D_offset': 1e-4, # Å²/s - baseline diffusion
            'gamma0': 0.01,  # s⁻¹ - typical shear rate
            'beta': 0.8,     # dimensionless - shear rate time dependence
            'gamma_offset': 0.001, # s⁻¹ - baseline shear
            'phi0': 0.0      # radians - angular offset
        }

        # Experimental parameters
        self.experimental_params = {
            'q_value': 0.1,  # Å⁻¹ - scattering vector magnitude
            'contrast': 0.95, # dimensionless - detector contrast
            'offset': 1.0     # dimensionless - baseline correlation
        }

    @pytest.mark.skipif(not SCIENTIFIC_MODULES_AVAILABLE, reason="Scientific modules not available")
    def test_correlation_function_bounds(self):
        """Test that correlation functions respect physical bounds."""
        # g1 correlation should be between 0 and 1
        t1, t2 = 1.0, 2.0
        phi = np.pi / 4
        q = self.experimental_params['q_value']

        g1 = compute_g1_correlation_numba(
            t1, t2, phi, q,
            self.realistic_params['D0'],
            self.realistic_params['alpha'],
            self.realistic_params['D_offset'],
            self.realistic_params['gamma0'],
            self.realistic_params['beta'],
            self.realistic_params['gamma_offset'],
            self.realistic_params['phi0']
        )

        assert 0.0 <= g1 <= 1.0, f"g1 correlation out of bounds: {g1}"

        # g2 correlation should be >= 1 (Siegert relation)
        g2 = self.experimental_params['offset'] + self.experimental_params['contrast'] * g1**2
        assert g2 >= 1.0, f"g2 correlation violates Siegert relation: {g2}"

        # At zero time delay, g1 should be 1
        g1_zero_delay = compute_g1_correlation_numba(
            t1, t1, phi, q,
            self.realistic_params['D0'],
            self.realistic_params['alpha'],
            self.realistic_params['D_offset'],
            self.realistic_params['gamma0'],
            self.realistic_params['beta'],
            self.realistic_params['gamma_offset'],
            self.realistic_params['phi0']
        )

        assert_allclose(g1_zero_delay, 1.0, rtol=1e-10)

    @pytest.mark.skipif(not SCIENTIFIC_MODULES_AVAILABLE, reason="Scientific modules not available")
    def test_time_decay_monotonicity(self):
        """Test that correlation functions decay monotonically with time."""
        phi = 0.0  # Isotropic case for simplicity
        q = self.experimental_params['q_value']
        t1 = 1.0

        # Test correlation at increasing time delays
        time_delays = np.linspace(0.0, 5.0, 20)
        correlations = []

        for dt in time_delays:
            t2 = t1 + dt
            g1 = compute_g1_correlation_numba(
                t1, t2, phi, q,
                self.realistic_params['D0'],
                self.realistic_params['alpha'],
                self.realistic_params['D_offset'],
                self.realistic_params['gamma0'],
                self.realistic_params['beta'],
                self.realistic_params['gamma_offset'],
                self.realistic_params['phi0']
            )
            correlations.append(g1)

        correlations = np.array(correlations)

        # Correlation should be monotonically decreasing (allowing for numerical noise)
        for i in range(len(correlations) - 1):
            # Allow small increases due to numerical precision
            assert correlations[i] >= correlations[i+1] - 1e-10, \
                f"Non-monotonic decay at index {i}: {correlations[i]} -> {correlations[i+1]}"

        # First value should be 1, last should be significantly smaller
        assert_allclose(correlations[0], 1.0, rtol=1e-10)
        assert correlations[-1] < 0.9 * correlations[0]

    @pytest.mark.skipif(not SCIENTIFIC_MODULES_AVAILABLE, reason="Scientific modules not available")
    def test_angular_dependence_symmetry(self):
        """Test angular dependence and symmetry properties."""
        t1, t2 = 1.0, 3.0
        q = self.experimental_params['q_value']

        # Test correlation at different angles
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        correlations = []

        for phi in angles:
            g1 = compute_g1_correlation_numba(
                t1, t2, phi, q,
                self.realistic_params['D0'],
                self.realistic_params['alpha'],
                self.realistic_params['D_offset'],
                self.realistic_params['gamma0'],
                self.realistic_params['beta'],
                self.realistic_params['gamma_offset'],
                self.realistic_params['phi0']
            )
            correlations.append(g1)

        correlations = np.array(correlations)

        # Should show 2π periodicity
        g1_0 = correlations[0]
        g1_2pi = compute_g1_correlation_numba(
            t1, t2, 2*np.pi, q,
            self.realistic_params['D0'],
            self.realistic_params['alpha'],
            self.realistic_params['D_offset'],
            self.realistic_params['gamma0'],
            self.realistic_params['beta'],
            self.realistic_params['gamma_offset'],
            self.realistic_params['phi0']
        )

        assert_allclose(g1_0, g1_2pi, rtol=1e-10)

        # For flow systems, should show angular variation
        if self.realistic_params['gamma0'] > 0:
            correlation_variation = np.max(correlations) - np.min(correlations)
            assert correlation_variation > 1e-6, "No angular dependence detected in flow system"

    @pytest.mark.skipif(not SCIENTIFIC_MODULES_AVAILABLE, reason="Scientific modules not available")
    def test_diffusion_coefficient_positivity(self):
        """Test that diffusion coefficients remain positive."""
        time_points = np.logspace(-2, 2, 50)  # 0.01 to 100 seconds

        for t in time_points:
            D_t = calculate_diffusion_coefficient_numba(
                t,
                self.realistic_params['D0'],
                self.realistic_params['alpha'],
                self.realistic_params['D_offset']
            )

            assert D_t > 0, f"Negative diffusion coefficient at t={t}: {D_t}"

            # Should be finite
            assert np.isfinite(D_t), f"Non-finite diffusion coefficient at t={t}: {D_t}"

    @pytest.mark.skipif(not SCIENTIFIC_MODULES_AVAILABLE, reason="Scientific modules not available")
    def test_shear_rate_behavior(self):
        """Test shear rate behavior and physical constraints."""
        time_points = np.logspace(-2, 2, 50)

        for t in time_points:
            gamma_t = calculate_shear_rate_numba(
                t,
                self.realistic_params['gamma0'],
                self.realistic_params['beta'],
                self.realistic_params['gamma_offset']
            )

            # Shear rate should be non-negative for physical systems
            assert gamma_t >= 0, f"Negative shear rate at t={t}: {gamma_t}"

            # Should be finite
            assert np.isfinite(gamma_t), f"Non-finite shear rate at t={t}: {gamma_t}"

    @pytest.mark.skipif(not SCIENTIFIC_MODULES_AVAILABLE, reason="Scientific modules not available")
    def test_static_limit_consistency(self):
        """Test consistency in static limit (no flow)."""
        # Set flow parameters to zero
        static_params = self.realistic_params.copy()
        static_params['gamma0'] = 0.0
        static_params['gamma_offset'] = 0.0

        t1, t2 = 1.0, 3.0
        q = self.experimental_params['q_value']

        # Correlation should be independent of angle in static case
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        correlations = []

        for phi in angles:
            g1 = compute_g1_correlation_numba(
                t1, t2, phi, q,
                static_params['D0'],
                static_params['alpha'],
                static_params['D_offset'],
                static_params['gamma0'],
                static_params['beta'],
                static_params['gamma_offset'],
                static_params['phi0']
            )
            correlations.append(g1)

        correlations = np.array(correlations)

        # All correlations should be nearly identical in static case
        correlation_std = np.std(correlations)
        assert correlation_std < 1e-10, f"Angular dependence in static case: std = {correlation_std}"


class TestNumericalAccuracy:
    """Test numerical accuracy and stability."""

    def setup_method(self):
        """Setup numerical accuracy tests."""
        self.tolerance = 1e-12  # High precision requirement

    @pytest.mark.skipif(not SCIENTIFIC_MODULES_AVAILABLE, reason="Scientific modules not available")
    def test_sinc_function_accuracy(self):
        """Test accuracy of sinc squared function against reference implementation."""
        # Test points including special cases
        test_points = np.array([
            0.0,           # sinc(0) = 1
            np.pi,         # sinc(π) = 0
            np.pi/2,       # sinc(π/2) = 2/π
            2*np.pi,       # sinc(2π) = 0
            0.1, 0.5, 1.0, 1.5, 2.0, 3.0
        ])

        for x in test_points:
            computed = compute_sinc_squared_numba(x)

            # Reference implementation using scipy
            if x == 0:
                expected = 1.0
            else:
                expected = (np.sin(x) / x) ** 2

            assert_allclose(computed, expected, rtol=self.tolerance,
                           err_msg=f"Sinc accuracy error at x={x}")

    @pytest.mark.skipif(not SCIENTIFIC_MODULES_AVAILABLE, reason="Scientific modules not available")
    def test_integration_accuracy(self):
        """Test accuracy of time integration components."""
        # Test against analytical solutions where possible

        # For constant diffusion (α = 0), integral should be D_offset * |t2 - t1|
        t1, t2 = 1.0, 3.0
        D0 = 0.0  # No time-dependent part
        alpha = 0.0
        D_offset = 1e-3

        # The integral ∫|t2-t1| D(t') dt' should equal D_offset * |t2 - t1|
        expected_integral = D_offset * abs(t2 - t1)

        # Test through correlation function with very small q (linear regime)
        q_small = 1e-6
        phi = 0.0

        g1 = compute_g1_correlation_numba(
            t1, t2, phi, q_small, D0, alpha, D_offset,
            0.0, 0.0, 0.0, 0.0  # No shear
        )

        # In the small q limit: g1 ≈ exp(-q²/2 * integral)
        # So: integral ≈ -2 * ln(g1) / q²
        computed_integral = -2 * np.log(g1) / (q_small**2)

        assert_allclose(computed_integral, expected_integral, rtol=1e-3,
                       err_msg="Integration accuracy error in constant diffusion case")

    @pytest.mark.skipif(not SCIENTIFIC_MODULES_AVAILABLE, reason="Scientific modules not available")
    def test_parameter_scaling_independence(self):
        """Test that results are independent of parameter scaling."""
        # Test with different scales of the same physical parameters
        base_params = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]

        # Scale all parameters by factors
        scale_factors = [0.1, 1.0, 10.0]
        t1, t2 = 1.0, 2.0
        phi = np.pi / 4
        q = 0.1

        correlations = []

        for scale in scale_factors:
            # Scale diffusion and shear parameters appropriately
            scaled_params = base_params.copy()
            scaled_params[0] *= scale  # D0
            scaled_params[2] *= scale  # D_offset
            scaled_params[3] *= scale  # gamma0
            scaled_params[5] *= scale  # gamma_offset

            g1 = compute_g1_correlation_numba(t1, t2, phi, q, *scaled_params)
            correlations.append(g1)

        # Results should be similar (though not identical due to nonlinear effects)
        correlations = np.array(correlations)
        relative_variation = np.std(correlations) / np.mean(correlations)

        # Allow some variation due to nonlinear physics, but should be consistent
        assert relative_variation < 0.5, f"Large parameter scaling dependence: {relative_variation}"

    @pytest.mark.skipif(not SCIENTIFIC_MODULES_AVAILABLE, reason="Scientific modules not available")
    def test_numerical_stability_extreme_parameters(self):
        """Test numerical stability with extreme but valid parameters."""
        extreme_cases = [
            # Very small diffusion
            [1e-10, 0.9, 1e-12, 1e-6, 0.8, 1e-8, 0.0],
            # Very large diffusion
            [1e-1, 0.9, 1e-3, 1e-1, 0.8, 1e-3, 0.0],
            # Extreme time dependencies
            [1e-3, 0.1, 1e-4, 0.01, 0.1, 0.001, 0.0],
            [1e-3, 1.9, 1e-4, 0.01, 1.9, 0.001, 0.0],
        ]

        t1, t2 = 1.0, 2.0
        phi = 0.0
        q = 0.1

        for params in extreme_cases:
            g1 = compute_g1_correlation_numba(t1, t2, phi, q, *params)

            # Should produce finite, physically reasonable results
            assert np.isfinite(g1), f"Non-finite result with extreme parameters: {params}"
            assert 0.0 <= g1 <= 1.0, f"Unphysical correlation with extreme parameters: {g1}"

    def test_reproducibility(self):
        """Test reproducibility of calculations."""
        if not SCIENTIFIC_MODULES_AVAILABLE:
            pytest.skip("Scientific modules not available")

        # Same calculation should give identical results
        params = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]
        t1, t2 = 1.0, 2.0
        phi = np.pi / 4
        q = 0.1

        # Perform calculation multiple times
        results = []
        for _ in range(10):
            g1 = compute_g1_correlation_numba(t1, t2, phi, q, *params)
            results.append(g1)

        results = np.array(results)

        # All results should be identical
        assert np.all(results == results[0]), "Calculation not reproducible"


class TestMathematicalCorrectness:
    """Test mathematical correctness and consistency."""

    @pytest.mark.skipif(not SCIENTIFIC_MODULES_AVAILABLE, reason="Scientific modules not available")
    def test_siegert_relation(self):
        """Test Siegert relation g2 = 1 + β|g1|²."""
        # Generate realistic correlation data
        t1, t2 = 1.0, 3.0
        phi = np.pi / 6
        q = 0.1
        params = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]

        g1 = compute_g1_correlation_numba(t1, t2, phi, q, *params)

        # Apply Siegert relation
        contrast = 0.95
        offset = 1.0
        g2_theoretical = offset + contrast * g1**2

        # Verify physical consistency
        assert g2_theoretical >= 1.0, "Siegert relation violated: g2 < 1"

        # At zero delay, should have g2 = 1 + β
        g1_zero = compute_g1_correlation_numba(t1, t1, phi, q, *params)
        g2_zero = offset + contrast * g1_zero**2
        expected_g2_zero = 1.0 + contrast  # Since g1(0) = 1

        assert_allclose(g2_zero, expected_g2_zero, rtol=1e-10)

    @pytest.mark.skipif(not SCIENTIFIC_MODULES_AVAILABLE, reason="Scientific modules not available")
    def test_time_reversal_symmetry(self):
        """Test time reversal symmetry in correlation functions."""
        phi = 0.0
        q = 0.1
        params = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]

        # Test correlation for (t1, t2) and (t2, t1)
        t1, t2 = 1.0, 3.0

        g1_forward = compute_g1_correlation_numba(t1, t2, phi, q, *params)
        g1_backward = compute_g1_correlation_numba(t2, t1, phi, q, *params)

        # Should be identical (correlation depends on |t2 - t1|)
        assert_allclose(g1_forward, g1_backward, rtol=1e-12,
                       err_msg="Time reversal symmetry violated")

    @pytest.mark.skipif(not SCIENTIFIC_MODULES_AVAILABLE, reason="Scientific modules not available")
    def test_scaling_relations(self):
        """Test scaling relations in correlation functions."""
        # Test q-scaling: correlation should depend on q² in the exponential
        t1, t2 = 1.0, 2.0
        phi = 0.0
        params = [1e-3, 0.9, 1e-4, 0.0, 0.0, 0.0, 0.0]  # Pure diffusion

        q_values = np.array([0.05, 0.1, 0.2])
        correlations = []

        for q in q_values:
            g1 = compute_g1_correlation_numba(t1, t2, phi, q, *params)
            correlations.append(g1)

        correlations = np.array(correlations)

        # In the pure diffusion case: g1 ∝ exp(-Aq²) where A is a constant
        # So ln(g1) should be proportional to q²
        log_correlations = np.log(correlations)
        q_squared = q_values**2

        # Fit linear relation
        slope, intercept = np.polyfit(q_squared, log_correlations, 1)

        # Should show good linear correlation
        correlation_coeff = np.corrcoef(q_squared, log_correlations)[0, 1]
        assert abs(correlation_coeff) > 0.99, f"Poor q² scaling: correlation = {correlation_coeff}"

        # Slope should be negative (decay with q²)
        assert slope < 0, f"Incorrect q² scaling: slope = {slope}"

    @pytest.mark.skipif(not SCIENTIFIC_MODULES_AVAILABLE, reason="Scientific modules not available")
    def test_flow_anisotropy(self):
        """Test anisotropic effects of flow on correlation functions."""
        t1, t2 = 1.0, 3.0
        q = 0.1

        # Compare isotropic (no flow) vs anisotropic (with flow) cases
        params_static = [1e-3, 0.9, 1e-4, 0.0, 0.0, 0.0, 0.0]  # No flow
        params_flow = [1e-3, 0.9, 1e-4, 0.02, 0.8, 0.002, 0.0]  # With flow

        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)

        # Static case should be isotropic
        correlations_static = []
        for phi in angles:
            g1 = compute_g1_correlation_numba(t1, t2, phi, q, *params_static)
            correlations_static.append(g1)

        correlations_static = np.array(correlations_static)
        static_variation = np.std(correlations_static)

        # Flow case should be anisotropic
        correlations_flow = []
        for phi in angles:
            g1 = compute_g1_correlation_numba(t1, t2, phi, q, *params_flow)
            correlations_flow.append(g1)

        correlations_flow = np.array(correlations_flow)
        flow_variation = np.std(correlations_flow)

        # Static case should have minimal angular variation
        assert static_variation < 1e-10, f"Unexpected anisotropy in static case: {static_variation}"

        # Flow case should have significant angular variation
        assert flow_variation > 1e-6, f"No anisotropy detected in flow case: {flow_variation}"


class TestPhysicalLimits:
    """Test behavior in various physical limits."""

    @pytest.mark.skipif(not SCIENTIFIC_MODULES_AVAILABLE, reason="Scientific modules not available")
    def test_zero_scattering_vector_limit(self):
        """Test behavior as scattering vector approaches zero."""
        t1, t2 = 1.0, 2.0
        phi = 0.0
        params = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]

        # Test with increasingly small q values
        q_values = np.logspace(-6, -1, 10)
        correlations = []

        for q in q_values:
            g1 = compute_g1_correlation_numba(t1, t2, phi, q, *params)
            correlations.append(g1)

        correlations = np.array(correlations)

        # As q → 0, correlation should approach 1
        assert correlations[-1] > 0.99, f"Correlation doesn't approach 1 as q → 0: {correlations[-1]}"

        # Should be monotonically increasing as q decreases
        for i in range(len(correlations) - 1):
            assert correlations[i] <= correlations[i+1] + 1e-10, \
                f"Non-monotonic behavior as q → 0 at index {i}"

    @pytest.mark.skipif(not SCIENTIFIC_MODULES_AVAILABLE, reason="Scientific modules not available")
    def test_long_time_limit(self):
        """Test behavior in the long time limit."""
        t1 = 1.0
        phi = 0.0
        q = 0.1
        params = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]

        # Test with increasing time separations
        time_separations = np.logspace(0, 3, 20)  # 1 to 1000 seconds
        correlations = []

        for dt in time_separations:
            t2 = t1 + dt
            g1 = compute_g1_correlation_numba(t1, t2, phi, q, *params)
            correlations.append(g1)

        correlations = np.array(correlations)

        # Should decay towards zero for long times
        assert correlations[-1] < 0.5 * correlations[0], \
            "Insufficient decay in long time limit"

        # Should be monotonically decreasing
        for i in range(len(correlations) - 1):
            assert correlations[i] >= correlations[i+1] - 1e-10, \
                f"Non-monotonic decay at index {i}"

    @pytest.mark.skipif(not SCIENTIFIC_MODULES_AVAILABLE, reason="Scientific modules not available")
    def test_high_shear_limit(self):
        """Test behavior in high shear rate limit."""
        t1, t2 = 1.0, 2.0
        q = 0.1

        # Compare low vs high shear cases
        params_low_shear = [1e-3, 0.9, 1e-4, 0.001, 0.8, 0.0001, 0.0]
        params_high_shear = [1e-3, 0.9, 1e-4, 1.0, 0.8, 0.1, 0.0]

        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)

        # High shear should show stronger angular dependence
        correlations_low = []
        correlations_high = []

        for phi in angles:
            g1_low = compute_g1_correlation_numba(t1, t2, phi, q, *params_low_shear)
            g1_high = compute_g1_correlation_numba(t1, t2, phi, q, *params_high_shear)
            correlations_low.append(g1_low)
            correlations_high.append(g1_high)

        variation_low = np.std(correlations_low)
        variation_high = np.std(correlations_high)

        # High shear should have larger angular variation
        assert variation_high > variation_low, \
            f"High shear doesn't show increased anisotropy: {variation_high} vs {variation_low}"


class TestValidationAgainstTheory:
    """Validate against known theoretical results."""

    @pytest.mark.skipif(not SCIENTIFIC_MODULES_AVAILABLE, reason="Scientific modules not available")
    def test_pure_brownian_motion(self):
        """Test against known result for pure Brownian motion."""
        # For pure Brownian motion: g1(τ) = exp(-Dq²τ) where τ = |t2 - t1|
        t1, t2 = 1.0, 3.0
        tau = abs(t2 - t1)
        phi = 0.0
        q = 0.1

        # Pure Brownian motion parameters
        D = 1e-3
        params = [D, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # α=1 for normal diffusion

        g1_computed = compute_g1_correlation_numba(t1, t2, phi, q, *params)
        g1_theoretical = np.exp(-D * q**2 * tau)

        assert_allclose(g1_computed, g1_theoretical, rtol=1e-6,
                       err_msg="Deviation from pure Brownian motion theory")

    @pytest.mark.skipif(not SCIENTIFIC_MODULES_AVAILABLE, reason="Scientific modules not available")
    def test_anomalous_diffusion_scaling(self):
        """Test scaling behavior for anomalous diffusion."""
        # For anomalous diffusion with α ≠ 1, test scaling with time
        t1 = 1.0
        phi = 0.0
        q = 0.05  # Small q for cleaner scaling

        # Anomalous diffusion parameters
        D0 = 1e-3
        alpha = 0.7  # Sub-diffusive
        D_offset = 0.0
        params = [D0, alpha, D_offset, 0.0, 0.0, 0.0, 0.0]

        # Test at different time scales
        time_separations = np.array([1.0, 2.0, 4.0, 8.0])
        correlations = []

        for dt in time_separations:
            t2 = t1 + dt
            g1 = compute_g1_correlation_numba(t1, t2, phi, q, *params)
            correlations.append(g1)

        # For anomalous diffusion: integral ∝ t^(α+1)
        # So correlation decay should scale as exp(-const * t^(α+1))
        log_correlations = np.log(correlations)
        time_powers = time_separations**(alpha + 1)

        # Should show good correlation between ln(g1) and t^(α+1)
        correlation_coeff = np.corrcoef(time_powers, log_correlations)[0, 1]
        assert abs(correlation_coeff) > 0.95, \
            f"Poor anomalous diffusion scaling: correlation = {correlation_coeff}"


class TestCrossValidation:
    """Cross-validation tests against different implementations."""

    @pytest.mark.skipif(not SCIENTIFIC_MODULES_AVAILABLE, reason="Scientific modules not available")
    def test_against_scipy_integration(self):
        """Test against scipy numerical integration for simple cases."""
        # For constant parameters, can use scipy to integrate and compare
        t1, t2 = 1.0, 3.0
        phi = 0.0
        q = 0.1

        # Constant diffusion case
        D_const = 1e-3
        params = [0.0, 0.0, D_const, 0.0, 0.0, 0.0, 0.0]

        g1_computed = compute_g1_correlation_numba(t1, t2, phi, q, *params)

        # Reference calculation using scipy
        def integrand(t):
            return D_const

        integral, _ = integrate.quad(integrand, min(t1, t2), max(t1, t2))
        g1_reference = np.exp(-q**2 / 2 * integral)

        assert_allclose(g1_computed, g1_reference, rtol=1e-10,
                       err_msg="Disagreement with scipy integration")

    def test_internal_consistency(self):
        """Test internal consistency between different calculation paths."""
        if not SCIENTIFIC_MODULES_AVAILABLE:
            pytest.skip("Scientific modules not available")

        # Test that the same physical system gives consistent results
        # through different parameter combinations

        # Case 1: D(t) = D0*t^α + D_offset vs equivalent formulation
        t1, t2 = 1.0, 2.0
        phi = 0.0
        q = 0.1

        # Original parameters
        D0_1, alpha_1, D_offset_1 = 1e-3, 0.9, 1e-4
        params1 = [D0_1, alpha_1, D_offset_1, 0.0, 0.0, 0.0, 0.0]

        # Equivalent parameters (should give same physics)
        # D(t) = 2*D0*t^α + 0.5*D_offset is equivalent to different D0, D_offset
        D0_2, alpha_2, D_offset_2 = 2e-3, 0.9, 0.5e-4
        params2 = [D0_2, alpha_2, D_offset_2, 0.0, 0.0, 0.0, 0.0]

        g1_1 = compute_g1_correlation_numba(t1, t2, phi, q, *params1)
        g1_2 = compute_g1_correlation_numba(t1, t2, phi, q, *params2)

        # Should give the same result
        assert_allclose(g1_1, g1_2, rtol=1e-10,
                       err_msg="Internal inconsistency between equivalent parameterizations")


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.skipif(not SCIENTIFIC_MODULES_AVAILABLE, reason="Scientific modules not available")
    def test_invalid_parameter_handling(self):
        """Test handling of invalid parameters."""
        t1, t2 = 1.0, 2.0
        phi = 0.0
        q = 0.1

        # Test with NaN parameters
        params_nan = [np.nan, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]

        with pytest.warns(None):  # May issue warnings
            result = compute_g1_correlation_numba(t1, t2, phi, q, *params_nan)
            # Should either handle gracefully or return NaN
            assert np.isnan(result) or np.isfinite(result)

        # Test with infinite parameters
        params_inf = [np.inf, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]

        with pytest.warns(None):
            result = compute_g1_correlation_numba(t1, t2, phi, q, *params_inf)
            # Should handle gracefully
            assert np.isfinite(result) or np.isnan(result)

    @pytest.mark.skipif(not SCIENTIFIC_MODULES_AVAILABLE, reason="Scientific modules not available")
    def test_edge_case_time_values(self):
        """Test with edge case time values."""
        phi = 0.0
        q = 0.1
        params = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]

        # Test with very small time values
        t1_small, t2_small = 1e-10, 2e-10
        result_small = compute_g1_correlation_numba(t1_small, t2_small, phi, q, *params)
        assert np.isfinite(result_small)
        assert 0.0 <= result_small <= 1.0

        # Test with very large time values
        t1_large, t2_large = 1e6, 2e6
        result_large = compute_g1_correlation_numba(t1_large, t2_large, phi, q, *params)
        assert np.isfinite(result_large)
        assert 0.0 <= result_large <= 1.0

        # Test with zero time difference
        result_zero = compute_g1_correlation_numba(t1_small, t1_small, phi, q, *params)
        assert_allclose(result_zero, 1.0, rtol=1e-10)
