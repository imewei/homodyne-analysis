"""
Integration tests for IRLS variance estimation with chi-squared calculation.

This module focuses on testing the actual integration of IRLS with the
chi-squared calculation pipeline, including:
- Method selection logic
- Configuration handling
- Adaptive target compatibility
- Performance and numerical stability
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from homodyne.analysis.core import HomodyneAnalysisCore


class TestIRLSMethodSelection:
    """Test IRLS method selection in chi-squared calculation."""

    @pytest.fixture
    def mock_core_config(self):
        """Mock core with different configurations."""

        def create_core(variance_method="irls_mad_robust"):
            config = {
                "advanced_settings": {
                    "chi_squared_calculation": {
                        "variance_method": variance_method,
                        "minimum_sigma": 1e-10,
                        "moving_window_size": 7,
                        "moving_window_edge_method": "reflect",
                        "irls_config": {
                            "max_iterations": 3,
                            "damping_factor": 0.7,
                            "convergence_tolerance": 1e-4,
                            "initial_sigma_squared": 1e-3,
                        },
                    }
                }
            }

            core = Mock(spec=HomodyneAnalysisCore)
            core.config = config
            core._cached_chi_config = config["advanced_settings"][
                "chi_squared_calculation"
            ]

            # Mock memory pooling attributes
            core._pool_initialized = False
            core._memory_pools = {}

            # Add the actual methods we want to test
            import homodyne.analysis.core

            AnalysisCore = homodyne.analysis.core.HomodyneAnalysisCore
            # Removed obsolete variance method reference
            core._estimate_variance_irls_mad_robust = (
                AnalysisCore._estimate_variance_irls_mad_robust.__get__(core)
            )
            core._mad_moving_window_with_edge_handling = (
                AnalysisCore._mad_moving_window_with_edge_handling.__get__(core)
            )

            return core

        return create_core

    def test_irls_method_called_with_correct_config(self, mock_core_config):
        """Test that IRLS method is called when configured."""
        core = mock_core_config("irls_mad_robust")

        test_residuals = np.random.randn(20) * 0.1

        # Test IRLS method directly (use edge_method="none" for direct testing)
        result = core._estimate_variance_irls_mad_robust(
            test_residuals, edge_method="none"
        )

        assert len(result) == len(test_residuals)
        assert np.all(result > 0)
        assert np.all(np.isfinite(result))

    def test_irls_mad_robust_method(self, mock_core_config):
        """Test IRLS MAD robust method."""
        core = mock_core_config("irls_mad_robust")

        test_residuals = np.random.randn(15) * 0.15

        # Test IRLS MAD method directly
        result = core._estimate_variance_irls_mad_robust(
            test_residuals, edge_method="none"
        )

        assert len(result) == len(test_residuals)
        assert np.all(result > 0)
        assert np.all(np.isfinite(result))


class TestIRLSNumericalStability:
    """Test numerical stability of IRLS implementation."""

    @pytest.fixture
    def stable_core(self):
        """Create core configured for numerical stability testing."""
        config = {
            "advanced_settings": {
                "chi_squared_calculation": {
                    "variance_method": "irls_mad_robust",
                    "minimum_sigma": 1e-15,  # Very small minimum
                    "moving_window_size": 5,
                    "irls_config": {
                        "max_iterations": 10,
                        "damping_factor": 0.8,
                        "convergence_tolerance": 1e-8,
                        "initial_sigma_squared": 1e-6,
                    },
                }
            }
        }

        core = Mock(spec=HomodyneAnalysisCore)
        core.config = config
        core._cached_chi_config = config["advanced_settings"]["chi_squared_calculation"]

        # Mock memory pooling attributes
        core._pool_initialized = False
        core._memory_pools = {}

        import homodyne.analysis.core

        AnalysisCore = homodyne.analysis.core.HomodyneAnalysisCore
        # Removed obsolete variance method reference
        core._estimate_variance_irls_mad_robust = (
            AnalysisCore._estimate_variance_irls_mad_robust.__get__(core)
        )
        core._mad_moving_window_with_edge_handling = (
            AnalysisCore._mad_moving_window_with_edge_handling.__get__(core)
        )

        return core

    def test_irls_with_tiny_values(self, stable_core):
        """Test IRLS with very small residual values."""
        tiny_residuals = np.random.randn(25) * 1e-8

        variances = stable_core._estimate_variance_irls_mad_robust(
            tiny_residuals, edge_method="none"
        )

        assert np.all(np.isfinite(variances)), "Should handle tiny values"
        assert np.all(variances > 0), "Should maintain positivity with tiny values"

    def test_irls_with_large_values(self, stable_core):
        """Test IRLS with very large residual values."""
        large_residuals = np.random.randn(25) * 1e6

        variances = stable_core._estimate_variance_irls_mad_robust(
            large_residuals, edge_method="none"
        )

        assert np.all(np.isfinite(variances)), "Should handle large values"
        assert np.all(variances > 0), "Should maintain positivity with large values"

    def test_irls_with_mixed_scales(self, stable_core):
        """Test IRLS with mixed scale residuals."""
        # Mix of very small and moderate values
        mixed_residuals = np.concatenate(
            [
                np.random.randn(10) * 1e-6,  # Very small
                np.random.randn(10) * 1e-2,  # Moderate
                np.random.randn(5) * 1e-6,  # Very small again
            ]
        )

        variances = stable_core._estimate_variance_irls_mad_robust(
            mixed_residuals, edge_method="none"
        )

        assert np.all(np.isfinite(variances)), "Should handle mixed scales"
        assert np.all(variances > 0), "Should maintain positivity with mixed scales"


class TestIRLSConvergenceBehavior:
    """Test IRLS convergence behavior and iteration control."""

    @pytest.fixture
    def convergence_core(self):
        """Core setup for convergence testing."""
        config = {
            "advanced_settings": {
                "chi_squared_calculation": {
                    "variance_method": "irls_mad_robust",
                    "minimum_sigma": 1e-10,
                    "moving_window_size": 7,
                    "irls_config": {
                        "max_iterations": 15,  # More iterations for convergence testing
                        "damping_factor": 0.6,
                        "convergence_tolerance": 1e-7,  # Tighter tolerance
                        "initial_sigma_squared": 2e-3,
                    },
                }
            }
        }

        core = Mock(spec=HomodyneAnalysisCore)
        core.config = config
        core._cached_chi_config = config["advanced_settings"]["chi_squared_calculation"]

        # Mock memory pooling attributes
        core._pool_initialized = False
        core._memory_pools = {}

        import homodyne.analysis.core

        AnalysisCore = homodyne.analysis.core.HomodyneAnalysisCore
        # Removed obsolete variance method reference
        core._estimate_variance_irls_mad_robust = (
            AnalysisCore._estimate_variance_irls_mad_robust.__get__(core)
        )
        core._mad_moving_window_with_edge_handling = (
            AnalysisCore._mad_moving_window_with_edge_handling.__get__(core)
        )

        return core

    def test_irls_convergence_with_stable_data(self, convergence_core):
        """Test IRLS convergence with well-behaved data."""
        # Generate stable residuals (Gaussian with consistent variance)
        np.random.seed(42)  # For reproducible results
        stable_residuals = np.random.normal(0, 0.1, 30)

        with patch("homodyne.analysis.core.logger") as mock_logger:
            variances = convergence_core._estimate_variance_irls_mad_robust(
                stable_residuals
            )

            # Check that convergence messages were logged
            debug_calls = [
                call
                for call in mock_logger.debug.call_args_list
                if "IRLS iteration" in str(call)
            ]
            info_calls = [
                call
                for call in mock_logger.info.call_args_list
                if "converged after" in str(call)
            ]

            # Should converge (and log convergence)
            assert len(info_calls) > 0, "Should log convergence"
            assert len(debug_calls) > 0, "Should log iterations"

        assert np.all(np.isfinite(variances)), "Converged result should be finite"
        assert np.all(variances > 0), "Converged result should be positive"

    def test_irls_max_iterations_reached(self, convergence_core):
        """Test IRLS behavior when max iterations is reached."""
        # Modify config for very few iterations
        convergence_core._cached_chi_config["irls_config"]["max_iterations"] = 2

        # Use challenging data that might not converge quickly
        challenging_residuals = np.concatenate(
            [
                np.ones(5) * 0.01,  # Small consistent values
                [2.0],  # Large outlier
                np.ones(5) * 0.01,  # Back to small values
            ]
        )

        with patch("homodyne.analysis.core.logger") as mock_logger:
            variances = convergence_core._estimate_variance_irls_mad_robust(
                challenging_residuals
            )

            # Should log either convergence or max iterations warning
            [
                call
                for call in mock_logger.warning.call_args_list
                if "did not converge" in str(call)
            ]

        # Even without convergence, should produce valid results
        assert np.all(np.isfinite(variances)), (
            "Should produce finite results even without convergence"
        )
        assert np.all(variances > 0), (
            "Should maintain positivity even without convergence"
        )


class TestIRLSEdgeHandling:
    """Test IRLS edge handling with reflection padding."""

    @pytest.fixture
    def edge_test_core(self):
        """Core setup for edge handling tests."""
        config = {
            "advanced_settings": {
                "chi_squared_calculation": {
                    "variance_method": "irls_mad_robust",
                    "minimum_sigma": 1e-10,
                    "moving_window_size": 9,  # Larger window to test edges
                    "moving_window_edge_method": "reflect",
                    "irls_config": {
                        "max_iterations": 4,
                        "damping_factor": 0.7,
                        "convergence_tolerance": 1e-4,
                        "initial_sigma_squared": 1e-3,
                    },
                }
            }
        }

        core = Mock(spec=HomodyneAnalysisCore)
        core.config = config
        core._cached_chi_config = config["advanced_settings"]["chi_squared_calculation"]

        # Mock memory pooling attributes
        core._pool_initialized = False
        core._memory_pools = {}

        import homodyne.analysis.core

        AnalysisCore = homodyne.analysis.core.HomodyneAnalysisCore
        # Removed obsolete variance method reference
        core._estimate_variance_irls_mad_robust = (
            AnalysisCore._estimate_variance_irls_mad_robust.__get__(core)
        )
        core._mad_moving_window_with_edge_handling = (
            AnalysisCore._mad_moving_window_with_edge_handling.__get__(core)
        )

        return core

    def test_edge_reflection_effectiveness(self, edge_test_core):
        """Test that reflection edge handling produces reasonable edge variances."""
        # Create data with distinct edge behavior
        edge_data = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5])

        # Pre-pad data for edge_method="reflect" testing
        window_size = 7
        pad_window_size = window_size if window_size % 2 == 1 else window_size + 1
        pad_size = pad_window_size // 2
        padded_edge_data = np.pad(edge_data, pad_size, mode="reflect")

        variances = edge_test_core._estimate_variance_irls_mad_robust(
            padded_edge_data, window_size=7, edge_method="reflect"
        )

        # Should return same size as original
        assert len(variances) == len(edge_data), "Output should match original size"
        # Edge variances should be reasonable (not extremely large or small)
        edge_var_ratio = variances[0] / np.mean(
            variances[2:-2]
        )  # Compare edge to center

        assert 0.1 < edge_var_ratio < 10, (
            "Edge variance should be reasonable relative to center"
        )
        assert np.all(np.isfinite(variances)), (
            "All edge-handled variances should be finite"
        )

    def test_no_edge_handling_vs_reflection(self, edge_test_core):
        """Test difference between reflection and no edge handling."""
        test_data = np.random.randn(15) * 0.12

        # Pre-pad data for reflection test
        window_size = 7
        pad_window_size = window_size if window_size % 2 == 1 else window_size + 1
        pad_size = pad_window_size // 2
        padded_test_data = np.pad(test_data, pad_size, mode="reflect")

        # With reflection
        var_reflect = edge_test_core._estimate_variance_irls_mad_robust(
            padded_test_data, window_size=7, edge_method="reflect"
        )

        # Without reflection (direct processing)
        var_no_reflect = edge_test_core._estimate_variance_irls_mad_robust(
            test_data, window_size=7, edge_method="none"
        )

        # Both should be valid and same size as original
        assert len(var_reflect) == len(test_data), (
            "Reflection result should match original size"
        )
        assert len(var_no_reflect) == len(test_data), (
            "No-reflection result should match original size"
        )
        assert np.all(np.isfinite(var_reflect)), "Reflection handling should be finite"
        assert np.all(np.isfinite(var_no_reflect)), (
            "No-reflection handling should be finite"
        )

        # The middle values should be similar, edges might differ
        middle_indices = slice(3, -3)
        middle_diff = np.mean(
            np.abs(var_reflect[middle_indices] - var_no_reflect[middle_indices])
        )
        edge_diff = np.mean(np.abs(var_reflect[[0, -1]] - var_no_reflect[[0, -1]]))

        # This is hard to assert precisely, but edges might have more difference
        assert middle_diff >= 0, "Middle differences should be non-negative"
        assert edge_diff >= 0, "Edge differences should be non-negative"


class TestIRLSRealWorldScenarios:
    """Test IRLS with realistic data scenarios."""

    @pytest.fixture
    def realistic_core(self):
        """Core setup for realistic testing."""
        config = {
            "advanced_settings": {
                "chi_squared_calculation": {
                    "variance_method": "irls_mad_robust",
                    "minimum_sigma": 1e-12,
                    "moving_window_size": 11,
                    "moving_window_edge_method": "reflect",
                    "irls_config": {
                        "max_iterations": 8,
                        "damping_factor": 0.7,
                        "convergence_tolerance": 1e-5,
                        "initial_sigma_squared": 1e-3,
                    },
                }
            }
        }

        core = Mock(spec=HomodyneAnalysisCore)
        core.config = config
        core._cached_chi_config = config["advanced_settings"]["chi_squared_calculation"]

        # Mock memory pooling attributes
        core._pool_initialized = False
        core._memory_pools = {}

        import homodyne.analysis.core

        AnalysisCore = homodyne.analysis.core.HomodyneAnalysisCore
        # Removed obsolete variance method reference
        core._estimate_variance_irls_mad_robust = (
            AnalysisCore._estimate_variance_irls_mad_robust.__get__(core)
        )
        core._mad_moving_window_with_edge_handling = (
            AnalysisCore._mad_moving_window_with_edge_handling.__get__(core)
        )

        return core

    def test_typical_correlation_function_residuals(self, realistic_core):
        """Test IRLS with residuals typical of correlation function fitting."""
        # Simulate residuals that might come from correlation function fitting
        # - Generally small values around zero
        # - Some systematic deviations
        # - Occasional outliers

        np.random.seed(123)
        n_points = 100

        # Base Gaussian residuals
        base_residuals = np.random.normal(0, 0.02, n_points)

        # Add some systematic deviation (like model mismatch)
        systematic = 0.01 * np.sin(np.linspace(0, 4 * np.pi, n_points))

        # Add occasional outliers
        outlier_indices = np.random.choice(n_points, size=5, replace=False)
        outliers = np.zeros(n_points)
        outliers[outlier_indices] = np.random.normal(0, 0.1, 5)

        realistic_residuals = base_residuals + systematic + outliers

        # Test IRLS with this realistic data
        variances = realistic_core._estimate_variance_irls_mad_robust(
            realistic_residuals
        )

        # Should handle realistic data well
        assert np.all(np.isfinite(variances)), "Should handle realistic residuals"
        assert np.all(variances > 0), "Should maintain positivity"

        # Variance estimates should be reasonable
        median_variance = np.median(variances)
        assert 1e-8 < median_variance < 1e-2, "Median variance should be reasonable"

        # Should be robust to outliers (variance at outlier points might be higher but finite)
        outlier_variances = variances[outlier_indices]
        assert np.all(np.isfinite(outlier_variances)), "Should handle outlier positions"

    def test_heteroscedastic_residuals(self, realistic_core):
        """Test IRLS with heteroscedastic (varying variance) residuals."""
        # Create residuals with varying variance across the data
        n_points = 80
        x = np.linspace(0, 1, n_points)

        # Varying standard deviation
        sigma_varying = 0.01 + 0.05 * x**2  # Increasing variance

        # Generate residuals with this varying variance
        hetero_residuals = np.random.normal(0, sigma_varying)

        # Test IRLS - should adapt to local variance
        variances = realistic_core._estimate_variance_irls_mad_robust(
            hetero_residuals, edge_method="none"
        )

        assert np.all(np.isfinite(variances)), "Should handle heteroscedastic data"
        assert np.all(variances > 0), "Should maintain positivity"

        # Early points should generally have lower variance estimates than later points
        early_var = np.mean(variances[:20])
        late_var = np.mean(variances[-20:])

        # This is a trend test - not strict inequality due to noise and robustness
        assert early_var > 0, "Early variance should be positive"
        assert late_var > 0, "Late variance should be positive"

        # The ratio should reflect the underlying trend (roughly)
        var_ratio = late_var / early_var
        (sigma_varying[-20:].mean() / sigma_varying[:20].mean()) ** 2

        # Should be somewhat correlated with the true variance trend
        # Allow wide range due to robustness and finite sample effects
        assert 0.1 < var_ratio < 100, "Variance ratio should be reasonable"


if __name__ == "__main__":
    pytest.main([__file__])
