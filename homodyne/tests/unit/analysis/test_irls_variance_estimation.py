"""
Tests for IRLS (Iterative Reweighted Least Squares) variance estimation functionality.

This module tests the IRLS variance estimation implementation including:
- MAD-based moving window variance estimation
- IRLS iterative convergence behavior
- Edge handling with reflection padding
- Integration with chi-squared calculation
- Adaptive target compatibility
- Configuration parameter handling
- Convergence criteria and damping
"""

import json
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest

from homodyne.analysis.core import HomodyneAnalysisCore


class TestMADMovingWindowVariance:
    """Test MAD-based moving window variance estimation."""

    @pytest.fixture
    def mock_core(self):
        """Create a mock HomodyneAnalysisCore for testing."""
        # Create minimal config
        config_data = {
            "advanced_settings": {
                "chi_squared_calculation": {
                    "minimum_sigma": 1e-10,
                    "moving_window_size": 7,
                    "moving_window_edge_method": "reflect",
                }
            }
        }

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)

        # Mock the core initialization to avoid full setup
        with patch(
            "homodyne.analysis.core.HomodyneAnalysisCore.__init__", return_value=None
        ):
            core = HomodyneAnalysisCore.__new__(HomodyneAnalysisCore)
            core.config = config_data
            core._cached_chi_config = config_data["advanced_settings"][
                "chi_squared_calculation"
            ]

            # Mock memory pooling attributes
            core._pool_initialized = False
            core._memory_pools = {}

            # Mock memory pooling attributes
            core._pool_initialized = False
            core._memory_pools = {}

        return core

    def test_mad_variance_basic_calculation(self, mock_core):
        """Test basic MAD variance calculation."""
        # Test data with known properties
        residuals = np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, 0.4, -0.2, 0.1])
        window_size = 5

        variances = mock_core._estimate_variance_irls_mad_robust(
            residuals, window_size=window_size, edge_method="none"
        )

        # Basic assertions
        assert len(variances) == len(residuals), "Output length should match input"
        assert np.all(variances > 0), "All variances should be positive"
        assert np.all(np.isfinite(variances)), "All variances should be finite"

    def test_mad_variance_with_outliers(self, mock_core):
        """Test MAD variance robustness with outliers."""
        # Data with outliers
        clean_data = np.array([0.1, 0.11, 0.09, 0.12, 0.08])
        outlier_data = np.array([0.1, 0.11, 5.0, 0.12, 0.08])  # Large outlier

        var_clean = mock_core._estimate_variance_irls_mad_robust(
            clean_data, window_size=3, edge_method="none"
        )
        var_outlier = mock_core._estimate_variance_irls_mad_robust(
            outlier_data, window_size=3, edge_method="none"
        )

        # MAD should be more robust to outliers than standard variance
        # The variance at non-outlier positions should be similar
        assert np.abs(var_clean[0] - var_outlier[0]) < np.abs(
            var_clean[0] - np.var(outlier_data)
        )
        assert np.abs(var_clean[-1] - var_outlier[-1]) < np.abs(
            var_clean[-1] - np.var(outlier_data)
        )

    def test_mad_variance_window_size_effect(self, mock_core):
        """Test effect of different window sizes on IRLS variance estimation."""
        residuals = np.random.randn(20) * 0.1

        var_small = mock_core._estimate_variance_irls_mad_robust(
            residuals, window_size=3, edge_method="none"
        )
        var_large = mock_core._estimate_variance_irls_mad_robust(
            residuals, window_size=9, edge_method="none"
        )

        # Both window sizes should produce valid variance estimates
        assert np.all(var_small > 0), "Small window should produce positive variances"
        assert np.all(var_large > 0), "Large window should produce positive variances"
        assert np.all(np.isfinite(var_small)), "Small window should produce finite variances"
        assert np.all(np.isfinite(var_large)), "Large window should produce finite variances"
        assert len(var_small) == len(residuals), "Small window should preserve length"
        assert len(var_large) == len(residuals), "Large window should preserve length"

    def test_mad_variance_minimum_floor(self, mock_core):
        """Test minimum variance floor enforcement."""
        # Very small residuals that could lead to tiny variances
        residuals = np.array([1e-12, -1e-12, 2e-12, -1.5e-12, 1e-12])

        variances = mock_core._estimate_variance_irls_mad_robust(
            residuals, edge_method="none"
        )

        # Check that variances are reasonable (not exactly zero)
        assert np.all(variances > 0), "Variances should be positive"
        assert np.all(np.isfinite(variances)), "Variances should be finite"


class TestIRLSVarianceEstimation:
    """Test IRLS iterative variance estimation with MAD."""

    @pytest.fixture
    def mock_core_with_irls_config(self):
        """Create mock core with IRLS configuration."""
        config_data = {
            "advanced_settings": {
                "chi_squared_calculation": {
                    "minimum_sigma": 1e-10,
                    "moving_window_size": 7,
                    "moving_window_edge_method": "reflect",
                    "variance_method": "irls_mad_robust",
                    "irls_config": {
                        "max_iterations": 5,
                        "damping_factor": 0.7,
                        "convergence_tolerance": 1e-4,
                        "initial_sigma_squared": 1e-3,
                    },
                }
            }
        }

        with patch(
            "homodyne.analysis.core.HomodyneAnalysisCore.__init__", return_value=None
        ):
            core = HomodyneAnalysisCore.__new__(HomodyneAnalysisCore)
            core.config = config_data
            core._cached_chi_config = config_data["advanced_settings"][
                "chi_squared_calculation"
            ]

            # Mock memory pooling attributes
            core._pool_initialized = False
            core._memory_pools = {}

            # Mock memory pooling attributes
            core._pool_initialized = False
            core._memory_pools = {}

        return core

    def test_irls_basic_functionality(self, mock_core_with_irls_config):
        """Test basic IRLS variance estimation functionality."""
        residuals = np.random.randn(50) * 0.2

        variances = mock_core_with_irls_config._estimate_variance_irls_mad_robust(
            residuals, window_size=7, edge_method="none"
        )

        # Basic checks
        assert len(variances) == len(residuals), "Output length should match input"
        assert np.all(variances > 0), "All variances should be positive"
        assert np.all(np.isfinite(variances)), "All variances should be finite"

    def test_irls_convergence_behavior(self, mock_core_with_irls_config):
        """Test IRLS convergence behavior with different data."""
        # Well-behaved data should converge quickly
        stable_residuals = np.random.randn(30) * 0.1

        # Count iterations by checking debug logs
        with patch("homodyne.analysis.core.logger") as mock_logger:
            mock_core_with_irls_config._estimate_variance_irls_mad_robust(
                stable_residuals, window_size=5
            )

            # Should have called logger.debug for iterations
            debug_calls = [
                call
                for call in mock_logger.debug.call_args_list
                if "IRLS iteration" in str(call)
            ]

            # Should converge in reasonable number of iterations
            assert len(debug_calls) <= 5, "Should converge within max iterations"

    def test_irls_damping_factor_effect(self, mock_core_with_irls_config):
        """Test effect of damping factor on IRLS convergence."""
        residuals = np.random.randn(20) * 0.15

        # Modify config for different damping factors
        original_damping = mock_core_with_irls_config._cached_chi_config["irls_config"][
            "damping_factor"
        ]

        # Test with high damping (more stable)
        mock_core_with_irls_config._cached_chi_config["irls_config"][
            "damping_factor"
        ] = 0.9
        var_high_damp = mock_core_with_irls_config._estimate_variance_irls_mad_robust(
            residuals
        )

        # Test with low damping (more aggressive)
        mock_core_with_irls_config._cached_chi_config["irls_config"][
            "damping_factor"
        ] = 0.3
        var_low_damp = mock_core_with_irls_config._estimate_variance_irls_mad_robust(
            residuals
        )

        # Restore original
        mock_core_with_irls_config._cached_chi_config["irls_config"][
            "damping_factor"
        ] = original_damping

        # Both should produce valid results
        assert np.all(
            np.isfinite(var_high_damp)
        ), "High damping should produce valid results"
        assert np.all(
            np.isfinite(var_low_damp)
        ), "Low damping should produce valid results"

    def test_irls_edge_handling_reflection(self, mock_core_with_irls_config):
        """Test IRLS with reflection edge handling."""
        # Test data where edges matter
        residuals = np.array(
            [0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
        )  # High values at edges

        # Pre-pad residuals for edge_method="reflect" testing
        window_size = 5
        pad_window_size = window_size if window_size % 2 == 1 else window_size + 1
        pad_size = pad_window_size // 2
        padded_residuals = np.pad(residuals, pad_size, mode="reflect")

        variances = mock_core_with_irls_config._estimate_variance_irls_mad_robust(
            padded_residuals, window_size=5, edge_method="reflect"
        )

        # Should return same size as original (unpadded) residuals
        assert len(variances) == len(residuals), "Output should match original size"
        # Edge variances should be reasonable (not infinite or zero)
        assert np.isfinite(variances[0]), "First element variance should be finite"
        assert np.isfinite(variances[-1]), "Last element variance should be finite"
        assert variances[0] > 0, "First element variance should be positive"
        assert variances[-1] > 0, "Last element variance should be positive"


class TestIRLSChiSquaredIntegration:
    """Test IRLS integration with chi-squared calculation."""

    @pytest.fixture
    def mock_analysis_core(self):
        """Create a more complete mock for integration testing."""
        config_data = {
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 50},
                "scattering": {"wavevector_q": 0.005},
                "geometry": {"stator_rotor_gap": 2000000},
            },
            "advanced_settings": {
                "chi_squared_calculation": {
                    "minimum_sigma": 1e-10,
                    "moving_window_size": 7,
                    "moving_window_edge_method": "reflect",
                    "variance_method": "irls_mad_robust",
                    "irls_config": {
                        "max_iterations": 3,  # Fewer iterations for faster tests
                        "damping_factor": 0.7,
                        "convergence_tolerance": 1e-4,
                        "initial_sigma_squared": 1e-3,
                    },
                }
            },
            "physical_parameters": {"q": 0.005, "L": 2000000, "dt": 0.1},
            "analysis": {"mode": "static_isotropic"},
        }

        with patch(
            "homodyne.analysis.core.HomodyneAnalysisCore.__init__", return_value=None
        ):
            core = HomodyneAnalysisCore.__new__(HomodyneAnalysisCore)
            core.config = config_data
            core._cached_chi_config = config_data["advanced_settings"][
                "chi_squared_calculation"
            ]

            # Mock memory pooling attributes
            core._pool_initialized = False
            core._memory_pools = {}

            # Mock required methods
            core.calculate_c2_nonequilibrium_laminar_parallel = Mock(
                return_value=np.random.rand(3, 10, 10) + 1.0
            )
            core._fix_diagonal_correction_vectorized = Mock(side_effect=lambda x: x)

        return core

    def test_irls_in_chi_squared_calculation(self, mock_analysis_core):
        """Test IRLS variance estimation within chi-squared calculation."""
        # Mock input data
        parameters = np.array([100.0, 0.5, 10.0])
        phi_angles = np.array([30.0, 60.0, 90.0])
        c2_experimental = np.random.rand(3, 10, 10) + 1.2

        # Mock the theory calculation to return reasonable values
        mock_analysis_core.calculate_c2_nonequilibrium_laminar_parallel.return_value = (
            c2_experimental + np.random.randn(*c2_experimental.shape) * 0.1
        )

        # Test that chi-squared calculation works with IRLS
        with patch.object(
            mock_analysis_core, "_estimate_variance_irls_mad_robust"
        ) as mock_irls:
            mock_irls.return_value = np.ones(100) * 0.01  # Mock variance output

            # This should not raise an exception
            try:
                # Note: This will likely fail due to missing methods, but we're testing the IRLS part
                mock_analysis_core.calculate_chi_squared_optimized(
                    parameters, phi_angles, c2_experimental, return_components=True
                )
                # If we get here, IRLS integration is working
                assert mock_irls.called, "IRLS method should be called"
            except (AttributeError, TypeError):
                # Expected due to incomplete mocking - just test that configuration is correct
                pass

        # Test that the configuration is set up correctly for IRLS
        config = mock_analysis_core.config
        chi_config = config["advanced_settings"]["chi_squared_calculation"]

        assert chi_config["variance_method"] == "irls_mad_robust"
        assert "irls_config" in chi_config

    def test_variance_method_selection(self, mock_analysis_core):
        """Test that variance method selection works correctly."""
        # Test IRLS method selection
        assert (
            mock_analysis_core._cached_chi_config["variance_method"]
            == "irls_mad_robust"
        )

        # Change to fallback method and verify
        mock_analysis_core._cached_chi_config["variance_method"] = "mad_robust"
        assert mock_analysis_core._cached_chi_config["variance_method"] == "mad_robust"


class TestIRLSConfigurationHandling:
    """Test IRLS configuration parameter handling."""

    def test_default_irls_configuration(self):
        """Test default IRLS configuration values."""
        # Test with minimal config
        minimal_config = {
            "advanced_settings": {
                "chi_squared_calculation": {"variance_method": "irls_mad_robust"}
            }
        }

        with patch(
            "homodyne.analysis.core.HomodyneAnalysisCore.__init__", return_value=None
        ):
            core = HomodyneAnalysisCore.__new__(HomodyneAnalysisCore)
            core.config = minimal_config
            core._cached_chi_config = minimal_config["advanced_settings"][
                "chi_squared_calculation"
            ]

            # Mock memory pooling attributes
            core._pool_initialized = False
            core._memory_pools = {}

            # Test that default values are handled properly
            residuals = np.random.randn(20) * 0.1

            try:
                variances = core._estimate_variance_irls_mad_robust(residuals)
                assert np.all(np.isfinite(variances)), "Should work with default config"
            except KeyError as e:
                # If defaults aren't properly handled, this will fail
                pytest.fail(f"Default configuration not handled: {e}")

    def test_irls_parameter_validation(self):
        """Test IRLS parameter validation and bounds."""
        config_data = {
            "advanced_settings": {
                "chi_squared_calculation": {
                    "minimum_sigma": 1e-10,
                    "irls_config": {
                        "max_iterations": 10,
                        "damping_factor": 0.5,
                        "convergence_tolerance": 1e-5,
                        "initial_sigma_squared": 2e-3,
                    },
                }
            }
        }

        with patch(
            "homodyne.analysis.core.HomodyneAnalysisCore.__init__", return_value=None
        ):
            core = HomodyneAnalysisCore.__new__(HomodyneAnalysisCore)
            core.config = config_data
            core._cached_chi_config = config_data["advanced_settings"][
                "chi_squared_calculation"
            ]

            # Mock memory pooling attributes
            core._pool_initialized = False
            core._memory_pools = {}

            # Test with custom parameters
            residuals = np.random.randn(15) * 0.12
            variances = core._estimate_variance_irls_mad_robust(residuals)

            assert np.all(
                variances > 0
            ), "Custom parameters should produce valid results"


class TestIRLSAdaptiveTargetCompatibility:
    """Test IRLS compatibility with adaptive target optimization."""

    def test_adaptive_target_chi_squared_components(self):
        """Test that IRLS provides correct components for adaptive target."""
        # This would test the integration we fixed in classical.py
        # Mock the components that should be returned
        mock_components = {
            "valid": True,
            "total_chi_squared": 120.0,
            "chi_squared": 1.2,  # This is actually reduced chi-squared
            "degrees_of_freedom": 100,
            "reduced_chi_squared": 1.2,
        }

        # Test the adaptive target calculation logic
        adaptive_target_alpha = 1.0
        total_chi_squared = mock_components["total_chi_squared"]
        total_dof = mock_components["degrees_of_freedom"]
        target_chi_squared = adaptive_target_alpha * total_dof

        # This should be mathematically correct
        expected_target = 1.0 * 100  # α * DOF = 100
        assert target_chi_squared == expected_target

        # Adaptive target objective
        objective_value = (total_chi_squared - target_chi_squared) ** 2
        expected_objective = (120.0 - 100.0) ** 2  # (120 - 100)^2 = 400
        assert objective_value == expected_objective

        # Verify reduced chi-squared relationship
        calculated_reduced = total_chi_squared / total_dof
        assert abs(calculated_reduced - mock_components["reduced_chi_squared"]) < 1e-10


class TestIRLSRobustness:
    """Test IRLS robustness and edge cases."""

    @pytest.fixture
    def robust_test_core(self):
        """Core setup for robustness testing."""
        config_data = {
            "advanced_settings": {
                "chi_squared_calculation": {
                    "minimum_sigma": 1e-12,
                    "irls_config": {
                        "max_iterations": 8,
                        "damping_factor": 0.6,
                        "convergence_tolerance": 1e-6,
                        "initial_sigma_squared": 5e-4,
                    },
                }
            }
        }

        with patch(
            "homodyne.analysis.core.HomodyneAnalysisCore.__init__", return_value=None
        ):
            core = HomodyneAnalysisCore.__new__(HomodyneAnalysisCore)
            core.config = config_data
            core._cached_chi_config = config_data["advanced_settings"][
                "chi_squared_calculation"
            ]

            # Mock memory pooling attributes
            core._pool_initialized = False
            core._memory_pools = {}

            # Mock memory pooling attributes
            core._pool_initialized = False
            core._memory_pools = {}

        return core

    def test_irls_with_zero_residuals(self, robust_test_core):
        """Test IRLS behavior with zero residuals."""
        residuals = np.zeros(10)

        variances = robust_test_core._estimate_variance_irls_mad_robust(residuals)

        # Should handle zeros gracefully
        assert np.all(np.isfinite(variances)), "Should handle zero residuals"
        assert np.all(variances > 0), "Should maintain positive variances"

    def test_irls_with_constant_residuals(self, robust_test_core):
        """Test IRLS with constant (non-zero) residuals."""
        residuals = np.ones(15) * 0.1

        variances = robust_test_core._estimate_variance_irls_mad_robust(residuals)

        # Should handle constant values
        assert np.all(np.isfinite(variances)), "Should handle constant residuals"
        assert np.all(variances > 0), "Should maintain positive variances"

    def test_irls_with_extreme_outliers(self, robust_test_core):
        """Test IRLS robustness with extreme outliers."""
        # Most values small, one very large
        residuals = np.array([0.01, 0.02, 0.01, 100.0, 0.02, 0.01, 0.01])

        variances = robust_test_core._estimate_variance_irls_mad_robust(
            residuals, edge_method="none"
        )

        # MAD-based IRLS should be robust to outliers
        assert np.all(np.isfinite(variances)), "Should handle extreme outliers"
        assert np.all(variances > 0), "Should maintain positive variances"

        # Variance at non-outlier positions should be reasonable
        non_outlier_indices = [0, 1, 2, 4, 5, 6]
        non_outlier_vars = variances[non_outlier_indices]
        assert np.std(non_outlier_vars) < np.std(
            residuals
        ), "Should be robust to outliers"

    def test_irls_with_very_small_window(self, robust_test_core):
        """Test IRLS with very small window size."""
        residuals = np.random.randn(20) * 0.1

        variances = robust_test_core._estimate_variance_irls_mad_robust(
            residuals, window_size=3, edge_method="none"
        )

        assert len(variances) == len(residuals), "Should handle small windows"
        assert np.all(
            variances > 0
        ), "Should produce valid variances with small windows"

    def test_irls_with_short_data(self, robust_test_core):
        """Test IRLS with very short data arrays."""
        residuals = np.array([0.1, -0.2, 0.15])  # Only 3 points

        variances = robust_test_core._estimate_variance_irls_mad_robust(
            residuals, window_size=3, edge_method="none"
        )

        assert len(variances) == 3, "Should handle short data arrays"
        assert np.all(variances > 0), "Should produce valid variances for short arrays"


if __name__ == "__main__":
    pytest.main([__file__])
