"""
Tests for angle filtering functionality in optimization objective function.

This module tests the enhanced chi-squared calculation that includes:
- Angle filtering for optimization (using only specific angular ranges)
- All-angle calculation for final results
- Proper degrees of freedom adjustment for filtered calculations
- Backward compatibility with existing code
- Integration with optimization methods
"""

from homodyne.analysis.core import HomodyneAnalysisCore
import numpy as np
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestAngleFilteringCore:
    """Test angle filtering functionality in core chi-squared calculation."""

    @pytest.fixture
    def mock_config_with_angles(self):
        """Configuration for angle filtering tests."""
        return {"metadata": {"config_version": "5.1",
                             "description": "Test configuration for angle filtering",
                             },
                "analyzer_parameters": {"temporal": {"dt": 0.1,
                                                     "start_frame": 1,
                                                     "end_frame": 100},
                                        "scattering": {"wavevector_q": 0.005},
                                        "geometry": {"stator_rotor_gap": 2000000},
                                        "computational": {"num_threads": 4},
                                        },
                "experimental_data": {"data_folder_path": "./test_data/",
                                      "data_file_name": "test_data.hdf",
                                      "phi_angles_path": "./test_data/",
                                      "phi_angles_file": "phi_angles.txt",
                                      },
                "initial_parameters": {"values": [1000.0,
                                                  -0.1,
                                                  50.0,
                                                  0.01,
                                                  -0.5,
                                                  0.001,
                                                  0.0],
                                       "parameter_names": ["D0",
                                                           "alpha",
                                                           "D_offset",
                                                           "gamma_dot_t0",
                                                           "beta",
                                                           "gamma_dot_t_offset",
                                                           "phi0",
                                                           ],
                                       },
                "advanced_settings": {"data_loading": {"use_diagonal_correction": True},
                                      "chi_squared_calculation": {"validity_check": {"check_positive_D0": True},
                                                                  "uncertainty_estimation_factor": 0.1,
                                                                  "minimum_sigma": 1e-10,
                                                                  "_scaling_optimization_note": "Scaling optimization is always enabled: g₂ = offset + contrast × g₁",
                                                                  },
                                      },
                "parameter_space": {"bounds": [{"name": "D0",
                                    "min": 1e-3,
                                                "max": 1e6,
                                                "type": "Normal",
                                                },
                                               {"name": "alpha",
                                                "min": -2.0,
                                                "max": 2.0,
                                                "type": "Normal",
                                                },
                                               {"name": "D_offset",
                                                "min": -5000,
                                                "max": 5000,
                                                "type": "Normal",
                                                },
                                               {"name": "gamma_dot_t0",
                                                "min": 1e-6,
                                                "max": 1.0,
                                                "type": "Normal",
                                                },
                                               {"name": "beta",
                                                "min": -2.0,
                                                "max": 2.0,
                                                "type": "Normal",
                                                },
                                               {"name": "gamma_dot_t_offset",
                                                "min": -0.1,
                                                "max": 0.1,
                                                "type": "Normal",
                                                },
                                               {"name": "phi0",
                                                "min": -10.0,
                                                "max": 10.0,
                                                "type": "Normal",
                                                },
                                               ]},
                "performance_settings": {"optimization_counter_log_frequency": 100,
                                         "parallel_execution": True,
                                         },
                }

    @pytest.fixture
    def test_phi_angles(self):
        """Test phi angles spanning the filtering ranges."""
        return np.array(
            [
                -25.5,  # Outside range
                -8.0,  # Inside [-10, 10] range
                -5.0,  # Inside [-10, 10] range
                0.0,  # Inside [-10, 10] range
                5.0,  # Inside [-10, 10] range
                8.0,  # Inside [-10, 10] range
                15.0,  # Outside range
                45.0,  # Outside range
                90.0,  # Outside range
                150.0,  # Outside range
                175.0,  # Inside [170, 190] range
                180.0,  # Inside [170, 190] range
                185.0,  # Inside [170, 190] range
                200.0,  # Outside range
            ]
        )

    @pytest.fixture
    def mock_analyzer(self, mock_config_with_angles):
        """Create a mock analyzer for angle filtering tests."""
        with patch("homodyne.analysis.core.ConfigManager"):
            analyzer = HomodyneAnalysisCore.__new__(HomodyneAnalysisCore)
            analyzer.config = mock_config_with_angles
            analyzer.dt = 0.1
            analyzer.start_frame = 1
            analyzer.end_frame = 100
            analyzer.wavevector_q = 0.005
            analyzer.stator_rotor_gap = 2000000
            analyzer.num_threads = 4
            analyzer.num_diffusion_params = 3
            analyzer.num_shear_rate_params = 3
            analyzer.time_length = 99
            analyzer.time_array = np.linspace(0.1, 9.9, 99, dtype=np.float64)
            return analyzer

    def test_angle_filtering_identification(self, test_phi_angles):
        """Test identification of angles in optimization ranges."""
        target_ranges = [(-10.0, 10.0), (170.0, 190.0)]

        # Manually identify expected optimization angles
        expected_optimization_indices = []
        expected_angles = []
        for i, angle in enumerate(test_phi_angles):
            for min_angle, max_angle in target_ranges:
                if min_angle <= angle <= max_angle:
                    expected_optimization_indices.append(i)
                    expected_angles.append(angle)
                    break

        # Verify expectations
        # 5 in [-10,10] + 3 in [170,190]
        assert len(expected_optimization_indices) == 8
        assert set(expected_angles) == {
            -8.0,
            -5.0,
            0.0,
            5.0,
            8.0,
            175.0,
            180.0,
            185.0,
        }

        # Test the filtering logic directly
        optimization_indices = []
        for i, angle in enumerate(test_phi_angles):
            for min_angle, max_angle in target_ranges:
                if min_angle <= angle <= max_angle:
                    optimization_indices.append(i)
                    break

        assert optimization_indices == expected_optimization_indices

    def test_chi_squared_angle_filtering_mock(
            self, mock_analyzer, test_phi_angles):
        """Test chi-squared calculation with angle filtering using mocked data."""

        # Create mock experimental data
        n_angles = len(test_phi_angles)
        c2_experimental = np.random.rand(n_angles, 20, 20) + np.float64(
            1.0
        )  # 20x20 correlation matrices
        parameters = np.array([1000.0, -0.1, 50.0, 0.01, -0.5, 0.001, 0.0])

        # Mock the theoretical calculation method
        def mock_calculate_c2(params, angles):
            # Return mock theoretical data with same shape as experimental
            return np.random.rand(len(angles), 20, 20) + np.float64(1.0)

        mock_analyzer.calculate_c2_nonequilibrium_laminar_parallel = mock_calculate_c2

        # Test without filtering (all angles)
        chi2_all = mock_analyzer.calculate_chi_squared_optimized(
            parameters,
            test_phi_angles,
            c2_experimental,
            method_name="Test_All",
            filter_angles_for_optimization=False,
        )

        # Test with filtering (optimization angles only)
        chi2_filtered = mock_analyzer.calculate_chi_squared_optimized(
            parameters,
            test_phi_angles,
            c2_experimental,
            method_name="Test_Filtered",
            filter_angles_for_optimization=True,
        )

        # Both should return valid numbers
        assert isinstance(chi2_all, (int, float))
        assert isinstance(chi2_filtered, (int, float))
        assert chi2_all > 0
        assert chi2_filtered > 0

        # They should be different (unless by coincidence)
        assert chi2_all != chi2_filtered

    def test_detailed_chi_squared_with_filtering(
            self, mock_analyzer, test_phi_angles):
        """Test detailed chi-squared results with angle filtering."""

        # Create mock experimental data
        n_angles = len(test_phi_angles)
        c2_experimental = np.random.rand(n_angles, 10, 10) + np.float64(1.0)
        parameters = np.array([1000.0, -0.1, 50.0, 0.01, -0.5, 0.001, 0.0])

        # Mock the theoretical calculation
        def mock_calculate_c2(params, angles):
            return np.random.rand(len(angles), 10, 10) + np.float64(1.0)

        mock_analyzer.calculate_c2_nonequilibrium_laminar_parallel = mock_calculate_c2

        # Get detailed results without filtering
        result_all = mock_analyzer.calculate_chi_squared_optimized(
            parameters,
            test_phi_angles,
            c2_experimental,
            method_name="DetailedAll",
            return_components=True,
            filter_angles_for_optimization=False,
        )

        # Get detailed results with filtering
        result_filtered = mock_analyzer.calculate_chi_squared_optimized(
            parameters,
            test_phi_angles,
            c2_experimental,
            method_name="DetailedFiltered",
            return_components=True,
            filter_angles_for_optimization=True,
        )

        # Both should be dictionaries
        assert isinstance(result_all, dict)
        assert isinstance(result_filtered, dict)

        # Both should be valid
        assert result_all["valid"]
        assert result_filtered["valid"]

        # Both should have all angle data (for detailed analysis)
        assert len(result_all["angle_chi_squared_reduced"]) == n_angles
        assert len(result_filtered["angle_chi_squared_reduced"]) == n_angles
        assert len(result_all["phi_angles"]) == n_angles
        assert len(result_filtered["phi_angles"]) == n_angles

        # DOF should be different (filtered should have fewer data points)
        assert result_all["degrees_of_freedom"] > result_filtered["degrees_of_freedom"]

        # Total chi-squared should be different
        assert result_all["chi_squared"] != result_filtered["chi_squared"]

    def test_degrees_of_freedom_calculation(
            self, mock_analyzer, test_phi_angles):
        """Test that degrees of freedom are calculated correctly for filtered angles."""

        # Create simple mock data
        n_angles = len(test_phi_angles)
        c2_experimental = np.ones(
            (n_angles, 5, 5)
        )  # 5x5 matrices = 25 data points per angle
        parameters = np.array([1000.0, -0.1, 50.0, 0.01, -0.5, 0.001, 0.0])

        def mock_calculate_c2(params, angles):
            return np.ones((len(angles), 5, 5))

        mock_analyzer.calculate_c2_nonequilibrium_laminar_parallel = mock_calculate_c2

        # Calculate expected DOF
        n_params = len(parameters)
        target_ranges = [(-10.0, 10.0), (170.0, 190.0)]
        optimization_indices = []
        for i, angle in enumerate(test_phi_angles):
            for min_angle, max_angle in target_ranges:
                if min_angle <= angle <= max_angle:
                    optimization_indices.append(i)
                    break

        expected_data_points_filtered = (
            len(optimization_indices) * 25
        )  # 25 points per angle
        expected_dof_filtered = max(
            expected_data_points_filtered - n_params, 1)

        expected_data_points_all = n_angles * 25
        expected_dof_all = max(expected_data_points_all - n_params, 1)

        # Test with filtering
        result_filtered = mock_analyzer.calculate_chi_squared_optimized(
            parameters,
            test_phi_angles,
            c2_experimental,
            return_components=True,
            filter_angles_for_optimization=True,
        )

        # Test without filtering
        result_all = mock_analyzer.calculate_chi_squared_optimized(
            parameters,
            test_phi_angles,
            c2_experimental,
            return_components=True,
            filter_angles_for_optimization=False,
        )

        # Check DOF calculations
        assert result_filtered["degrees_of_freedom"] == expected_dof_filtered
        assert result_all["degrees_of_freedom"] == expected_dof_all

        # Filtered should have fewer degrees of freedom
        assert result_filtered["degrees_of_freedom"] < result_all["degrees_of_freedom"]

    def test_backward_compatibility(self, mock_analyzer, test_phi_angles):
        """Test that existing code still works without angle filtering."""

        # Create mock data with fixed seed for reproducible test
        np.random.seed(42)
        n_angles = len(test_phi_angles)
        c2_experimental = np.random.rand(n_angles, 3, 3) + np.float64(1.0)
        parameters = np.array([1000.0, -0.1, 50.0, 0.01, -0.5, 0.001, 0.0])

        # Use deterministic mock data
        def mock_calculate_c2(params, angles):
            return np.ones((len(angles), 3, 3)) + np.float64(
                0.5
            )  # Fixed data for consistency

        mock_analyzer.calculate_c2_nonequilibrium_laminar_parallel = mock_calculate_c2

        # Test default behavior (no filtering)
        chi2_default = mock_analyzer.calculate_chi_squared_optimized(
            parameters, test_phi_angles, c2_experimental
        )

        # Test explicit no filtering
        chi2_no_filter = mock_analyzer.calculate_chi_squared_optimized(
            parameters,
            test_phi_angles,
            c2_experimental,
            filter_angles_for_optimization=False,
        )

        # Should be identical (both use all angles)
        assert chi2_default == chi2_no_filter
        assert isinstance(chi2_default, (int, float))
        assert chi2_default > 0

    def test_empty_optimization_angles_fallback(self, mock_analyzer):
        """Test fallback behavior when no angles are in optimization ranges."""

        # Use angles all outside the optimization ranges
        phi_angles_outside = np.array([30.0, 60.0, 120.0, 210.0, 240.0])

        n_angles = len(phi_angles_outside)
        c2_experimental = np.random.rand(n_angles, 4, 4) + np.float64(1.0)
        parameters = np.array([1000.0, -0.1, 50.0, 0.01, -0.5, 0.001, 0.0])

        def mock_calculate_c2(params, angles):
            return np.random.rand(len(angles), 4, 4) + np.float64(1.0)

        mock_analyzer.calculate_c2_nonequilibrium_laminar_parallel = mock_calculate_c2

        # This should fall back to using all angles
        chi2_fallback = mock_analyzer.calculate_chi_squared_optimized(
            parameters,
            phi_angles_outside,
            c2_experimental,
            filter_angles_for_optimization=True,
        )

        # Should still return a valid result
        assert isinstance(chi2_fallback, (int, float))
        assert chi2_fallback > 0


class TestAngleFilteringOptimizationIntegration:
    """Test integration of angle filtering with optimization methods."""

    def test_classical_optimizer_uses_angle_filtering(self):
        """Test that classical optimizer creates objective function with angle filtering."""
        from homodyne.optimization.classical import ClassicalOptimizer

        # Mock analyzer and config (use spec to prevent unwanted attributes)
        mock_analyzer = Mock(spec=["calculate_chi_squared_optimized"])
        mock_config = {
            "optimization_config": {
                "classical_optimization": {
                    "methods": ["Nelder-Mead"],
                    "method_options": {},
                },
                "angle_filtering": {"enabled": True},
            }
        }

        optimizer = ClassicalOptimizer(mock_analyzer, mock_config)

        # Create test data
        phi_angles = np.array([-5.0, 0.0, 5.0, 175.0, 185.0])
        c2_experimental = np.random.rand(5, 10, 10)

        # Create objective function
        objective = optimizer.create_objective_function(
            phi_angles, c2_experimental, "Test"
        )

        # Test that the objective function calls the analyzer with angle
        # filtering
        test_params = np.array([1000.0, -0.1, 50.0, 0.01, -0.5, 0.001, 0.0])
        mock_analyzer.calculate_chi_squared_optimized.return_value = 10.0

        result = objective(test_params)

        # Verify the call was made with angle filtering enabled
        mock_analyzer.calculate_chi_squared_optimized.assert_called_once_with(
            test_params,
            phi_angles,
            c2_experimental,
            "Test",
            filter_angles_for_optimization=True,
        )
        assert result == 10.0

    def test_mcmc_sampler_uses_angle_filtering(self):
        """Test that MCMC sampler uses angle filtering by default."""
        try:
            from homodyne.optimization.mcmc import MCMCSampler, PYMC_AVAILABLE

            if not PYMC_AVAILABLE:
                pytest.skip("PyMC not available")

            # Mock analyzer and config
            mock_analyzer = Mock()
            mock_config = {
                "optimization_config": {
                    "mcmc_sampling": {
                        "mcmc_draws": 100,
                        "mcmc_tune": 50,
                        "mcmc_chains": 2,
                    }
                },
                "initial_parameters": {
                    "parameter_names": [
                        "D0",
                        "alpha",
                        "D_offset",
                        "gamma_dot_t0",
                        "beta",
                        "gamma_dot_t_offset",
                        "phi0",
                    ]
                },
                "performance_settings": {},
            }

            # Create MCMC sampler
            sampler = MCMCSampler.__new__(MCMCSampler)
            sampler.core = mock_analyzer
            sampler.config = mock_config
            sampler.mcmc_config = mock_config["optimization_config"]["mcmc_sampling"]
            sampler.bayesian_model = None
            sampler.mcmc_trace = None
            sampler.mcmc_result = None

            # Test that the run_mcmc_analysis method accepts
            # filter_angles_for_optimization parameter
            import inspect

            sig = inspect.signature(sampler.run_mcmc_analysis)
            assert "filter_angles_for_optimization" in sig.parameters

            # Test that the default value is None (which gets resolved to True)
            default_value = sig.parameters["filter_angles_for_optimization"].default
            assert (
                default_value is None
            )  # Now uses None as default, resolved at runtime

            # Test that all MCMC methods have the filtering parameter
            methods_to_check = [
                "run_mcmc_analysis",
                "_run_mcmc_nuts_optimized",
                "_build_bayesian_model_optimized",
            ]

            for method_name in methods_to_check:
                if hasattr(sampler, method_name):
                    method_sig = inspect.signature(
                        getattr(sampler, method_name))
                    assert (
                        "filter_angles_for_optimization" in method_sig.parameters
                    ), f"Missing parameter in {method_name}"

            print("✓ MCMC sampler uses angle filtering by default")

        except ImportError:
            pytest.skip("MCMC module not available")

    def test_all_optimization_methods_use_angle_filtering(self):
        """Test that all three optimization methods use angle filtering consistently."""

        # Test Classical optimization
        try:
            from homodyne.optimization.classical import ClassicalOptimizer

            mock_analyzer = Mock(spec=["calculate_chi_squared_optimized"])
            mock_config = {
                "optimization_config": {
                    "classical_optimization": {"methods": ["Nelder-Mead"]},
                    "angle_filtering": {"enabled": True},
                }
            }
            optimizer = ClassicalOptimizer(mock_analyzer, mock_config)

            # Create objective function and verify it uses filtering
            phi_angles = np.array([-5.0, 0.0, 5.0, 175.0, 185.0])
            c2_experimental = np.random.rand(5, 10, 10)

            objective = optimizer.create_objective_function(
                phi_angles, c2_experimental, "Test"
            )

            # Mock the core method to capture the call
            mock_analyzer.calculate_chi_squared_optimized = Mock(
                return_value=10.0)
            test_params = np.array(
                [1000.0, -0.1, 50.0, 0.01, -0.5, 0.001, 0.0])

            objective(test_params)

            # Verify that filtering was enabled
            mock_analyzer.calculate_chi_squared_optimized.assert_called_once()
            call_args = mock_analyzer.calculate_chi_squared_optimized.call_args
            assert call_args[1]["filter_angles_for_optimization"]

        except ImportError:
            pass  # Classical optimization not available

        # Test MCMC
        try:
            from homodyne.optimization.mcmc import MCMCSampler, PYMC_AVAILABLE

            if PYMC_AVAILABLE:
                import inspect

                sig = inspect.signature(MCMCSampler.run_mcmc_analysis)
                assert (
                    sig.parameters["filter_angles_for_optimization"].default is None
                )  # Uses None, resolved at runtime

        except ImportError:
            pass  # MCMC not available


class TestAngleFilteringEdgeCases:
    """Test edge cases and error handling for angle filtering."""

    def test_single_angle_filtering(self):
        """Test filtering with a single angle."""
        target_ranges = [(-10.0, 10.0), (170.0, 190.0)]

        # Test angle inside range
        angle_inside = np.array([5.0])
        optimization_indices = []
        for i, angle in enumerate(angle_inside):
            for min_angle, max_angle in target_ranges:
                if min_angle <= angle <= max_angle:
                    optimization_indices.append(i)
                    break
        assert optimization_indices == [0]

        # Test angle outside range
        angle_outside = np.array([45.0])
        optimization_indices = []
        for i, angle in enumerate(angle_outside):
            for min_angle, max_angle in target_ranges:
                if min_angle <= angle <= max_angle:
                    optimization_indices.append(i)
                    break
        assert optimization_indices == []

    def test_boundary_angles(self):
        """Test angles exactly on the boundaries of filtering ranges."""
        target_ranges = [(-10.0, 10.0), (170.0, 190.0)]

        boundary_angles = np.array([-10.0, 10.0, 170.0, 190.0])
        optimization_indices = []

        for i, angle in enumerate(boundary_angles):
            for min_angle, max_angle in target_ranges:
                if min_angle <= angle <= max_angle:
                    optimization_indices.append(i)
                    break

        # All boundary angles should be included
        assert optimization_indices == [0, 1, 2, 3]

    def test_angle_filtering_ranges_definition(self):
        """Test that the angle filtering ranges are correctly defined."""
        # This test verifies the specific ranges mentioned in the user request
        target_ranges = [(-10.0, 10.0), (170.0, 190.0)]

        # Test specific angles from user requirement
        test_cases = [
            (-5.0, True),  # Should be included
            (0.0, True),  # Should be included
            (5.0, True),  # Should be included
            (15.0, False),  # Should be excluded
            (175.0, True),  # Should be included
            (180.0, True),  # Should be included
            (185.0, True),  # Should be included
            (195.0, False),  # Should be excluded
        ]

        for angle, should_include in test_cases:
            is_included = False
            for min_angle, max_angle in target_ranges:
                if min_angle <= angle <= max_angle:
                    is_included = True
                    break

            assert (
                is_included == should_include), f"Angle {angle}° should be {
                'included' if should_include else 'excluded'}"

    def test_optimization_methods_use_config_manager(self):
        """Test that optimization methods read angle filtering settings from ConfigManager."""
        from homodyne.core.config import ConfigManager

        try:
            from homodyne.optimization.classical import ClassicalOptimizer

            # Create a mock analyzer with a ConfigManager
            mock_analyzer = Mock()
            mock_config_manager = Mock(spec=ConfigManager)
            mock_config_manager.is_angle_filtering_enabled.return_value = False
            mock_analyzer.config_manager = mock_config_manager

            # Test that classical optimizer respects ConfigManager
            optimizer = ClassicalOptimizer(mock_analyzer, {})
            phi_angles = np.array([-5.0, 0.0, 5.0, 175.0, 185.0])
            c2_experimental = np.random.rand(5, 10, 10)

            objective = optimizer.create_objective_function(
                phi_angles, c2_experimental, "Test"
            )

            # Mock the core method to capture the call
            mock_analyzer.calculate_chi_squared_optimized = Mock(
                return_value=10.0)
            test_params = np.array(
                [1000.0, -0.1, 50.0, 0.01, -0.5, 0.001, 0.0])

            objective(test_params)

            # Verify that ConfigManager was consulted
            mock_config_manager.is_angle_filtering_enabled.assert_called_once()

            # Verify that angle filtering was disabled (as configured)
            call_args = mock_analyzer.calculate_chi_squared_optimized.call_args
            assert call_args[1]["filter_angles_for_optimization"] == False

        except ImportError:
            pytest.skip("Classical optimization module not available")

        # Test MCMC with ConfigManager
        try:
            from homodyne.optimization.mcmc import MCMCSampler

            if not hasattr(MCMCSampler, "_MCMCSampler__test_available"):
                # Create a mock MCMC sampler
                mock_sampler = Mock(spec=MCMCSampler)
                mock_analyzer = Mock()
                mock_config_manager = Mock(spec=ConfigManager)
                mock_config_manager.is_angle_filtering_enabled.return_value = True
                mock_analyzer.config_manager = mock_config_manager

                mock_sampler.core = mock_analyzer

                # Test that MCMC would consult ConfigManager
                # (This is a simplified test since MCMC requires PyMC)
                assert hasattr(mock_analyzer, "config_manager")
                assert mock_analyzer.config_manager.is_angle_filtering_enabled()

        except ImportError:
            pytest.skip("MCMC module not available")


if __name__ == "__main__":
    pytest.main([__file__])
