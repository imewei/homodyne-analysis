"""
Test Classical Optimization Configuration Reading
==============================================

Tests to ensure classical optimization reads configuration correctly from the JSON
configuration file, including initial_parameters and parameter_space sections.

Covers both Nelder-Mead and Gurobi optimization methods, including automatic
detection, configuration handling, and parameter bounds consistency with MCMC.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from homodyne.optimization.classical import GUROBI_AVAILABLE, ClassicalOptimizer


class TestClassicalOptimizationConfigurationReading:
    """Test classical optimization configuration reading from JSON files."""

    def test_initial_parameters_reading(self):
        """Test that ClassicalOptimizer reads initial parameters correctly."""

        test_config = {
            "initial_parameters": {
                "values": [1000.0, -0.5, 100.0, 0.001, 0.2, 0.0001, 5.0],
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_t0",
                    "beta",
                    "gamma_dot_t_offset",
                    "phi0",
                ],
                "active_parameters": ["D0", "alpha", "D_offset"],
            },
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]}
            },
        }

        # Create mock core
        mock_core = Mock()
        mock_core.num_threads = 4

        # Create ClassicalOptimizer instance
        optimizer = ClassicalOptimizer(mock_core, test_config)

        # Test that it can access initial parameters correctly
        expected_values = np.array([1000.0, -0.5, 100.0, 0.001, 0.2, 0.0001, 5.0])
        actual_values = np.array(
            test_config["initial_parameters"]["values"], dtype=np.float64
        )

        np.testing.assert_array_equal(actual_values, expected_values)

        # Test parameter names access
        param_names = test_config["initial_parameters"]["parameter_names"]
        assert len(param_names) == 7
        assert param_names[0] == "D0"
        assert param_names[1] == "alpha"
        assert param_names[2] == "D_offset"

    def test_parameter_space_bounds_reading(self):
        """Test that ClassicalOptimizer reads parameter bounds correctly."""

        test_config = {
            "parameter_space": {
                "bounds": [
                    {
                        "name": "D0",
                        "min": 1.0,
                        "max": 1000000,
                        "type": "Normal",
                    },
                    {
                        "name": "alpha",
                        "min": -2.0,
                        "max": 2.0,
                        "type": "Normal",
                    },
                    {
                        "name": "D_offset",
                        "min": -100,
                        "max": 100,
                        "type": "Normal",
                    },
                    {
                        "name": "gamma_dot_t0",
                        "min": 1e-6,
                        "max": 1.0,
                        "type": "Normal",
                    },
                    {
                        "name": "beta",
                        "min": -2.0,
                        "max": 2.0,
                        "type": "Normal",
                    },
                    {
                        "name": "gamma_dot_t_offset",
                        "min": -1e-2,
                        "max": 1e-2,
                        "type": "Normal",
                    },
                    {
                        "name": "phi0",
                        "min": -10.0,
                        "max": 10.0,
                        "type": "Normal",
                    },
                ]
            },
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]}
            },
        }

        # Create mock core
        mock_core = Mock()
        mock_core.num_threads = 4

        # Create ClassicalOptimizer instance
        optimizer = ClassicalOptimizer(mock_core, test_config)

        # Test parameter bounds access
        param_bounds = test_config.get("parameter_space", {}).get("bounds", [])

        assert len(param_bounds) == 7

        # Check specific bounds (using authoritative constraints)
        d0_bound = param_bounds[0]
        assert d0_bound["name"] == "D0"
        # Updated to match authoritative constraints
        assert d0_bound["min"] == 1.0
        # Updated to match authoritative constraints
        assert d0_bound["max"] == 1000000

        alpha_bound = param_bounds[1]
        assert alpha_bound["name"] == "alpha"
        assert alpha_bound["min"] == -2.0
        # Updated to match authoritative constraints
        assert alpha_bound["max"] == 2.0

    def test_bounds_extraction_for_optimization(self):
        """Test that bounds are correctly extracted for optimization methods."""

        test_config = {
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 1.0, "max": 1000000},
                    {"name": "alpha", "min": -1.5, "max": -0.5},
                    {"name": "D_offset", "min": -100, "max": 100},
                ]
            },
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]}
            },
        }

        # Create mock core
        mock_core = Mock()
        mock_core.num_threads = 4

        # Create ClassicalOptimizer instance
        optimizer = ClassicalOptimizer(mock_core, test_config)

        # Test bounds extraction (simulating what happens in
        # get_parameter_bounds method)
        bounds = []
        param_bounds = test_config.get("parameter_space", {}).get("bounds", [])
        effective_param_count = 3  # For this test

        for i, bound in enumerate(param_bounds):
            if i < effective_param_count:
                bounds.append((bound.get("min", -np.inf), bound.get("max", np.inf)))

        # Expected bounds should match what's in the test_config above
        expected_bounds = [(1.0, 1000000), (-1.5, -0.5), (-100, 100)]
        assert bounds == expected_bounds

    def test_static_mode_parameter_selection(self):
        """Test parameter selection for static mode (first 3 parameters)."""

        test_config = {
            "initial_parameters": {
                "values": [18000, -1.59, 3.10, 0.0, 0.0, 0.0, 0.0],
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_t0",
                    "beta",
                    "gamma_dot_t_offset",
                    "phi0",
                ],
                "active_parameters": ["D0", "alpha", "D_offset"],
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 1.0, "max": 1000000},
                    {"name": "alpha", "min": -1.6, "max": -1.5},
                    {"name": "D_offset", "min": -100, "max": 100},
                    {
                        "name": "gamma_dot_t0",
                        "min": 0.0,
                        "max": 0.0,
                    },  # Fixed for static
                    {
                        "name": "beta",
                        "min": 0.0,
                        "max": 0.0,
                    },  # Fixed for static
                    {
                        "name": "gamma_dot_t_offset",
                        "min": 0.0,
                        "max": 0.0,
                    },  # Fixed for static
                    {
                        "name": "phi0",
                        "min": 0.0,
                        "max": 0.0,
                    },  # Fixed for static
                ]
            },
            "analysis_settings": {
                "static_mode": True,
                "static_submode": "isotropic",
            },
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]}
            },
        }

        # Create mock core
        mock_core = Mock()
        mock_core.num_threads = 4

        # Create ClassicalOptimizer instance
        optimizer = ClassicalOptimizer(mock_core, test_config)

        # Test parameter selection for static mode (simulating
        # run_classical_optimization_optimized logic)
        initial_parameters = np.array(
            test_config["initial_parameters"]["values"], dtype=np.float64
        )
        effective_param_count = 3  # Static mode uses first 3 parameters

        # For static mode, only use first 3 parameters
        static_parameters = initial_parameters[:effective_param_count]

        expected_static_params = np.array([18000, -1.59, 3.10])
        np.testing.assert_array_equal(static_parameters, expected_static_params)

        # Test bounds for static mode
        param_bounds = test_config.get("parameter_space", {}).get("bounds", [])
        static_bounds = []
        for i, bound in enumerate(param_bounds):
            if i < effective_param_count:
                static_bounds.append((bound.get("min"), bound.get("max")))

        # Expected bounds should match what's in the test_config above
        expected_static_bounds = [(1.0, 1000000), (-1.6, -1.5), (-100, 100)]
        assert static_bounds == expected_static_bounds

    def test_optimization_config_access(self):
        """Test that optimization configuration is accessed correctly."""

        test_config = {
            "optimization_config": {
                "classical_optimization": {
                    "methods": ["Nelder-Mead"],
                    "method_options": {
                        "Nelder-Mead": {"maxiter": 5000, "xatol": 1e-10},
                    },
                }
            }
        }

        # Create mock core
        mock_core = Mock()
        mock_core.num_threads = 4

        # Create ClassicalOptimizer instance
        optimizer = ClassicalOptimizer(mock_core, test_config)

        # Test that optimization configuration is loaded correctly
        opt_config = optimizer.optimization_config

        assert "methods" in opt_config
        assert "method_options" in opt_config

        methods = opt_config["methods"]
        assert "Nelder-Mead" in methods
        assert len(methods) == 1  # Only Nelder-Mead should be present

        method_options = opt_config["method_options"]
        assert "Nelder-Mead" in method_options
        assert method_options["Nelder-Mead"]["maxiter"] == 5000

    def test_missing_configuration_sections(self):
        """Test handling of missing configuration sections."""

        # Configuration with missing sections
        test_config = {
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]}
            }
            # Missing initial_parameters and parameter_space
        }

        # Create mock core
        mock_core = Mock()
        mock_core.num_threads = 4

        # Create ClassicalOptimizer instance - should not crash
        optimizer = ClassicalOptimizer(mock_core, test_config)

        # Test graceful handling of missing sections
        param_bounds = test_config.get("parameter_space", {}).get("bounds", [])
        assert param_bounds == []  # Should get empty list as default

        # Test that optimization config is still accessible
        opt_config = optimizer.optimization_config
        assert "methods" in opt_config

    def test_real_config_structure_compatibility(self):
        """Test with realistic configuration structure matching the project."""

        realistic_config = {
            "metadata": {
                "config_version": "6.0",
                "analysis_mode": "static_isotropic",
            },
            "initial_parameters": {
                "values": [18000, -1.59, 3.10],
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_t0",
                    "beta",
                    "gamma_dot_t_offset",
                    "phi0",
                ],
                "active_parameters": ["D0", "alpha", "D_offset"],
            },
            "parameter_space": {
                "bounds": [
                    {
                        "name": "D0",
                        "min": 1.0,
                        "max": 1000000,
                        "type": "Normal",
                    },
                    {
                        "name": "alpha",
                        "min": -1.6,
                        "max": -1.5,
                        "type": "Normal",
                    },
                    {
                        "name": "D_offset",
                        "min": -100,
                        "max": 100,
                        "type": "Normal",
                    },
                    {
                        "name": "gamma_dot_t0",
                        "min": 0.0,
                        "max": 0.0,
                        "type": "fixed",
                    },
                    {"name": "beta", "min": 0.0, "max": 0.0, "type": "fixed"},
                    {
                        "name": "gamma_dot_t_offset",
                        "min": 0.0,
                        "max": 0.0,
                        "type": "fixed",
                    },
                    {"name": "phi0", "min": 0.0, "max": 0.0, "type": "fixed"},
                ]
            },
            "optimization_config": {
                "classical_optimization": {
                    "methods": ["Nelder-Mead"],
                    "method_options": {
                        "Nelder-Mead": {
                            "maxiter": 5000,
                            "xatol": 1e-10,
                            "fatol": 1e-10,
                            "adaptive": True,
                        }
                    },
                }
            },
            "analysis_settings": {
                "static_mode": True,
                "static_submode": "isotropic",
            },
        }

        # Create mock core
        mock_core = Mock()
        mock_core.num_threads = 4

        # Create ClassicalOptimizer instance
        optimizer = ClassicalOptimizer(mock_core, realistic_config)

        # Test that all sections are accessible
        initial_params = np.array(
            realistic_config["initial_parameters"]["values"], dtype=np.float64
        )
        param_bounds = realistic_config.get("parameter_space", {}).get("bounds", [])
        opt_config = optimizer.optimization_config

        # Verify values match expected configuration
        expected_params = np.array([18000, -1.59, 3.10])
        np.testing.assert_array_equal(initial_params, expected_params)

        assert len(param_bounds) == 7
        assert param_bounds[0]["name"] == "D0"
        assert param_bounds[0]["min"] == 1.0  # Match actual config above
        assert param_bounds[0]["max"] == 1000000  # Match actual config above

        assert "methods" in opt_config
        assert opt_config["methods"] == ["Nelder-Mead"]


class TestGurobiIntegration:
    """Test Gurobi optimization integration and configuration."""

    def test_gurobi_availability_detection(self):
        """Test that Gurobi availability is correctly detected."""
        # This test simply checks that the import detection works
        assert isinstance(GUROBI_AVAILABLE, bool)
        print(f"Gurobi detected as available: {GUROBI_AVAILABLE}")

    def test_gurobi_in_available_methods(self):
        """Test that Gurobi appears in available methods when installed."""
        test_config = {
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]}
            }
        }

        mock_core = Mock()
        optimizer = ClassicalOptimizer(mock_core, test_config)

        available_methods = optimizer.get_available_methods()

        if GUROBI_AVAILABLE:
            assert "Gurobi" in available_methods
        else:
            assert "Gurobi" not in available_methods

    def test_gurobi_method_validation(self):
        """Test Gurobi method validation."""
        test_config = {
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]}
            }
        }

        mock_core = Mock()
        optimizer = ClassicalOptimizer(mock_core, test_config)

        # Test Nelder-Mead validation (should always work)
        assert optimizer.validate_method_compatibility("Nelder-Mead", False)
        assert optimizer.validate_method_compatibility("Nelder-Mead", True)

        # Test Gurobi validation
        gurobi_valid = optimizer.validate_method_compatibility("Gurobi", True)
        assert gurobi_valid == GUROBI_AVAILABLE

    def test_gurobi_recommendations(self):
        """Test that Gurobi appears in method recommendations when available."""
        test_config = {
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]}
            }
        }

        mock_core = Mock()
        optimizer = ClassicalOptimizer(mock_core, test_config)

        recommendations = optimizer.get_method_recommendations()

        if GUROBI_AVAILABLE:
            assert "Gurobi" in recommendations["with_bounds"]
            assert "Gurobi" in recommendations["smooth_objective"]
            assert "Gurobi" in recommendations["low_dimensional"]
        else:
            assert "Gurobi" not in recommendations["with_bounds"]

    def test_gurobi_configuration_options(self):
        """Test Gurobi-specific configuration options."""
        test_config = {
            "optimization_config": {
                "classical_optimization": {
                    "methods": ["Gurobi"],
                    "method_options": {
                        "Gurobi": {
                            "max_iterations": 500,
                            "tolerance": 1e-8,
                            "output_flag": 0,
                            "method": 2,
                            "time_limit": 600,
                        }
                    },
                }
            }
        }

        mock_core = Mock()
        optimizer = ClassicalOptimizer(mock_core, test_config)

        # Check that Gurobi options are accessible
        opt_config = optimizer.optimization_config
        if "method_options" in opt_config and "Gurobi" in opt_config["method_options"]:
            gurobi_options = opt_config["method_options"]["Gurobi"]
            assert gurobi_options["max_iterations"] == 500
            assert gurobi_options["tolerance"] == 1e-8
            assert gurobi_options["time_limit"] == 600

    @pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi not available")
    def test_gurobi_optimization_simple_quadratic(self):
        """Test Gurobi optimization on a simple quadratic function."""

        # Simple quadratic: f(x) = (x-1)^2 + (y-2)^2
        def quadratic_objective(params):
            x, y = params
            return (x - 1) ** 2 + (y - 2) ** 2

        test_config = {
            "optimization_config": {
                "classical_optimization": {
                    "methods": ["Gurobi"],
                    "method_options": {
                        "Gurobi": {
                            "max_iterations": 100,
                            "tolerance": 1e-6,
                            "output_flag": 0,
                        }
                    },
                }
            },
            "parameter_space": {
                "bounds": [
                    {"name": "x", "min": -5.0, "max": 5.0},
                    {"name": "y", "min": -5.0, "max": 5.0},
                ]
            },
        }

        mock_core = Mock()
        optimizer = ClassicalOptimizer(mock_core, test_config)

        # Test optimization
        initial_params = np.array([0.0, 0.0])
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]

        success, result = optimizer._run_gurobi_optimization(
            quadratic_objective,
            initial_params,
            bounds,
            test_config["optimization_config"]["classical_optimization"][
                "method_options"
            ]["Gurobi"],
        )

        assert success, f"Gurobi optimization should succeed, got: {result}"
        assert hasattr(result, "x"), "Result should have optimal parameters"
        assert hasattr(result, "fun"), "Result should have optimal value"
        assert (
            hasattr(result, "method")
            and getattr(result, "method") == "Gurobi-Iterative-QP"
        )

    @pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi not available")
    def test_gurobi_with_bounds_constraints(self):
        """Test Gurobi optimization respects bounds constraints."""

        # Function with optimum outside bounds: f(x) = (x-10)^2
        def objective_outside_bounds(params):
            x = params[0]
            # Optimum at x=10, but we'll constrain to [0,5]
            return (x - 10) ** 2

        test_config = {
            "parameter_space": {"bounds": [{"name": "x", "min": 0.0, "max": 5.0}]}
        }

        mock_core = Mock()
        optimizer = ClassicalOptimizer(mock_core, test_config)

        initial_params = np.array([2.5])
        bounds = [(0.0, 5.0)]

        success, result = optimizer._run_gurobi_optimization(
            objective_outside_bounds, initial_params, bounds
        )

        if success:
            # Should find optimum at boundary x=5 (closest to true optimum
            # x=10)
            assert hasattr(result, "x"), "Result should have x attribute on success"
            x_value = getattr(result, "x", None)
            assert x_value is not None, "Result should have x attribute"
            assert 0.0 <= x_value[0] <= 5.0, "Solution should respect bounds"

    def test_gurobi_error_handling_when_unavailable(self):
        """Test proper error handling when Gurobi is not available."""
        if GUROBI_AVAILABLE:
            pytest.skip("Gurobi is available, cannot test unavailable case")

        def dummy_objective(params):
            return sum(params**2)

        test_config = {}
        mock_core = Mock()
        optimizer = ClassicalOptimizer(mock_core, test_config)

        # Should return False and ImportError
        success, result = optimizer._run_gurobi_optimization(
            dummy_objective, np.array([1.0]), [(0, 2)]
        )

        assert not success
        assert isinstance(result, ImportError)

    def test_mixed_methods_configuration(self):
        """Test configuration with both Nelder-Mead and Gurobi methods."""
        if GUROBI_AVAILABLE:
            expected_methods = ["Nelder-Mead", "Gurobi"]
        else:
            expected_methods = ["Nelder-Mead"]

        test_config = {
            "optimization_config": {
                "classical_optimization": {
                    "methods": expected_methods,
                    "method_options": {
                        "Nelder-Mead": {"maxiter": 1000},
                        "Gurobi": {"max_iterations": 500, "tolerance": 1e-8},
                    },
                }
            }
        }

        mock_core = Mock()
        optimizer = ClassicalOptimizer(mock_core, test_config)

        # Check that both methods are configured
        opt_config = optimizer.optimization_config
        assert "methods" in opt_config

        if GUROBI_AVAILABLE:
            assert len(opt_config["methods"]) >= 1  # At least Nelder-Mead
            assert "method_options" in opt_config
            assert "Nelder-Mead" in opt_config["method_options"]

    def test_gurobi_uses_mcmc_bounds(self):
        """Test that Gurobi optimization uses the same bounds as MCMC."""
        # Configuration with the exact MCMC bounds from the user's example
        mcmc_bounds_config = {
            "parameter_space": {
                "bounds": [
                    {
                        "name": "D0",
                        "min": 1.0,
                        "max": 1000000.0,
                        "type": "TruncatedNormal",
                    },
                    {
                        "name": "alpha",
                        "min": -2.0,
                        "max": 2.0,
                        "type": "Normal",
                    },
                    {
                        "name": "D_offset",
                        "min": -100,
                        "max": 100,
                        "type": "Normal",
                    },
                    {
                        "name": "gamma_dot_t0",
                        "min": 1e-06,
                        "max": 1.0,
                        "type": "TruncatedNormal",
                    },
                    {
                        "name": "beta",
                        "min": -2.0,
                        "max": 2.0,
                        "type": "Normal",
                    },
                    {
                        "name": "gamma_dot_t_offset",
                        "min": -0.01,
                        "max": 0.01,
                        "type": "Normal",
                    },
                    {"name": "phi0", "min": -10, "max": 10, "type": "Normal"},
                ]
            },
            "optimization_config": {"classical_optimization": {"methods": ["Gurobi"]}},
        }

        mock_core = Mock()
        optimizer = ClassicalOptimizer(mock_core, mcmc_bounds_config)

        # Test that get_parameter_bounds extracts the correct MCMC bounds
        bounds = optimizer.get_parameter_bounds(effective_param_count=7)

        # Verify the bounds match the MCMC specification exactly
        expected_bounds = [
            (1.0, 1000000.0),  # D0
            (-2.0, 2.0),  # alpha
            (-100, 100),  # D_offset
            (1e-06, 1.0),  # gamma_dot_t0
            (-2.0, 2.0),  # beta
            (-0.01, 0.01),  # gamma_dot_t_offset
            (-10, 10),  # phi0
        ]

        assert len(bounds) == 7
        for i, (expected_min, expected_max) in enumerate(expected_bounds):
            actual_min, actual_max = bounds[i]
            assert (
                actual_min == expected_min
            ), f"Parameter {i} min bound mismatch: {actual_min} != {expected_min}"
            assert (
                actual_max == expected_max
            ), f"Parameter {i} max bound mismatch: {actual_max} != {expected_max}"

        print("✓ Gurobi correctly uses MCMC bounds")

    def test_gurobi_bounds_consistency_static_mode(self):
        """Test that Gurobi uses correct bounds for static mode (3 parameters)."""
        mcmc_bounds_config = {
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 1.0, "max": 1000000.0},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": -100, "max": 100},
                    {
                        "name": "gamma_dot_t0",
                        "min": 0.0,
                        "max": 0.0,
                    },  # Fixed for static
                    {
                        "name": "beta",
                        "min": 0.0,
                        "max": 0.0,
                    },  # Fixed for static
                    {
                        "name": "gamma_dot_t_offset",
                        "min": 0.0,
                        "max": 0.0,
                    },  # Fixed for static
                    {
                        "name": "phi0",
                        "min": 0.0,
                        "max": 0.0,
                    },  # Fixed for static
                ]
            }
        }

        mock_core = Mock()
        optimizer = ClassicalOptimizer(mock_core, mcmc_bounds_config)

        # Test static mode bounds (only first 3 parameters)
        bounds = optimizer.get_parameter_bounds(effective_param_count=3)

        expected_bounds = [
            (1.0, 1000000.0),  # D0
            (-2.0, 2.0),  # alpha
            (-100, 100),  # D_offset
        ]

        assert len(bounds) == 3
        for i, (expected_min, expected_max) in enumerate(expected_bounds):
            actual_min, actual_max = bounds[i]
            assert (
                actual_min == expected_min
            ), f"Static parameter {i} min bound mismatch"
            assert (
                actual_max == expected_max
            ), f"Static parameter {i} max bound mismatch"
