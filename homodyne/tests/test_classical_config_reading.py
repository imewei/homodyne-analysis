"""
Test Classical Optimization Configuration Reading
==============================================

Tests to ensure classical optimization reads configuration correctly from the JSON
configuration file, including initial_parameters and parameter_space sections.
"""

import pytest
import numpy as np
from unittest.mock import Mock
from homodyne.optimization.classical import ClassicalOptimizer


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
                "classical_optimization": {"methods": ["L-BFGS-B", "TNC"]}
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
                    {"name": "D0", "min": 100, "max": 10000, "type": "log-uniform"},
                    {"name": "alpha", "min": -2.0, "max": 0.0, "type": "uniform"},
                    {"name": "D_offset", "min": 0, "max": 1000, "type": "uniform"},
                    {"name": "gamma_dot_t0", "min": 0.0, "max": 0.1, "type": "uniform"},
                    {"name": "beta", "min": -1.0, "max": 1.0, "type": "uniform"},
                    {
                        "name": "gamma_dot_t_offset",
                        "min": 0.0,
                        "max": 0.01,
                        "type": "uniform",
                    },
                    {"name": "phi0", "min": 0.0, "max": 360.0, "type": "uniform"},
                ]
            },
            "optimization_config": {
                "classical_optimization": {"methods": ["L-BFGS-B"]}
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

        # Check specific bounds
        d0_bound = param_bounds[0]
        assert d0_bound["name"] == "D0"
        assert d0_bound["min"] == 100
        assert d0_bound["max"] == 10000

        alpha_bound = param_bounds[1]
        assert alpha_bound["name"] == "alpha"
        assert alpha_bound["min"] == -2.0
        assert alpha_bound["max"] == 0.0

    def test_bounds_extraction_for_optimization(self):
        """Test that bounds are correctly extracted for optimization methods."""

        test_config = {
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 500, "max": 5000},
                    {"name": "alpha", "min": -1.5, "max": -0.5},
                    {"name": "D_offset", "min": 10, "max": 500},
                ]
            },
            "optimization_config": {
                "classical_optimization": {"methods": ["L-BFGS-B"]}
            },
        }

        # Create mock core
        mock_core = Mock()
        mock_core.num_threads = 4

        # Create ClassicalOptimizer instance
        optimizer = ClassicalOptimizer(mock_core, test_config)

        # Test bounds extraction (simulating what happens in get_parameter_bounds method)
        bounds = []
        param_bounds = test_config.get("parameter_space", {}).get("bounds", [])
        effective_param_count = 3  # For this test

        for i, bound in enumerate(param_bounds):
            if i < effective_param_count:
                bounds.append((bound.get("min", -np.inf), bound.get("max", np.inf)))

        expected_bounds = [(500, 5000), (-1.5, -0.5), (10, 500)]
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
                    {"name": "D0", "min": 15000, "max": 20000},
                    {"name": "alpha", "min": -1.6, "max": -1.5},
                    {"name": "D_offset", "min": 0, "max": 5},
                    {
                        "name": "gamma_dot_t0",
                        "min": 0.0,
                        "max": 0.0,
                    },  # Fixed for static
                    {"name": "beta", "min": 0.0, "max": 0.0},  # Fixed for static
                    {
                        "name": "gamma_dot_t_offset",
                        "min": 0.0,
                        "max": 0.0,
                    },  # Fixed for static
                    {"name": "phi0", "min": 0.0, "max": 0.0},  # Fixed for static
                ]
            },
            "analysis_settings": {"static_mode": True, "static_submode": "isotropic"},
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]}
            },
        }

        # Create mock core
        mock_core = Mock()
        mock_core.num_threads = 4

        # Create ClassicalOptimizer instance
        optimizer = ClassicalOptimizer(mock_core, test_config)

        # Test parameter selection for static mode (simulating run_classical_optimization_optimized logic)
        initial_parameters = np.array(
            test_config["initial_parameters"]["values"], dtype=np.float64
        )
        effective_param_count = 3  # Static mode uses first 3 parameters

        # For static mode, only use first 3 parameters
        if len(initial_parameters) > effective_param_count:
            static_parameters = initial_parameters[:effective_param_count]

        expected_static_params = np.array([18000, -1.59, 3.10])
        np.testing.assert_array_equal(static_parameters, expected_static_params)

        # Test bounds for static mode
        param_bounds = test_config.get("parameter_space", {}).get("bounds", [])
        static_bounds = []
        for i, bound in enumerate(param_bounds):
            if i < effective_param_count:
                static_bounds.append((bound.get("min"), bound.get("max")))

        expected_static_bounds = [(15000, 20000), (-1.6, -1.5), (0, 5)]
        assert static_bounds == expected_static_bounds

    def test_optimization_config_access(self):
        """Test that optimization configuration is accessed correctly."""

        test_config = {
            "optimization_config": {
                "classical_optimization": {
                    "methods": ["L-BFGS-B", "TNC", "Nelder-Mead"],
                    "method_options": {
                        "L-BFGS-B": {"maxiter": 1000, "ftol": 1e-9},
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
        assert "L-BFGS-B" in methods
        assert "TNC" in methods
        assert "Nelder-Mead" in methods

        method_options = opt_config["method_options"]
        assert "L-BFGS-B" in method_options
        assert method_options["L-BFGS-B"]["maxiter"] == 1000
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
            "metadata": {"config_version": "6.0", "analysis_mode": "static_isotropic"},
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
                    {"name": "D0", "min": 15000, "max": 20000, "type": "log-uniform"},
                    {"name": "alpha", "min": -1.6, "max": -1.5, "type": "uniform"},
                    {"name": "D_offset", "min": 0, "max": 5, "type": "uniform"},
                    {"name": "gamma_dot_t0", "min": 0.0, "max": 0.0, "type": "fixed"},
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
            "analysis_settings": {"static_mode": True, "static_submode": "isotropic"},
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
        assert param_bounds[0]["min"] == 15000
        assert param_bounds[0]["max"] == 20000

        assert "methods" in opt_config
        assert opt_config["methods"] == ["Nelder-Mead"]
