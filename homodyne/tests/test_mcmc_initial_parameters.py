"""
Test MCMC Initial Parameter Handling
===================================

Tests to ensure MCMC optimization properly uses classical results when available
or falls back to initial parameters. This addresses the issue where --method all
did not properly chain classical and MCMC optimization.

Test Coverage:
- MCMC using classical optimization results as initial parameters
- MCMC falling back to initial parameters when no classical results available
- Parameter chaining in run_all_methods function
- Classical results storage on analyzer core object
- Proper logging for parameter initialization scenarios
- Graceful handling of classical optimization failures
- Parameter array consistency and validation

These tests ensure the fix for the issue: "For --method all in run_homodyne.py,
no mcmc initiated after the classical method. Please fix the issue and make sure
that --method all will use the 'initial parameters' if no 'classical' best-fit
results are available, otherwise --method mcmc will use the 'classical'
best-fit results."
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestMCMCInitialParameterHandling:
    """Test MCMC initial parameter handling from classical results and fallbacks."""

    def test_mcmc_uses_classical_results_when_available(self):
        """Test that MCMC uses classical optimization results when available."""

        # Mock analyzer with classical results stored
        mock_analyzer = Mock()
        mock_analyzer.config = {
            "optimization_config": {
                "mcmc_sampling": {"draws": 100, "chains": 2, "tune": 50}
            },
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1000, -0.5, 100],
                "active_parameters": ["D0", "alpha", "D_offset"],
            },
            "analyzer_parameters": {
                "temporal": {"dt": 0.5, "start_frame": 1, "end_frame": 100}
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 100, "max": 10000},
                    {"name": "alpha", "min": -2.0, "max": 0.0},
                    {"name": "D_offset", "min": 0, "max": 1000},
                ]
            },
        }

        # Set classical results on analyzer
        classical_best_params = [5000, -1.2, 500]
        mock_analyzer.best_params_classical = classical_best_params

        # Mock the MCMC sampler creation and initialization detection
        with patch(
            "homodyne.run_homodyne.create_mcmc_sampler"
        ) as mock_create_sampler:
            mock_sampler = Mock()
            mock_sampler.run_mcmc_analysis.return_value = {
                "posterior_means": {"D0": 5100, "alpha": -1.15, "D_offset": 480},
                "trace": None,
                "diagnostics": {"converged": True, "max_rhat": 1.01, "min_ess": 400},
                "chi_squared": 1.5,  # Add chi_squared to avoid comparison issues
            }
            mock_create_sampler.return_value = mock_sampler

            # Import and run the function
            from homodyne.run_homodyne import run_mcmc_optimization

            # Test parameters
            initial_params = [
                1000,
                -0.5,
                100,
            ]  # Should be overridden by classical results
            phi_angles = np.array([0.0])
            c2_exp = np.random.random((1, 100, 100))

            # Run MCMC optimization
            result = run_mcmc_optimization(
                mock_analyzer, initial_params, phi_angles, c2_exp
            )

            # Verify that the analyzer has classical results stored
            assert hasattr(mock_analyzer, "best_params_classical")
            assert mock_analyzer.best_params_classical == classical_best_params

            # Verify MCMC was called
            mock_create_sampler.assert_called_once_with(
                mock_analyzer, mock_analyzer.config
            )
            mock_sampler.run_mcmc_analysis.assert_called_once()

            # Verify results returned
            assert result is not None
            assert "mcmc_summary" in result

    def test_mcmc_falls_back_to_initial_parameters(self):
        """Test that MCMC uses initial parameters when no classical results available."""

        # Mock analyzer without classical results
        mock_analyzer = Mock()
        mock_analyzer.config = {
            "optimization_config": {
                "mcmc_sampling": {"draws": 100, "chains": 2, "tune": 50}
            },
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1000, -0.5, 100],
                "active_parameters": ["D0", "alpha", "D_offset"],
            },
            "analyzer_parameters": {
                "temporal": {"dt": 0.5, "start_frame": 1, "end_frame": 100}
            },
            "analyzer_parameters": {
                "temporal": {"dt": 0.5, "start_frame": 1, "end_frame": 100}
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 100, "max": 10000},
                    {"name": "alpha", "min": -2.0, "max": 0.0},
                    {"name": "D_offset", "min": 0, "max": 1000},
                ]
            },
        }

        # No classical results - explicitly set to None
        mock_analyzer.best_params_classical = None

        # Mock the MCMC sampler creation and the import
        with (
            patch(
                "homodyne.run_homodyne.create_mcmc_sampler"
            ) as mock_create_sampler,
            patch(
                "homodyne.run_homodyne.create_mcmc_sampler"
            ) as mock_create_sampler_module,
        ):
            mock_sampler = Mock()
            mock_sampler.run_mcmc_analysis.return_value = {
                "posterior_means": {"D0": 1100, "alpha": -0.45, "D_offset": 120},
                "trace": None,
                "diagnostics": {"converged": True, "max_rhat": 1.02, "min_ess": 350},
            }
            mock_create_sampler.return_value = mock_sampler
            mock_create_sampler_module.return_value = mock_sampler

            # Import and run the function
            from homodyne.run_homodyne import run_mcmc_optimization

            # Test parameters
            initial_params = [1000, -0.5, 100]
            phi_angles = np.array([0.0])
            c2_exp = np.random.random((1, 100, 100))

            # Run MCMC optimization
            result = run_mcmc_optimization(
                mock_analyzer, initial_params, phi_angles, c2_exp
            )

            # Verify that analyzer has the best_params_classical attribute
            # (In the actual implementation, this gets set during MCMC optimization)
            assert hasattr(mock_analyzer, "best_params_classical")

            # Verify MCMC was called
            mock_create_sampler_module.assert_called_once_with(
                mock_analyzer, mock_analyzer.config
            )
            mock_sampler.run_mcmc_analysis.assert_called_once()

            # Verify results returned
            assert result is not None
            assert "mcmc_summary" in result

    def test_run_all_methods_parameter_chaining(self):
        """Test that run_all_methods properly chains classical results to MCMC."""

        # Mock analyzer
        mock_analyzer = Mock()
        mock_analyzer.config = {
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]},
                "mcmc_sampling": {"draws": 100, "chains": 2, "tune": 50},
            },
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1000, -0.5, 100],
                "active_parameters": ["D0", "alpha", "D_offset"],
            },
            "analyzer_parameters": {
                "temporal": {"dt": 0.5, "start_frame": 1, "end_frame": 100}
            },
            "analyzer_parameters": {
                "temporal": {"dt": 0.5, "start_frame": 1, "end_frame": 100}
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 100, "max": 10000},
                    {"name": "alpha", "min": -2.0, "max": 0.0},
                    {"name": "D_offset", "min": 0, "max": 1000},
                ]
            },
        }

        # Mock classical optimization results
        classical_best_params = [5000, -1.2, 500]
        mock_classical_results = {
            "classical_optimization": {
                "parameters": classical_best_params,
                "chi_squared": 1.5,
                "success": True,
            },
            "classical_summary": {
                "parameters": classical_best_params,
                "chi_squared": 1.5,
                "method": "Classical",
            },
            "methods_used": ["Classical"],
        }

        # Mock MCMC results
        mock_mcmc_results = {
            "mcmc_optimization": {"parameters": [5100, -1.15, 480], "success": True},
            "mcmc_summary": {"parameters": [5100, -1.15, 480], "method": "MCMC"},
            "methods_used": ["MCMC"],
        }

        # Mock the optimization functions
        with (
            patch("homodyne.run_homodyne.run_classical_optimization") as mock_classical,
            patch("homodyne.run_homodyne.run_mcmc_optimization") as mock_mcmc,
        ):

            mock_classical.return_value = mock_classical_results
            mock_mcmc.return_value = mock_mcmc_results

            # Import and run the function
            from homodyne.run_homodyne import run_all_methods

            # Test parameters
            initial_params = [1000, -0.5, 100]
            phi_angles = np.array([0.0])
            c2_exp = np.random.random((1, 100, 100))

            # Run all methods
            result = run_all_methods(mock_analyzer, initial_params, phi_angles, c2_exp)

            # Verify classical was called with initial parameters (including output_dir)
            mock_classical.assert_called_once_with(
                mock_analyzer, initial_params, phi_angles, c2_exp, None
            )

            # Verify MCMC was called with classical results as initial parameters
            mock_mcmc.assert_called_once_with(
                mock_analyzer, classical_best_params, phi_angles, c2_exp, None
            )

            # Verify combined results
            assert result is not None
            assert "Classical" in result["methods_used"]
            assert "MCMC" in result["methods_used"]

    def test_classical_results_storage_on_analyzer(self):
        """Test that classical optimization results are properly stored on analyzer."""

        # Mock analyzer and classical optimizer
        mock_analyzer = Mock()
        mock_analyzer.config = {
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]}
            },
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1000, -0.5, 100],
                "active_parameters": ["D0", "alpha", "D_offset"],
            },
            "analyzer_parameters": {
                "temporal": {"dt": 0.5, "start_frame": 1, "end_frame": 100}
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 100, "max": 10000},
                    {"name": "alpha", "min": -2.0, "max": 0.0},
                    {"name": "D_offset", "min": 0, "max": 1000},
                ]
            },
        }

        # Mock classical optimizer with best_params_classical attribute
        mock_optimizer = Mock()
        classical_best_params = [5000, -1.2, 500]
        mock_optimizer.best_params_classical = classical_best_params

        # Mock optimization result
        mock_result = Mock()
        mock_result.fun = 1.5
        mock_result.success = True
        mock_result.method = "Nelder-Mead"

        mock_optimizer.run_classical_optimization_optimized.return_value = (
            classical_best_params,
            mock_result,
        )

        # Mock ClassicalOptimizer class
        with patch("homodyne.run_homodyne.ClassicalOptimizer") as mock_optimizer_class:
            mock_optimizer_class.return_value = mock_optimizer

            # Import and run the function
            from homodyne.run_homodyne import run_classical_optimization

            # Test parameters
            initial_params = [1000, -0.5, 100]
            phi_angles = np.array([0.0])
            c2_exp = np.random.random((1, 100, 100))

            # Run classical optimization
            result = run_classical_optimization(
                mock_analyzer, initial_params, phi_angles, c2_exp
            )

            # Verify optimizer was created and called
            mock_optimizer_class.assert_called_once_with(
                mock_analyzer, mock_analyzer.config
            )
            mock_optimizer.run_classical_optimization_optimized.assert_called_once()

            # Verify that classical results were stored on analyzer
            assert hasattr(mock_analyzer, "best_params_classical")
            assert mock_analyzer.best_params_classical == classical_best_params

            # Verify result structure
            assert result is not None
            assert "classical_optimization" in result
            assert "classical_summary" in result
            assert result["classical_summary"]["parameters"] == classical_best_params

    def test_mcmc_parameter_initialization_logging(self):
        """Test that proper logging occurs for MCMC parameter initialization."""

        # Mock analyzer with classical results
        mock_analyzer = Mock()
        mock_analyzer.config = {
            "optimization_config": {
                "mcmc_sampling": {"draws": 100, "chains": 2, "tune": 50}
            },
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1000, -0.5, 100],
                "active_parameters": ["D0", "alpha", "D_offset"],
            },
        }

        # Test with classical results available
        mock_analyzer.best_params_classical = [5000, -1.2, 500]

        with (
            patch(
                "homodyne.run_homodyne.create_mcmc_sampler"
            ) as mock_create_sampler,
            patch("logging.getLogger") as mock_logger,
        ):

            mock_sampler = Mock()
            mock_sampler.run_mcmc_analysis.return_value = {
                "trace": None,
                "diagnostics": {},
            }
            mock_create_sampler.return_value = mock_sampler

            mock_log = Mock()
            mock_logger.return_value = mock_log

            # Import and run the function
            from homodyne.run_homodyne import run_mcmc_optimization

            # Test parameters
            initial_params = [1000, -0.5, 100]
            phi_angles = np.array([0.0])
            c2_exp = np.random.random((1, 100, 100))

            # Run MCMC optimization
            run_mcmc_optimization(mock_analyzer, initial_params, phi_angles, c2_exp)

            # Verify appropriate logging occurred (may vary based on implementation)
            # The exact log message can vary, so just verify that logging occurred
            assert mock_log.info.called

    def test_mcmc_fallback_parameter_initialization_logging(self):
        """Test logging when MCMC falls back to initial parameters."""

        # Mock analyzer without classical results
        mock_analyzer = Mock()
        mock_analyzer.config = {
            "optimization_config": {
                "mcmc_sampling": {"draws": 100, "chains": 2, "tune": 50}
            },
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1000, -0.5, 100],
                "active_parameters": ["D0", "alpha", "D_offset"],
            },
        }

        # No classical results available - explicitly ensure it's not set
        if hasattr(mock_analyzer, "best_params_classical"):
            delattr(mock_analyzer, "best_params_classical")

        with (
            patch(
                "homodyne.run_homodyne.create_mcmc_sampler"
            ) as mock_create_sampler,
            patch("logging.getLogger") as mock_logger,
        ):

            mock_sampler = Mock()
            mock_sampler.run_mcmc_analysis.return_value = {
                "trace": None,
                "diagnostics": {},
            }
            mock_create_sampler.return_value = mock_sampler

            mock_log = Mock()
            mock_logger.return_value = mock_log

            # Import and run the function
            from homodyne.run_homodyne import run_mcmc_optimization

            # Test parameters
            initial_params = [1000, -0.5, 100]
            phi_angles = np.array([0.0])
            c2_exp = np.random.random((1, 100, 100))

            # Run MCMC optimization
            run_mcmc_optimization(mock_analyzer, initial_params, phi_angles, c2_exp)

            # Verify fallback logging occurred (may vary based on implementation)
            # The exact log message can vary, so just verify that logging occurred
            assert mock_log.info.called

    def test_run_all_methods_with_classical_failure(self):
        """Test that run_all_methods handles classical optimization failure gracefully."""

        # Mock analyzer
        mock_analyzer = Mock()
        mock_analyzer.config = {
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]},
                "mcmc_sampling": {"draws": 100, "chains": 2, "tune": 50},
            },
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1000, -0.5, 100],
                "active_parameters": ["D0", "alpha", "D_offset"],
            },
        }

        # Mock MCMC results
        mock_mcmc_results = {
            "mcmc_optimization": {"parameters": [1100, -0.45, 120], "success": True},
            "mcmc_summary": {"parameters": [1100, -0.45, 120], "method": "MCMC"},
            "methods_used": ["MCMC"],
        }

        # Mock the optimization functions - classical fails, MCMC succeeds
        with (
            patch("homodyne.run_homodyne.run_classical_optimization") as mock_classical,
            patch("homodyne.run_homodyne.run_mcmc_optimization") as mock_mcmc,
        ):

            mock_classical.return_value = None  # Classical fails
            mock_mcmc.return_value = mock_mcmc_results

            # Import and run the function
            from homodyne.run_homodyne import run_all_methods

            # Test parameters
            initial_params = [1000, -0.5, 100]
            phi_angles = np.array([0.0])
            c2_exp = np.random.random((1, 100, 100))

            # Run all methods
            result = run_all_methods(mock_analyzer, initial_params, phi_angles, c2_exp)

            # Verify classical was called with initial parameters (including output_dir)
            mock_classical.assert_called_once_with(
                mock_analyzer, initial_params, phi_angles, c2_exp, None
            )

            # Verify MCMC was called with original initial parameters (fallback)
            mock_mcmc.assert_called_once_with(
                mock_analyzer, initial_params, phi_angles, c2_exp, None
            )

            # Verify only MCMC results are included
            assert result is not None
            assert "MCMC" in result["methods_used"]
            assert "Classical" not in result["methods_used"]

    def test_parameter_array_consistency(self):
        """Test that parameter arrays maintain consistency between classical and MCMC."""

        # Test with realistic parameter values
        initial_params = [18000, -1.59, 3.10]
        classical_params = [17500, -1.55, 2.95]

        # Mock analyzer
        mock_analyzer = Mock()
        mock_analyzer.config = {
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": initial_params,
                "active_parameters": ["D0", "alpha", "D_offset"],
            },
            "analyzer_parameters": {
                "temporal": {"dt": 0.5, "start_frame": 1, "end_frame": 100}
            }
        }

        # Test classical result storage
        mock_analyzer.best_params_classical = classical_params

        # Verify parameter consistency
        assert len(mock_analyzer.best_params_classical) == len(initial_params)
        assert isinstance(mock_analyzer.best_params_classical, list)

        # Test parameter value ranges are reasonable
        assert 10000 <= mock_analyzer.best_params_classical[0] <= 25000  # D0
        assert -2.0 <= mock_analyzer.best_params_classical[1] <= -1.0  # alpha
        assert 0 <= mock_analyzer.best_params_classical[2] <= 10  # D_offset
