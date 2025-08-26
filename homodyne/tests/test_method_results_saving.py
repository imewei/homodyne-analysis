"""
Test suite for individual method results saving functionality.

Tests the saving of fitted parameters with uncertainties for all classical
and robust optimization methods.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from homodyne.run_homodyne import (_estimate_parameter_uncertainties,
                                   _save_individual_method_results,
                                   _save_individual_robust_method_results)

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestMethodResultsSaving:
    """Test individual method results saving functionality."""

    @pytest.fixture
    def mock_analyzer(self):
        """Create a mock analyzer with required methods and config."""
        analyzer = Mock()
        analyzer.config = {
            "initial_parameters": {
                "parameter_names": ["amplitude", "frequency", "phase"],
                "parameter_units": {
                    "amplitude": "arb",
                    "frequency": "Hz",
                    "phase": "rad",
                },
            }
        }

        # Mock dt for time array calculation
        analyzer.dt = 0.1

        # Mock the compute_c2_correlation_optimized method
        def mock_compute_c2(params, phi_angles):
            # Return synthetic data based on parameters
            return np.random.randn(len(phi_angles), 100) * params[0]

        analyzer.compute_c2_correlation_optimized = Mock(
            side_effect=mock_compute_c2)
        return analyzer

    @pytest.fixture
    def mock_result_classical(self):
        """Create a mock classical optimization result with method_results."""
        result = Mock()
        result.x = np.array([1.5, 2.0, 0.5])
        result.fun = 0.5
        result.success = True
        result.best_method = "Gurobi"

        # Add method_results dictionary
        result.method_results = {
            "Nelder-Mead": {
                "parameters": [1.48, 1.98, 0.52],
                "chi_squared": 0.55,
                "success": True,
                "iterations": 150,
                "function_evaluations": 280,
                "message": "Optimization terminated successfully",
            },
            "Gurobi": {
                "parameters": [1.5, 2.0, 0.5],
                "chi_squared": 0.5,
                "success": True,
                "iterations": 50,
                "function_evaluations": 100,
                "message": "Optimal solution found",
            },
            "Robust-Wasserstein": {
                "parameters": [1.52, 2.02, 0.48],
                "chi_squared": 0.58,
                "success": True,
                "iterations": 80,
                "function_evaluations": 160,
                "message": "Robust optimization converged",
            },
        }

        return result

    @pytest.fixture
    def mock_result_robust(self):
        """Create a mock robust optimization result with method_results."""
        result = Mock()
        result.x = np.array([1.51, 2.01, 0.49])
        result.fun = 0.52
        result.success = True
        result.best_method = "Robust-Wasserstein"

        # Add robust method_results dictionary
        result.method_results = {
            "Robust-Wasserstein": {
                "parameters": [1.51, 2.01, 0.49],
                "chi_squared": 0.52,
                "success": True,
                "iterations": 75,
                "solve_time": 2.5,
                "status": "optimal",
                "robust_info": {
                    "uncertainty_radius": 0.03,
                    "regularization_alpha": 0.01,
                },
            },
            "Robust-Scenario": {
                "parameters": [1.49, 1.99, 0.51],
                "chi_squared": 0.54,
                "success": True,
                "iterations": 60,
                "solve_time": 3.2,
                "status": "optimal",
                "robust_info": {"n_scenarios": 15, "worst_case_value": 0.65},
            },
        }

        return result

    @pytest.fixture
    def test_data(self):
        """Create test experimental data."""
        phi_angles = np.array([0, 45, 90, 135])
        c2_exp = np.random.randn(4, 100)
        return phi_angles, c2_exp

    @pytest.fixture
    def time_arrays(self):
        """Create proper time arrays with dt = 0.1."""
        dt = 0.1
        n_t1, n_t2 = 100, 100  # Match c2_exp shape from test_data
        t1 = np.arange(n_t1) * dt
        t2 = np.arange(n_t2) * dt
        return t1, t2

    def test_save_individual_classical_methods(
        self,
        mock_analyzer,
        mock_result_classical,
        test_data,
        time_arrays,
        tmp_path,
    ):
        """Test saving individual classical method results."""
        phi_angles, c2_exp = test_data
        t1, t2 = time_arrays

        # Call the function
        _save_individual_method_results(
            mock_analyzer,
            mock_result_classical,
            phi_angles,
            c2_exp,
            tmp_path,
            t1,
            t2,
        )

        # Check that method directories were created
        classical_dir = tmp_path / "classical"
        assert classical_dir.exists()

        # Check for each method's directory
        for method_name in ["nelder_mead", "gurobi", "robust_wasserstein"]:
            method_dir = classical_dir / method_name
            assert method_dir.exists(
            ), f"Directory for {method_name} not created"

            # Check for parameters.json
            params_file = method_dir / "parameters.json"
            assert (
                params_file.exists()
            ), f"parameters.json not created for {method_name}"

            # Load and validate JSON content
            with open(params_file, "r") as f:
                method_info = json.load(f)

            assert "parameters" in method_info
            assert "goodness_of_fit" in method_info
            assert "convergence_info" in method_info
            assert "data_info" in method_info

            # Check parameter structure
            for param_name in ["amplitude", "frequency", "phase"]:
                assert param_name in method_info["parameters"]
                param_data = method_info["parameters"][param_name]
                assert "value" in param_data
                assert "uncertainty" in param_data
                assert "unit" in param_data

            # Check for fitted_data.npz
            data_file = method_dir / "fitted_data.npz"
            assert data_file.exists(
            ), f"fitted_data.npz not created for {method_name}"

            # Load and validate numpy data
            data = np.load(data_file)
            assert "c2_fitted" in data
            assert "c2_experimental" in data
            assert "residuals" in data
            assert "parameters" in data
            assert "uncertainties" in data
            assert "chi_squared" in data
            assert "t1" in data
            assert "t2" in data

            # Validate time arrays - should be proper dt-based time arrays
            assert len(data["t1"]) > 0, "t1 array should not be empty"
            assert len(data["t2"]) > 0, "t2 array should not be empty"
            # Check that t1 and t2 start from 0 and have constant spacing (dt)
            if len(data["t1"]) > 1:
                assert data["t1"][0] == 0.0, "t1 should start from 0"
                dt_t1 = data["t1"][1] - data["t1"][0]
                assert dt_t1 > 0, "dt should be positive"
            if len(data["t2"]) > 1:
                assert data["t2"][0] == 0.0, "t2 should start from 0"
                dt_t2 = data["t2"][1] - data["t2"][0]
                assert dt_t2 > 0, "dt should be positive"

        # Check for summary file
        summary_file = classical_dir / "all_classical_methods_summary.json"
        assert summary_file.exists()

        with open(summary_file, "r") as f:
            summary = json.load(f)

        assert summary["analysis_type"] == "Classical Optimization"
        assert len(summary["methods_analyzed"]) == 3
        assert "results" in summary

    def test_save_individual_robust_methods(
        self,
        mock_analyzer,
        mock_result_robust,
        test_data,
        time_arrays,
        tmp_path,
    ):
        """Test saving individual robust method results."""
        phi_angles, c2_exp = test_data
        t1, t2 = time_arrays

        # Call the function
        _save_individual_robust_method_results(
            mock_analyzer,
            mock_result_robust,
            phi_angles,
            c2_exp,
            tmp_path,
            t1,
            t2,
        )

        # Check that robust directory was created
        robust_dir = tmp_path / "robust"
        assert robust_dir.exists()

        # Check for each method's directory (using actual naming convention)
        for method_name in ["wasserstein", "scenario"]:
            method_dir = robust_dir / method_name
            assert method_dir.exists(
            ), f"Directory for {method_name} not created"

            # Check for parameters.json
            params_file = method_dir / "parameters.json"
            assert params_file.exists()

            # Load and validate JSON content
            with open(params_file, "r") as f:
                method_info = json.load(f)

            assert method_info["method_type"] == "Robust Optimization"
            assert "robust_specific" in method_info

            # Check robust-specific information
            robust_info = method_info["robust_specific"]
            if "wasserstein" in method_name:
                assert "uncertainty_radius" in robust_info
            elif "scenario" in method_name:
                assert "n_scenarios" in robust_info

            # Check for fitted_data.npz
            data_file = method_dir / "fitted_data.npz"
            assert data_file.exists(
            ), f"fitted_data.npz not created for {method_name}"

            # Load and validate numpy data
            data = np.load(data_file)
            assert "c2_fitted" in data
            assert "c2_experimental" in data
            assert "residuals" in data
            assert "parameters" in data
            assert "uncertainties" in data
            assert "chi_squared" in data
            assert "t1" in data
            assert "t2" in data

            # Validate time arrays - should be proper dt-based time arrays
            assert len(data["t1"]) > 0, "t1 array should not be empty"
            assert len(data["t2"]) > 0, "t2 array should not be empty"
            # Check that t1 and t2 start from 0 and have constant spacing (dt)
            if len(data["t1"]) > 1:
                assert data["t1"][0] == 0.0, "t1 should start from 0"
                dt_t1 = data["t1"][1] - data["t1"][0]
                assert dt_t1 > 0, "dt should be positive"
            if len(data["t2"]) > 1:
                assert data["t2"][0] == 0.0, "t2 should start from 0"
                dt_t2 = data["t2"][1] - data["t2"][0]
                assert dt_t2 > 0, "dt should be positive"

    def test_estimate_parameter_uncertainties(self, mock_analyzer, test_data):
        """Test parameter uncertainty estimation."""
        phi_angles, c2_exp = test_data
        params = np.array([1.5, 2.0, 0.5])
        chi_squared_min = 0.5

        # Call the uncertainty estimation
        uncertainties = _estimate_parameter_uncertainties(
            mock_analyzer, params, phi_angles, c2_exp, chi_squared_min
        )

        # Check that uncertainties were calculated
        assert len(uncertainties) == len(params)
        assert all(
            u > 0 for u in uncertainties), "All uncertainties should be positive"
        assert all(
            u < abs(p) for u, p in zip(uncertainties, params)
        ), "Uncertainties should be reasonable relative to parameter values"

    def test_handles_missing_method_results(
        self, mock_analyzer, test_data, time_arrays, tmp_path
    ):
        """Test handling of results without method_results attribute."""
        phi_angles, c2_exp = test_data
        t1, t2 = time_arrays

        # Create result without method_results
        result = Mock()
        result.x = np.array([1.5, 2.0, 0.5])
        # No method_results attribute

        # Should handle gracefully without error
        _save_individual_method_results(
            mock_analyzer, result, phi_angles, c2_exp, tmp_path, t1, t2
        )

        # Check that classical directory was created but no method subdirs
        classical_dir = tmp_path / "classical"
        assert classical_dir.exists()

        # Should not have method subdirectories
        method_dirs = [d for d in classical_dir.iterdir() if d.is_dir()]
        assert len(method_dirs) == 0

    def test_handles_failed_methods(
        self, mock_analyzer, test_data, time_arrays, tmp_path
    ):
        """Test that failed methods are skipped."""
        phi_angles, c2_exp = test_data
        t1, t2 = time_arrays

        # Create result with mix of successful and failed methods
        result = Mock()
        result.x = np.array([1.5, 2.0, 0.5])
        result.method_results = {
            "Nelder-Mead": {
                "parameters": [1.5, 2.0, 0.5],
                "success": True,
                "chi_squared": 0.5,
            },
            "Gurobi": {
                "parameters": None,
                "success": False,
                "error": "Optimization failed",
            },
        }

        _save_individual_method_results(
            mock_analyzer, result, phi_angles, c2_exp, tmp_path, t1, t2
        )

        classical_dir = tmp_path / "classical"

        # Should only create directory for successful method
        assert (classical_dir / "nelder_mead").exists()
        assert not (classical_dir / "gurobi").exists()

    def test_parameter_names_fallback(
        self,
        mock_analyzer,
        mock_result_classical,
        test_data,
        time_arrays,
        tmp_path,
    ):
        """Test fallback parameter naming when names not in config."""
        phi_angles, c2_exp = test_data
        t1, t2 = time_arrays

        # Remove parameter names from config
        mock_analyzer.config["initial_parameters"]["parameter_names"] = []

        _save_individual_method_results(
            mock_analyzer,
            mock_result_classical,
            phi_angles,
            c2_exp,
            tmp_path,
            t1,
            t2,
        )

        # Load a method's parameters file
        params_file = tmp_path / "classical" / "nelder_mead" / "parameters.json"
        with open(params_file, "r") as f:
            method_info = json.load(f)

        # Should have default parameter names
        assert "param_0" in method_info["parameters"]
        assert "param_1" in method_info["parameters"]
        assert "param_2" in method_info["parameters"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
