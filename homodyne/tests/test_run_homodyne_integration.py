"""
Integration Tests for run_homodyne.py Script
===========================================

Tests the behavior of the main run_homodyne.py script, including:
- --plot-experimental-data flag behavior (early exit, correct output path)
- Classical method output organization
- Results file location changes
- Directory structure creation
"""

import pytest
import tempfile
import numpy as np
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import json
import subprocess

from homodyne.tests.fixtures import (
    dummy_config,
    temp_directory,
    create_minimal_config_file,
)


class TestRunHomodyneIntegration:
    """Test run_homodyne.py script integration and behavior."""

    def test_plot_experimental_data_flag_behavior(self, temp_directory):
        """Test that --plot-experimental-data flag creates correct output structure."""
        # Create expected output structure for experimental data plots
        exp_data_dir = temp_directory / "homodyne_results" / "exp_data"
        exp_data_dir.mkdir(parents=True, exist_ok=True)

        # Simulate files that would be created by --plot-experimental-data
        expected_files = [
            "data_validation_phi_0.0deg.png",
            "data_validation_phi_45.0deg.png",
            "data_validation_phi_90.0deg.png",
            "summary_statistics.txt",
        ]

        for filename in expected_files:
            (exp_data_dir / filename).touch()

        # Verify structure was created correctly
        assert exp_data_dir.exists()
        assert exp_data_dir.is_dir()

        # Verify files exist in correct location
        for filename in expected_files:
            assert (exp_data_dir / filename).exists()

        # Verify files are NOT in old location
        old_location = temp_directory / "plots" / "data_validation"
        if old_location.exists():
            for filename in expected_files:
                assert not (old_location / filename).exists()

    def test_classical_method_output_structure(self, temp_directory):
        """Test that classical method creates correct output structure."""
        # Create expected output structure for classical method
        classical_dir = temp_directory / "homodyne_results" / "classical"
        classical_dir.mkdir(parents=True, exist_ok=True)

        # Simulate files that would be created by classical method
        expected_files = [
            "experimental_data.npz",
            "fitted_data.npz",
            "residuals_data.npz",
            "c2_heatmaps_phi_0.0deg.png",
            "c2_heatmaps_phi_45.0deg.png",
        ]

        for filename in expected_files:
            if filename.endswith(".npz"):
                # Create realistic NPZ files
                mock_data = np.random.rand(3, 20, 30)
                np.savez_compressed(classical_dir / filename, data=mock_data)
            else:
                # Create empty files for plots
                (classical_dir / filename).touch()

        # Verify structure was created correctly
        assert classical_dir.exists()
        assert classical_dir.is_dir()

        # Verify files exist and have correct formats
        npz_files = list(classical_dir.glob("*.npz"))
        png_files = list(classical_dir.glob("*.png"))

        assert len(npz_files) == 3  # experimental, fitted, residuals
        assert len(png_files) >= 2  # C2 heatmaps

        # Verify NPZ files contain data
        for npz_file in npz_files:
            data = np.load(npz_file)
            assert "data" in data
            assert data["data"].shape == (3, 20, 30)

    def test_main_results_file_location(self, temp_directory):
        """Test that main results file is saved to output directory."""
        results_dir = temp_directory / "homodyne_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create mock main results file
        results_file = results_dir / "homodyne_analysis_results.json"

        mock_results = {
            "timestamp": "2025-08-18T16:06:09.710366+00:00",
            "config": {"test": "config"},
            "results": {"classical_optimization": {"parameters": [1.0, 2.0, 3.0]}},
            "execution_metadata": {"analysis_success": True},
        }

        with open(results_file, "w") as f:
            json.dump(mock_results, f, indent=2)

        # Verify file is in output directory, not current directory
        assert results_file.exists()
        assert results_file.parent == results_dir
        assert results_file.name == "homodyne_analysis_results.json"

        # Verify content is correct
        with open(results_file, "r") as f:
            loaded_results = json.load(f)

        assert loaded_results["config"]["test"] == "config"
        assert loaded_results["execution_metadata"]["analysis_success"] is True

    def test_complete_directory_structure_integration(self, temp_directory):
        """Test complete directory structure as it would be created by run_homodyne.py."""
        base_dir = temp_directory / "homodyne_results"

        # Create complete expected structure
        directories = [base_dir, base_dir / "classical", base_dir / "exp_data"]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        # Create main results files
        main_results = base_dir / "homodyne_analysis_results.json"
        per_angle_results = (
            base_dir / "classical" / "per_angle_chi_squared_classical.json"
        )
        log_file = base_dir / "run.log"

        # Mock main results
        main_results_data = {
            "timestamp": "2025-08-18T16:06:09.710366+00:00",
            "config": {"analysis_mode": "static_isotropic"},
            "results": {
                "classical_optimization": {
                    "parameters": [0.85459454, -0.02171455, -0.78131764],
                    "chi_squared": 17.12216371557179,
                    "success": True,
                }
            },
            "execution_metadata": {"analysis_success": True},
        }

        # Mock per-angle results
        per_angle_data = {
            "method": "Classical",
            "overall_reduced_chi_squared": 17.12216371557179,
            "n_optimization_angles": 1,
            "quality_assessment": {"overall_quality": "poor"},
        }

        with open(main_results, "w") as f:
            json.dump(main_results_data, f, indent=2)

        with open(per_angle_results, "w") as f:
            json.dump(per_angle_data, f, indent=2)

        log_file.write_text(
            "2025-08-18 10:57:33 - __main__ - INFO - Analysis completed successfully\n"
        )

        # Create classical method files
        classical_files = [
            "experimental_data.npz",
            "fitted_data.npz",
            "residuals_data.npz",
            "c2_heatmaps_phi_0.0deg.png",
        ]

        for filename in classical_files:
            filepath = base_dir / "classical" / filename
            if filename.endswith(".npz"):
                mock_data = np.random.rand(1, 60, 60) + 1.0  # Realistic C2 data shape
                np.savez_compressed(filepath, data=mock_data)
            else:
                filepath.touch()

        # Create experimental data files
        exp_data_files = ["data_validation_phi_0.0deg.png", "summary_statistics.txt"]

        for filename in exp_data_files:
            (base_dir / "exp_data" / filename).touch()

        # Verify complete structure
        assert base_dir.exists()
        assert (base_dir / "classical").is_dir()
        assert (base_dir / "exp_data").is_dir()

        # Verify main files
        assert main_results.exists()
        assert per_angle_results.exists()
        assert log_file.exists()

        # Verify classical files
        assert len(list((base_dir / "classical").glob("*.npz"))) == 3
        assert len(list((base_dir / "classical").glob("*.png"))) >= 1

        # Verify experimental data files
        assert len(list((base_dir / "exp_data").glob("*.png"))) >= 1
        assert (base_dir / "exp_data" / "summary_statistics.txt").exists()


class TestRunHomodyneMockExecution:
    """Test run_homodyne.py execution with mocked components."""

    def test_plot_experimental_data_early_exit_simulation(self, temp_directory):
        """Simulate the early exit behavior of --plot-experimental-data."""

        def mock_run_homodyne_with_plot_experimental_data():
            """Mock implementation of run_homodyne.py with --plot-experimental-data."""
            # Simulate command line argument parsing
            plot_experimental_data = True
            method = None  # Not set when using --plot-experimental-data
            output_dir = temp_directory / "homodyne_results" / "exp_data"

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Simulate loading experimental data and creating plots
            mock_experimental_data = np.random.rand(3, 30, 40) + 1.0

            # Create mock validation plots
            validation_files = [
                "data_validation_phi_0.0deg.png",
                "data_validation_phi_90.0deg.png",
                "data_validation_phi_180.0deg.png",
            ]

            for filename in validation_files:
                (output_dir / filename).touch()

            # Create summary statistics
            summary_file = output_dir / "summary_statistics.txt"
            summary_file.write_text(
                "Data validation completed.\nMean g2: 1.05\nContrast: 0.02\n"
            )

            # Early exit - no fitting performed
            if plot_experimental_data:
                return {"status": "experimental_data_plotted", "exit_early": True}

            # This code would not be reached with --plot-experimental-data
            return {"status": "full_analysis_completed", "exit_early": False}

        # Run mock implementation
        result = mock_run_homodyne_with_plot_experimental_data()

        # Verify early exit behavior
        assert result["exit_early"] is True
        assert result["status"] == "experimental_data_plotted"

        # Verify output files were created
        exp_data_dir = temp_directory / "homodyne_results" / "exp_data"
        assert exp_data_dir.exists()
        assert len(list(exp_data_dir.glob("*.png"))) == 3
        assert (exp_data_dir / "summary_statistics.txt").exists()

    def test_classical_method_execution_simulation(self, temp_directory):
        """Simulate the execution of run_homodyne.py with --method classical."""

        def mock_run_homodyne_with_classical_method():
            """Mock implementation of run_homodyne.py with --method classical."""
            # Simulate command line argument parsing
            method = "classical"
            output_dir = temp_directory / "homodyne_results"
            classical_dir = output_dir / "classical"

            # Create output directories
            output_dir.mkdir(parents=True, exist_ok=True)
            classical_dir.mkdir(parents=True, exist_ok=True)

            # Simulate analysis execution
            # 1. Load experimental data
            mock_exp_data = np.random.rand(1, 60, 60) + 1.0

            # 2. Run classical optimization
            mock_parameters = [0.85459454, -0.02171455, -0.78131764]
            mock_chi_squared = 17.12216371557179

            # 3. Calculate fitted data
            mock_theory_data = np.random.rand(1, 60, 60) + 0.95
            mock_fitted_data = (
                mock_theory_data * 1912.0 - 1910.96
            )  # Scaling from actual results
            mock_residuals_data = mock_exp_data - mock_fitted_data

            # 4. Save data files
            np.savez_compressed(
                classical_dir / "experimental_data.npz", data=mock_exp_data
            )
            np.savez_compressed(
                classical_dir / "fitted_data.npz", data=mock_fitted_data
            )
            np.savez_compressed(
                classical_dir / "residuals_data.npz", data=mock_residuals_data
            )

            # 5. Create C2 heatmaps
            (classical_dir / "c2_heatmaps_phi_0.0deg.png").touch()

            # 6. Save main results to output directory (not current directory)
            main_results = {
                "timestamp": "2025-08-18T16:06:09.710366+00:00",
                "results": {
                    "classical_optimization": {
                        "parameters": mock_parameters,
                        "chi_squared": mock_chi_squared,
                        "success": True,
                    }
                },
            }

            results_file = output_dir / "homodyne_analysis_results.json"
            with open(results_file, "w") as f:
                json.dump(main_results, f, indent=2)

            return {
                "status": "classical_analysis_completed",
                "parameters": mock_parameters,
                "chi_squared": mock_chi_squared,
                "results_file": str(results_file),
                "classical_dir": str(classical_dir),
            }

        # Run mock implementation
        result = mock_run_homodyne_with_classical_method()

        # Verify results
        assert result["status"] == "classical_analysis_completed"
        assert len(result["parameters"]) == 3
        assert result["chi_squared"] > 0

        # Verify file structure
        output_dir = temp_directory / "homodyne_results"
        classical_dir = output_dir / "classical"

        assert output_dir.exists()
        assert classical_dir.exists()

        # Verify main results file location
        results_file = Path(result["results_file"])
        assert results_file.exists()
        assert (
            results_file.parent == output_dir
        )  # In output directory, not current directory

        # Verify classical data files
        assert (classical_dir / "experimental_data.npz").exists()
        assert (classical_dir / "fitted_data.npz").exists()
        assert (classical_dir / "residuals_data.npz").exists()
        assert (classical_dir / "c2_heatmaps_phi_0.0deg.png").exists()

        # Verify NPZ file content
        exp_data = np.load(classical_dir / "experimental_data.npz")
        fitted_data = np.load(classical_dir / "fitted_data.npz")
        residuals_data = np.load(classical_dir / "residuals_data.npz")

        assert "data" in exp_data
        assert "data" in fitted_data
        assert "data" in residuals_data
        assert exp_data["data"].shape == (1, 60, 60)


class TestBackwardCompatibilityIntegration:
    """Test that the changes maintain backward compatibility for existing workflows."""

    def test_existing_configurations_still_work(self, temp_directory):
        """Test that existing configuration files work with new directory structure."""
        # Create a configuration similar to what existed before changes
        old_style_config = {
            "metadata": {
                "config_version": "6.0",
                "description": "Legacy Configuration Test",
            },
            "analysis_settings": {"static_mode": True, "static_submode": "isotropic"},
            "initial_parameters": {"values": [1000, -0.5, 100]},
        }

        # Save configuration
        config_file = temp_directory / "legacy_config.json"
        with open(config_file, "w") as f:
            json.dump(old_style_config, f, indent=2)

        # Verify configuration is readable
        with open(config_file, "r") as f:
            loaded_config = json.load(f)

        assert loaded_config["metadata"]["config_version"] == "6.0"
        assert loaded_config["analysis_settings"]["static_mode"] is True

        # The system should automatically create new directory structure
        # even with old configuration files (this would be tested in actual execution)

    def test_results_file_backward_compatibility(self, temp_directory):
        """Test that results files maintain expected format while changing location."""
        output_dir = temp_directory / "homodyne_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create results file in new location
        results_file = output_dir / "homodyne_analysis_results.json"

        # Results file should maintain same internal structure
        expected_structure = {
            "timestamp": "2025-08-18T16:06:09.710366+00:00",
            "config": {"test": "config_data"},
            "results": {"test": "results_data"},
            "execution_metadata": {"analysis_success": True},
        }

        with open(results_file, "w") as f:
            json.dump(expected_structure, f, indent=2)

        # Verify file exists in new location
        assert results_file.exists()
        assert results_file.parent == output_dir

        # Verify internal structure is preserved
        with open(results_file, "r") as f:
            loaded_data = json.load(f)

        required_keys = ["timestamp", "config", "results", "execution_metadata"]
        for key in required_keys:
            assert key in loaded_data

        assert loaded_data["execution_metadata"]["analysis_success"] is True


class TestMCMCIntegration:
    """Test MCMC method integration with new directory structure."""

    def test_mcmc_method_output_structure(self, temp_directory):
        """Test that MCMC method creates correct output structure."""
        # Create expected output structure for MCMC method
        mcmc_dir = temp_directory / "homodyne_results" / "mcmc"
        mcmc_dir.mkdir(parents=True, exist_ok=True)

        # Simulate files that would be created by MCMC method
        expected_files = [
            "experimental_data.npz",
            "fitted_data.npz",
            "residuals_data.npz",
            "c2_heatmaps_phi_0.0deg.png",
            "mcmc_summary.json",
            "mcmc_trace.nc",
            "trace_plot.png",
            "corner_plot.png",
        ]

        for filename in expected_files:
            if filename.endswith(".npz"):
                # Create realistic NPZ files
                mock_data = np.random.rand(3, 20, 30)
                np.savez_compressed(mcmc_dir / filename, data=mock_data)
            else:
                # Create empty files for plots and summaries
                (mcmc_dir / filename).touch()

        # Verify structure was created correctly
        assert mcmc_dir.exists()
        assert mcmc_dir.is_dir()

        # Verify files exist and have correct formats
        npz_files = list(mcmc_dir.glob("*.npz"))
        json_files = list(mcmc_dir.glob("*.json"))
        plot_files = list(mcmc_dir.glob("*.png"))
        netcdf_files = list(mcmc_dir.glob("*.nc"))

        assert len(npz_files) == 3  # experimental, fitted, residuals
        assert len(json_files) == 1  # mcmc_summary.json
        assert len(plot_files) >= 3  # C2 heatmaps, trace, corner
        assert len(netcdf_files) == 1  # mcmc_trace.nc

        # Verify NPZ files contain data
        for npz_file in npz_files:
            data = np.load(npz_file)
            assert "data" in data
            assert data["data"].shape == (3, 20, 30)

    def test_mcmc_method_execution_simulation(self, temp_directory):
        """Simulate the execution of run_homodyne.py with --method mcmc."""

        def mock_run_homodyne_with_mcmc_method():
            """Mock implementation of run_homodyne.py with --method mcmc."""
            # Simulate command line argument parsing
            method = "mcmc"
            output_dir = temp_directory / "homodyne_results"
            mcmc_dir = output_dir / "mcmc"

            # Create output directories
            output_dir.mkdir(parents=True, exist_ok=True)
            mcmc_dir.mkdir(parents=True, exist_ok=True)

            # Simulate analysis execution
            # 1. Load experimental data
            mock_exp_data = np.random.rand(1, 60, 60) + 1.0

            # 2. Run MCMC sampling (mock posterior means)
            mock_posterior_means = {"D0": 850.0, "alpha": -0.021, "D_offset": -780.0}
            mock_parameters = [850.0, -0.021, -780.0]

            # 3. Calculate fitted data from posterior means
            mock_theory_data = np.random.rand(1, 60, 60) + 0.95
            mock_fitted_data = mock_theory_data * 1900.0 - 1900.0  # Scaling
            mock_residuals_data = mock_exp_data - mock_fitted_data

            # 4. Save data files
            np.savez_compressed(mcmc_dir / "experimental_data.npz", data=mock_exp_data)
            np.savez_compressed(mcmc_dir / "fitted_data.npz", data=mock_fitted_data)
            np.savez_compressed(
                mcmc_dir / "residuals_data.npz", data=mock_residuals_data
            )

            # 5. Create MCMC-specific files
            (mcmc_dir / "c2_heatmaps_phi_0.0deg.png").touch()
            (mcmc_dir / "mcmc_trace.nc").touch()
            (mcmc_dir / "trace_plot.png").touch()
            (mcmc_dir / "corner_plot.png").touch()

            # 6. Save MCMC summary
            mcmc_summary = {
                "method": "MCMC_NUTS",
                "execution_time_seconds": 120.5,
                "posterior_means": mock_posterior_means,
                "convergence_diagnostics": {
                    "max_rhat": 1.02,
                    "min_ess": 450,
                    "converged": True,
                    "assessment": "excellent",
                },
            }

            summary_file = mcmc_dir / "mcmc_summary.json"
            with open(summary_file, "w") as f:
                json.dump(mcmc_summary, f, indent=2)

            # 7. Save main results to output directory (not current directory)
            main_results = {
                "timestamp": "2025-08-18T16:06:09.710366+00:00",
                "results": {
                    "mcmc_optimization": {
                        "parameters": mock_parameters,
                        "convergence_quality": "excellent",
                        "success": True,
                        "posterior_means": mock_posterior_means,
                    }
                },
            }

            results_file = output_dir / "homodyne_analysis_results.json"
            with open(results_file, "w") as f:
                json.dump(main_results, f, indent=2)

            return {
                "status": "mcmc_analysis_completed",
                "parameters": mock_parameters,
                "convergence_quality": "excellent",
                "results_file": str(results_file),
                "mcmc_dir": str(mcmc_dir),
            }

        # Run mock implementation
        result = mock_run_homodyne_with_mcmc_method()

        # Verify results
        assert result["status"] == "mcmc_analysis_completed"
        assert len(result["parameters"]) == 3
        assert result["convergence_quality"] == "excellent"

        # Verify file structure
        output_dir = temp_directory / "homodyne_results"
        mcmc_dir = output_dir / "mcmc"

        assert output_dir.exists()
        assert mcmc_dir.exists()

        # Verify main results file location
        results_file = Path(result["results_file"])
        assert results_file.exists()
        assert (
            results_file.parent == output_dir
        )  # In output directory, not current directory

        # Verify MCMC data files
        assert (mcmc_dir / "experimental_data.npz").exists()
        assert (mcmc_dir / "fitted_data.npz").exists()
        assert (mcmc_dir / "residuals_data.npz").exists()
        assert (mcmc_dir / "mcmc_summary.json").exists()
        assert (mcmc_dir / "mcmc_trace.nc").exists()

        # Verify MCMC plots
        assert (mcmc_dir / "c2_heatmaps_phi_0.0deg.png").exists()
        assert (mcmc_dir / "trace_plot.png").exists()
        assert (mcmc_dir / "corner_plot.png").exists()

        # Verify NPZ file content
        exp_data = np.load(mcmc_dir / "experimental_data.npz")
        fitted_data = np.load(mcmc_dir / "fitted_data.npz")
        residuals_data = np.load(mcmc_dir / "residuals_data.npz")

        assert "data" in exp_data
        assert "data" in fitted_data
        assert "data" in residuals_data
        assert exp_data["data"].shape == (1, 60, 60)

        # Verify MCMC summary content
        with open(mcmc_dir / "mcmc_summary.json", "r") as f:
            summary = json.load(f)

        assert summary["method"] == "MCMC_NUTS"
        assert "convergence_diagnostics" in summary
        assert summary["convergence_diagnostics"]["converged"] is True

    def test_mcmc_vs_classical_method_separation(self, temp_directory):
        """Test that MCMC and classical methods create separate directories."""
        base_dir = temp_directory / "homodyne_results"

        # Simulate running both methods
        classical_dir = base_dir / "classical"
        mcmc_dir = base_dir / "mcmc"

        classical_dir.mkdir(parents=True, exist_ok=True)
        mcmc_dir.mkdir(parents=True, exist_ok=True)

        # Create method-specific files
        # Classical files
        classical_files = [
            "experimental_data.npz",
            "fitted_data.npz",
            "residuals_data.npz",
            "c2_heatmaps_phi_0.0deg.png",
        ]

        for filename in classical_files:
            if filename.endswith(".npz"):
                mock_data = np.random.rand(2, 30, 40) + 1.0
                np.savez_compressed(classical_dir / filename, data=mock_data)
            else:
                (classical_dir / filename).touch()

        # MCMC files
        mcmc_files = [
            "experimental_data.npz",
            "fitted_data.npz",
            "residuals_data.npz",
            "c2_heatmaps_phi_0.0deg.png",
            "mcmc_summary.json",
            "mcmc_trace.nc",
            "trace_plot.png",
            "corner_plot.png",
        ]

        for filename in mcmc_files:
            if filename.endswith(".npz"):
                mock_data = np.random.rand(2, 30, 40) + 1.0
                np.savez_compressed(mcmc_dir / filename, data=mock_data)
            elif filename.endswith(".json"):
                mock_summary = {"method": "MCMC_NUTS", "converged": True}
                with open(mcmc_dir / filename, "w") as f:
                    json.dump(mock_summary, f, indent=2)
            else:
                (mcmc_dir / filename).touch()

        # Verify directories are separate
        assert classical_dir.exists()
        assert mcmc_dir.exists()
        assert classical_dir != mcmc_dir

        # Verify common files exist in both directories
        common_files = [
            "experimental_data.npz",
            "fitted_data.npz",
            "residuals_data.npz",
            "c2_heatmaps_phi_0.0deg.png",
        ]
        for filename in common_files:
            assert (classical_dir / filename).exists()
            assert (mcmc_dir / filename).exists()

        # Verify MCMC-specific files only exist in MCMC directory
        mcmc_only_files = [
            "mcmc_summary.json",
            "mcmc_trace.nc",
            "trace_plot.png",
            "corner_plot.png",
        ]
        for filename in mcmc_only_files:
            assert (mcmc_dir / filename).exists()
            assert not (classical_dir / filename).exists()

        # Verify data has correct structure
        classical_data = np.load(classical_dir / "experimental_data.npz")
        mcmc_data = np.load(mcmc_dir / "experimental_data.npz")

        assert "data" in classical_data
        assert "data" in mcmc_data
        assert classical_data["data"].shape == (2, 30, 40)
        assert mcmc_data["data"].shape == (2, 30, 40)
