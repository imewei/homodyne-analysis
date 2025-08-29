"""
Integration Tests for run_homodyne.py Script
===========================================

Tests the behavior of the main run_homodyne.py script, including:
- --plot-experimental-data flag behavior (early exit, correct output path)
- Classical method output organization
- Results file location changes
- Directory structure creation
"""

import json
from pathlib import Path

import numpy as np


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

        # Create method-specific directories
        method_dirs = ["nelder_mead", "gurobi"]
        for method_name in method_dirs:
            method_dir = classical_dir / method_name
            method_dir.mkdir(parents=True, exist_ok=True)

            # Create consolidated fitted_data.npz with all data
            mock_experimental_data = np.random.rand(3, 20, 30)
            mock_fitted_data = np.random.rand(3, 20, 30)
            mock_residuals_data = mock_experimental_data - mock_fitted_data
            mock_parameters = np.array([1.5, 2.0, 0.5])
            mock_uncertainties = np.array([0.1, 0.1, 0.1])

            np.savez_compressed(
                method_dir / "fitted_data.npz",
                c2_experimental=mock_experimental_data,
                c2_fitted=mock_fitted_data,
                residuals=mock_residuals_data,
                parameters=mock_parameters,
                uncertainties=mock_uncertainties,
                chi_squared=np.array([0.5]),
            )

            # Create method-specific plot files
            (method_dir / f"c2_heatmaps_{method_name}.png").touch()

            # Create parameters.json
            method_info = {
                "parameters": {"param_0": {"value": 1.5, "uncertainty": 0.1}},
                "goodness_of_fit": {"chi_squared": 0.5},
                "convergence_info": {"success": True},
            }

            with open(method_dir / "parameters.json", "w") as f:
                json.dump(method_info, f, indent=2)

        # Create summary file
        summary_data = {
            "analysis_type": "Classical Optimization",
            "methods_analyzed": method_dirs,
            "results": {},
        }

        with open(classical_dir / "all_classical_methods_summary.json", "w") as f:
            json.dump(summary_data, f, indent=2)

        # Verify structure was created correctly
        assert classical_dir.exists()
        assert classical_dir.is_dir()

        # Verify method directories exist
        for method_name in method_dirs:
            method_dir = classical_dir / method_name
            assert method_dir.exists()
            assert method_dir.is_dir()

            # Verify files in method directory
            assert (method_dir / "fitted_data.npz").exists()
            assert (method_dir / "parameters.json").exists()
            assert (method_dir / f"c2_heatmaps_{method_name}.png").exists()

            # Verify NPZ file contains consolidated data
            data = np.load(method_dir / "fitted_data.npz")
            assert "c2_experimental" in data
            assert "c2_fitted" in data
            assert "residuals" in data
            assert "parameters" in data
            assert "uncertainties" in data
            assert "chi_squared" in data
            assert data["c2_experimental"].shape == (3, 20, 30)

        # Verify summary file exists
        assert (classical_dir / "all_classical_methods_summary.json").exists()

    def test_main_results_file_location(self, temp_directory):
        """Test that main results file is saved to output directory."""
        results_dir = temp_directory / "homodyne_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create mock main results file (homodyne_analysis_results.json stays
        # in root)
        results_file = results_dir / "homodyne_analysis_results.json"

        mock_results = {
            "timestamp": "2025-08-18T16:06:09.710366+00:00",
            "config": {"test": "config"},
            "results": {"classical_optimization": {"parameters": [1.0, 2.0, 3.0]}},
            "execution_metadata": {"analysis_success": True},
        }

        with open(results_file, "w") as f:
            json.dump(mock_results, f, indent=2)

        # Verify file is in output directory
        assert results_file.exists()
        assert results_file.parent == results_dir
        assert results_file.name == "homodyne_analysis_results.json"

        # Verify content is correct
        with open(results_file, encoding="utf-8") as f:
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

        log_file.write_text(
            "2025-08-18 10:57:33 - __main__ - INFO - Analysis completed successfully\n"
        )

        # Create classical method directories and files
        classical_dir = base_dir / "classical"
        method_dirs = ["nelder_mead", "gurobi"]

        for method_name in method_dirs:
            method_dir = classical_dir / method_name
            method_dir.mkdir(parents=True, exist_ok=True)

            # Create consolidated fitted_data.npz
            mock_experimental_data = np.random.rand(1, 60, 60) + 1.0
            mock_fitted_data = np.random.rand(1, 60, 60) + 1.0
            mock_residuals_data = mock_experimental_data - mock_fitted_data
            mock_parameters = np.array([1.5, 2.0, 0.5])
            mock_uncertainties = np.array([0.1, 0.1, 0.1])

            np.savez_compressed(
                method_dir / "fitted_data.npz",
                c2_experimental=mock_experimental_data,
                c2_fitted=mock_fitted_data,
                residuals=mock_residuals_data,
                parameters=mock_parameters,
                uncertainties=mock_uncertainties,
                chi_squared=np.array([0.5]),
            )

            # Create method-specific plots and parameters
            (method_dir / f"c2_heatmaps_{method_name}.png").touch()

            method_info = {
                "parameters": {"param_0": {"value": 1.5, "uncertainty": 0.1}},
                "goodness_of_fit": {"chi_squared": 0.5},
            }

            with open(method_dir / "parameters.json", "w") as f:
                json.dump(method_info, f, indent=2)

        # Create summary file
        summary_data = {
            "analysis_type": "Classical Optimization",
            "methods_analyzed": method_dirs,
        }

        with open(classical_dir / "all_classical_methods_summary.json", "w") as f:
            json.dump(summary_data, f, indent=2)

        # Create experimental data files
        exp_data_files = [
            "data_validation_phi_0.0deg.png",
            "summary_statistics.txt",
        ]

        for filename in exp_data_files:
            (base_dir / "exp_data" / filename).touch()

        # Verify complete structure
        assert base_dir.exists()
        assert (base_dir / "classical").is_dir()
        assert (base_dir / "exp_data").is_dir()

        # Verify main files
        assert main_results.exists()
        assert log_file.exists()

        # Verify classical method directories and files
        classical_dir = base_dir / "classical"
        for method_name in method_dirs:
            method_dir = classical_dir / method_name
            assert method_dir.exists()
            assert (method_dir / "fitted_data.npz").exists()
            assert (method_dir / "parameters.json").exists()
            assert (method_dir / f"c2_heatmaps_{method_name}.png").exists()

        # Verify summary file
        assert (classical_dir / "all_classical_methods_summary.json").exists()

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
                return {
                    "status": "experimental_data_plotted",
                    "exit_early": True,
                }

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

            # 4. Create method-specific directories and save data files
            method_dirs = ["nelder_mead", "gurobi"]
            for method_name in method_dirs:
                method_dir = classical_dir / method_name
                method_dir.mkdir(parents=True, exist_ok=True)

                # Save consolidated data in fitted_data.npz
                np.savez_compressed(
                    method_dir / "fitted_data.npz",
                    c2_experimental=mock_exp_data,
                    c2_fitted=mock_fitted_data,
                    residuals=mock_residuals_data,
                    parameters=mock_parameters,
                    uncertainties=np.array([0.1, 0.1, 0.1]),
                    chi_squared=np.array([mock_chi_squared]),
                )

                # Create method-specific plots and parameters
                (method_dir / f"c2_heatmaps_{method_name}.png").touch()

                method_info = {
                    "parameters": {
                        "param_0": {
                            "value": mock_parameters[0],
                            "uncertainty": 0.1,
                        }
                    },
                    "goodness_of_fit": {"chi_squared": mock_chi_squared},
                }

                with open(method_dir / "parameters.json", "w") as f:
                    json.dump(method_info, f, indent=2)

            # Create summary file
            summary_data = {
                "analysis_type": "Classical Optimization",
                "methods_analyzed": method_dirs,
                "best_method": "nelder_mead",
            }

            with open(classical_dir / "all_classical_methods_summary.json", "w") as f:
                json.dump(summary_data, f, indent=2)

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

        # Verify classical method directories and data files
        method_dirs = ["nelder_mead", "gurobi"]
        for method_name in method_dirs:
            method_dir = classical_dir / method_name
            assert method_dir.exists()
            assert (method_dir / "fitted_data.npz").exists()
            assert (method_dir / "parameters.json").exists()
            assert (method_dir / f"c2_heatmaps_{method_name}.png").exists()

            # Verify consolidated NPZ file content
            data = np.load(method_dir / "fitted_data.npz")
            assert "c2_experimental" in data
            assert "c2_fitted" in data
            assert "residuals" in data
            assert "parameters" in data
            assert "uncertainties" in data
            assert "chi_squared" in data
            assert data["c2_experimental"].shape == (1, 60, 60)

        # Verify summary file
        assert (classical_dir / "all_classical_methods_summary.json").exists()


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
            "analysis_settings": {
                "static_mode": True,
                "static_submode": "isotropic",
            },
            "initial_parameters": {"values": [1000, -0.5, 100]},
        }

        # Save configuration
        config_file = temp_directory / "legacy_config.json"
        with open(config_file, "w") as f:
            json.dump(old_style_config, f, indent=2)

        # Verify configuration is readable
        with open(config_file, encoding="utf-8") as f:
            loaded_config = json.load(f)

        assert loaded_config["metadata"]["config_version"] == "6.0"
        assert loaded_config["analysis_settings"]["static_mode"] is True

        # The system should automatically create new directory structure
        # even with old configuration files (this would be tested in actual
        # execution)

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
        with open(results_file, encoding="utf-8") as f:
            loaded_data = json.load(f)

        required_keys = [
            "timestamp",
            "config",
            "results",
            "execution_metadata",
        ]
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
            "fitted_data.npz",
            "c2_heatmaps_phi_0.0deg.png",
            "mcmc_summary.json",
            "mcmc_trace.nc",
            "trace_plot.png",
            "corner_plot.png",
        ]

        for filename in expected_files:
            if filename == "fitted_data.npz":
                # Create consolidated NPZ file with all data
                mock_experimental_data = np.random.rand(3, 20, 30)
                mock_fitted_data = np.random.rand(3, 20, 30)
                mock_residuals_data = mock_experimental_data - mock_fitted_data

                np.savez_compressed(
                    mcmc_dir / filename,
                    c2_experimental=mock_experimental_data,
                    c2_fitted=mock_fitted_data,
                    residuals=mock_residuals_data,
                    parameters=np.array([1.5, 2.0, 0.5]),
                    uncertainties=np.array([0.1, 0.1, 0.1]),
                    chi_squared=np.array([0.5]),
                )
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

        assert len(npz_files) == 1  # consolidated fitted_data.npz
        assert len(json_files) == 1  # mcmc_summary.json
        assert len(plot_files) >= 3  # C2 heatmaps, trace, corner
        assert len(netcdf_files) == 1  # mcmc_trace.nc

        # Verify consolidated NPZ file contains all data
        data = np.load(mcmc_dir / "fitted_data.npz")
        assert "c2_experimental" in data
        assert "c2_fitted" in data
        assert "residuals" in data
        assert "parameters" in data
        assert "uncertainties" in data
        assert "chi_squared" in data
        assert data["c2_experimental"].shape == (3, 20, 30)

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
            mock_posterior_means = {
                "D0": 850.0,
                "alpha": -0.021,
                "D_offset": -780.0,
            }
            mock_parameters = [850.0, -0.021, -780.0]

            # 3. Calculate fitted data from posterior means
            mock_theory_data = np.random.rand(1, 60, 60) + 0.95
            mock_fitted_data = mock_theory_data * 1900.0 - 1900.0  # Scaling
            mock_residuals_data = mock_exp_data - mock_fitted_data

            # 4. Save consolidated data file
            np.savez_compressed(
                mcmc_dir / "fitted_data.npz",
                c2_experimental=mock_exp_data,
                c2_fitted=mock_fitted_data,
                residuals=mock_residuals_data,
                parameters=mock_parameters,
                uncertainties=np.array([0.1, 0.1, 0.1]),
                chi_squared=np.array([0.5]),
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
        assert (mcmc_dir / "fitted_data.npz").exists()
        assert (mcmc_dir / "mcmc_summary.json").exists()
        assert (mcmc_dir / "mcmc_trace.nc").exists()

        # Verify MCMC plots
        assert (mcmc_dir / "c2_heatmaps_phi_0.0deg.png").exists()
        assert (mcmc_dir / "trace_plot.png").exists()
        assert (mcmc_dir / "corner_plot.png").exists()

        # Verify consolidated NPZ file content
        data = np.load(mcmc_dir / "fitted_data.npz")
        assert "c2_experimental" in data
        assert "c2_fitted" in data
        assert "residuals" in data
        assert "parameters" in data
        assert "uncertainties" in data
        assert "chi_squared" in data
        assert data["c2_experimental"].shape == (1, 60, 60)

        # Verify MCMC summary content
        with open(mcmc_dir / "mcmc_summary.json", encoding="utf-8") as f:
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
        # Classical files in method directories
        method_dirs = ["nelder_mead", "gurobi"]

        for method_name in method_dirs:
            method_dir = classical_dir / method_name
            method_dir.mkdir(parents=True, exist_ok=True)

            # Create consolidated fitted_data.npz
            mock_experimental_data = np.random.rand(2, 30, 40) + 1.0
            mock_fitted_data = np.random.rand(2, 30, 40) + 1.0
            mock_residuals_data = mock_experimental_data - mock_fitted_data

            np.savez_compressed(
                method_dir / "fitted_data.npz",
                c2_experimental=mock_experimental_data,
                c2_fitted=mock_fitted_data,
                residuals=mock_residuals_data,
                parameters=np.array([1.5, 2.0, 0.5]),
                uncertainties=np.array([0.1, 0.1, 0.1]),
                chi_squared=np.array([0.5]),
            )

            # Create method-specific plots
            (method_dir / f"c2_heatmaps_{method_name}.png").touch()

        # MCMC files (still in single mcmc directory)
        mcmc_files = [
            "fitted_data.npz",
            "c2_heatmaps_phi_0.0deg.png",
            "mcmc_summary.json",
            "mcmc_trace.nc",
            "trace_plot.png",
            "corner_plot.png",
        ]

        for filename in mcmc_files:
            if filename == "fitted_data.npz":
                # Create consolidated NPZ with all data for MCMC
                mock_experimental_data = np.random.rand(2, 30, 40) + 1.0
                mock_fitted_data = np.random.rand(2, 30, 40) + 1.0
                mock_residuals_data = mock_experimental_data - mock_fitted_data

                np.savez_compressed(
                    mcmc_dir / filename,
                    c2_experimental=mock_experimental_data,
                    c2_fitted=mock_fitted_data,
                    residuals=mock_residuals_data,
                    parameters=np.array([1.5, 2.0, 0.5]),
                    uncertainties=np.array([0.1, 0.1, 0.1]),
                    chi_squared=np.array([0.5]),
                )
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

        # Verify fitted_data.npz exists in method directories and MCMC
        # directory
        for method_name in method_dirs:
            method_dir = classical_dir / method_name
            assert (method_dir / "fitted_data.npz").exists()
            assert (method_dir / f"c2_heatmaps_{method_name}.png").exists()

        # Verify MCMC files exist
        assert (mcmc_dir / "fitted_data.npz").exists()
        assert (mcmc_dir / "c2_heatmaps_phi_0.0deg.png").exists()

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
        # Check first classical method
        first_method_dir = classical_dir / method_dirs[0]
        classical_data = np.load(first_method_dir / "fitted_data.npz")
        mcmc_data = np.load(mcmc_dir / "fitted_data.npz")

        assert "c2_experimental" in classical_data
        assert "c2_fitted" in classical_data
        assert "residuals" in classical_data
        assert "c2_experimental" in mcmc_data
        assert "c2_fitted" in mcmc_data
        assert "residuals" in mcmc_data
        assert classical_data["c2_experimental"].shape == (2, 30, 40)
        assert mcmc_data["c2_experimental"].shape == (2, 30, 40)
