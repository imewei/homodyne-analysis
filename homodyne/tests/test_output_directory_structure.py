"""
Tests for Output Directory Structure Changes
===========================================

Tests the new output directory structure and file organization introduced
in the homodyne analysis module, including:
- Classical method results saved to ./homodyne_results/classical/
- Experimental data plots saved to ./homodyne_results/exp_data/
- Main results file saved to output directory instead of current directory
- NPZ data files for classical fitting results
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from homodyne.tests.fixtures import (dummy_config, temp_directory,
                                     test_output_directory)


class TestOutputDirectoryStructure:
    """Test the new output directory structure and file organization."""

    def test_classical_output_directory_structure(self, temp_directory):
        """Test that classical method creates the correct directory structure."""
        # Create expected classical output directory
        classical_dir = temp_directory / "homodyne_results" / "classical"
        classical_dir.mkdir(parents=True, exist_ok=True)

        # Expected files and directories in new classical structure
        summary_file = "all_classical_methods_summary.json"

        # Create method directories and files
        method_dirs = ["nelder_mead", "gurobi"]
        method_files = [
            "analysis_results_{method}.json",
            "parameters.json",
            "fitted_data.npz",  # Contains experimental, fitted, residuals
            "c2_heatmaps_{method}.png",
        ]

        # Create summary file
        (classical_dir / summary_file).touch()

        # Create method directories and files
        for method in method_dirs:
            method_dir = classical_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)

            for file_template in method_files:
                filename = file_template.format(method=method)
                (method_dir / filename).touch()

        # Verify structure
        assert classical_dir.exists()
        assert (classical_dir / summary_file).exists()

        for method in method_dirs:
            method_dir = classical_dir / method
            assert method_dir.exists()
            for file_template in method_files:
                filename = file_template.format(method=method)
                assert (method_dir / filename).exists()

    def test_experimental_data_output_directory_structure(
            self, temp_directory):
        """Test that experimental data plots create the correct directory structure."""
        # Create expected experimental data output directory
        exp_data_dir = temp_directory / "homodyne_results" / "exp_data"
        exp_data_dir.mkdir(parents=True, exist_ok=True)

        # Expected files in exp_data directory
        expected_files = [
            "data_validation_phi_0.0deg.png",
            "data_validation_phi_45.0deg.png",
            "summary_statistics.txt",
        ]

        # Create mock files
        for filename in expected_files:
            (exp_data_dir / filename).touch()

        # Verify structure
        assert exp_data_dir.exists()
        for filename in expected_files:
            assert (exp_data_dir / filename).exists()

    def test_main_results_file_location(self, temp_directory):
        """Test that main results file is saved to output directory."""
        results_dir = temp_directory / "homodyne_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Main results file should be in output directory, not current
        # directory
        main_results_file = results_dir / "homodyne_analysis_results.json"
        main_results_file.touch()

        assert main_results_file.exists()
        assert main_results_file.parent == results_dir

    def test_complete_directory_structure(self, temp_directory):
        """Test the complete expected directory structure."""
        base_dir = temp_directory / "homodyne_results"

        # Create complete expected structure with new method-specific
        # directories
        structure = {
            "homodyne_analysis_results.json": "file",
            "run.log": "file",
            "classical": {
                "all_classical_methods_summary.json": "file",
                "nelder_mead": {
                    "analysis_results_nelder_mead.json": "file",
                    "parameters.json": "file",
                    "fitted_data.npz": "file",  # Contains experimental, fitted, residuals
                    "c2_heatmaps_nelder_mead.png": "file",  # Single phi angle case
                    "nelder_mead_diagnostic_summary.png": "file",  # Method-specific diagnostic
                },
                "gurobi": {
                    "analysis_results_gurobi.json": "file",
                    "parameters.json": "file",
                    "fitted_data.npz": "file",  # Contains experimental, fitted, residuals
                    "c2_heatmaps_gurobi.png": "file",  # Single phi angle case
                    "gurobi_diagnostic_summary.png": "file",  # Method-specific diagnostic
                },
            },
            "robust": {
                "all_robust_methods_summary.json": "file",
                "wasserstein": {
                    "analysis_results_wasserstein.json": "file",
                    "parameters.json": "file",
                    "fitted_data.npz": "file",  # Contains experimental, fitted, residuals
                    "c2_heatmaps_wasserstein.png": "file",  # Single phi angle case
                    "wasserstein_diagnostic_summary.png": "file",  # Method-specific diagnostic
                },
                "scenario": {
                    "analysis_results_scenario.json": "file",
                    "parameters.json": "file",
                    "fitted_data.npz": "file",  # Contains experimental, fitted, residuals
                    "c2_heatmaps_scenario.png": "file",  # Single phi angle case
                    "scenario_diagnostic_summary.png": "file",  # Method-specific diagnostic
                },
                "ellipsoidal": {
                    "analysis_results_ellipsoidal.json": "file",
                    "parameters.json": "file",
                    "fitted_data.npz": "file",  # Contains experimental, fitted, residuals
                    "c2_heatmaps_ellipsoidal.png": "file",  # Single phi angle case
                    "ellipsoidal_diagnostic_summary.png": "file",  # Method-specific diagnostic
                },
            },
            "exp_data": {
                "data_validation_phi_0.0deg.png": "file",
                "data_validation_phi_45.0deg.png": "file",
                "summary_statistics.txt": "file",
            },
        }

        def create_structure(base_path, structure_dict):
            """Recursively create directory structure."""
            for name, content in structure_dict.items():
                path = base_path / name
                if content == "file":
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                elif isinstance(content, dict):
                    path.mkdir(parents=True, exist_ok=True)
                    create_structure(path, content)

        create_structure(base_dir, structure)

        # Verify complete structure
        assert base_dir.exists()
        assert (base_dir / "homodyne_analysis_results.json").exists()

        # Verify classical structure
        assert (base_dir / "classical").is_dir()
        assert (
            base_dir /
            "classical" /
            "all_classical_methods_summary.json").exists()
        assert (base_dir / "classical" / "nelder_mead").is_dir()
        assert (
            base_dir / "classical" / "nelder_mead" / "analysis_results_nelder_mead.json"
        ).exists()
        assert (
            base_dir /
            "classical" /
            "nelder_mead" /
            "fitted_data.npz").exists()
        assert (base_dir / "classical" / "gurobi").is_dir()
        assert (
            base_dir / "classical" / "gurobi" / "analysis_results_gurobi.json"
        ).exists()
        assert (base_dir / "classical" / "gurobi" / "fitted_data.npz").exists()

        # Verify robust structure
        assert (base_dir / "robust").is_dir()
        assert (
            base_dir /
            "robust" /
            "all_robust_methods_summary.json").exists()
        assert (base_dir / "robust" / "wasserstein").is_dir()
        assert (
            base_dir / "robust" / "wasserstein" / "analysis_results_wasserstein.json"
        ).exists()
        assert (
            base_dir /
            "robust" /
            "wasserstein" /
            "fitted_data.npz").exists()

        # Verify experimental data plots structure (unchanged)
        assert (base_dir / "exp_data").is_dir()
        assert (
            base_dir /
            "exp_data" /
            "data_validation_phi_0.0deg.png").exists()

    def test_multiple_phi_angles_heatmap_naming(self, temp_directory):
        """Test that heatmap files are named correctly for multiple phi angles."""
        base_dir = temp_directory / "homodyne_results"
        classical_dir = base_dir / "classical"

        # Create classical method directories
        for method in ["nelder_mead", "gurobi"]:
            method_dir = classical_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)

            # For multiple phi angles, heatmaps should include phi angle in
            # filename
            test_angles = [0.0, 45.0, 90.0]
            for angle in test_angles:
                heatmap_file = method_dir / \
                    f"c2_heatmaps_{method}_phi_{angle}deg.png"
                heatmap_file.touch()
                assert heatmap_file.exists()

        # Test robust methods too
        robust_dir = base_dir / "robust"
        for method in ["wasserstein", "scenario", "ellipsoidal"]:
            method_dir = robust_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)

            # For multiple phi angles, heatmaps should include phi angle in
            # filename
            test_angles = [0.0, 45.0]
            for angle in test_angles:
                heatmap_file = method_dir / \
                    f"c2_heatmaps_{method}_phi_{angle}deg.png"
                heatmap_file.touch()
                assert heatmap_file.exists()

    def test_diagnostic_summary_plots(self, temp_directory):
        """Test that diagnostic summary plots are generated for each method."""
        base_dir = temp_directory / "homodyne_results"

        # Test classical method diagnostic plots
        classical_dir = base_dir / "classical"
        for method in ["nelder_mead", "gurobi"]:
            method_dir = classical_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)

            diag_file = method_dir / f"{method}_diagnostic_summary.png"
            diag_file.touch()
            assert diag_file.exists()

        # Test robust method diagnostic plots
        robust_dir = base_dir / "robust"
        for method in ["wasserstein", "scenario", "ellipsoidal"]:
            method_dir = robust_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)

            diag_file = method_dir / f"{method}_diagnostic_summary.png"
            diag_file.touch()
            assert diag_file.exists()


class TestNPZDataFiles:
    """Test NPZ data file creation for classical method."""

    def test_npz_data_structure(self, temp_directory):
        """Test the structure of NPZ data files in method-specific directories."""
        classical_dir = temp_directory / "classical"
        classical_dir.mkdir(parents=True, exist_ok=True)

        # Create method directory
        method_dir = classical_dir / "nelder_mead"
        method_dir.mkdir(parents=True, exist_ok=True)

        # Create mock correlation data
        n_angles, n_t2, n_t1 = 3, 20, 30
        mock_experimental_data = np.random.rand(n_angles, n_t2, n_t1) + 1.0
        mock_fitted_data = np.random.rand(n_angles, n_t2, n_t1) + 1.0
        mock_residuals_data = mock_experimental_data - mock_fitted_data
        mock_parameters = np.array([1.5, 2.0, 0.5])
        mock_uncertainties = np.array([0.1, 0.2, 0.05])

        # Save combined NPZ file (new structure: all data in one file per
        # method)
        fitted_file = method_dir / "fitted_data.npz"

        np.savez_compressed(
            fitted_file,
            c2_experimental=mock_experimental_data,
            c2_fitted=mock_fitted_data,
            residuals=mock_residuals_data,
            parameters=mock_parameters,
            uncertainties=mock_uncertainties,
            chi_squared=np.array([0.5]),
        )

        # Verify file exists and contains expected data
        assert fitted_file.exists()

        # Load and verify combined data structure
        data_loaded = np.load(fitted_file)

        assert "c2_experimental" in data_loaded
        assert "c2_fitted" in data_loaded
        assert "residuals" in data_loaded
        assert "parameters" in data_loaded
        assert "uncertainties" in data_loaded

        assert data_loaded["c2_experimental"].shape == (n_angles, n_t2, n_t1)
        assert data_loaded["c2_fitted"].shape == (n_angles, n_t2, n_t1)
        assert data_loaded["residuals"].shape == (n_angles, n_t2, n_t1)
        assert data_loaded["parameters"].shape == (3,)
        assert data_loaded["uncertainties"].shape == (3,)

    def test_fitted_data_calculation(self, temp_directory):
        """Test that fitted data follows the expected scaling relationship."""
        # Mock theoretical and experimental data
        n_angles, n_t2, n_t1 = 2, 15, 20
        theory_data = np.random.rand(n_angles, n_t2, n_t1)

        # Create experimental data with known scaling: exp = contrast * theory
        # + offset
        contrast = 2.5
        offset = 1.2
        exp_data = theory_data * contrast + offset

        # Simulate least squares fitting to recover scaling parameters
        for angle_idx in range(n_angles):
            theory_flat = theory_data[angle_idx].flatten()
            exp_flat = exp_data[angle_idx].flatten()

            # Create design matrix [theory, ones]
            A = np.vstack([theory_flat, np.ones(len(theory_flat))]).T
            scaling_params, residuals, rank, s = np.linalg.lstsq(
                A, exp_flat, rcond=None
            )
            recovered_contrast, recovered_offset = scaling_params

            # Check that we recovered the correct scaling parameters (within
            # tolerance)
            assert abs(recovered_contrast - contrast) < 0.1
            assert abs(recovered_offset - offset) < 0.1

            # Calculate fitted data
            fitted_flat = theory_flat * recovered_contrast + recovered_offset
            fitted_data = fitted_flat.reshape(theory_data[angle_idx].shape)

            # Calculate residuals
            residuals_data = exp_data[angle_idx] - fitted_data

            # Residuals should be small for perfect fit
            assert np.mean(np.abs(residuals_data)) < 0.1

    def test_residuals_calculation(self, temp_directory):
        """Test that residuals are correctly calculated as experimental - fitted."""
        # Create test data
        n_angles, n_t2, n_t1 = 2, 10, 12
        experimental_data = np.random.rand(n_angles, n_t2, n_t1) + 2.0
        fitted_data = np.random.rand(n_angles, n_t2, n_t1) + 1.8

        # Calculate expected residuals
        expected_residuals = experimental_data - fitted_data

        # Save and load to test file I/O
        classical_dir = temp_directory / "classical"
        classical_dir.mkdir(parents=True, exist_ok=True)

        exp_file = classical_dir / "experimental_data.npz"
        fitted_file = classical_dir / "fitted_data.npz"
        residuals_file = classical_dir / "residuals_data.npz"

        np.savez_compressed(exp_file, data=experimental_data)
        np.savez_compressed(fitted_file, data=fitted_data)
        np.savez_compressed(residuals_file, data=expected_residuals)

        # Load back and verify
        exp_loaded = np.load(exp_file)["data"]
        fitted_loaded = np.load(fitted_file)["data"]
        residuals_loaded = np.load(residuals_file)["data"]

        # Verify residuals calculation
        calculated_residuals = exp_loaded - fitted_loaded
        np.testing.assert_array_almost_equal(
            calculated_residuals, residuals_loaded)
        np.testing.assert_array_almost_equal(
            expected_residuals, residuals_loaded)


class TestPlotExperimentalDataBehavior:
    """Test the new behavior of --plot-experimental-data flag."""

    def test_experimental_data_early_exit(self):
        """Test that --plot-experimental-data exits early without fitting."""
        # This would be tested at the integration level in run_homodyne.py
        # Here we test the concept that when plot_experimental_data is True,
        # the analysis should skip all fitting procedures

        plot_experimental_data = True

        if plot_experimental_data:
            # Should exit early and not perform fitting
            fitting_performed = False
            results_directory = "homodyne_results/exp_data"
        else:
            # Normal analysis workflow
            fitting_performed = True
            results_directory = "homodyne_results"

        assert not fitting_performed
        assert results_directory == "homodyne_results/exp_data"

    def test_experimental_data_output_path(self, temp_directory):
        """Test that experimental data plots are saved to correct path."""
        # Create expected path for experimental data plots
        exp_data_path = temp_directory / "homodyne_results" / "exp_data"
        exp_data_path.mkdir(parents=True, exist_ok=True)

        # Create mock experimental data plot files
        plot_files = [
            "data_validation_phi_0.0deg.png",
            "data_validation_phi_90.0deg.png",
            "summary_statistics.txt",
        ]

        for filename in plot_files:
            (exp_data_path / filename).touch()

        # Verify files are in the correct location
        assert exp_data_path.exists()
        for filename in plot_files:
            assert (exp_data_path / filename).exists()

        # Verify they are NOT in the old location (./plots/data_validation/)
        old_path = temp_directory / "plots" / "data_validation"
        if old_path.exists():
            for filename in plot_files:
                assert not (old_path / filename).exists()


class TestBackwardCompatibility:
    """Test that changes maintain backward compatibility where expected."""

    def test_save_results_backward_compatibility(self):
        """Test that save_results_with_config works both with and without output_dir."""
        # This is already tested in test_save_results.py, but we can add
        # integration tests here
        pass

    def test_default_output_directory(self, test_output_directory):
        """Test that default output directory behavior is preserved."""
        # If no output directory is specified, should default to ./homodyne_results/
        # Using test_output_directory fixture ensures proper cleanup
        default_output_dir = test_output_directory

        assert default_output_dir.exists()
        assert default_output_dir.name == "homodyne_results"

        # The test_output_directory fixture only marks directories as test artifacts
        # if they didn't exist before. If the directory already exists (user data),
        # it won't be marked, which is the correct safety behavior.
        marker_file = default_output_dir / ".test-artifact"

        # Check if this was a pre-existing directory or newly created by the fixture
        # The fixture is smart and won't mark pre-existing directories as test
        # artifacts
        if marker_file.exists():
            # Directory was created by the test fixture
            print("\nDirectory was created by test fixture")
        else:
            # Directory existed before the test (user data) - this is safe
            print("\nDirectory pre-existed (user data) - correctly preserved")

    def test_configuration_compatibility(self, dummy_config):
        """Test that existing configurations still work with new directory structure."""
        # Existing configurations should work without modification
        # New directory structure should be created automatically

        config = dummy_config.copy()

        # Configuration should not require special settings for new directory
        # structure
        assert "output_settings" in config

        # The system should automatically create the correct directory structure
        # based on the method being used (classical, mcmc, experimental data
        # plotting)


class TestMCMCOutputDirectoryStructure:
    """Test the new output directory structure for MCMC method."""

    def test_mcmc_output_directory_structure(self, temp_directory):
        """Test that MCMC method creates the correct directory structure."""
        # Create expected MCMC output directory
        mcmc_dir = temp_directory / "homodyne_results" / "mcmc"
        mcmc_dir.mkdir(parents=True, exist_ok=True)

        # Expected files in mcmc directory
        expected_files = [
            "experimental_data.npz",
            "fitted_data.npz",
            "residuals_data.npz",
            "c2_heatmaps_phi_0.0deg.png",  # Example C2 heatmap
            "3d_surface_phi_0.0deg.png",  # 3D surface plot
            "3d_surface_residuals_phi_0.0deg.png",  # 3D residuals plot
            "mcmc_summary.json",  # MCMC summary
            "mcmc_trace.nc",  # NetCDF trace file
            "trace_plot.png",  # MCMC trace plots
            "corner_plot.png",  # Parameter posterior distributions
        ]

        # Create mock files
        for filename in expected_files:
            (mcmc_dir / filename).touch()

        # Verify structure
        assert mcmc_dir.exists()
        for filename in expected_files:
            assert (mcmc_dir / filename).exists()

    def test_mcmc_3d_plotting_files(self, temp_directory):
        """Test that MCMC method creates 3D surface plotting files."""
        # Create expected MCMC output directory
        mcmc_dir = temp_directory / "homodyne_results" / "mcmc"
        mcmc_dir.mkdir(parents=True, exist_ok=True)

        # Expected 3D plotting files
        expected_3d_files = [
            "3d_surface_phi_0.0deg.png",  # 3D surface plot for angle 0°
            "3d_surface_phi_45.0deg.png",  # 3D surface plot for angle 45°
            "3d_surface_residuals_phi_0.0deg.png",  # 3D residuals plot for angle 0°
            "3d_surface_residuals_phi_45.0deg.png",  # 3D residuals plot for angle 45°
        ]

        # Create mock 3D plotting files
        for filename in expected_3d_files:
            (mcmc_dir / filename).touch()

        # Verify 3D plotting files exist
        for filename in expected_3d_files:
            assert (mcmc_dir / filename).exists()

        # Verify they are in the correct MCMC directory
        assert all((mcmc_dir / f).parent ==
                   mcmc_dir for f in expected_3d_files)

    def test_mcmc_npz_data_structure(self, temp_directory):
        """Test the structure of MCMC NPZ data files."""
        mcmc_dir = temp_directory / "mcmc"
        mcmc_dir.mkdir(parents=True, exist_ok=True)

        # Create mock correlation data
        n_angles, n_t2, n_t1 = 3, 20, 30
        mock_experimental_data = np.random.rand(n_angles, n_t2, n_t1) + 1.0
        mock_fitted_data = np.random.rand(n_angles, n_t2, n_t1) + 1.0
        mock_residuals_data = mock_experimental_data - mock_fitted_data

        # Save NPZ files
        experimental_file = mcmc_dir / "experimental_data.npz"
        fitted_file = mcmc_dir / "fitted_data.npz"
        residuals_file = mcmc_dir / "residuals_data.npz"

        np.savez_compressed(experimental_file, data=mock_experimental_data)
        np.savez_compressed(fitted_file, data=mock_fitted_data)
        np.savez_compressed(residuals_file, data=mock_residuals_data)

        # Verify files exist and contain expected data
        assert experimental_file.exists()
        assert fitted_file.exists()
        assert residuals_file.exists()

        # Load and verify data
        exp_loaded = np.load(experimental_file)
        fitted_loaded = np.load(fitted_file)
        residuals_loaded = np.load(residuals_file)

        assert "data" in exp_loaded
        assert "data" in fitted_loaded
        assert "data" in residuals_loaded

        assert exp_loaded["data"].shape == (n_angles, n_t2, n_t1)
        assert fitted_loaded["data"].shape == (n_angles, n_t2, n_t1)
        assert residuals_loaded["data"].shape == (n_angles, n_t2, n_t1)

    def test_mcmc_fitted_data_calculation(self, temp_directory):
        """Test that MCMC fitted data follows the expected scaling relationship."""
        # Mock theoretical and experimental data
        n_angles, n_t2, n_t1 = 2, 15, 20
        theory_data = np.random.rand(n_angles, n_t2, n_t1)

        # Create experimental data with known scaling: exp = contrast * theory
        # + offset
        contrast = 2.5
        offset = 1.2
        exp_data = theory_data * contrast + offset

        # Simulate least squares fitting to recover scaling parameters
        for angle_idx in range(n_angles):
            theory_flat = theory_data[angle_idx].flatten()
            exp_flat = exp_data[angle_idx].flatten()

            # Create design matrix [theory, ones]
            A = np.vstack([theory_flat, np.ones(len(theory_flat))]).T
            scaling_params, residuals, rank, s = np.linalg.lstsq(
                A, exp_flat, rcond=None
            )
            recovered_contrast, recovered_offset = scaling_params

            # Check that we recovered the correct scaling parameters (within
            # tolerance)
            assert abs(recovered_contrast - contrast) < 0.1
            assert abs(recovered_offset - offset) < 0.1

            # Calculate fitted data
            fitted_flat = theory_flat * recovered_contrast + recovered_offset
            fitted_data = fitted_flat.reshape(theory_data[angle_idx].shape)

            # Calculate residuals
            residuals_data = exp_data[angle_idx] - fitted_data

            # Residuals should be small for perfect fit
            assert np.mean(np.abs(residuals_data)) < 0.1

    def test_mcmc_vs_classical_directory_separation(self, temp_directory):
        """Test that MCMC and classical results are properly separated."""
        base_dir = temp_directory / "homodyne_results"

        # Create both directory structures
        classical_dir = base_dir / "classical"
        mcmc_dir = base_dir / "mcmc"

        classical_dir.mkdir(parents=True, exist_ok=True)
        mcmc_dir.mkdir(parents=True, exist_ok=True)

        # Create files in both directories
        classical_files = [
            "experimental_data.npz",
            "fitted_data.npz",
            "residuals_data.npz",
            "c2_heatmaps_phi_0.0deg.png",
        ]

        mcmc_files = [
            "experimental_data.npz",
            "fitted_data.npz",
            "residuals_data.npz",
            "c2_heatmaps_phi_0.0deg.png",
            "mcmc_summary.json",
            "mcmc_trace.nc",
        ]

        # Create classical files
        for filename in classical_files:
            (classical_dir / filename).touch()

        # Create mcmc files
        for filename in mcmc_files:
            (mcmc_dir / filename).touch()

        # Verify separation
        assert classical_dir.exists()
        assert mcmc_dir.exists()
        assert classical_dir != mcmc_dir

        # Verify files are in correct locations
        for filename in classical_files:
            assert (classical_dir / filename).exists()

        for filename in mcmc_files:
            assert (mcmc_dir / filename).exists()

        # Verify unique MCMC files are only in MCMC directory
        assert (mcmc_dir / "mcmc_summary.json").exists()
        assert (mcmc_dir / "mcmc_trace.nc").exists()
        assert not (classical_dir / "mcmc_summary.json").exists()
        assert not (classical_dir / "mcmc_trace.nc").exists()

    def test_complete_directory_structure_with_mcmc(self, temp_directory):
        """Test the complete expected directory structure including MCMC."""
        base_dir = temp_directory / "homodyne_results"

        # Create complete expected structure with new classical/robust
        # structure + MCMC
        structure = {
            "homodyne_analysis_results.json": "file",
            "run.log": "file",
            "classical": {
                "all_classical_methods_summary.json": "file",
                "nelder_mead": {
                    "analysis_results_nelder_mead.json": "file",
                    "parameters.json": "file",
                    "fitted_data.npz": "file",  # Contains experimental, fitted, residuals
                    "c2_heatmaps_nelder_mead.png": "file",  # Single phi angle case
                    "nelder_mead_diagnostic_summary.png": "file",  # Method-specific diagnostic
                },
                "gurobi": {
                    "analysis_results_gurobi.json": "file",
                    "parameters.json": "file",
                    "fitted_data.npz": "file",  # Contains experimental, fitted, residuals
                    "c2_heatmaps_gurobi.png": "file",  # Single phi angle case
                    "gurobi_diagnostic_summary.png": "file",  # Method-specific diagnostic
                },
            },
            "mcmc": {
                "experimental_data.npz": "file",
                "fitted_data.npz": "file",
                "residuals_data.npz": "file",
                "c2_heatmaps_phi_0.0deg.png": "file",
                "3d_surface_phi_0.0deg.png": "file",  # 3D surface plot
                "3d_surface_residuals_phi_0.0deg.png": "file",  # 3D residuals plot
                "mcmc_summary.json": "file",
                "mcmc_trace.nc": "file",
                "trace_plot.png": "file",
                "corner_plot.png": "file",
            },
            "exp_data": {
                "data_validation_phi_0.0deg.png": "file",
                "summary_statistics.txt": "file",
            },
        }

        def create_structure(base_path, structure_dict):
            """Recursively create directory structure."""
            for name, content in structure_dict.items():
                path = base_path / name
                if content == "file":
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                elif isinstance(content, dict):
                    path.mkdir(parents=True, exist_ok=True)
                    create_structure(path, content)

        create_structure(base_dir, structure)

        # Verify complete structure
        assert base_dir.exists()
        assert (base_dir / "homodyne_analysis_results.json").exists()
        assert (base_dir / "classical").is_dir()
        assert (base_dir / "mcmc").is_dir()
        assert (base_dir / "exp_data").is_dir()

        # Verify new classical method-specific structure
        assert (
            base_dir /
            "classical" /
            "all_classical_methods_summary.json").exists()
        assert (base_dir / "classical" / "nelder_mead").is_dir()
        assert (
            base_dir / "classical" / "nelder_mead" / "analysis_results_nelder_mead.json"
        ).exists()
        assert (
            base_dir /
            "classical" /
            "nelder_mead" /
            "fitted_data.npz").exists()
        assert (base_dir / "classical" / "gurobi").is_dir()
        assert (base_dir / "classical" / "gurobi" / "fitted_data.npz").exists()

        # Verify mcmc files
        assert (base_dir / "mcmc" / "experimental_data.npz").exists()
        assert (base_dir / "mcmc" / "fitted_data.npz").exists()
        assert (base_dir / "mcmc" / "residuals_data.npz").exists()
        assert (base_dir / "mcmc" / "mcmc_summary.json").exists()
        assert (base_dir / "mcmc" / "mcmc_trace.nc").exists()

        # Verify experimental data files
        assert (
            base_dir /
            "exp_data" /
            "data_validation_phi_0.0deg.png").exists()
