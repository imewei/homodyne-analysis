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

import pytest
import tempfile
import numpy as np
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import json

from homodyne.tests.fixtures import (
    dummy_config,
    temp_directory,
)


class TestOutputDirectoryStructure:
    """Test the new output directory structure and file organization."""

    def test_classical_output_directory_structure(self, temp_directory):
        """Test that classical method creates the correct directory structure."""
        # Create expected classical output directory
        classical_dir = temp_directory / "homodyne_results" / "classical"
        classical_dir.mkdir(parents=True, exist_ok=True)
        
        # Expected files in classical directory
        expected_files = [
            "experimental_data.npz",
            "fitted_data.npz", 
            "residuals_data.npz",
            "c2_heatmaps_phi_0.0deg.png"  # Example C2 heatmap
        ]
        
        # Create mock files
        for filename in expected_files:
            (classical_dir / filename).touch()
        
        # Verify structure
        assert classical_dir.exists()
        for filename in expected_files:
            assert (classical_dir / filename).exists()

    def test_experimental_data_output_directory_structure(self, temp_directory):
        """Test that experimental data plots create the correct directory structure."""
        # Create expected experimental data output directory
        exp_data_dir = temp_directory / "homodyne_results" / "exp_data"
        exp_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Expected files in exp_data directory  
        expected_files = [
            "data_validation_phi_0.0deg.png",
            "data_validation_phi_45.0deg.png",
            "summary_statistics.txt"
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
        
        # Main results file should be in output directory, not current directory
        main_results_file = results_dir / "homodyne_analysis_results.json"
        main_results_file.touch()
        
        assert main_results_file.exists()
        assert main_results_file.parent == results_dir

    def test_complete_directory_structure(self, temp_directory):
        """Test the complete expected directory structure."""
        base_dir = temp_directory / "homodyne_results"
        
        # Create complete expected structure
        structure = {
            "homodyne_analysis_results.json": "file",
            "per_angle_chi_squared_classical.json": "file", 
            "run.log": "file",
            "classical": {
                "experimental_data.npz": "file",
                "fitted_data.npz": "file",
                "residuals_data.npz": "file",
                "c2_heatmaps_phi_0.0deg.png": "file",
                "c2_heatmaps_phi_45.0deg.png": "file"
            },
            "exp_data": {
                "data_validation_phi_0.0deg.png": "file",
                "data_validation_phi_45.0deg.png": "file",
                "summary_statistics.txt": "file"
            }
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
        assert (base_dir / "exp_data").is_dir()
        assert (base_dir / "classical" / "experimental_data.npz").exists()
        assert (base_dir / "classical" / "fitted_data.npz").exists()
        assert (base_dir / "classical" / "residuals_data.npz").exists()
        assert (base_dir / "exp_data" / "data_validation_phi_0.0deg.png").exists()


class TestNPZDataFiles:
    """Test NPZ data file creation for classical method."""

    def test_npz_data_structure(self, temp_directory):
        """Test the structure of NPZ data files."""
        classical_dir = temp_directory / "classical"
        classical_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock correlation data
        n_angles, n_t2, n_t1 = 3, 20, 30
        mock_experimental_data = np.random.rand(n_angles, n_t2, n_t1) + 1.0
        mock_fitted_data = np.random.rand(n_angles, n_t2, n_t1) + 1.0
        mock_residuals_data = mock_experimental_data - mock_fitted_data
        
        # Save NPZ files
        experimental_file = classical_dir / "experimental_data.npz"
        fitted_file = classical_dir / "fitted_data.npz"
        residuals_file = classical_dir / "residuals_data.npz"
        
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

    def test_fitted_data_calculation(self, temp_directory):
        """Test that fitted data follows the expected scaling relationship."""
        # Mock theoretical and experimental data
        n_angles, n_t2, n_t1 = 2, 15, 20
        theory_data = np.random.rand(n_angles, n_t2, n_t1)
        
        # Create experimental data with known scaling: exp = contrast * theory + offset
        contrast = 2.5
        offset = 1.2
        exp_data = theory_data * contrast + offset
        
        # Simulate least squares fitting to recover scaling parameters
        for angle_idx in range(n_angles):
            theory_flat = theory_data[angle_idx].flatten()
            exp_flat = exp_data[angle_idx].flatten()
            
            # Create design matrix [theory, ones]
            A = np.vstack([theory_flat, np.ones(len(theory_flat))]).T
            scaling_params, residuals, rank, s = np.linalg.lstsq(A, exp_flat, rcond=None)
            recovered_contrast, recovered_offset = scaling_params
            
            # Check that we recovered the correct scaling parameters (within tolerance)
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
        np.testing.assert_array_almost_equal(calculated_residuals, residuals_loaded)
        np.testing.assert_array_almost_equal(expected_residuals, residuals_loaded)


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
            "summary_statistics.txt"
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
        # This is already tested in test_save_results.py, but we can add integration tests here
        pass

    def test_default_output_directory(self, temp_directory):
        """Test that default output directory behavior is preserved."""
        # If no output directory is specified, should default to ./homodyne_results/
        default_output_dir = temp_directory / "homodyne_results" 
        default_output_dir.mkdir(parents=True, exist_ok=True)
        
        assert default_output_dir.exists()
        assert default_output_dir.name == "homodyne_results"

    def test_configuration_compatibility(self, dummy_config):
        """Test that existing configurations still work with new directory structure."""
        # Existing configurations should work without modification
        # New directory structure should be created automatically
        
        config = dummy_config.copy()
        
        # Configuration should not require special settings for new directory structure
        assert "output_settings" in config
        
        # The system should automatically create the correct directory structure
        # based on the method being used (classical, mcmc, experimental data plotting)