"""
Tests for IO Utilities Module
=============================

Tests directory creation, file existence, and data saving operations.
"""

import json
import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

# Import the modules to test
from homodyne.core.io_utils import (
    ensure_dir,
    get_output_directory,
    save_analysis_results,
    save_fig,
    save_json,
    save_numpy,
    save_pickle,
    timestamped_filename,
)
from homodyne.tests.fixtures import dummy_config


class TestDirectoryCreation:
    """Test directory creation and file existence functionality."""

    def test_ensure_dir_creates_new_directory(self, temp_directory):
        """Test that ensure_dir creates a new directory."""
        new_dir = temp_directory / "new_test_dir"
        assert not new_dir.exists()

        result = ensure_dir(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir

    def test_ensure_dir_nested_directories(self, temp_directory):
        """Test creation of nested directory structures."""
        nested_path = temp_directory / "level1" / "level2" / "level3"
        assert not nested_path.exists()

        result = ensure_dir(nested_path)

        assert nested_path.exists()
        assert nested_path.is_dir()
        assert result == nested_path

        # Check that all parent directories were created
        assert (temp_directory / "level1").exists()
        assert (temp_directory / "level1" / "level2").exists()

    def test_ensure_dir_existing_directory(self, temp_directory):
        """Test that ensure_dir handles existing directories gracefully."""
        existing_dir = temp_directory / "existing"
        existing_dir.mkdir()
        assert existing_dir.exists()

        result = ensure_dir(existing_dir)

        assert existing_dir.exists()
        assert result == existing_dir

    def test_ensure_dir_permissions(self, temp_directory):
        """Test directory creation with custom permissions."""
        if os.name == "nt":  # Skip permission tests on Windows
            pytest.skip("Permission tests not applicable on Windows")

        perm_dir = temp_directory / "perm_test"
        result = ensure_dir(perm_dir, permissions=0o755)

        assert result.exists()
        # Note: Actual permission checking can be tricky due to umask
        assert oct(result.stat().st_mode)[-3:] in [
            "755",
            "775",
        ]  # Account for umask

    def test_ensure_dir_with_string_path(self, temp_directory):
        """Test that ensure_dir works with string paths."""
        str_path = str(temp_directory / "string_path")
        result = ensure_dir(str_path)

        assert Path(str_path).exists()
        assert isinstance(result, Path)

    def test_ensure_dir_file_exists_error(self, temp_directory):
        """Test error when trying to create directory where file exists."""
        file_path = temp_directory / "test_file"
        file_path.write_text("test content")

        with pytest.raises(OSError, match="Path exists but is not a directory"):
            ensure_dir(file_path)


class TestFilenameGeneration:
    """Test timestamped filename generation."""

    def test_timestamped_filename_basic(self):
        """Test basic filename generation with timestamp."""
        config = {
            "output_settings": {
                "file_naming": {
                    "timestamp_format": "%Y%m%d_%H%M%S",
                    "include_chi_squared": False,
                    "include_config_name": False,
                }
            }
        }

        result = timestamped_filename("test_file", config=config)

        assert result.startswith("test_file_")
        assert len(result) > len("test_file_")
        # Basic format check (YYYYMMDD_HHMMSS)
        timestamp_part = result.replace("test_file_", "")
        assert len(timestamp_part) == 15  # YYYYMMDD_HHMMSS

    def test_timestamped_filename_with_chi2(self):
        """Test filename generation with chi-squared value."""
        config = {
            "output_settings": {
                "file_naming": {
                    "timestamp_format": "%Y%m%d_%H%M%S",
                    "include_chi_squared": True,
                    "include_config_name": False,
                }
            }
        }

        result = timestamped_filename("results", chi2=1.234567, config=config)

        assert "chi2_1.234567" in result
        assert result.startswith("results_")

    def test_timestamped_filename_with_config_name(self):
        """Test filename generation with config version."""
        config = {
            "metadata": {"config_version": "5.1-test"},
            "output_settings": {
                "file_naming": {
                    "timestamp_format": "%Y%m%d_%H%M%S",
                    "include_chi_squared": False,
                    "include_config_name": True,
                }
            },
        }

        result = timestamped_filename("analysis", config=config)

        assert "v5.1-test" in result
        assert result.startswith("analysis_")

    def test_timestamped_filename_no_config(self):
        """Test filename generation with no config (uses defaults)."""
        result = timestamped_filename("default_test")

        assert result.startswith("default_test_")
        assert len(result) > len("default_test_")

    def test_timestamped_filename_custom_format(self):
        """Test filename generation with custom timestamp format."""
        config = {
            "output_settings": {
                "file_naming": {
                    "timestamp_format": "%Y-%m-%d",
                    "include_chi_squared": False,
                    "include_config_name": False,
                }
            }
        }

        result = timestamped_filename("custom", config=config)

        # Should contain date in YYYY-MM-DD format
        import re

        assert re.search(r"\d{4}-\d{2}-\d{2}", result)


class TestDataSaving:
    """Test data saving functions."""

    def test_save_json_success(self, temp_directory):
        """Test successful JSON saving."""
        data = {
            "test": "data",
            "numbers": [1, 2, 3],
            "nested": {"key": "value"},
        }
        filepath = temp_directory / "test.json"

        result = save_json(data, filepath)

        assert result is True
        assert filepath.exists()

        # Verify content
        with open(filepath, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        assert loaded_data == data

    def test_save_json_creates_directory(self, temp_directory):
        """Test that save_json creates parent directories."""
        data = {"test": "data"}
        filepath = temp_directory / "nested" / "directory" / "test.json"

        result = save_json(data, filepath)

        assert result is True
        assert filepath.exists()
        assert filepath.parent.exists()

    def test_save_json_non_serializable(self, temp_directory):
        """Test JSON saving with non-serializable data."""
        # Complex number is not JSON serializable by default
        data = {"complex": complex(1, 2)}
        filepath = temp_directory / "test.json"

        result = save_json(data, filepath)

        assert result is False
        # File may exist but should be empty or contain partial data
        if filepath.exists():
            # If file exists, it should be very small (essentially empty)
            assert filepath.stat().st_size < 100  # Very small file

    def test_save_numpy_compressed(self, temp_directory):
        """Test NumPy array saving with compression."""
        data = np.random.rand(10, 10)
        filepath = temp_directory / "test_array.npz"

        result = save_numpy(data, filepath, compressed=True)

        assert result is True
        assert filepath.exists()

        # Verify content
        loaded = np.load(filepath)
        np.testing.assert_array_equal(loaded["data"], data)

    def test_save_numpy_uncompressed(self, temp_directory):
        """Test NumPy array saving without compression."""
        data = np.random.rand(5, 5)
        filepath = temp_directory / "test_array.npy"

        result = save_numpy(data, filepath, compressed=False)

        assert result is True
        assert filepath.exists()

        # Verify content
        loaded = np.load(filepath)
        np.testing.assert_array_equal(loaded, data)

    def test_save_pickle_success(self, temp_directory):
        """Test successful pickle saving."""
        data = {"complex_object": [1, 2, {"nested": set([3, 4, 5])}]}
        filepath = temp_directory / "test.pkl"

        result = save_pickle(data, filepath)

        assert result is True
        assert filepath.exists()

        # Verify content
        # Note: This is safe as we're only loading test data we just created
        with open(filepath, "rb") as f:
            loaded_data = pickle.load(f)  # nosec B301 - Safe: loading trusted test data
        assert loaded_data == data

    def test_save_figure_success(self, temp_directory):
        """Test matplotlib figure saving."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Test Plot")
        filepath = temp_directory / "test_plot.png"

        result = save_fig(fig, filepath, dpi=100)

        assert result is True
        assert filepath.exists()

        # Check file size (should be > 0 for a valid PNG)
        assert filepath.stat().st_size > 1000

        plt.close(fig)

    def test_save_figure_with_format(self, temp_directory):
        """Test figure saving with specific format."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        filepath = temp_directory / "test_plot"  # No extension

        result = save_fig(fig, filepath, format="pdf", dpi=150)

        assert result is True
        # The actual saved file should have the format
        saved_files = list(temp_directory.glob("test_plot*"))
        assert len(saved_files) >= 1

        plt.close(fig)

    def test_save_figure_invalid_object(self, temp_directory):
        """Test figure saving with invalid figure object."""
        filepath = temp_directory / "invalid.png"

        result = save_fig("not_a_figure", filepath)

        assert result is False
        assert not filepath.exists()


class TestOutputDirectory:
    """Test output directory management."""

    def test_get_output_directory_from_config(self, temp_directory, dummy_config):
        """Test getting output directory from configuration."""
        dummy_config["output_settings"]["results_directory"] = str(
            temp_directory / "custom_results"
        )

        result = get_output_directory(dummy_config)

        assert result.exists()
        assert result.name == "custom_results"

    def test_get_output_directory_default(self, temp_directory):
        """Test getting default output directory."""
        # Change to temp directory to avoid creating "./results" in project
        original_cwd = Path.cwd()
        os.chdir(temp_directory)

        try:
            result = get_output_directory(None)
            assert result.exists()
            assert result.name == "homodyne_results"
        finally:
            os.chdir(original_cwd)

    def test_get_output_directory_no_config(self, temp_directory):
        """Test output directory with empty config."""
        original_cwd = Path.cwd()
        os.chdir(temp_directory)

        try:
            result = get_output_directory({})
            assert result.exists()
            assert result.name == "homodyne_results"
        finally:
            os.chdir(original_cwd)


class TestAnalysisResultsSaving:
    """Test complete analysis results saving."""

    def test_save_analysis_results_complete(self, temp_directory, dummy_config):
        """Test saving complete analysis results."""
        dummy_config["output_settings"]["results_directory"] = str(
            temp_directory / "test_output"
        )

        results = {
            "best_chi_squared": 1.234,
            "best_parameters": {"D0": 100.0, "alpha": -0.1},
            "correlation_data": np.random.rand(10, 10),
            "mcmc_trace": {"param1": [1, 2, 3], "param2": [4, 5, 6]},
        }

        save_status = save_analysis_results(results, dummy_config, "test_analysis")

        # Check that various save operations were attempted
        assert "json" in save_status
        assert "numpy" in save_status
        assert "pickle" in save_status

        # Verify files exist
        output_dir = Path(dummy_config["output_settings"]["results_directory"])
        json_files = list(output_dir.glob("*.json"))
        npz_files = list(output_dir.glob("*_data.npz"))
        pkl_files = list(output_dir.glob("*_full.pkl"))

        assert len(json_files) >= 1
        assert len(npz_files) >= 1
        assert len(pkl_files) >= 1

    def test_save_analysis_results_minimal(self, temp_directory):
        """Test saving minimal analysis results."""
        original_cwd = Path.cwd()
        os.chdir(temp_directory)

        try:
            results = {"chi_squared": 5.678}

            save_status = save_analysis_results(results, None, "minimal_test")

            assert "json" in save_status
            assert save_status["json"] is True

            # Check results directory was created
            results_dir = temp_directory / "homodyne_results"
            assert results_dir.exists()

        finally:
            os.chdir(original_cwd)


class TestErrorHandling:
    """Test error handling in IO operations."""

    def test_save_json_permission_error(self, temp_directory):
        """Test JSON saving with permission error."""
        if os.name == "nt":  # Skip on Windows due to different permission model
            pytest.skip("Permission tests not reliable on Windows")

        # Create a read-only directory
        readonly_dir = temp_directory / "readonly"
        readonly_dir.mkdir(mode=0o444)
        filepath = readonly_dir / "test.json"

        try:
            result = save_json({"test": "data"}, filepath)
            assert result is False
        finally:
            # Clean up - make writable again
            readonly_dir.chmod(0o755)

    @patch("builtins.open", side_effect=OSError("Simulated IO error"))
    def test_save_json_io_error(self, mock_open_func, temp_directory):
        """Test JSON saving with IO error."""
        filepath = temp_directory / "test.json"

        result = save_json({"test": "data"}, filepath)

        assert result is False

    def test_save_numpy_invalid_data(self, temp_directory):
        """Test NumPy saving with invalid data."""
        # Object arrays with unhashable types can cause issues
        invalid_data = np.array([{"unhashable": set([1, 2, 3])}], dtype=object)
        filepath = temp_directory / "invalid.npz"

        result = save_numpy(invalid_data, filepath)

        # This might succeed or fail depending on NumPy version
        # The test mainly checks that the function handles it gracefully
        assert isinstance(result, bool)

    def test_ensure_dir_permission_denied(self, temp_directory):
        """Test directory creation when permission is denied."""
        if os.name == "nt":
            pytest.skip("Permission tests not reliable on Windows")

        # Create a directory with no write permissions
        no_write_dir = temp_directory / "no_write"
        no_write_dir.mkdir(mode=0o444)

        try:
            with pytest.raises(OSError):
                ensure_dir(no_write_dir / "should_fail")
        finally:
            # Clean up
            no_write_dir.chmod(0o755)
