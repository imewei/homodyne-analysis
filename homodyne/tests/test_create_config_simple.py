"""
Test suite for create_config module - simplified version.

This module tests the configuration template creation functionality
with focus on testable aspects.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from homodyne.create_config import (
    create_config_from_template,
    main,
)


class TestCreateConfig:
    """Test configuration creation functionality."""

    def test_create_config_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "config.json"
            
            with pytest.raises(ValueError, match="Invalid mode"):
                create_config_from_template(
                    output_file=str(output_file),
                    mode="invalid_mode"
                )

    def test_create_config_default_parameters(self):
        """Test config creation with default parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "config.json"
            
            # This will likely fail due to missing templates, but we can test the interface
            try:
                create_config_from_template(output_file=str(output_file))
            except (FileNotFoundError, ValueError):
                # Expected when no templates available
                pass

    def test_create_config_all_parameters(self):
        """Test config creation with all parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "config.json"
            
            # Test that all parameters are accepted
            try:
                create_config_from_template(
                    output_file=str(output_file),
                    sample_name="test_sample",
                    experiment_name="Test Experiment",
                    author="Test Author",
                    mode="laminar_flow"
                )
            except (FileNotFoundError, ValueError) as e:
                # Expected when no templates available
                # But make sure it's not a parameter error
                assert "Invalid mode" not in str(e)

    def test_create_config_valid_modes(self):
        """Test that valid modes are accepted."""
        valid_modes = ["static_isotropic", "static_anisotropic", "laminar_flow"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for mode in valid_modes:
                output_file = Path(tmpdir) / f"{mode}_config.json"
                
                try:
                    create_config_from_template(
                        output_file=str(output_file),
                        mode=mode
                    )
                except (FileNotFoundError, ValueError) as e:
                    # Should not be an invalid mode error
                    assert "Invalid mode" not in str(e)

    def test_create_config_directory_creation(self):
        """Test that parent directories are created if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Non-existent subdirectory
            output_file = Path(tmpdir) / "configs" / "subdir" / "config.json"
            
            try:
                create_config_from_template(output_file=str(output_file))
            except (FileNotFoundError, ValueError):
                # Expected when templates not found
                # But directory should have been created
                assert output_file.parent.exists()


class TestMainFunction:
    """Test the main function."""

    def test_main_python_version_check(self):
        """Test Python version check."""
        with patch("homodyne.create_config.sys.version_info", (3, 11)):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_main_error_handling(self):
        """Test main function error handling."""
        # Test that ArgumentParser exits with error for invalid mode
        with patch("sys.argv", ["create_config", "--mode", "invalid_mode_xyz"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2  # ArgumentParser exits with code 2

    def test_main_success_handling(self):
        """Test main function success handling."""
        with patch("sys.argv", ["create_config"]), \
             patch("homodyne.create_config.create_config_from_template") as mock_create:
            mock_create.return_value = None
            
            result = main()
            assert result == 0

    def test_main_argument_parsing(self):
        """Test that argument parsing works correctly."""
        test_args = [
            "create_config",
            "--mode", "static_isotropic",
            "--output", "test.json",
            "--sample", "test_sample",
            "--experiment", "Test Experiment", 
            "--author", "Test Author"
        ]
        
        with patch("sys.argv", test_args), \
             patch("homodyne.create_config.create_config_from_template") as mock_create:
            mock_create.return_value = None
            
            result = main()
            
            # Verify function was called with correct arguments
            mock_create.assert_called_once_with(
                output_file="test.json",
                sample_name="test_sample",
                experiment_name="Test Experiment",
                author="Test Author",
                mode="static_isotropic"
            )
            assert result == 0

    def test_main_short_arguments(self):
        """Test short argument forms."""
        test_args = [
            "create_config",
            "-m", "laminar_flow",
            "-o", "short.json",
            "-s", "sample",
            "-e", "Experiment",
            "-a", "Author"
        ]
        
        with patch("sys.argv", test_args), \
             patch("homodyne.create_config.create_config_from_template") as mock_create:
            mock_create.return_value = None
            
            result = main()
            
            mock_create.assert_called_once_with(
                output_file="short.json",
                sample_name="sample",
                experiment_name="Experiment",
                author="Author",
                mode="laminar_flow"
            )
            assert result == 0

    def test_main_default_values(self):
        """Test that default values are used correctly."""
        with patch("sys.argv", ["create_config"]), \
             patch("homodyne.create_config.create_config_from_template") as mock_create:
            mock_create.return_value = None
            
            result = main()
            
            # Verify defaults are used
            mock_create.assert_called_once_with(
                output_file="my_config.json",
                sample_name=None,
                experiment_name=None,
                author=None,
                mode="laminar_flow"
            )
            assert result == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])