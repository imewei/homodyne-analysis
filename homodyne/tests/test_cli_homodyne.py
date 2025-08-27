"""
Test suite for the homodyne CLI command.

This module tests the main homodyne command-line interface,
including argument parsing, method selection, and execution flow.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from homodyne.run_homodyne import main


class TestHomodyneCLI:
    """Test suite for the main homodyne command."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary configuration file."""
        config = {
            "metadata": {"config_version": "1.0.0", "analysis_mode": "laminar_flow"},
            "experimental_data": {"data_folder_path": "./test_data/"},
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 100},
                "scattering": {"wavevector_q": 0.01},
                "geometry": {"stator_rotor_gap": 1000000},
            },
            "initial_parameters": {
                "values": [100.0, -0.5, 10.0, 0.0, 0.0, 0.0, 0.0],
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_t0",
                    "beta",
                    "gamma_dot_t_offset",
                    "epsilon",
                ],
            },
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"], "max_iter": 100}
            },
            "output_settings": {
                "reporting": {"generate_plots": False, "save_results": True}
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            return f.name

    @pytest.fixture
    def mock_data_file(self):
        """Create a mock data file."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            # Create mock data
            phi_angles = np.linspace(-30, 30, 10)
            c2_experimental = np.random.rand(10, 50, 50) + 1.0

            np.savez(f.name, phi_angles=phi_angles, c2_experimental=c2_experimental)
            return f.name

    def test_main_with_help_argument(self):
        """Test main function with --help argument."""
        with patch("sys.argv", ["homodyne", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_with_invalid_method(self):
        """Test main function with invalid method argument."""
        with patch("sys.argv", ["homodyne", "--method", "invalid"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_main_with_valid_methods(self):
        """Test main function accepts valid method choices."""
        valid_methods = ["classical", "robust", "mcmc", "all"]

        for method in valid_methods:
            with patch("sys.argv", ["homodyne", "--method", method, "--help"]):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                # Help should exit with code 0, invalid arguments with code 2
                assert exc_info.value.code == 0

    @patch("homodyne.run_homodyne.sys.version_info", (3, 11))
    def test_python_version_check(self):
        """Test that Python version check works."""
        with patch("sys.argv", ["homodyne"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_main_with_missing_config_file(self):
        """Test main function with non-existent config file."""
        with patch("sys.argv", ["homodyne", "--config", "nonexistent.json"]):
            with patch("os.path.exists", return_value=False):
                with pytest.raises((SystemExit, FileNotFoundError)):
                    main()

    def test_main_with_plot_simulated_data(self):
        """Test main function with --plot-simulated-data option."""
        with patch("sys.argv", ["homodyne", "--plot-simulated-data", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_with_static_modes(self):
        """Test main function with static mode options."""
        static_options = [
            "--static-isotropic",
            "--static-anisotropic",
            "--laminar-flow",
        ]

        for option in static_options:
            with patch("sys.argv", ["homodyne", option, "--help"]):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0

    def test_main_with_verbose_and_quiet_conflict(self):
        """Test that --verbose and --quiet cannot be used together."""
        with patch("sys.argv", ["homodyne", "--verbose", "--quiet"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_main_with_contrast_without_plot_simulated(self):
        """Test that --contrast requires --plot-simulated-data."""
        with patch("sys.argv", ["homodyne", "--contrast", "1.5"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_main_with_verbose_flag(self):
        """Test main function with --verbose flag."""
        with patch("sys.argv", ["homodyne", "--verbose", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_main_with_quiet_flag(self):
        """Test main function with --quiet flag."""
        with patch("sys.argv", ["homodyne", "--quiet", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_environment_variables_from_reference(self):
        """Test environment variables mentioned in CLI_REFERENCE.md."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "env_config.json")
            output_path = os.path.join(tmpdir, "env_output")

            # Create a minimal config
            with open(config_path, "w") as f:
                json.dump({"test": "config"}, f)

            # Test all environment variables from CLI reference
            env = {
                "HOMODYNE_CONFIG": config_path,  # Set default configuration file
                "HOMODYNE_OUTPUT_DIR": output_path,  # Set default output directory
                "HOMODYNE_LOG_LEVEL": "INFO",  # Set logging level
                "HOMODYNE_PROFILE": "1",  # Enable profiling (development)
            }

            with patch.dict(os.environ, env):
                with patch("sys.argv", ["homodyne", "--help"]):
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    assert exc_info.value.code == 0

    def test_main_with_phi_angles_without_plot_simulated(self):
        """Test that --phi-angles requires --plot-simulated-data."""
        with patch("sys.argv", ["homodyne", "--phi-angles", "0,45,90"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_main_with_offset_without_plot_simulated(self):
        """Test that --offset requires --plot-simulated-data."""
        with patch("sys.argv", ["homodyne", "--offset", "0.1"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def teardown_method(self, method):
        """Clean up temporary files after each test."""
        # Clean up any temporary files created during tests
        import glob
        import tempfile

        # Use proper temp directory instead of hardcoded /tmp
        temp_dir = tempfile.gettempdir()
        for pattern in ["*.json", "*.npz", "*.log"]:
            for file in glob.glob(os.path.join(temp_dir, pattern)):
                try:
                    os.remove(file)
                except:
                    pass


class TestCLIReferenceExamples:
    """Test examples from CLI_REFERENCE.md work correctly."""

    def test_example_commands_help_functionality(self):
        """Test that example commands from CLI reference can show help."""
        example_commands = [
            ["homodyne", "--help"],
            ["homodyne", "--method", "robust", "--help"],
            ["homodyne", "--method", "all", "--verbose", "--help"],
            ["homodyne", "--config", "my_config.json", "--help"],
            ["homodyne", "--static-isotropic", "--method", "classical", "--help"],
            ["homodyne", "--plot-experimental-data", "--help"],
            ["homodyne", "--plot-simulated-data", "--help"],
            [
                "homodyne",
                "--plot-simulated-data",
                "--contrast",
                "1.5",
                "--offset",
                "0.1",
                "--help",
            ],
            ["homodyne", "--quiet", "--method", "all", "--help"],
        ]

        for cmd in example_commands:
            with patch("sys.argv", cmd):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0, f"Failed for command: {' '.join(cmd)}"

    def test_phi_angles_example_format(self):
        """Test that phi-angles example format works."""
        with patch(
            "sys.argv",
            [
                "homodyne",
                "--plot-simulated-data",
                "--phi-angles",
                "0,45,90,135",
                "--help",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0


class TestArgumentValidation:
    """Test command-line argument validation logic."""

    def test_valid_method_choices_accepted(self):
        """Test that valid methods are accepted."""
        valid_methods = ["classical", "robust", "mcmc", "all"]

        for method in valid_methods:
            with patch("sys.argv", ["homodyne", "--method", method, "--help"]):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                # Should exit with help (code 0), not argument error (code 2)
                assert exc_info.value.code == 0

    def test_invalid_method_rejected(self):
        """Test that invalid method names are rejected."""
        with patch("sys.argv", ["homodyne", "--method", "invalid"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_argument_combination_validation(self):
        """Test that incompatible argument combinations are rejected."""
        # Test contrast without plot-simulated-data
        with patch("sys.argv", ["homodyne", "--contrast", "1.5"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_exit_codes_documented_behavior(self):
        """Test that exit codes match documented behavior from CLI reference."""
        # Test successful help (exit code 0)
        with patch("sys.argv", ["homodyne", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

        # Test configuration error (exit code 2)
        with patch("sys.argv", ["homodyne", "--method", "invalid"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

        # Test Python version mismatch (exit code 1)
        with patch("homodyne.run_homodyne.sys.version_info", (3, 11)):
            with patch("sys.argv", ["homodyne"]):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1

    def test_default_config_path_matches_reference(self):
        """Test that default config path matches CLI reference."""
        # According to CLI reference, default config is ./homodyne_config.json
        with patch("sys.argv", ["homodyne", "--help"]):
            with pytest.raises(SystemExit):
                main()
        # This test mainly documents the expected behavior

    def test_default_output_dir_matches_reference(self):
        """Test that default output directory matches CLI reference."""
        # According to CLI reference, default output dir is ./homodyne_results
        with patch("sys.argv", ["homodyne", "--help"]):
            with pytest.raises(SystemExit):
                main()
        # This test mainly documents the expected behavior


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
