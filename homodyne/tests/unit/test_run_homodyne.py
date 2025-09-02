"""
Tests for homodyne/run_homodyne.py - CLI interface and main entry point.

This module tests the command-line interface for homodyne scattering analysis,
including argument parsing, configuration handling, analysis orchestration,
and error handling.
"""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

# Import the module under test
from homodyne.run_homodyne import main, plot_simulated_data, print_banner, run_analysis


class TestArgumentParsing:
    """Test command-line argument parsing functionality."""

    def test_default_arguments(self):
        """Test that default arguments are properly set."""
        with patch("sys.argv", ["run_homodyne.py"]):
            with patch("homodyne.run_homodyne.run_analysis") as mock_run:
                with patch("homodyne.run_homodyne.setup_logging"):
                    with patch("homodyne.run_homodyne.print_banner"):
                        with patch("homodyne.run_homodyne.logging.getLogger"):
                            with patch("sys.exit"):
                                main()

                # Verify run_analysis was called with expected defaults
                args = mock_run.call_args[0][0]
                assert args.method == "classical"
                assert args.config == Path("./homodyne_config.json")
                assert args.output_dir == Path("./homodyne_results")
                assert args.verbose is False
                assert args.quiet is False
                assert args.static_isotropic is False
                assert args.laminar_flow is False
                assert args.plot_simulated_data is False
                assert args.contrast == 1.0
                assert args.offset == 0.0

    def test_method_argument_parsing(self):
        """Test parsing of different method arguments."""
        methods = ["classical", "mcmc", "robust", "all"]

        for method in methods:
            with patch("sys.argv", ["run_homodyne.py", "--method", method]):
                with patch("homodyne.run_homodyne.run_analysis") as mock_run:
                    with patch("homodyne.run_homodyne.setup_logging"):
                        with patch("homodyne.run_homodyne.print_banner"):
                            with patch("homodyne.run_homodyne.logging.getLogger"):
                                with patch("sys.exit"):
                                    main()

                    args = mock_run.call_args[0][0]
                    assert args.method == method

    def test_config_path_parsing(self):
        """Test parsing of config file path argument."""
        config_path = "/custom/path/config.json"

        with patch("sys.argv", ["run_homodyne.py", "--config", config_path]):
            with patch("homodyne.run_homodyne.run_analysis") as mock_run:
                with patch("homodyne.run_homodyne.setup_logging"):
                    with patch("homodyne.run_homodyne.print_banner"):
                        with patch("homodyne.run_homodyne.logging.getLogger"):
                            with patch("sys.exit"):
                                main()

                args = mock_run.call_args[0][0]
                assert args.config == Path(config_path)

    def test_output_directory_parsing(self):
        """Test parsing of output directory argument."""
        output_dir = "/custom/output/path"

        with patch("sys.argv", ["run_homodyne.py", "--output-dir", output_dir]):
            with patch("homodyne.run_homodyne.run_analysis") as mock_run:
                with patch("homodyne.run_homodyne.setup_logging"):
                    with patch("homodyne.run_homodyne.print_banner"):
                        with patch("homodyne.run_homodyne.logging.getLogger"):
                            with patch("sys.exit"):
                                main()

                args = mock_run.call_args[0][0]
                assert args.output_dir == Path(output_dir)

    def test_verbose_and_quiet_flags(self):
        """Test parsing of verbose and quiet flags."""
        # Test verbose flag
        with patch("sys.argv", ["run_homodyne.py", "--verbose"]):
            with patch("homodyne.run_homodyne.run_analysis") as mock_run:
                with patch("homodyne.run_homodyne.setup_logging"):
                    with patch("homodyne.run_homodyne.print_banner"):
                        with patch("homodyne.run_homodyne.logging.getLogger"):
                            with patch("sys.exit"):
                                main()

                args = mock_run.call_args[0][0]
                assert args.verbose is True
                assert args.quiet is False

        # Test quiet flag
        with patch("sys.argv", ["run_homodyne.py", "--quiet"]):
            with patch("homodyne.run_homodyne.run_analysis") as mock_run:
                with patch("homodyne.run_homodyne.setup_logging"):
                    with patch("homodyne.run_homodyne.print_banner"):
                        with patch("homodyne.run_homodyne.logging.getLogger"):
                            with patch("sys.exit"):
                                main()

                args = mock_run.call_args[0][0]
                assert args.verbose is False
                assert args.quiet is True

    def test_mutually_exclusive_logging_flags_error(self):
        """Test that verbose and quiet flags are mutually exclusive."""
        with patch("sys.argv", ["run_homodyne.py", "--verbose", "--quiet"]):
            with patch("sys.stderr"):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 2  # argparse error code

    def test_analysis_mode_flags(self):
        """Test parsing of analysis mode flags."""
        modes = [
            ("--static-isotropic", "static_isotropic"),
            ("--static-anisotropic", "static_anisotropic"),
            ("--laminar-flow", "laminar_flow"),
        ]

        for flag, attr in modes:
            with patch("sys.argv", ["run_homodyne.py", flag]):
                with patch("homodyne.run_homodyne.run_analysis") as mock_run:
                    with patch("homodyne.run_homodyne.setup_logging"):
                        with patch("homodyne.run_homodyne.print_banner"):
                            with patch("homodyne.run_homodyne.logging.getLogger"):
                                with patch("sys.exit"):
                                    main()

                    args = mock_run.call_args[0][0]
                    assert getattr(args, attr) is True

    def test_plotting_flags(self):
        """Test parsing of plotting flags."""
        # Test plot-experimental-data flag
        with patch("sys.argv", ["run_homodyne.py", "--plot-experimental-data"]):
            with patch("homodyne.run_homodyne.run_analysis") as mock_run:
                with patch("homodyne.run_homodyne.setup_logging"):
                    with patch("homodyne.run_homodyne.print_banner"):
                        with patch("homodyne.run_homodyne.logging.getLogger"):
                            with patch("sys.exit"):
                                main()

                args = mock_run.call_args[0][0]
                assert args.plot_experimental_data is True

        # Test plot-simulated-data flag (should trigger special handling)
        with patch("sys.argv", ["run_homodyne.py", "--plot-simulated-data"]):
            with patch("homodyne.run_homodyne.plot_simulated_data") as mock_plot:
                with patch("homodyne.run_homodyne.setup_logging"):
                    with patch("homodyne.run_homodyne.print_banner"):
                        with patch("homodyne.run_homodyne.logging.getLogger"):
                            with patch("sys.exit"):
                                main()

                assert mock_plot.called

    def test_contrast_and_offset_parameters(self):
        """Test parsing of contrast and offset parameters."""
        contrast = 1.5
        offset = 0.1

        with patch(
            "sys.argv",
            [
                "run_homodyne.py",
                "--plot-simulated-data",
                "--contrast",
                str(contrast),
                "--offset",
                str(offset),
            ],
        ):
            with patch("homodyne.run_homodyne.plot_simulated_data") as mock_plot:
                with patch("homodyne.run_homodyne.setup_logging"):
                    with patch("homodyne.run_homodyne.print_banner"):
                        with patch("homodyne.run_homodyne.logging.getLogger"):
                            with patch("sys.exit"):
                                main()

                args = mock_plot.call_args[0][0]
                assert args.contrast == contrast
                assert args.offset == offset

    def test_phi_angles_parameter(self):
        """Test parsing of phi angles parameter."""
        phi_angles = "0,45,90,135"

        with patch(
            "sys.argv",
            ["run_homodyne.py", "--plot-simulated-data", "--phi-angles", phi_angles],
        ):
            with patch("homodyne.run_homodyne.plot_simulated_data") as mock_plot:
                with patch("homodyne.run_homodyne.setup_logging"):
                    with patch("homodyne.run_homodyne.print_banner"):
                        with patch("homodyne.run_homodyne.logging.getLogger"):
                            with patch("sys.exit"):
                                main()

                args = mock_plot.call_args[0][0]
                assert args.phi_angles == phi_angles

    def test_contrast_offset_without_plot_simulated_error(self):
        """Test that contrast/offset without plot-simulated-data raises error."""
        with patch("sys.argv", ["run_homodyne.py", "--contrast", "1.5"]):
            with patch("sys.stderr"):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 2  # argparse error code

    def test_phi_angles_without_plot_simulated_error(self):
        """Test that phi-angles without plot-simulated-data raises error."""
        with patch("sys.argv", ["run_homodyne.py", "--phi-angles", "0,45,90"]):
            with patch("sys.stderr"):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 2  # argparse error code


class TestPrintBanner:
    """Test the print_banner function."""

    def test_print_banner_basic(self, capsys):
        """Test that print_banner outputs expected content."""
        args = argparse.Namespace(
            method="classical",
            config=Path("./test_config.json"),
            output_dir=Path("./test_output"),
            quiet=False,
            verbose=False,
            static_isotropic=False,
            static_anisotropic=False,
            laminar_flow=False,
        )

        print_banner(args)

        captured = capsys.readouterr()
        assert "HOMODYNE ANALYSIS RUNNER" in captured.out
        assert "classical" in captured.out
        assert "test_config.json" in captured.out

    def test_print_banner_different_methods(self, capsys):
        """Test print_banner with different methods."""
        methods = ["classical", "mcmc", "robust", "all"]

        for method in methods:
            args = argparse.Namespace(
                method=method,
                config=Path("./test_config.json"),
                output_dir=Path("./test_output"),
                quiet=False,
                verbose=False,
                static_isotropic=False,
                static_anisotropic=False,
                laminar_flow=False,
            )
            print_banner(args)

            captured = capsys.readouterr()
            assert method in captured.out


class TestRunAnalysis:
    """Test the run_analysis function."""

    def create_temp_config(self, config_data: dict[str, Any]) -> Path:
        """Helper to create temporary config file."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(config_data, temp_file)
        temp_file.flush()
        return Path(temp_file.name)

    def test_run_analysis_config_file_not_found(self, tmp_path):
        """Test run_analysis with missing config file."""
        args = argparse.Namespace(
            config=tmp_path / "nonexistent.json",
            output_dir=tmp_path,
            static_isotropic=False,
            static_anisotropic=False,
            laminar_flow=False,
            plot_experimental_data=False,
            method="classical",
        )

        with patch("homodyne.run_homodyne.logging.getLogger") as mock_logger:
            logger_instance = Mock()
            mock_logger.return_value = logger_instance

            with pytest.raises(SystemExit) as exc_info:
                run_analysis(args)

            assert exc_info.value.code == 1
            logger_instance.error.assert_called()

    def test_run_analysis_config_path_not_file(self, tmp_path):
        """Test run_analysis when config path is a directory."""
        config_dir = tmp_path / "config_dir"
        config_dir.mkdir()

        args = argparse.Namespace(
            config=config_dir,
            output_dir=tmp_path,
            static_isotropic=False,
            static_anisotropic=False,
            laminar_flow=False,
            plot_experimental_data=False,
            method="classical",
        )

        with patch("homodyne.run_homodyne.logging.getLogger") as mock_logger:
            logger_instance = Mock()
            mock_logger.return_value = logger_instance

            with pytest.raises(SystemExit) as exc_info:
                run_analysis(args)

            assert exc_info.value.code == 1
            logger_instance.error.assert_called()

    @patch("homodyne.run_homodyne.HomodyneAnalysisCore")
    def test_run_analysis_success_classical(self, mock_core_class, tmp_path):
        """Test successful run_analysis with classical method."""
        # Create temporary config file
        config_data = {
            "analysis_settings": {"analysis_mode": "static_isotropic"},
            "data_settings": {"data_file": "test.h5"},
            "parameters": {"D_parallel": 1.0},
        }
        config_path = self.create_temp_config(config_data)

        args = argparse.Namespace(
            config=config_path,
            output_dir=tmp_path,
            static_isotropic=False,
            static_anisotropic=False,
            laminar_flow=False,
            plot_experimental_data=False,
            method="classical",
        )

        # Mock the analysis core
        mock_core = Mock()
        mock_core_class.return_value = mock_core
        mock_core.load_experimental_data.return_value = (Mock(), Mock(), Mock(), Mock())
        mock_core.config = {"initial_parameters": {"values": [1.0, 2.0, 3.0]}}
        mock_core.calculate_chi_squared_optimized.return_value = 1.5
        mock_core.save_results_with_config = Mock()
        mock_core.analyze_per_angle_chi_squared = Mock()

        # Mock the optimization function to return successful results
        with patch(
            "homodyne.run_homodyne.run_classical_optimization"
        ) as mock_classical_opt:
            mock_classical_opt.return_value = {
                "methods_used": ["Classical"],
                "classical_optimization": {
                    "parameters": [1.0, 2.0, 3.0],
                    "success": True,
                },
            }

            with patch("homodyne.run_homodyne.logging.getLogger") as mock_logger:
                logger_instance = Mock()
                mock_logger.return_value = logger_instance

                # Should complete without raising SystemExit
                run_analysis(args)

                # Verify core was created and classical method called
                mock_core_class.assert_called_once()
                mock_classical_opt.assert_called_once()

    @patch("homodyne.run_homodyne.HomodyneAnalysisCore")
    def test_run_analysis_with_mode_overrides(self, mock_core_class, tmp_path):
        """Test run_analysis with different mode overrides."""
        config_data = {"analysis_settings": {"analysis_mode": "laminar_flow"}}
        config_path = self.create_temp_config(config_data)

        modes = [
            ("static_isotropic", True, False, False),
            ("static_anisotropic", False, True, False),
            ("laminar_flow", False, False, True),
        ]

        for mode_name, static_iso, static_aniso, laminar in modes:
            args = argparse.Namespace(
                config=config_path,
                output_dir=tmp_path,
                static_isotropic=static_iso,
                static_anisotropic=static_aniso,
                laminar_flow=laminar,
                plot_experimental_data=False,
                method="classical",
            )

            mock_core = Mock()
            mock_core_class.return_value = mock_core
            mock_core.load_experimental_data.return_value = (
                Mock(),
                Mock(),
                Mock(),
                Mock(),
            )
            mock_core.config = {"initial_parameters": {"values": [1.0, 2.0, 3.0]}}
            mock_core.calculate_chi_squared_optimized.return_value = 1.5
            mock_core.save_results_with_config = Mock()
            mock_core.analyze_per_angle_chi_squared = Mock()

            # Mock the optimization function to return successful results
            with patch(
                "homodyne.run_homodyne.run_classical_optimization"
            ) as mock_classical_opt:
                mock_classical_opt.return_value = {
                    "methods_used": ["Classical"],
                    "classical_optimization": {
                        "parameters": [1.0, 2.0, 3.0],
                        "success": True,
                    },
                }

                with patch("homodyne.run_homodyne.logging.getLogger"):
                    run_analysis(args)

                    # Verify core was called with config override
                    call_args = mock_core_class.call_args
                    if any([static_iso, static_aniso, laminar]):
                        assert call_args[1]["config_override"] is not None

                mock_core_class.reset_mock()

    @patch("homodyne.run_homodyne.HomodyneAnalysisCore", None)
    def test_run_analysis_core_unavailable(self, tmp_path):
        """Test run_analysis when HomodyneAnalysisCore is unavailable."""
        config_data = {"analysis_settings": {"analysis_mode": "static_isotropic"}}
        config_path = self.create_temp_config(config_data)

        args = argparse.Namespace(
            config=config_path,
            output_dir=tmp_path,
            static_isotropic=False,
            static_anisotropic=False,
            laminar_flow=False,
            plot_experimental_data=False,
            method="classical",
        )

        with patch("homodyne.run_homodyne.logging.getLogger") as mock_logger:
            logger_instance = Mock()
            mock_logger.return_value = logger_instance

            with pytest.raises(SystemExit) as exc_info:
                run_analysis(args)

            assert exc_info.value.code == 1
            logger_instance.error.assert_called()

    @patch("homodyne.run_homodyne.HomodyneAnalysisCore")
    def test_run_analysis_different_methods(self, mock_core_class, tmp_path):
        """Test run_analysis with different optimization methods."""
        config_data = {"analysis_settings": {"analysis_mode": "static_isotropic"}}
        config_path = self.create_temp_config(config_data)

        methods = ["classical", "mcmc", "robust", "all"]
        expected_calls = [
            "run_classical_optimization",
            "run_mcmc_optimization",
            "run_robust_optimization",
            "run_all_methods",
        ]

        for method, expected_call in zip(methods, expected_calls, strict=False):
            args = argparse.Namespace(
                config=config_path,
                output_dir=tmp_path,
                static_isotropic=False,
                static_anisotropic=False,
                laminar_flow=False,
                plot_experimental_data=False,
                method=method,
            )

            mock_core = Mock()
            mock_core_class.return_value = mock_core
            mock_core.load_experimental_data.return_value = (
                Mock(),
                Mock(),
                Mock(),
                Mock(),
            )
            mock_core.config = {"initial_parameters": {"values": [1.0, 2.0, 3.0]}}
            mock_core.calculate_chi_squared_optimized.return_value = 1.5
            mock_core.save_results_with_config = Mock()
            mock_core.analyze_per_angle_chi_squared = Mock()

            # Mock the specific optimization function
            optimization_function_name = f"run_{expected_call.replace('run_', '').replace('_optimization', '')}_optimization"
            if expected_call == "run_all_methods":
                optimization_function_name = "run_all_methods"

            with patch(
                f"homodyne.run_homodyne.{optimization_function_name}"
            ) as mock_opt:
                mock_opt.return_value = {
                    "methods_used": [method.capitalize()],
                    f"{method}_optimization": {
                        "parameters": [1.0, 2.0, 3.0],
                        "success": True,
                    },
                }

                with patch("homodyne.run_homodyne.logging.getLogger"):
                    run_analysis(args)

                    # Verify correct optimization function was called
                    mock_opt.assert_called_once()

            mock_core_class.reset_mock()

    def teardown_method(self):
        """Clean up temporary files."""
        import os

        for temp_file in getattr(self, "_temp_files", []):
            try:
                os.unlink(temp_file)
            except (OSError, FileNotFoundError):
                pass


class TestPlotSimulatedData:
    """Test the plot_simulated_data function."""

    def create_temp_config(self, config_data: dict[str, Any]) -> Path:
        """Helper to create temporary config file."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(config_data, temp_file)
        temp_file.flush()
        return Path(temp_file.name)

    @patch("homodyne.run_homodyne.HomodyneAnalysisCore")
    def test_plot_simulated_data_success(self, mock_core_class, tmp_path):
        """Test successful plot_simulated_data execution."""
        config_data = {
            "analysis_settings": {"analysis_mode": "static_isotropic"},
            "parameters": {"D_parallel": 1.0},
        }
        config_path = self.create_temp_config(config_data)

        args = argparse.Namespace(
            config=config_path,
            output_dir=tmp_path,
            static_isotropic=False,
            static_anisotropic=False,
            laminar_flow=False,
            contrast=1.0,
            offset=0.0,
            phi_angles=None,
        )

        mock_core = Mock()
        mock_core_class.return_value = mock_core
        mock_core.calculate_c2_single_angle_optimized.return_value = (
            1.0  # Return float instead of Mock
        )

        with patch("homodyne.run_homodyne.logging.getLogger"):
            # Should complete without raising exception
            plot_simulated_data(args)

            # Verify core was created
            mock_core_class.assert_called_once()

    @patch("homodyne.run_homodyne.HomodyneAnalysisCore")
    def test_plot_simulated_data_with_custom_parameters(
        self, mock_core_class, tmp_path
    ):
        """Test plot_simulated_data with custom contrast, offset, and phi angles."""
        config_data = {
            "analysis_settings": {"analysis_mode": "static_isotropic"},
            "parameters": {"D_parallel": 1.0},
        }
        config_path = self.create_temp_config(config_data)

        args = argparse.Namespace(
            config=config_path,
            output_dir=tmp_path,
            static_isotropic=False,
            static_anisotropic=False,
            laminar_flow=False,
            contrast=1.5,
            offset=0.1,
            phi_angles="0,45,90,135",
        )

        mock_core = Mock()
        mock_core_class.return_value = mock_core
        mock_core.calculate_c2_single_angle_optimized.return_value = (
            1.0  # Return float instead of Mock
        )

        with patch("homodyne.run_homodyne.logging.getLogger"):
            plot_simulated_data(args)

            # Verify core was created
            mock_core_class.assert_called_once()

    @patch("homodyne.run_homodyne.HomodyneAnalysisCore")
    def test_plot_simulated_data_missing_config(self, mock_core_class, tmp_path):
        """Test plot_simulated_data with missing config file uses defaults."""
        args = argparse.Namespace(
            config=tmp_path / "nonexistent.json",
            output_dir=tmp_path,
            static_isotropic=False,
            static_anisotropic=False,
            laminar_flow=False,
            contrast=1.0,
            offset=0.0,
            phi_angles=None,
        )

        mock_core = Mock()
        mock_core_class.return_value = mock_core
        mock_core.calculate_c2_single_angle_optimized.return_value = 1.0

        with patch("homodyne.run_homodyne.logging.getLogger") as mock_logger:
            logger_instance = Mock()
            mock_logger.return_value = logger_instance

            # Should complete successfully with default config
            plot_simulated_data(args)

            # Verify warning was logged about missing config
            logger_instance.warning.assert_called()
            # Verify core was created with default config
            mock_core_class.assert_called_once()


class TestMainIntegration:
    """Integration tests for the main function."""

    def test_main_successful_execution(self, tmp_path):
        """Test complete main function execution."""
        # Create temporary config file
        config_data = {
            "analysis_settings": {"analysis_mode": "static_isotropic"},
            "data_settings": {"data_file": "test.h5"},
            "parameters": {"D_parallel": 1.0},
        }
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(config_data, temp_file)
        temp_file.flush()
        config_path = temp_file.name

        with patch("sys.argv", ["run_homodyne.py", "--config", config_path]):
            with patch("homodyne.run_homodyne.run_analysis") as mock_run_analysis:
                with patch("homodyne.run_homodyne.setup_logging"):
                    with patch("sys.exit") as mock_exit:
                        main()

                        # Verify successful exit
                        mock_exit.assert_called_with(0)
                        # Verify run_analysis was called
                        mock_run_analysis.assert_called_once()

    def test_main_plot_simulated_data_mode(self, tmp_path):
        """Test main function in plot-simulated-data mode."""
        # Create temporary config file
        config_data = {
            "analysis_settings": {"analysis_mode": "static_isotropic"},
            "parameters": {"D_parallel": 1.0},
        }
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(config_data, temp_file)
        temp_file.flush()
        config_path = temp_file.name

        with patch(
            "sys.argv",
            ["run_homodyne.py", "--plot-simulated-data", "--config", config_path],
        ):
            with patch("homodyne.run_homodyne.plot_simulated_data") as mock_plot:
                with patch("homodyne.run_homodyne.setup_logging"):
                    with patch("homodyne.run_homodyne.print_banner"):
                        with patch("homodyne.run_homodyne.logging.getLogger"):
                            with pytest.raises(SystemExit) as exc_info:
                                main()

                            # Verify successful exit code
                            assert exc_info.value.code == 0
                            # Verify plot function was called
                            mock_plot.assert_called_once()

    def test_main_analysis_failure(self, tmp_path):
        """Test main function with analysis failure."""
        # Create temporary config file
        config_data = {
            "analysis_settings": {"analysis_mode": "static_isotropic"},
            "data_settings": {"data_file": "test.h5"},
            "parameters": {"D_parallel": 1.0},
        }
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(config_data, temp_file)
        temp_file.flush()
        config_path = temp_file.name

        with patch("sys.argv", ["run_homodyne.py", "--config", config_path]):
            with patch(
                "homodyne.run_homodyne.run_analysis",
                side_effect=RuntimeError("Analysis failed"),
            ):
                with patch("homodyne.run_homodyne.setup_logging"):
                    with patch("homodyne.run_homodyne.print_banner"):
                        with patch("homodyne.run_homodyne.logging.getLogger"):
                            with patch("sys.exit") as mock_exit:
                                main()

                                # Verify failure exit
                                mock_exit.assert_called_with(1)

    def test_main_system_exit_preservation(self, tmp_path):
        """Test that SystemExit exceptions are properly preserved."""
        # Create temporary config file
        config_data = {
            "analysis_settings": {"analysis_mode": "static_isotropic"},
            "data_settings": {"data_file": "test.h5"},
            "parameters": {"D_parallel": 1.0},
        }
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(config_data, temp_file)
        temp_file.flush()
        config_path = temp_file.name

        with patch("sys.argv", ["run_homodyne.py", "--config", config_path]):
            with patch(
                "homodyne.run_homodyne.run_analysis", side_effect=SystemExit(42)
            ):
                with patch("homodyne.run_homodyne.setup_logging"):
                    with patch("homodyne.run_homodyne.print_banner"):
                        with patch("homodyne.run_homodyne.logging.getLogger"):
                            with pytest.raises(SystemExit) as exc_info:
                                main()

                            # Verify original exit code is preserved
                            assert exc_info.value.code == 42


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_method_argument(self):
        """Test handling of invalid method argument."""
        with patch("sys.argv", ["run_homodyne.py", "--method", "invalid"]):
            with patch("sys.stderr"):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 2  # argparse error

    def test_mutually_exclusive_mode_arguments(self):
        """Test handling of mutually exclusive mode arguments."""
        with patch(
            "sys.argv", ["run_homodyne.py", "--static-isotropic", "--laminar-flow"]
        ):
            with patch("sys.stderr"):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 2  # argparse error

    def test_missing_required_dependencies(self):
        """Test handling when required dependencies are missing."""
        with patch("sys.argv", ["run_homodyne.py"]):
            with patch("homodyne.run_homodyne.HomodyneAnalysisCore", None):
                with patch("homodyne.run_homodyne.setup_logging"):
                    with patch("homodyne.run_homodyne.print_banner"):
                        with patch("sys.exit") as mock_exit:
                            main()

                            # Should exit with failure code
                            mock_exit.assert_called_with(1)


if __name__ == "__main__":
    pytest.main([__file__])
