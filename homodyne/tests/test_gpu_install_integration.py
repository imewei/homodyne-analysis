"""
Tests for GPU installation and conda environment integration.
"""

import platform
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.skipif(
    platform.system() != "Linux", reason="GPU installation requires Linux"
)
class TestGPUInstallIntegration:
    """Test GPU installation script and conda integration."""

    def test_create_gpu_activation_script(self):
        """Test GPU activation script creation."""
        from scripts.install_gpu_autoload import create_gpu_activation_script

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            script_path = create_gpu_activation_script(config_dir)

            assert script_path.exists()
            assert script_path.is_file()
            assert script_path.stat().st_mode & 0o755  # Should be executable

            # Check script content
            content = script_path.read_text()
            assert "homodyne_gpu_activate" in content
            assert "CUDA_ROOT" in content
            assert "JAX_PLATFORMS" in content

    def test_create_config_script(self):
        """Test configuration script creation."""
        from scripts.install_gpu_autoload import create_config_script

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            script_path = create_config_script(config_dir)

            assert script_path.exists()
            assert script_path.is_file()

            # Check script content
            content = script_path.read_text()
            assert "CONDA_PREFIX" in content
            assert "homodyne_config.sh" in content
            assert "homodyne_gpu_status" in content

    def test_copy_completion_script(self):
        """Test completion script creation."""
        from scripts.install_gpu_autoload import copy_completion_script

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            script_path = copy_completion_script(config_dir)

            assert script_path is not None
            assert script_path.exists()
            assert script_path.is_file()

            # Check script content
            content = script_path.read_text()
            assert "homodyne_gpu_activate" in content
            assert "alias hm=" in content
            assert "alias hga=" in content

    def test_create_conda_activation_scripts(self):
        """Test conda activation/deactivation script creation."""
        from scripts.install_gpu_autoload import (
            create_conda_activate_script,
            create_conda_deactivate_script,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            conda_dir = Path(temp_dir) / "conda"
            conda_dir.mkdir(parents=True, exist_ok=True)

            # Test activation and deactivation scripts (mock sys.prefix to use our temp directory)
            with patch("sys.prefix", str(temp_dir)):
                activate_path = create_conda_activate_script(config_dir)
                assert activate_path.exists()
                content = activate_path.read_text()
                assert str(config_dir) in content
                assert "homodyne_config.sh" in content

                # Test deactivation script
                deactivate_path = create_conda_deactivate_script(config_dir)
                assert deactivate_path.exists()
                content = deactivate_path.read_text()
                assert "HOMODYNE_GPU_ACTIVATED" in content

    @patch("platform.system")
    @patch("pathlib.Path.exists")
    @patch("sys.argv", ["install_gpu_autoload.py"])
    def test_main_installation_linux_conda(self, mock_exists, mock_platform):
        """Test main installation function in Linux conda environment."""
        from scripts.install_gpu_autoload import main

        mock_platform.return_value = "Linux"
        mock_exists.return_value = False  # Directories don't exist yet

        with (
            patch.dict("os.environ", {"CONDA_PREFIX": "/test/conda/env"}),
            patch("scripts.install_gpu_autoload.get_shell") as mock_shell,
            patch(
                "scripts.install_gpu_autoload.create_gpu_activation_script"
            ) as mock_gpu,
            patch(
                "scripts.install_gpu_autoload.copy_completion_script"
            ) as mock_completion,
            patch("scripts.install_gpu_autoload.create_config_script") as mock_config,
            patch(
                "scripts.install_gpu_autoload.create_conda_activation_scripts"
            ) as mock_conda_scripts,
            patch("pathlib.Path.mkdir") as mock_mkdir,
            patch("builtins.print") as mock_print,
        ):
            mock_shell.return_value = "zsh"
            mock_gpu.return_value = Path("/test/gpu.sh")
            mock_completion.return_value = Path("/test/completion.zsh")
            mock_config.return_value = Path("/test/config.sh")
            mock_conda_scripts.return_value = (
                Path("/test/activate.sh"),
                Path("/test/deactivate.sh"),
            )

            result = main()

            assert result == 0
            mock_gpu.assert_called_once()
            mock_completion.assert_called_once()
            mock_config.assert_called_once()
            mock_conda_scripts.assert_called_once()

    @patch("platform.system")
    @patch("sys.argv", ["install_gpu_autoload.py"])
    def test_main_installation_non_linux(self, mock_platform):
        """Test main installation skips on non-Linux platforms."""
        from scripts.install_gpu_autoload import main

        mock_platform.return_value = "Windows"

        with patch("builtins.print") as mock_print:
            result = main()

            assert result == 0
            # Should print platform info message
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("Linux" in call for call in print_calls)

    @patch("platform.system")
    @patch("sys.argv", ["install_gpu_autoload.py"])
    def test_main_installation_no_conda(self, mock_platform):
        """Test main installation without conda environment."""
        from scripts.install_gpu_autoload import main

        mock_platform.return_value = "Linux"

        with (
            patch.dict("os.environ", {}, clear=True),  # No CONDA_PREFIX
            patch("builtins.print") as mock_print,
        ):
            result = main()

            assert result == 0
            # Should print virtual environment info message
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("virtual environment" in call for call in print_calls)


@pytest.mark.skipif(
    platform.system() != "Linux", reason="GPU functionality requires Linux"
)
class TestCondaIntegrationHelpers:
    """Test helper functions for conda integration."""

    def test_get_shell(self):
        """Test shell detection."""
        from scripts.install_gpu_autoload import get_shell

        with patch.dict("os.environ", {"SHELL": "/bin/zsh"}):
            shell = get_shell()
            assert shell == "zsh"

        with patch.dict("os.environ", {"SHELL": "/usr/bin/bash"}):
            shell = get_shell()
            assert shell == "bash"

    def test_get_shell_rc_file(self):
        """Test shell RC file detection."""
        from scripts.install_gpu_autoload import get_shell_rc_file

        with (
            patch("pathlib.Path.home") as mock_home,
            patch("pathlib.Path.exists") as mock_exists,
        ):
            mock_home_path = MagicMock()
            mock_home.return_value = mock_home_path
            mock_exists.return_value = True

            rc_file = get_shell_rc_file()
            assert rc_file is not None

    def test_is_virtual_environment(self):
        """Test virtual environment detection."""
        from scripts.install_gpu_autoload import main

        # Test conda environment
        with patch.dict("os.environ", {"CONDA_DEFAULT_ENV": "test"}):
            # Should detect as virtual environment
            with (
                patch("scripts.install_gpu_autoload.get_shell") as mock_shell,
                patch("platform.system") as mock_platform,
                patch("builtins.print") as mock_print,
                patch("sys.argv", ["install_gpu_autoload.py"]),
            ):
                mock_shell.return_value = "zsh"
                mock_platform.return_value = "Linux"

                result = main()
                assert result == 0


class TestScriptTemplates:
    """Test script template generation."""

    def test_gpu_activation_script_template(self):
        """Test GPU activation script has required components."""
        from scripts.install_gpu_autoload import create_gpu_activation_script

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            script_path = create_gpu_activation_script(config_dir)
            content = script_path.read_text()

            # Check required function and variables
            required_elements = [
                "homodyne_gpu_activate",
                "CUDA_ROOT",
                "CUDA_HOME",
                "LD_LIBRARY_PATH",
                "XLA_FLAGS",
                "JAX_PLATFORMS",
                "HOMODYNE_GPU_ACTIVATED",
                "alias homodyne-gpu-activate",
            ]

            for element in required_elements:
                assert element in content, f"Missing required element: {element}"

    def test_config_script_template(self):
        """Test configuration script has required components."""
        from scripts.install_gpu_autoload import create_config_script

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            script_path = create_config_script(config_dir)
            content = script_path.read_text()

            # Check required components
            required_elements = [
                "CONDA_PREFIX",
                "gpu_activation.sh",
                "homodyne_completion_bypass.zsh",
                "homodyne_gpu_status",
                "SHELL",
                "ZSH_VERSION",
            ]

            for element in required_elements:
                assert element in content, f"Missing required element: {element}"

    def test_completion_script_template(self):
        """Test completion script has required components."""
        from scripts.install_gpu_autoload import copy_completion_script

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            script_path = copy_completion_script(config_dir)
            content = script_path.read_text()

            # Check required aliases and functions
            required_elements = [
                "alias hm=",
                "alias hga=",
                "alias hconfig=",
                "homodyne_gpu_activate",
                "homodyne_help",
                "_homodyne_complete",
                "compdef",
            ]

            for element in required_elements:
                assert element in content, f"Missing required element: {element}"
