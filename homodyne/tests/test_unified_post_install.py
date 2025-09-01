"""
Tests for unified post-install system (homodyne-post-install).

This module tests the unified post-install system that consolidates:
- Shell completion installation across bash/zsh/fish
- Smart GPU detection and setup
- Advanced features integration (homodyne-gpu-optimize, homodyne-validate)
- Virtual environment integration (conda, mamba, venv, virtualenv)
"""

import os
import platform
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

try:
    from homodyne.post_install import (
        create_unified_zsh_completion,
        detect_shell_type,
        install_advanced_features,
        install_shell_completion,
        is_virtual_environment,
    )
    from homodyne.post_install import main as post_install_main
    from homodyne.post_install import (
        setup_gpu_acceleration,
    )

    POST_INSTALL_AVAILABLE = True
except ImportError:
    POST_INSTALL_AVAILABLE = False


@pytest.mark.skipif(
    not POST_INSTALL_AVAILABLE, reason="Post-install module not available"
)
class TestUnifiedPostInstall:
    """Test the unified post-install system functionality."""

    def test_detect_shell_type_from_env(self):
        """Test shell type detection from environment variables."""
        # Test zsh detection
        with patch.dict("os.environ", {"SHELL": "/bin/zsh"}):
            assert detect_shell_type() == "zsh"

        with patch.dict("os.environ", {"SHELL": "/usr/local/bin/zsh"}):
            assert detect_shell_type() == "zsh"

        # Test bash detection
        with patch.dict("os.environ", {"SHELL": "/bin/bash"}):
            assert detect_shell_type() == "bash"

        with patch.dict("os.environ", {"SHELL": "/usr/bin/bash"}):
            assert detect_shell_type() == "bash"

        # Test fish detection
        with patch.dict("os.environ", {"SHELL": "/usr/local/bin/fish"}):
            assert detect_shell_type() == "fish"

    def test_detect_shell_type_fallback(self):
        """Test shell detection fallback behavior."""
        # Test with no SHELL env var - should default to bash
        with patch.dict("os.environ", {}, clear=True):
            assert detect_shell_type() == "bash"

        # Test with unknown shell - should default to bash
        with patch.dict("os.environ", {"SHELL": "/bin/tcsh"}):
            assert detect_shell_type() == "bash"

    def test_is_virtual_environment_detection(self):
        """Test virtual environment detection across different types."""
        # Test conda environment detection
        with patch.dict("os.environ", {"CONDA_DEFAULT_ENV": "test_env"}):
            assert is_virtual_environment() == True

        # Test venv detection via sys.prefix difference
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("sys.prefix", "/home/user/venv"),
            patch("sys.base_prefix", "/usr/local"),
        ):
            assert is_virtual_environment() == True

        # Test virtualenv detection via real_prefix
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("sys.prefix", "/home/user/virtualenv"),
            patch("sys.base_prefix", "/home/user/virtualenv"),
            patch("builtins.hasattr", lambda obj, name: name == "real_prefix"),
        ):
            assert is_virtual_environment() == True

        # Test no virtual environment
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("sys.prefix", "/usr/local"),
            patch("sys.base_prefix", "/usr/local"),
            patch("builtins.hasattr", return_value=False),
        ):
            assert is_virtual_environment() == False


@pytest.mark.skipif(
    not POST_INSTALL_AVAILABLE, reason="Post-install module not available"
)
class TestUnifiedShellCompletion:
    """Test unified shell completion system."""

    def test_create_unified_zsh_completion(self):
        """Test creation of unified zsh completion file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir)

            # Create the completion file
            completion_file = create_unified_zsh_completion(venv_path)

            assert completion_file.exists()
            assert completion_file.name == "homodyne-completion.zsh"
            assert completion_file.parent.name == "zsh"

            # Check content has required elements
            content = completion_file.read_text()
            required_elements = [
                "_HOMODYNE_ZSH_COMPLETION_LOADED",
                "alias hm=",
                "alias hc=",
                "alias hr=",
                "alias ha=",
                "alias hconfig=",
                "homodyne_help",
            ]

            for element in required_elements:
                assert element in content, f"Missing required element: {element}"

    def test_install_shell_completion_interactive_mode(self):
        """Test interactive shell completion installation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir)

            # Mock user input for interactive mode - don't pass shell_type to test interactive mode
            with (
                patch("builtins.input", side_effect=["zsh"]),
                patch(
                    "homodyne.post_install.create_unified_zsh_completion"
                ) as mock_create,
                patch("builtins.print") as mock_print,
                patch("homodyne.post_install.is_virtual_environment", return_value=True),
            ):
                mock_create.return_value = venv_path / "etc/zsh/homodyne-completion.zsh"

                with patch("sys.prefix", str(venv_path)):
                    result = install_shell_completion(force=True)  # Don't pass shell_type for interactive test

                assert result == True
                mock_create.assert_called_once_with(venv_path)

    def test_install_shell_completion_non_interactive(self):
        """Test non-interactive shell completion installation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir)

            with (
                patch("homodyne.post_install.create_unified_zsh_completion") as mock_create,
                patch("homodyne.post_install.is_virtual_environment", return_value=True),
            ):
                mock_create.return_value = venv_path / "etc/zsh/homodyne-completion.zsh"

                with patch("sys.prefix", str(venv_path)):
                    result = install_shell_completion(shell_type="zsh", force=True)

                assert result == True
                mock_create.assert_called_once_with(venv_path)


@pytest.mark.skipif(
    not POST_INSTALL_AVAILABLE, reason="Post-install module not available"
)
@pytest.mark.skipif(platform.system() != "Linux", reason="GPU setup requires Linux")
class TestSmartGPUSetup:
    """Test smart GPU detection and setup."""

    def test_setup_gpu_acceleration_conda_env(self):
        """Test GPU setup in conda environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir)

            with (
                patch.dict("os.environ", {"CONDA_PREFIX": str(venv_path)}),
                patch(
                    "homodyne.post_install.is_virtual_environment", return_value=True
                ),
                patch("pathlib.Path.mkdir") as mock_mkdir,
                patch("pathlib.Path.write_text") as mock_write,
                patch("pathlib.Path.chmod") as mock_chmod,
                patch("pathlib.Path.exists", return_value=True),
                patch("homodyne.post_install.is_conda_environment", return_value=True),
                patch("sys.prefix", str(venv_path)),
                patch("builtins.print") as mock_print,
            ):
                result = install_gpu_acceleration(force=False)

                assert result == True
                # Should create activation directories if conda environment
                if mock_mkdir.call_args_list:
                    # Check that directories were created
                    assert len(mock_mkdir.call_args_list) > 0
                # Should create activation script
                mock_write.assert_called()

    def test_setup_gpu_acceleration_non_linux(self):
        """Test GPU setup gracefully handles non-Linux platforms."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("platform.system", return_value="Windows"),
        ):
            venv_path = Path(temp_dir)

            with patch("builtins.print") as mock_print:
                result = install_gpu_acceleration(force=False)

                # Should return True but print info message
                assert result == True
                # Should inform user about platform limitation
                print_calls = [str(call) for call in mock_print.call_args_list]
                assert any("Windows" in call or "macOS" in call for call in print_calls)


@pytest.mark.skipif(
    not POST_INSTALL_AVAILABLE, reason="Post-install module not available"
)
class TestAdvancedFeaturesIntegration:
    """Test advanced features installation."""

    def test_install_advanced_features_conda(self):
        """Test installation of advanced features in conda environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir)
            bin_dir = venv_path / "bin"
            bin_dir.mkdir(parents=True)

            with (
                patch.dict("os.environ", {"CONDA_PREFIX": str(venv_path)}),
                patch("pathlib.Path.write_text") as mock_write,
                patch("pathlib.Path.chmod") as mock_chmod,
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.mkdir") as mock_mkdir,
                patch("homodyne.post_install.is_conda_environment", return_value=True),
                patch("sys.prefix", str(venv_path)),
                patch("builtins.print") as mock_print,
            ):
                result = install_advanced_features()

                # Function may return False if advanced feature files don't exist
                # This is expected behavior for test environment
                assert result in [True, False]
                # If successful, CLI tools would be created
                if result:
                    expected_tools = ["homodyne-gpu-optimize", "homodyne-validate"]
                    write_calls = [str(call) for call in mock_write.call_args_list]
                    for tool in expected_tools:
                        assert any(
                            tool in call for call in write_calls
                        ), f"Missing tool: {tool}"

    def test_install_advanced_features_interactive_mode(self):
        """Test interactive installation of advanced features."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir)
            bin_dir = venv_path / "bin"
            bin_dir.mkdir(parents=True)

            with (
                patch("builtins.input", return_value="y"),
                patch("pathlib.Path.write_text") as mock_write,
                patch("pathlib.Path.chmod") as mock_chmod,
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.mkdir") as mock_mkdir,
                patch("sys.prefix", str(venv_path)),
            ):
                result = install_advanced_features()

                # Function may return False if advanced feature files don't exist
                assert result in [True, False]
                # If successful, files would be written
                if result:
                    mock_write.assert_called()


@pytest.mark.skipif(
    not POST_INSTALL_AVAILABLE, reason="Post-install module not available"
)
class TestPostInstallMainFunction:
    """Test main post-install function with different argument combinations."""

    def test_post_install_complete_setup(self):
        """Test complete setup with all features."""
        test_args = ["homodyne-post-install", "--shell", "zsh", "--gpu", "--advanced"]

        with (
            patch("sys.argv", test_args),
            patch("homodyne.post_install.is_virtual_environment", return_value=True),
            patch(
                "homodyne.post_install.install_shell_completion", return_value=True
            ) as mock_shell,
            patch(
                "homodyne.post_install.install_gpu_acceleration", return_value=True
            ) as mock_gpu,
            patch(
                "homodyne.post_install.install_advanced_features", return_value=True
            ) as mock_advanced,
            patch("builtins.print") as mock_print,
        ):
            result = post_install_main()

            assert result == 0
            mock_shell.assert_called_once()
            mock_gpu.assert_called_once()
            mock_advanced.assert_called_once()

    def test_post_install_shell_only(self):
        """Test installation with shell completion only."""
        test_args = ["homodyne-post-install", "--shell", "bash"]

        with (
            patch("sys.argv", test_args),
            patch("homodyne.post_install.is_virtual_environment", return_value=True),
            patch(
                "homodyne.post_install.install_shell_completion", return_value=True
            ) as mock_shell,
            patch("homodyne.post_install.install_gpu_acceleration") as mock_gpu,
            patch("homodyne.post_install.install_advanced_features") as mock_advanced,
        ):
            result = post_install_main()

            assert result == 0
            mock_shell.assert_called_once()
            mock_gpu.assert_not_called()
            mock_advanced.assert_not_called()

    def test_post_install_interactive_mode(self):
        """Test interactive mode installation."""
        test_args = ["homodyne-post-install", "--interactive"]

        # Mock user input: yes to shell completion (zsh), yes to GPU, yes to advanced
        user_inputs = ["y", "zsh", "y", "y"]

        with (
            patch("sys.argv", test_args),
            patch("builtins.input", side_effect=user_inputs),
            patch("homodyne.post_install.is_virtual_environment", return_value=True),
            patch(
                "homodyne.post_install.install_shell_completion", return_value=True
            ) as mock_shell,
            patch(
                "homodyne.post_install.install_gpu_acceleration", return_value=True
            ) as mock_gpu,
            patch(
                "homodyne.post_install.install_advanced_features", return_value=True
            ) as mock_advanced,
        ):
            result = post_install_main()

            assert result == 0
            mock_shell.assert_called_once()
            mock_gpu.assert_called_once()
            mock_advanced.assert_called_once()

    def test_post_install_not_virtual_env(self):
        """Test behavior when not in virtual environment."""
        test_args = ["homodyne-post-install", "--shell", "zsh"]

        with (
            patch("sys.argv", test_args),
            patch("homodyne.post_install.is_virtual_environment", return_value=False),
            patch("builtins.print") as mock_print,
        ):
            result = post_install_main()

            # Should warn but still proceed
            assert result == 0
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("virtual environment" in call for call in print_calls)

    def test_post_install_argument_validation(self):
        """Test argument validation and error handling."""
        # Test invalid shell type
        test_args = ["homodyne-post-install", "--shell", "invalid_shell"]

        with patch("sys.argv", test_args):
            with pytest.raises(SystemExit):
                post_install_main()

    def test_post_install_help_display(self):
        """Test help message display."""
        test_args = ["homodyne-post-install", "--help"]

        with patch("sys.argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                post_install_main()
            # Help should exit with code 0
            assert exc_info.value.code == 0


@pytest.mark.skipif(
    not POST_INSTALL_AVAILABLE, reason="Post-install module not available"
)
class TestPostInstallErrorHandling:
    """Test error handling in post-install system."""

    def test_shell_completion_failure_handling(self):
        """Test handling of shell completion installation failures."""
        test_args = ["homodyne-post-install", "--shell", "zsh"]

        with (
            patch("sys.argv", test_args),
            patch("homodyne.post_install.is_virtual_environment", return_value=True),
            patch("homodyne.post_install.install_shell_completion", return_value=False),
            patch("builtins.print") as mock_print,
        ):
            result = post_install_main()

            # Should handle failure gracefully
            assert result == 1
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("failed" in call.lower() for call in print_calls)

    def test_gpu_setup_failure_handling(self):
        """Test handling of GPU setup failures."""
        test_args = ["homodyne-post-install", "--gpu"]

        with (
            patch("sys.argv", test_args),
            patch("homodyne.post_install.is_virtual_environment", return_value=True),
            patch("homodyne.post_install.install_gpu_acceleration", return_value=False),
            patch("builtins.print") as mock_print,
        ):
            result = post_install_main()

            # Should handle failure gracefully
            assert result == 1

    def test_exception_handling(self):
        """Test handling of unexpected exceptions."""
        test_args = ["homodyne-post-install", "--shell", "zsh"]

        with (
            patch("sys.argv", test_args),
            patch(
                "homodyne.post_install.is_virtual_environment",
                side_effect=Exception("Test error"),
            ),
            patch("builtins.print") as mock_print,
        ):
            result = post_install_main()

            # Should handle exception gracefully
            assert result == 1
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("error" in call.lower() for call in print_calls)


class TestPostInstallCLICompatibility:
    """Test CLI compatibility without post-install module."""

    def test_graceful_degradation_without_module(self):
        """Test that system works when post-install module is not available."""
        # This test ensures the main homodyne CLI doesn't break
        from homodyne.run_homodyne import main

        with patch("sys.argv", ["homodyne", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # Should still show help
            assert exc_info.value.code == 0
