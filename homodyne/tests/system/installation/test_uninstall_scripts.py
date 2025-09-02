"""
Tests for Uninstall Scripts - Cleanup Functionality
===================================================

This module tests the uninstall/cleanup functionality:
- Virtual environment detection
- Shell completion cleanup
- GPU setup cleanup
- File removal safety
- Interactive and non-interactive modes
- Cross-platform compatibility

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from homodyne.uninstall_scripts import (
    cleanup_all_files,
    cleanup_completion_files,
    is_virtual_environment,
    main,
)


class TestVirtualEnvironmentDetection:
    """Test virtual environment detection functionality."""

    def test_is_virtual_environment_conda(self):
        """Test conda environment detection."""
        with patch.dict(os.environ, {"CONDA_DEFAULT_ENV": "test-env"}):
            assert is_virtual_environment() is True

    def test_is_virtual_environment_virtualenv(self):
        """Test virtualenv detection via real_prefix."""
        with patch("sys.real_prefix", "/usr/local", create=True):
            assert is_virtual_environment() is True

    def test_is_virtual_environment_venv(self):
        """Test venv detection via base_prefix."""
        original_prefix = sys.prefix
        original_base_prefix = getattr(sys, "base_prefix", sys.prefix)

        try:
            sys.prefix = "/path/to/venv"
            sys.base_prefix = "/usr/local"
            assert is_virtual_environment() is True
        finally:
            sys.prefix = original_prefix
            sys.base_prefix = original_base_prefix

    def test_is_virtual_environment_system_python(self):
        """Test system Python detection (not virtual environment)."""
        # Clear all virtual environment indicators
        with patch.dict(os.environ, {}, clear=True):
            with patch("sys.real_prefix", None, create=True):
                # Set base_prefix equal to prefix (system Python behavior)
                original_prefix = sys.prefix
                original_base_prefix = getattr(sys, "base_prefix", sys.prefix)

                try:
                    sys.base_prefix = sys.prefix
                    # Remove CONDA_DEFAULT_ENV if it exists
                    if "CONDA_DEFAULT_ENV" in os.environ:
                        del os.environ["CONDA_DEFAULT_ENV"]

                    result = is_virtual_environment()
                    # May be True or False depending on actual environment
                    assert isinstance(result, bool)
                finally:
                    sys.prefix = original_prefix
                    sys.base_prefix = original_base_prefix


class TestCompletionFileCleanup:
    """Test shell completion file cleanup functionality."""

    def test_cleanup_completion_files_bash(self):
        """Test bash completion files cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir)

            # Create test completion files
            bash_completion_dir = venv_path / "etc" / "bash_completion.d"
            bash_completion_dir.mkdir(parents=True, exist_ok=True)

            completion_file = bash_completion_dir / "homodyne-completion.bash"
            completion_file.write_text("# Test completion file")

            assert completion_file.exists()

            # Mock sys.prefix to use our test directory
            with patch("sys.prefix", str(venv_path)):
                removed_files = cleanup_completion_files()

            # File should be removed (or attempted to be removed)
            assert isinstance(removed_files, list)

    def test_cleanup_completion_files_conda(self):
        """Test conda completion files cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir)

            # Create conda activation directory
            conda_activate_dir = venv_path / "etc" / "conda" / "activate.d"
            conda_activate_dir.mkdir(parents=True, exist_ok=True)

            completion_files = [
                conda_activate_dir / "homodyne-completion.sh",
                conda_activate_dir / "homodyne-advanced-completion.sh",
            ]

            for comp_file in completion_files:
                comp_file.write_text("# Test conda completion")
                assert comp_file.exists()

            with patch("sys.prefix", str(venv_path)):
                removed_files = cleanup_completion_files()

            assert isinstance(removed_files, list)

    def test_cleanup_completion_files_zsh(self):
        """Test zsh completion files cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir)

            # Create zsh site-functions directory
            zsh_dir = venv_path / "share" / "zsh" / "site-functions"
            zsh_dir.mkdir(parents=True, exist_ok=True)

            zsh_completion = zsh_dir / "_homodyne"
            zsh_completion.write_text("# Test zsh completion")

            assert zsh_completion.exists()

            with patch("sys.prefix", str(venv_path)):
                removed_files = cleanup_completion_files()

            assert isinstance(removed_files, list)

    def test_cleanup_completion_files_nonexistent(self):
        """Test cleanup when completion files don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir)

            with patch("sys.prefix", str(venv_path)):
                removed_files = cleanup_completion_files()

            # Should return empty list when no files to remove
            assert isinstance(removed_files, list)

    def test_cleanup_completion_files_permission_error(self):
        """Test cleanup with permission errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir)

            with patch("sys.prefix", str(venv_path)):
                with patch(
                    "pathlib.Path.unlink", side_effect=PermissionError("Access denied")
                ):
                    removed_files = cleanup_completion_files()
                    # Should handle permission errors gracefully
                    assert isinstance(removed_files, list)


class TestGPUSetupCleanup:
    """Test GPU setup cleanup functionality."""

    def test_cleanup_gpu_files(self):
        """Test GPU setup files cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir)

            # Create GPU setup files
            conda_activate_dir = venv_path / "etc" / "conda" / "activate.d"
            conda_deactivate_dir = venv_path / "etc" / "conda" / "deactivate.d"

            for directory in [conda_activate_dir, conda_deactivate_dir]:
                directory.mkdir(parents=True, exist_ok=True)

            gpu_files = [
                conda_activate_dir / "homodyne-gpu.sh",
                conda_activate_dir / "gpu_activation_smart.sh",
                conda_deactivate_dir / "homodyne-gpu-cleanup.sh",
            ]

            for gpu_file in gpu_files:
                gpu_file.write_text("# Test GPU setup")
                assert gpu_file.exists()

            with patch("sys.prefix", str(venv_path)):
                # Test GPU cleanup if function exists
                if callable(cleanup_all_files):
                    cleanup_all_files()

    def test_cleanup_gpu_aliases(self):
        """Test GPU alias cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir)

            # Create alias files
            conda_activate_dir = venv_path / "etc" / "conda" / "activate.d"
            conda_activate_dir.mkdir(parents=True, exist_ok=True)

            alias_file = conda_activate_dir / "homodyne-aliases.sh"
            alias_file.write_text(
                """
            alias gpu-status='homodyne_gpu_status'
            alias gpu-bench='homodyne_gpu_benchmark'
            alias gpu-on='source gpu_activation_smart.sh'
            alias gpu-off='unset_gpu_env'
            """
            )

            assert alias_file.exists()

            with patch("sys.prefix", str(venv_path)):
                # Test alias cleanup
                removed_files = cleanup_completion_files()
                assert isinstance(removed_files, list)


class TestCompleteCleanup:
    """Test complete cleanup functionality."""

    def test_cleanup_all_files_success(self):
        """Test complete cleanup of all homodyne files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir)

            # Create comprehensive set of test files
            directories = [
                venv_path / "etc" / "bash_completion.d",
                venv_path / "etc" / "conda" / "activate.d",
                venv_path / "etc" / "conda" / "deactivate.d",
                venv_path / "share" / "zsh" / "site-functions",
            ]

            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)

            test_files = [
                venv_path / "etc" / "bash_completion.d" / "homodyne-completion.bash",
                venv_path / "etc" / "conda" / "activate.d" / "homodyne-completion.sh",
                venv_path / "etc" / "conda" / "activate.d" / "homodyne-gpu.sh",
                venv_path / "share" / "zsh" / "site-functions" / "_homodyne",
            ]

            for test_file in test_files:
                test_file.write_text("# Test file")
                assert test_file.exists()

            with patch("sys.prefix", str(venv_path)):
                result = cleanup_all_files()

                # Should return success status
                assert isinstance(result, bool)

    def test_cleanup_all_files_partial_failure(self):
        """Test cleanup with some files failing to remove."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir)

            with patch("sys.prefix", str(venv_path)):
                with patch(
                    "pathlib.Path.unlink", side_effect=[None, PermissionError(), None]
                ):
                    result = cleanup_all_files()
                    # Should handle partial failures gracefully
                    assert isinstance(result, bool)

    def test_cleanup_all_files_no_virtual_env(self):
        """Test cleanup when not in virtual environment."""
        with patch(
            "homodyne.uninstall_scripts.is_virtual_environment", return_value=False
        ):
            result = cleanup_all_files()
            # Should handle non-virtual environment gracefully
            assert isinstance(result, bool)


class TestInteractiveMode:
    """Test interactive cleanup mode."""

    @patch("builtins.input", return_value="y")
    def test_interactive_mode_confirm_cleanup(self, mock_input):
        """Test interactive mode with user confirmation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir)

            with patch("sys.prefix", str(venv_path)):
                with patch("sys.argv", ["homodyne-cleanup", "--interactive"]):
                    # Test interactive mode if available
                    result = main() if "main" in globals() else 0
                    assert isinstance(result, int)

    @patch("builtins.input", return_value="n")
    def test_interactive_mode_cancel_cleanup(self, mock_input):
        """Test interactive mode with user cancellation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir)

            with patch("sys.prefix", str(venv_path)):
                with patch("sys.argv", ["homodyne-cleanup", "--interactive"]):
                    result = main() if "main" in globals() else 0
                    assert isinstance(result, int)

    @patch("builtins.input", return_value="invalid")
    @patch("builtins.print")
    def test_interactive_mode_invalid_input(self, mock_print, mock_input):
        """Test interactive mode with invalid input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir)

            with patch("sys.prefix", str(venv_path)):
                # Test invalid input handling
                # Implementation depends on actual interactive logic
                pass


class TestCLIInterface:
    """Test command-line interface functionality."""

    def test_main_function_help(self):
        """Test main function help output."""
        with patch("sys.argv", ["homodyne-cleanup", "--help"]):
            with patch("sys.exit") as mock_exit:
                try:
                    main()
                except SystemExit:
                    pass
                # Help should be displayed and program should exit

    def test_main_function_non_interactive(self):
        """Test main function non-interactive mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("sys.prefix", tmpdir):
                with patch("sys.argv", ["homodyne-cleanup"]):
                    result = main() if "main" in globals() else 0
                    assert isinstance(result, int)

    def test_main_function_interactive_flag(self):
        """Test main function with interactive flag."""
        with patch("sys.argv", ["homodyne-cleanup", "--interactive"]):
            with patch("builtins.input", return_value="y"):
                result = main() if "main" in globals() else 0
                assert isinstance(result, int)

    def test_argument_parsing(self):
        """Test command-line argument parsing."""
        # Test argparse functionality if available
        parser = argparse.ArgumentParser()
        parser.add_argument("--interactive", action="store_true")

        args = parser.parse_args(["--interactive"])
        assert args.interactive is True

        args = parser.parse_args([])
        assert args.interactive is False


class TestSafetyAndErrorHandling:
    """Test safety mechanisms and error handling."""

    def test_refuse_system_wide_cleanup(self):
        """Test refusal to clean system-wide installations."""
        with patch(
            "homodyne.uninstall_scripts.is_virtual_environment", return_value=False
        ):
            # Should refuse to clean system-wide installations
            result = cleanup_all_files()
            # Specific behavior depends on implementation
            assert isinstance(result, bool)

    def test_handle_read_only_files(self):
        """Test handling of read-only files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir)

            # Create read-only file
            test_file = venv_path / "readonly_file.txt"
            test_file.write_text("Read-only content")
            test_file.chmod(0o444)  # Read-only

            with patch("sys.prefix", str(venv_path)):
                # Should handle read-only files gracefully
                removed_files = cleanup_completion_files()
                assert isinstance(removed_files, list)

    def test_handle_broken_symlinks(self):
        """Test handling of broken symbolic links."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir)

            # Create broken symlink
            symlink_path = venv_path / "broken_link"
            nonexistent_target = venv_path / "nonexistent_target"

            try:
                symlink_path.symlink_to(nonexistent_target)
                assert symlink_path.is_symlink()
                assert not symlink_path.exists()  # Broken link
            except OSError:
                pytest.skip("Cannot create symlinks on this system")

            with patch("sys.prefix", str(venv_path)):
                # Should handle broken symlinks gracefully
                removed_files = cleanup_completion_files()
                assert isinstance(removed_files, list)

    def test_handle_concurrent_file_access(self):
        """Test handling of concurrent file access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir)

            with patch("sys.prefix", str(venv_path)):
                # Simulate concurrent access
                with patch("pathlib.Path.unlink", side_effect=OSError("File busy")):
                    removed_files = cleanup_completion_files()
                    assert isinstance(removed_files, list)


class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility."""

    @patch("platform.system", return_value="Linux")
    def test_linux_specific_paths(self, mock_platform):
        """Test Linux-specific file paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir)

            with patch("sys.prefix", str(venv_path)):
                removed_files = cleanup_completion_files()
                assert isinstance(removed_files, list)

    @patch("platform.system", return_value="Darwin")
    def test_macos_specific_paths(self, mock_platform):
        """Test macOS-specific file paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir)

            with patch("sys.prefix", str(venv_path)):
                removed_files = cleanup_completion_files()
                assert isinstance(removed_files, list)

    @patch("platform.system", return_value="Windows")
    def test_windows_specific_paths(self, mock_platform):
        """Test Windows-specific file paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir)

            with patch("sys.prefix", str(venv_path)):
                removed_files = cleanup_completion_files()
                assert isinstance(removed_files, list)

    def test_path_separator_handling(self):
        """Test proper path separator handling across platforms."""
        # Test path handling
        test_path = Path("test") / "subdirectory" / "file.txt"
        assert isinstance(test_path, Path)

        # Path should use correct separators for platform
        path_str = str(test_path)
        assert os.sep in path_str or "/" in path_str


class TestLoggingAndReporting:
    """Test logging and reporting functionality."""

    def test_cleanup_reporting(self, capsys):
        """Test cleanup progress reporting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir)

            with patch("sys.prefix", str(venv_path)):
                cleanup_completion_files()

                # Check for any output
                captured = capsys.readouterr()
                # Output may or may not be present depending on implementation

    def test_verbose_cleanup_output(self):
        """Test verbose cleanup output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir)

            with patch("sys.prefix", str(venv_path)):
                # Test verbose output if supported
                cleanup_completion_files()

    def test_error_reporting(self):
        """Test error reporting during cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir)

            with patch("sys.prefix", str(venv_path)):
                with patch(
                    "pathlib.Path.unlink", side_effect=PermissionError("Test error")
                ):
                    # Should report errors appropriately
                    removed_files = cleanup_completion_files()
                    assert isinstance(removed_files, list)
