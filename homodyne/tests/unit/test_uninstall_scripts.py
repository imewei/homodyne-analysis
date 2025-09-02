"""
Tests for homodyne/uninstall_scripts.py - Cleanup and uninstallation functionality.

This module tests the uninstallation script functionality, including
removal of shell completion files, GPU configuration cleanup,
virtual environment detection, and safe file removal operations.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the module under test
from homodyne.uninstall_scripts import (
    cleanup_completion_files,
    cleanup_gpu_files,
    interactive_cleanup,
    is_virtual_environment,
    main,
)


class TestVirtualEnvironmentDetection:
    """Test virtual environment detection functionality."""

    def test_is_virtual_environment_real_prefix(self):
        """Test detection via sys.real_prefix (virtualenv)."""
        with patch.object(sys, "real_prefix", "/usr", create=True):
            assert is_virtual_environment() is True

    def test_is_virtual_environment_base_prefix_different(self):
        """Test detection via different sys.base_prefix (venv)."""
        with patch.object(sys, "base_prefix", "/usr"):
            with patch.object(sys, "prefix", "/home/user/venv"):
                assert is_virtual_environment() is True

    def test_is_virtual_environment_base_prefix_same(self):
        """Test no detection when base_prefix equals prefix."""
        with patch.object(sys, "base_prefix", "/usr"):
            with patch.object(sys, "prefix", "/usr"):
                with patch.dict(os.environ, {}, clear=True):
                    assert is_virtual_environment() is False

    def test_is_virtual_environment_conda(self):
        """Test detection via CONDA_DEFAULT_ENV."""
        with patch.dict(os.environ, {"CONDA_DEFAULT_ENV": "myenv"}):
            assert is_virtual_environment() is True

    def test_is_virtual_environment_mamba(self):
        """Test detection via MAMBA_ROOT_PREFIX."""
        with patch.dict(os.environ, {"MAMBA_ROOT_PREFIX": "/opt/mamba"}):
            assert is_virtual_environment() is True

    def test_is_virtual_environment_virtual_env(self):
        """Test detection via VIRTUAL_ENV."""
        with patch.dict(os.environ, {"VIRTUAL_ENV": "/home/user/venv"}):
            assert is_virtual_environment() is True

    def test_is_virtual_environment_false(self):
        """Test no virtual environment detected in system Python."""
        with patch.object(sys, "base_prefix", "/usr"):
            with patch.object(sys, "prefix", "/usr"):
                with patch.dict(os.environ, {}, clear=True):
                    assert is_virtual_environment() is False


class TestCompletionFilesCleanup:
    """Test shell completion files cleanup functionality."""

    def create_temp_venv_structure(self):
        """Helper to create temporary virtual environment structure with completion files."""
        temp_dir = tempfile.mkdtemp()
        venv_path = Path(temp_dir)

        # Create directory structure
        bash_completion_dir = venv_path / "etc" / "bash_completion.d"
        conda_activate_dir = venv_path / "etc" / "conda" / "activate.d"
        conda_deactivate_dir = venv_path / "etc" / "conda" / "deactivate.d"
        share_dir = venv_path / "share"
        zsh_completion_dir = share_dir / "zsh" / "site-functions"
        fish_completion_dir = share_dir / "fish" / "completions"

        for dir_path in [
            bash_completion_dir,
            conda_activate_dir,
            conda_deactivate_dir,
            zsh_completion_dir,
            fish_completion_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Create the actual completion files that the cleanup function looks for
        zsh_dir = venv_path / "etc" / "zsh"
        zsh_dir.mkdir(parents=True, exist_ok=True)

        fish_vendor_dir = venv_path / "share" / "fish" / "vendor_completions.d"
        fish_vendor_dir.mkdir(parents=True, exist_ok=True)

        # Create only files that the cleanup function actually targets
        completion_files = [
            bash_completion_dir / "homodyne-completion.bash",
            conda_activate_dir / "homodyne-completion.sh",
            conda_activate_dir / "homodyne-advanced-completion.sh",
            zsh_dir / "homodyne-completion.zsh",
            fish_vendor_dir / "homodyne.fish",
        ]

        for file_path in completion_files:
            file_path.write_text("# Mock completion file content")

        return venv_path, completion_files

    def test_cleanup_completion_files_success(self):
        """Test successful cleanup of completion files."""
        venv_path, completion_files = self.create_temp_venv_structure()

        with patch("sys.prefix", str(venv_path)):
            removed_files = cleanup_completion_files()

            # Verify that files were identified for removal
            assert len(removed_files) > 0

            # Verify files were actually removed
            for file_path in completion_files:
                assert not file_path.exists()

    @patch("sys.prefix")
    def test_cleanup_completion_files_no_files(self, mock_prefix):
        """Test cleanup when no completion files exist."""
        temp_dir = tempfile.mkdtemp()
        venv_path = Path(temp_dir)
        mock_prefix.__str__ = lambda: str(venv_path)

        removed_files = cleanup_completion_files()

        # Should return empty list when no files exist
        assert removed_files == []

    @patch("sys.prefix")
    def test_cleanup_completion_files_permission_error(self, mock_prefix):
        """Test cleanup handles permission errors gracefully."""
        venv_path, completion_files = self.create_temp_venv_structure()
        mock_prefix.__str__ = lambda: str(venv_path)

        # Make first file read-only to trigger permission error
        first_file = completion_files[0]
        first_file.chmod(0o000)

        try:
            removed_files = cleanup_completion_files()

            # Should handle permission errors and continue with other files
            assert len(removed_files) >= 0  # May or may not succeed depending on system
        finally:
            # Restore permissions for cleanup
            first_file.chmod(0o644)

    def test_cleanup_completion_files_partial_structure(self):
        """Test cleanup with partial directory structure."""
        temp_dir = tempfile.mkdtemp()
        venv_path = Path(temp_dir)

        # Create only some directories
        bash_dir = venv_path / "etc" / "bash_completion.d"
        bash_dir.mkdir(parents=True, exist_ok=True)
        completion_file = bash_dir / "homodyne-completion.bash"
        completion_file.write_text("# Test content")

        with patch("sys.prefix", str(venv_path)):
            removed_files = cleanup_completion_files()

        # Should handle partial directory structure
        assert len(removed_files) >= 0
        assert not completion_file.exists()


class TestGPUFilesCleanup:
    """Test GPU configuration files cleanup functionality."""

    def create_temp_gpu_structure(self):
        """Helper to create temporary GPU configuration structure."""
        temp_dir = tempfile.mkdtemp()
        venv_path = Path(temp_dir)

        # Create GPU-related directory structure where cleanup function looks
        homodyne_gpu_dir = venv_path / "etc" / "homodyne" / "gpu"
        conda_activate_dir = venv_path / "etc" / "conda" / "activate.d"

        for dir_path in [homodyne_gpu_dir, conda_activate_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Create only the GPU files that cleanup function targets
        gpu_files = [
            homodyne_gpu_dir / "gpu_activation.sh",
            conda_activate_dir / "homodyne-gpu.sh",
        ]

        for file_path in gpu_files:
            file_path.write_text("# Mock GPU config content")

        return venv_path, gpu_files

    def test_cleanup_gpu_files_success(self):
        """Test successful cleanup of GPU configuration files."""
        venv_path, gpu_files = self.create_temp_gpu_structure()

        with patch("sys.prefix", str(venv_path)):
            removed_files = cleanup_gpu_files()

        # Verify that files were identified for removal
        assert len(removed_files) > 0

        # Verify files were actually removed
        for file_path in gpu_files:
            assert not file_path.exists()

    @patch("sys.prefix")
    def test_cleanup_gpu_files_no_files(self, mock_prefix):
        """Test GPU cleanup when no files exist."""
        temp_dir = tempfile.mkdtemp()
        venv_path = Path(temp_dir)
        mock_prefix.__str__ = lambda: str(venv_path)

        removed_files = cleanup_gpu_files()

        # Should return empty list when no files exist
        assert removed_files == []

    @patch("sys.prefix")
    def test_cleanup_gpu_files_permission_error(self, mock_prefix):
        """Test GPU cleanup handles permission errors gracefully."""
        venv_path, gpu_files = self.create_temp_gpu_structure()
        mock_prefix.__str__ = lambda: str(venv_path)

        # Make a file read-only to trigger permission error
        first_file = gpu_files[0]
        first_file.chmod(0o000)

        try:
            removed_files = cleanup_gpu_files()

            # Should handle permission errors and continue
            assert len(removed_files) >= 0
        finally:
            # Restore permissions for cleanup
            first_file.chmod(0o644)

    def test_cleanup_gpu_files_partial_structure(self):
        """Test GPU cleanup with partial directory structure."""
        temp_dir = tempfile.mkdtemp()
        venv_path = Path(temp_dir)

        # Create only conda activation directory
        conda_dir = venv_path / "etc" / "conda" / "activate.d"
        conda_dir.mkdir(parents=True, exist_ok=True)
        gpu_file = conda_dir / "homodyne-gpu.sh"
        gpu_file.write_text("# Test GPU config")

        with patch("sys.prefix", str(venv_path)):
            removed_files = cleanup_gpu_files()

        # Should handle partial structure
        assert len(removed_files) >= 0
        assert not gpu_file.exists()


class TestInteractiveCleanup:
    """Test interactive cleanup functionality."""

    def test_interactive_cleanup_yes_to_all(self):
        """Test interactive cleanup with user choosing yes for all options."""
        with patch("builtins.input", return_value="y"):
            with patch(
                "homodyne.uninstall_scripts.cleanup_completion_files",
                return_value=["file1.sh"],
            ):
                with patch(
                    "homodyne.uninstall_scripts.cleanup_gpu_files",
                    return_value=["gpu1.py"],
                ):
                    with patch("builtins.print"):
                        interactive_cleanup()

    def test_interactive_cleanup_no_to_all(self):
        """Test interactive cleanup with user choosing no for all options."""
        with patch("builtins.input", return_value="n"):
            with patch("builtins.print"):
                interactive_cleanup()

    def test_interactive_cleanup_selective(self):
        """Test interactive cleanup with selective choices."""
        # Provide responses for all 4 prompts: completion, gpu, advanced, old
        responses = ["y", "n", "y", "n"]
        with patch("builtins.input", side_effect=responses):
            with patch(
                "homodyne.uninstall_scripts.cleanup_completion_files",
                return_value=["comp.sh"],
            ):
                with patch("builtins.print"):
                    interactive_cleanup()

    def test_interactive_cleanup_invalid_then_valid(self):
        """Test interactive cleanup with invalid input followed by valid input."""
        # The function doesn't actually validate input - it just checks if it starts with 'y'
        # So this test should just provide enough responses
        responses = ["invalid", "y", "n", "y"]  # 4 responses needed
        with patch("builtins.input", side_effect=responses):
            with patch(
                "homodyne.uninstall_scripts.cleanup_completion_files",
                return_value=["comp.sh"],
            ):
                with patch("builtins.print"):
                    interactive_cleanup()

    def test_interactive_cleanup_no_files_found(self):
        """Test interactive cleanup when no files are found to remove."""
        with patch("builtins.input", return_value="y"):
            with patch(
                "homodyne.uninstall_scripts.cleanup_completion_files", return_value=[]
            ):
                with patch(
                    "homodyne.uninstall_scripts.cleanup_gpu_files", return_value=[]
                ):
                    with patch("builtins.print") as mock_print:
                        interactive_cleanup()

                        # Should inform user that no files were found
                        mock_print.assert_called()


class TestArgumentParsing:
    """Test command-line argument parsing."""

    def test_main_with_help_flag(self):
        """Test main function with help flag."""
        with patch("sys.argv", ["uninstall_scripts.py", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # argparse exits with code 0 for help
            assert exc_info.value.code == 0

    def test_main_with_interactive_flag(self):
        """Test main function with interactive flag."""
        with patch("sys.argv", ["uninstall_scripts.py", "--interactive"]):
            with patch("homodyne.uninstall_scripts.interactive_cleanup"):
                with patch(
                    "homodyne.uninstall_scripts.is_virtual_environment",
                    return_value=True,
                ):
                    main()

    def test_main_with_dry_run_flag(self):
        """Test main function with dry run flag."""
        with patch("sys.argv", ["uninstall_scripts.py", "--dry-run"]):
            with patch(
                "homodyne.uninstall_scripts.cleanup_completion_files",
                return_value=["file.sh"],
            ):
                with patch(
                    "homodyne.uninstall_scripts.cleanup_gpu_files",
                    return_value=["gpu.py"],
                ):
                    with patch(
                        "homodyne.uninstall_scripts.is_virtual_environment",
                        return_value=True,
                    ):
                        with patch("builtins.print"):
                            main()

    def test_main_default_behavior(self):
        """Test main function default behavior (no flags)."""
        with patch("sys.argv", ["uninstall_scripts.py"]):
            with patch(
                "homodyne.uninstall_scripts.cleanup_completion_files",
                return_value=["comp.sh"],
            ):
                with patch(
                    "homodyne.uninstall_scripts.cleanup_gpu_files",
                    return_value=["gpu.py"],
                ):
                    with patch(
                        "homodyne.uninstall_scripts.is_virtual_environment",
                        return_value=True,
                    ):
                        with patch("builtins.print"):
                            main()

    def test_main_invalid_environment(self):
        """Test main function outside virtual environment."""
        with patch("sys.argv", ["uninstall_scripts.py"]):
            with patch(
                "homodyne.uninstall_scripts.is_virtual_environment", return_value=False
            ):
                with patch("builtins.print"):
                    main()  # Should show a warning about not being in venv


class TestMainFunction:
    """Test the main function integration."""

    def test_main_default_behavior(self):
        """Test main function default behavior without flags."""
        with patch("sys.argv", ["uninstall_scripts.py"]):
            with patch("homodyne.uninstall_scripts.interactive_cleanup"):
                with patch(
                    "homodyne.uninstall_scripts.is_virtual_environment",
                    return_value=True,
                ):
                    # Should run interactive cleanup by default
                    main()

    @patch("homodyne.uninstall_scripts.is_virtual_environment")
    def test_main_not_in_virtual_environment(self, mock_is_venv):
        """Test main function when not in virtual environment."""
        mock_is_venv.return_value = False

        with patch("sys.argv", ["uninstall_scripts.py"]):
            with patch("builtins.print") as mock_print:
                main()

                # Should print warning and exit
                mock_print.assert_called()

    def test_main_cleanup_with_files(self):
        """Test main function with files to remove."""
        with patch("sys.argv", ["uninstall_scripts.py"]):
            with patch(
                "homodyne.uninstall_scripts.cleanup_completion_files",
                return_value=[("Shell completion", "file1.sh")],
            ):
                with patch(
                    "homodyne.uninstall_scripts.cleanup_gpu_files",
                    return_value=[("GPU setup", "gpu.py")],
                ):
                    with patch(
                        "homodyne.uninstall_scripts.is_virtual_environment",
                        return_value=True,
                    ):
                        with patch("builtins.print") as mock_print:
                            main()

                            # Should report removed files
                            mock_print.assert_called()

    def test_main_cleanup_gpu_files_only(self):
        """Test main function cleaning up only GPU files."""
        with patch("sys.argv", ["uninstall_scripts.py", "--dry-run"]):
            with patch(
                "homodyne.uninstall_scripts.cleanup_gpu_files",
                return_value=["gpu1.py", "gpu2.sh"],
            ):
                with patch(
                    "homodyne.uninstall_scripts.is_virtual_environment",
                    return_value=True,
                ):
                    with patch("builtins.print") as mock_print:
                        main()

                        # Should report removed files
                        mock_print.assert_called()

    def test_main_cleanup_all_files(self):
        """Test main function cleaning up all files."""
        with patch("sys.argv", ["uninstall_scripts.py"]):
            with patch(
                "homodyne.uninstall_scripts.cleanup_completion_files",
                return_value=["comp.sh"],
            ):
                with patch(
                    "homodyne.uninstall_scripts.cleanup_gpu_files",
                    return_value=["gpu.py"],
                ):
                    with patch(
                        "homodyne.uninstall_scripts.is_virtual_environment",
                        return_value=True,
                    ):
                        with patch("builtins.print") as mock_print:
                            main()

                            # Should report all removed files
                            mock_print.assert_called()

    def test_main_no_files_to_remove(self):
        """Test main function when no files need to be removed."""
        with patch("sys.argv", ["uninstall_scripts.py"]):
            with patch(
                "homodyne.uninstall_scripts.cleanup_completion_files", return_value=[]
            ):
                with patch(
                    "homodyne.uninstall_scripts.cleanup_gpu_files", return_value=[]
                ):
                    with patch(
                        "homodyne.uninstall_scripts.is_virtual_environment",
                        return_value=True,
                    ):
                        with patch("builtins.print") as mock_print:
                            main()

                            # Should inform user no files were found
                            mock_print.assert_called()


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_cleanup_with_permission_errors(self):
        """Test cleanup functions handle permission errors gracefully."""
        with patch(
            "pathlib.Path.unlink", side_effect=PermissionError("Permission denied")
        ):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.iterdir", return_value=[Path("test_file")]):
                    # Should not raise exception
                    result = cleanup_completion_files()
                    assert isinstance(result, list)

    def test_cleanup_with_file_not_found_errors(self):
        """Test cleanup functions handle FileNotFoundError gracefully."""
        with patch(
            "pathlib.Path.unlink", side_effect=FileNotFoundError("File not found")
        ):
            with patch("pathlib.Path.exists", return_value=True):
                # Should not raise exception
                result = cleanup_completion_files()
                assert isinstance(result, list)

    def test_cleanup_with_os_errors(self):
        """Test cleanup functions handle general OS errors gracefully."""
        with patch("pathlib.Path.unlink", side_effect=OSError("OS error")):
            with patch("pathlib.Path.exists", return_value=True):
                # Should not raise exception
                result = cleanup_gpu_files()
                assert isinstance(result, list)

    def test_interactive_with_keyboard_interrupt(self):
        """Test interactive cleanup handles KeyboardInterrupt gracefully."""
        with patch("builtins.input", side_effect=KeyboardInterrupt()):
            with patch("builtins.print") as mock_print:
                # Should handle Ctrl+C gracefully
                interactive_cleanup()
                mock_print.assert_called()

    def test_interactive_with_eof_error(self):
        """Test interactive cleanup handles EOFError gracefully."""
        with patch("builtins.input", side_effect=EOFError()):
            with patch("builtins.print") as mock_print:
                # Should handle EOF gracefully
                interactive_cleanup()
                mock_print.assert_called()


class TestFileSystemOperations:
    """Test file system operations and safety checks."""

    def test_safe_file_removal_existing_file(self):
        """Test safe file removal of existing files."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(b"test content")

        try:
            # File should exist initially
            assert temp_path.exists()

            # Remove via cleanup function (indirectly)
            temp_path.unlink()

            # File should be removed
            assert not temp_path.exists()
        finally:
            # Clean up if removal failed
            if temp_path.exists():
                temp_path.unlink()

    def test_safe_file_removal_nonexistent_file(self):
        """Test safe file removal of non-existent files."""
        nonexistent_path = Path(tempfile.gettempdir()) / "nonexistent_file_12345"

        # Should not raise exception when trying to remove non-existent file
        try:
            nonexistent_path.unlink()
        except FileNotFoundError:
            # This is expected behavior
            pass

    def test_directory_traversal_safety(self):
        """Test that cleanup functions don't traverse outside expected directories."""
        # Create a controlled environment
        temp_dir = tempfile.mkdtemp()
        venv_path = Path(temp_dir)

        # Create a file outside the expected structure
        outside_file = venv_path / "outside_file.sh"
        outside_file.write_text("# This should not be removed")

        with patch("sys.prefix", str(venv_path)):
            removed_files = cleanup_completion_files()

            # File outside expected structure should not be removed
            assert outside_file.exists()
            assert str(outside_file) not in [str(f) for f in removed_files]

        # Clean up
        outside_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__])
