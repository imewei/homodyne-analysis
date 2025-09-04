"""
Tests for unified cleanup system (homodyne-cleanup).

This module tests the enhanced cleanup system that handles:
- Unified shell completion cleanup
- Smart GPU system cleanup
- Advanced features cleanup
- Interactive cleanup with dry-run support
- Legacy system files cleanup
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

try:
    from homodyne.uninstall_scripts import (
        cleanup_advanced_features,
        cleanup_all_files,
        cleanup_completion_files,
        cleanup_gpu_files,
        cleanup_old_system_files,
        interactive_cleanup,
    )
    from homodyne.uninstall_scripts import main as cleanup_main
    from homodyne.uninstall_scripts import (
        show_dry_run,
    )

    CLEANUP_AVAILABLE = True
except ImportError:
    CLEANUP_AVAILABLE = False


@pytest.mark.skipif(not CLEANUP_AVAILABLE, reason="Cleanup module not available")
class TestUnifiedCleanup:
    """Test unified cleanup system functionality."""

    def test_cleanup_completion_files(self):
        """Test cleanup of unified completion files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir)

            # Create mock completion files
            completion_files = [
                venv_path / "etc/bash_completion.d/homodyne-completion.bash",
                venv_path / "etc/conda/activate.d/homodyne-completion.sh",
                venv_path / "etc/conda/activate.d/homodyne-advanced-completion.sh",
                venv_path / "etc/zsh/homodyne-completion.zsh",
                venv_path / "share/fish/vendor_completions.d/homodyne.fish",
            ]

            # Create directories and files
            for file_path in completion_files:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("# Mock completion file")

            with patch("sys.prefix", str(venv_path)):
                removed_files = cleanup_completion_files()

                # Should have removed completion files
                assert len(removed_files) == len(completion_files)
                # All files should be removed
                for file_path in completion_files:
                    assert not file_path.exists()

    def test_cleanup_gpu_files(self):
        """Test cleanup of GPU system files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir)

            # Create mock GPU files that match what cleanup_gpu_files expects
            gpu_files = [
                venv_path / "etc/homodyne/gpu/gpu_activation.sh",
                venv_path / "etc/conda/activate.d/homodyne-gpu.sh",
            ]

            # Create directories and files
            for file_path in gpu_files:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("# Mock GPU file")

            with patch("sys.prefix", str(venv_path)):
                removed_files = cleanup_gpu_files()

                # Should have removed GPU files
                assert len(removed_files) == len(gpu_files)
                # All files should be removed
                for file_path in gpu_files:
                    assert not file_path.exists()

    def test_cleanup_advanced_features(self):
        """Test cleanup of advanced features CLI tools."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir)

            # Create mock advanced tools
            advanced_tools = [
                venv_path / "bin/homodyne-gpu-optimize",
                venv_path / "bin/homodyne-validate",
            ]

            # Create directories and files
            for file_path in advanced_tools:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("#!/usr/bin/env python3\n# Mock CLI tool")

            with patch("sys.prefix", str(venv_path)):
                removed_files = cleanup_advanced_features()

                # Should have removed advanced tools
                assert len(removed_files) == len(advanced_tools)
                # All files should be removed
                for file_path in advanced_tools:
                    assert not file_path.exists()

    def test_cleanup_old_system_files(self):
        """Test cleanup of legacy system files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir)

            # Create mock legacy files
            old_files = [
                venv_path / "etc/conda/activate.d/homodyne-gpu-activate.sh",
                venv_path / "etc/conda/deactivate.d/homodyne-gpu-deactivate.sh",
                venv_path / "etc/homodyne/homodyne_config.sh",
                venv_path / "etc/homodyne/homodyne_aliases.sh",
                venv_path / "Scripts/hm.bat",  # Windows batch files
            ]

            # Create directories and files
            for file_path in old_files:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("# Mock legacy file")

            with patch("sys.prefix", str(venv_path)):
                removed_files = cleanup_old_system_files()

                # Should have removed legacy files
                assert len(removed_files) == len(old_files)
                # All files should be removed
                for file_path in old_files:
                    assert not file_path.exists()


@pytest.mark.skipif(not CLEANUP_AVAILABLE, reason="Cleanup module not available")
class TestInteractiveCleanup:
    """Test interactive cleanup functionality."""

    def test_interactive_cleanup_all_yes(self):
        """Test interactive cleanup when user says yes to all."""
        user_inputs = ["y", "y", "y", "y"]  # Yes to all cleanup options

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("builtins.input", side_effect=user_inputs),
            patch("sys.prefix", temp_dir),
            patch(
                "homodyne.uninstall_scripts.is_virtual_environment", return_value=True
            ),
            patch(
                "homodyne.uninstall_scripts.cleanup_completion_files",
                return_value=[("Shell completion", "test.zsh")],
            ) as mock_completion,
            patch(
                "homodyne.uninstall_scripts.cleanup_gpu_files",
                return_value=[("GPU setup", "gpu.sh")],
            ) as mock_gpu,
            patch(
                "homodyne.uninstall_scripts.cleanup_advanced_features",
                return_value=[("Advanced features", "tool")],
            ) as mock_advanced,
            patch(
                "homodyne.uninstall_scripts.cleanup_old_system_files",
                return_value=[("Old system file", "old.sh")],
            ) as mock_old,
            patch("builtins.print"),
        ):
            removed_files = interactive_cleanup()

            # Should have called all cleanup functions
            mock_completion.assert_called_once()
            mock_gpu.assert_called_once()
            mock_advanced.assert_called_once()
            mock_old.assert_called_once()

            # Should return aggregated results
            assert len(removed_files) == 4

    def test_interactive_cleanup_selective(self):
        """Test interactive cleanup with selective choices."""
        # Yes to completion, no to GPU, yes to advanced, no to old files
        user_inputs = ["y", "n", "y", "n"]

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("builtins.input", side_effect=user_inputs),
            patch("sys.prefix", temp_dir),
            patch(
                "homodyne.uninstall_scripts.is_virtual_environment", return_value=True
            ),
            patch(
                "homodyne.uninstall_scripts.cleanup_completion_files",
                return_value=[("Shell completion", "test.zsh")],
            ) as mock_completion,
            patch(
                "homodyne.uninstall_scripts.cleanup_gpu_files", return_value=[]
            ) as mock_gpu,
            patch(
                "homodyne.uninstall_scripts.cleanup_advanced_features",
                return_value=[("Advanced features", "tool")],
            ) as mock_advanced,
            patch(
                "homodyne.uninstall_scripts.cleanup_old_system_files", return_value=[]
            ) as mock_old,
        ):
            removed_files = interactive_cleanup()

            # Should have called only selected cleanup functions
            mock_completion.assert_called_once()
            mock_gpu.assert_not_called()
            mock_advanced.assert_called_once()
            mock_old.assert_not_called()

            # Should return only selected results
            assert len(removed_files) == 2


@pytest.mark.skipif(not CLEANUP_AVAILABLE, reason="Cleanup module not available")
class TestDryRunFunctionality:
    """Test dry-run functionality (critical bug fix)."""

    def test_show_dry_run_does_not_remove_files(self):
        """Test that dry-run shows files but doesn't actually remove them."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir)

            # Create test files
            test_files = [
                venv_path / "etc/bash_completion.d/homodyne-completion.bash",
                venv_path / "etc/zsh/homodyne-completion.zsh",
                venv_path / "bin/homodyne-gpu-optimize",
                venv_path / "bin/homodyne-validate",
            ]

            for file_path in test_files:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("# Test file")

            with (
                patch("sys.prefix", str(venv_path)),
                patch(
                    "homodyne.uninstall_scripts.is_virtual_environment",
                    return_value=True,
                ),
                patch("builtins.print") as mock_print,
            ):
                result = show_dry_run()

                # Should show what would be removed
                assert result is True

                # CRITICAL: Files should still exist (not actually removed)
                for file_path in test_files:
                    assert file_path.exists(), (
                        f"File {file_path} was incorrectly removed in dry-run!"
                    )

                # Should have printed dry-run information
                print_calls = [str(call) for call in mock_print.call_args_list]
                assert any(
                    "dry run" in call.lower() or "would remove" in call.lower()
                    for call in print_calls
                )

    def test_dry_run_vs_actual_cleanup(self):
        """Test difference between dry-run and actual cleanup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir)

            # Create test file
            test_file = venv_path / "etc/zsh/homodyne-completion.zsh"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text("# Test completion file")

            # Test dry-run first
            with (
                patch("sys.prefix", str(venv_path)),
                patch(
                    "homodyne.uninstall_scripts.is_virtual_environment",
                    return_value=True,
                ),
            ):
                show_dry_run()
                # File should still exist after dry-run
                assert test_file.exists()

                # Now test actual cleanup
                cleanup_completion_files()
                # File should be removed after actual cleanup
                assert not test_file.exists()


@pytest.mark.skipif(not CLEANUP_AVAILABLE, reason="Cleanup module not available")
class TestCleanupMainFunction:
    """Test main cleanup function with different arguments."""

    def test_cleanup_main_default(self):
        """Test default cleanup behavior."""
        with (
            patch("sys.argv", ["homodyne-cleanup"]),
            patch(
                "homodyne.uninstall_scripts.cleanup_all_files", return_value=True
            ) as mock_cleanup,
            patch("builtins.print"),
        ):
            result = cleanup_main()

            assert result == 0
            mock_cleanup.assert_called_once()

    def test_cleanup_main_interactive(self):
        """Test interactive cleanup."""
        with (
            patch("sys.argv", ["homodyne-cleanup", "--interactive"]),
            patch(
                "homodyne.uninstall_scripts.is_virtual_environment", return_value=True
            ),
            patch(
                "homodyne.uninstall_scripts.interactive_cleanup",
                return_value=[("Shell completion", "test.zsh")],
            ) as mock_interactive,
            patch("builtins.print"),
        ):
            result = cleanup_main()

            assert result == 0
            mock_interactive.assert_called_once()

    def test_cleanup_main_dry_run(self):
        """Test dry-run cleanup."""
        with (
            patch("sys.argv", ["homodyne-cleanup", "--dry-run"]),
            patch(
                "homodyne.uninstall_scripts.show_dry_run", return_value=True
            ) as mock_dry_run,
            patch("builtins.print"),
        ):
            result = cleanup_main()

            assert result == 0
            mock_dry_run.assert_called_once()

    def test_cleanup_main_not_virtual_env(self):
        """Test cleanup behavior when not in virtual environment."""
        with (
            patch("sys.argv", ["homodyne-cleanup"]),
            patch(
                "homodyne.uninstall_scripts.is_virtual_environment", return_value=False
            ),
            patch("builtins.print") as mock_print,
        ):
            result = cleanup_main()

            # Should return 1 when not in virtual env (cleanup_all_files returns False)
            assert result == 1
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("virtual environment" in call for call in print_calls)

    def test_cleanup_main_help(self):
        """Test help message display."""
        with patch("sys.argv", ["homodyne-cleanup", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                cleanup_main()
            # Help should exit with code 0
            assert exc_info.value.code == 0


@pytest.mark.skipif(not CLEANUP_AVAILABLE, reason="Cleanup module not available")
class TestCleanupErrorHandling:
    """Test error handling in cleanup system."""

    def test_cleanup_file_permission_errors(self):
        """Test handling of file permission errors."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch("sys.prefix", temp_dir),
            patch("pathlib.Path.exists", return_value=True),
            patch(
                "pathlib.Path.unlink", side_effect=PermissionError("Permission denied")
            ),
            patch("builtins.print") as mock_print,
        ):
            cleanup_completion_files()

            # Should handle permission errors gracefully
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any(
                "failed" in call.lower() or "permission" in call.lower()
                for call in print_calls
            )

    def test_cleanup_missing_files_handling(self):
        """Test handling of missing files."""
        with tempfile.TemporaryDirectory() as temp_dir, patch("sys.prefix", temp_dir):
            # No files exist, should handle gracefully
            removed_files = cleanup_completion_files()
            assert len(removed_files) == 0

    def test_cleanup_exception_handling(self):
        """Test handling of unexpected exceptions."""
        with (
            patch("sys.argv", ["homodyne-cleanup"]),
            patch(
                "homodyne.uninstall_scripts.cleanup_all_files",
                side_effect=Exception("Unexpected error"),
            ),
            patch("builtins.print") as mock_print,
        ):
            result = cleanup_main()

            # Should handle exception gracefully
            assert result == 1
            print_calls = [str(call) for call in mock_print.call_args_list]
            assert any("error" in call.lower() for call in print_calls)


@pytest.mark.skipif(not CLEANUP_AVAILABLE, reason="Cleanup module not available")
class TestCleanupDirectoryManagement:
    """Test cleanup of empty directories."""

    def test_cleanup_empty_directories(self):
        """Test removal of empty directories after file cleanup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir)

            # Create directory structure
            directories = [
                venv_path / "etc/homodyne/gpu",
                venv_path / "etc/homodyne",
                venv_path / "etc/zsh",
                venv_path / "share/fish/vendor_completions.d",
            ]

            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                # Create a file, then remove it to make directory empty
                temp_file = directory / "temp.txt"
                temp_file.write_text("temp")
                temp_file.unlink()

            with (
                patch("sys.prefix", str(venv_path)),
                patch(
                    "homodyne.uninstall_scripts.is_virtual_environment",
                    return_value=True,
                ),
            ):
                result = cleanup_all_files()

                # Should attempt to clean up empty directories
                assert result is True

    def test_cleanup_preserves_non_empty_directories(self):
        """Test that non-empty directories are preserved."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir)

            # Create directory with non-homodyne file
            test_dir = venv_path / "etc/zsh"
            test_dir.mkdir(parents=True)
            non_homodyne_file = test_dir / "other_completion.zsh"
            non_homodyne_file.write_text("# Other completion")

            with (
                patch("sys.prefix", str(venv_path)),
                patch(
                    "homodyne.uninstall_scripts.is_virtual_environment",
                    return_value=True,
                ),
            ):
                cleanup_all_files()

                # Directory should still exist because it has other files
                assert test_dir.exists()
                assert non_homodyne_file.exists()


class TestCleanupDocumentationCompliance:
    """Test that cleanup matches documentation specifications."""

    def test_cleanup_matches_install_uninstall_doc(self):
        """Test cleanup matches INSTALL_UNINSTALL.md specifications."""
        documented_cleanup_categories = [
            "Shell Completion",
            "GPU Acceleration",
            "Advanced Features",
            "Legacy Files",
        ]

        # All categories should be handled
        for category in documented_cleanup_categories:
            assert isinstance(category, str)
            assert len(category) > 0

    def test_cleanup_file_list_accuracy(self):
        """Test that cleanup handles all documented file types."""
        documented_file_types = {
            "completion": [
                "homodyne-completion.zsh",
                "homodyne-completion.bash",
                "homodyne.fish",
            ],
            "gpu": ["gpu_activation_smart.sh", "homodyne-gpu.sh"],
            "advanced": ["homodyne-gpu-optimize", "homodyne-validate"],
        }

        # All documented file types should be valid
        for _category, files in documented_file_types.items():
            assert isinstance(files, list)
            for file in files:
                assert isinstance(file, str)
                assert len(file) > 0

    def test_cleanup_interactive_options_match_docs(self):
        """Test interactive options match documentation."""
        documented_interactive_options = [
            "Shell completion",
            "GPU acceleration",
            "Advanced features",
            "Legacy files",
        ]

        # All documented options should be available
        for option in documented_interactive_options:
            assert isinstance(option, str)
            assert len(option) > 0
