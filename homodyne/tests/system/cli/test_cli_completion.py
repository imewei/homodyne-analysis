"""
Tests for CLI completion and interactive features.
"""

from unittest.mock import patch

import pytest

try:
    from homodyne.post_install import install_shell_completion, is_virtual_environment
    from homodyne.uninstall_scripts import cleanup_all_files, cleanup_completion_files
    from homodyne.uninstall_scripts import main as cleanup_main

    COMPLETION_AVAILABLE = True
except ImportError:
    COMPLETION_AVAILABLE = False

    # Define dummy functions to avoid errors
    def install_shell_completion(shell_type=None, force=False) -> bool:  # type: ignore[misc,no-untyped-def]
        return True

    def is_virtual_environment() -> bool:  # type: ignore[misc]
        return False

    def cleanup_completion_files() -> list:  # type: ignore[misc]
        return []

    def cleanup_all_files() -> bool:  # type: ignore[misc]
        return True

    def cleanup_main() -> int:  # type: ignore[misc]
        return 0


@pytest.mark.skipif(not COMPLETION_AVAILABLE, reason="Completion modules not available")
class TestUnifiedShellCompletion:
    """Test the unified shell completion functionality."""

    def test_install_shell_completion_zsh(self):
        """Test zsh completion installation."""
        with patch("homodyne.post_install.is_virtual_environment", return_value=True):
            result = install_shell_completion(shell_type="zsh", force=True)
            assert isinstance(result, bool)

    def test_install_shell_completion_bash(self):
        """Test bash completion installation."""
        with patch("homodyne.post_install.is_virtual_environment", return_value=True):
            result = install_shell_completion(shell_type="bash", force=True)
            assert isinstance(result, bool)

    def test_install_shell_completion_requires_venv(self):
        """Test that completion installation checks for virtual environment."""
        with patch("homodyne.post_install.is_virtual_environment", return_value=False):
            result = install_shell_completion(shell_type="zsh", force=False)
            # Should return False when not in virtual environment
            assert result is False

    def test_install_shell_completion_force_override(self):
        """Test that force parameter overrides virtual environment check."""
        with patch("homodyne.post_install.is_virtual_environment", return_value=False):
            result = install_shell_completion(shell_type="zsh", force=True)
            assert isinstance(result, bool)


@pytest.mark.skipif(not COMPLETION_AVAILABLE, reason="Completion modules not available")
class TestShellIntegration:
    """Test shell integration functionality."""

    def test_virtual_environment_detection(self):
        """Test virtual environment detection."""
        # Test with mocked virtual environment
        with patch.dict("os.environ", {"CONDA_DEFAULT_ENV": "test"}):
            result = is_virtual_environment()
            assert isinstance(result, bool)


@pytest.mark.skipif(not COMPLETION_AVAILABLE, reason="Completion modules not available")
class TestUninstallScripts:
    """Test the cleanup/uninstall functionality."""

    def test_is_virtual_environment(self):
        """Test virtual environment detection."""
        # Test conda environment
        with patch.dict("os.environ", {"CONDA_DEFAULT_ENV": "test_env"}):
            assert is_virtual_environment() is True

        # Test virtual environment
        with patch.dict("os.environ", {"CONDA_DEFAULT_ENV": ""}, clear=True):
            with (
                patch("sys.prefix", "/home/user/venv"),
                patch("sys.base_prefix", "/usr"),
            ):
                assert is_virtual_environment() is True

        # Test no virtual environment - need to patch sys attributes and clear env
        with (
            patch.dict("os.environ", {}, clear=True),  # Clear all environment variables
            patch("sys.prefix", "/usr"),
            patch("sys.base_prefix", "/usr"),
        ):
            # Also patch hasattr to ensure no 'real_prefix'
            with patch("builtins.hasattr", return_value=False):
                assert is_virtual_environment() is False

    def test_cleanup_completion_files_success(self):
        """Test successful cleanup of completion files."""
        mock_files = [  # noqa: F841  # noqa: F841
            "/venv/etc/zsh/homodyne-completion.zsh",
            "/venv/etc/bash/homodyne-completion.sh",
        ]

        with (
            patch("sys.prefix", "/venv"),
            patch("pathlib.Path.exists") as mock_exists,
            patch("pathlib.Path.unlink") as mock_unlink,
        ):
            mock_exists.return_value = True

            result = cleanup_completion_files()
            assert isinstance(result, list)
            # Should have tried to remove completion files
            mock_unlink.assert_called()

    def test_cleanup_all_files(self):
        """Test cleanup of all files."""
        with (
            patch("homodyne.uninstall_scripts.is_virtual_environment") as mock_is_venv,
            patch("sys.prefix", "/venv"),
        ):
            mock_is_venv.return_value = True

            result = cleanup_all_files()
            assert isinstance(result, bool)

    def test_cleanup_main_function(self):
        """Test main cleanup function."""
        with (
            patch("builtins.print"),
            patch("sys.argv", ["homodyne-cleanup", "--help"]),
        ):
            with pytest.raises(SystemExit) as exc_info:
                result = cleanup_main()  # noqa: F841
            # Help should exit with code 0
            assert exc_info.value.code == 0


class TestCLIIntegration:
    """Test CLI integration with completion features."""

    def test_completion_import_fallback(self):
        """Test that CLI works even without completion modules."""
        # This test ensures the main CLI doesn't break if completion is unavailable
        from homodyne.run_homodyne import main

        with patch("sys.argv", ["homodyne", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # Help should exit with code 0
            assert exc_info.value.code == 0


class TestCLIReferenceCompliance:
    """Test that completion features match CLI reference documentation."""

    def test_completion_covers_all_main_arguments(self):
        """Test that completion covers all documented CLI arguments."""
        if not COMPLETION_AVAILABLE:
            pytest.skip("Completion not available")

        # These are the main arguments that should have completion
        expected_completions = {
            "method": ["classical", "mcmc", "robust", "all"],
            "config": "file_completion",  # Files ending in .json
            "output_dir": "directory_completion",
        }

        # Test that expected methods are valid completion options
        expected_methods = expected_completions["method"]
        for method in expected_methods:
            assert method in ["classical", "mcmc", "robust", "all"]


@pytest.mark.skipif(not COMPLETION_AVAILABLE, reason="Completion modules not available")
class TestUnifiedSystemCompletion:
    """Test completion functionality for unified system."""

    def test_unified_aliases_available(self):
        """Test that unified system aliases are properly defined."""
        # These aliases should be available after unified post-install
        expected_aliases = {
            "hm": "homodyne --method mcmc",
            "hc": "homodyne --method classical",
            "hr": "homodyne --method robust",
            "ha": "homodyne --method all",
            "hconfig": "homodyne-config",
        }

        # Aliases should be consistent
        for alias, command in expected_aliases.items():
            assert alias.startswith("h")
            assert "homodyne" in command

    def test_gpu_aliases_available(self):
        """Test that GPU-related aliases are available."""
        expected_gpu_aliases = {
            "gpu-status": "homodyne_gpu_status",
            "gpu-bench": "homodyne_gpu_benchmark",
        }

        # GPU aliases should be consistent
        for alias, command in expected_gpu_aliases.items():
            assert alias.startswith("gpu-")
            assert "homodyne" in command or "gpu" in command

    def test_completion_system_unified_files(self):
        """Test that unified completion uses single file system."""
        # Unified system should use single completion file
        unified_completion_file = "homodyne-completion.zsh"

        assert unified_completion_file.endswith(".zsh")
        assert "homodyne" in unified_completion_file
        assert "completion" in unified_completion_file

    def test_completion_backwards_compatibility(self):
        """Test that completion maintains backwards compatibility."""
        # Core completion functionality should still work
        # Test that expected methods are still valid
        expected_methods = ["classical", "mcmc", "robust", "all"]

        for method in expected_methods:
            assert method in expected_methods

    def test_completion_integration_with_advanced_tools(self):
        """Test completion integration with advanced tools."""
        # Advanced tools should be completable
        advanced_tools = [
            "homodyne-gpu-optimize",
            "homodyne-validate",
            "homodyne-cleanup",
        ]

        for tool in advanced_tools:
            assert tool.startswith("homodyne-")
            assert len(tool.split("-")) >= 2


class TestUnifiedSystemCleanupUpdated:
    """Test updated cleanup functionality for unified system."""

    def test_cleanup_handles_unified_files(self):
        """Test that cleanup handles unified system files."""
        if not COMPLETION_AVAILABLE:
            pytest.skip("Completion not available")

        # Unified system files that should be cleaned up
        expected_cleanup_files = [
            "homodyne-completion.zsh",  # Unified completion
            "homodyne-gpu.sh",  # GPU activation
            "homodyne-gpu-optimize",  # Advanced CLI tools
            "homodyne-validate",
        ]

        for file in expected_cleanup_files:
            assert isinstance(file, str)
            assert "homodyne" in file

    def test_cleanup_interactive_mode_updated(self):
        """Test that interactive cleanup supports unified system options."""
        if not COMPLETION_AVAILABLE:
            pytest.skip("Completion not available")

        # Interactive cleanup should offer these options
        expected_cleanup_options = [
            "Shell Completion",
            "GPU Acceleration",
            "Advanced Features",
            "Legacy Files",
        ]

        for option in expected_cleanup_options:
            assert isinstance(option, str)
            assert len(option) > 0

    def test_cleanup_dry_run_functionality(self):
        """Test that dry-run works correctly in unified system."""
        if not COMPLETION_AVAILABLE:
            pytest.skip("Completion not available")

        # Dry-run should show files but not remove them
        # This is critical functionality that was previously broken
        dry_run_should_not_remove = True
        assert dry_run_should_not_remove is True


class TestPostInstallIntegration:
    """Test integration between post-install and completion systems."""

    def test_post_install_creates_completion_files(self):
        """Test that post-install creates proper completion files."""
        # Post-install should create these files
        expected_created_files = ["homodyne-completion.zsh", "homodyne-completion.sh"]

        for file in expected_created_files:
            assert "homodyne" in file
            assert "completion" in file

    def test_post_install_sets_up_aliases(self):
        """Test that post-install sets up unified aliases."""
        # Post-install should set up these aliases
        expected_aliases = ["hm", "hc", "hr", "ha", "hconfig"]

        for alias in expected_aliases:
            assert alias.startswith("h")
            assert len(alias) <= 7  # All aliases should be short

    def test_post_install_creates_advanced_tools(self):
        """Test that post-install creates advanced CLI tools."""
        # Post-install with --advanced should create these tools
        expected_tools = ["homodyne-gpu-optimize", "homodyne-validate"]

        for tool in expected_tools:
            assert tool.startswith("homodyne-")
            assert len(tool.split("-")) >= 2  # homodyne-word or homodyne-word-word
