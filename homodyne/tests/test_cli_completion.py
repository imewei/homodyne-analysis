"""
Tests for CLI completion and interactive features.
"""

import argparse
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

try:
    from homodyne.cli_completion import (
        HomodyneCompleter,
        setup_shell_completion,
    )
    from homodyne.uninstall_scripts import (
        cleanup_conda_scripts,
        is_virtual_environment,
    )
    from homodyne.uninstall_scripts import main as cleanup_main

    COMPLETION_AVAILABLE = True
except ImportError:
    COMPLETION_AVAILABLE = False
    # Define dummy classes/functions to avoid errors

    class _DummyHomodyneCompleter:  # type: ignore[misc]
        def method_completer(self, prefix: str, parsed_args: Any) -> list[str]:
            return []

        def config_files_completer(self, prefix: str, parsed_args: Any) -> list[str]:
            return []

        def output_dir_completer(self, prefix: str, parsed_args: Any) -> list[str]:
            return []

        def analysis_mode_completer(self, prefix: str, parsed_args: Any) -> list[str]:
            return []

    HomodyneCompleter = _DummyHomodyneCompleter  # type: ignore[misc]

    def setup_shell_completion(parser: Any) -> None:  # type: ignore[misc]
        pass

    def cleanup_conda_scripts() -> bool:  # type: ignore[misc]
        return True

    def is_virtual_environment() -> bool:  # type: ignore[misc]
        return False

    def cleanup_main() -> int:  # type: ignore[misc]
        return 0


@pytest.mark.skipif(not COMPLETION_AVAILABLE, reason="Completion modules not available")
class TestHomodyneCompleter:
    """Test the completion functionality."""

    def test_method_completer(self):
        """Test method completion."""
        completer = HomodyneCompleter()

        # Test full matches
        results = completer.method_completer("classical", argparse.Namespace())
        assert "classical" in results

        # Test partial matches
        results = completer.method_completer("mc", argparse.Namespace())
        assert "mcmc" in results

        # Test all methods
        results = completer.method_completer("", argparse.Namespace())
        expected = ["classical", "mcmc", "robust", "all"]
        assert all(method in results for method in expected)

    def test_config_files_completer(self):
        """Test config file completion."""
        completer = HomodyneCompleter()

        # Mock the completion_fast module
        with patch("homodyne.cli_completion.complete_config") as mock_complete:
            mock_complete.return_value = ["config.json", "my_config.json", "test.json"]

            results = completer.config_files_completer("config", argparse.Namespace())
            assert results == ["config.json", "my_config.json", "test.json"]
            mock_complete.assert_called_with("config")

            results = completer.config_files_completer("my", argparse.Namespace())
            assert results == ["config.json", "my_config.json", "test.json"]
            mock_complete.assert_called_with("my")

    def test_output_dir_completer(self):
        """Test directory completion."""
        completer = HomodyneCompleter()

        # Mock the completion_fast module
        with patch("homodyne.cli_completion.complete_output_dir") as mock_complete:
            mock_complete.return_value = ["results/", "data/", "output/"]

            results = completer.output_dir_completer("res", argparse.Namespace())
            assert results == ["results/", "data/", "output/"]
            mock_complete.assert_called_with("res")

    def test_analysis_mode_completer(self):
        """Test analysis mode completion."""
        completer = HomodyneCompleter()

        results = completer.analysis_mode_completer("static", argparse.Namespace())
        assert "static_isotropic" in results
        assert "static_anisotropic" in results

        results = completer.analysis_mode_completer("laminar", argparse.Namespace())
        assert "laminar_flow" in results


@pytest.mark.skipif(not COMPLETION_AVAILABLE, reason="Completion modules not available")
class TestShellCompletion:
    """Test shell completion installation and conda integration."""

    def test_setup_shell_completion(self):
        """Test setup of shell completion on parser."""
        with patch("argparse.ArgumentParser") as mock_parser:
            parser = mock_parser.return_value
            parser._actions = []

            # Should not raise any errors
            setup_shell_completion(parser)


@pytest.mark.skipif(not COMPLETION_AVAILABLE, reason="Completion modules not available")
class TestUninstallScripts:
    """Test the cleanup/uninstall functionality."""

    def test_is_virtual_environment(self):
        """Test virtual environment detection."""
        # Test conda environment
        with patch.dict("os.environ", {"CONDA_DEFAULT_ENV": "test_env"}):
            assert is_virtual_environment() == True

        # Test virtual environment
        with patch.dict("os.environ", {"CONDA_DEFAULT_ENV": ""}, clear=True):
            with (
                patch("sys.prefix", "/home/user/venv"),
                patch("sys.base_prefix", "/usr"),
            ):
                assert is_virtual_environment() == True

        # Test no virtual environment - need to patch sys attributes and clear env
        with (
            patch.dict("os.environ", {}, clear=True),  # Clear all environment variables
            patch("sys.prefix", "/usr"),
            patch("sys.base_prefix", "/usr"),
        ):
            # Also patch hasattr to ensure no 'real_prefix'
            with patch("builtins.hasattr", return_value=False):
                assert is_virtual_environment() == False

    def test_cleanup_conda_scripts_success(self):
        """Test successful cleanup of conda scripts."""
        mock_scripts = [
            "/conda/prefix/etc/conda/activate.d/homodyne-gpu-activate.sh",
            "/conda/prefix/etc/conda/deactivate.d/homodyne-gpu-deactivate.sh",
            "/conda/prefix/etc/homodyne/gpu_activation.sh",
            "/conda/prefix/etc/homodyne/homodyne_aliases.sh",
            "/conda/prefix/etc/homodyne/homodyne_completion.zsh",
            "/conda/prefix/etc/homodyne/homodyne_config.sh",
        ]

        with (
            patch("homodyne.uninstall_scripts.is_virtual_environment") as mock_is_venv,
            patch("platform.system") as mock_platform,
            patch("sys.prefix", "/conda/prefix"),
            patch("pathlib.Path.exists") as mock_exists,
            patch("pathlib.Path.unlink") as mock_unlink,
            patch("pathlib.Path.rmdir") as mock_rmdir,
            patch("pathlib.Path.iterdir") as mock_iterdir,
        ):
            mock_is_venv.return_value = True
            mock_platform.return_value = "Linux"
            mock_exists.return_value = True
            mock_iterdir.return_value = []  # Empty directory

            result = cleanup_conda_scripts()
            assert result == True
            # Should have called unlink for each script file
            assert mock_unlink.call_count == len(mock_scripts)

    def test_cleanup_conda_scripts_not_virtual_env(self):
        """Test cleanup skips when not in virtual environment."""
        with patch("homodyne.uninstall_scripts.is_virtual_environment") as mock_is_venv:
            mock_is_venv.return_value = False

            result = cleanup_conda_scripts()
            assert result == False

    def test_cleanup_conda_scripts_not_linux(self):
        """Test cleanup skips when not on Linux."""
        with (
            patch("homodyne.uninstall_scripts.is_virtual_environment") as mock_is_venv,
            patch("platform.system") as mock_platform,
        ):
            mock_is_venv.return_value = True
            mock_platform.return_value = "Windows"

            result = cleanup_conda_scripts()
            assert result == True  # Returns True but prints info message

    def test_cleanup_main_function(self):
        """Test main cleanup function."""
        with (
            patch("homodyne.uninstall_scripts.cleanup_conda_scripts") as mock_cleanup,
            patch("builtins.print") as mock_print,
        ):
            mock_cleanup.return_value = True

            result = cleanup_main()
            assert result == 0
            mock_cleanup.assert_called_once()

    def test_cleanup_main_function_failure(self):
        """Test main cleanup function handles failures."""
        with (
            patch("homodyne.uninstall_scripts.cleanup_conda_scripts") as mock_cleanup,
            patch("builtins.print") as mock_print,
        ):
            mock_cleanup.return_value = False

            result = cleanup_main()
            assert result == 1


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

        completer = HomodyneCompleter()

        # Test method completion
        methods = completer.method_completer("", argparse.Namespace())
        for expected_method in expected_completions["method"]:
            assert (
                expected_method in methods
            ), f"Method {expected_method} missing from completion"
