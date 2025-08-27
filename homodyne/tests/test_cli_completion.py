"""
Tests for CLI completion and interactive features.
"""

import argparse
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

try:
    from homodyne.cli_completion import (
        HomodyneCompleter,
        install_shell_completion,
        setup_shell_completion,
    )

    COMPLETION_AVAILABLE = True
except ImportError:
    COMPLETION_AVAILABLE = False
    # Define dummy classes/functions to avoid errors

    class HomodyneCompleter:  # type: ignore[misc]
        def method_completer(self, prefix: str, parsed_args: Any) -> List[str]:
            return []

        def config_files_completer(self, prefix: str, parsed_args: Any) -> List[str]:
            return []

        def output_dir_completer(self, prefix: str, parsed_args: Any) -> List[str]:
            return []

        def analysis_mode_completer(self, prefix: str, parsed_args: Any) -> List[str]:
            return []

    def setup_shell_completion(parser: Any) -> None:  # type: ignore[misc]
        pass

    def install_shell_completion(shell: str) -> int:  # type: ignore[misc]
        return 1


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

        # Test json file completion
        with patch("glob.glob") as mock_glob:
            mock_glob.return_value = ["config.json", "my_config.json", "test.json"]

            results = completer.config_files_completer("config", argparse.Namespace())
            assert "config.json" in results

            results = completer.config_files_completer("my", argparse.Namespace())
            assert "my_config.json" in results

    def test_output_dir_completer(self):
        """Test directory completion."""
        completer = HomodyneCompleter()

        with patch("os.listdir") as mock_listdir, patch("os.path.isdir") as mock_isdir:

            mock_listdir.return_value = ["results", "data", "config.json"]
            mock_isdir.side_effect = lambda x: not x.endswith(".json")

            results = completer.output_dir_completer("res", argparse.Namespace())
            # Should return directories only
            assert any("results" in r for r in results)

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
    """Test shell completion installation."""

    def test_install_completion_bash(self):
        """Test bash completion installation."""
        with (
            patch("homodyne.cli_completion.ARGCOMPLETE_AVAILABLE", True),
            patch("pathlib.Path.home") as mock_home,
            patch("pathlib.Path.mkdir") as mock_mkdir,
            patch("pathlib.Path.exists"),
            patch("pathlib.Path.read_text"),
            patch("builtins.open", create=True) as mock_open,
        ):

            # Setup mocks
            mock_home_path = MagicMock(spec=Path)
            mock_home.return_value = mock_home_path
            mock_config_file = MagicMock(spec=Path)
            mock_config_file.parent.mkdir = mock_mkdir
            mock_config_file.exists.return_value = False

            # Mock Path constructor to return our mock config file
            with patch("homodyne.cli_completion.Path") as mock_path_constructor:
                mock_path_constructor.return_value = mock_config_file

                mock_file = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_file

                result = install_shell_completion("bash")
                assert result == 0
                mock_open.assert_called()

    def test_install_completion_unsupported_shell(self):
        """Test error handling for unsupported shell."""
        result = install_shell_completion("unsupported")
        assert result == 1

    def test_setup_shell_completion(self):
        """Test setup of shell completion on parser."""
        with patch("argparse.ArgumentParser") as mock_parser:
            parser = mock_parser.return_value
            parser._actions = []

            # Should not raise any errors
            setup_shell_completion(parser)


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

    def test_install_completion_argument(self):
        """Test --install-completion argument parsing."""
        from homodyne.run_homodyne import main

        with patch("sys.argv", ["homodyne", "--install-completion", "bash"]):
            with patch("homodyne.run_homodyne.COMPLETION_AVAILABLE", True):
                with patch(
                    "homodyne.run_homodyne.install_shell_completion"
                ) as mock_install:
                    mock_install.return_value = 0
                    result = main()
                    assert result == 0
                    mock_install.assert_called_once_with("bash")


    def test_completion_unavailable_error_messages(self):
        """Test error messages when completion features are unavailable."""
        from homodyne.run_homodyne import main

        with patch("sys.argv", ["homodyne", "--install-completion", "bash"]):
            with patch("homodyne.run_homodyne.COMPLETION_AVAILABLE", False):
                with patch("builtins.print") as mock_print:
                    result = main()
                    assert result == 1
                    mock_print.assert_called()
                    # Check that helpful error message was printed
                    printed_args = [call[0][0] for call in mock_print.call_args_list]
                    assert any("argcomplete" in msg for msg in printed_args)





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

