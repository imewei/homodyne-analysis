"""
Test suite for completion_fast module.

This module tests the ultra-lightweight shell completion functionality.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from homodyne.completion_fast import (
    FastCache,
    complete_config,
    complete_method,
    complete_mode,
    complete_output_dir,
    main,
)


class TestFastCache:
    """Test the FastCache class."""

    def test_cache_init_with_fresh_scan(self):
        """Test cache initialization when no cache file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("homodyne.completion_fast.Path.home") as mock_home:
                mock_home.return_value = Path(tmpdir)

                # Change to a directory with test files
                with tempfile.TemporaryDirectory() as test_cwd:
                    test_dir = Path(test_cwd)
                    (test_dir / "config.json").touch()
                    (test_dir / "test.json").touch()
                    (test_dir / "output").mkdir()
                    (test_dir / "data").mkdir()

                    with patch("homodyne.completion_fast.Path.cwd") as mock_cwd:
                        mock_cwd.return_value = test_dir
                        cache = FastCache()

                        # Verify cache was populated
                        assert "config.json" in cache.get_files(".")
                        assert "test.json" in cache.get_files(".")
                        assert "output" in cache.get_dirs(".")
                        assert "data" in cache.get_dirs(".")

    def test_cache_load_from_existing_fresh(self):
        """Test loading from existing fresh cache file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".cache" / "homodyne"
            cache_dir.mkdir(parents=True)
            cache_file = cache_dir / "completion_cache.json"

            # Create a fresh cache file
            cache_data = {
                "timestamp": time.time() - 1,  # 1 second ago (fresh)
                "files": {".": ["test_config.json"]},
                "dirs": {".": ["test_output"]},
            }
            cache_file.write_text(json.dumps(cache_data))

            with patch("homodyne.completion_fast.Path.home") as mock_home:
                mock_home.return_value = Path(tmpdir)
                cache = FastCache()

                # Should load from cache
                assert "test_config.json" in cache.get_files(".")
                assert "test_output" in cache.get_dirs(".")

    def test_cache_load_from_existing_stale(self):
        """Test cache falls back to scan when cache is stale."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".cache" / "homodyne"
            cache_dir.mkdir(parents=True)
            cache_file = cache_dir / "completion_cache.json"

            # Create a stale cache file
            cache_data = {
                "timestamp": time.time() - 10,  # 10 seconds ago (stale)
                "files": {".": ["old_config.json"]},
                "dirs": {".": ["old_output"]},
            }
            cache_file.write_text(json.dumps(cache_data))

            with tempfile.TemporaryDirectory() as test_cwd:
                test_dir = Path(test_cwd)
                (test_dir / "new_config.json").touch()
                (test_dir / "new_output").mkdir()

                with (
                    patch("homodyne.completion_fast.Path.home") as mock_home,
                    patch("homodyne.completion_fast.Path.cwd") as mock_cwd,
                ):
                    mock_home.return_value = Path(tmpdir)
                    mock_cwd.return_value = test_dir

                    cache = FastCache()

                    # Should scan fresh, not load stale cache
                    assert "new_config.json" in cache.get_files(".")
                    assert "old_config.json" not in cache.get_files(".")

    def test_cache_error_handling(self):
        """Test cache handles errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create cache file with invalid JSON
            cache_dir = Path(tmpdir) / ".cache" / "homodyne"
            cache_dir.mkdir(parents=True)
            cache_file = cache_dir / "completion_cache.json"
            cache_file.write_text("invalid json")

            with (
                patch("homodyne.completion_fast.Path.home") as mock_home,
                patch("homodyne.completion_fast.Path.cwd") as mock_cwd,
            ):
                mock_home.return_value = Path(tmpdir)
                mock_cwd.return_value = Path("/nonexistent")  # Will cause scan to fail

                cache = FastCache()

                # Should have empty data but not crash
                assert cache.get_files(".") == []
                assert cache.get_dirs(".") == []

    def test_get_files_prioritization(self):
        """Test that common config files are prioritized."""
        cache = FastCache()
        cache._data = {
            "files": {
                ".": [
                    "z_other.json",
                    "config.json",
                    "a_another.json",
                    "homodyne_config.json",
                ]
            },
            "dirs": {".": []},
            "timestamp": time.time(),
        }

        files = cache.get_files(".")
        # Common configs should come first
        assert files[0] == "config.json"
        assert files[1] == "homodyne_config.json"
        assert "z_other.json" in files
        assert "a_another.json" in files

    def test_get_dirs_prioritization(self):
        """Test that common output dirs are prioritized."""
        cache = FastCache()
        cache._data = {
            "files": {".": []},
            "dirs": {".": ["z_other", "output", "a_another", "results"]},
            "timestamp": time.time(),
        }

        dirs = cache.get_dirs(".")
        # Common dirs should come first
        assert dirs[0] == "output"
        assert dirs[1] == "results"
        assert "z_other" in dirs
        assert "a_another" in dirs


class TestCompletionFunctions:
    """Test the completion functions."""

    def test_complete_method(self):
        """Test method name completion."""
        # Test with no prefix
        methods = complete_method("")
        assert "classical" in methods
        assert "mcmc" in methods
        assert "robust" in methods
        assert "all" in methods

        # Test with prefix
        assert complete_method("c") == ["classical"]
        assert complete_method("m") == ["mcmc"]
        assert complete_method("r") == ["robust"]
        assert complete_method("a") == ["all"]

        # Test case insensitive
        assert complete_method("C") == ["classical"]

        # Test no matches
        assert complete_method("xyz") == []

    def test_complete_mode(self):
        """Test analysis mode completion."""
        # Test with no prefix
        modes = complete_mode("")
        assert "static_isotropic" in modes
        assert "static_anisotropic" in modes
        assert "laminar_flow" in modes

        # Test with prefix
        assert complete_mode("static") == ["static_isotropic", "static_anisotropic"]
        assert complete_mode("lam") == ["laminar_flow"]

        # Test case insensitive
        assert complete_mode("STATIC") == ["static_isotropic", "static_anisotropic"]

    def test_complete_config(self):
        """Test config file completion."""
        with patch("homodyne.completion_fast._cache") as mock_cache:
            mock_cache.get_files.return_value = ["config.json", "test.json"]

            # Test with no prefix
            configs = complete_config("")
            assert "config.json" in configs
            assert "test.json" in configs

            # Test with prefix
            mock_cache.get_files.return_value = ["config.json", "custom.json"]
            assert "config.json" in complete_config("con")
            assert "custom.json" not in complete_config("con")

    def test_complete_config_with_directory(self):
        """Test config completion with directory prefix."""
        with patch("homodyne.completion_fast._cache") as mock_cache:
            mock_cache.get_files.return_value = ["config.json"]

            # Test with directory prefix
            configs = complete_config("data/con")
            mock_cache.get_files.assert_called_with("data")
            # Use os.path.join for platform-agnostic path comparison
            expected_path = os.path.join("data", "config.json")
            assert configs == [expected_path]

    def test_complete_output_dir(self):
        """Test output directory completion."""
        with patch("homodyne.completion_fast._cache") as mock_cache:
            mock_cache.get_dirs.return_value = ["output", "results"]

            # Test with no prefix
            dirs = complete_output_dir("")
            assert "output" + os.sep in dirs
            assert "results" + os.sep in dirs

            # Test with prefix
            mock_cache.get_dirs.return_value = ["output", "other"]
            assert "output" + os.sep in complete_output_dir("out")
            assert "other" + os.sep not in complete_output_dir("out")

    def test_complete_output_dir_with_directory(self):
        """Test output directory completion with directory prefix."""
        with patch("homodyne.completion_fast._cache") as mock_cache:
            mock_cache.get_dirs.return_value = ["output"]

            # Test with parent directory
            dirs = complete_output_dir("project/out")
            mock_cache.get_dirs.assert_called_with("project")
            # Use os.path.join for platform-agnostic path comparison
            expected_path = os.path.join("project", "output") + os.sep
            assert dirs == [expected_path]


class TestMainFunction:
    """Test the main function."""

    def test_main_method_completion(self, capsys):
        """Test main function with method completion."""
        with patch("sys.argv", ["completion_fast", "method", "c"]):
            main()

        captured = capsys.readouterr()
        assert "classical" in captured.out

    def test_main_mode_completion(self, capsys):
        """Test main function with mode completion."""
        with patch("sys.argv", ["completion_fast", "mode", "static"]):
            main()

        captured = capsys.readouterr()
        assert "static_isotropic" in captured.out
        assert "static_anisotropic" in captured.out

    def test_main_config_completion(self, capsys):
        """Test main function with config completion."""
        with patch("homodyne.completion_fast._cache") as mock_cache:
            mock_cache.get_files.return_value = ["config.json"]

            with patch("sys.argv", ["completion_fast", "config", ""]):
                main()

        captured = capsys.readouterr()
        assert "config.json" in captured.out

    def test_main_output_dir_completion(self, capsys):
        """Test main function with output dir completion."""
        with patch("homodyne.completion_fast._cache") as mock_cache:
            mock_cache.get_dirs.return_value = ["output"]

            with patch("sys.argv", ["completion_fast", "output_dir", ""]):
                main()

        captured = capsys.readouterr()
        assert "output" + os.sep in captured.out

    def test_main_unknown_completion(self, capsys):
        """Test main function with unknown completion type."""
        with patch("sys.argv", ["completion_fast", "unknown", "test"]):
            main()

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_main_no_arguments(self, capsys):
        """Test main function with no arguments."""
        with patch("sys.argv", ["completion_fast"]):
            main()

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_main_only_completion_type(self, capsys):
        """Test main function with only completion type."""
        with patch("sys.argv", ["completion_fast", "method"]):
            main()

        captured = capsys.readouterr()
        # Should complete with empty prefix
        assert "classical" in captured.out
        assert "mcmc" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
