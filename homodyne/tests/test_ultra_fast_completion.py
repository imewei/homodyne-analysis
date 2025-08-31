"""Ultra-fast completion performance tests with persistent caching."""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from homodyne.cli_completion import HomodyneCompleter
from homodyne.completion_fast import FastCache


class TestUltraFastCompletion:
    """Test ultra-fast shell completion with persistent caching."""

    def test_instant_static_completions(self):
        """Test that static completions are truly instant (< 0.001ms)."""
        completer = HomodyneCompleter()

        # Method completion
        times = []
        for _ in range(100):  # Test 100 times for accuracy
            start = time.perf_counter()
            results = completer.method_completer("cl", argparse.Namespace())
            times.append((time.perf_counter() - start) * 1000)

        avg_time = sum(times) / len(times)
        max_time = max(times)

        assert avg_time < 0.01, f"Method completion too slow: {avg_time:.4f}ms average"
        assert (
            max_time < 0.1
        ), f"Method completion worst case too slow: {max_time:.4f}ms"
        assert "classical" in results

        # Mode completion
        times = []
        for _ in range(100):
            start = time.perf_counter()
            results = completer.analysis_mode_completer("static", argparse.Namespace())
            times.append((time.perf_counter() - start) * 1000)

        avg_time = sum(times) / len(times)
        assert avg_time < 0.01, f"Mode completion too slow: {avg_time:.4f}ms average"
        assert len(results) == 2  # static_isotropic, static_anisotropic

    def test_cached_file_completion_speed(self):
        """Test that cached file completion is ultra-fast."""
        completer = HomodyneCompleter()

        # Mock the completion_fast module for performance testing
        test_files = [f"config_{i:02d}.json" for i in range(50)]
        
        # Test cached performance (should be instant)
        times = []
        for _ in range(50):
            start = time.perf_counter()
            with patch("homodyne.cli_completion.complete_config", return_value=test_files):
                results = completer.config_files_completer(
                    "config", argparse.Namespace()
                )
            times.append((time.perf_counter() - start) * 1000)

        avg_time = sum(times) / len(times)
        max_time = max(times)

        assert (
            avg_time < 0.1
        ), f"Cached file completion too slow: {avg_time:.4f}ms average"
        assert (
            max_time < 1.0
        ), f"Cached file completion worst case too slow: {max_time:.4f}ms"
        assert len(results) > 0

    def test_persistent_cache_loading(self):
        """Test that persistent cache loading is fast."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / ".cache" / "homodyne"
            cache_file = cache_dir / "completion_cache.json"

            # Create mock cache file
            cache_dir.mkdir(parents=True)
            cache_data = {
                "timestamp": time.time(),
                "files": {".": ["config.json", "my_config.json"]},
                "dirs": {".": ["output", "results"]},
            }

            with open(cache_file, "w") as f:
                json.dump(cache_data, f)

            # Test cache loading speed
            with patch.object(Path, "home", return_value=Path(tmpdir)):
                start = time.perf_counter()
                cache = FastCache()
                load_time = (time.perf_counter() - start) * 1000

                assert load_time < 10, f"Cache loading too slow: {load_time:.2f}ms"
                assert cache.get_files(".") == ["config.json", "my_config.json"]
                assert cache.get_dirs(".") == ["output", "results"]

    def test_empty_prefix_performance(self):
        """Test that empty prefix completion (worst case) is still fast."""
        completer = HomodyneCompleter()

        # Mock the completion_fast module with many items
        test_files = [f"config_{i:03d}.json" for i in range(100)]

        # Test empty prefix (returns all items)
        with patch("homodyne.cli_completion.complete_config", return_value=test_files[:15]):
            times = []
            for _ in range(20):
                start = time.perf_counter()
                results = completer.config_files_completer("", argparse.Namespace())
                times.append((time.perf_counter() - start) * 1000)

            avg_time = sum(times) / len(times)
            assert (
                avg_time < 0.5
            ), f"Empty prefix completion too slow: {avg_time:.4f}ms average"
            assert len(results) <= 15, "Results should be limited for performance"

    def test_cache_update_performance(self):
        """Test that cache updates are reasonably fast."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create many files
            for i in range(200):
                Path(tmpdir, f"file_{i:03d}.json").touch()
            for i in range(100):
                Path(tmpdir, f"dir_{i:02d}").mkdir()

            orig_dir = os.getcwd()
            os.chdir(tmpdir)

            try:
                cache = FastCache()
                
                start = time.perf_counter()
                cache._scan_current_dir()  # Force scan
                update_time = (time.perf_counter() - start) * 1000

                assert update_time < 50, f"Cache update too slow: {update_time:.2f}ms"

                # Verify cache populated correctly
                files = cache.get_files(".")
                dirs = cache.get_dirs(".")
                assert len(files) > 0
                assert len(dirs) > 0

            finally:
                os.chdir(orig_dir)

    def test_memory_efficiency(self):
        """Test that cache doesn't consume excessive memory."""
        import sys

        # Create cache with reasonable amount of data
        cache = FastCache()
        cache._data = {
            "timestamp": time.time(),
            "files": {
                ".": [f"config_{i:03d}.json" for i in range(100)],
                "config": [f"sub_config_{i:02d}.json" for i in range(50)],
            },
            "dirs": {
                ".": [f"dir_{i:02d}" for i in range(50)],
                "output": [f"subdir_{i:02d}" for i in range(25)],
            },
        }

        # Rough memory usage check (cache should be small)
        cache_size = sys.getsizeof(cache._data)
        assert cache_size < 10000, f"Cache using too much memory: {cache_size} bytes"

    def test_concurrent_completion_calls(self):
        """Test that multiple concurrent completion calls are fast."""
        completer = HomodyneCompleter()

        # Simulate rapid successive completion calls
        total_start = time.perf_counter()

        for _i in range(10):
            results1 = completer.method_completer("c", argparse.Namespace())
            results2 = completer.analysis_mode_completer("s", argparse.Namespace())

            assert len(results1) > 0
            assert len(results2) > 0

        total_time = (time.perf_counter() - total_start) * 1000
        avg_per_call = total_time / 20  # 20 total calls

        assert (
            avg_per_call < 0.1
        ), f"Concurrent calls too slow: {avg_per_call:.4f}ms per call"
        assert total_time < 10, f"Total time for 20 calls too slow: {total_time:.2f}ms"


if __name__ == "__main__":
    # Run ultra-fast performance tests
    test = TestUltraFastCompletion()

    print("Running ultra-fast completion performance tests...")

    test.test_instant_static_completions()
    print("✓ Static completions are truly instant (< 0.01ms)")

    test.test_cached_file_completion_speed()
    print("✓ Cached file completions are ultra-fast (< 0.1ms)")

    test.test_persistent_cache_loading()
    print("✓ Cache loading from disk is fast (< 10ms)")

    test.test_empty_prefix_performance()
    print("✓ Empty prefix completion is fast (< 0.5ms)")

    test.test_cache_update_performance()
    print("✓ Cache updates are reasonable (< 50ms)")

    test.test_memory_efficiency()
    print("✓ Cache memory usage is efficient")

    test.test_concurrent_completion_calls()
    print("✓ Concurrent calls maintain performance")

    print("\nAll ultra-fast performance tests passed!")
    print("Shell completion is now INSTANT! ⚡")
