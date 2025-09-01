"""
Performance Test Configuration
==============================

Configuration for performance and benchmark tests.
"""

import json
import time
from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    """Automatically mark all tests in performance directory."""
    for item in items:
        # Add performance marker to all tests in this directory
        item.add_marker(pytest.mark.performance)
        item.add_marker(pytest.mark.benchmark)


class PerformanceTracker:
    """Track performance metrics across test runs."""

    def __init__(self):
        self.metrics = {}
        self.baseline_file = Path(__file__).parent / "performance_baseline.json"
        self.load_baseline()

    def load_baseline(self):
        """Load performance baseline if it exists."""
        if self.baseline_file.exists():
            with open(self.baseline_file) as f:
                self.baseline = json.load(f)
        else:
            self.baseline = {}

    def record(self, test_name, duration, memory_mb=None):
        """Record performance metrics for a test."""
        self.metrics[test_name] = {"duration": duration, "memory_mb": memory_mb}

        # Check against baseline if available
        if test_name in self.baseline:
            baseline_duration = self.baseline[test_name].get("duration", 0)
            if duration > baseline_duration * 1.5:  # 50% regression threshold
                pytest.warning(
                    f"Performance regression: {test_name} took {duration:.3f}s "
                    f"(baseline: {baseline_duration:.3f}s)"
                )

    def save_metrics(self):
        """Save current metrics as new baseline."""
        with open(self.baseline_file, "w") as f:
            json.dump(self.metrics, f, indent=2)


@pytest.fixture(scope="session")
def performance_tracker():
    """Session-wide performance tracker."""
    return PerformanceTracker()


@pytest.fixture
def benchmark_timer(performance_tracker, request):
    """Time individual test execution."""
    start = time.perf_counter()
    yield
    duration = time.perf_counter() - start
    performance_tracker.record(request.node.name, duration)
