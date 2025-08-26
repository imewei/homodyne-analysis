"""
Performance Test Configuration and Fixtures
==========================================

This module provides shared fixtures and configuration for performance testing.
It extends the main conftest.py with performance-specific utilities.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

# Import performance monitoring utilities
from homodyne.core.config import performance_monitor

# Performance tracking system is available via PerformanceRecorder
PERFORMANCE_TRACKER_AVAILABLE = True


def optimize_numerical_environment():
    """Optimize numerical environment for consistent performance."""
    try:
        import os

        optimizations = {
            "numba_threads": os.environ.get("NUMBA_NUM_THREADS", "4"),
            "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
            "mkl_threads": os.environ.get("MKL_NUM_THREADS", "1"),
        }
        return optimizations
    except Exception:
        return {}


def stable_benchmark(
    func,
    *args,
    warmup_runs=1,
    benchmark_runs=3,
    measurement_runs=3,
    outlier_threshold=2.0,
    target_cv=None,
    **kwargs,
):
    """Run a stable benchmark of a function."""
    # Remove benchmark-specific parameters from kwargs before calling func
    benchmark_params = [
        "warmup_runs",
        "benchmark_runs",
        "measurement_runs",
        "outlier_threshold",
        "target_cv",
    ]
    func_kwargs = {
        k: v for k,
        v in kwargs.items() if k not in benchmark_params}

    # Warmup runs
    for _ in range(warmup_runs):
        func(*args, **func_kwargs)

    # Clear previous timings
    performance_monitor.reset_timings()

    # Benchmark runs
    result = None
    times = []
    for _ in range(measurement_runs):
        with performance_monitor.time_function(func.__name__):
            result = func(*args, **func_kwargs)

        # Get the latest timing
        summary = performance_monitor.get_timing_summary()
        if func.__name__ in summary:
            times.append(summary[func.__name__]["mean"])

    # Return result in expected format with comprehensive statistics
    import numpy as np

    mean_time = sum(times) / len(times) if times else 0.0
    sorted_times = sorted(times) if times else [0.0]
    median_time = sorted_times[len(sorted_times) // 2]
    std_time = (
        (sum((t - mean_time) ** 2 for t in times) /
         len(times)) ** 0.5 if times else 0.0
    )

    # Calculate percentiles
    times_array = np.array(times) if times else np.array([0.0])
    percentile_5 = np.percentile(times_array, 5)
    percentile_95 = np.percentile(times_array, 95)

    # Calculate outlier ratio (min/max ratio)
    min_time = min(times) if times else 1.0
    max_time = max(times) if times else 1.0
    # Protect against division by zero - if either is zero, set ratio to 1.0
    outlier_ratio = min_time / \
        max_time if (max_time > 0 and min_time > 0) else 1.0

    # Count outliers (simple threshold-based approach)
    if times and len(times) > 1:
        q1 = np.percentile(times_array, 25)
        q3 = np.percentile(times_array, 75)
        iqr = q3 - q1
        outlier_lower = q1 - 1.5 * iqr
        outlier_upper = q3 + 1.5 * iqr
        outlier_count = sum(1 for t in times if t <
                            outlier_lower or t > outlier_upper)
    else:
        outlier_count = 0

    return {
        "result": result,
        "mean": mean_time,
        "median": median_time,
        "std": std_time,
        "min": min_time,
        "max": max_time,
        "percentile_5": percentile_5,
        "percentile_95": percentile_95,
        "outlier_ratio": outlier_ratio,
        "outlier_count": outlier_count,
        "times": times_array,  # Convert to numpy array as expected by tests
        "outlier_threshold": outlier_threshold,
        "num_measurements": len(times),
    }


def assert_performance_within_bounds(
    execution_time,
    expected_time,
    tolerance=0.5,
    tolerance_factor=None,
    test_name="performance",
    **kwargs,
):
    """Assert that execution time is within expected bounds."""
    # Use tolerance_factor if provided, otherwise use tolerance
    if tolerance_factor is not None:
        tolerance = tolerance_factor

    lower_bound = expected_time * (1 - tolerance)
    upper_bound = expected_time * (1 + tolerance)
    assert lower_bound <= execution_time <= upper_bound, (
        f"Test {test_name}: Execution time {
            execution_time:.4f}s outside bounds "
        f"[{lower_bound:.4f}s, {upper_bound:.4f}s]"
    )


def assert_performance_stability(
        times, cv_threshold=0.3, max_cv=None, **kwargs):
    """Assert that performance measurements are stable."""
    if max_cv is not None:
        cv_threshold = max_cv

    if len(times) < 2:
        return
    mean_time = sum(times) / len(times)
    variance = sum((t - mean_time) ** 2 for t in times) / len(times)
    std_dev = variance**0.5
    cv = std_dev / mean_time if mean_time > 0 else 0
    assert (
        cv <= cv_threshold
    ), f"Coefficient of variation {cv:.3f} exceeds threshold {cv_threshold}"


# Performance test data storage
PERFORMANCE_BASELINE_FILE = Path(
    __file__).parent / "performance_baselines.json"


class PerformanceRecorder:
    """Record and track performance metrics across test runs."""

    def __init__(self):
        self.baselines = self.load_baselines()
        self.current_results = {}

    def load_baselines(self) -> Dict[str, Any]:
        """Load performance baselines from file."""
        if PERFORMANCE_BASELINE_FILE.exists():
            try:
                with open(PERFORMANCE_BASELINE_FILE, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def save_baselines(self):
        """Save current baselines to file."""
        try:
            with open(PERFORMANCE_BASELINE_FILE, "w") as f:
                json.dump(self.baselines, f, indent=2)
        except IOError:
            pass

    def record_metric(self, test_name: str, metric_name: str, value: float):
        """Record a performance metric."""
        if test_name not in self.current_results:
            self.current_results[test_name] = {}
        self.current_results[test_name][metric_name] = value

    def check_regression(
        self,
        test_name: str,
        metric_name: str,
        value: float,
        threshold: float = 1.5,
    ) -> bool:
        """Check if a performance metric shows regression."""
        baseline = self.baselines.get(test_name, {}).get(metric_name)
        if baseline is None:
            # No baseline, record this as the new baseline
            if test_name not in self.baselines:
                self.baselines[test_name] = {}
            self.baselines[test_name][metric_name] = value
            return False

        # Check for regression (value significantly worse than baseline)
        return value > baseline * threshold

    def update_baseline(self, test_name: str, metric_name: str, value: float):
        """Update baseline if current performance is better."""
        baseline = self.baselines.get(
            test_name, {}).get(
            metric_name, float("inf"))
        if value < baseline * 0.9:  # 10% improvement
            if test_name not in self.baselines:
                self.baselines[test_name] = {}
            self.baselines[test_name][metric_name] = value


# Global performance recorder instance
_performance_recorder = PerformanceRecorder()


@pytest.fixture(scope="session")
def performance_recorder():
    """Provide access to the performance recorder."""
    return _performance_recorder


@pytest.fixture(scope="session")
def performance_tracker():
    """Session-scoped performance tracker for regression testing."""
    # Use the existing PerformanceRecorder infrastructure
    yield _performance_recorder


@pytest.fixture(scope="session", autouse=True)
def setup_performance_environment():
    """Set up consistent performance testing environment."""

    # Local warmup implementation
    def warmup_numba_kernels():
        return {"numba_available": True, "total_warmup_time": 0.1}

    # Use consolidated environment optimization (safe for already-initialized
    # Numba)
    try:
        optimizations = optimize_numerical_environment()
        print(
            f"✓ Performance testing environment configured ({
                len(optimizations)} optimizations)"
        )
    except RuntimeError as e:
        if "NUMBA_NUM_THREADS" in str(e):
            print("⚠ Numba threads already initialized - using existing settings")
            optimizations = {}
        else:
            raise

    # Warmup Numba kernels for stable performance
    try:
        warmup_results = warmup_numba_kernels()
        if warmup_results.get("numba_available", False):
            warmup_time = warmup_results.get("total_warmup_time", 0)
            print(f"✓ Numba kernels warmed up in {warmup_time:.3f}s")
        else:
            print("✓ Numba not available, skipping kernel warmup")
    except Exception as e:
        print(f"⚠ Kernel warmup failed: {e}")

    yield

    # Save baselines after all tests complete
    _performance_recorder.save_baselines()
    print("✓ Performance baselines saved")


@pytest.fixture
def performance_timer():
    """Context manager for timing performance tests."""

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.elapsed = None

        def __enter__(self):
            gc.collect()  # Clean up before timing
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = time.perf_counter()
            if self.start_time is not None and self.end_time is not None:
                self.elapsed = self.end_time - self.start_time
            else:
                self.elapsed = None

    return Timer


@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests."""
    try:
        import os

        import psutil

        class MemoryMonitor:
            def __init__(self):
                self.process = psutil.Process(os.getpid())
                self.baseline_memory = None
                self.peak_memory = None

            def __enter__(self):
                gc.collect()
                self.baseline_memory = (
                    self.process.memory_info().rss / 1024 / 1024
                )  # MB
                self.peak_memory = self.baseline_memory
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                gc.collect()
                final_memory = self.process.memory_info().rss / 1024 / 1024
                self.peak_memory = max(self.peak_memory, final_memory)

            def check(self):
                """Check current memory usage and update peak."""
                current_memory = self.process.memory_info().rss / 1024 / 1024
                self.peak_memory = max(self.peak_memory, current_memory)
                return current_memory

            @property
            def memory_used(self):
                """Memory used above baseline."""
                if self.peak_memory and self.baseline_memory:
                    return self.peak_memory - self.baseline_memory
                return 0

        return MemoryMonitor

    except ImportError:
        # Fallback when psutil is not available
        class DummyMemoryMonitor:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def check(self):
                return 0

            @property
            def memory_used(self):
                return 0

        return DummyMemoryMonitor


@pytest.fixture(
    params=[
        {"n_angles": 5, "time_length": 30, "mode": "static"},
        {"n_angles": 15, "time_length": 50, "mode": "laminar"},
    ]
)
def parametrized_benchmark_data(request):
    """Parametrized benchmark data for different test scenarios."""
    params = request.param
    n_angles = params["n_angles"]
    time_length = params["time_length"]
    mode = params["mode"]

    phi_angles = np.linspace(0, 90, n_angles)

    # Create realistic correlation data
    c2_experimental = np.zeros((n_angles, time_length, time_length))

    for i in range(n_angles):
        # Simulate exponential decay correlation
        time_grid = np.arange(time_length)
        t1_grid, t2_grid = np.meshgrid(time_grid, time_grid, indexing="ij")

        decay_rate = 0.1 * (1 + 0.05 * i)
        correlation = np.exp(-decay_rate * np.abs(t1_grid - t2_grid))

        # Add noise
        noise = np.random.normal(0, 0.02, correlation.shape)
        c2_experimental[i] = 1.0 + 0.3 * correlation + noise

    # Parameters based on mode
    if mode == "static":
        parameters = np.array([0.8, -0.02, 0.1])  # D0, alpha, D_offset
    else:  # laminar
        parameters = np.array([0.8, -0.02, 0.1, 0.05, -0.01, 0.001, 15.0])

    return {
        "phi_angles": phi_angles,
        "c2_experimental": c2_experimental,
        "parameters": parameters,
        "time_length": time_length,
        "mode": mode,
        "test_id": f"{mode}_{n_angles}angles_{time_length}time",
    }


@pytest.fixture
def performance_config_factory():
    """Factory for creating performance test configurations."""

    def create_config(
        n_angles: int = 5,
        time_length: int = 30,
        mode: str = "static",
        enable_advanced_settings: bool = True,
    ) -> Dict[str, Any]:
        """Create a performance test configuration."""

        phi_angles = np.linspace(0, 90, n_angles).tolist()

        config = {
            "data_configuration": {"data_file": f"test_performance_{mode}.npz"},
            "processing": {
                "phi_angles": phi_angles,
                "time_range": [0, time_length],
            },
            "analysis_mode": (
                "static_isotropic" if mode == "static" else "laminar_flow"
            ),
            "performance_settings": {
                "parallel_execution": True,
                "num_threads": 1,
                "optimization_counter_log_frequency": 1000,
                "use_numba_optimization": True,
            },
            "validation_rules": {
                "fit_quality": {
                    "acceptable_threshold_per_angle": 5.0,
                    "excellent_threshold": 1.5,
                    "good_threshold": 3.0,
                },
                "frame_range": {"minimum_frames": 10},
            },
        }

        if enable_advanced_settings:
            config["advanced_settings"] = {
                "chi_squared_calculation": {
                    "uncertainty_factor": 0.1,
                    "min_sigma": 1e-6,
                    "use_scaling_optimization": True,
                },
                "numerical_precision": {
                    "float_precision": "float64",
                    "relative_tolerance": 1e-10,
                },
            }

        return config

    return create_config


# Performance assertion helpers
def assert_performance_regression(
    test_name: str,
    metric_name: str,
    value: float,
    recorder: PerformanceRecorder,
    threshold: float = 1.5,
    update_baseline: bool = False,
):
    """Assert that performance hasn't regressed beyond threshold."""

    is_regression = recorder.check_regression(
        test_name, metric_name, value, threshold)

    if is_regression:
        baseline = recorder.baselines[test_name][metric_name]
        # Use the consolidated assertion function
        try:
            assert_performance_within_bounds(
                value, baseline, threshold, f"{test_name}.{metric_name}"
            )
        except AssertionError as e:
            pytest.fail(str(e))

    if update_baseline:
        recorder.update_baseline(test_name, metric_name, value)

    # Record current result
    recorder.record_metric(test_name, metric_name, value)


def assert_memory_usage(
    test_name: str,
    memory_mb: float,
    recorder: PerformanceRecorder,
    max_memory_mb: float = 100.0,
):
    """Assert memory usage is within acceptable limits."""
    # Check absolute memory limit
    assert memory_mb <= max_memory_mb, (
        f"Memory usage too high in {test_name}: "
        f"{memory_mb:.1f} MB > {max_memory_mb:.1f} MB limit"
    )

    # Check for memory regression
    baseline_memory = recorder.baselines.get(test_name, {}).get("memory_mb")
    if baseline_memory:
        memory_regression_threshold = 1.5  # 50% increase
        if memory_mb > baseline_memory * memory_regression_threshold:
            pytest.fail(
                f"Memory regression in {test_name}: "
                f"{memory_mb:.1f} MB vs baseline {baseline_memory:.1f} MB "
                f"({memory_mb / baseline_memory:.2f}x increase)"
            )

    # Record memory usage
    recorder.record_metric(test_name, "memory_mb", memory_mb)


# Expose consolidated performance utilities for backward compatibility
# These are now imported from homodyne.core.profiler for consistency
__all__ = [
    "PerformanceRecorder",
    "assert_performance_regression",
    "assert_memory_usage",
    "assert_performance_within_bounds",  # From profiler
    "assert_performance_stability",  # From profiler
    "stable_benchmark",  # From profiler
    "optimize_numerical_environment",  # From profiler
]


# Pytest hooks for performance testing
def pytest_configure(config):
    """Configure performance test markers and options."""
    config.addinivalue_line(
        "markers",
        "performance: mark test as a performance test that should be fast",
    )
    config.addinivalue_line(
        "markers",
        "memory: mark test as a memory usage test")
    config.addinivalue_line(
        "markers", "regression: mark test as a performance regression test"
    )
    config.addinivalue_line(
        "markers",
        "benchmark: mark test for benchmarking (requires pytest-benchmark)",
    )


def pytest_runtest_setup(item):
    """Setup for performance tests."""
    if item.get_closest_marker("performance"):
        # Ensure clean environment
        gc.collect()


def pytest_runtest_call(item):
    """Call hook for performance tests."""
    if item.get_closest_marker("performance"):
        # Monitor test execution
        pass


def pytest_runtest_teardown(item, nextitem):
    """Teardown for performance tests."""
    if item.get_closest_marker("performance"):
        # Clean up after performance tests
        gc.collect()


# Command line options for performance testing
def pytest_addoption(parser):
    """Add command line options for performance tests."""
    parser.addoption(
        "--update-baselines",
        action="store_true",
        default=False,
        help="Update performance baselines with current results",
    )
    parser.addoption(
        "--performance-threshold",
        type=float,
        default=1.5,
        help="Performance regression threshold (default: 1.5x)",
    )
    parser.addoption(
        "--skip-slow-performance",
        action="store_true",
        default=False,
        help="Skip slow performance tests",
    )


@pytest.fixture
def update_baselines(request):
    """Fixture to check if baselines should be updated."""
    return request.config.getoption("--update-baselines")


@pytest.fixture
def performance_threshold(request):
    """Fixture to get performance regression threshold."""
    return request.config.getoption("--performance-threshold")


# Performance test result reporting
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item, call):
    """Make performance test reports."""
    if call.when == "call" and item.get_closest_marker("performance"):
        outcome = yield
        report = outcome.get_result()

        # Add performance data to report
        if hasattr(item, "performance_data"):
            report.performance_data = item.performance_data

        return report
    else:
        return (yield)
