"""
Performance Test Configuration and Fixtures
==========================================

This module provides shared fixtures and configuration for performance testing.
It extends the main conftest.py with performance-specific utilities for:
- Classical optimization performance testing
- Robust optimization benchmarking
- MCMC sampling performance analysis
- Cross-method performance comparisons

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import gc
import json
import os
import time
from pathlib import Path
from typing import Any

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
    func_kwargs = {k: v for k, v in kwargs.items() if k not in benchmark_params}

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
        (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5 if times else 0.0
    )

    # Calculate percentiles
    times_array = np.array(times) if times else np.array([0.0])
    percentile_5 = np.percentile(times_array, 5)
    percentile_95 = np.percentile(times_array, 95)

    # Calculate outlier ratio (min/max ratio)
    min_time = min(times) if times else 1.0
    max_time = max(times) if times else 1.0
    # Protect against division by zero - if either is zero, set ratio to 1.0
    outlier_ratio = min_time / max_time if (max_time > 0 and min_time > 0) else 1.0

    # Count outliers (simple threshold-based approach)
    if times and len(times) > 1:
        q1 = np.percentile(times_array, 25)
        q3 = np.percentile(times_array, 75)
        iqr = q3 - q1
        outlier_lower = q1 - 1.5 * iqr
        outlier_upper = q3 + 1.5 * iqr
        outlier_count = sum(1 for t in times if t < outlier_lower or t > outlier_upper)
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
        f"Test {test_name}: Execution time {execution_time:.4f}s outside bounds "
        f"[{lower_bound:.4f}s, {upper_bound:.4f}s]"
    )


def assert_performance_stability(times, cv_threshold=0.3, max_cv=None, **kwargs):
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
PERFORMANCE_BASELINE_FILE = Path(__file__).parent / "performance_baselines.json"


class PerformanceRecorder:
    """Record and track performance metrics across test runs."""

    def __init__(self):
        self.baselines = self.load_baselines()
        self.current_results = {}

    def load_baselines(self) -> dict[str, Any]:
        """Load performance baselines from file."""
        if PERFORMANCE_BASELINE_FILE.exists():
            try:
                with open(PERFORMANCE_BASELINE_FILE) as f:
                    data: dict[str, Any] = json.load(f)
                    return data
            except (OSError, json.JSONDecodeError):
                pass
        return {}

    def save_baselines(self):
        """Save current baselines to file."""
        try:
            with open(PERFORMANCE_BASELINE_FILE, "w") as f:
                json.dump(self.baselines, f, indent=2)
        except OSError:
            pass

    def record_metric(self, test_name: str, metric_name: str, value: float) -> None:
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
        return bool(value > baseline * threshold)

    def update_baseline(self, test_name: str, metric_name: str, value: float) -> None:
        """Update baseline if current performance is better."""
        baseline = self.baselines.get(test_name, {}).get(metric_name, float("inf"))
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
                len(optimizations)
            } optimizations)"
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
    ) -> dict[str, Any]:
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
) -> None:
    """Assert that performance hasn't regressed beyond threshold."""

    is_regression = recorder.check_regression(test_name, metric_name, value, threshold)

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
) -> None:
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
    "assert_memory_usage",
    "assert_performance_regression",
    "assert_performance_stability",  # From profiler
    "assert_performance_within_bounds",  # From profiler
    "optimize_numerical_environment",  # From profiler
    "stable_benchmark",  # From profiler
]


# Pytest hooks for performance testing
def pytest_configure(config):
    """Configure performance test markers and options."""
    config.addinivalue_line(
        "markers",
        "performance: mark test as a performance test that should be fast",
    )
    config.addinivalue_line("markers", "memory: mark test as a memory usage test")
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


# Enhanced performance utilities for MCMC and Robust Optimization

# Performance test configuration specific to optimization methods
OPTIMIZATION_PERFORMANCE_CONFIG = {
    "timeouts": {
        "classical_optimization": 30.0,
        "robust_optimization": 90.0,
        "mcmc_sampling": 180.0,
    },
    "thresholds": {
        "chi_squared_computation": 0.02,  # Max time for chi-squared calculation
        "model_building": 1.0,  # Max time for MCMC model building
        "parameter_bounds": 0.001,  # Max time for bounds extraction
        "convergence_diagnostics": 0.5,  # Max time for MCMC diagnostics
        "solver_fallback": 45.0,  # Max time including solver fallback
    },
    "dataset_sizes": {
        "small": {"n_angles": 15, "n_times": 50},
        "medium": {"n_angles": 20, "n_times": 80},
        "large": {"n_angles": 35, "n_times": 120},
    },
    "sample_counts": {
        "quick_test": {"draws": 10, "tune": 5, "chains": 1},
        "standard_test": {"draws": 20, "tune": 10, "chains": 1},
        "thorough_test": {"draws": 50, "tune": 25, "chains": 2},
        "benchmark_test": {"draws": 200, "tune": 100, "chains": 4},
    },
    "ci_adjustments": {
        "timeout_multiplier": 2.0,
        "skip_stress_tests": True,
        "reduce_sample_counts": True,
    },
}


def get_optimization_timeout(method: str) -> float:
    """Get timeout for optimization method with CI adjustments."""
    timeouts = OPTIMIZATION_PERFORMANCE_CONFIG["timeouts"]
    base_timeout: float = timeouts.get(method, 60.0)  # type: ignore[attr-defined]
    is_ci = any(os.getenv(var) for var in ["CI", "GITHUB_ACTIONS", "TRAVIS"])

    if is_ci:
        ci_adjustments = OPTIMIZATION_PERFORMANCE_CONFIG["ci_adjustments"]
        multiplier: float = ci_adjustments["timeout_multiplier"]  # type: ignore[index]
        return base_timeout * multiplier

    return base_timeout


def get_performance_threshold_for_metric(metric: str) -> float:
    """Get performance threshold for specific metric."""
    thresholds = OPTIMIZATION_PERFORMANCE_CONFIG["thresholds"]
    base_threshold: float = thresholds.get(metric, 1.0)  # type: ignore[attr-defined]
    is_ci = any(os.getenv(var) for var in ["CI", "GITHUB_ACTIONS", "TRAVIS"])

    if is_ci:
        return base_threshold * 2.0  # More lenient in CI

    return base_threshold


def assert_optimization_performance(
    elapsed_time: float, method: str, custom_threshold: float | None = None
) -> None:
    """Assert optimization performance is within bounds."""
    threshold = custom_threshold or get_optimization_timeout(method)

    assert elapsed_time <= threshold, (
        f"Performance regression in {method}: took {elapsed_time:.3f}s "
        f"(threshold: {threshold:.3f}s)"
    )


class OptimizationBenchmarkContext:
    """Context manager for optimization benchmarking."""

    def __init__(self, method_name: str, test_name: str = ""):
        self.method_name = method_name
        self.test_name = test_name
        self.start_time = 0.0
        self.end_time = 0.0
        self.memory_start = 0.0
        self.memory_peak = 0.0
        self.results: dict[str, float] = {}

    def __enter__(self):
        # Optimize environment for consistent benchmarking
        optimize_numerical_environment()
        gc.collect()

        self.start_time = time.perf_counter()
        self.memory_start = self._get_memory_usage()
        self.memory_peak = self.memory_start

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        gc.collect()

        # Record results
        self.results = {
            "elapsed_time": self.end_time - self.start_time,
            "memory_start_mb": self.memory_start,
            "memory_peak_mb": self.memory_peak,
            "memory_used_mb": self._get_memory_usage() - self.memory_start,
            "method": self.method_name,
            "success": exc_type is None,
        }

        return False

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process(os.getpid())
            return float(process.memory_info().rss / 1024 / 1024)
        except ImportError:
            return 0.0

    def update_peak_memory(self):
        """Update peak memory usage."""
        current_memory = self._get_memory_usage()
        self.memory_peak = max(self.memory_peak, current_memory)


@pytest.fixture
def optimization_benchmark_context():
    """Provide optimization benchmark context manager."""

    def _create_context(
        method_name: str, test_name: str = ""
    ) -> OptimizationBenchmarkContext:
        return OptimizationBenchmarkContext(method_name, test_name)

    return _create_context


# Enhanced mock cores for cross-method testing
class UniversalOptimizationMockCore:
    """Universal mock core that works with all optimization methods."""

    def __init__(self, config, n_angles=20, n_times=80, seed=42, noise_level=0.01):
        self.config = config
        self.n_angles = n_angles
        self.n_times = n_times

        # Set reproducible seed
        np.random.seed(seed)

        # Generate consistent test data
        self.phi_angles = np.linspace(-45, 45, n_angles)
        self.time_delays = np.linspace(0, 10, n_times)
        self.true_parameters = np.array([125.0, -0.65, 18.0])
        self.c2_experimental = self._generate_test_data(noise_level)

    def _generate_test_data(self, noise_level=0.01):
        """Generate test data with known ground truth."""
        D0_true, alpha_true, D_offset_true = self.true_parameters
        c2_data = np.zeros((self.n_angles, self.n_times))

        for i, phi in enumerate(self.phi_angles):
            # Realistic correlation function
            D_eff = D0_true * self.time_delays ** abs(alpha_true) + D_offset_true
            decay = np.exp(-0.01 * D_eff * self.time_delays)

            angular_factor = 1.0 + 0.08 * np.cos(np.radians(phi))
            contrast = 0.22 * angular_factor
            c2_data[i, :] = 1.0 + contrast * decay

        # Add noise
        noise = np.random.normal(0, noise_level, c2_data.shape)
        c2_data += noise * np.mean(c2_data)

        return c2_data

    def compute_c2_correlation_optimized(self, params, phi_angles):
        """Compute correlation function (classical/MCMC compatibility)."""
        n_angles = len(phi_angles)
        c2_theory = np.zeros((n_angles, self.n_times))

        D0, alpha, D_offset = params[0], params[1], params[2]

        for i, phi in enumerate(phi_angles):
            D_eff = D0 * self.time_delays ** abs(alpha) + D_offset * self.time_delays
            decay = np.exp(-0.01 * D_eff * self.time_delays)

            angular_factor = 1.0 + 0.08 * np.cos(np.radians(phi))
            contrast = 0.22 * angular_factor
            c2_theory[i, :] = 1.0 + contrast * decay

        return c2_theory

    def calculate_chi_squared_optimized(self, params, phi_angles, c2_experimental):
        """Calculate chi-squared for optimization."""
        c2_theory = self.compute_c2_correlation_optimized(params, phi_angles)
        residuals = c2_experimental - c2_theory
        return np.sum(residuals**2) / c2_experimental.size

    def calculate_c2_nonequilibrium_laminar_parallel(self, params, phi_angles):
        """3D correlation function for robust optimization."""
        n_angles = len(phi_angles)
        c2_theory = np.zeros((n_angles, self.n_times, self.n_times))

        D0, alpha, D_offset = params[0], params[1], params[2]

        for i, phi in enumerate(phi_angles):
            angular_factor = 1.0 + 0.08 * np.cos(np.radians(phi))
            contrast = 0.22 * angular_factor

            for j in range(self.n_times):
                for k in range(self.n_times):
                    abs(self.time_delays[j] - self.time_delays[k])
                    D_eff = D0 * self.time_delays[min(j, k)] ** abs(alpha) + D_offset
                    decay = np.exp(-0.01 * D_eff)
                    c2_theory[i, j, k] = 1.0 + contrast * decay

        return c2_theory

    def is_static_mode(self):
        return True

    def get_effective_parameter_count(self):
        return 3

    @property
    def time_length(self):
        return self.n_times


@pytest.fixture
def universal_mock_core_small():
    """Small dataset universal mock core."""
    config = {"analysis_settings": {"mode": "static", "num_parameters": 3}}
    size_config = OPTIMIZATION_PERFORMANCE_CONFIG["dataset_sizes"]["small"]
    return UniversalOptimizationMockCore(
        config,
        n_angles=size_config["n_angles"],
        n_times=size_config["n_times"],
        seed=42,
    )


@pytest.fixture
def universal_mock_core_medium():
    """Medium dataset universal mock core."""
    config = {"analysis_settings": {"mode": "static", "num_parameters": 3}}
    size_config = OPTIMIZATION_PERFORMANCE_CONFIG["dataset_sizes"]["medium"]
    return UniversalOptimizationMockCore(
        config,
        n_angles=size_config["n_angles"],
        n_times=size_config["n_times"],
        seed=42,
    )


@pytest.fixture
def universal_mock_core_large():
    """Large dataset universal mock core."""
    config = {"analysis_settings": {"mode": "static", "num_parameters": 3}}
    size_config = OPTIMIZATION_PERFORMANCE_CONFIG["dataset_sizes"]["large"]
    return UniversalOptimizationMockCore(
        config,
        n_angles=size_config["n_angles"],
        n_times=size_config["n_times"],
        seed=42,
    )


def log_optimization_performance(method: str, results: dict[str, Any]) -> None:
    """Log optimization performance results."""
    import logging

    logger = logging.getLogger(__name__)

    logger.info(f"Performance Results for {method}:")
    for metric, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")


def compare_optimization_methods(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare performance across optimization methods."""
    if not results:
        return {}

    comparison = {
        "fastest_method": min(
            results, key=lambda x: x.get("elapsed_time", float("inf"))
        ),
        "most_memory_efficient": min(
            results, key=lambda x: x.get("memory_used_mb", float("inf"))
        ),
        "success_rate": sum(1 for r in results if r.get("success", False))
        / len(results),
        "average_time": np.mean(
            [r.get("elapsed_time", 0) for r in results if r.get("success", False)]
        ),
    }

    return comparison
