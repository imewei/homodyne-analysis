"""
Performance Profiling Utilities for Homodyne Package
===================================================

This module provides performance profiling and monitoring tools to help
identify bottlenecks and track optimization improvements.

Features:
- Function execution timing
- Memory usage monitoring
- Cache performance tracking
- Batch operation profiling

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import time
import functools
import logging
from typing import Dict, Any, Optional, Callable
import gc
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Global performance statistics
_performance_stats = {
    "function_times": {},
    "function_calls": {},
    "memory_usage": {},
    "cache_stats": {},
}


def profile_execution_time(func_name: Optional[str] = None):
    """
    Decorator to profile function execution time.

    Parameters
    ----------
    func_name : Optional[str]
        Custom name for the function (defaults to actual function name)

    Returns
    -------
    decorator
        Decorated function with timing
    """

    def decorator(func: Callable) -> Callable:
        name = func_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                execution_time = end_time - start_time

                # Update statistics
                if name not in _performance_stats["function_times"]:
                    _performance_stats["function_times"][name] = []
                    _performance_stats["function_calls"][name] = 0

                _performance_stats["function_times"][name].append(execution_time)
                _performance_stats["function_calls"][name] += 1

                # Log slow operations
                if execution_time > 1.0:  # Log operations taking more than 1 second
                    logger.info(f"Performance: {name} took {execution_time:.3f}s")
                elif execution_time > 0.1:  # Debug log for operations > 100ms
                    logger.debug(f"Performance: {name} took {execution_time:.3f}s")

        return wrapper

    return decorator


@contextmanager
def profile_memory_usage(operation_name: str):
    """
    Context manager to profile memory usage of an operation.

    Parameters
    ----------
    operation_name : str
        Name of the operation being profiled
    """
    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        yield

        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_diff = memory_after - memory_before

        # Update statistics
        _performance_stats["memory_usage"][operation_name] = {
            "before_mb": memory_before,
            "after_mb": memory_after,
            "diff_mb": memory_diff,
        }

        if abs(memory_diff) > 10:  # Log significant memory changes
            logger.info(
                f"Memory: {operation_name} changed memory by {memory_diff:.1f} MB"
            )

    except ImportError:
        logger.warning("psutil not available for memory profiling")
        yield


def profile_batch_operation(batch_size: int = 100):
    """
    Decorator to profile batch operations and find optimal batch sizes.

    Parameters
    ----------
    batch_size : int
        Size of batches to process

    Returns
    -------
    decorator
        Decorated function with batch profiling
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract data that needs to be batched (assume first argument is data)
            if args:
                data = args[0]
                if hasattr(data, "__len__") and len(data) > batch_size:
                    # Process in batches
                    results = []
                    total_items = len(data)

                    start_time = time.perf_counter()
                    for i in range(0, total_items, batch_size):
                        batch_data = data[i : i + batch_size]
                        batch_args = (batch_data,) + args[1:]
                        batch_result = func(*batch_args, **kwargs)
                        results.append(batch_result)

                    end_time = time.perf_counter()

                    # Log batch performance
                    total_time = end_time - start_time
                    items_per_second = total_items / total_time if total_time > 0 else 0
                    logger.debug(
                        f"Batch processing: {total_items} items in {batch_size} batches, "
                        f"{items_per_second:.1f} items/sec"
                    )

                    # Combine results if they are lists/arrays
                    if results and hasattr(results[0], "__len__"):
                        try:
                            import numpy as np

                            return np.concatenate(results)
                        except (ImportError, ValueError):
                            return [item for sublist in results for item in sublist]

                    return results

            # Fall back to normal execution
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_performance_summary() -> Dict[str, Any]:
    """
    Get a summary of performance statistics.

    Returns
    -------
    Dict[str, Any]
        Performance statistics summary
    """
    summary = {}

    # Function timing statistics
    for func_name, times in _performance_stats["function_times"].items():
        if times:
            import statistics

            summary[func_name] = {
                "calls": _performance_stats["function_calls"][func_name],
                "total_time": sum(times),
                "avg_time": statistics.mean(times),
                "median_time": statistics.median(times),
                "max_time": max(times),
                "min_time": min(times),
            }

    # Memory usage statistics
    summary["memory_usage"] = _performance_stats["memory_usage"]

    return summary


def clear_performance_stats():
    """Clear all performance statistics."""
    global _performance_stats
    _performance_stats = {
        "function_times": {},
        "function_calls": {},
        "memory_usage": {},
        "cache_stats": {},
    }


def log_performance_summary():
    """Log a summary of performance statistics."""
    summary = get_performance_summary()

    if summary:
        logger.info("=== Performance Summary ===")
        for func_name, stats in summary.items():
            if isinstance(stats, dict) and "calls" in stats:
                logger.info(
                    f"{func_name}: {stats['calls']} calls, "
                    f"avg: {stats['avg_time']:.3f}s, "
                    f"total: {stats['total_time']:.3f}s"
                )

        if summary.get("memory_usage"):
            logger.info("Memory usage changes:")
            for op_name, mem_stats in summary["memory_usage"].items():
                logger.info(f"{op_name}: {mem_stats['diff_mb']:.1f} MB")

    logger.info("=========================")


def stable_benchmark(
    func: Callable,
    warmup_runs: int = 3,
    measurement_runs: int = 10,
    outlier_threshold: float = 2.0,
) -> Dict[str, Any]:
    """
    Perform stable benchmarking with outlier filtering and warmup.

    This function provides more robust benchmarking than simple timing by:
    - Running warmup iterations to stabilize JIT compilation
    - Filtering statistical outliers for more reliable measurements
    - Providing comprehensive statistics including percentiles

    Parameters
    ----------
    func : callable
        Function to benchmark
    warmup_runs : int, default=3
        Number of warmup runs before measurement
    measurement_runs : int, default=10
        Number of measurement runs
    outlier_threshold : float, default=2.0
        Standard deviations beyond which results are considered outliers

    Returns
    -------
    dict
        Benchmark results including mean, median, std, and filtered statistics

    Examples
    --------
    >>> def compute_something():
    ...     return np.sum(np.random.rand(1000, 1000))
    >>> results = stable_benchmark(compute_something, warmup_runs=3, measurement_runs=10)
    >>> print(f"Mean: {results['mean']:.4f}s, Outliers: {results['outlier_count']}")
    """
    import numpy as np

    # Warmup runs to stabilize performance (JIT, cache, etc.)
    logger.debug(f"Performing {warmup_runs} warmup runs...")
    for i in range(warmup_runs):
        try:
            _ = func()
            gc.collect()  # Consistent garbage collection state
        except Exception as e:
            logger.warning(f"Warmup run {i+1} failed: {e}")

    # Measurement runs
    logger.debug(f"Performing {measurement_runs} measurement runs...")
    times = []
    result = None  # Ensure result is always defined
    for i in range(measurement_runs):
        gc.collect()
        start_time = time.perf_counter()
        result = func()
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    times = np.array(times)

    # Calculate statistics
    mean_time = np.mean(times)
    median_time = np.median(times)
    std_time = np.std(times)

    # Filter outliers
    outlier_mask = np.abs(times - mean_time) > outlier_threshold * std_time
    filtered_times = times[~outlier_mask]

    if len(filtered_times) > 0:
        filtered_mean = np.mean(filtered_times)
        filtered_std = np.std(filtered_times)
    else:
        filtered_mean = mean_time
        filtered_std = std_time

    return {
        "result": result,  # Last result for validation
        "times": times,
        "mean": mean_time,
        "median": median_time,
        "std": std_time,
        "min": np.min(times),
        "max": np.max(times),
        "outlier_ratio": (
            np.max(times) / np.min(times) if np.min(times) > 0 else float("inf")
        ),
        "outlier_count": np.sum(outlier_mask),
        "filtered_mean": filtered_mean,
        "filtered_std": filtered_std,
        "percentile_95": np.percentile(times, 95),
        "percentile_99": np.percentile(times, 99),
    }


def optimize_numerical_environment() -> Dict[str, str]:
    """
    Optimize the numerical computation environment for consistent performance.

    This function sets environment variables and configurations that help
    reduce performance variance in numerical computations by:
    - Controlling threading in BLAS libraries
    - Setting consistent random seeds
    - Optimizing garbage collection
    - Configuring JIT compilation settings
    - Setting CPU affinity and memory optimizations

    Returns
    -------
    dict
        Dictionary of applied optimizations

    Examples
    --------
    >>> optimizations = optimize_numerical_environment()
    >>> print(f"Applied {len(optimizations)} optimizations")
    """
    import os

    optimizations = {}

    # Conservative threading optimizations for maximum stability
    import multiprocessing

    n_cores = min(
        multiprocessing.cpu_count() // 2, 4
    )  # Use half cores, cap at 4 for stability

    threading_vars = {
        "OPENBLAS_NUM_THREADS": str(n_cores),
        "MKL_NUM_THREADS": str(n_cores),
        "NUMEXPR_NUM_THREADS": str(n_cores),
        "OMP_NUM_THREADS": str(n_cores),
        "NUMBA_NUM_THREADS": str(n_cores),
    }

    for var, value in threading_vars.items():
        if var not in os.environ:
            os.environ[var] = value
            optimizations[var] = value
            logger.debug(f"Set {var}={value}")
        else:
            logger.debug(f"Skipped {var} (already set to {os.environ[var]})")

    # Numba JIT optimization settings balanced for performance and stability
    numba_vars = {
        "NUMBA_CACHE_DIR": os.path.join(os.path.expanduser("~"), ".numba_cache"),
        "NUMBA_DISABLE_INTEL_SVML": "1",  # Disable SVML for consistent performance
        "NUMBA_DISABLE_TBB": "0",  # Enable TBB for better parallel performance
        "NUMBA_BOUNDSCHECK": "0",  # Disable bounds checking for performance
        "NUMBA_FASTMATH": "0",  # Disable fast math for better numerical stability
        "NUMBA_LOOP_VECTORIZE": "1",  # Enable loop vectorization (stable)
        "NUMBA_OPT": "2",  # Moderate optimization level for balance
    }

    for var, value in numba_vars.items():
        if var not in os.environ:
            os.environ[var] = value
            optimizations[var] = value
            logger.debug(f"Set {var}={value}")

    # NumPy optimizations
    try:
        import numpy as np

        # Use consistent random seed for reproducible benchmarks
        np.random.seed(42)
        optimizations["numpy_random_seed"] = "42"

        # Configure numpy error handling for consistent behavior
        old_settings = np.seterr(all="warn")  # Store old settings
        optimizations["numpy_error_handling"] = "configured"

        # Set BLAS threading model
        try:
            np.show_config()  # This helps ensure BLAS is properly configured
            optimizations["numpy_blas_check"] = "verified"
        except:
            pass

        logger.debug("Configured NumPy for consistent performance")

    except ImportError:
        logger.debug("NumPy not available for optimization")

    # Garbage collection optimization for consistent memory behavior
    old_threshold = gc.get_threshold()
    gc.set_threshold(1000, 15, 15)  # Slightly less aggressive GC for performance
    optimizations["gc_threshold"] = f"{old_threshold} -> (1000, 15, 15)"

    # Python optimization settings
    import sys

    if hasattr(sys, "setswitchinterval"):
        sys.setswitchinterval(0.01)  # Reduce thread switching overhead
        optimizations["python_switch_interval"] = "0.01"

    logger.info(f"Applied {len(optimizations)} numerical environment optimizations")
    return optimizations


def assert_performance_within_bounds(
    measured_time: float,
    expected_time: float,
    tolerance_factor: float = 2.0,
    test_name: str = "performance_test",
    enable_baseline_tracking: bool = False,
):
    """
    Assert that measured performance is within acceptable bounds.

    This function provides a standardized way to validate performance in tests
    by checking that execution time doesn't exceed expected bounds.

    Parameters
    ----------
    measured_time : float
        Measured execution time in seconds
    expected_time : float
        Expected execution time in seconds
    tolerance_factor : float, default=2.0
        Acceptable factor by which measured time can exceed expected time
    test_name : str
        Name of the test for error messaging
    enable_baseline_tracking : bool, default=False
        Whether to track baselines for regression detection

    Raises
    ------
    AssertionError
        If measured time exceeds tolerance bounds

    Examples
    --------
    >>> measured = 0.05  # 50ms
    >>> expected = 0.02  # 20ms
    >>> assert_performance_within_bounds(measured, expected, tolerance_factor=3.0)
    """
    import warnings

    max_acceptable_time = expected_time * tolerance_factor

    assert measured_time <= max_acceptable_time, (
        f"{test_name} performance regression: "
        f"measured {measured_time:.4f}s > expected {expected_time:.4f}s * {tolerance_factor} = {max_acceptable_time:.4f}s"
    )

    # Also check if performance is suspiciously good (might indicate incorrect measurement)
    min_reasonable_time = expected_time / 100  # Allow up to 100x speedup
    if measured_time < min_reasonable_time:
        warnings.warn(
            f"{test_name} suspiciously fast: {measured_time:.6f}s << expected {expected_time:.4f}s. "
            "Check measurement accuracy.",
            RuntimeWarning,
        )

    # Optional baseline tracking for regression detection
    if enable_baseline_tracking:
        try:
            from pathlib import Path
            import json

            baseline_file = Path("performance_baselines.json")
            baselines = {}

            if baseline_file.exists():
                try:
                    with open(baseline_file) as f:
                        baselines = json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass

            # Record current measurement
            if test_name not in baselines:
                baselines[test_name] = {}
            baselines[test_name]["measured_time"] = measured_time
            baselines[test_name]["expected_time"] = expected_time

            # Save updated baselines
            try:
                with open(baseline_file, "w") as f:
                    json.dump(baselines, f, indent=2)
            except IOError:
                logger.warning(
                    f"Could not save performance baselines to {baseline_file}"
                )

        except Exception as e:
            logger.debug(f"Baseline tracking failed for {test_name}: {e}")

    logger.debug(
        f"Performance assertion passed: {test_name} = {measured_time:.4f}s (expected ~{expected_time:.4f}s)"
    )


def assert_performance_stability(
    times: list,
    max_cv: float = 0.5,  # 50% coefficient of variation
    test_name: str = "stability_test",
):
    """
    Assert that performance measurements are stable (low variance).

    This function validates that a series of performance measurements
    have acceptably low variance, indicating consistent performance.

    Parameters
    ----------
    times : list of float
        List of measured execution times
    max_cv : float, default=0.5
        Maximum acceptable coefficient of variation (std/mean)
    test_name : str
        Name of the test for error messaging

    Raises
    ------
    AssertionError
        If performance variance is too high

    Examples
    --------
    >>> times = [0.020, 0.021, 0.019, 0.022, 0.020]  # Consistent times
    >>> assert_performance_stability(times, max_cv=0.1)  # Allow 10% variation
    """
    import numpy as np

    times_array = np.array(times)
    mean_time = np.mean(times_array)
    std_time = np.std(times_array)
    cv = std_time / mean_time if mean_time > 0 else float("inf")

    assert cv <= max_cv, (
        f"{test_name} performance too variable: "
        f"coefficient of variation {cv:.3f} > max allowed {max_cv:.3f} "
        f"(std={std_time:.4f}s, mean={mean_time:.4f}s)"
    )

    logger.debug(
        f"Performance stability assertion passed: {test_name} CV = {cv:.3f} (max {max_cv:.3f})"
    )


def jit_warmup(func: Callable, *args, warmup_runs: int = 5, **kwargs) -> Any:
    """
    Perform JIT compilation warmup for Numba-compiled functions.

    This function helps stabilize performance by ensuring JIT compilation
    is complete before benchmarking or critical calculations.

    Parameters
    ----------
    func : callable
        Function to warm up (should be JIT-compiled)
    *args : tuple
        Arguments to pass to the function
    warmup_runs : int, default=5
        Number of warmup calls to perform
    **kwargs : dict
        Keyword arguments to pass to the function

    Returns
    -------
    Any
        Result of the last warmup call

    Examples
    --------
    >>> @njit
    >>> def compute_matrix(x):
    ...     return x @ x.T
    >>> x = np.random.rand(100, 100)
    >>> result = jit_warmup(compute_matrix, x, warmup_runs=3)
    """
    logger.debug(
        f"JIT warmup: performing {warmup_runs} warmup calls for {func.__name__}"
    )

    result = None
    for i in range(warmup_runs):
        try:
            result = func(*args, **kwargs)
            if i == 0:
                logger.debug(f"JIT warmup: first call completed for {func.__name__}")
            gc.collect()  # Clean up after each warmup call
        except Exception as e:
            logger.warning(
                f"JIT warmup failed on call {i+1}/{warmup_runs} for {func.__name__}: {e}"
            )
            if i == warmup_runs - 1:  # Last attempt
                raise

    logger.debug(f"JIT warmup completed for {func.__name__}")
    return result


def create_stable_benchmark_config(test_type: str = "default") -> Dict[str, int]:
    """
    Create stable benchmarking configuration based on test type.

    Different types of tests need different benchmarking strategies for
    optimal stability vs. speed tradeoff.

    Parameters
    ----------
    test_type : str, default="default"
        Type of test: "quick", "default", "thorough", "ci"

    Returns
    -------
    dict
        Benchmarking configuration with warmup_runs, measurement_runs, etc.

    Examples
    --------
    >>> config = create_stable_benchmark_config("thorough")
    >>> results = stable_benchmark(func, **config)
    """
    import os

    # Detect CI environment
    is_ci = any(
        var in os.environ for var in ["CI", "GITHUB_ACTIONS", "TRAVIS", "APPVEYOR"]
    )

    configs = {
        "quick": {"warmup_runs": 2, "measurement_runs": 5, "outlier_threshold": 3.0},
        "default": {"warmup_runs": 5, "measurement_runs": 10, "outlier_threshold": 2.5},
        "thorough": {
            "warmup_runs": 10,
            "measurement_runs": 20,
            "outlier_threshold": 2.0,
        },
        "ci": {
            "warmup_runs": 8,
            "measurement_runs": 15,
            "outlier_threshold": 3.0,  # More tolerance for CI variability
        },
    }

    # Override with CI config if in CI environment
    if is_ci and test_type != "ci":
        logger.debug(f"CI environment detected, using CI config instead of {test_type}")
        test_type = "ci"

    config = configs.get(test_type, configs["default"])
    logger.debug(f"Created {test_type} benchmark config: {config}")
    return config


def adaptive_stable_benchmark(
    func: Callable,
    initial_runs: int = 5,
    target_cv: float = 0.15,
    max_runs: int = 50,
    min_runs: int = 10,
) -> Dict[str, Any]:
    """
    Perform adaptive benchmarking that continues until stable results are achieved.

    This function automatically determines the optimal number of measurement runs
    needed to achieve stable performance measurements within a target CV.

    Parameters
    ----------
    func : callable
        Function to benchmark
    initial_runs : int, default=5
        Initial number of measurement runs
    target_cv : float, default=0.15
        Target coefficient of variation (15%)
    max_runs : int, default=50
        Maximum number of measurement runs
    min_runs : int, default=10
        Minimum number of measurement runs

    Returns
    -------
    dict
        Benchmark results with adaptive statistics

    Examples
    --------
    >>> def compute_something():
    ...     return np.sum(np.random.rand(1000, 1000))
    >>> results = adaptive_stable_benchmark(compute_something, target_cv=0.1)
    >>> print(f"Achieved CV: {results['cv']:.3f} in {results['total_runs']} runs")
    """
    import numpy as np

    logger.debug(f"Starting adaptive benchmark with target CV={target_cv:.3f}")

    # Initial warmup
    config = create_stable_benchmark_config("quick")
    for _ in range(config["warmup_runs"]):
        try:
            _ = func()
            gc.collect()
        except Exception as e:
            logger.warning(f"Adaptive benchmark warmup failed: {e}")

    # Collect initial measurements
    times = []
    result = None

    for run in range(max_runs):
        gc.collect()
        start_time = time.perf_counter()
        result = func()
        end_time = time.perf_counter()
        times.append(end_time - start_time)

        # Check if we have enough runs to assess stability
        if len(times) >= min_runs:
            times_array = np.array(times)
            mean_time = np.mean(times_array)
            std_time = np.std(times_array)
            cv = std_time / mean_time if mean_time > 0 else float("inf")

            # Check if we've achieved target stability
            if cv <= target_cv:
                logger.debug(
                    f"Adaptive benchmark achieved target CV={cv:.3f} in {len(times)} runs"
                )
                break

            # Also check if we've done enough runs
            if len(times) >= initial_runs * 2 and cv <= target_cv * 1.5:
                logger.debug(
                    f"Adaptive benchmark achieved acceptable CV={cv:.3f} in {len(times)} runs"
                )
                break

    # Calculate final statistics
    times_array = np.array(times)
    mean_time = np.mean(times_array)
    std_time = np.std(times_array)
    cv = std_time / mean_time if mean_time > 0 else float("inf")

    # Filter outliers for final analysis
    outlier_threshold = 2.0
    outlier_mask = np.abs(times_array - mean_time) > outlier_threshold * std_time
    filtered_times = times_array[~outlier_mask]

    return {
        "result": result,
        "times": times_array,
        "mean": mean_time,
        "median": np.median(times_array),
        "std": std_time,
        "cv": cv,
        "total_runs": len(times),
        "target_cv": target_cv,
        "achieved_target": cv <= target_cv,
        "outlier_count": np.sum(outlier_mask),
        "filtered_mean": (
            np.mean(filtered_times) if len(filtered_times) > 0 else mean_time
        ),
        "percentile_95": np.percentile(times_array, 95),
    }


def memory_efficient_operation(max_memory_mb: float = 1000.0):
    """
    Decorator to ensure operations don't exceed memory limits.

    This decorator monitors memory usage and triggers garbage collection
    if memory usage exceeds the specified threshold.

    Parameters
    ----------
    max_memory_mb : float, default=1000.0
        Maximum memory usage in MB before triggering cleanup

    Examples
    --------
    >>> @memory_efficient_operation(max_memory_mb=500.0)
    >>> def process_large_array(arr):
    ...     return arr @ arr.T
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                import psutil
                import os

                process = psutil.Process(os.getpid())
                initial_memory = process.memory_info().rss / 1024 / 1024

                result = func(*args, **kwargs)

                current_memory = process.memory_info().rss / 1024 / 1024
                memory_diff = current_memory - initial_memory

                if current_memory > max_memory_mb:
                    logger.debug(
                        f"Memory threshold exceeded ({current_memory:.1f}MB > {max_memory_mb:.1f}MB), triggering GC"
                    )
                    gc.collect()

                    # Check memory again after GC
                    post_gc_memory = process.memory_info().rss / 1024 / 1024
                    logger.debug(
                        f"Memory after GC: {post_gc_memory:.1f}MB (freed {current_memory - post_gc_memory:.1f}MB)"
                    )

                return result

            except ImportError:
                # Fallback without memory monitoring
                return func(*args, **kwargs)

        return wrapper

    return decorator


class PerformanceCache:
    """
    Smart performance-aware cache with automatic cleanup.

    This cache implementation considers both memory usage and access patterns
    to optimize performance while preventing memory bloat.
    """

    def __init__(self, max_items: int = 100, max_memory_mb: float = 500.0):
        self.max_items = max_items
        self.max_memory_mb = max_memory_mb
        self.cache = {}
        self.access_times = {}
        self.memory_usage = {}
        self._total_memory = 0.0

    def _estimate_memory(self, obj) -> float:
        """Estimate memory usage of an object in MB."""
        try:
            import sys

            size_bytes = sys.getsizeof(obj)

            # For numpy arrays, use more accurate size
            if hasattr(obj, "nbytes"):
                size_bytes = obj.nbytes

            return size_bytes / 1024 / 1024  # Convert to MB
        except:
            return 1.0  # Fallback estimate

    def _cleanup_cache(self):
        """Remove least recently used items to free memory."""
        if not self.cache:
            return

        # Sort by access time (oldest first)
        items_by_access = sorted(self.access_times.items(), key=lambda x: x[1])

        # Remove oldest items until we're under limits
        for key, _ in items_by_access:
            if (
                len(self.cache) <= self.max_items * 0.8
                and self._total_memory <= self.max_memory_mb * 0.8
            ):
                break

            if key in self.cache:
                memory_freed = self.memory_usage.get(key, 0)
                del self.cache[key]
                del self.access_times[key]
                if key in self.memory_usage:
                    del self.memory_usage[key]
                self._total_memory -= memory_freed

        logger.debug(
            f"Cache cleanup: {len(self.cache)} items, {self._total_memory:.1f}MB"
        )

    def get(self, key: str, default=None):
        """Get item from cache with access time tracking."""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return default

    def put(self, key: str, value):
        """Put item in cache with memory tracking."""
        memory_size = self._estimate_memory(value)

        # Remove existing item if present
        if key in self.cache:
            self._total_memory -= self.memory_usage.get(key, 0)

        # Add new item
        self.cache[key] = value
        self.access_times[key] = time.time()
        self.memory_usage[key] = memory_size
        self._total_memory += memory_size

        # Cleanup if necessary
        if len(self.cache) > self.max_items or self._total_memory > self.max_memory_mb:
            self._cleanup_cache()

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.access_times.clear()
        self.memory_usage.clear()
        self._total_memory = 0.0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "items": len(self.cache),
            "memory_mb": self._total_memory,
            "max_items": self.max_items,
            "max_memory_mb": self.max_memory_mb,
            "utilization": (
                len(self.cache) / self.max_items if self.max_items > 0 else 0
            ),
            "memory_utilization": (
                self._total_memory / self.max_memory_mb if self.max_memory_mb > 0 else 0
            ),
        }


# Global performance cache instance
_performance_cache = PerformanceCache()


def get_performance_cache() -> PerformanceCache:
    """Get the global performance cache instance."""
    return _performance_cache


def performance_monitor(
    monitor_memory: bool = True,
    monitor_time: bool = True,
    log_threshold_seconds: float = 1.0,
):
    """
    Advanced performance monitoring decorator.

    This decorator provides comprehensive performance monitoring including
    memory usage, execution time, and stability tracking.

    Parameters
    ----------
    monitor_memory : bool, default=True
        Whether to monitor memory usage
    monitor_time : bool, default=True
        Whether to monitor execution time
    log_threshold_seconds : float, default=1.0
        Threshold for logging slow operations

    Examples
    --------
    >>> @performance_monitor(monitor_memory=True, log_threshold_seconds=0.5)
    >>> def heavy_computation(data):
    ...     return process_data(data)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter() if monitor_time else None
            initial_memory = None

            if monitor_memory:
                try:
                    import psutil
                    import os

                    process = psutil.Process(os.getpid())
                    initial_memory = process.memory_info().rss / 1024 / 1024
                except ImportError:
                    pass

            try:
                result = func(*args, **kwargs)

                # Performance logging
                if monitor_time and start_time is not None:
                    execution_time = time.perf_counter() - start_time

                    # Update global stats
                    func_name = func.__name__
                    if func_name not in _performance_stats["function_times"]:
                        _performance_stats["function_times"][func_name] = []
                        _performance_stats["function_calls"][func_name] = 0

                    _performance_stats["function_times"][func_name].append(
                        execution_time
                    )
                    _performance_stats["function_calls"][func_name] += 1

                    # Log if above threshold
                    if execution_time >= log_threshold_seconds:
                        logger.info(
                            f"Performance monitor: {func_name} took {execution_time:.3f}s"
                        )

                if monitor_memory and initial_memory is not None:
                    try:
                        final_memory = process.memory_info().rss / 1024 / 1024
                        memory_diff = final_memory - initial_memory

                        if (
                            abs(memory_diff) > 50
                        ):  # Log significant memory changes (>50MB)
                            logger.info(
                                f"Memory monitor: {func.__name__} changed memory by {memory_diff:+.1f}MB"
                            )

                    except:
                        pass

                return result

            except Exception as e:
                if monitor_time and start_time is not None:
                    execution_time = time.perf_counter() - start_time
                    logger.error(
                        f"Function {func.__name__} failed after {execution_time:.3f}s: {e}"
                    )
                raise

        return wrapper

    return decorator


# Auto-cleanup when module is garbage collected
import atexit


def cleanup_performance_resources():
    """Clean up performance monitoring resources."""
    clear_performance_stats()
    _performance_cache.clear()


atexit.register(cleanup_performance_resources)
