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
Institution: Argonne National Laboratory & University of Chicago
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


def stable_benchmark(func: Callable, 
                    warmup_runs: int = 3,
                    measurement_runs: int = 10,
                    outlier_threshold: float = 2.0) -> Dict[str, Any]:
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
        'result': result,  # Last result for validation
        'times': times,
        'mean': mean_time,
        'median': median_time, 
        'std': std_time,
        'min': np.min(times),
        'max': np.max(times),
        'outlier_ratio': np.max(times) / np.min(times) if np.min(times) > 0 else float('inf'),
        'outlier_count': np.sum(outlier_mask),
        'filtered_mean': filtered_mean,
        'filtered_std': filtered_std,
        'percentile_95': np.percentile(times, 95),
        'percentile_99': np.percentile(times, 99)
    }


def optimize_numerical_environment() -> Dict[str, str]:
    """
    Optimize the numerical computation environment for consistent performance.
    
    This function sets environment variables and configurations that help
    reduce performance variance in numerical computations by:
    - Controlling threading in BLAS libraries
    - Setting consistent random seeds
    - Optimizing garbage collection
    
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
    
    # Threading optimizations for consistency
    threading_vars = {
        'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 
        'NUMEXPR_NUM_THREADS': '1',
        'OMP_NUM_THREADS': '1'
    }
    
    for var, value in threading_vars.items():
        if var not in os.environ:
            os.environ[var] = value
            optimizations[var] = value
            logger.debug(f"Set {var}={value}")
            
    # NumPy optimizations
    try:
        import numpy as np
        
        # Use consistent random seed for reproducible benchmarks
        np.random.seed(42)
        optimizations['numpy_random_seed'] = '42'
        
        # Configure numpy error handling for consistent behavior
        old_settings = np.seterr(all='warn')  # Store old settings
        optimizations['numpy_error_handling'] = 'configured'
        
        logger.debug("Configured NumPy for consistent performance")
        
    except ImportError:
        logger.debug("NumPy not available for optimization")
        
    # Garbage collection optimization for consistent memory behavior
    old_threshold = gc.get_threshold()
    gc.set_threshold(700, 10, 10)  # More frequent GC for consistent memory
    optimizations['gc_threshold'] = f"{old_threshold} -> (700, 10, 10)"
    
    logger.info(f"Applied {len(optimizations)} numerical environment optimizations")
    return optimizations


def assert_performance_within_bounds(measured_time: float,
                                    expected_time: float, 
                                    tolerance_factor: float = 2.0,
                                    test_name: str = "performance_test",
                                    enable_baseline_tracking: bool = False):
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
            RuntimeWarning
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
                logger.warning(f"Could not save performance baselines to {baseline_file}")
                
        except Exception as e:
            logger.debug(f"Baseline tracking failed for {test_name}: {e}")
    
    logger.debug(f"Performance assertion passed: {test_name} = {measured_time:.4f}s (expected ~{expected_time:.4f}s)")


def assert_performance_stability(times: list,
                               max_cv: float = 0.5,  # 50% coefficient of variation
                               test_name: str = "stability_test"):
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
    cv = std_time / mean_time if mean_time > 0 else float('inf')
    
    assert cv <= max_cv, (
        f"{test_name} performance too variable: "
        f"coefficient of variation {cv:.3f} > max allowed {max_cv:.3f} "
        f"(std={std_time:.4f}s, mean={mean_time:.4f}s)"
    )
    
    logger.debug(f"Performance stability assertion passed: {test_name} CV = {cv:.3f} (max {max_cv:.3f})")


# Auto-cleanup when module is garbage collected
import atexit

atexit.register(clear_performance_stats)
