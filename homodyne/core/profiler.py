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


# Auto-cleanup when module is garbage collected
import atexit

atexit.register(clear_performance_stats)
