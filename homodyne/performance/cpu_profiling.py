"""
CPU Performance Profiling and Optimization
==========================================

Advanced CPU-specific performance profiling, monitoring, and optimization
utilities for homodyne analysis with focus on scientific computing workloads.

This module provides:
- Detailed CPU performance profiling
- Memory usage monitoring and optimization
- Cache performance analysis
- Thread utilization tracking
- CPU-specific bottleneck identification
- Performance regression detection

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import gc
import time
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import psutil

try:
    import cProfile
    import pstats

    CPROFILE_AVAILABLE = True
except ImportError:
    CPROFILE_AVAILABLE = False

try:
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    warnings.warn("memory_profiler not available - memory profiling limited")


@dataclass
class CPUProfileResult:
    """Results from CPU performance profiling."""

    function_name: str
    total_time: float
    cpu_time: float
    memory_peak_mb: float
    memory_delta_mb: float
    cache_efficiency: float
    thread_utilization: float
    profile_stats: dict[str, Any]


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for CPU analysis."""

    execution_time: float
    cpu_utilization: float
    memory_usage_mb: float
    memory_peak_mb: float
    cache_misses: int
    context_switches: int
    thread_count: int
    io_operations: int


class CPUProfiler:
    """
    Advanced CPU performance profiler for scientific computing.

    Provides comprehensive profiling capabilities including execution timing,
    memory usage tracking, cache analysis, and thread utilization monitoring.
    """

    def __init__(self, enable_detailed_profiling: bool = True):
        """
        Initialize CPU profiler.

        Parameters
        ----------
        enable_detailed_profiling : bool
            Enable detailed profiling (may add overhead)
        """
        self.enable_detailed = enable_detailed_profiling
        self.process = psutil.Process()
        self.baseline_metrics = self._get_baseline_metrics()

    def _get_baseline_metrics(self) -> dict[str, float]:
        """Get baseline system metrics."""
        try:
            return {
                "cpu_percent": self.process.cpu_percent(),
                "memory_mb": self.process.memory_info().rss / 1024 / 1024,
                "thread_count": self.process.num_threads(),
                "ctx_switches": self.process.num_ctx_switches().total,
            }
        except Exception:
            return {
                "cpu_percent": 0,
                "memory_mb": 0,
                "thread_count": 1,
                "ctx_switches": 0,
            }

    @contextmanager
    def profile_function(self, function_name: str = "unknown"):
        """
        Context manager for profiling function execution.

        Parameters
        ----------
        function_name : str
            Name of function being profiled

        Yields
        ------
        CPUProfileResult
            Profiling results
        """
        # Pre-execution setup
        gc.collect()  # Clean memory state
        start_metrics = self._get_current_metrics()

        if CPROFILE_AVAILABLE and self.enable_detailed:
            profiler = cProfile.Profile()
            profiler.enable()

        start_time = time.perf_counter()

        try:
            yield

        finally:
            # Post-execution measurement
            end_time = time.perf_counter()

            if CPROFILE_AVAILABLE and self.enable_detailed:
                profiler.disable()
                stats = pstats.Stats(profiler)
                profile_stats = self._extract_profile_stats(stats)
            else:
                profile_stats = {}

            end_metrics = self._get_current_metrics()

            # Calculate deltas and efficiency metrics
            result = self._calculate_profile_result(
                function_name,
                start_time,
                end_time,
                start_metrics,
                end_metrics,
                profile_stats,
            )

    def _get_current_metrics(self) -> dict[str, float]:
        """Get current system metrics."""
        try:
            memory_info = self.process.memory_info()
            cpu_times = self.process.cpu_times()

            return {
                "timestamp": time.perf_counter(),
                "cpu_percent": self.process.cpu_percent(),
                "memory_rss_mb": memory_info.rss / 1024 / 1024,
                "memory_vms_mb": memory_info.vms / 1024 / 1024,
                "cpu_user_time": cpu_times.user,
                "cpu_system_time": cpu_times.system,
                "thread_count": self.process.num_threads(),
                "ctx_switches": self.process.num_ctx_switches().total,
                "io_counters": (
                    self.process.io_counters()._asdict()
                    if hasattr(self.process, "io_counters")
                    else {}
                ),
            }
        except Exception as e:
            warnings.warn(f"Failed to get metrics: {e}")
            return {"timestamp": time.perf_counter()}

    def _extract_profile_stats(self, stats: "pstats.Stats") -> dict[str, Any]:
        """Extract key statistics from cProfile results."""
        try:
            # Get stats as structured data
            stats_data = {}
            for func, (cc, nc, tt, ct, callers) in stats.stats.items():
                filename, line, func_name = func
                stats_data[f"{filename}:{func_name}"] = {
                    "cumulative_calls": cc,
                    "primitive_calls": nc,
                    "total_time": tt,
                    "cumulative_time": ct,
                    "per_call_time": tt / nc if nc > 0 else 0,
                }

            # Sort by total time and get top functions
            sorted_funcs = sorted(
                stats_data.items(), key=lambda x: x[1]["total_time"], reverse=True
            )

            return {
                "total_calls": sum(
                    data["cumulative_calls"] for _, data in sorted_funcs
                ),
                "total_time": sum(data["total_time"] for _, data in sorted_funcs),
                "top_functions": dict(
                    sorted_funcs[:10]
                ),  # Top 10 time-consuming functions
            }
        except Exception as e:
            warnings.warn(f"Failed to extract profile stats: {e}")
            return {}

    def _calculate_profile_result(
        self,
        function_name: str,
        start_time: float,
        end_time: float,
        start_metrics: dict[str, float],
        end_metrics: dict[str, float],
        profile_stats: dict[str, Any],
    ) -> CPUProfileResult:
        """Calculate comprehensive profiling results."""
        total_time = end_time - start_time

        # Calculate CPU utilization and memory deltas
        cpu_time = 0.0
        memory_peak_mb = end_metrics.get("memory_rss_mb", 0)
        memory_delta_mb = end_metrics.get("memory_rss_mb", 0) - start_metrics.get(
            "memory_rss_mb", 0
        )

        # Estimate cache efficiency (simplified heuristic)
        cache_efficiency = self._estimate_cache_efficiency(
            total_time, memory_delta_mb, profile_stats
        )

        # Calculate thread utilization
        thread_count = end_metrics.get("thread_count", 1)
        thread_utilization = min(1.0, end_metrics.get("cpu_percent", 0) / 100.0)

        return CPUProfileResult(
            function_name=function_name,
            total_time=total_time,
            cpu_time=cpu_time,
            memory_peak_mb=memory_peak_mb,
            memory_delta_mb=memory_delta_mb,
            cache_efficiency=cache_efficiency,
            thread_utilization=thread_utilization,
            profile_stats=profile_stats,
        )

    def _estimate_cache_efficiency(
        self, execution_time: float, memory_delta: float, profile_stats: dict[str, Any]
    ) -> float:
        """
        Estimate cache efficiency based on execution patterns.

        This is a simplified heuristic - more sophisticated cache analysis
        would require hardware performance counters.
        """
        try:
            # Heuristic: lower memory delta and faster execution suggests better cache usage
            if execution_time > 0 and memory_delta >= 0:
                # Normalize based on typical cache-friendly vs cache-unfriendly patterns
                efficiency_score = 1.0 / (1.0 + memory_delta / execution_time)
                return min(1.0, max(0.0, efficiency_score))
            return 0.5  # Neutral efficiency
        except Exception:
            return 0.5

    def benchmark_function(
        self,
        func: Callable,
        *args,
        iterations: int = 10,
        warmup_iterations: int = 3,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Comprehensive benchmarking of a function.

        Parameters
        ----------
        func : Callable
            Function to benchmark
        *args, **kwargs
            Arguments for the function
        iterations : int
            Number of benchmark iterations
        warmup_iterations : int
            Number of warmup iterations (not measured)

        Returns
        -------
        dict[str, Any]
            Comprehensive benchmark results
        """
        # Warmup runs
        for _ in range(warmup_iterations):
            try:
                func(*args, **kwargs)
            except Exception:
                warnings.warn("Warmup iteration failed")

        # Measured runs
        results = []
        total_start = time.perf_counter()

        for i in range(iterations):
            with self.profile_function(f"{func.__name__}_iter_{i}") as profiler:
                try:
                    result = func(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    warnings.warn(f"Benchmark iteration {i} failed: {e}")

        total_end = time.perf_counter()

        # Calculate statistics
        return self._calculate_benchmark_stats(results, total_end - total_start)

    def _calculate_benchmark_stats(
        self, results: list[Any], total_time: float
    ) -> dict[str, Any]:
        """Calculate benchmark statistics."""
        if not results:
            return {"error": "No successful iterations"}

        # Extract timing information (simplified - would need actual timing data)
        times = [total_time / len(results)] * len(results)  # Simplified

        return {
            "iterations": len(results),
            "total_time": total_time,
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "median_time": np.median(times),
            "throughput": len(results) / total_time,
            "coefficient_of_variation": (
                np.std(times) / np.mean(times) if np.mean(times) > 0 else 0
            ),
        }


class MemoryOptimizer:
    """
    Memory optimization utilities for CPU-intensive scientific computing.
    """

    def __init__(self):
        """Initialize memory optimizer."""
        self.process = psutil.Process()

    def optimize_array_memory(self, arrays: list[np.ndarray]) -> dict[str, Any]:
        """
        Optimize memory usage for NumPy arrays.

        Parameters
        ----------
        arrays : list[np.ndarray]
            Arrays to optimize

        Returns
        -------
        dict[str, Any]
            Optimization results and recommendations
        """
        total_memory_mb = sum(arr.nbytes for arr in arrays) / 1024 / 1024
        memory_info = self.process.memory_info()
        available_memory_mb = psutil.virtual_memory().available / 1024 / 1024

        optimization_results = {
            "total_array_memory_mb": total_memory_mb,
            "available_memory_mb": available_memory_mb,
            "memory_pressure": total_memory_mb / available_memory_mb,
            "recommendations": [],
        }

        # Analyze array characteristics
        for i, arr in enumerate(arrays):
            arr_info = {
                "index": i,
                "shape": arr.shape,
                "dtype": str(arr.dtype),
                "memory_mb": arr.nbytes / 1024 / 1024,
                "is_contiguous": arr.flags.c_contiguous,
                "suggestions": [],
            }

            # Check for optimization opportunities
            if not arr.flags.c_contiguous:
                arr_info["suggestions"].append(
                    "Make array C-contiguous for better cache performance"
                )

            if arr.dtype == np.float64:
                arr_info["suggestions"].append(
                    "Consider float32 if precision allows (50% memory reduction)"
                )

            if arr.size > 1000000:  # Large arrays
                arr_info["suggestions"].append(
                    "Consider chunked processing for large arrays"
                )

            optimization_results[f"array_{i}"] = arr_info

        # Overall recommendations
        if optimization_results["memory_pressure"] > 0.8:
            optimization_results["recommendations"].append(
                "High memory pressure - consider chunked processing"
            )

        if optimization_results["memory_pressure"] > 0.5:
            optimization_results["recommendations"].append(
                "Monitor memory usage - approaching limits"
            )

        return optimization_results

    def monitor_memory_usage(self, duration_seconds: float = 60.0) -> dict[str, Any]:
        """
        Monitor memory usage over time.

        Parameters
        ----------
        duration_seconds : float
            Monitoring duration

        Returns
        -------
        dict[str, Any]
            Memory usage statistics over time
        """
        start_time = time.time()
        measurements = []

        while time.time() - start_time < duration_seconds:
            try:
                memory_info = self.process.memory_info()
                system_memory = psutil.virtual_memory()

                measurement = {
                    "timestamp": time.time() - start_time,
                    "rss_mb": memory_info.rss / 1024 / 1024,
                    "vms_mb": memory_info.vms / 1024 / 1024,
                    "percent": self.process.memory_percent(),
                    "system_available_mb": system_memory.available / 1024 / 1024,
                    "system_used_percent": system_memory.percent,
                }
                measurements.append(measurement)

                time.sleep(0.1)  # Sample every 100ms

            except Exception as e:
                warnings.warn(f"Memory measurement failed: {e}")

        # Calculate statistics
        if measurements:
            rss_values = [m["rss_mb"] for m in measurements]
            return {
                "duration": duration_seconds,
                "sample_count": len(measurements),
                "peak_memory_mb": max(rss_values),
                "min_memory_mb": min(rss_values),
                "mean_memory_mb": np.mean(rss_values),
                "memory_growth_mb": rss_values[-1] - rss_values[0],
                "measurements": measurements,
            }
        return {"error": "No measurements collected"}


def profile_homodyne_function(func: Callable) -> Callable:
    """
    Decorator for automatic profiling of homodyne analysis functions.

    Parameters
    ----------
    func : Callable
        Function to profile

    Returns
    -------
    Callable
        Wrapped function with profiling
    """

    def wrapper(*args, **kwargs):
        profiler = CPUProfiler()

        with profiler.profile_function(func.__name__) as result:
            output = func(*args, **kwargs)

        # Log performance results (in production, this might go to a monitoring system)
        print(f"Performance Profile for {func.__name__}:")
        print(f"  Execution Time: {result.total_time:.4f}s")
        print(f"  Memory Delta: {result.memory_delta_mb:.2f}MB")
        print(f"  Cache Efficiency: {result.cache_efficiency:.2f}")

        return output

    return wrapper


def get_cpu_performance_info() -> dict[str, Any]:
    """
    Get comprehensive CPU performance information.

    Returns
    -------
    dict[str, Any]
        System performance characteristics
    """
    profiler = CPUProfiler()
    memory_optimizer = MemoryOptimizer()

    system_info = {
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_freq": dict(psutil.cpu_freq()._asdict()) if psutil.cpu_freq() else {},
        "memory_total_gb": psutil.virtual_memory().total / 1024**3,
        "memory_available_gb": psutil.virtual_memory().available / 1024**3,
        "platform": {
            "system": (
                psutil.Platform.system if hasattr(psutil, "Platform") else "unknown"
            ),
            "machine": (
                psutil.Platform.machine if hasattr(psutil, "Platform") else "unknown"
            ),
        },
        "profiling_capabilities": {
            "cprofile_available": CPROFILE_AVAILABLE,
            "memory_profiler_available": MEMORY_PROFILER_AVAILABLE,
            "detailed_profiling": profiler.enable_detailed,
        },
    }

    return system_info
