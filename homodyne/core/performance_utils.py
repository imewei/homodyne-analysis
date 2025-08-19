"""
Performance Optimization Utilities
===================================

Utilities for improving performance consistency and handling JIT compilation
overhead in numerical computations.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory & University of Chicago
"""

import gc
import time
import warnings
from functools import wraps
from typing import Callable, Any, Dict, List, Optional
import numpy as np


def jit_warmup(warmup_runs: int = 3, gc_between_runs: bool = True):
    """
    Decorator to add JIT compilation warmup for stable performance benchmarking.
    
    This decorator ensures that Numba JIT compilation is completed before 
    performance measurement begins, reducing variance in benchmark results.
    
    Parameters
    ----------
    warmup_runs : int, default=3
        Number of warmup runs to perform before actual measurement
    gc_between_runs : bool, default=True
        Whether to run garbage collection between warmup runs
        
    Returns
    -------
    callable
        Decorated function with JIT warmup capability
        
    Examples
    --------
    >>> @jit_warmup(warmup_runs=5)
    ... def compute_correlation(params, angles):
    ...     return analyzer.calculate_c2_nonequilibrium_laminar_parallel(params, angles)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Store original function for warmup
            original_func = func
            
            # Perform warmup runs
            for i in range(warmup_runs):
                try:
                    _ = original_func(*args, **kwargs)
                    if gc_between_runs:
                        gc.collect()
                except Exception as e:
                    warnings.warn(f"Warmup run {i+1} failed: {e}", RuntimeWarning)
                    
            # Return the actual function call result
            return original_func(*args, **kwargs)
            
        return wrapper
    return decorator


def stable_benchmark(func: Callable, 
                    warmup_runs: int = 3,
                    measurement_runs: int = 10,
                    outlier_threshold: float = 2.0) -> Dict[str, Any]:
    """
    Perform stable benchmarking with outlier filtering and warmup.
    
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
    """
    # Warmup runs
    print(f"Performing {warmup_runs} warmup runs...")
    for i in range(warmup_runs):
        try:
            _ = func()
            gc.collect()
        except Exception as e:
            warnings.warn(f"Warmup run {i+1} failed: {e}", RuntimeWarning)
    
    # Measurement runs
    print(f"Performing {measurement_runs} measurement runs...")
    times = []
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
        'outlier_ratio': np.max(times) / np.min(times),
        'outlier_count': np.sum(outlier_mask),
        'filtered_mean': filtered_mean,
        'filtered_std': filtered_std,
        'percentile_95': np.percentile(times, 95),
        'percentile_99': np.percentile(times, 99)
    }


def memory_efficient_computation(pre_allocate: bool = False, 
                               clear_cache: bool = True):
    """
    Decorator for memory-efficient computation with optional pre-allocation.
    
    Parameters
    ----------
    pre_allocate : bool, default=False
        Whether to pre-allocate result arrays
    clear_cache : bool, default=True
        Whether to clear caches after computation
        
    Returns
    -------
    callable
        Decorated function with memory optimization
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            initial_memory = None
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            except ImportError:
                pass
                
            # Clear caches if requested
            if clear_cache:
                gc.collect()
                
            # Execute function
            result = func(*args, **kwargs)
            
            # Memory cleanup
            if clear_cache:
                gc.collect()
                
            final_memory = None
            if initial_memory is not None:
                try:
                    final_memory = process.memory_info().rss / 1024 / 1024
                    memory_delta = final_memory - initial_memory
                    if hasattr(result, '__memory_info__'):
                        result.__memory_info__ = {
                            'initial_mb': initial_memory,
                            'final_mb': final_memory, 
                            'delta_mb': memory_delta
                        }
                except:
                    pass
                    
            return result
        return wrapper
    return decorator


class PerformanceMonitor:
    """
    Context manager for monitoring performance characteristics.
    
    Provides detailed performance monitoring including timing, memory usage,
    and system resource utilization.
    
    Examples
    --------
    >>> with PerformanceMonitor("correlation_calculation") as monitor:
    ...     result = analyzer.calculate_c2_nonequilibrium_laminar_parallel(params, angles)
    >>> print(f"Execution time: {monitor.elapsed_time:.4f}s")
    """
    
    def __init__(self, operation_name: str, verbose: bool = False):
        self.operation_name = operation_name
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.memory_info = {}
        
    def __enter__(self):
        if self.verbose:
            print(f"Starting performance monitoring for: {self.operation_name}")
            
        # Record initial state
        self.start_time = time.perf_counter()
        
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            self.memory_info['initial_rss_mb'] = process.memory_info().rss / 1024 / 1024
            self.memory_info['initial_vms_mb'] = process.memory_info().vms / 1024 / 1024
        except ImportError:
            if self.verbose:
                print("psutil not available - memory monitoring disabled")
                
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        
        # Record final state
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            final_rss = process.memory_info().rss / 1024 / 1024
            final_vms = process.memory_info().vms / 1024 / 1024
            
            self.memory_info['final_rss_mb'] = final_rss
            self.memory_info['final_vms_mb'] = final_vms
            
            if 'initial_rss_mb' in self.memory_info:
                self.memory_info['rss_delta_mb'] = final_rss - self.memory_info['initial_rss_mb']
                self.memory_info['vms_delta_mb'] = final_vms - self.memory_info['initial_vms_mb']
                
        except ImportError:
            pass
            
        if self.verbose:
            print(f"Performance monitoring complete for: {self.operation_name}")
            print(f"  Execution time: {self.elapsed_time:.4f}s")
            if self.memory_info:
                if 'rss_delta_mb' in self.memory_info:
                    print(f"  Memory delta (RSS): {self.memory_info['rss_delta_mb']:.1f} MB")
                    print(f"  Memory delta (VMS): {self.memory_info['vms_delta_mb']:.1f} MB")


def optimize_numerical_environment():
    """
    Optimize the numerical computation environment for consistent performance.
    
    This function sets environment variables and configurations that help
    reduce performance variance in numerical computations.
    
    Returns
    -------
    dict
        Dictionary of applied optimizations
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
            
    # NumPy optimizations
    try:
        import numpy as np
        
        # Use consistent random seed for reproducible benchmarks
        np.random.seed(42)
        optimizations['numpy_random_seed'] = 42
        
        # Disable numpy warnings for cleaner benchmark output
        np.seterr(all='ignore')
        optimizations['numpy_warnings'] = 'disabled'
        
    except ImportError:
        pass
        
    # Garbage collection optimization
    gc.set_threshold(700, 10, 10)  # More frequent GC for consistent memory
    optimizations['gc_threshold'] = (700, 10, 10)
    
    return optimizations


# Performance assertion helpers
def assert_performance_within_bounds(measured_time: float,
                                    expected_time: float, 
                                    tolerance_factor: float = 2.0,
                                    test_name: str = "performance_test"):
    """
    Assert that measured performance is within acceptable bounds.
    
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
        
    Raises
    ------
    AssertionError
        If measured time exceeds tolerance bounds
    """
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


def assert_performance_stability(times: List[float],
                               max_cv: float = 0.5,  # 50% coefficient of variation
                               test_name: str = "stability_test"):
    """
    Assert that performance measurements are stable (low variance).
    
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
    """
    times_array = np.array(times)
    mean_time = np.mean(times_array)
    std_time = np.std(times_array)
    cv = std_time / mean_time if mean_time > 0 else float('inf')
    
    assert cv <= max_cv, (
        f"{test_name} performance too variable: "
        f"coefficient of variation {cv:.3f} > max allowed {max_cv:.3f} "
        f"(std={std_time:.4f}s, mean={mean_time:.4f}s)"
    )
