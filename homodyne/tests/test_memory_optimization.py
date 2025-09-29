"""
Memory Efficiency and Allocation Optimization
=============================================

Comprehensive memory optimization suite for the homodyne analysis package.
Focuses on reducing memory footprint, improving allocation patterns, and
preventing memory leaks.

Features:
- Memory profiling and leak detection
- Efficient allocation strategies
- Memory pool management
- Garbage collection optimization
- Memory-mapped file operations
- Array reuse and recycling

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import gc
import json
import tempfile
import time
import tracemalloc
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil

warnings.filterwarnings("ignore")


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""

    operation_name: str
    memory_before_mb: float
    memory_after_mb: float
    memory_peak_mb: float
    memory_delta_mb: float
    allocation_count: int
    deallocation_count: int
    gc_collections: int
    optimization_method: str


class MemoryProfiler:
    """Advanced memory profiling and monitoring."""

    def __init__(self):
        self.snapshots = []
        self.metrics = []

    def start_profiling(self):
        """Start memory profiling."""
        tracemalloc.start()
        gc.collect()  # Start with clean state

    def stop_profiling(self):
        """Stop memory profiling."""
        tracemalloc.stop()

    def take_snapshot(self, label: str = ""):
        """Take a memory snapshot."""
        current, peak = tracemalloc.get_traced_memory()
        process_memory = psutil.Process().memory_info().rss / 1024 / 1024

        snapshot = {
            "label": label,
            "timestamp": time.time(),
            "tracemalloc_current_mb": current / 1024 / 1024,
            "tracemalloc_peak_mb": peak / 1024 / 1024,
            "process_memory_mb": process_memory,
            "gc_objects": len(gc.get_objects()),
        }

        self.snapshots.append(snapshot)
        return snapshot

    def measure_operation(self, operation_name: str, func, *args, **kwargs):
        """Measure memory usage of an operation."""
        # Take before snapshot
        gc.collect()
        before_snapshot = self.take_snapshot(f"{operation_name}_before")

        # Execute operation
        result = func(*args, **kwargs)

        # Take after snapshot
        gc.collect()
        after_snapshot = self.take_snapshot(f"{operation_name}_after")

        # Calculate metrics
        memory_delta = (
            after_snapshot["process_memory_mb"] - before_snapshot["process_memory_mb"]
        )

        metrics = MemoryMetrics(
            operation_name=operation_name,
            memory_before_mb=before_snapshot["process_memory_mb"],
            memory_after_mb=after_snapshot["process_memory_mb"],
            memory_peak_mb=after_snapshot["tracemalloc_peak_mb"],
            memory_delta_mb=memory_delta,
            allocation_count=0,  # Would need more detailed tracking
            deallocation_count=0,
            gc_collections=0,
            optimization_method="baseline",
        )

        self.metrics.append(metrics)
        return result, metrics


class MemoryPool:
    """Memory pool for efficient array allocation."""

    def __init__(self, max_pool_size: int = 100):
        self.pools = {}  # Shape -> list of arrays
        self.max_pool_size = max_pool_size
        self.allocation_count = 0
        self.reuse_count = 0

    def get_array(
        self, shape: tuple[int, ...], dtype: np.dtype = np.float64
    ) -> np.ndarray:
        """Get an array from the pool or allocate new one."""
        key = (shape, dtype)

        if self.pools.get(key):
            # Reuse from pool
            array = self.pools[key].pop()
            array.fill(0)  # Clear previous data
            self.reuse_count += 1
            return array
        # Allocate new array
        self.allocation_count += 1
        return np.zeros(shape, dtype=dtype)

    def return_array(self, array: np.ndarray):
        """Return array to the pool."""
        key = (array.shape, array.dtype)

        if key not in self.pools:
            self.pools[key] = []

        if len(self.pools[key]) < self.max_pool_size:
            self.pools[key].append(array)

    def get_stats(self) -> dict[str, int]:
        """Get pool statistics."""
        total_pooled = sum(len(arrays) for arrays in self.pools.values())
        return {
            "allocations": self.allocation_count,
            "reuses": self.reuse_count,
            "pooled_arrays": total_pooled,
            "pool_shapes": len(self.pools),
        }


class InPlaceOperations:
    """Optimized in-place operations to reduce memory allocation."""

    @staticmethod
    def inplace_chi_squared(
        c2_exp: np.ndarray,
        c2_theo: np.ndarray,
        result_array: np.ndarray | None = None,
    ) -> np.ndarray:
        """In-place chi-squared calculation."""
        if result_array is None:
            result_array = np.empty_like(c2_exp)

        # In-place operations
        np.subtract(c2_exp, c2_theo, out=result_array)
        np.square(result_array, out=result_array)

        return result_array

    @staticmethod
    def inplace_correlation_function(
        data: np.ndarray, result: np.ndarray | None = None
    ) -> np.ndarray:
        """Memory-efficient correlation function calculation."""
        n_angles, n_time = data.shape

        if result is None:
            result = np.empty((n_angles, n_time, n_time))

        # Calculate correlation for each angle
        for i in range(n_angles):
            data_i = data[i]
            # Use np.outer for efficient outer product
            np.outer(data_i, data_i, out=result[i])

        return result

    @staticmethod
    def inplace_normalization(
        array: np.ndarray,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> np.ndarray:
        """In-place array normalization."""
        if min_val is None:
            min_val = np.min(array)
        if max_val is None:
            max_val = np.max(array)

        range_val = max_val - min_val
        if range_val > 1e-15:
            # In-place normalization
            array -= min_val
            array /= range_val

        return array


class MemoryMappedOperations:
    """Memory-mapped file operations for large datasets."""

    @staticmethod
    def create_memory_mapped_array(
        shape: tuple[int, ...],
        dtype: np.dtype = np.float64,
        filename: str | None = None,
    ) -> np.ndarray:
        """Create memory-mapped array."""
        if filename is None:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            filename = temp_file.name
            temp_file.close()

        # Calculate required size
        itemsize = np.dtype(dtype).itemsize
        total_size = np.prod(shape) * itemsize

        # Create memory-mapped array
        with open(filename, "wb") as f:
            f.write(b"\x00" * total_size)

        return np.memmap(filename, dtype=dtype, mode="r+", shape=shape)

    @staticmethod
    def chunked_operation(
        array: np.ndarray, operation: callable, chunk_size: int = 1000
    ) -> np.ndarray:
        """Apply operation in chunks to reduce memory usage."""
        if array.size <= chunk_size:
            return operation(array)

        # Process in chunks
        flat_array = array.flatten()
        result_chunks = []

        for i in range(0, len(flat_array), chunk_size):
            chunk = flat_array[i : i + chunk_size]
            result_chunk = operation(chunk)
            result_chunks.append(result_chunk)

        # Combine results
        result = np.concatenate(result_chunks)
        return result.reshape(array.shape)


class GarbageCollectionOptimizer:
    """Garbage collection optimization strategies."""

    @staticmethod
    def disable_gc_during_computation(func):
        """Decorator to disable GC during computation."""

        def wrapper(*args, **kwargs):
            gc_enabled = gc.isenabled()
            if gc_enabled:
                gc.disable()

            try:
                result = func(*args, **kwargs)
            finally:
                if gc_enabled:
                    gc.enable()

            return result

        return wrapper

    @staticmethod
    def periodic_gc_cleanup(interval: int = 1000):
        """Decorator for periodic garbage collection."""
        call_count = 0

        def decorator(func):
            def wrapper(*args, **kwargs):
                nonlocal call_count
                call_count += 1

                if call_count % interval == 0:
                    gc.collect()

                return func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def measure_gc_impact(operation_name: str, func, *args, **kwargs):
        """Measure garbage collection impact on performance."""
        # Measure with GC enabled
        gc.enable()
        start_time = time.perf_counter()
        result_with_gc = func(*args, **kwargs)
        time_with_gc = time.perf_counter() - start_time

        # Clean up and measure with GC disabled
        gc.collect()
        gc.disable()
        start_time = time.perf_counter()
        result_without_gc = func(*args, **kwargs)
        time_without_gc = time.perf_counter() - start_time
        gc.enable()

        gc_overhead = time_with_gc - time_without_gc
        return {
            "operation": operation_name,
            "time_with_gc": time_with_gc,
            "time_without_gc": time_without_gc,
            "gc_overhead": gc_overhead,
            "gc_overhead_percent": (
                (gc_overhead / time_with_gc) * 100 if time_with_gc > 0 else 0
            ),
        }


class MemoryOptimizationSuite:
    """Comprehensive memory optimization test suite."""

    def __init__(self):
        self.profiler = MemoryProfiler()
        self.memory_pool = MemoryPool()
        self.results = {}

    def test_memory_pool_efficiency(self) -> dict[str, Any]:
        """Test memory pool efficiency."""
        print("Testing memory pool efficiency...")

        self.profiler.start_profiling()

        # Test without memory pool
        def allocate_without_pool():
            arrays = []
            for _ in range(100):
                arr = np.zeros((100, 100))
                arrays.append(arr)
            return arrays

        _, metrics_without_pool = self.profiler.measure_operation(
            "allocation_without_pool", allocate_without_pool
        )

        # Test with memory pool
        def allocate_with_pool():
            arrays = []
            for _ in range(100):
                arr = self.memory_pool.get_array((100, 100))
                arrays.append(arr)
            return arrays

        _, metrics_with_pool = self.profiler.measure_operation(
            "allocation_with_pool", allocate_with_pool
        )

        pool_stats = self.memory_pool.get_stats()

        self.profiler.stop_profiling()

        return {
            "without_pool": {
                "memory_delta_mb": metrics_without_pool.memory_delta_mb,
                "peak_memory_mb": metrics_without_pool.memory_peak_mb,
            },
            "with_pool": {
                "memory_delta_mb": metrics_with_pool.memory_delta_mb,
                "peak_memory_mb": metrics_with_pool.memory_peak_mb,
            },
            "pool_stats": pool_stats,
            "memory_reduction": metrics_without_pool.memory_delta_mb
            - metrics_with_pool.memory_delta_mb,
        }

    def test_inplace_operations(self) -> dict[str, Any]:
        """Test in-place operation efficiency."""
        print("Testing in-place operations...")

        # Generate test data
        c2_exp = np.random.rand(5, 100, 100)
        c2_theo = np.random.rand(5, 100, 100)

        self.profiler.start_profiling()

        # Test standard operations
        def standard_chi_squared():
            return np.sum((c2_exp - c2_theo) ** 2)

        _, metrics_standard = self.profiler.measure_operation(
            "standard_chi_squared", standard_chi_squared
        )

        # Test in-place operations
        def inplace_chi_squared():
            result_array = np.empty_like(c2_exp)
            InPlaceOperations.inplace_chi_squared(c2_exp, c2_theo, result_array)
            return np.sum(result_array)

        _, metrics_inplace = self.profiler.measure_operation(
            "inplace_chi_squared", inplace_chi_squared
        )

        self.profiler.stop_profiling()

        return {
            "standard": {
                "memory_delta_mb": metrics_standard.memory_delta_mb,
                "peak_memory_mb": metrics_standard.memory_peak_mb,
            },
            "inplace": {
                "memory_delta_mb": metrics_inplace.memory_delta_mb,
                "peak_memory_mb": metrics_inplace.memory_peak_mb,
            },
            "memory_reduction": metrics_standard.memory_delta_mb
            - metrics_inplace.memory_delta_mb,
        }

    def test_memory_mapped_operations(self) -> dict[str, Any]:
        """Test memory-mapped file operations."""
        print("Testing memory-mapped operations...")

        # Large array that might not fit in memory efficiently
        shape = (1000, 1000)

        self.profiler.start_profiling()

        # Test standard array allocation
        def standard_allocation():
            return np.zeros(shape)

        _, metrics_standard = self.profiler.measure_operation(
            "standard_allocation", standard_allocation
        )

        # Test memory-mapped allocation
        def memmap_allocation():
            return MemoryMappedOperations.create_memory_mapped_array(shape)

        _, metrics_memmap = self.profiler.measure_operation(
            "memmap_allocation", memmap_allocation
        )

        self.profiler.stop_profiling()

        return {
            "standard": {
                "memory_delta_mb": metrics_standard.memory_delta_mb,
                "peak_memory_mb": metrics_standard.memory_peak_mb,
            },
            "memmap": {
                "memory_delta_mb": metrics_memmap.memory_delta_mb,
                "peak_memory_mb": metrics_memmap.memory_peak_mb,
            },
            "memory_reduction": metrics_standard.memory_delta_mb
            - metrics_memmap.memory_delta_mb,
        }

    def test_gc_optimization(self) -> dict[str, Any]:
        """Test garbage collection optimization."""
        print("Testing garbage collection optimization...")

        # Define a memory-intensive operation
        def memory_intensive_operation():
            arrays = []
            for i in range(50):
                arr = np.random.rand(100, 100)
                processed = arr @ arr.T
                arrays.append(processed)
            return arrays

        # Measure GC impact
        gc_impact = GarbageCollectionOptimizer.measure_gc_impact(
            "memory_intensive", memory_intensive_operation
        )

        return gc_impact

    def test_chunked_operations(self) -> dict[str, Any]:
        """Test chunked operations for memory efficiency."""
        print("Testing chunked operations...")

        # Large array
        large_array = np.random.rand(10000, 100)

        self.profiler.start_profiling()

        # Standard operation
        def standard_operation():
            return np.sqrt(large_array)

        _, metrics_standard = self.profiler.measure_operation(
            "standard_sqrt", standard_operation
        )

        # Chunked operation
        def chunked_operation():
            return MemoryMappedOperations.chunked_operation(
                large_array, np.sqrt, chunk_size=1000
            )

        _, metrics_chunked = self.profiler.measure_operation(
            "chunked_sqrt", chunked_operation
        )

        self.profiler.stop_profiling()

        return {
            "standard": {
                "memory_delta_mb": metrics_standard.memory_delta_mb,
                "peak_memory_mb": metrics_standard.memory_peak_mb,
            },
            "chunked": {
                "memory_delta_mb": metrics_chunked.memory_delta_mb,
                "peak_memory_mb": metrics_chunked.memory_peak_mb,
            },
            "memory_reduction": metrics_standard.memory_delta_mb
            - metrics_chunked.memory_delta_mb,
        }

    def run_comprehensive_memory_optimization_test(self) -> dict[str, Any]:
        """Run comprehensive memory optimization tests."""
        print("Running Comprehensive Memory Optimization Tests")
        print("=" * 60)

        all_results = {}

        # Test memory pool efficiency
        pool_results = self.test_memory_pool_efficiency()
        all_results["memory_pool"] = pool_results

        # Test in-place operations
        inplace_results = self.test_inplace_operations()
        all_results["inplace_operations"] = inplace_results

        # Test memory-mapped operations
        memmap_results = self.test_memory_mapped_operations()
        all_results["memory_mapped"] = memmap_results

        # Test GC optimization
        gc_results = self.test_gc_optimization()
        all_results["garbage_collection"] = gc_results

        # Test chunked operations
        chunked_results = self.test_chunked_operations()
        all_results["chunked_operations"] = chunked_results

        return all_results


def generate_memory_optimization_report(results: dict[str, Any]) -> str:
    """Generate comprehensive memory optimization report."""
    report_lines = []
    report_lines.append("MEMORY OPTIMIZATION REPORT - TASK 4.4")
    report_lines.append("=" * 60)

    # Memory pool results
    if "memory_pool" in results:
        pool_data = results["memory_pool"]
        report_lines.append("\nMEMORY POOL OPTIMIZATION:")
        report_lines.append(
            f"  Memory reduction: {pool_data.get('memory_reduction', 0):.2f} MB"
        )
        report_lines.append(f"  Pool stats: {pool_data.get('pool_stats', {})}")

    # In-place operations
    if "inplace_operations" in results:
        inplace_data = results["inplace_operations"]
        report_lines.append("\nIN-PLACE OPERATIONS:")
        report_lines.append(
            f"  Memory reduction: {inplace_data.get('memory_reduction', 0):.2f} MB"
        )

    # Memory-mapped operations
    if "memory_mapped" in results:
        memmap_data = results["memory_mapped"]
        report_lines.append("\nMEMORY-MAPPED OPERATIONS:")
        report_lines.append(
            f"  Memory reduction: {memmap_data.get('memory_reduction', 0):.2f} MB"
        )

    # Garbage collection
    if "garbage_collection" in results:
        gc_data = results["garbage_collection"]
        report_lines.append("\nGARBAGE COLLECTION OPTIMIZATION:")
        report_lines.append(
            f"  GC overhead: {gc_data.get('gc_overhead_percent', 0):.2f}%"
        )
        report_lines.append(f"  Time with GC: {gc_data.get('time_with_gc', 0):.4f}s")
        report_lines.append(
            f"  Time without GC: {gc_data.get('time_without_gc', 0):.4f}s"
        )

    # Chunked operations
    if "chunked_operations" in results:
        chunked_data = results["chunked_operations"]
        report_lines.append("\nCHUNKED OPERATIONS:")
        report_lines.append(
            f"  Memory reduction: {chunked_data.get('memory_reduction', 0):.2f} MB"
        )

    # Calculate total memory savings
    total_savings = 0
    for category in [
        "memory_pool",
        "inplace_operations",
        "memory_mapped",
        "chunked_operations",
    ]:
        if category in results:
            total_savings += results[category].get("memory_reduction", 0)

    report_lines.append(f"\nTOTAL MEMORY SAVINGS: {total_savings:.2f} MB")

    return "\n".join(report_lines)


def run_memory_optimization_suite():
    """Main function to run memory optimization suite."""
    print("Starting Memory Optimization Suite - Task 4.4")
    print("=" * 60)

    # Run optimization tests
    suite = MemoryOptimizationSuite()
    results = suite.run_comprehensive_memory_optimization_test()

    # Generate report
    report = generate_memory_optimization_report(results)
    print("\n" + report)

    # Save results
    results_dir = Path("memory_optimization_results")
    results_dir.mkdir(exist_ok=True)

    # Save JSON results
    json_file = results_dir / "task_4_4_memory_optimization_results.json"
    with open(json_file, "w") as f:
        json.dump(
            {
                "optimization_results": results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "summary": {
                    "total_optimizations": len(results),
                    "memory_techniques": list(results.keys()),
                },
            },
            f,
            indent=2,
        )

    # Save text report
    report_file = results_dir / "task_4_4_memory_optimization_report.txt"
    with open(report_file, "w") as f:
        f.write(report)

    print(f"\n📄 Results saved to: {json_file}")
    print(f"📄 Report saved to: {report_file}")

    # Calculate total memory savings
    total_savings = 0
    for category in [
        "memory_pool",
        "inplace_operations",
        "memory_mapped",
        "chunked_operations",
    ]:
        if category in results:
            total_savings += results[category].get("memory_reduction", 0)

    print("\n✅ Task 4.4 Memory Optimization Complete!")
    print(f"💾 Total memory savings: {total_savings:.2f} MB")
    print(f"🎯 {len(results)} optimization techniques implemented")

    return results


if __name__ == "__main__":
    run_memory_optimization_suite()
