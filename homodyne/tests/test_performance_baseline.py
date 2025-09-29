"""
Performance Baseline Measurement System
=======================================

Comprehensive performance baseline measurement and tracking system for the
homodyne analysis package. This system establishes benchmarks for all major
components and tracks performance evolution over time.

Features:
- Core algorithm performance measurement
- Memory usage profiling
- Startup time analysis
- I/O operation benchmarking
- Scalability analysis across data sizes
- Automated performance regression detection

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import gc
import json
import os
import tempfile
import time
import tracemalloc
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import psutil

# Suppress warnings for cleaner benchmark output
warnings.filterwarnings("ignore")


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""
    operation_name: str
    execution_time: float  # seconds
    memory_peak: float     # MB
    memory_current: float  # MB
    cpu_percent: float     # percentage
    data_size: int         # input size
    iterations: int        # number of iterations
    timestamp: str         # ISO timestamp
    system_info: Dict[str, Any]  # system information


@dataclass
class PerformanceBaseline:
    """Performance baseline data structure."""
    baseline_id: str
    creation_date: str
    system_info: Dict[str, Any]
    metrics: Dict[str, PerformanceMetrics]
    summary_statistics: Dict[str, float]


class SystemProfiler:
    """System information and resource monitoring."""

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information."""
        try:
            cpu_info = {
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            }

            memory_info = psutil.virtual_memory()._asdict()

            return {
                "cpu": cpu_info,
                "memory": memory_info,
                "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}",
                "platform": __import__('platform').platform(),
                "architecture": __import__('platform').architecture(),
            }
        except Exception:
            return {"error": "Could not gather system information"}

    @staticmethod
    def get_memory_usage() -> Tuple[float, float]:
        """Get current and peak memory usage in MB."""
        process = psutil.Process()
        memory_info = process.memory_info()
        current_mb = memory_info.rss / 1024 / 1024

        # Try to get peak memory if available
        try:
            peak_mb = process.memory_info().peak_wss / 1024 / 1024
        except AttributeError:
            peak_mb = current_mb

        return current_mb, peak_mb

    @staticmethod
    def get_cpu_usage() -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)


class PerformanceBenchmarker:
    """High-precision performance benchmarking system."""

    def __init__(self):
        self.profiler = SystemProfiler()
        self.baseline_dir = Path("performance_baselines")
        self.baseline_dir.mkdir(exist_ok=True)

    @contextmanager
    def measure_performance(self, operation_name: str, data_size: int = 0, iterations: int = 1):
        """Context manager for measuring operation performance."""
        # Start memory tracking
        tracemalloc.start()
        gc.collect()  # Clean memory before measurement

        # Record initial state
        start_memory_current, _ = self.profiler.get_memory_usage()
        start_time = time.perf_counter()

        try:
            yield
        finally:
            # Record final state
            end_time = time.perf_counter()
            end_memory_current, memory_peak = self.profiler.get_memory_usage()
            cpu_percent = self.profiler.get_cpu_usage()

            # Stop memory tracking
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Create metrics
            execution_time = end_time - start_time
            peak_memory_mb = max(memory_peak, peak_mem / 1024 / 1024)
            current_memory_mb = end_memory_current

            metrics = PerformanceMetrics(
                operation_name=operation_name,
                execution_time=execution_time,
                memory_peak=peak_memory_mb,
                memory_current=current_memory_mb,
                cpu_percent=cpu_percent,
                data_size=data_size,
                iterations=iterations,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                system_info=self.profiler.get_system_info()
            )

            # Store metrics for later retrieval
            self._last_metrics = metrics

    def get_last_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the last measured performance metrics."""
        return getattr(self, '_last_metrics', None)

    def benchmark_function(self, func: Callable, args: Tuple = (), kwargs: Dict = None,
                          iterations: int = 10, warmup: int = 3) -> PerformanceMetrics:
        """Benchmark a function with multiple iterations."""
        if kwargs is None:
            kwargs = {}

        # Warmup runs
        for _ in range(warmup):
            try:
                func(*args, **kwargs)
            except Exception:
                pass  # Ignore warmup errors

        # Actual benchmarking
        times = []
        memories = []

        for i in range(iterations):
            with self.measure_performance(func.__name__, data_size=len(args), iterations=1):
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    print(f"Benchmark iteration {i+1} failed: {e}")
                    continue

            metrics = self.get_last_metrics()
            if metrics:
                times.append(metrics.execution_time)
                memories.append(metrics.memory_peak)

        if not times:
            # Fallback metrics if all iterations failed
            return PerformanceMetrics(
                operation_name=func.__name__,
                execution_time=float('inf'),
                memory_peak=0.0,
                memory_current=0.0,
                cpu_percent=0.0,
                data_size=len(args),
                iterations=iterations,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                system_info=self.profiler.get_system_info()
            )

        # Calculate statistics
        avg_time = np.mean(times)
        avg_memory = np.mean(memories) if memories else 0.0
        current_memory, _ = self.profiler.get_memory_usage()
        cpu_percent = self.profiler.get_cpu_usage()

        return PerformanceMetrics(
            operation_name=func.__name__,
            execution_time=avg_time,
            memory_peak=avg_memory,
            memory_current=current_memory,
            cpu_percent=cpu_percent,
            data_size=len(args),
            iterations=iterations,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            system_info=self.profiler.get_system_info()
        )


class CoreAlgorithmBenchmarks:
    """Benchmarks for core homodyne analysis algorithms."""

    def __init__(self):
        self.benchmarker = PerformanceBenchmarker()

    def benchmark_chi_squared_calculation(self) -> Dict[str, PerformanceMetrics]:
        """Benchmark chi-squared calculation with different data sizes."""
        results = {}

        # Test different data sizes
        data_sizes = [(3, 10, 10), (5, 20, 20), (8, 50, 50), (10, 100, 100)]

        for n_angles, n_time1, n_time2 in data_sizes:
            # Generate test data
            c2_exp = np.random.rand(n_angles, n_time1, n_time2)
            c2_theo = np.random.rand(n_angles, n_time1, n_time2)

            def chi_squared_calc():
                return np.sum((c2_exp - c2_theo)**2)

            size_key = f"chi_squared_{n_angles}x{n_time1}x{n_time2}"
            results[size_key] = self.benchmarker.benchmark_function(
                chi_squared_calc, iterations=100
            )

        return results

    def benchmark_matrix_operations(self) -> Dict[str, PerformanceMetrics]:
        """Benchmark core matrix operations."""
        results = {}

        # Test different matrix sizes
        matrix_sizes = [10, 50, 100, 500]

        for size in matrix_sizes:
            # Matrix multiplication benchmark
            A = np.random.rand(size, size)
            B = np.random.rand(size, size)

            def matrix_multiply():
                return A @ B

            results[f"matrix_multiply_{size}x{size}"] = self.benchmarker.benchmark_function(
                matrix_multiply, iterations=10
            )

            # Eigenvalue decomposition benchmark
            def eigenvalue_decomp():
                return np.linalg.eig(A)

            results[f"eigenvalue_decomp_{size}x{size}"] = self.benchmarker.benchmark_function(
                eigenvalue_decomp, iterations=5
            )

        return results

    def benchmark_array_operations(self) -> Dict[str, PerformanceMetrics]:
        """Benchmark NumPy array operations."""
        results = {}

        # Test different array sizes
        array_sizes = [1000, 10000, 100000, 1000000]

        for size in array_sizes:
            data = np.random.rand(size)

            # Statistical operations
            def array_mean():
                return np.mean(data)

            def array_std():
                return np.std(data)

            def array_sum():
                return np.sum(data)

            results[f"array_mean_{size}"] = self.benchmarker.benchmark_function(
                array_mean, iterations=100
            )
            results[f"array_std_{size}"] = self.benchmarker.benchmark_function(
                array_std, iterations=100
            )
            results[f"array_sum_{size}"] = self.benchmarker.benchmark_function(
                array_sum, iterations=100
            )

        return results


class IOBenchmarks:
    """Benchmarks for I/O operations."""

    def __init__(self):
        self.benchmarker = PerformanceBenchmarker()

    def benchmark_file_operations(self) -> Dict[str, PerformanceMetrics]:
        """Benchmark file I/O operations."""
        results = {}

        # Test different file sizes
        data_sizes = [1000, 10000, 100000]

        for size in data_sizes:
            # Generate test data
            test_data = np.random.rand(size)

            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_path = tmp_file.name

            try:
                # NumPy save benchmark
                def numpy_save():
                    np.save(tmp_path, test_data)

                results[f"numpy_save_{size}"] = self.benchmarker.benchmark_function(
                    numpy_save, iterations=10
                )

                # NumPy load benchmark
                np.save(tmp_path, test_data)  # Ensure file exists

                def numpy_load():
                    return np.load(tmp_path + '.npy')

                results[f"numpy_load_{size}"] = self.benchmarker.benchmark_function(
                    numpy_load, iterations=10
                )

            finally:
                # Clean up
                for ext in ['', '.npy']:
                    try:
                        os.unlink(tmp_path + ext)
                    except FileNotFoundError:
                        pass

        return results

    def benchmark_json_operations(self) -> Dict[str, PerformanceMetrics]:
        """Benchmark JSON I/O operations."""
        results = {}

        # Test different JSON data sizes
        for size in [100, 1000, 10000]:
            test_data = {f"key_{i}": f"value_{i}" for i in range(size)}

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
                tmp_path = tmp_file.name

            try:
                # JSON write benchmark
                def json_write():
                    with open(tmp_path, 'w') as f:
                        json.dump(test_data, f)

                results[f"json_write_{size}"] = self.benchmarker.benchmark_function(
                    json_write, iterations=10
                )

                # JSON read benchmark
                with open(tmp_path, 'w') as f:
                    json.dump(test_data, f)

                def json_read():
                    with open(tmp_path, 'r') as f:
                        return json.load(f)

                results[f"json_read_{size}"] = self.benchmarker.benchmark_function(
                    json_read, iterations=10
                )

            finally:
                os.unlink(tmp_path)

        return results


class StartupBenchmarks:
    """Benchmarks for package startup and import performance."""

    def __init__(self):
        self.benchmarker = PerformanceBenchmarker()

    def benchmark_import_times(self) -> Dict[str, PerformanceMetrics]:
        """Benchmark import times for major modules."""
        results = {}

        # Test core module imports
        import_tests = [
            ("numpy", "import numpy"),
            ("scipy", "import scipy"),
            ("json", "import json"),
            ("pathlib", "from pathlib import Path"),
        ]

        # Test homodyne module imports if available
        try:
            homodyne_imports = [
                ("homodyne_core", "from homodyne.core import composition"),
                ("homodyne_analysis", "from homodyne.analysis import core"),
                ("homodyne_optimization", "from homodyne.optimization import classical"),
            ]
            import_tests.extend(homodyne_imports)
        except ImportError:
            pass

        for module_name, import_statement in import_tests:
            def import_module():
                exec(import_statement)

            results[f"import_{module_name}"] = self.benchmarker.benchmark_function(
                import_module, iterations=5
            )

        return results


class PerformanceBaselineManager:
    """Manages performance baselines and comparisons."""

    def __init__(self, baseline_dir: str = "performance_baselines"):
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(exist_ok=True)

    def create_comprehensive_baseline(self, baseline_id: str) -> PerformanceBaseline:
        """Create a comprehensive performance baseline."""
        print(f"Creating comprehensive performance baseline: {baseline_id}")

        # Initialize benchmarks
        core_benchmarks = CoreAlgorithmBenchmarks()
        io_benchmarks = IOBenchmarks()
        startup_benchmarks = StartupBenchmarks()

        all_metrics = {}

        # Run core algorithm benchmarks
        print("Running core algorithm benchmarks...")
        core_results = core_benchmarks.benchmark_chi_squared_calculation()
        all_metrics.update(core_results)

        matrix_results = core_benchmarks.benchmark_matrix_operations()
        all_metrics.update(matrix_results)

        array_results = core_benchmarks.benchmark_array_operations()
        all_metrics.update(array_results)

        # Run I/O benchmarks
        print("Running I/O benchmarks...")
        file_results = io_benchmarks.benchmark_file_operations()
        all_metrics.update(file_results)

        json_results = io_benchmarks.benchmark_json_operations()
        all_metrics.update(json_results)

        # Run startup benchmarks
        print("Running startup benchmarks...")
        import_results = startup_benchmarks.benchmark_import_times()
        all_metrics.update(import_results)

        # Calculate summary statistics
        execution_times = [m.execution_time for m in all_metrics.values() if m.execution_time != float('inf')]
        memory_peaks = [m.memory_peak for m in all_metrics.values() if m.memory_peak > 0]

        summary_stats = {
            "total_operations": len(all_metrics),
            "avg_execution_time": np.mean(execution_times) if execution_times else 0.0,
            "median_execution_time": np.median(execution_times) if execution_times else 0.0,
            "max_execution_time": np.max(execution_times) if execution_times else 0.0,
            "avg_memory_peak": np.mean(memory_peaks) if memory_peaks else 0.0,
            "max_memory_peak": np.max(memory_peaks) if memory_peaks else 0.0,
        }

        # Create baseline
        baseline = PerformanceBaseline(
            baseline_id=baseline_id,
            creation_date=time.strftime("%Y-%m-%d %H:%M:%S"),
            system_info=SystemProfiler.get_system_info(),
            metrics=all_metrics,
            summary_statistics=summary_stats
        )

        # Save baseline
        self.save_baseline(baseline)

        return baseline

    def save_baseline(self, baseline: PerformanceBaseline):
        """Save performance baseline to disk."""
        filename = f"performance_baseline_{baseline.baseline_id}.json"
        filepath = self.baseline_dir / filename

        # Convert to JSON-serializable format
        baseline_dict = asdict(baseline)

        # Convert PerformanceMetrics objects to dicts
        metrics_dict = {}
        for key, metrics in baseline.metrics.items():
            metrics_dict[key] = asdict(metrics)
        baseline_dict["metrics"] = metrics_dict

        with open(filepath, 'w') as f:
            json.dump(baseline_dict, f, indent=2)

        print(f"Baseline saved to: {filepath}")

    def load_baseline(self, baseline_id: str) -> Optional[PerformanceBaseline]:
        """Load performance baseline from disk."""
        filename = f"performance_baseline_{baseline_id}.json"
        filepath = self.baseline_dir / filename

        if not filepath.exists():
            return None

        with open(filepath, 'r') as f:
            baseline_dict = json.load(f)

        # Convert metrics back to PerformanceMetrics objects
        metrics = {}
        for key, metrics_dict in baseline_dict["metrics"].items():
            metrics[key] = PerformanceMetrics(**metrics_dict)

        baseline_dict["metrics"] = metrics

        return PerformanceBaseline(**baseline_dict)

    def generate_baseline_report(self, baseline: PerformanceBaseline) -> str:
        """Generate a comprehensive baseline report."""
        report = []
        report.append("=" * 80)
        report.append(f"PERFORMANCE BASELINE REPORT: {baseline.baseline_id}")
        report.append("=" * 80)

        report.append(f"\nCreation Date: {baseline.creation_date}")
        report.append(f"System Info: {baseline.system_info.get('platform', 'Unknown')}")
        report.append(f"CPU Cores: {baseline.system_info.get('cpu', {}).get('cpu_count', 'Unknown')}")
        report.append(f"Memory: {baseline.system_info.get('memory', {}).get('total', 0) / (1024**3):.1f} GB")

        report.append(f"\nSUMMARY STATISTICS:")
        report.append(f"  Total Operations: {baseline.summary_statistics['total_operations']}")
        report.append(f"  Average Execution Time: {baseline.summary_statistics['avg_execution_time']:.4f} seconds")
        report.append(f"  Median Execution Time: {baseline.summary_statistics['median_execution_time']:.4f} seconds")
        report.append(f"  Maximum Execution Time: {baseline.summary_statistics['max_execution_time']:.4f} seconds")
        report.append(f"  Average Memory Peak: {baseline.summary_statistics['avg_memory_peak']:.2f} MB")
        report.append(f"  Maximum Memory Peak: {baseline.summary_statistics['max_memory_peak']:.2f} MB")

        report.append(f"\nDETAILED METRICS:")
        report.append(f"{'Operation':<30} {'Time (s)':<12} {'Memory (MB)':<12} {'Data Size':<12}")
        report.append("-" * 66)

        # Sort by execution time (descending)
        sorted_metrics = sorted(baseline.metrics.items(),
                              key=lambda x: x[1].execution_time, reverse=True)

        for op_name, metrics in sorted_metrics:
            time_str = f"{metrics.execution_time:.4f}" if metrics.execution_time != float('inf') else "FAILED"
            report.append(f"{op_name:<30} {time_str:<12} {metrics.memory_peak:<12.2f} {metrics.data_size:<12}")

        return "\n".join(report)


def run_performance_baseline_creation():
    """Main function to create performance baselines."""
    print("Starting Performance Baseline Creation for Task 4.1")
    print("=" * 60)

    manager = PerformanceBaselineManager()

    # Create initial baseline
    baseline_id = "task_4_1_initial"
    baseline = manager.create_comprehensive_baseline(baseline_id)

    # Generate and display report
    report = manager.generate_baseline_report(baseline)
    print("\n" + report)

    # Save report to file
    report_file = f"performance_baseline_report_{baseline_id}.txt"
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"\n📄 Detailed report saved to: {report_file}")
    print(f"🎯 Baseline '{baseline_id}' created successfully!")

    return baseline


if __name__ == "__main__":
    run_performance_baseline_creation()
