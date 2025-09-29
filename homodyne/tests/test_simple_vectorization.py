"""
Simplified Vectorization and Parallel Processing Optimizations
==============================================================

Simplified vectorization and parallel processing optimization suite
focusing on threading and vectorization without multiprocessing issues.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


@dataclass
class OptimizationResult:
    """Results of optimization."""

    method: str
    execution_time: float
    speedup_factor: float
    data_size: int
    num_workers: int


def compute_chi_squared_single(c2_exp, c2_theo):
    """Global function for chi-squared calculation."""
    return float(np.sum((c2_exp - c2_theo) ** 2))


def compute_correlation_single(data):
    """Global function for correlation calculation."""
    n_angles, n_time = data.shape
    result = np.zeros((n_angles, n_time, n_time))
    for i in range(n_angles):
        result[i] = np.outer(data[i], data[i])
    return result


class VectorizationOptimizer:
    """Vectorization optimizations."""

    @staticmethod
    def vectorized_chi_squared_batch(c2_exp_batch, c2_theo_batch):
        """Vectorized chi-squared for batch processing."""
        diff = c2_exp_batch - c2_theo_batch
        return np.sum(diff * diff, axis=(1, 2, 3))

    @staticmethod
    def vectorized_statistical_operations(data_batch):
        """Vectorized statistical operations."""
        return {
            "means": np.mean(data_batch, axis=-1),
            "stds": np.std(data_batch, axis=-1),
            "vars": np.var(data_batch, axis=-1),
        }

    @staticmethod
    def vectorized_matrix_operations(matrices):
        """Vectorized matrix operations."""
        # Batch matrix multiplication
        transposes = np.transpose(matrices, (0, 2, 1))
        products = np.matmul(matrices, transposes)

        return {
            "products": products,
            "traces": np.trace(matrices, axis1=1, axis2=2),
            "determinants": np.linalg.det(matrices),
        }


class ThreadingOptimizer:
    """Threading-based optimizations."""

    def __init__(self, num_threads=4):
        self.num_threads = num_threads

    def parallel_chi_squared(self, c2_exp_list, c2_theo_list):
        """Parallel chi-squared calculation using threads."""
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for c2_exp, c2_theo in zip(c2_exp_list, c2_theo_list, strict=False):
                future = executor.submit(compute_chi_squared_single, c2_exp, c2_theo)
                futures.append(future)

            results = [future.result() for future in futures]

        return results

    def parallel_correlations(self, data_list):
        """Parallel correlation calculation using threads."""
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for data in data_list:
                future = executor.submit(compute_correlation_single, data)
                futures.append(future)

            results = [future.result() for future in futures]

        return results


class PerformanceBenchmarker:
    """Performance benchmarking system."""

    def __init__(self):
        self.results = []

    def benchmark_vectorization_vs_loops(self):
        """Benchmark vectorization vs loop-based operations."""
        print("Benchmarking vectorization vs loops...")

        # Test data
        batch_size = 20
        c2_exp_list = [np.random.rand(3, 30, 30) for _ in range(batch_size)]
        c2_theo_list = [np.random.rand(3, 30, 30) for _ in range(batch_size)]

        # Loop-based calculation
        start_time = time.perf_counter()
        loop_results = []
        for c2_exp, c2_theo in zip(c2_exp_list, c2_theo_list, strict=False):
            chi2 = compute_chi_squared_single(c2_exp, c2_theo)
            loop_results.append(chi2)
        loop_time = time.perf_counter() - start_time

        # Vectorized calculation
        c2_exp_batch = np.array(c2_exp_list)
        c2_theo_batch = np.array(c2_theo_list)

        start_time = time.perf_counter()
        vectorized_results = VectorizationOptimizer.vectorized_chi_squared_batch(
            c2_exp_batch, c2_theo_batch
        )
        vectorized_time = time.perf_counter() - start_time

        speedup = loop_time / vectorized_time if vectorized_time > 0 else 0

        return {
            "loop_time": loop_time,
            "vectorized_time": vectorized_time,
            "speedup": speedup,
            "accuracy_preserved": np.allclose(
                loop_results, vectorized_results, rtol=1e-10
            ),
        }

    def benchmark_threading_speedup(self):
        """Benchmark threading speedup."""
        print("Benchmarking threading speedup...")

        # Test data
        num_datasets = 16
        c2_exp_list = [np.random.rand(5, 40, 40) for _ in range(num_datasets)]
        c2_theo_list = [np.random.rand(5, 40, 40) for _ in range(num_datasets)]

        # Serial execution
        start_time = time.perf_counter()
        serial_results = []
        for c2_exp, c2_theo in zip(c2_exp_list, c2_theo_list, strict=False):
            result = compute_chi_squared_single(c2_exp, c2_theo)
            serial_results.append(result)
        serial_time = time.perf_counter() - start_time

        # Test different thread counts
        thread_results = {}
        for num_threads in [1, 2, 4, 8]:
            threading_optimizer = ThreadingOptimizer(num_threads)

            start_time = time.perf_counter()
            threaded_results = threading_optimizer.parallel_chi_squared(
                c2_exp_list, c2_theo_list
            )
            threaded_time = time.perf_counter() - start_time

            speedup = serial_time / threaded_time if threaded_time > 0 else 0
            efficiency = speedup / num_threads

            thread_results[f"threads_{num_threads}"] = {
                "time": threaded_time,
                "speedup": speedup,
                "efficiency": efficiency,
                "accuracy_preserved": np.allclose(
                    serial_results, threaded_results, rtol=1e-10
                ),
            }

        return thread_results

    def benchmark_statistical_vectorization(self):
        """Benchmark statistical operation vectorization."""
        print("Benchmarking statistical vectorization...")

        # Test data
        batch_size = 100
        data_size = 1000
        data_list = [np.random.rand(data_size) for _ in range(batch_size)]

        # Loop-based statistical calculations
        start_time = time.perf_counter()
        loop_stats = []
        for data in data_list:
            stats = {"mean": np.mean(data), "std": np.std(data), "var": np.var(data)}
            loop_stats.append(stats)
        loop_time = time.perf_counter() - start_time

        # Vectorized statistical calculations
        data_batch = np.array(data_list)

        start_time = time.perf_counter()
        vectorized_stats = VectorizationOptimizer.vectorized_statistical_operations(
            data_batch
        )
        vectorized_time = time.perf_counter() - start_time

        speedup = loop_time / vectorized_time if vectorized_time > 0 else 0

        # Check accuracy
        loop_means = [s["mean"] for s in loop_stats]
        accuracy = np.allclose(loop_means, vectorized_stats["means"], rtol=1e-10)

        return {
            "loop_time": loop_time,
            "vectorized_time": vectorized_time,
            "speedup": speedup,
            "accuracy_preserved": accuracy,
        }

    def benchmark_matrix_operations(self):
        """Benchmark matrix operation vectorization."""
        print("Benchmarking matrix operations...")

        # Test data
        batch_size = 20
        matrix_size = 50
        matrices = np.random.rand(batch_size, matrix_size, matrix_size)

        # Loop-based matrix operations
        start_time = time.perf_counter()
        loop_results = []
        for i in range(batch_size):
            matrix = matrices[i]
            result = {
                "product": matrix @ matrix.T,
                "trace": np.trace(matrix),
                "determinant": np.linalg.det(matrix),
            }
            loop_results.append(result)
        loop_time = time.perf_counter() - start_time

        # Vectorized matrix operations
        start_time = time.perf_counter()
        vectorized_results = VectorizationOptimizer.vectorized_matrix_operations(
            matrices
        )
        vectorized_time = time.perf_counter() - start_time

        speedup = loop_time / vectorized_time if vectorized_time > 0 else 0

        # Check accuracy for traces
        loop_traces = [r["trace"] for r in loop_results]
        accuracy = np.allclose(loop_traces, vectorized_results["traces"], rtol=1e-10)

        return {
            "loop_time": loop_time,
            "vectorized_time": vectorized_time,
            "speedup": speedup,
            "accuracy_preserved": accuracy,
        }

    def run_comprehensive_benchmark(self):
        """Run comprehensive vectorization and threading benchmarks."""
        print("Running Comprehensive Vectorization and Threading Benchmarks")
        print("=" * 65)

        results = {}

        # Vectorization vs loops
        results["vectorization"] = self.benchmark_vectorization_vs_loops()

        # Threading speedup
        results["threading"] = self.benchmark_threading_speedup()

        # Statistical vectorization
        results["statistical"] = self.benchmark_statistical_vectorization()

        # Matrix operations
        results["matrix_operations"] = self.benchmark_matrix_operations()

        return results


def generate_optimization_report(results):
    """Generate comprehensive optimization report."""
    report_lines = []
    report_lines.append("VECTORIZATION AND THREADING OPTIMIZATION REPORT - TASK 4.5")
    report_lines.append("=" * 70)

    # Vectorization results
    if "vectorization" in results:
        vec_data = results["vectorization"]
        report_lines.append("\nVECTORIZATION OPTIMIZATION:")
        report_lines.append(f"  Speedup: {vec_data['speedup']:.2f}x")
        report_lines.append(f"  Loop time: {vec_data['loop_time']:.4f}s")
        report_lines.append(f"  Vectorized time: {vec_data['vectorized_time']:.4f}s")
        report_lines.append(
            f"  Accuracy preserved: {'✓' if vec_data['accuracy_preserved'] else '✗'}"
        )

    # Threading results
    if "threading" in results:
        report_lines.append("\nTHREADING OPTIMIZATION:")
        for thread_config, data in results["threading"].items():
            num_threads = thread_config.split("_")[1]
            report_lines.append(f"  {num_threads} threads:")
            report_lines.append(f"    Speedup: {data['speedup']:.2f}x")
            report_lines.append(f"    Efficiency: {data['efficiency']:.2f}")
            report_lines.append(f"    Time: {data['time']:.4f}s")

    # Statistical operations
    if "statistical" in results:
        stats_data = results["statistical"]
        report_lines.append("\nSTATISTICAL OPERATIONS:")
        report_lines.append(f"  Speedup: {stats_data['speedup']:.2f}x")
        report_lines.append(
            f"  Accuracy preserved: {'✓' if stats_data['accuracy_preserved'] else '✗'}"
        )

    # Matrix operations
    if "matrix_operations" in results:
        matrix_data = results["matrix_operations"]
        report_lines.append("\nMATRIX OPERATIONS:")
        report_lines.append(f"  Speedup: {matrix_data['speedup']:.2f}x")
        report_lines.append(
            f"  Accuracy preserved: {'✓' if matrix_data['accuracy_preserved'] else '✗'}"
        )

    # Calculate overall statistics
    speedups = []
    if "vectorization" in results:
        speedups.append(results["vectorization"]["speedup"])
    if "statistical" in results:
        speedups.append(results["statistical"]["speedup"])
    if "matrix_operations" in results:
        speedups.append(results["matrix_operations"]["speedup"])

    if speedups:
        avg_speedup = np.mean(speedups)
        max_speedup = np.max(speedups)
        report_lines.append("\nOVERALL PERFORMANCE:")
        report_lines.append(f"  Average speedup: {avg_speedup:.2f}x")
        report_lines.append(f"  Maximum speedup: {max_speedup:.2f}x")

    return "\n".join(report_lines)


def run_simple_vectorization_suite():
    """Main function to run simplified vectorization suite."""
    print("Starting Simplified Vectorization and Threading Suite - Task 4.5")
    print("=" * 70)

    # Run benchmarks
    benchmarker = PerformanceBenchmarker()
    results = benchmarker.run_comprehensive_benchmark()

    # Generate report
    report = generate_optimization_report(results)
    print("\n" + report)

    # Save results
    results_dir = Path("vectorization_simple_results")
    results_dir.mkdir(exist_ok=True)

    # Save JSON results
    json_file = results_dir / "task_4_5_simple_vectorization_results.json"
    with open(json_file, "w") as f:
        json.dump(
            {
                "optimization_results": results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "summary": {
                    "techniques_tested": list(results.keys()),
                    "total_benchmarks": len(results),
                },
            },
            f,
            indent=2,
        )

    # Save text report
    report_file = results_dir / "task_4_5_simple_vectorization_report.txt"
    with open(report_file, "w") as f:
        f.write(report)

    print(f"\n📄 Results saved to: {json_file}")
    print(f"📄 Report saved to: {report_file}")

    # Calculate summary statistics
    speedups = []
    if "vectorization" in results:
        speedups.append(results["vectorization"]["speedup"])
    if "statistical" in results:
        speedups.append(results["statistical"]["speedup"])
    if "matrix_operations" in results:
        speedups.append(results["matrix_operations"]["speedup"])

    avg_speedup = np.mean(speedups) if speedups else 0
    max_speedup = np.max(speedups) if speedups else 0

    print("\n✅ Task 4.5 Vectorization and Threading Complete!")
    print(f"🚀 Average speedup achieved: {avg_speedup:.2f}x")
    print(f"🎯 Maximum speedup achieved: {max_speedup:.2f}x")
    print(f"⚡ {len(results)} optimization techniques tested")

    return results


if __name__ == "__main__":
    run_simple_vectorization_suite()
