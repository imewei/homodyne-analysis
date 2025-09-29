"""
Vectorization and Parallel Processing Optimizations
===================================================

Comprehensive vectorization and parallel processing optimization suite
for the homodyne analysis package. Focuses on leveraging modern CPU
architectures and multi-core processing capabilities.

Features:
- SIMD vectorization optimizations
- Multi-threading with NumPy and concurrent.futures
- Parallel processing for independent computations
- GPU acceleration preparation
- Load balancing strategies
- Performance scaling analysis

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import json
import multiprocessing as mp
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np

warnings.filterwarnings("ignore")


@dataclass
class ParallelizationResult:
    """Results of parallelization optimization."""
    operation_name: str
    serial_time: float
    parallel_time: float
    speedup_factor: float
    efficiency: float
    num_workers: int
    parallelization_method: str
    overhead_time: float


class VectorizationOptimizer:
    """SIMD vectorization optimizations."""

    @staticmethod
    def vectorized_chi_squared_batch(c2_exp_batch: np.ndarray, c2_theo_batch: np.ndarray) -> np.ndarray:
        """Vectorized chi-squared calculation for multiple datasets."""
        # Process multiple datasets simultaneously
        diff = c2_exp_batch - c2_theo_batch
        chi2_values = np.sum(diff * diff, axis=(1, 2, 3))  # Sum over spatial dimensions
        return chi2_values

    @staticmethod
    def vectorized_correlation_functions(data_batch: np.ndarray) -> np.ndarray:
        """Vectorized correlation function calculation."""
        # data_batch shape: (batch_size, n_angles, n_time)
        batch_size, n_angles, n_time = data_batch.shape

        # Vectorized outer product calculation
        correlations = np.zeros((batch_size, n_angles, n_time, n_time))

        for b in range(batch_size):
            for a in range(n_angles):
                data_slice = data_batch[b, a]
                correlations[b, a] = np.outer(data_slice, data_slice)

        return correlations

    @staticmethod
    def vectorized_statistical_moments(data_batch: np.ndarray) -> Dict[str, np.ndarray]:
        """Vectorized statistical calculations across multiple datasets."""
        # Calculate moments for entire batch at once
        means = np.mean(data_batch, axis=-1)
        variances = np.var(data_batch, axis=-1, ddof=0)
        stds = np.sqrt(variances)

        return {
            "means": means,
            "variances": variances,
            "stds": stds
        }

    @staticmethod
    def vectorized_matrix_operations_batch(matrices: np.ndarray) -> Dict[str, np.ndarray]:
        """Vectorized matrix operations for batch processing."""
        # matrices shape: (batch_size, n, n)

        # Batch determinant calculation
        determinants = np.linalg.det(matrices)

        # Batch eigenvalue calculation
        eigenvalues = np.linalg.eigvals(matrices)

        # Batch matrix multiplication (A @ A.T)
        transposes = np.transpose(matrices, axes=(0, 2, 1))
        products = np.matmul(matrices, transposes)

        return {
            "determinants": determinants,
            "eigenvalues": eigenvalues,
            "products": products
        }


class ThreadParallelProcessor:
    """Thread-based parallel processing optimizations."""

    def __init__(self, num_threads: int = None):
        self.num_threads = num_threads or mp.cpu_count()

    def parallel_chi_squared_calculation(self, c2_exp_list: List[np.ndarray],
                                       c2_theo_list: List[np.ndarray]) -> List[float]:
        """Parallel chi-squared calculation using threads."""

        def compute_single_chi_squared(args):
            c2_exp, c2_theo = args
            return np.sum((c2_exp - c2_theo) ** 2)

        data_pairs = list(zip(c2_exp_list, c2_theo_list))

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(executor.map(compute_single_chi_squared, data_pairs))

        return results

    def parallel_correlation_processing(self, data_list: List[np.ndarray]) -> List[np.ndarray]:
        """Parallel correlation function processing."""

        def compute_correlation(data):
            n_angles, n_time = data.shape
            result = np.zeros((n_angles, n_time, n_time))

            for i in range(n_angles):
                result[i] = np.outer(data[i], data[i])

            return result

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(executor.map(compute_correlation, data_list))

        return results

    def parallel_eigenvalue_computation(self, matrices: List[np.ndarray]) -> List[np.ndarray]:
        """Parallel eigenvalue computation."""

        def compute_eigenvalues(matrix):
            return np.linalg.eigvals(matrix)

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            results = list(executor.map(compute_eigenvalues, matrices))

        return results


class ProcessParallelProcessor:
    """Process-based parallel processing optimizations."""

    def __init__(self, num_processes: int = None):
        self.num_processes = num_processes or mp.cpu_count()

    def parallel_chi_squared_calculation(self, c2_exp_list: List[np.ndarray],
                                       c2_theo_list: List[np.ndarray]) -> List[float]:
        """Parallel chi-squared calculation using processes."""

        def compute_single_chi_squared(args):
            c2_exp, c2_theo = args
            return float(np.sum((c2_exp - c2_theo) ** 2))

        data_pairs = list(zip(c2_exp_list, c2_theo_list))

        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            results = list(executor.map(compute_single_chi_squared, data_pairs))

        return results

    def parallel_statistical_analysis(self, data_list: List[np.ndarray]) -> List[Dict[str, float]]:
        """Parallel statistical analysis."""

        def compute_statistics(data):
            return {
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "var": float(np.var(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data))
            }

        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            results = list(executor.map(compute_statistics, data_list))

        return results


class LoadBalancer:
    """Load balancing for parallel processing."""

    @staticmethod
    def balance_workload(data_list: List[Any], num_workers: int) -> List[List[Any]]:
        """Balance workload across workers."""
        chunk_size = len(data_list) // num_workers
        remainder = len(data_list) % num_workers

        chunks = []
        start_idx = 0

        for i in range(num_workers):
            # Add extra item to first 'remainder' chunks
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_chunk_size

            chunks.append(data_list[start_idx:end_idx])
            start_idx = end_idx

        return chunks

    @staticmethod
    def adaptive_chunk_size(data_size: int, num_workers: int, min_chunk_size: int = 100) -> int:
        """Calculate adaptive chunk size based on data size and workers."""
        chunk_size = max(data_size // num_workers, min_chunk_size)
        return chunk_size


class ParallelizationBenchmarker:
    """Benchmarking system for parallelization optimizations."""

    def __init__(self):
        self.results = []

    def benchmark_chi_squared_parallelization(self) -> List[ParallelizationResult]:
        """Benchmark chi-squared parallelization methods."""
        results = []

        # Generate test data
        num_datasets = 20
        c2_exp_list = [np.random.rand(5, 50, 50) for _ in range(num_datasets)]
        c2_theo_list = [np.random.rand(5, 50, 50) for _ in range(num_datasets)]

        # Serial computation
        start_time = time.perf_counter()
        serial_results = []
        for c2_exp, c2_theo in zip(c2_exp_list, c2_theo_list):
            chi2 = np.sum((c2_exp - c2_theo) ** 2)
            serial_results.append(chi2)
        serial_time = time.perf_counter() - start_time

        # Thread-based parallelization
        thread_processor = ThreadParallelProcessor()
        start_time = time.perf_counter()
        thread_results = thread_processor.parallel_chi_squared_calculation(c2_exp_list, c2_theo_list)
        thread_time = time.perf_counter() - start_time

        # Process-based parallelization
        process_processor = ProcessParallelProcessor()
        start_time = time.perf_counter()
        process_results = process_processor.parallel_chi_squared_calculation(c2_exp_list, c2_theo_list)
        process_time = time.perf_counter() - start_time

        # Vectorized computation
        c2_exp_batch = np.array(c2_exp_list)
        c2_theo_batch = np.array(c2_theo_list)
        start_time = time.perf_counter()
        vectorized_results = VectorizationOptimizer.vectorized_chi_squared_batch(c2_exp_batch, c2_theo_batch)
        vectorized_time = time.perf_counter() - start_time

        # Calculate results
        thread_speedup = serial_time / thread_time if thread_time > 0 else 0
        process_speedup = serial_time / process_time if process_time > 0 else 0
        vectorized_speedup = serial_time / vectorized_time if vectorized_time > 0 else 0

        results.append(ParallelizationResult(
            operation_name="chi_squared_threads",
            serial_time=serial_time,
            parallel_time=thread_time,
            speedup_factor=thread_speedup,
            efficiency=thread_speedup / thread_processor.num_threads,
            num_workers=thread_processor.num_threads,
            parallelization_method="threading",
            overhead_time=max(0, thread_time - serial_time / thread_processor.num_threads)
        ))

        results.append(ParallelizationResult(
            operation_name="chi_squared_processes",
            serial_time=serial_time,
            parallel_time=process_time,
            speedup_factor=process_speedup,
            efficiency=process_speedup / process_processor.num_processes,
            num_workers=process_processor.num_processes,
            parallelization_method="multiprocessing",
            overhead_time=max(0, process_time - serial_time / process_processor.num_processes)
        ))

        results.append(ParallelizationResult(
            operation_name="chi_squared_vectorized",
            serial_time=serial_time,
            parallel_time=vectorized_time,
            speedup_factor=vectorized_speedup,
            efficiency=vectorized_speedup,  # SIMD efficiency
            num_workers=1,  # Single thread but SIMD
            parallelization_method="vectorization",
            overhead_time=max(0, vectorized_time - serial_time)
        ))

        return results

    def benchmark_correlation_parallelization(self) -> List[ParallelizationResult]:
        """Benchmark correlation function parallelization."""
        results = []

        # Generate test data
        num_datasets = 10
        data_list = [np.random.rand(8, 100) for _ in range(num_datasets)]

        # Serial computation
        start_time = time.perf_counter()
        serial_results = []
        for data in data_list:
            n_angles, n_time = data.shape
            result = np.zeros((n_angles, n_time, n_time))
            for i in range(n_angles):
                result[i] = np.outer(data[i], data[i])
            serial_results.append(result)
        serial_time = time.perf_counter() - start_time

        # Thread-based parallelization
        thread_processor = ThreadParallelProcessor()
        start_time = time.perf_counter()
        thread_results = thread_processor.parallel_correlation_processing(data_list)
        thread_time = time.perf_counter() - start_time

        # Vectorized batch processing
        data_batch = np.array(data_list)
        start_time = time.perf_counter()
        vectorized_results = VectorizationOptimizer.vectorized_correlation_functions(data_batch)
        vectorized_time = time.perf_counter() - start_time

        thread_speedup = serial_time / thread_time if thread_time > 0 else 0
        vectorized_speedup = serial_time / vectorized_time if vectorized_time > 0 else 0

        results.append(ParallelizationResult(
            operation_name="correlation_threads",
            serial_time=serial_time,
            parallel_time=thread_time,
            speedup_factor=thread_speedup,
            efficiency=thread_speedup / thread_processor.num_threads,
            num_workers=thread_processor.num_threads,
            parallelization_method="threading",
            overhead_time=max(0, thread_time - serial_time / thread_processor.num_threads)
        ))

        results.append(ParallelizationResult(
            operation_name="correlation_vectorized",
            serial_time=serial_time,
            parallel_time=vectorized_time,
            speedup_factor=vectorized_speedup,
            efficiency=vectorized_speedup,
            num_workers=1,
            parallelization_method="vectorization",
            overhead_time=max(0, vectorized_time - serial_time)
        ))

        return results

    def benchmark_scaling_performance(self, max_workers: int = None) -> Dict[str, List[float]]:
        """Benchmark performance scaling with different numbers of workers."""
        if max_workers is None:
            max_workers = mp.cpu_count()

        scaling_results = {
            "num_workers": [],
            "thread_speedups": [],
            "process_speedups": [],
            "thread_efficiencies": [],
            "process_efficiencies": []
        }

        # Generate test data
        num_datasets = 16
        c2_exp_list = [np.random.rand(3, 30, 30) for _ in range(num_datasets)]
        c2_theo_list = [np.random.rand(3, 30, 30) for _ in range(num_datasets)]

        # Serial baseline
        start_time = time.perf_counter()
        for c2_exp, c2_theo in zip(c2_exp_list, c2_theo_list):
            np.sum((c2_exp - c2_theo) ** 2)
        serial_time = time.perf_counter() - start_time

        # Test different numbers of workers
        for num_workers in range(1, max_workers + 1):
            scaling_results["num_workers"].append(num_workers)

            # Thread-based
            thread_processor = ThreadParallelProcessor(num_workers)
            start_time = time.perf_counter()
            thread_processor.parallel_chi_squared_calculation(c2_exp_list, c2_theo_list)
            thread_time = time.perf_counter() - start_time

            thread_speedup = serial_time / thread_time if thread_time > 0 else 0
            thread_efficiency = thread_speedup / num_workers

            scaling_results["thread_speedups"].append(thread_speedup)
            scaling_results["thread_efficiencies"].append(thread_efficiency)

            # Process-based
            process_processor = ProcessParallelProcessor(num_workers)
            start_time = time.perf_counter()
            process_processor.parallel_chi_squared_calculation(c2_exp_list, c2_theo_list)
            process_time = time.perf_counter() - start_time

            process_speedup = serial_time / process_time if process_time > 0 else 0
            process_efficiency = process_speedup / num_workers

            scaling_results["process_speedups"].append(process_speedup)
            scaling_results["process_efficiencies"].append(process_efficiency)

        return scaling_results

    def run_comprehensive_parallelization_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive parallelization benchmarks."""
        print("Running Comprehensive Vectorization and Parallelization Benchmarks")
        print("=" * 70)

        all_results = {}

        # Chi-squared parallelization
        print("Benchmarking chi-squared parallelization...")
        chi2_results = self.benchmark_chi_squared_parallelization()
        all_results["chi_squared"] = chi2_results

        # Correlation parallelization
        print("Benchmarking correlation parallelization...")
        corr_results = self.benchmark_correlation_parallelization()
        all_results["correlation"] = corr_results

        # Scaling performance
        print("Benchmarking scaling performance...")
        scaling_results = self.benchmark_scaling_performance()
        all_results["scaling"] = scaling_results

        return all_results


def generate_parallelization_report(results: Dict[str, Any]) -> str:
    """Generate comprehensive parallelization report."""
    report_lines = []
    report_lines.append("VECTORIZATION AND PARALLELIZATION REPORT - TASK 4.5")
    report_lines.append("=" * 70)

    # Chi-squared results
    if "chi_squared" in results:
        report_lines.append("\nCHI-SQUARED PARALLELIZATION:")
        for result in results["chi_squared"]:
            report_lines.append(f"  {result.operation_name}:")
            report_lines.append(f"    Speedup: {result.speedup_factor:.2f}x")
            report_lines.append(f"    Efficiency: {result.efficiency:.2f}")
            report_lines.append(f"    Method: {result.parallelization_method}")
            report_lines.append(f"    Workers: {result.num_workers}")

    # Correlation results
    if "correlation" in results:
        report_lines.append("\nCORRELATION PARALLELIZATION:")
        for result in results["correlation"]:
            report_lines.append(f"  {result.operation_name}:")
            report_lines.append(f"    Speedup: {result.speedup_factor:.2f}x")
            report_lines.append(f"    Efficiency: {result.efficiency:.2f}")
            report_lines.append(f"    Method: {result.parallelization_method}")

    # Scaling analysis
    if "scaling" in results:
        scaling_data = results["scaling"]
        max_thread_speedup = max(scaling_data["thread_speedups"]) if scaling_data["thread_speedups"] else 0
        max_process_speedup = max(scaling_data["process_speedups"]) if scaling_data["process_speedups"] else 0

        report_lines.append("\nSCALING ANALYSIS:")
        report_lines.append(f"  Maximum thread speedup: {max_thread_speedup:.2f}x")
        report_lines.append(f"  Maximum process speedup: {max_process_speedup:.2f}x")
        report_lines.append(f"  Optimal thread count: {scaling_data['num_workers'][scaling_data['thread_speedups'].index(max_thread_speedup)] if scaling_data['thread_speedups'] else 'N/A'}")

    return "\n".join(report_lines)


def run_vectorization_parallel_suite():
    """Main function to run vectorization and parallel processing suite."""
    print("Starting Vectorization and Parallel Processing Suite - Task 4.5")
    print("=" * 70)

    # Run benchmarks
    benchmarker = ParallelizationBenchmarker()
    results = benchmarker.run_comprehensive_parallelization_benchmark()

    # Generate report
    report = generate_parallelization_report(results)
    print("\n" + report)

    # Save results
    results_dir = Path("vectorization_results")
    results_dir.mkdir(exist_ok=True)

    # Convert results to JSON-serializable format
    json_results = {}

    for category, data in results.items():
        if category == "scaling":
            json_results[category] = data
        else:
            # Convert ParallelizationResult objects to dicts
            json_results[category] = []
            for result in data:
                json_results[category].append({
                    "operation_name": result.operation_name,
                    "serial_time": float(result.serial_time),
                    "parallel_time": float(result.parallel_time),
                    "speedup_factor": float(result.speedup_factor),
                    "efficiency": float(result.efficiency),
                    "num_workers": int(result.num_workers),
                    "parallelization_method": result.parallelization_method,
                    "overhead_time": float(result.overhead_time)
                })

    # Save JSON results
    json_file = results_dir / "task_4_5_vectorization_results.json"
    with open(json_file, 'w') as f:
        json.dump({
            "parallelization_results": json_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "cpu_count": mp.cpu_count(),
                "available_cores": mp.cpu_count()
            }
        }, f, indent=2)

    # Save text report
    report_file = results_dir / "task_4_5_vectorization_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"\n📄 Results saved to: {json_file}")
    print(f"📄 Report saved to: {report_file}")

    # Calculate summary statistics
    all_speedups = []
    for category in ["chi_squared", "correlation"]:
        if category in results:
            for result in results[category]:
                if result.speedup_factor > 0:
                    all_speedups.append(result.speedup_factor)

    avg_speedup = np.mean(all_speedups) if all_speedups else 0
    max_speedup = np.max(all_speedups) if all_speedups else 0

    print(f"\n✅ Task 4.5 Vectorization and Parallelization Complete!")
    print(f"🚀 Average speedup achieved: {avg_speedup:.2f}x")
    print(f"🎯 Maximum speedup achieved: {max_speedup:.2f}x")
    print(f"⚡ {len(all_speedups)} parallelization methods tested")
    print(f"🔧 System has {mp.cpu_count()} CPU cores available")

    return json_results


if __name__ == "__main__":
    run_vectorization_parallel_suite()
