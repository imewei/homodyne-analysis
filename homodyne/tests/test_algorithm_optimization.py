"""
Core Algorithm Optimization Suite
=================================

Comprehensive optimization of computational bottlenecks in homodyne analysis
core algorithms. Focuses on numerical performance improvements while maintaining
mathematical accuracy.

Features:
- Bottleneck identification and profiling
- Vectorization optimizations
- Memory-efficient implementations
- Numerical stability improvements
- Performance comparisons

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import json
import time

import numpy as np

# Conditional numba import
try:
    from numba import jit
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    # Mock decorators if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if args else decorator

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if args else decorator

    NUMBA_AVAILABLE = False
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")


@dataclass
class OptimizationResult:
    """Results of algorithm optimization."""

    original_time: float
    optimized_time: float
    speedup_factor: float
    memory_reduction: float
    accuracy_preserved: bool
    optimization_method: str


class MathematicalOptimizer:
    """Optimizations for core mathematical operations."""

    @staticmethod
    @njit
    def chi_squared_optimized(c2_exp: np.ndarray, c2_theo: np.ndarray) -> float:
        """Optimized chi-squared calculation using Numba JIT."""
        return np.sum((c2_exp - c2_theo) ** 2)

    @staticmethod
    def chi_squared_vectorized(c2_exp: np.ndarray, c2_theo: np.ndarray) -> float:
        """Vectorized chi-squared calculation."""
        diff = c2_exp - c2_theo
        return np.sum(diff * diff)  # Slightly faster than **2

    @staticmethod
    def chi_squared_original(c2_exp: np.ndarray, c2_theo: np.ndarray) -> float:
        """Original chi-squared calculation."""
        chi2 = 0.0
        flat_exp = c2_exp.flatten()
        flat_theo = c2_theo.flatten()
        for i in range(len(flat_exp)):
            diff = flat_exp[i] - flat_theo[i]
            chi2 += diff * diff
        return chi2

    @staticmethod
    @njit
    def matrix_multiply_optimized(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Optimized matrix multiplication."""
        return np.dot(A, B)

    @staticmethod
    def correlation_function_optimized(
        data: np.ndarray, angles: np.ndarray
    ) -> np.ndarray:
        """Optimized correlation function calculation."""
        n_angles, n_time = data.shape[0], data.shape[1]
        result = np.zeros((n_angles, n_time, n_time))

        # Vectorized calculation
        for i in range(n_angles):
            # Use broadcasting for efficiency
            data_i = data[i]
            result[i] = np.outer(data_i, data_i)

        return result

    @staticmethod
    def correlation_function_original(
        data: np.ndarray, angles: np.ndarray
    ) -> np.ndarray:
        """Original correlation function calculation."""
        n_angles, n_time = data.shape[0], data.shape[1]
        result = np.zeros((n_angles, n_time, n_time))

        for i in range(n_angles):
            for j in range(n_time):
                for k in range(n_time):
                    result[i, j, k] = data[i, j] * data[i, k]

        return result

    @staticmethod
    @njit
    def eigenvalue_computation_optimized(matrix: np.ndarray) -> np.ndarray:
        """Optimized eigenvalue computation using Numba."""
        # Note: Numba doesn't support scipy.linalg.eig directly
        # This is a placeholder for eigenvalue optimization
        return np.linalg.eigvals(matrix)

    @staticmethod
    def statistical_moments_optimized(data: np.ndarray) -> dict[str, float]:
        """Optimized statistical moment calculations."""
        # Use single-pass algorithm for efficiency
        n = len(data)
        if n == 0:
            return {"mean": 0.0, "variance": 0.0, "std": 0.0}

        # Single pass for mean and variance
        mean = np.mean(data)
        variance = np.var(data, ddof=0)
        std = np.sqrt(variance)

        return {"mean": float(mean), "variance": float(variance), "std": float(std)}

    @staticmethod
    def statistical_moments_original(data: np.ndarray) -> dict[str, float]:
        """Original statistical moment calculations."""
        # Multi-pass algorithm
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std = variance**0.5

        return {"mean": mean, "variance": variance, "std": std}


class MemoryOptimizer:
    """Memory efficiency optimizations."""

    @staticmethod
    def in_place_operations(data: np.ndarray) -> np.ndarray:
        """Use in-place operations to reduce memory allocation."""
        # Instead of creating new arrays, modify in place
        data *= 2.0  # In-place multiplication
        data += 1.0  # In-place addition
        np.sqrt(data, out=data)  # In-place square root
        return data

    @staticmethod
    def memory_efficient_correlation(
        data: np.ndarray, chunk_size: int = 100
    ) -> np.ndarray:
        """Memory-efficient correlation calculation using chunking."""
        n_angles, n_time = data.shape[0], data.shape[1]

        if n_time <= chunk_size:
            # Small enough to process directly
            return MathematicalOptimizer.correlation_function_optimized(data, None)

        # Process in chunks
        result = np.zeros((n_angles, n_time, n_time))

        for i in range(0, n_time, chunk_size):
            end_i = min(i + chunk_size, n_time)
            for j in range(0, n_time, chunk_size):
                end_j = min(j + chunk_size, n_time)

                # Process chunk
                for angle_idx in range(n_angles):
                    chunk_i = data[angle_idx, i:end_i]
                    chunk_j = data[angle_idx, j:end_j]
                    result[angle_idx, i:end_i, j:end_j] = np.outer(chunk_i, chunk_j)

        return result

    @staticmethod
    def pre_allocate_arrays(
        shape: tuple[int, ...], dtype: np.dtype = np.float64
    ) -> np.ndarray:
        """Pre-allocate arrays with optimal memory layout."""
        # Use C-contiguous arrays for better cache performance
        return np.zeros(shape, dtype=dtype, order="C")


class NumericalStabilityOptimizer:
    """Optimizations for numerical stability."""

    @staticmethod
    def stable_chi_squared(
        c2_exp: np.ndarray, c2_theo: np.ndarray, epsilon: float = 1e-15
    ) -> float:
        """Numerically stable chi-squared calculation."""
        # Avoid overflow and underflow
        diff = c2_exp - c2_theo

        # Clamp extreme values
        diff = np.clip(diff, -1e10, 1e10)

        # Use Kahan summation for better numerical accuracy
        chi2 = 0.0
        compensation = 0.0

        for val in diff.flat:
            val_squared = val * val
            y = val_squared - compensation
            t = chi2 + y
            compensation = (t - chi2) - y
            chi2 = t

        return chi2

    @staticmethod
    def robust_eigenvalue_computation(matrix: np.ndarray) -> np.ndarray:
        """Robust eigenvalue computation with condition number checking."""
        # Check condition number
        cond_num = np.linalg.cond(matrix)

        if cond_num > 1e12:
            # Matrix is ill-conditioned, use regularization
            regularization = 1e-10 * np.eye(matrix.shape[0])
            matrix_reg = matrix + regularization
            return np.linalg.eigvals(matrix_reg)
        return np.linalg.eigvals(matrix)

    @staticmethod
    def stable_correlation_normalization(correlation: np.ndarray) -> np.ndarray:
        """Numerically stable correlation normalization."""
        # Avoid division by zero
        max_val = np.max(np.abs(correlation))
        if max_val < 1e-15:
            return correlation

        # Normalize by maximum value first, then rescale
        normalized = correlation / max_val
        min_val = np.min(normalized)
        range_val = np.max(normalized) - min_val

        if range_val < 1e-15:
            return normalized

        return (normalized - min_val) / range_val


class AlgorithmBenchmarker:
    """Benchmarking system for algorithm optimizations."""

    def __init__(self):
        self.results = {}

    def benchmark_chi_squared_optimizations(self) -> dict[str, OptimizationResult]:
        """Benchmark chi-squared calculation optimizations."""
        results = {}

        # Test different data sizes
        data_sizes = [(5, 20, 20), (10, 50, 50), (15, 100, 100)]

        for n_angles, n_time1, n_time2 in data_sizes:
            # Generate test data
            c2_exp = np.random.rand(n_angles, n_time1, n_time2)
            c2_theo = np.random.rand(n_angles, n_time1, n_time2)

            size_key = f"chi_squared_{n_angles}x{n_time1}x{n_time2}"

            # Benchmark original
            start_time = time.perf_counter()
            for _ in range(100):
                result_orig = MathematicalOptimizer.chi_squared_original(
                    c2_exp, c2_theo
                )
            orig_time = time.perf_counter() - start_time

            # Benchmark vectorized
            start_time = time.perf_counter()
            for _ in range(100):
                result_vec = MathematicalOptimizer.chi_squared_vectorized(
                    c2_exp, c2_theo
                )
            vec_time = time.perf_counter() - start_time

            # Benchmark JIT optimized
            start_time = time.perf_counter()
            for _ in range(100):
                result_jit = MathematicalOptimizer.chi_squared_optimized(
                    c2_exp, c2_theo
                )
            jit_time = time.perf_counter() - start_time

            # Check accuracy
            accuracy_vec = abs(result_orig - result_vec) < 1e-10
            accuracy_jit = abs(result_orig - result_jit) < 1e-10

            # Store results
            results[f"{size_key}_vectorized"] = OptimizationResult(
                original_time=orig_time,
                optimized_time=vec_time,
                speedup_factor=orig_time / vec_time if vec_time > 0 else float("inf"),
                memory_reduction=0.0,  # Same memory usage
                accuracy_preserved=accuracy_vec,
                optimization_method="vectorization",
            )

            results[f"{size_key}_jit"] = OptimizationResult(
                original_time=orig_time,
                optimized_time=jit_time,
                speedup_factor=orig_time / jit_time if jit_time > 0 else float("inf"),
                memory_reduction=0.0,
                accuracy_preserved=accuracy_jit,
                optimization_method="numba_jit",
            )

        return results

    def benchmark_correlation_optimizations(self) -> dict[str, OptimizationResult]:
        """Benchmark correlation function optimizations."""
        results = {}

        # Test different data sizes
        data_sizes = [(3, 50), (5, 100), (8, 200)]

        for n_angles, n_time in data_sizes:
            # Generate test data
            data = np.random.rand(n_angles, n_time)
            angles = np.linspace(0, 180, n_angles)

            size_key = f"correlation_{n_angles}x{n_time}"

            # Benchmark original
            start_time = time.perf_counter()
            result_orig = MathematicalOptimizer.correlation_function_original(
                data, angles
            )
            orig_time = time.perf_counter() - start_time

            # Benchmark optimized
            start_time = time.perf_counter()
            result_opt = MathematicalOptimizer.correlation_function_optimized(
                data, angles
            )
            opt_time = time.perf_counter() - start_time

            # Check accuracy
            accuracy = np.allclose(result_orig, result_opt, rtol=1e-10)

            results[size_key] = OptimizationResult(
                original_time=orig_time,
                optimized_time=opt_time,
                speedup_factor=orig_time / opt_time if opt_time > 0 else float("inf"),
                memory_reduction=0.0,
                accuracy_preserved=accuracy,
                optimization_method="vectorization",
            )

        return results

    def benchmark_statistical_optimizations(self) -> dict[str, OptimizationResult]:
        """Benchmark statistical computation optimizations."""
        results = {}

        # Test different data sizes
        data_sizes = [1000, 10000, 100000]

        for size in data_sizes:
            data = np.random.rand(size)
            size_key = f"statistics_{size}"

            # Benchmark original
            start_time = time.perf_counter()
            for _ in range(100):
                result_orig = MathematicalOptimizer.statistical_moments_original(data)
            orig_time = time.perf_counter() - start_time

            # Benchmark optimized
            start_time = time.perf_counter()
            for _ in range(100):
                result_opt = MathematicalOptimizer.statistical_moments_optimized(data)
            opt_time = time.perf_counter() - start_time

            # Check accuracy
            accuracy = (
                abs(result_orig["mean"] - result_opt["mean"]) < 1e-10
                and abs(result_orig["variance"] - result_opt["variance"]) < 1e-10
            )

            results[size_key] = OptimizationResult(
                original_time=orig_time,
                optimized_time=opt_time,
                speedup_factor=orig_time / opt_time if opt_time > 0 else float("inf"),
                memory_reduction=0.0,
                accuracy_preserved=accuracy,
                optimization_method="single_pass_algorithm",
            )

        return results

    def run_comprehensive_optimization_benchmark(self) -> dict[str, Any]:
        """Run comprehensive optimization benchmarks."""
        print("Running Comprehensive Algorithm Optimization Benchmarks")
        print("=" * 60)

        all_results = {}

        # Chi-squared optimizations
        print("Benchmarking chi-squared optimizations...")
        chi2_results = self.benchmark_chi_squared_optimizations()
        all_results.update(chi2_results)

        # Correlation optimizations
        print("Benchmarking correlation function optimizations...")
        corr_results = self.benchmark_correlation_optimizations()
        all_results.update(corr_results)

        # Statistical optimizations
        print("Benchmarking statistical computation optimizations...")
        stats_results = self.benchmark_statistical_optimizations()
        all_results.update(stats_results)

        return all_results


def test_numerical_stability():
    """Test numerical stability optimizations."""
    print("\nTesting Numerical Stability Optimizations...")

    # Test with ill-conditioned data
    size = 100
    c2_exp = np.random.rand(size, size) * 1e10  # Large values
    c2_theo = c2_exp + np.random.rand(size, size) * 1e-10  # Small differences

    # Original calculation (may have numerical issues)
    try:
        result_orig = MathematicalOptimizer.chi_squared_original(c2_exp, c2_theo)
        orig_stable = np.isfinite(result_orig)
    except (OverflowError, RuntimeError):
        orig_stable = False
        result_orig = float("inf")

    # Stable calculation
    result_stable = NumericalStabilityOptimizer.stable_chi_squared(c2_exp, c2_theo)
    stable_stable = np.isfinite(result_stable)

    print(f"  Original calculation stable: {orig_stable}")
    print(f"  Optimized calculation stable: {stable_stable}")
    print(
        f"  Stability improvement: {'✓' if stable_stable and not orig_stable else '='}"
    )

    # Test eigenvalue stability
    # Create ill-conditioned matrix
    A = np.random.rand(50, 50)
    A = A @ A.T  # Make positive definite
    A += 1e-15 * np.eye(50)  # Make slightly ill-conditioned

    try:
        eig_orig = np.linalg.eigvals(A)
        orig_eig_stable = np.all(np.isfinite(eig_orig))
    except np.linalg.LinAlgError:
        orig_eig_stable = False

    eig_robust = NumericalStabilityOptimizer.robust_eigenvalue_computation(A)
    robust_eig_stable = np.all(np.isfinite(eig_robust))

    print(f"  Original eigenvalues stable: {orig_eig_stable}")
    print(f"  Robust eigenvalues stable: {robust_eig_stable}")


def generate_optimization_report(results: dict[str, OptimizationResult]) -> str:
    """Generate comprehensive optimization report."""
    report_lines = []
    report_lines.append("ALGORITHM OPTIMIZATION REPORT - TASK 4.3")
    report_lines.append("=" * 60)

    # Summary statistics
    speedups = [
        r.speedup_factor for r in results.values() if r.speedup_factor != float("inf")
    ]
    accuracy_preserved = sum(1 for r in results.values() if r.accuracy_preserved)

    report_lines.append("\nSUMMARY STATISTICS:")
    report_lines.append(f"  Total optimizations: {len(results)}")
    report_lines.append(f"  Accuracy preserved: {accuracy_preserved}/{len(results)}")
    report_lines.append(f"  Average speedup: {np.mean(speedups):.2f}x")
    report_lines.append(f"  Maximum speedup: {np.max(speedups):.2f}x")
    report_lines.append(f"  Minimum speedup: {np.min(speedups):.2f}x")

    # Detailed results
    report_lines.append("\nDETAILED RESULTS:")
    report_lines.append(
        f"{'Operation':<30} {'Speedup':<10} {'Method':<20} {'Accurate':<10}"
    )
    report_lines.append("-" * 70)

    for name, result in results.items():
        speedup_str = (
            f"{result.speedup_factor:.2f}x"
            if result.speedup_factor != float("inf")
            else "∞"
        )
        accurate_str = "✓" if result.accuracy_preserved else "✗"

        report_lines.append(
            f"{name:<30} {speedup_str:<10} {result.optimization_method:<20} {accurate_str:<10}"
        )

    return "\n".join(report_lines)


def run_algorithm_optimization_suite():
    """Main function to run algorithm optimization suite."""
    print("Starting Algorithm Optimization Suite - Task 4.3")
    print("=" * 60)

    # Run benchmarks
    benchmarker = AlgorithmBenchmarker()
    results = benchmarker.run_comprehensive_optimization_benchmark()

    # Test numerical stability
    test_numerical_stability()

    # Generate report
    report = generate_optimization_report(results)
    print("\n" + report)

    # Save results
    results_dir = Path("optimization_results")
    results_dir.mkdir(exist_ok=True)

    # Convert results to JSON-serializable format
    json_results = {}
    for name, result in results.items():
        json_results[name] = {
            "original_time": float(result.original_time),
            "optimized_time": float(result.optimized_time),
            "speedup_factor": (
                float(result.speedup_factor)
                if result.speedup_factor != float("inf")
                else None
            ),
            "memory_reduction": float(result.memory_reduction),
            "accuracy_preserved": bool(result.accuracy_preserved),
            "optimization_method": str(result.optimization_method),
        }

    # Save JSON results
    json_file = results_dir / "task_4_3_optimization_results.json"
    with open(json_file, "w") as f:
        json.dump(
            {
                "optimization_results": json_results,
                "summary": {
                    "total_optimizations": len(results),
                    "accuracy_preserved": sum(
                        1 for r in results.values() if r.accuracy_preserved
                    ),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            },
            f,
            indent=2,
        )

    # Save text report
    report_file = results_dir / "task_4_3_optimization_report.txt"
    with open(report_file, "w") as f:
        f.write(report)

    print(f"\n📄 Results saved to: {json_file}")
    print(f"📄 Report saved to: {report_file}")

    speedups = [
        r.speedup_factor for r in results.values() if r.speedup_factor != float("inf")
    ]
    avg_speedup = np.mean(speedups) if speedups else 0

    print("\n✅ Task 4.3 Algorithm Optimization Complete!")
    print(f"🚀 Average speedup achieved: {avg_speedup:.2f}x")
    print(f"🎯 {len(results)} optimizations implemented")
    print(
        f"✓ Numerical accuracy preserved in {sum(1 for r in results.values() if r.accuracy_preserved)}/{len(results)} cases"
    )

    return json_results


if __name__ == "__main__":
    run_algorithm_optimization_suite()
