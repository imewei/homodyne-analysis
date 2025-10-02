"""
Quick Performance Baseline System
=================================

Fast performance baseline measurement for Task 4.1.
Optimized for quick execution while capturing essential metrics.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import gc
import json
import time
import warnings
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil

warnings.filterwarnings("ignore")


@dataclass
class QuickMetrics:
    """Quick performance metrics."""

    name: str
    time_seconds: float
    memory_mb: float
    data_size: int
    status: str


class QuickBenchmarker:
    """Fast benchmarking system."""

    def __init__(self):
        self.results = {}

    def measure_quick(self, name: str, func, *args, iterations: int = 5):
        """Quick measurement with minimal overhead."""
        times = []
        memory_start = psutil.Process().memory_info().rss / 1024 / 1024

        # Warmup
        try:
            func(*args)
        except Exception:
            pass

        # Measure
        for _ in range(iterations):
            gc.collect()
            start = time.perf_counter()
            try:
                func(*args)
                end = time.perf_counter()
                times.append(end - start)
            except Exception as e:
                self.results[name] = QuickMetrics(
                    name=name,
                    time_seconds=float("inf"),
                    memory_mb=0.0,
                    data_size=len(args),
                    status=f"FAILED: {str(e)[:50]}",
                )
                return

        memory_end = psutil.Process().memory_info().rss / 1024 / 1024
        avg_time = np.mean(times) if times else float("inf")

        self.results[name] = QuickMetrics(
            name=name,
            time_seconds=avg_time,
            memory_mb=memory_end - memory_start,
            data_size=len(args),
            status="SUCCESS",
        )


def run_quick_baseline():
    """Run quick performance baseline."""
    print("Quick Performance Baseline Creation - Task 4.1")
    print("=" * 50)

    benchmarker = QuickBenchmarker()

    # Core mathematical operations
    print("Testing core operations...")

    # Small chi-squared calculation
    c2_exp_small = np.random.rand(3, 10, 10)
    c2_theo_small = np.random.rand(3, 10, 10)

    benchmarker.measure_quick(
        "chi_squared_small", lambda: np.sum((c2_exp_small - c2_theo_small) ** 2)
    )

    # Medium chi-squared calculation
    c2_exp_med = np.random.rand(5, 20, 20)
    c2_theo_med = np.random.rand(5, 20, 20)

    benchmarker.measure_quick(
        "chi_squared_medium", lambda: np.sum((c2_exp_med - c2_theo_med) ** 2)
    )

    # Matrix operations
    A_small = np.random.rand(50, 50)
    B_small = np.random.rand(50, 50)

    benchmarker.measure_quick("matrix_multiply_50x50", lambda: A_small @ B_small)

    A_med = np.random.rand(100, 100)

    benchmarker.measure_quick("eigenvalues_100x100", lambda: np.linalg.eigvals(A_med))

    # Array operations
    data_1k = np.random.rand(1000)
    data_10k = np.random.rand(10000)

    benchmarker.measure_quick("array_mean_1k", lambda: np.mean(data_1k))

    benchmarker.measure_quick("array_std_10k", lambda: np.std(data_10k))

    # Statistical operations
    benchmarker.measure_quick("array_sort_1k", lambda: np.sort(data_1k))

    benchmarker.measure_quick("fft_1k", lambda: np.fft.fft(data_1k))

    # Print results
    print("\nBASELINE RESULTS:")
    print(f"{'Operation':<25} {'Time (ms)':<12} {'Memory (MB)':<12} {'Status':<15}")
    print("-" * 64)

    total_time = 0
    success_count = 0

    for name, metrics in benchmarker.results.items():
        time_ms = (
            metrics.time_seconds * 1000
            if metrics.time_seconds != float("inf")
            else float("inf")
        )
        time_str = f"{time_ms:.2f}" if time_ms != float("inf") else "TIMEOUT"

        print(
            f"{name:<25} {time_str:<12} {metrics.memory_mb:<12.2f} {metrics.status:<15}"
        )

        if metrics.status == "SUCCESS":
            total_time += metrics.time_seconds
            success_count += 1

    print("-" * 64)
    print(f"Total successful operations: {success_count}/{len(benchmarker.results)}")
    print(f"Total execution time: {total_time * 1000:.2f} ms")

    # Save baseline
    baseline_data = {
        "baseline_id": "task_4_1_quick_baseline",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "platform": __import__("platform").platform(),
        },
        "metrics": {
            name: asdict(metrics) for name, metrics in benchmarker.results.items()
        },
        "summary": {
            "total_operations": len(benchmarker.results),
            "successful_operations": success_count,
            "total_time_ms": total_time * 1000,
            "avg_time_ms": (
                (total_time * 1000) / success_count if success_count > 0 else 0
            ),
        },
    }

    # Create baseline directory
    baseline_dir = Path("performance_baselines")
    baseline_dir.mkdir(exist_ok=True)

    # Save baseline
    baseline_file = baseline_dir / "quick_baseline_task_4_1.json"
    with open(baseline_file, "w") as f:
        json.dump(baseline_data, f, indent=2)

    print(f"\n📄 Baseline saved to: {baseline_file}")

    # Test import performance
    print("\nTesting import performance...")

    import_times = {}

    def test_import(module_name, import_stmt):
        start = time.perf_counter()
        try:
            exec(import_stmt)
            end = time.perf_counter()
            import_times[module_name] = (end - start) * 1000
            return "SUCCESS"
        except Exception as e:
            import_times[module_name] = float("inf")
            return f"FAILED: {str(e)[:30]}"

    print(f"{'Module':<20} {'Time (ms)':<12} {'Status':<20}")
    print("-" * 52)

    # Core imports
    imports_to_test = [
        ("numpy", "import numpy"),
        ("scipy", "import scipy"),
        ("json", "import json"),
        ("pathlib", "from pathlib import Path"),
    ]

    for module, stmt in imports_to_test:
        status = test_import(module, stmt)
        time_str = (
            f"{import_times[module]:.2f}"
            if import_times[module] != float("inf")
            else "TIMEOUT"
        )
        print(f"{module:<20} {time_str:<12} {status:<20}")

    baseline_data["import_times"] = import_times

    # Update saved baseline
    with open(baseline_file, "w") as f:
        json.dump(baseline_data, f, indent=2)

    print("\n✅ Task 4.1 Baseline Creation Complete!")
    print("🎯 Baseline ID: task_4_1_quick_baseline")
    print(f"📊 {success_count} operations benchmarked successfully")

    return baseline_data


if __name__ == "__main__":
    run_quick_baseline()
