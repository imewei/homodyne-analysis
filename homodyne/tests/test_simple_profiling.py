"""
Simplified Performance Profiling System
======================================

Lightweight performance profiling system for Task 4.2.
Provides essential profiling and monitoring capabilities.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import cProfile
import functools
import io
import json
import pstats
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil

warnings.filterwarnings("ignore")


@dataclass
class SimpleProfileData:
    """Simple profiling data."""

    function_name: str
    execution_time: float
    memory_usage: float
    call_count: int


class SimpleProfiler:
    """Lightweight function profiler."""

    def __init__(self):
        self.profiles = {}

    def profile_function(self, name: str | None = None):
        """Simple profiling decorator."""

        def decorator(func):
            profile_name = name or func.__name__

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Memory before
                mem_before = psutil.Process().memory_info().rss / 1024 / 1024

                # Time execution
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()

                # Memory after
                mem_after = psutil.Process().memory_info().rss / 1024 / 1024

                # Store profile data
                execution_time = end_time - start_time
                memory_delta = mem_after - mem_before

                if profile_name not in self.profiles:
                    self.profiles[profile_name] = SimpleProfileData(
                        function_name=profile_name,
                        execution_time=execution_time,
                        memory_usage=memory_delta,
                        call_count=1,
                    )
                else:
                    # Update existing profile
                    profile = self.profiles[profile_name]
                    profile.call_count += 1
                    profile.execution_time += execution_time
                    profile.memory_usage = max(profile.memory_usage, memory_delta)

                return result

            return wrapper

        return decorator

    def get_profile_summary(self) -> dict[str, Any]:
        """Get profile summary."""
        if not self.profiles:
            return {}

        profiles_dict = {}
        for name, profile in self.profiles.items():
            profiles_dict[name] = {
                "function_name": profile.function_name,
                "total_time": float(profile.execution_time),
                "avg_time": float(profile.execution_time / profile.call_count),
                "memory_usage": float(profile.memory_usage),
                "call_count": int(profile.call_count),
            }

        return profiles_dict


class PerformanceMonitor:
    """Simple performance monitoring."""

    def __init__(self):
        self.profiler = SimpleProfiler()
        self.system_metrics = []

    def capture_system_metrics(self):
        """Capture current system metrics."""
        try:
            process = psutil.Process()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = process.memory_info()

            metrics = {
                "timestamp": time.time(),
                "cpu_percent": float(cpu_percent),
                "memory_mb": float(memory_info.rss / 1024 / 1024),
                "memory_percent": float(psutil.virtual_memory().percent),
            }

            self.system_metrics.append(metrics)

            # Keep only recent metrics
            if len(self.system_metrics) > 100:
                self.system_metrics = self.system_metrics[-100:]

        except Exception:
            pass  # Ignore monitoring errors

    def get_monitoring_summary(self) -> dict[str, Any]:
        """Get monitoring summary."""
        if not self.system_metrics:
            return {}

        cpu_values = [m["cpu_percent"] for m in self.system_metrics]
        memory_values = [m["memory_mb"] for m in self.system_metrics]

        return {
            "metrics_collected": len(self.system_metrics),
            "avg_cpu_percent": float(np.mean(cpu_values)),
            "max_cpu_percent": float(np.max(cpu_values)),
            "avg_memory_mb": float(np.mean(memory_values)),
            "max_memory_mb": float(np.max(memory_values)),
        }


def test_profiling_system():
    """Test the simple profiling system."""
    print("Testing Simple Profiling System - Task 4.2")
    print("=" * 50)

    monitor = PerformanceMonitor()

    # Test functions with profiling
    @monitor.profiler.profile_function("matrix_operations")
    def matrix_test():
        A = np.random.rand(100, 100)
        B = np.random.rand(100, 100)
        return A @ B

    @monitor.profiler.profile_function("array_operations")
    def array_test():
        data = np.random.rand(5000)
        return np.sort(data)

    @monitor.profiler.profile_function("statistical_operations")
    def stats_test():
        data = np.random.rand(10000)
        return {"mean": np.mean(data), "std": np.std(data), "median": np.median(data)}

    # Run tests
    print("Running profiled operations...")

    # Capture initial system state
    monitor.capture_system_metrics()

    # Run matrix operations
    for i in range(5):
        matrix_test()
        monitor.capture_system_metrics()

    # Run array operations
    for i in range(10):
        array_test()
        monitor.capture_system_metrics()

    # Run statistical operations
    for i in range(3):
        stats_test()
        monitor.capture_system_metrics()

    # Get results
    profile_summary = monitor.profiler.get_profile_summary()
    monitoring_summary = monitor.get_monitoring_summary()

    # Display results
    print("\nPROFILING RESULTS:")
    print(
        f"{'Function':<25} {'Calls':<8} {'Total (ms)':<12} {'Avg (ms)':<12} {'Memory (MB)':<12}"
    )
    print("-" * 69)

    for name, data in profile_summary.items():
        total_ms = data["total_time"] * 1000
        avg_ms = data["avg_time"] * 1000
        print(
            f"{name:<25} {data['call_count']:<8} {total_ms:<12.2f} {avg_ms:<12.4f} {data['memory_usage']:<12.2f}"
        )

    print("\nSYSTEM MONITORING:")
    print(f"  Metrics collected: {monitoring_summary.get('metrics_collected', 0)}")
    print(f"  Average CPU: {monitoring_summary.get('avg_cpu_percent', 0):.1f}%")
    print(f"  Peak CPU: {monitoring_summary.get('max_cpu_percent', 0):.1f}%")
    print(f"  Average Memory: {monitoring_summary.get('avg_memory_mb', 0):.1f} MB")
    print(f"  Peak Memory: {monitoring_summary.get('max_memory_mb', 0):.1f} MB")

    # Save results
    results_dir = Path("performance_results")
    results_dir.mkdir(exist_ok=True)

    # Combine all data
    combined_results = {
        "profiling_summary": profile_summary,
        "monitoring_summary": monitoring_summary,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": "4.2_profiling_infrastructure",
    }

    # Save to JSON
    results_file = results_dir / "task_4_2_profiling_results.json"
    with open(results_file, "w") as f:
        json.dump(combined_results, f, indent=2)

    # Generate text report
    report_lines = []
    report_lines.append("TASK 4.2 PROFILING INFRASTRUCTURE REPORT")
    report_lines.append("=" * 50)
    report_lines.append(f"Generated: {combined_results['timestamp']}")
    report_lines.append("")

    report_lines.append("PROFILING RESULTS:")
    for name, data in profile_summary.items():
        report_lines.append(f"  {name}:")
        report_lines.append(f"    Calls: {data['call_count']}")
        report_lines.append(f"    Total Time: {data['total_time'] * 1000:.2f} ms")
        report_lines.append(f"    Average Time: {data['avg_time'] * 1000:.4f} ms")
        report_lines.append(f"    Memory Usage: {data['memory_usage']:.2f} MB")
        report_lines.append("")

    report_lines.append("SYSTEM MONITORING:")
    for key, value in monitoring_summary.items():
        report_lines.append(f"  {key}: {value}")

    # Save text report
    report_file = results_dir / "task_4_2_profiling_report.txt"
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\n📄 Results saved to: {results_file}")
    print(f"📄 Report saved to: {report_file}")

    # Test advanced profiling features
    print("\nTesting advanced profiling features...")

    # cProfile test
    pr = cProfile.Profile()
    pr.enable()

    # Run some operations to profile
    data = np.random.rand(1000, 100)
    np.linalg.svd(data)

    pr.disable()

    # Get cProfile stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(10)  # Top 10 functions

    profile_output = s.getvalue()

    # Save cProfile output
    cprofile_file = results_dir / "task_4_2_cprofile_output.txt"
    with open(cprofile_file, "w") as f:
        f.write("CPROFILE OUTPUT - SVD Operation\n")
        f.write("=" * 40 + "\n")
        f.write(profile_output)

    print(f"📄 cProfile output saved to: {cprofile_file}")

    print("\n✅ Task 4.2 Profiling Infrastructure Complete!")
    print("🎯 Advanced profiling and monitoring system established")
    print(f"📊 {len(profile_summary)} functions profiled")
    print(
        f"📈 {monitoring_summary.get('metrics_collected', 0)} system metrics collected"
    )

    return combined_results


if __name__ == "__main__":
    test_profiling_system()
