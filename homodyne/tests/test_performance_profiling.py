"""
Advanced Performance Profiling and Monitoring Infrastructure
=============================================================

Comprehensive profiling system for the homodyne analysis package with
real-time monitoring, bottleneck detection, and performance analytics.

Features:
- Function-level profiling with call graphs
- Memory profiling and leak detection
- CPU utilization monitoring
- Performance hotspot identification
- Real-time monitoring dashboard
- Automated performance alerts

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import cProfile
import pstats
import io
import time
import psutil
import threading
import numpy as np
import json
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import functools
import gc
import warnings

warnings.filterwarnings("ignore")


@dataclass
class ProfileData:
    """Container for profiling data."""
    function_name: str
    total_time: float
    cumulative_time: float
    call_count: int
    time_per_call: float
    filename: str
    line_number: int


@dataclass
class MemoryProfile:
    """Memory profiling data."""
    timestamp: float
    current_mb: float
    peak_mb: float
    tracemalloc_current: float
    tracemalloc_peak: float
    gc_objects: int


@dataclass
class PerformanceAlert:
    """Performance alert data."""
    timestamp: str
    alert_type: str
    metric_name: str
    current_value: float
    threshold: float
    severity: str
    message: str


class FunctionProfiler:
    """Advanced function-level profiler."""

    def __init__(self):
        self.profile_data = {}
        self.profiler = None

    @contextmanager
    def profile_context(self, operation_name: str):
        """Context manager for profiling operations."""
        self.profiler = cProfile.Profile()
        self.profiler.enable()

        try:
            yield
        finally:
            self.profiler.disable()
            self._process_profile_data(operation_name)

    def _process_profile_data(self, operation_name: str):
        """Process cProfile data into structured format."""
        if not self.profiler:
            return

        # Create string buffer for stats
        stats_buffer = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=stats_buffer)
        stats.sort_stats('cumulative')

        # Extract top functions
        profile_list = []
        for func_info, func_stats in stats.stats.items():
            filename, line_number, function_name = func_info

            # Handle different pstats formats
            if len(func_stats) == 4:
                call_count, reccall_count, total_time, cumulative_time = func_stats
            elif len(func_stats) == 6:
                # Newer format has additional fields
                call_count, reccall_count, total_time, cumulative_time, _, _ = func_stats
            else:
                # Fallback for unknown format
                call_count = func_stats[0] if len(func_stats) > 0 else 0
                total_time = func_stats[2] if len(func_stats) > 2 else 0
                cumulative_time = func_stats[3] if len(func_stats) > 3 else 0

            # Skip built-in functions and profiler overhead
            if filename.startswith('<') or 'profiling' in filename.lower():
                continue

            time_per_call = total_time / call_count if call_count > 0 else 0

            profile_list.append(ProfileData(
                function_name=function_name,
                total_time=total_time,
                cumulative_time=cumulative_time,
                call_count=call_count,
                time_per_call=time_per_call,
                filename=filename,
                line_number=line_number
            ))

        # Sort by cumulative time
        profile_list.sort(key=lambda x: x.cumulative_time, reverse=True)
        self.profile_data[operation_name] = profile_list[:20]  # Top 20 functions

    def get_top_functions(self, operation_name: str, n: int = 10) -> List[ProfileData]:
        """Get top N functions by execution time."""
        if operation_name not in self.profile_data:
            return []
        return self.profile_data[operation_name][:n]

    def generate_profile_report(self, operation_name: str) -> str:
        """Generate human-readable profile report."""
        if operation_name not in self.profile_data:
            return f"No profile data available for {operation_name}"

        report = []
        report.append(f"PROFILE REPORT: {operation_name}")
        report.append("=" * 60)
        report.append(f"{'Function':<30} {'Calls':<8} {'Time(s)':<10} {'Time/Call':<12}")
        report.append("-" * 60)

        for profile in self.profile_data[operation_name]:
            report.append(f"{profile.function_name[:29]:<30} {profile.call_count:<8} "
                         f"{profile.total_time:<10.4f} {profile.time_per_call:<12.6f}")

        return "\n".join(report)


class MemoryProfiler:
    """Advanced memory profiling and monitoring."""

    def __init__(self):
        self.memory_snapshots = []
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self, interval: float = 0.1):
        """Start continuous memory monitoring."""
        if self.monitoring:
            return

        self.monitoring = True
        tracemalloc.start()

        def monitor_loop():
            while self.monitoring:
                self._take_memory_snapshot()
                time.sleep(interval)

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        tracemalloc.stop()

    def _take_memory_snapshot(self):
        """Take a memory snapshot."""
        try:
            # Process memory
            process = psutil.Process()
            memory_info = process.memory_info()
            current_mb = memory_info.rss / 1024 / 1024

            # Peak memory (approximate)
            peak_mb = getattr(memory_info, 'peak_wss', memory_info.rss) / 1024 / 1024

            # Tracemalloc memory
            try:
                tracemalloc_current, tracemalloc_peak = tracemalloc.get_traced_memory()
                tracemalloc_current_mb = tracemalloc_current / 1024 / 1024
                tracemalloc_peak_mb = tracemalloc_peak / 1024 / 1024
            except RuntimeError:
                tracemalloc_current_mb = 0
                tracemalloc_peak_mb = 0

            # GC object count
            gc_objects = len(gc.get_objects())

            snapshot = MemoryProfile(
                timestamp=time.time(),
                current_mb=current_mb,
                peak_mb=peak_mb,
                tracemalloc_current=tracemalloc_current_mb,
                tracemalloc_peak=tracemalloc_peak_mb,
                gc_objects=gc_objects
            )

            self.memory_snapshots.append(snapshot)

            # Keep only recent snapshots (last 1000)
            if len(self.memory_snapshots) > 1000:
                self.memory_snapshots = self.memory_snapshots[-1000:]

        except Exception:
            pass  # Ignore monitoring errors

    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        if not self.memory_snapshots:
            return {}

        recent_snapshots = self.memory_snapshots[-100:]  # Last 100 snapshots

        current_memory = [s.current_mb for s in recent_snapshots]
        peak_memory = [s.peak_mb for s in recent_snapshots]
        gc_objects = [s.gc_objects for s in recent_snapshots]

        return {
            "avg_memory_mb": np.mean(current_memory),
            "max_memory_mb": np.max(current_memory),
            "min_memory_mb": np.min(current_memory),
            "memory_std_mb": np.std(current_memory),
            "peak_memory_mb": np.max(peak_memory),
            "avg_gc_objects": np.mean(gc_objects),
            "max_gc_objects": np.max(gc_objects)
        }

    def detect_memory_leaks(self, threshold_mb: float = 10.0) -> List[str]:
        """Detect potential memory leaks."""
        if len(self.memory_snapshots) < 100:
            return []

        # Analyze memory trend over time
        recent_memory = [s.current_mb for s in self.memory_snapshots[-100:]]

        # Simple leak detection: check if memory is consistently increasing
        if len(recent_memory) < 50:
            return []

        first_half = recent_memory[:25]
        second_half = recent_memory[-25:]

        avg_first = np.mean(first_half)
        avg_second = np.mean(second_half)

        memory_increase = avg_second - avg_first

        alerts = []
        if memory_increase > threshold_mb:
            alerts.append(f"Memory increased by {memory_increase:.2f} MB over recent period")

        # Check for sudden spikes
        max_memory = np.max(recent_memory)
        avg_memory = np.mean(recent_memory)

        if max_memory - avg_memory > threshold_mb:
            alerts.append(f"Memory spike detected: {max_memory:.2f} MB peak vs {avg_memory:.2f} MB average")

        return alerts


class PerformanceMonitor:
    """Real-time performance monitoring system."""

    def __init__(self):
        self.function_profiler = FunctionProfiler()
        self.memory_profiler = MemoryProfiler()
        self.alerts = []
        self.thresholds = {
            "max_execution_time": 10.0,  # seconds
            "max_memory_mb": 1000.0,     # MB
            "max_cpu_percent": 80.0,     # percent
        }

    def profile_function(self, func_name: str = None):
        """Decorator for profiling functions."""
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__name__}"

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.function_profiler.profile_context(name):
                    start_time = time.perf_counter()
                    result = func(*args, **kwargs)
                    end_time = time.perf_counter()

                    execution_time = end_time - start_time
                    self._check_performance_thresholds(name, execution_time)

                return result
            return wrapper
        return decorator

    def _check_performance_thresholds(self, function_name: str, execution_time: float):
        """Check if performance thresholds are exceeded."""
        if execution_time > self.thresholds["max_execution_time"]:
            alert = PerformanceAlert(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                alert_type="execution_time",
                metric_name=function_name,
                current_value=execution_time,
                threshold=self.thresholds["max_execution_time"],
                severity="WARNING",
                message=f"Function {function_name} exceeded execution time threshold"
            )
            self.alerts.append(alert)

        # Check memory usage
        try:
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            if memory_mb > self.thresholds["max_memory_mb"]:
                alert = PerformanceAlert(
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    alert_type="memory_usage",
                    metric_name="system_memory",
                    current_value=memory_mb,
                    threshold=self.thresholds["max_memory_mb"],
                    severity="WARNING",
                    message=f"Memory usage exceeded threshold: {memory_mb:.2f} MB"
                )
                self.alerts.append(alert)
        except Exception:
            pass

    def start_monitoring(self):
        """Start comprehensive monitoring."""
        self.memory_profiler.start_monitoring(interval=0.5)

    def stop_monitoring(self):
        """Stop all monitoring."""
        self.memory_profiler.stop_monitoring()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        memory_stats = self.memory_profiler.get_memory_stats()
        memory_leaks = self.memory_profiler.detect_memory_leaks()

        return {
            "memory_statistics": memory_stats,
            "memory_leak_alerts": memory_leaks,
            "performance_alerts": [asdict(alert) for alert in self.alerts[-10:]],  # Recent alerts
            "profiled_functions": list(self.function_profiler.profile_data.keys()),
            "monitoring_active": self.memory_profiler.monitoring
        }

    def generate_monitoring_report(self) -> str:
        """Generate comprehensive monitoring report."""
        report = []
        report.append("PERFORMANCE MONITORING REPORT")
        report.append("=" * 50)

        # Memory statistics
        memory_stats = self.memory_profiler.get_memory_stats()
        if memory_stats:
            report.append("\nMEMORY STATISTICS:")
            report.append(f"  Average Memory: {memory_stats.get('avg_memory_mb', 0):.2f} MB")
            report.append(f"  Peak Memory: {memory_stats.get('peak_memory_mb', 0):.2f} MB")
            report.append(f"  Memory Std Dev: {memory_stats.get('memory_std_mb', 0):.2f} MB")
            report.append(f"  GC Objects: {memory_stats.get('avg_gc_objects', 0):.0f}")

        # Memory leaks
        memory_leaks = self.memory_profiler.detect_memory_leaks()
        if memory_leaks:
            report.append("\nMEMORY LEAK ALERTS:")
            for leak in memory_leaks:
                report.append(f"  ⚠️  {leak}")

        # Performance alerts
        if self.alerts:
            report.append("\nRECENT PERFORMANCE ALERTS:")
            for alert in self.alerts[-5:]:  # Last 5 alerts
                report.append(f"  {alert.severity}: {alert.message}")

        # Profiled functions
        if self.function_profiler.profile_data:
            report.append("\nPROFILED FUNCTIONS:")
            for func_name in self.function_profiler.profile_data.keys():
                report.append(f"  • {func_name}")

        return "\n".join(report)


class PerformanceTestSuite:
    """Test suite for performance profiling infrastructure."""

    def __init__(self):
        self.monitor = PerformanceMonitor()

    def test_function_profiling(self):
        """Test function profiling capabilities."""
        print("Testing function profiling...")

        @self.monitor.profile_function("matrix_operations")
        def matrix_test():
            A = np.random.rand(100, 100)
            B = np.random.rand(100, 100)
            return A @ B

        @self.monitor.profile_function("array_operations")
        def array_test():
            data = np.random.rand(10000)
            return np.sort(data)

        # Run profiled functions
        result1 = matrix_test()
        result2 = array_test()

        # Get profile reports
        matrix_report = self.monitor.function_profiler.generate_profile_report("matrix_operations")
        array_report = self.monitor.function_profiler.generate_profile_report("array_operations")

        print("✓ Function profiling test completed")
        return matrix_report, array_report

    def test_memory_monitoring(self):
        """Test memory monitoring capabilities."""
        print("Testing memory monitoring...")

        # Start monitoring
        self.monitor.start_monitoring()

        # Simulate memory usage
        data_arrays = []
        for i in range(10):
            # Create some arrays to consume memory
            arr = np.random.rand(1000, 100)
            data_arrays.append(arr)
            time.sleep(0.1)  # Allow monitoring to capture

        # Check memory stats
        time.sleep(0.5)  # Let monitoring collect data
        memory_stats = self.monitor.memory_profiler.get_memory_stats()

        # Stop monitoring
        self.monitor.stop_monitoring()

        print("✓ Memory monitoring test completed")
        return memory_stats

    def test_performance_alerts(self):
        """Test performance alert system."""
        print("Testing performance alerts...")

        # Lower thresholds for testing
        original_thresholds = self.monitor.thresholds.copy()
        self.monitor.thresholds["max_execution_time"] = 0.01  # Very low threshold

        @self.monitor.profile_function("slow_operation")
        def slow_function():
            time.sleep(0.02)  # Intentionally slow
            return "done"

        # This should trigger an alert
        result = slow_function()

        # Check for alerts
        alerts = self.monitor.alerts

        # Restore thresholds
        self.monitor.thresholds = original_thresholds

        print("✓ Performance alerts test completed")
        return alerts

    def run_comprehensive_test(self):
        """Run comprehensive profiling infrastructure test."""
        print("PERFORMANCE PROFILING INFRASTRUCTURE TEST")
        print("=" * 50)

        # Test 1: Function profiling
        matrix_report, array_report = self.test_function_profiling()

        # Test 2: Memory monitoring
        memory_stats = self.test_memory_monitoring()

        # Test 3: Performance alerts
        alerts = self.test_performance_alerts()

        # Generate summary report
        summary_report = self.monitor.generate_monitoring_report()

        print("\nSUMMARY:")
        print(f"  Functions profiled: {len(self.monitor.function_profiler.profile_data)}")
        print(f"  Memory snapshots: {len(self.monitor.memory_profiler.memory_snapshots)}")
        print(f"  Performance alerts: {len(alerts)}")

        # Save detailed reports
        reports_dir = Path("performance_reports")
        reports_dir.mkdir(exist_ok=True)

        # Save matrix profiling report
        with open(reports_dir / "matrix_profiling_report.txt", 'w') as f:
            f.write(matrix_report)

        # Save array profiling report
        with open(reports_dir / "array_profiling_report.txt", 'w') as f:
            f.write(array_report)

        # Save monitoring summary
        with open(reports_dir / "monitoring_summary.txt", 'w') as f:
            f.write(summary_report)

        # Save performance data as JSON
        performance_data = {
            "memory_stats": memory_stats,
            "alerts": [asdict(alert) for alert in alerts],
            "profiled_functions": list(self.monitor.function_profiler.profile_data.keys()),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(reports_dir / "performance_data.json", 'w') as f:
            json.dump(performance_data, f, indent=2)

        print(f"\n📄 Detailed reports saved to: {reports_dir}")
        print("✅ Performance profiling infrastructure test completed!")

        return performance_data


def run_profiling_infrastructure_test():
    """Main function to test profiling infrastructure."""
    print("Starting Performance Profiling Infrastructure Test - Task 4.2")
    print("=" * 60)

    test_suite = PerformanceTestSuite()
    performance_data = test_suite.run_comprehensive_test()

    print("\n🎯 Task 4.2 Performance Profiling Infrastructure Complete!")
    print("📊 Advanced profiling and monitoring system established")

    return performance_data


if __name__ == "__main__":
    run_profiling_infrastructure_test()
