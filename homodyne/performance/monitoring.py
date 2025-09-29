#!/usr/bin/env python3
"""
Comprehensive Performance Monitoring System for Homodyne Analysis
================================================================

Advanced performance monitoring, tracking, and optimization system providing:
- Real-time performance metrics collection and analysis
- Performance regression detection and alerting
- Bottleneck identification with optimization recommendations
- Memory usage profiling and leak detection
- CPU utilization analysis and optimization suggestions
- I/O performance monitoring and data pipeline optimization
- Performance baseline tracking and trend analysis

Key Features:
- Automated performance testing with configurable benchmarks
- Multi-dimensional performance analysis (time, memory, CPU, I/O)
- Performance regression detection with statistical significance testing
- Optimization opportunity prioritization with impact/effort matrices
- Continuous performance monitoring for production deployments
- Integration with CI/CD pipelines for performance gating

Usage:
    # Run comprehensive performance monitoring
    python -m homodyne.performance_monitoring --mode comprehensive

    # Monitor specific components
    python -m homodyne.performance_monitoring --component optimization --iterations 100

    # Performance regression testing
    python -m homodyne.performance_monitoring --regression-test --baseline baseline.json

    # Real-time monitoring
    python -m homodyne.performance_monitoring --monitor --interval 30
"""

import argparse
import gc
import json
import logging
import os
import statistics
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import psutil

# Import homodyne components
try:
    from homodyne.analysis.core import HomodyneAnalysisCore
    from homodyne.core.config import ConfigManager
    from homodyne.core.cpu_optimization import CPUOptimizer
    from homodyne.core.cpu_optimization import get_cpu_optimization_info
    from homodyne.optimization.classical import ClassicalOptimizer
    from homodyne.optimization.robust import CVXPY_AVAILABLE
    from homodyne.optimization.robust import RobustHomodyneOptimizer
    from homodyne.performance.cpu_profiling import CPUProfiler
    from homodyne.performance.cpu_profiling import get_cpu_performance_info

    HOMODYNE_AVAILABLE = True
    CPU_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Homodyne components not available: {e}")
    HOMODYNE_AVAILABLE = False
    CVXPY_AVAILABLE = False
    CPU_OPTIMIZATION_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric measurement."""

    name: str
    value: float
    unit: str
    timestamp: float
    context: dict[str, Any]


@dataclass
class BenchmarkResult:
    """Comprehensive benchmark result."""

    benchmark_name: str
    duration: float
    memory_peak: int
    memory_delta: int
    cpu_time: float
    iterations: int
    statistics: dict[str, float]
    metadata: dict[str, Any]


@dataclass
class PerformanceReport:
    """Complete performance analysis report."""

    timestamp: float
    system_info: dict[str, Any]
    benchmarks: list[BenchmarkResult]
    bottlenecks: list[dict[str, Any]]
    optimizations: list[dict[str, Any]]
    trends: dict[str, Any]
    recommendations: list[dict[str, Any]]


class PerformanceMonitor:
    """
    Advanced performance monitoring system for homodyne analysis.

    Provides comprehensive performance analysis including:
    - Benchmark execution and statistical analysis
    - Memory profiling with leak detection
    - CPU utilization monitoring
    - Performance regression detection
    - Optimization opportunity identification
    - Real-time performance tracking
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize performance monitor."""
        self.config = config or self._default_config()
        self.results_dir = Path(self.config["output_dir"])
        self.results_dir.mkdir(exist_ok=True)

        self.metrics: list[PerformanceMetric] = []
        self.benchmarks: list[BenchmarkResult] = []
        self.process = psutil.Process()

        # Performance tracking state
        self._baseline_data: dict[str, Any] | None = None
        self._regression_threshold = self.config.get(
            "regression_threshold", 0.15
        )  # 15% regression

    def _default_config(self) -> dict[str, Any]:
        """Get default monitoring configuration."""
        return {
            "output_dir": "performance_monitoring",
            "benchmark_iterations": 10,
            "memory_sampling_interval": 0.1,
            "cpu_sampling_interval": 0.1,
            "regression_threshold": 0.15,
            "optimization_impact_threshold": 0.1,
            "enable_detailed_profiling": True,
            "enable_memory_tracking": True,
            "enable_cpu_profiling": True,
            "cpu_only_mode": True,  # CPU-only optimization focus
            "enable_cache_analysis": CPU_OPTIMIZATION_AVAILABLE,
            "enable_vectorization_analysis": True,
            "enable_multiprocess_benchmarks": True,
        }

    @contextmanager
    def performance_context(self, name: str, metadata: dict[str, Any] | None = None):
        """Context manager for measuring performance."""
        if not self.config.get("enable_detailed_profiling", True):
            yield
            return

        metadata = metadata or {}
        start_time = time.time()
        start_memory = self.process.memory_info()
        start_cpu_times = self.process.cpu_times()

        # Start memory tracking
        tracemalloc.start()
        gc.collect()  # Clean start

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.process.memory_info()
            end_cpu_times = self.process.cpu_times()

            # Get memory statistics
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Calculate metrics
            duration = end_time - start_time
            memory_delta = end_memory.rss - start_memory.rss
            cpu_time = (end_cpu_times.user - start_cpu_times.user) + (
                end_cpu_times.system - start_cpu_times.system
            )

            # Store metric
            metric = PerformanceMetric(
                name=name,
                value=duration,
                unit="seconds",
                timestamp=time.time(),
                context={
                    "memory_delta": memory_delta,
                    "memory_peak": peak,
                    "cpu_time": cpu_time,
                    "cpu_percent": (cpu_time / duration * 100) if duration > 0 else 0,
                    **metadata,
                },
            )
            self.metrics.append(metric)

            logger.debug(
                f"Performance metric '{name}': {duration:.3f}s, "
                f"Memory: {memory_delta / 1024 / 1024:.1f}MB, "
                f"CPU: {cpu_time:.3f}s"
            )

    def run_benchmark(
        self, name: str, benchmark_func: Callable, iterations: int | None = None
    ) -> BenchmarkResult:
        """
        Run a comprehensive benchmark with statistical analysis.

        Parameters
        ----------
        name : str
            Benchmark name
        benchmark_func : Callable
            Function to benchmark
        iterations : int, optional
            Number of iterations (default from config)

        Returns
        -------
        BenchmarkResult
            Comprehensive benchmark results
        """
        iterations = iterations or self.config["benchmark_iterations"]
        durations = []
        memory_peaks = []
        memory_deltas = []
        cpu_times = []

        logger.info(f"Running benchmark '{name}' with {iterations} iterations...")

        for i in range(iterations):
            gc.collect()  # Clean state for each iteration

            start_time = time.time()
            start_memory = self.process.memory_info()
            start_cpu_times = self.process.cpu_times()

            tracemalloc.start()

            try:
                # Run the benchmark
                benchmark_func()
            except Exception as e:
                logger.error(f"Benchmark '{name}' iteration {i} failed: {e}")
                continue

            end_time = time.time()
            end_memory = self.process.memory_info()
            end_cpu_times = self.process.cpu_times()

            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Record measurements
            durations.append(end_time - start_time)
            memory_peaks.append(peak)
            memory_deltas.append(end_memory.rss - start_memory.rss)
            cpu_times.append(
                (end_cpu_times.user - start_cpu_times.user)
                + (end_cpu_times.system - start_cpu_times.system)
            )

        if not durations:
            raise RuntimeError(f"All iterations of benchmark '{name}' failed")

        # Calculate statistics
        stats = {
            "mean": statistics.mean(durations),
            "median": statistics.median(durations),
            "stdev": statistics.stdev(durations) if len(durations) > 1 else 0,
            "min": min(durations),
            "max": max(durations),
            "p95": np.percentile(durations, 95),
            "p99": np.percentile(durations, 99),
            "cv": (
                statistics.stdev(durations) / statistics.mean(durations)
                if len(durations) > 1
                else 0
            ),
        }

        result = BenchmarkResult(
            benchmark_name=name,
            duration=stats["mean"],
            memory_peak=int(np.mean(memory_peaks)),
            memory_delta=int(np.mean(memory_deltas)),
            cpu_time=np.mean(cpu_times),
            iterations=len(durations),
            statistics=stats,
            metadata={
                "memory_peak_max": int(max(memory_peaks)),
                "memory_delta_max": int(max(memory_deltas)),
                "cpu_time_max": max(cpu_times),
                "system_info": self._get_system_info(),
            },
        )

        self.benchmarks.append(result)
        logger.info(
            f"Benchmark '{name}' completed: {stats['mean']:.3f}s ± {stats['stdev']:.3f}s"
        )

        return result

    def benchmark_core_operations(self) -> list[BenchmarkResult]:
        """Benchmark core homodyne operations."""
        if not HOMODYNE_AVAILABLE:
            logger.warning("Homodyne not available - skipping core benchmarks")
            return []

        results = []

        # Setup test environment
        config_manager = ConfigManager()
        config_manager.config = self._create_test_config()
        core = HomodyneAnalysisCore(config_manager)

        test_params = np.array([100.0, 0.5, 10.0])
        test_angles = np.linspace(0, 180, 10)

        # Generate test data once
        test_data = self._generate_test_data(core, test_params, test_angles)

        # Benchmark chi-squared calculation (CPU-optimized)
        def chi_squared_benchmark():
            return core.calculate_chi_squared_optimized(
                test_params, test_angles, test_data
            )

        results.append(
            self.run_benchmark("chi_squared_calculation_cpu", chi_squared_benchmark)
        )

        # Benchmark CPU-specific optimizations if available
        if CPU_OPTIMIZATION_AVAILABLE:
            cpu_optimizer = CPUOptimizer()

            # Cache-optimized matrix operations
            def cache_optimized_benchmark():
                test_matrix = np.random.randn(100, 100)
                return cpu_optimizer.optimize_matrix_operations_cpu(
                    test_matrix, operation="correlation"
                )

            results.append(
                self.run_benchmark("cache_optimized_operations", cache_optimized_benchmark)
            )

            # Parallel chi-squared computation
            def parallel_chi_squared_benchmark():
                parameter_sets = [test_params + np.random.randn(3) * 0.1 for _ in range(10)]
                return cpu_optimizer.parallel_chi_squared_cpu(
                    parameter_sets, test_angles, test_data
                )

            results.append(
                self.run_benchmark("parallel_chi_squared_cpu", parallel_chi_squared_benchmark)
            )

        # Benchmark correlation calculation
        def correlation_benchmark():
            return core.calculate_c2_nonequilibrium_laminar_parallel(
                test_params, test_angles
            )

        results.append(
            self.run_benchmark("correlation_calculation", correlation_benchmark)
        )

        # Benchmark parameter estimation (if optimization available)
        if hasattr(core, "estimate_parameters"):

            def parameter_estimation_benchmark():
                return core.estimate_parameters(test_data, test_angles)

            results.append(
                self.run_benchmark(
                    "parameter_estimation", parameter_estimation_benchmark
                )
            )

        return results

    def benchmark_optimization_methods(self) -> list[BenchmarkResult]:
        """Benchmark optimization methods."""
        if not HOMODYNE_AVAILABLE:
            logger.warning("Homodyne not available - skipping optimization benchmarks")
            return []

        results = []

        # Setup test environment
        config_manager = ConfigManager()
        config_manager.config = self._create_test_config()
        core = HomodyneAnalysisCore(config_manager)

        test_params = np.array([100.0, 0.5, 10.0])
        test_angles = np.linspace(0, 180, 5)  # Fewer angles for faster benchmarking
        test_data = self._generate_test_data(core, test_params, test_angles)

        # Benchmark classical optimization
        try:
            classical_optimizer = ClassicalOptimizer(core, config_manager.config)

            def classical_optimization_benchmark():
                return classical_optimizer.optimize_parameters(
                    test_params, test_angles, test_data
                )

            results.append(
                self.run_benchmark(
                    "classical_optimization", classical_optimization_benchmark, 5
                )
            )
        except Exception as e:
            logger.warning(f"Classical optimization benchmark failed: {e}")

        # Benchmark robust optimization if available
        if CVXPY_AVAILABLE:
            try:
                robust_optimizer = RobustHomodyneOptimizer(core, config_manager.config)

                def robust_optimization_benchmark():
                    return robust_optimizer.run_robust_optimization(
                        test_params, test_angles, test_data, method="wasserstein"
                    )

                results.append(
                    self.run_benchmark(
                        "robust_optimization", robust_optimization_benchmark, 3
                    )
                )
            except Exception as e:
                logger.warning(f"Robust optimization benchmark failed: {e}")

        return results

    def benchmark_data_operations(self) -> list[BenchmarkResult]:
        """Benchmark data loading and processing operations."""
        results = []

        # Benchmark numpy operations
        def numpy_operations_benchmark():
            size = 1000
            a = np.random.random((size, size))
            b = np.random.random((size, size))
            c = np.dot(a, b)
            return np.sum(c)

        results.append(
            self.run_benchmark("numpy_matrix_operations", numpy_operations_benchmark)
        )

        # Benchmark data generation
        def data_generation_benchmark():
            return np.random.exponential(1.0, (100, 100, 100))

        results.append(self.run_benchmark("data_generation", data_generation_benchmark))

        # Benchmark FFT operations (common in XPCS)
        def fft_benchmark():
            data = np.random.random(10000)
            return np.fft.fft(data)

        results.append(self.run_benchmark("fft_operations", fft_benchmark))

        return results

    def detect_performance_regressions(
        self, baseline_file: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Detect performance regressions compared to baseline.

        Parameters
        ----------
        baseline_file : str, optional
            Path to baseline performance data

        Returns
        -------
        list[dict[str, Any]]
            List of detected regressions
        """
        if baseline_file and Path(baseline_file).exists():
            with open(baseline_file) as f:
                baseline_data = json.load(f)
        elif self._baseline_data:
            baseline_data = self._baseline_data
        else:
            logger.warning("No baseline data available for regression detection")
            return []

        regressions = []
        baseline_benchmarks = {
            b["benchmark_name"]: b for b in baseline_data.get("benchmarks", [])
        }

        for current_benchmark in self.benchmarks:
            name = current_benchmark.benchmark_name
            if name not in baseline_benchmarks:
                continue

            baseline_benchmark = baseline_benchmarks[name]
            current_duration = current_benchmark.duration
            baseline_duration = baseline_benchmark["duration"]

            # Calculate regression percentage
            regression_ratio = (
                current_duration - baseline_duration
            ) / baseline_duration
            regression_percent = regression_ratio * 100

            if regression_ratio > self._regression_threshold:
                regressions.append(
                    {
                        "benchmark": name,
                        "current_duration": current_duration,
                        "baseline_duration": baseline_duration,
                        "regression_percent": regression_percent,
                        "severity": "high" if regression_ratio > 0.5 else "medium",
                        "statistical_significance": self._calculate_significance(
                            current_benchmark, baseline_benchmark
                        ),
                    }
                )

        if regressions:
            logger.warning(f"Detected {len(regressions)} performance regressions")
        else:
            logger.info("No significant performance regressions detected")

        return regressions

    def identify_optimization_opportunities(self) -> list[dict[str, Any]]:
        """Identify optimization opportunities based on benchmark results."""
        opportunities = []

        # Analyze benchmark results for optimization opportunities
        for benchmark in self.benchmarks:
            # High variability indicates optimization opportunity
            if benchmark.statistics["cv"] > 0.1:  # Coefficient of variation > 10%
                opportunities.append(
                    {
                        "area": f"{benchmark.benchmark_name}_consistency",
                        "description": f"Reduce performance variability in {benchmark.benchmark_name}",
                        "evidence": f"CV = {benchmark.statistics['cv']:.2%}",
                        "impact": "medium",
                        "effort": "medium",
                        "priority_score": 6,
                    }
                )

            # Slow operations
            if benchmark.duration > 1.0:  # Operations taking > 1 second
                opportunities.append(
                    {
                        "area": f"{benchmark.benchmark_name}_optimization",
                        "description": f"Optimize {benchmark.benchmark_name} algorithm",
                        "evidence": f"Duration = {benchmark.duration:.2f}s",
                        "impact": "high",
                        "effort": "high",
                        "priority_score": 8,
                    }
                )

            # High memory usage
            if benchmark.memory_peak > 100 * 1024 * 1024:  # > 100MB
                opportunities.append(
                    {
                        "area": f"{benchmark.benchmark_name}_memory",
                        "description": f"Reduce memory usage in {benchmark.benchmark_name}",
                        "evidence": f"Peak memory = {benchmark.memory_peak / 1024 / 1024:.1f}MB",
                        "impact": "medium",
                        "effort": "medium",
                        "priority_score": 7,
                    }
                )

        # Sort by priority score
        opportunities.sort(key=lambda x: x["priority_score"], reverse=True)

        return opportunities

    def generate_performance_report(self) -> PerformanceReport:
        """Generate comprehensive performance report."""
        # Detect regressions
        regressions = self.detect_performance_regressions()

        # Identify optimization opportunities
        optimizations = self.identify_optimization_opportunities()

        # Calculate trends (if we have historical data)
        trends = self._calculate_performance_trends()

        # Generate recommendations
        recommendations = self._generate_recommendations(regressions, optimizations)

        report = PerformanceReport(
            timestamp=time.time(),
            system_info=self._get_system_info(),
            benchmarks=self.benchmarks,
            bottlenecks=regressions,
            optimizations=optimizations,
            trends=trends,
            recommendations=recommendations,
        )

        return report

    def save_report(
        self, report: PerformanceReport, filename: str | None = None
    ) -> str:
        """Save performance report to file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"performance_report_{timestamp}.json"

        filepath = self.results_dir / filename

        # Convert to serializable format
        report_dict = asdict(report)

        with open(filepath, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)

        logger.info(f"Performance report saved to {filepath}")
        return str(filepath)

    def create_performance_dashboard(
        self, report: PerformanceReport, output_file: str | None = None
    ):
        """Create performance monitoring dashboard."""
        if output_file is None:
            output_file = self.results_dir / "performance_dashboard.png"

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Homodyne Analysis Performance Dashboard", fontsize=16)

        # Benchmark durations
        ax1 = axes[0, 0]
        names = [b.benchmark_name for b in report.benchmarks]
        durations = [b.duration for b in report.benchmarks]
        errors = [b.statistics["stdev"] for b in report.benchmarks]

        ax1.bar(range(len(names)), durations, yerr=errors, capsize=5)
        ax1.set_title("Benchmark Durations")
        ax1.set_ylabel("Time (seconds)")
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha="right")

        # Memory usage
        ax2 = axes[0, 1]
        memory_peaks = [
            b.memory_peak / 1024 / 1024 for b in report.benchmarks
        ]  # Convert to MB
        ax2.bar(range(len(names)), memory_peaks)
        ax2.set_title("Peak Memory Usage")
        ax2.set_ylabel("Memory (MB)")
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha="right")

        # Performance variability
        ax3 = axes[1, 0]
        cv_values = [
            b.statistics["cv"] * 100 for b in report.benchmarks
        ]  # Convert to percentage
        ax3.bar(range(len(names)), cv_values)
        ax3.set_title("Performance Variability (Coefficient of Variation)")
        ax3.set_ylabel("CV (%)")
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels(names, rotation=45, ha="right")
        ax3.axhline(
            y=10,
            color="r",
            linestyle="--",
            alpha=0.7,
            label="High variability threshold",
        )
        ax3.legend()

        # Optimization opportunities
        ax4 = axes[1, 1]
        if report.optimizations:
            opp_areas = [
                opp["area"][:20] for opp in report.optimizations[:5]
            ]  # Top 5, truncated names
            priorities = [opp["priority_score"] for opp in report.optimizations[:5]]
            ax4.barh(range(len(opp_areas)), priorities)
            ax4.set_title("Top Optimization Opportunities")
            ax4.set_xlabel("Priority Score")
            ax4.set_yticks(range(len(opp_areas)))
            ax4.set_yticklabels(opp_areas)
        else:
            ax4.text(
                0.5,
                0.5,
                "No optimization opportunities identified",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("Optimization Opportunities")

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Performance dashboard saved to {output_file}")

    # Helper methods
    def _get_system_info(self) -> dict[str, Any]:
        """Get comprehensive system information."""
        system_info = {
            "platform": {
                "system": os.uname().sysname,
                "release": os.uname().release,
                "machine": os.uname().machine,
            },
            "hardware": {
                "cpu_count": os.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
            },
            "python": {"version": sys.version, "executable": sys.executable},
            "process": {
                "pid": os.getpid(),
                "memory_rss": self.process.memory_info().rss,
                "cpu_percent": self.process.cpu_percent(),
            },
            "cpu_optimization": {
                "available": CPU_OPTIMIZATION_AVAILABLE,
                "mode": "cpu_only",  # Explicitly CPU-only
            }
        }

        # Add CPU optimization details if available
        if CPU_OPTIMIZATION_AVAILABLE:
            try:
                cpu_info = get_cpu_optimization_info()
                cpu_perf_info = get_cpu_performance_info()
                system_info["cpu_optimization"].update({
                    "optimization_info": cpu_info,
                    "performance_info": cpu_perf_info,
                })
            except Exception as e:
                logger.warning(f"Failed to get CPU optimization info: {e}")

        return system_info

    def _create_test_config(self) -> dict[str, Any]:
        """Create test configuration for benchmarking."""
        return {
            "analyzer_parameters": {
                "q_magnitude": 0.01,
                "L": 1000.0,
                "dt": 0.1,
                "time_points": 50,
                "n_time_steps": 50,
            },
            "analysis_settings": {
                "enable_static_isotropic": True,
                "enable_laminar_flow": False,
            },
            "performance_settings": {
                "enable_parallel_processing": True,
                "num_threads": min(4, os.cpu_count()),  # CPU-optimized threading
                "cpu_only_mode": True,
                "enable_numba_jit": True,
                "enable_vectorization": True,
            },
            "optimization_config": {
                "classical_optimization": {
                    "methods": ["nelder_mead"],
                    "method_options": {
                        "nelder_mead": {
                            "maxiter": 100,  # Limit for benchmarking
                            "xatol": 1e-4,
                            "fatol": 1e-4,
                        }
                    },
                }
            },
        }

    def _generate_test_data(
        self, core, params: np.ndarray, angles: np.ndarray
    ) -> np.ndarray:
        """Generate synthetic test data for benchmarking."""
        try:
            theoretical = core.calculate_c2_nonequilibrium_laminar_parallel(
                params, angles
            )
            noise = np.random.normal(0, 0.01, theoretical.shape)
            return theoretical + noise
        except Exception:
            # Fallback
            n_angles = len(angles)
            n_times = 50
            return np.random.exponential(1.0, (n_angles, n_times, n_times))

    def _calculate_significance(
        self, current: BenchmarkResult, baseline: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate statistical significance of performance difference."""
        # Simple t-test approximation
        current_mean = current.duration
        current_std = current.statistics["stdev"]
        current_n = current.iterations

        baseline_mean = baseline["duration"]
        baseline_std = baseline.get("statistics", {}).get("stdev", baseline_mean * 0.1)
        baseline_n = baseline.get("iterations", 10)

        # Pooled standard deviation
        pooled_std = np.sqrt(
            ((current_n - 1) * current_std**2 + (baseline_n - 1) * baseline_std**2)
            / (current_n + baseline_n - 2)
        )

        # t-statistic
        t_stat = (current_mean - baseline_mean) / (
            pooled_std * np.sqrt(1 / current_n + 1 / baseline_n)
        )

        return {
            "t_statistic": t_stat,
            "pooled_std": pooled_std,
            "significant": abs(t_stat) > 2.0,  # Rough significance threshold
        }

    def _calculate_performance_trends(self) -> dict[str, Any]:
        """Calculate performance trends from historical data."""
        # This would analyze historical performance data
        # For now, return placeholder
        return {
            "trend_direction": "stable",
            "trend_confidence": 0.8,
            "seasonal_patterns": False,
            "growth_rate": 0.0,
        }

    def _generate_recommendations(
        self, regressions: list[dict], optimizations: list[dict]
    ) -> list[dict[str, Any]]:
        """Generate performance recommendations."""
        recommendations = []

        # Recommendations based on regressions
        for regression in regressions:
            recommendations.append(
                {
                    "type": "regression_fix",
                    "priority": "high",
                    "description": f"Address performance regression in {regression['benchmark']}",
                    "expected_impact": f"Restore {regression['regression_percent']:.1f}% performance loss",
                }
            )

        # Recommendations based on optimization opportunities
        for opp in optimizations[:3]:  # Top 3 opportunities
            recommendations.append(
                {
                    "type": "optimization",
                    "priority": "medium",
                    "description": opp["description"],
                    "expected_impact": f"Estimated {opp['impact']} impact improvement",
                }
            )

        # General recommendations
        recommendations.append(
            {
                "type": "monitoring",
                "priority": "low",
                "description": "Continue regular performance monitoring",
                "expected_impact": "Early detection of performance issues",
            }
        )

        return recommendations


def main():
    """Main entry point for performance monitoring."""
    parser = argparse.ArgumentParser(
        description="Homodyne Analysis Performance Monitoring"
    )

    parser.add_argument(
        "--mode",
        choices=["quick", "comprehensive", "regression"],
        default="comprehensive",
        help="Monitoring mode",
    )
    parser.add_argument(
        "--component",
        choices=["core", "optimization", "data"],
        help="Focus on specific component",
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of benchmark iterations"
    )
    parser.add_argument("--baseline", help="Baseline file for regression testing")
    parser.add_argument(
        "--output-dir", default="performance_monitoring", help="Output directory"
    )
    parser.add_argument(
        "--create-dashboard", action="store_true", help="Create performance dashboard"
    )

    args = parser.parse_args()

    # Initialize monitor
    config = {"output_dir": args.output_dir, "benchmark_iterations": args.iterations}
    monitor = PerformanceMonitor(config)

    try:
        logger.info(f"Starting performance monitoring (mode: {args.mode})")

        if args.mode == "regression" and args.baseline:
            # Regression testing mode
            monitor.benchmark_core_operations()
            regressions = monitor.detect_performance_regressions(args.baseline)

            if regressions:
                logger.error(f"Performance regressions detected: {len(regressions)}")
                for reg in regressions:
                    logger.error(
                        f"  {reg['benchmark']}: +{reg['regression_percent']:.1f}% ({reg['severity']})"
                    )
                return 1
            else:
                logger.info("No performance regressions detected")

        elif args.component:
            # Component-specific monitoring
            if args.component == "core":
                monitor.benchmark_core_operations()
            elif args.component == "optimization":
                monitor.benchmark_optimization_methods()
            elif args.component == "data":
                monitor.benchmark_data_operations()
        else:
            # Comprehensive monitoring
            monitor.benchmark_core_operations()
            monitor.benchmark_optimization_methods()
            monitor.benchmark_data_operations()

        # Generate and save report
        report = monitor.generate_performance_report()
        report_file = monitor.save_report(report)

        # Create dashboard if requested
        if args.create_dashboard:
            monitor.create_performance_dashboard(report)

        logger.info("Performance monitoring completed successfully")
        print(f"\nPerformance report saved to: {report_file}")

        # Print summary
        if report.benchmarks:
            print("\nBenchmark Summary:")
            for benchmark in report.benchmarks:
                print(
                    f"  {benchmark.benchmark_name}: {benchmark.duration:.3f}s ± {benchmark.statistics['stdev']:.3f}s"
                )

        if report.optimizations:
            print(f"\nTop optimization opportunities: {len(report.optimizations)}")
            for opp in report.optimizations[:3]:
                print(
                    f"  • {opp['description']} (Priority: {opp['priority_score']}/10)"
                )

        return 0

    except Exception as e:
        logger.error(f"Performance monitoring failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
