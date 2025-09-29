#!/usr/bin/env python3
"""
Performance Baseline Analysis for Homodyne Analysis
==================================================

Comprehensive performance monitoring and baseline establishment for the
homodyne analysis Python scientific computing application.

This module provides:
1. Current performance metrics collection (response times, throughput, resource usage)
2. Performance bottleneck identification across the stack
3. User experience impact assessment
4. Performance baseline report generation for optimization tracking
5. Optimization opportunity prioritization by impact/effort ratio

Key Performance Areas:
- Python application performance profiling
- Scientific computing workload analysis
- Memory usage patterns and optimization
- CPU utilization during computations
- I/O operations and data processing bottlenecks
- Test execution times and performance regression detection

Usage:
    python -m homodyne.performance_baseline --profile-mode comprehensive
    python -m homodyne.performance_baseline --quick-analysis
    python -m homodyne.performance_baseline --memory-analysis
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import psutil

# Import homodyne components for testing with explicit error handling
try:
    from homodyne.analysis.core import HomodyneAnalysisCore
    from homodyne.core.config import ConfigManager
    from homodyne.optimization.classical import ClassicalOptimizer

    HOMODYNE_AVAILABLE = True
except ImportError as e:
    import warnings

    warnings.warn(
        f"Homodyne components not available for baseline testing: {e}",
        ImportWarning,
        stacklevel=2,
    )
    HOMODYNE_AVAILABLE = False

    # Explicit None assignments for missing components
    HomodyneAnalysisCore = None
    ConfigManager = None
    ClassicalOptimizer = None

# Check for robust optimization separately with explicit handling
try:
    from homodyne.optimization.robust import CVXPY_AVAILABLE
    from homodyne.optimization.robust import RobustHomodyneOptimizer

    ROBUST_AVAILABLE = True
except ImportError as e:
    import warnings

    warnings.warn(
        f"Robust optimization not available for baseline testing: {e}",
        ImportWarning,
        stacklevel=2,
    )
    ROBUST_AVAILABLE = False
    CVXPY_AVAILABLE = False
    RobustHomodyneOptimizer = None

# Performance monitoring tools with explicit availability checking
try:
    import numba  # noqa: F401

    NUMBA_AVAILABLE = True
except ImportError as e:
    import warnings

    warnings.warn(
        f"Numba not available for performance optimization: {e}",
        ImportWarning,
        stacklevel=2,
    )
    NUMBA_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """
    Comprehensive performance profiler for homodyne analysis application.

    Monitors CPU usage, memory consumption, I/O operations, and computational
    bottlenecks to establish performance baselines and identify optimization
    opportunities.
    """

    def __init__(self, output_dir: str = "performance_reports"):
        """Initialize performance profiler."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.metrics: dict[str, Any] = {
            "system_info": self._collect_system_info(),
            "baseline_metrics": {},
            "bottlenecks": [],
            "optimization_opportunities": [],
            "test_performance": {},
            "memory_analysis": {},
            "cpu_analysis": {},
            "io_analysis": {},
        }

        # Performance tracking
        self.start_time = time.time()
        self.process = psutil.Process()

    def _collect_system_info(self) -> dict[str, Any]:
        """Collect system information for baseline context."""
        try:
            return {
                "platform": {
                    "system": os.uname().sysname,
                    "release": os.uname().release,
                    "machine": os.uname().machine,
                    "processor": os.uname().machine,
                },
                "python": {
                    "version": sys.version,
                    "executable": sys.executable,
                    "path": sys.path[:3],  # First 3 paths only
                },
                "hardware": {
                    "cpu_count": os.cpu_count(),
                    "cpu_freq": (
                        psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
                    ),
                    "memory_total": psutil.virtual_memory().total,
                    "memory_available": psutil.virtual_memory().available,
                    "disk_usage": psutil.disk_usage("/")._asdict(),
                },
                "dependencies": {
                    "numpy": self._get_package_version("numpy"),
                    "scipy": self._get_package_version("scipy"),
                    "matplotlib": self._get_package_version("matplotlib"),
                    "numba": (
                        self._get_package_version("numba")
                        if NUMBA_AVAILABLE
                        else "Not available"
                    ),
                    "cvxpy": (
                        self._get_package_version("cvxpy")
                        if CVXPY_AVAILABLE
                        else "Not available"
                    ),
                    "psutil": self._get_package_version("psutil"),
                },
            }
        except Exception as e:
            logger.error(f"Error collecting system info: {e}")
            return {"error": str(e)}

    def _get_package_version(self, package_name: str) -> str:
        """Get version of a package."""
        try:
            import importlib.metadata

            return importlib.metadata.version(package_name)
        except Exception:
            return "Unknown"

    @contextmanager
    def monitor_performance(self, operation_name: str):
        """Context manager for monitoring operation performance."""
        start_time = time.time()
        start_memory = self.process.memory_info()
        start_cpu_times = self.process.cpu_times()

        # Start memory tracing
        tracemalloc.start()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.process.memory_info()
            end_cpu_times = self.process.cpu_times()

            # Get memory peak
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Calculate metrics
            duration = end_time - start_time
            memory_delta = end_memory.rss - start_memory.rss
            cpu_time = (end_cpu_times.user - start_cpu_times.user) + (
                end_cpu_times.system - start_cpu_times.system
            )

            # Store metrics
            self.metrics["baseline_metrics"][operation_name] = {
                "duration": duration,
                "memory_delta": memory_delta,
                "memory_peak": peak,
                "cpu_time": cpu_time,
                "cpu_percent": (cpu_time / duration * 100) if duration > 0 else 0,
                "timestamp": time.time(),
            }

            logger.info(
                f"{operation_name}: {duration:.3f}s, "
                f"Memory: {memory_delta / 1024 / 1024:.1f}MB, "
                f"CPU: {cpu_time:.3f}s"
            )

    def profile_import_performance(self):
        """Profile module import performance."""
        logger.info("Profiling import performance...")

        import_times = {}

        # Test core imports
        imports_to_test = [
            ("numpy", "import numpy"),
            ("scipy", "import scipy"),
            ("matplotlib", "import matplotlib"),
            (
                "homodyne.core.config",
                "from homodyne.core.config import HomodyneConfigManager",
            ),
            (
                "homodyne.analysis.core",
                "from homodyne.analysis.core import HomodyneAnalysisCore",
            ),
            (
                "homodyne.optimization.classical",
                "from homodyne.optimization.classical import ClassicalOptimizer",
            ),
        ]

        if ROBUST_AVAILABLE and CVXPY_AVAILABLE:
            imports_to_test.append(
                (
                    "homodyne.optimization.robust",
                    "from homodyne.optimization.robust import RobustHomodyneOptimizer",
                )
            )

        for module_name, import_stmt in imports_to_test:
            try:
                start_time = time.time()
                exec(import_stmt)
                end_time = time.time()
                import_times[module_name] = end_time - start_time
                logger.info(f"Import {module_name}: {import_times[module_name]:.3f}s")
            except ImportError as e:
                import_times[module_name] = f"Failed: {e}"
                logger.warning(f"Failed to import {module_name}: {e}")

        self.metrics["baseline_metrics"]["import_times"] = import_times
        return import_times

    def profile_computation_performance(self):
        """Profile core computational operations."""
        if not HOMODYNE_AVAILABLE:
            logger.warning("Homodyne not available - skipping computation profiling")
            return

        logger.info("Profiling computational performance...")

        try:
            # Create a minimal test configuration
            test_config = self._create_test_config()

            with self.monitor_performance("config_creation"):
                config_manager = ConfigManager()
            config_manager.config = test_config

            with self.monitor_performance("core_initialization"):
                core = HomodyneAnalysisCore(config_manager)

            # Test parameter estimation performance
            test_params = np.array([100.0, 0.5, 10.0])  # D0, alpha, D_offset
            test_angles = np.linspace(0, 180, 10)  # 10 angles for testing

            # Generate synthetic test data
            with self.monitor_performance("synthetic_data_generation"):
                test_data = self._generate_synthetic_data(
                    core, test_params, test_angles
                )

            # Test chi-squared calculation performance
            with self.monitor_performance("chi_squared_calculation"):
                for _i in range(10):  # Multiple iterations for better measurement
                    chi2 = core.calculate_chi_squared_optimized(
                        test_params, test_angles, test_data
                    )

            # Test correlation function calculation
            with self.monitor_performance("correlation_calculation"):
                for _i in range(5):  # Multiple iterations
                    core.calculate_c2_nonequilibrium_laminar_parallel(
                        test_params, test_angles
                    )

            logger.info(f"Chi-squared value: {chi2:.6f}")

        except Exception as e:
            logger.error(f"Error in computation profiling: {e}")
            self.metrics["bottlenecks"].append(
                {
                    "operation": "computation_profiling",
                    "error": str(e),
                    "severity": "high",
                }
            )

    def profile_optimization_performance(self):
        """Profile optimization method performance."""
        if not HOMODYNE_AVAILABLE:
            logger.warning("Homodyne not available - skipping optimization profiling")
            return

        logger.info("Profiling optimization performance...")

        try:
            # Setup test environment
            test_config = self._create_test_config()
            config_manager = ConfigManager()
            config_manager.config = test_config
            core = HomodyneAnalysisCore(config_manager)

            test_params = np.array([100.0, 0.5, 10.0])
            test_angles = np.linspace(0, 180, 5)  # Fewer angles for faster testing
            test_data = self._generate_synthetic_data(core, test_params, test_angles)

            # Test classical optimization
            with self.monitor_performance("classical_optimizer_init"):
                classical_optimizer = ClassicalOptimizer(core, test_config)

            with self.monitor_performance("classical_nelder_mead"):
                try:
                    result = classical_optimizer.run_nelder_mead_optimization(
                        test_params, test_angles, test_data
                    )
                    logger.info(
                        f"Classical optimization result: {result[1].get('status', 'unknown')}"
                    )
                except Exception as e:
                    logger.warning(f"Classical optimization failed: {e}")

            # Test robust optimization if available
            if ROBUST_AVAILABLE and CVXPY_AVAILABLE:
                try:
                    with self.monitor_performance("robust_optimizer_init"):
                        robust_optimizer = RobustHomodyneOptimizer(core, test_config)

                    with self.monitor_performance("robust_wasserstein"):
                        result = robust_optimizer.run_robust_optimization(
                            test_params, test_angles, test_data, method="wasserstein"
                        )
                        logger.info(
                            f"Robust optimization result: {result[1].get('status', 'unknown')}"
                        )
                except Exception as e:
                    logger.warning(f"Robust optimization failed: {e}")
            else:
                logger.info(
                    "Robust optimization not available - skipping robust optimization profiling"
                )

        except Exception as e:
            logger.error(f"Error in optimization profiling: {e}")
            self.metrics["bottlenecks"].append(
                {
                    "operation": "optimization_profiling",
                    "error": str(e),
                    "severity": "high",
                }
            )

    def profile_memory_usage(self):
        """Profile memory usage patterns."""
        logger.info("Profiling memory usage...")

        gc.collect()  # Clean up before measuring
        initial_memory = self.process.memory_info().rss

        memory_samples = []

        # Monitor memory during different operations
        operations = [
            ("baseline", lambda: time.sleep(0.1)),
            ("numpy_operations", self._test_numpy_operations),
            ("data_loading_simulation", self._test_data_loading),
        ]

        if HOMODYNE_AVAILABLE:
            operations.extend(
                [
                    ("homodyne_core_creation", self._test_homodyne_core),
                    ("computation_intensive", self._test_computation_intensive),
                ]
            )

        for op_name, operation in operations:
            gc.collect()
            start_memory = self.process.memory_info().rss

            try:
                operation()
                end_memory = self.process.memory_info().rss
                memory_delta = end_memory - start_memory

                memory_samples.append(
                    {
                        "operation": op_name,
                        "memory_delta": memory_delta,
                        "memory_total": end_memory,
                        "memory_percent": (end_memory / psutil.virtual_memory().total)
                        * 100,
                    }
                )

                logger.info(
                    f"Memory {op_name}: {memory_delta / 1024 / 1024:.1f}MB delta"
                )

            except Exception as e:
                logger.warning(f"Memory test {op_name} failed: {e}")
                memory_samples.append({"operation": op_name, "error": str(e)})

        self.metrics["memory_analysis"] = {
            "initial_memory": initial_memory,
            "samples": memory_samples,
            "peak_memory": max(
                sample.get("memory_total", 0) for sample in memory_samples
            ),
        }

    def profile_cpu_usage(self):
        """Profile CPU usage patterns."""
        logger.info("Profiling CPU usage...")

        cpu_samples = []

        # Monitor CPU during different workloads
        workloads = [
            ("idle", lambda: time.sleep(0.5)),
            ("cpu_intensive", self._test_cpu_intensive),
            ("scientific_computing", self._test_scientific_computing),
        ]

        for workload_name, workload in workloads:
            # Monitor CPU usage
            cpu_start = time.time()
            cpu_times_start = self.process.cpu_times()

            try:
                workload()

                cpu_end = time.time()
                cpu_times_end = self.process.cpu_times()

                wall_time = cpu_end - cpu_start
                cpu_time = (cpu_times_end.user - cpu_times_start.user) + (
                    cpu_times_end.system - cpu_times_start.system
                )
                cpu_percent = (cpu_time / wall_time * 100) if wall_time > 0 else 0

                cpu_samples.append(
                    {
                        "workload": workload_name,
                        "wall_time": wall_time,
                        "cpu_time": cpu_time,
                        "cpu_percent": cpu_percent,
                        "efficiency": cpu_time / wall_time if wall_time > 0 else 0,
                    }
                )

                logger.info(f"CPU {workload_name}: {cpu_percent:.1f}% utilization")

            except Exception as e:
                logger.warning(f"CPU test {workload_name} failed: {e}")
                cpu_samples.append({"workload": workload_name, "error": str(e)})

        self.metrics["cpu_analysis"] = {
            "samples": cpu_samples,
            "system_cpu_count": os.cpu_count(),
        }

    def run_performance_tests(self):
        """Run the existing test suite and measure performance."""
        logger.info("Running performance tests...")

        test_start_time = time.time()

        try:
            import subprocess

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "homodyne/tests/",
                    "-v",
                    "--tb=short",
                    "-q",
                ],
                check=False, capture_output=True,
                text=True,
                timeout=300,
            )

            test_end_time = time.time()
            test_duration = test_end_time - test_start_time

            self.metrics["test_performance"] = {
                "duration": test_duration,
                "return_code": result.returncode,
                "stdout_lines": len(result.stdout.split("\n")) if result.stdout else 0,
                "stderr_lines": len(result.stderr.split("\n")) if result.stderr else 0,
                "success": result.returncode == 0,
            }

            logger.info(f"Test suite completed in {test_duration:.1f}s")

            if result.returncode != 0:
                logger.warning("Some tests failed")
                self.metrics["bottlenecks"].append(
                    {
                        "operation": "test_execution",
                        "issue": "Test failures detected",
                        "severity": "medium",
                    }
                )

        except Exception as e:
            logger.error(f"Error running tests: {e}")
            self.metrics["test_performance"] = {"error": str(e)}

    def identify_bottlenecks(self):
        """Identify performance bottlenecks based on collected metrics."""
        logger.info("Identifying performance bottlenecks...")

        bottlenecks = []

        # Analyze baseline metrics for slow operations
        for operation, metrics in self.metrics["baseline_metrics"].items():
            if isinstance(metrics, dict) and "duration" in metrics:
                duration = metrics["duration"]

                # Flag slow operations (threshold-based)
                if duration > 5.0:  # Operations taking more than 5 seconds
                    bottlenecks.append(
                        {
                            "operation": operation,
                            "issue": f"Slow operation: {duration:.1f}s",
                            "severity": "high",
                            "optimization_opportunity": "Algorithm optimization or caching",
                        }
                    )
                elif duration > 1.0:  # Operations taking more than 1 second
                    bottlenecks.append(
                        {
                            "operation": operation,
                            "issue": f"Moderate latency: {duration:.1f}s",
                            "severity": "medium",
                            "optimization_opportunity": "Performance tuning",
                        }
                    )

        # Analyze memory usage
        memory_analysis = self.metrics.get("memory_analysis", {})
        if "samples" in memory_analysis:
            for sample in memory_analysis["samples"]:
                if "memory_delta" in sample:
                    memory_delta_mb = sample["memory_delta"] / 1024 / 1024
                    if memory_delta_mb > 100:  # More than 100MB delta
                        bottlenecks.append(
                            {
                                "operation": sample["operation"],
                                "issue": f"High memory usage: {memory_delta_mb:.1f}MB",
                                "severity": "medium",
                                "optimization_opportunity": "Memory optimization or streaming",
                            }
                        )

        # Analyze import times
        import_times = self.metrics["baseline_metrics"].get("import_times", {})
        for module, import_time in import_times.items():
            if isinstance(import_time, (int, float)) and import_time > 2.0:
                bottlenecks.append(
                    {
                        "operation": f"import_{module}",
                        "issue": f"Slow import: {import_time:.1f}s",
                        "severity": "low",
                        "optimization_opportunity": "Lazy loading or import optimization",
                    }
                )

        self.metrics["bottlenecks"].extend(bottlenecks)

    def prioritize_optimizations(self):
        """Prioritize optimization opportunities by impact/effort ratio."""
        logger.info("Prioritizing optimization opportunities...")

        # Define optimization opportunities with impact/effort estimates
        optimizations = [
            {
                "area": "Chi-squared calculation",
                "description": "Optimize chi-squared computation with vectorization",
                "impact": "high",  # Frequently called operation
                "effort": "medium",
                "estimated_improvement": "20-40% speedup",
                "priority_score": 8,
            },
            {
                "area": "Memory allocation",
                "description": "Implement memory pooling for frequent allocations",
                "impact": "medium",
                "effort": "medium",
                "estimated_improvement": "10-20% memory reduction",
                "priority_score": 6,
            },
            {
                "area": "Import optimization",
                "description": "Implement lazy loading for heavy modules",
                "impact": "medium",
                "effort": "low",
                "estimated_improvement": "50-80% startup time reduction",
                "priority_score": 7,
            },
            {
                "area": "Numba JIT compilation",
                "description": "Apply JIT compilation to computational kernels",
                "impact": "high",
                "effort": "high",
                "estimated_improvement": "2-5x speedup for computations",
                "priority_score": 7,
            },
            {
                "area": "Data structure optimization",
                "description": "Optimize data structures for cache efficiency",
                "impact": "medium",
                "effort": "high",
                "estimated_improvement": "15-30% performance improvement",
                "priority_score": 5,
            },
        ]

        # Sort by priority score
        optimizations.sort(key=lambda x: x["priority_score"], reverse=True)

        self.metrics["optimization_opportunities"] = optimizations

    def generate_baseline_report(self) -> str:
        """Generate comprehensive baseline performance report."""
        logger.info("Generating baseline performance report...")

        report_data = {
            "report_metadata": {
                "generated_at": time.time(),
                "duration": time.time() - self.start_time,
                "homodyne_available": HOMODYNE_AVAILABLE,
                "numba_available": NUMBA_AVAILABLE,
                "cvxpy_available": CVXPY_AVAILABLE,
                "robust_available": ROBUST_AVAILABLE,
            },
            "system_info": self.metrics["system_info"],
            "performance_metrics": self.metrics["baseline_metrics"],
            "bottlenecks": self.metrics["bottlenecks"],
            "optimization_opportunities": self.metrics["optimization_opportunities"],
            "memory_analysis": self.metrics["memory_analysis"],
            "cpu_analysis": self.metrics["cpu_analysis"],
            "test_performance": self.metrics["test_performance"],
        }

        # Save JSON report
        report_file = self.output_dir / f"performance_baseline_{int(time.time())}.json"
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        # Generate human-readable summary
        summary = self._generate_summary_report(report_data)

        # Save summary
        summary_file = self.output_dir / f"performance_summary_{int(time.time())}.txt"
        with open(summary_file, "w") as f:
            f.write(summary)

        logger.info(f"Reports saved to {self.output_dir}/")

        return summary

    def _generate_summary_report(self, data: dict) -> str:
        """Generate human-readable summary report."""
        summary = []
        summary.append("=" * 80)
        summary.append("HOMODYNE ANALYSIS PERFORMANCE BASELINE REPORT")
        summary.append("=" * 80)
        summary.append("")

        # System information
        summary.append("SYSTEM INFORMATION")
        summary.append("-" * 40)
        sys_info = data["system_info"]
        summary.append(
            f"Platform: {sys_info.get('platform', {}).get('system', 'Unknown')}"
        )
        summary.append(
            f"CPU Count: {sys_info.get('hardware', {}).get('cpu_count', 'Unknown')}"
        )
        summary.append(
            f"Memory: {sys_info.get('hardware', {}).get('memory_total', 0) / 1024**3:.1f} GB"
        )
        summary.append("")

        # Performance metrics
        summary.append("PERFORMANCE METRICS")
        summary.append("-" * 40)
        metrics = data["performance_metrics"]
        for operation, metric in metrics.items():
            if isinstance(metric, dict) and "duration" in metric:
                summary.append(f"{operation}: {metric['duration']:.3f}s")
            elif operation == "import_times":
                summary.append("Import times:")
                for module, time_val in metric.items():
                    if isinstance(time_val, (int, float)):
                        summary.append(f"  {module}: {time_val:.3f}s")
        summary.append("")

        # Bottlenecks
        summary.append("IDENTIFIED BOTTLENECKS")
        summary.append("-" * 40)
        bottlenecks = data["bottlenecks"]
        if bottlenecks:
            for bottleneck in bottlenecks:
                summary.append(
                    f"• {bottleneck.get('operation', 'Unknown')}: {bottleneck.get('issue', 'Unknown issue')}"
                )
                summary.append(f"  Severity: {bottleneck.get('severity', 'Unknown')}")
                if "optimization_opportunity" in bottleneck:
                    summary.append(
                        f"  Opportunity: {bottleneck['optimization_opportunity']}"
                    )
                summary.append("")
        else:
            summary.append("No significant bottlenecks identified.")
            summary.append("")

        # Top optimization opportunities
        summary.append("TOP OPTIMIZATION OPPORTUNITIES")
        summary.append("-" * 40)
        opportunities = data["optimization_opportunities"]
        for i, opp in enumerate(opportunities[:5], 1):
            summary.append(f"{i}. {opp['area']} (Priority: {opp['priority_score']}/10)")
            summary.append(f"   Description: {opp['description']}")
            summary.append(f"   Impact: {opp['impact']}, Effort: {opp['effort']}")
            summary.append(f"   Estimated improvement: {opp['estimated_improvement']}")
            summary.append("")

        # Summary statistics
        summary.append("SUMMARY STATISTICS")
        summary.append("-" * 40)
        report_meta = data["report_metadata"]
        summary.append(f"Analysis duration: {report_meta['duration']:.1f}s")
        summary.append(
            f"Components available: Homodyne={report_meta['homodyne_available']}, "
            f"Numba={report_meta['numba_available']}, CVXPY={report_meta['cvxpy_available']}, "
            f"Robust={report_meta['robust_available']}"
        )

        test_perf = data.get("test_performance", {})
        if "duration" in test_perf:
            summary.append(
                f"Test suite execution: {test_perf['duration']:.1f}s "
                f"({'PASSED' if test_perf['success'] else 'FAILED'})"
            )

        summary.append("")
        summary.append("=" * 80)

        return "\n".join(summary)

    # Helper methods for testing different scenarios
    def _create_test_config(self) -> dict[str, Any]:
        """Create minimal test configuration."""
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
                "num_threads": 2,
            },
            "optimization_config": {
                "classical_optimization": {"methods": ["nelder_mead"]}
            },
        }

    def _generate_synthetic_data(
        self, core, params: np.ndarray, angles: np.ndarray
    ) -> np.ndarray:
        """Generate synthetic test data."""
        try:
            # Use the core to generate theoretical data
            theoretical = core.calculate_c2_nonequilibrium_laminar_parallel(
                params, angles
            )
            # Add some noise
            noise = np.random.normal(0, 0.01, theoretical.shape)
            return theoretical + noise
        except Exception:
            # Fallback to simple synthetic data
            n_angles = len(angles)
            n_times = 50
            return np.random.exponential(1.0, (n_angles, n_times, n_times))

    def _test_numpy_operations(self):
        """Test numpy operations performance."""
        size = 1000
        a = np.random.random((size, size))
        b = np.random.random((size, size))
        c = np.dot(a, b)
        return c.sum()

    def _test_data_loading(self):
        """Simulate data loading operations."""
        # Simulate loading large datasets
        data = np.random.random((1000, 100, 100))
        return data.mean()

    def _test_homodyne_core(self):
        """Test homodyne core creation."""
        if HOMODYNE_AVAILABLE:
            config = self._create_test_config()
            config_manager = ConfigManager()
            config_manager.config = config
            core = HomodyneAnalysisCore(config_manager)
            return core
        return None

    def _test_computation_intensive(self):
        """Test computation-intensive operations."""
        # Matrix operations that simulate scientific computing
        size = 500
        a = np.random.random((size, size))
        b = np.linalg.inv(a + np.eye(size) * 0.1)  # Add regularization
        c = np.linalg.eigvals(b)
        return c.sum()

    def _test_cpu_intensive(self):
        """Test CPU-intensive operations."""
        # Pure Python computation
        result = sum(i * i for i in range(100000))
        return result

    def _test_scientific_computing(self):
        """Test scientific computing workload."""
        # Typical scientific operations
        x = np.linspace(0, 10, 10000)
        y = np.sin(x) * np.exp(-x / 10)
        z = np.fft.fft(y)
        return np.abs(z).sum()


def main():
    """Main entry point for performance baseline analysis."""
    parser = argparse.ArgumentParser(
        description="Homodyne Analysis Performance Baseline Tool"
    )

    parser.add_argument(
        "--profile-mode",
        choices=["quick", "comprehensive", "memory", "cpu"],
        default="comprehensive",
        help="Profiling mode (default: comprehensive)",
    )

    parser.add_argument(
        "--output-dir",
        default="performance_reports",
        help="Output directory for reports (default: performance_reports)",
    )

    parser.add_argument(
        "--quick-analysis", action="store_true", help="Run quick analysis only"
    )

    parser.add_argument(
        "--memory-analysis", action="store_true", help="Focus on memory analysis"
    )

    parser.add_argument(
        "--skip-tests", action="store_true", help="Skip running the test suite"
    )

    args = parser.parse_args()

    # Initialize profiler
    profiler = PerformanceProfiler(args.output_dir)

    print("Starting Homodyne Analysis Performance Baseline...")
    print(f"Mode: {args.profile_mode}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 60)

    try:
        # Always run basic profiling
        profiler.profile_import_performance()

        if args.quick_analysis or args.profile_mode == "quick":
            # Quick analysis mode
            profiler.profile_computation_performance()
            if not args.skip_tests:
                profiler.run_performance_tests()

        elif args.memory_analysis or args.profile_mode == "memory":
            # Memory-focused analysis
            profiler.profile_memory_usage()
            profiler.profile_computation_performance()

        elif args.profile_mode == "cpu":
            # CPU-focused analysis
            profiler.profile_cpu_usage()
            profiler.profile_computation_performance()

        else:  # comprehensive
            # Full comprehensive analysis
            profiler.profile_computation_performance()
            profiler.profile_optimization_performance()
            profiler.profile_memory_usage()
            profiler.profile_cpu_usage()

            if not args.skip_tests:
                profiler.run_performance_tests()

        # Always identify bottlenecks and prioritize optimizations
        profiler.identify_bottlenecks()
        profiler.prioritize_optimizations()

        # Generate final report
        summary = profiler.generate_baseline_report()

        print("\n" + "=" * 80)
        print("PERFORMANCE BASELINE ANALYSIS COMPLETE")
        print("=" * 80)
        print(summary)

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError during analysis: {e}")
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
