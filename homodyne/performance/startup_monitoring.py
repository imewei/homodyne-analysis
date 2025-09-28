"""
Startup Performance Monitoring and Baseline Management
======================================================

Advanced monitoring system for tracking package startup performance, establishing
baselines, and detecting performance regressions over time.

Features:
- Continuous startup performance monitoring
- Baseline establishment and management
- Performance regression detection
- Historical performance tracking
- Alert system for performance degradation
- Integration with CI/CD pipelines

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import json
import logging
import os
import statistics
import subprocess
import sys
import time
import warnings
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class StartupMetrics:
    """Comprehensive startup performance metrics."""
    timestamp: str
    import_time: float
    memory_usage_mb: float
    python_version: str
    package_version: str
    platform: str
    cpu_count: int
    optimization_enabled: bool
    import_errors: List[str]
    dependency_load_times: Dict[str, float]
    total_modules_loaded: int
    lazy_modules_count: int
    immediate_modules_count: int


@dataclass
class PerformanceBaseline:
    """Performance baseline configuration."""
    name: str
    target_import_time: float
    max_memory_usage_mb: float
    acceptable_variance_percent: float
    measurement_count: int
    environment_tags: List[str]
    created_at: str
    updated_at: str


@dataclass
class RegressionAlert:
    """Performance regression alert."""
    alert_id: str
    metric_name: str
    current_value: float
    baseline_value: float
    degradation_percent: float
    severity: str  # 'warning', 'critical'
    timestamp: str
    recommendations: List[str]


class StartupPerformanceMonitor:
    """
    Advanced startup performance monitoring system.

    Provides comprehensive monitoring of package startup performance with
    baseline management, regression detection, and historical tracking.
    """

    def __init__(self, package_name: str = "homodyne",
                 baseline_dir: Optional[Path] = None):
        """
        Initialize startup performance monitor.

        Parameters
        ----------
        package_name : str
            Name of package to monitor
        baseline_dir : Optional[Path]
            Directory to store baseline data
        """
        self.package_name = package_name
        self.baseline_dir = baseline_dir or Path.home() / ".homodyne" / "performance"
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

        self.baseline_file = self.baseline_dir / "startup_baselines.json"
        self.metrics_file = self.baseline_dir / "startup_metrics.jsonl"
        self.alerts_file = self.baseline_dir / "performance_alerts.json"

        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.recent_metrics: deque = deque(maxlen=100)  # Keep last 100 measurements
        self.load_baselines()

    def measure_startup_performance(self,
                                  iterations: int = 5,
                                  warmup_iterations: int = 2) -> StartupMetrics:
        """
        Measure comprehensive startup performance.

        Parameters
        ----------
        iterations : int
            Number of measurement iterations
        warmup_iterations : int
            Number of warmup iterations (not measured)

        Returns
        -------
        StartupMetrics
            Comprehensive startup performance metrics
        """
        logger.info(f"Measuring startup performance ({iterations} iterations)")

        # Warmup runs
        for i in range(warmup_iterations):
            self._single_startup_measurement()

        # Measured runs
        measurements = []
        memory_measurements = []
        dependency_times = defaultdict(list)
        import_errors = []

        for i in range(iterations):
            try:
                metrics = self._single_startup_measurement(detailed=True)
                measurements.append(metrics['import_time'])
                memory_measurements.append(metrics['memory_usage'])

                # Aggregate dependency times
                for dep, time_val in metrics.get('dependency_times', {}).items():
                    dependency_times[dep].append(time_val)

                import_errors.extend(metrics.get('import_errors', []))

            except Exception as e:
                logger.warning(f"Measurement iteration {i} failed: {e}")
                import_errors.append(str(e))

        if not measurements:
            raise RuntimeError("No successful startup measurements collected")

        # Calculate aggregated metrics
        avg_import_time = statistics.mean(measurements)
        avg_memory = statistics.mean(memory_measurements)

        # Average dependency times
        avg_dependency_times = {
            dep: statistics.mean(times)
            for dep, times in dependency_times.items()
        }

        # System information
        import platform
        system_info = self._get_system_info()

        # Create comprehensive metrics
        metrics = StartupMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            import_time=avg_import_time,
            memory_usage_mb=avg_memory,
            python_version=system_info['python_version'],
            package_version=system_info['package_version'],
            platform=system_info['platform'],
            cpu_count=system_info['cpu_count'],
            optimization_enabled=system_info['optimization_enabled'],
            import_errors=list(set(import_errors)),  # Remove duplicates
            dependency_load_times=avg_dependency_times,
            total_modules_loaded=system_info.get('total_modules', 0),
            lazy_modules_count=system_info.get('lazy_modules', 0),
            immediate_modules_count=system_info.get('immediate_modules', 0),
        )

        # Store metrics
        self._store_metrics(metrics)
        self.recent_metrics.append(metrics)

        logger.info(f"Startup performance: {avg_import_time:.3f}s, {avg_memory:.1f}MB")
        return metrics

    def _single_startup_measurement(self, detailed: bool = False) -> Dict[str, Any]:
        """Perform a single startup time measurement."""
        import psutil
        import subprocess

        # Prepare measurement script
        measurement_script = f'''
import time
import sys
import os

# Disable optional heavy dependencies for consistent measurement
sys.modules["numba"] = None
sys.modules["pymc"] = None
sys.modules["arviz"] = None
sys.modules["corner"] = None

# Enable optimization for measurement
os.environ["HOMODYNE_OPTIMIZE_STARTUP"] = "true"

# Measure import time
start_time = time.perf_counter()
try:
    import {self.package_name}

    # Get detailed metrics if requested
    if {detailed}:
        try:
            import_report = {self.package_name}.get_import_performance_report()
            startup_report = {self.package_name}.get_startup_performance_report()

            print("DETAILED_METRICS_START")
            import json
            print(json.dumps({{
                "import_report": import_report,
                "startup_report": startup_report
            }}))
            print("DETAILED_METRICS_END")
        except Exception as e:
            print(f"DETAILED_ERROR: {{e}}")

    end_time = time.perf_counter()
    print(f"IMPORT_TIME: {{end_time - start_time}}")

except Exception as e:
    print(f"IMPORT_ERROR: {{e}}")
    sys.exit(1)
'''

        # Measure memory before subprocess
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Run measurement in subprocess
        start_time = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "-c", measurement_script],
            capture_output=True,
            text=True,
            timeout=30  # 30-second timeout
        )
        end_time = time.perf_counter()

        if result.returncode != 0:
            raise RuntimeError(f"Import failed: {result.stderr}")

        # Parse results
        output_lines = result.stdout.strip().split('\n')
        import_time = None
        detailed_metrics = {}
        import_errors = []

        for line in output_lines:
            if line.startswith("IMPORT_TIME:"):
                import_time = float(line.split(":", 1)[1].strip())
            elif line.startswith("IMPORT_ERROR:"):
                import_errors.append(line.split(":", 1)[1].strip())
            elif line.startswith("DETAILED_ERROR:"):
                import_errors.append(line.split(":", 1)[1].strip())

        # Parse detailed metrics if available
        if detailed and "DETAILED_METRICS_START" in result.stdout:
            try:
                start_idx = result.stdout.find("DETAILED_METRICS_START") + len("DETAILED_METRICS_START\n")
                end_idx = result.stdout.find("DETAILED_METRICS_END")
                detailed_json = result.stdout[start_idx:end_idx].strip()
                detailed_metrics = json.loads(detailed_json)
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug(f"Failed to parse detailed metrics: {e}")

        if import_time is None:
            raise RuntimeError("Failed to measure import time")

        # Estimate memory usage (rough approximation)
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_usage = max(0, final_memory - initial_memory + 50)  # Add base estimate

        return {
            "import_time": import_time,
            "memory_usage": memory_usage,
            "dependency_times": detailed_metrics.get("import_report", {}).get("individual_imports", {}),
            "import_errors": import_errors,
            "detailed_metrics": detailed_metrics,
        }

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for metrics."""
        import platform

        try:
            # Get package version
            import importlib.metadata
            package_version = importlib.metadata.version(self.package_name)
        except Exception:
            package_version = "unknown"

        # Check if optimization is enabled
        optimization_enabled = os.environ.get("HOMODYNE_OPTIMIZE_STARTUP", "true").lower() in ("true", "1", "yes")

        return {
            "python_version": platform.python_version(),
            "package_version": package_version,
            "platform": f"{platform.system()} {platform.machine()}",
            "cpu_count": os.cpu_count() or 1,
            "optimization_enabled": optimization_enabled,
            "total_modules": len(sys.modules),
            "lazy_modules": 0,  # Would need actual count from lazy loader
            "immediate_modules": 0,  # Would need actual count
        }

    def establish_baseline(self,
                          name: str,
                          target_import_time: float,
                          max_memory_usage_mb: float,
                          acceptable_variance_percent: float = 10.0,
                          measurement_count: int = 10,
                          environment_tags: Optional[List[str]] = None) -> PerformanceBaseline:
        """
        Establish a new performance baseline.

        Parameters
        ----------
        name : str
            Baseline name
        target_import_time : float
            Target import time in seconds
        max_memory_usage_mb : float
            Maximum acceptable memory usage in MB
        acceptable_variance_percent : float
            Acceptable variance from target (%)
        measurement_count : int
            Number of measurements for baseline
        environment_tags : Optional[List[str]]
            Environment tags for this baseline

        Returns
        -------
        PerformanceBaseline
            Created baseline configuration
        """
        logger.info(f"Establishing performance baseline: {name}")

        # Validate current performance against targets
        current_metrics = self.measure_startup_performance(iterations=measurement_count)

        if current_metrics.import_time > target_import_time * (1 + acceptable_variance_percent / 100):
            warnings.warn(
                f"Current import time ({current_metrics.import_time:.3f}s) exceeds target "
                f"({target_import_time:.3f}s) by more than acceptable variance"
            )

        if current_metrics.memory_usage_mb > max_memory_usage_mb:
            warnings.warn(
                f"Current memory usage ({current_metrics.memory_usage_mb:.1f}MB) exceeds "
                f"maximum ({max_memory_usage_mb:.1f}MB)"
            )

        # Create baseline
        baseline = PerformanceBaseline(
            name=name,
            target_import_time=target_import_time,
            max_memory_usage_mb=max_memory_usage_mb,
            acceptable_variance_percent=acceptable_variance_percent,
            measurement_count=measurement_count,
            environment_tags=environment_tags or [],
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

        # Store baseline
        self.baselines[name] = baseline
        self.save_baselines()

        logger.info(f"Baseline '{name}' established successfully")
        return baseline

    def check_performance_regression(self,
                                   baseline_name: str,
                                   current_metrics: Optional[StartupMetrics] = None) -> List[RegressionAlert]:
        """
        Check for performance regressions against baseline.

        Parameters
        ----------
        baseline_name : str
            Name of baseline to check against
        current_metrics : Optional[StartupMetrics]
            Current metrics (measured if not provided)

        Returns
        -------
        List[RegressionAlert]
            List of performance regression alerts
        """
        if baseline_name not in self.baselines:
            raise ValueError(f"Baseline '{baseline_name}' not found")

        baseline = self.baselines[baseline_name]

        if current_metrics is None:
            current_metrics = self.measure_startup_performance()

        alerts = []

        # Check import time regression
        import_time_threshold = baseline.target_import_time * (1 + baseline.acceptable_variance_percent / 100)
        if current_metrics.import_time > import_time_threshold:
            degradation = ((current_metrics.import_time - baseline.target_import_time) / baseline.target_import_time) * 100
            severity = "critical" if degradation > baseline.acceptable_variance_percent * 2 else "warning"

            alerts.append(RegressionAlert(
                alert_id=f"import_time_{int(time.time())}",
                metric_name="import_time",
                current_value=current_metrics.import_time,
                baseline_value=baseline.target_import_time,
                degradation_percent=degradation,
                severity=severity,
                timestamp=datetime.now(timezone.utc).isoformat(),
                recommendations=[
                    "Review recent changes to import structure",
                    "Check for new heavy dependencies",
                    "Verify lazy loading is working correctly",
                    "Consider profiling import bottlenecks",
                ]
            ))

        # Check memory usage regression
        if current_metrics.memory_usage_mb > baseline.max_memory_usage_mb:
            degradation = ((current_metrics.memory_usage_mb - baseline.max_memory_usage_mb) / baseline.max_memory_usage_mb) * 100
            severity = "critical" if degradation > 50 else "warning"

            alerts.append(RegressionAlert(
                alert_id=f"memory_usage_{int(time.time())}",
                metric_name="memory_usage",
                current_value=current_metrics.memory_usage_mb,
                baseline_value=baseline.max_memory_usage_mb,
                degradation_percent=degradation,
                severity=severity,
                timestamp=datetime.now(timezone.utc).isoformat(),
                recommendations=[
                    "Review memory-intensive imports",
                    "Check for memory leaks in initialization",
                    "Optimize large data structures",
                    "Consider more aggressive lazy loading",
                ]
            ))

        # Store alerts if any
        if alerts:
            self._store_alerts(alerts)
            logger.warning(f"Performance regression detected: {len(alerts)} alerts")

        return alerts

    def get_performance_trend(self, days: int = 30) -> Dict[str, Any]:
        """
        Get performance trend over specified time period.

        Parameters
        ----------
        days : int
            Number of days to analyze

        Returns
        -------
        Dict[str, Any]
            Performance trend analysis
        """
        # Load historical metrics
        historical_metrics = self._load_historical_metrics(days)

        if len(historical_metrics) < 2:
            return {"error": "Insufficient historical data for trend analysis"}

        # Analyze trends
        import_times = [m.import_time for m in historical_metrics]
        memory_usage = [m.memory_usage_mb for m in historical_metrics]
        timestamps = [datetime.fromisoformat(m.timestamp.replace('Z', '+00:00')) for m in historical_metrics]

        # Calculate trend statistics
        trend_analysis = {
            "period_days": days,
            "total_measurements": len(historical_metrics),
            "import_time_trend": {
                "min": min(import_times),
                "max": max(import_times),
                "mean": statistics.mean(import_times),
                "median": statistics.median(import_times),
                "std_dev": statistics.stdev(import_times) if len(import_times) > 1 else 0,
                "latest": import_times[-1],
                "trend_direction": "improving" if import_times[-1] < import_times[0] else "degrading",
            },
            "memory_usage_trend": {
                "min": min(memory_usage),
                "max": max(memory_usage),
                "mean": statistics.mean(memory_usage),
                "median": statistics.median(memory_usage),
                "std_dev": statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0,
                "latest": memory_usage[-1],
                "trend_direction": "improving" if memory_usage[-1] < memory_usage[0] else "degrading",
            },
            "measurements_per_day": len(historical_metrics) / days,
            "first_measurement": timestamps[0].isoformat(),
            "last_measurement": timestamps[-1].isoformat(),
        }

        return trend_analysis

    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Returns
        -------
        Dict[str, Any]
            Comprehensive performance report
        """
        current_metrics = self.measure_startup_performance()

        # Check all baselines for regressions
        all_alerts = []
        baseline_status = {}

        for baseline_name in self.baselines:
            try:
                alerts = self.check_performance_regression(baseline_name, current_metrics)
                all_alerts.extend(alerts)
                baseline_status[baseline_name] = {
                    "status": "failed" if alerts else "passed",
                    "alerts_count": len(alerts),
                }
            except Exception as e:
                baseline_status[baseline_name] = {
                    "status": "error",
                    "error": str(e),
                }

        # Get performance trend
        trend_analysis = self.get_performance_trend()

        # Create comprehensive report
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "package_name": self.package_name,
            "current_metrics": asdict(current_metrics),
            "baseline_status": baseline_status,
            "alerts": [asdict(alert) for alert in all_alerts],
            "trend_analysis": trend_analysis,
            "baselines": {name: asdict(baseline) for name, baseline in self.baselines.items()},
            "summary": {
                "total_baselines": len(self.baselines),
                "passing_baselines": sum(1 for status in baseline_status.values() if status.get("status") == "passed"),
                "failing_baselines": sum(1 for status in baseline_status.values() if status.get("status") == "failed"),
                "total_alerts": len(all_alerts),
                "critical_alerts": sum(1 for alert in all_alerts if alert.severity == "critical"),
                "current_performance_rating": self._calculate_performance_rating(current_metrics, all_alerts),
            }
        }

        return report

    def _calculate_performance_rating(self,
                                    metrics: StartupMetrics,
                                    alerts: List[RegressionAlert]) -> str:
        """Calculate overall performance rating."""
        if any(alert.severity == "critical" for alert in alerts):
            return "poor"
        elif any(alert.severity == "warning" for alert in alerts):
            return "fair"
        elif metrics.import_time < 1.0 and metrics.memory_usage_mb < 100:
            return "excellent"
        elif metrics.import_time < 2.0 and metrics.memory_usage_mb < 200:
            return "good"
        else:
            return "fair"

    def _store_metrics(self, metrics: StartupMetrics) -> None:
        """Store metrics to persistent storage."""
        try:
            with open(self.metrics_file, 'a', encoding='utf-8') as f:
                json.dump(asdict(metrics), f)
                f.write('\n')
        except Exception as e:
            logger.warning(f"Failed to store metrics: {e}")

    def _store_alerts(self, alerts: List[RegressionAlert]) -> None:
        """Store alerts to persistent storage."""
        try:
            # Load existing alerts
            existing_alerts = []
            if self.alerts_file.exists():
                with open(self.alerts_file, 'r', encoding='utf-8') as f:
                    existing_alerts = json.load(f)

            # Add new alerts
            existing_alerts.extend([asdict(alert) for alert in alerts])

            # Keep only recent alerts (last 1000)
            existing_alerts = existing_alerts[-1000:]

            # Save back
            with open(self.alerts_file, 'w', encoding='utf-8') as f:
                json.dump(existing_alerts, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to store alerts: {e}")

    def _load_historical_metrics(self, days: int) -> List[StartupMetrics]:
        """Load historical metrics from storage."""
        try:
            if not self.metrics_file.exists():
                return []

            cutoff_time = datetime.now(timezone.utc).timestamp() - (days * 24 * 3600)
            metrics = []

            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))

                        if timestamp.timestamp() >= cutoff_time:
                            metrics.append(StartupMetrics(**data))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue

            return sorted(metrics, key=lambda m: m.timestamp)

        except Exception as e:
            logger.warning(f"Failed to load historical metrics: {e}")
            return []

    def load_baselines(self) -> None:
        """Load baselines from persistent storage."""
        try:
            if self.baseline_file.exists():
                with open(self.baseline_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.baselines = {
                        name: PerformanceBaseline(**baseline_data)
                        for name, baseline_data in data.items()
                    }
        except Exception as e:
            logger.warning(f"Failed to load baselines: {e}")
            self.baselines = {}

    def save_baselines(self) -> None:
        """Save baselines to persistent storage."""
        try:
            data = {name: asdict(baseline) for name, baseline in self.baselines.items()}
            with open(self.baseline_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save baselines: {e}")


# Global monitor instance
_global_monitor = None


def get_startup_monitor() -> StartupPerformanceMonitor:
    """Get global startup performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = StartupPerformanceMonitor()
    return _global_monitor


@contextmanager
def monitor_startup_performance():
    """Context manager for monitoring startup performance."""
    monitor = get_startup_monitor()
    start_time = time.perf_counter()

    try:
        yield monitor
    finally:
        end_time = time.perf_counter()
        logger.info(f"Startup monitoring completed in {end_time - start_time:.3f}s")


def establish_default_baselines() -> Dict[str, PerformanceBaseline]:
    """
    Establish default performance baselines.

    Returns
    -------
    Dict[str, PerformanceBaseline]
        Created default baselines
    """
    monitor = get_startup_monitor()

    # Default baselines for different environments
    baselines = {}

    # Development baseline (more lenient)
    baselines["development"] = monitor.establish_baseline(
        name="development",
        target_import_time=2.0,
        max_memory_usage_mb=200.0,
        acceptable_variance_percent=20.0,
        measurement_count=5,
        environment_tags=["development", "local"]
    )

    # Production baseline (strict)
    baselines["production"] = monitor.establish_baseline(
        name="production",
        target_import_time=1.5,
        max_memory_usage_mb=150.0,
        acceptable_variance_percent=10.0,
        measurement_count=10,
        environment_tags=["production", "optimized"]
    )

    # CI/CD baseline (for automated testing)
    baselines["ci"] = monitor.establish_baseline(
        name="ci",
        target_import_time=3.0,  # More lenient for CI environments
        max_memory_usage_mb=250.0,
        acceptable_variance_percent=25.0,
        measurement_count=3,
        environment_tags=["ci", "automated"]
    )

    logger.info(f"Established {len(baselines)} default baselines")
    return baselines


def check_startup_health() -> Dict[str, Any]:
    """
    Quick startup health check.

    Returns
    -------
    Dict[str, Any]
        Startup health status
    """
    monitor = get_startup_monitor()

    try:
        # Quick measurement (fewer iterations)
        metrics = monitor.measure_startup_performance(iterations=3)

        # Simple health assessment
        health_status = "healthy"
        issues = []

        if metrics.import_time > 3.0:
            health_status = "unhealthy"
            issues.append(f"Slow import time: {metrics.import_time:.3f}s")

        if metrics.memory_usage_mb > 300:
            health_status = "unhealthy"
            issues.append(f"High memory usage: {metrics.memory_usage_mb:.1f}MB")

        if metrics.import_errors:
            health_status = "unhealthy"
            issues.extend(f"Import error: {error}" for error in metrics.import_errors)

        return {
            "status": health_status,
            "import_time": metrics.import_time,
            "memory_usage_mb": metrics.memory_usage_mb,
            "optimization_enabled": metrics.optimization_enabled,
            "package_version": metrics.package_version,
            "issues": issues,
            "timestamp": metrics.timestamp,
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }