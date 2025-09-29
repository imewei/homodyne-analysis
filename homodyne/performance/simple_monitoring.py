"""
Simplified Startup Performance Monitoring
=========================================

Simplified version of startup performance monitoring focused on core functionality
without complex dependencies or baseline management that might have issues.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SimpleStartupMetrics:
    """Simple startup performance metrics."""

    timestamp: str
    import_time: float
    package_version: str
    python_version: str
    optimization_enabled: bool
    measurement_iterations: int


class SimpleStartupMonitor:
    """
    Simplified startup performance monitor.

    Focuses on essential monitoring without complex baseline management.
    """

    def __init__(self, package_name: str = "homodyne"):
        self.package_name = package_name

    def measure_startup_time(self, iterations: int = 3) -> SimpleStartupMetrics:
        """
        Measure package startup time.

        Parameters
        ----------
        iterations : int
            Number of measurement iterations

        Returns
        -------
        SimpleStartupMetrics
            Simple startup metrics
        """
        logger.info(f"Measuring startup time ({iterations} iterations)")

        times = []
        for i in range(iterations):
            try:
                start_time = time.perf_counter()
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        f"""
import sys
import os
sys.modules["numba"] = None
sys.modules["pymc"] = None
sys.modules["arviz"] = None
sys.modules["corner"] = None
os.environ["HOMODYNE_OPTIMIZE_STARTUP"] = "true"
import time
start = time.perf_counter()
import {self.package_name}
end = time.perf_counter()
print(f"IMPORT_TIME:{{end - start:.6f}}")
                    """,
                    ],
                    check=False, capture_output=True,
                    text=True,
                    timeout=30,
                )
                end_time = time.perf_counter()

                if result.returncode == 0:
                    # Parse import time from output
                    for line in result.stdout.split("\n"):
                        if line.startswith("IMPORT_TIME:"):
                            import_time = float(line.split(":")[1])
                            times.append(import_time)
                            break
                else:
                    logger.warning(f"Measurement {i} failed: {result.stderr}")

            except Exception as e:
                logger.warning(f"Measurement {i} failed: {e}")

        if not times:
            raise RuntimeError("No successful measurements collected")

        avg_time = sum(times) / len(times)

        # Get system information
        try:
            import importlib.metadata

            package_version = importlib.metadata.version(self.package_name)
        except Exception:
            package_version = "unknown"

        import platform

        python_version = platform.python_version()

        optimization_enabled = os.environ.get(
            "HOMODYNE_OPTIMIZE_STARTUP", "true"
        ).lower() in ("true", "1", "yes")

        metrics = SimpleStartupMetrics(
            timestamp=datetime.now(UTC).isoformat(),
            import_time=avg_time,
            package_version=package_version,
            python_version=python_version,
            optimization_enabled=optimization_enabled,
            measurement_iterations=len(times),
        )

        logger.info(f"Startup time: {avg_time:.3f}s (avg of {len(times)} measurements)")
        return metrics

    def check_startup_health(self) -> dict[str, Any]:
        """
        Simple startup health check.

        Returns
        -------
        Dict[str, Any]
            Health status information
        """
        try:
            metrics = self.measure_startup_time(iterations=2)

            # Simple health assessment
            if metrics.import_time < 1.0:
                status = "excellent"
            elif metrics.import_time < 2.0:
                status = "good"
            elif metrics.import_time < 3.0:
                status = "fair"
            else:
                status = "poor"

            return {
                "status": status,
                "import_time": metrics.import_time,
                "package_version": metrics.package_version,
                "python_version": metrics.python_version,
                "optimization_enabled": metrics.optimization_enabled,
                "timestamp": metrics.timestamp,
                "assessment": self._get_performance_assessment(metrics.import_time),
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    def _get_performance_assessment(self, import_time: float) -> str:
        """Get performance assessment message."""
        if import_time < 1.0:
            return "Excellent startup performance"
        if import_time < 2.0:
            return "Good startup performance"
        if import_time < 3.0:
            return "Fair startup performance - consider optimization"
        return "Poor startup performance - optimization needed"

    def create_simple_baseline(self, name: str, target_time: float) -> dict[str, Any]:
        """
        Create a simple performance baseline.

        Parameters
        ----------
        name : str
            Baseline name
        target_time : float
            Target import time in seconds

        Returns
        -------
        Dict[str, Any]
            Simple baseline information
        """
        current_metrics = self.measure_startup_time()

        baseline = {
            "name": name,
            "target_time": target_time,
            "current_time": current_metrics.import_time,
            "meets_target": current_metrics.import_time <= target_time,
            "created_at": current_metrics.timestamp,
            "package_version": current_metrics.package_version,
        }

        logger.info(
            f"Baseline '{name}' created: target {target_time}s, current {current_metrics.import_time:.3f}s"
        )
        return baseline

    def generate_simple_report(self) -> dict[str, Any]:
        """
        Generate a simple performance report.

        Returns
        -------
        Dict[str, Any]
            Simple performance report
        """
        health = self.check_startup_health()
        metrics = self.measure_startup_time(iterations=5)

        return {
            "report_generated_at": datetime.now(UTC).isoformat(),
            "package_name": self.package_name,
            "health_status": health["status"],
            "current_performance": {
                "import_time": metrics.import_time,
                "measurement_iterations": metrics.measurement_iterations,
                "optimization_enabled": metrics.optimization_enabled,
            },
            "assessment": health.get("assessment", "Unknown"),
            "recommendations": self._get_recommendations(metrics.import_time),
            "system_info": {
                "package_version": metrics.package_version,
                "python_version": metrics.python_version,
                "timestamp": metrics.timestamp,
            },
        }

    def _get_recommendations(self, import_time: float) -> list[str]:
        """Get performance recommendations."""
        recommendations = []

        if import_time > 3.0:
            recommendations.extend(
                [
                    "Consider enabling optimization with HOMODYNE_OPTIMIZE_STARTUP=true",
                    "Review import structure for heavy dependencies",
                    "Check for circular imports or unnecessary eager loading",
                ]
            )
        elif import_time > 2.0:
            recommendations.extend(
                [
                    "Monitor performance trends",
                    "Consider lazy loading for non-critical components",
                ]
            )
        else:
            recommendations.append("Performance is within acceptable range")

        return recommendations


# Global simple monitor instance
_simple_monitor = None


def get_simple_monitor() -> SimpleStartupMonitor:
    """Get global simple monitor instance."""
    global _simple_monitor
    if _simple_monitor is None:
        _simple_monitor = SimpleStartupMonitor()
    return _simple_monitor


def quick_startup_check() -> dict[str, Any]:
    """
    Quick startup performance check.

    Returns
    -------
    Dict[str, Any]
        Quick performance assessment
    """
    monitor = get_simple_monitor()
    return monitor.check_startup_health()


def measure_current_startup_performance(iterations: int = 3) -> dict[str, Any]:
    """
    Measure current startup performance.

    Parameters
    ----------
    iterations : int
        Number of measurement iterations

    Returns
    -------
    Dict[str, Any]
        Current performance metrics
    """
    monitor = get_simple_monitor()
    metrics = monitor.measure_startup_time(iterations)
    return asdict(metrics)


def create_performance_baseline(name: str, target_time: float = 2.0) -> dict[str, Any]:
    """
    Create a simple performance baseline.

    Parameters
    ----------
    name : str
        Baseline name
    target_time : float
        Target import time in seconds

    Returns
    -------
    Dict[str, Any]
        Created baseline information
    """
    monitor = get_simple_monitor()
    return monitor.create_simple_baseline(name, target_time)
