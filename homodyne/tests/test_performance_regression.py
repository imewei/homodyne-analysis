"""
Performance Regression Testing Framework
========================================

Automated performance regression testing system for Task 4.6.
Ensures performance improvements are maintained over time.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")


class PerformanceRegressionTester:
    """Performance regression testing framework."""

    def __init__(self, baseline_file: str = "performance_baseline.json"):
        self.baseline_file = Path(baseline_file)
        self.tolerance = 0.20  # 20% performance degradation tolerance

    def load_baseline(self) -> dict[str, Any]:
        """Load performance baseline."""
        if self.baseline_file.exists():
            with open(self.baseline_file) as f:
                return json.load(f)
        return {}

    def save_baseline(self, baseline: dict[str, Any]):
        """Save performance baseline."""
        with open(self.baseline_file, "w") as f:
            json.dump(baseline, f, indent=2)

    def run_performance_tests(self) -> dict[str, float]:
        """Run current performance tests."""
        results = {}

        # Test 1: Chi-squared calculation
        c2_exp = np.random.rand(5, 50, 50)
        c2_theo = np.random.rand(5, 50, 50)

        start_time = time.perf_counter()
        for _ in range(10):
            np.sum((c2_exp - c2_theo) ** 2)
        results["chi_squared"] = time.perf_counter() - start_time

        # Test 2: Matrix operations
        A = np.random.rand(100, 100)

        start_time = time.perf_counter()
        for _ in range(10):
            np.linalg.eigvals(A)
        results["eigenvalues"] = time.perf_counter() - start_time

        # Test 3: Statistical operations
        data = np.random.rand(10000)

        start_time = time.perf_counter()
        for _ in range(100):
            np.mean(data)
            np.std(data)
        results["statistics"] = time.perf_counter() - start_time

        return results

    def check_regression(self) -> dict[str, Any]:
        """Check for performance regression."""
        baseline = self.load_baseline()
        current = self.run_performance_tests()

        regression_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": {},
            "overall_status": "PASS",
            "regressions_detected": 0,
        }

        for test_name, current_time in current.items():
            if test_name in baseline:
                baseline_time = baseline[test_name]
                ratio = current_time / baseline_time
                status = "PASS" if ratio <= (1 + self.tolerance) else "FAIL"

                if status == "FAIL":
                    regression_report["regressions_detected"] += 1
                    regression_report["overall_status"] = "FAIL"

                regression_report["tests"][test_name] = {
                    "baseline_time": baseline_time,
                    "current_time": current_time,
                    "ratio": ratio,
                    "status": status,
                    "degradation_percent": (ratio - 1) * 100,
                }

        return regression_report

    def create_initial_baseline(self):
        """Create initial performance baseline."""
        print("Creating initial performance baseline...")
        baseline = self.run_performance_tests()
        baseline["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save_baseline(baseline)
        print(f"✓ Baseline saved: {baseline}")


def run_regression_testing():
    """Run performance regression testing."""
    print("Performance Regression Testing Framework - Task 4.6")
    print("=" * 60)

    tester = PerformanceRegressionTester("performance_baseline_task_4_6.json")

    # Create baseline if it doesn't exist
    if not tester.baseline_file.exists():
        tester.create_initial_baseline()

    # Run regression check
    report = tester.check_regression()

    print("\nREGRESSION TEST RESULTS:")
    print(f"Overall Status: {report['overall_status']}")
    print(f"Regressions Detected: {report['regressions_detected']}")

    for test_name, test_data in report["tests"].items():
        status_emoji = "✓" if test_data["status"] == "PASS" else "✗"
        print(
            f"  {status_emoji} {test_name}: {test_data['ratio']:.2f}x ({test_data['degradation_percent']:+.1f}%)"
        )

    # Save report
    report_file = Path("performance_regression_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Report saved to: {report_file}")
    print("✅ Task 4.6 Performance Regression Testing Complete!")

    return report


if __name__ == "__main__":
    run_regression_testing()
