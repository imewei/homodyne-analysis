#!/usr/bin/env python3
"""
Advanced Test Runner for Homodyne Analysis Package
==================================================

Provides comprehensive testing capabilities with performance monitoring,
security validation, and scientific computing accuracy verification.
"""

import argparse
import os
import sys

import pytest


class TestRunner:
    """Advanced test runner with comprehensive reporting."""

    def __init__(self):
        self.start_time = None
        self.results = {}
        self.failed_tests = []
        self.performance_data = {}

    def run_unit_tests(self, verbose: bool = False, coverage: bool = True) -> int:
        """Run unit tests with coverage reporting."""
        print("🧪 Running Unit Tests...")

        args = [
            "homodyne/tests/test_core_kernels.py",
            "homodyne/tests/test_analysis_core.py",
            "homodyne/tests/test_config_management.py",
            "-v" if verbose else "-q",
            "--tb=short",
        ]

        if coverage:
            args.extend([
                "--cov=homodyne.core",
                "--cov=homodyne.analysis",
                "--cov=homodyne.config",
                "--cov-report=term-missing",
            ])

        return pytest.main(args)

    def run_integration_tests(self, verbose: bool = False) -> int:
        """Run integration tests."""
        print("🔧 Running Integration Tests...")

        args = [
            "homodyne/tests/test_cli_integration.py",
            "-v" if verbose else "-q",
            "--tb=short",
            "-m", "integration",
        ]

        return pytest.main(args)

    def run_performance_tests(self, verbose: bool = False, benchmark: bool = True) -> int:
        """Run performance tests and benchmarks."""
        print("⚡ Running Performance Tests...")

        args = [
            "homodyne/tests/test_optimization_performance.py",
            "-v" if verbose else "-q",
            "--tb=short",
            "-m", "performance",
        ]

        if benchmark:
            args.extend(["--benchmark-only", "--benchmark-sort=mean"])

        return pytest.main(args)

    def run_security_tests(self, verbose: bool = False) -> int:
        """Run security and validation tests."""
        print("🔒 Running Security Tests...")

        args = [
            "homodyne/tests/test_security_validation.py",
            "-v" if verbose else "-q",
            "--tb=short",
            "-m", "security",
        ]

        return pytest.main(args)

    def run_scientific_tests(self, verbose: bool = False) -> int:
        """Run scientific computing validation tests."""
        print("🔬 Running Scientific Validation Tests...")

        args = [
            "homodyne/tests/test_scientific_validation.py",
            "-v" if verbose else "-q",
            "--tb=short",
            "-m", "scientific",
        ]

        return pytest.main(args)

    def run_all_tests(self,
                     verbose: bool = False,
                     coverage: bool = True,
                     parallel: bool = False,
                     markers: list[str] | None = None) -> int:
        """Run comprehensive test suite."""
        print("🚀 Running Comprehensive Test Suite...")

        args = [
            "homodyne/tests/",
            "-v" if verbose else "-q",
            "--tb=short",
            "--durations=10",
            "--maxfail=5",
        ]

        if coverage:
            args.extend([
                "--cov=homodyne",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml:coverage.xml",
                "--cov-fail-under=75",
            ])

        if parallel:
            args.extend(["-n", "auto"])

        if markers:
            marker_expr = " or ".join(markers)
            args.extend(["-m", marker_expr])

        return pytest.main(args)

    def run_fast_tests(self, verbose: bool = False) -> int:
        """Run only fast tests (excluding slow and performance tests)."""
        print("🏃 Running Fast Tests Only...")

        args = [
            "homodyne/tests/",
            "-v" if verbose else "-q",
            "--tb=short",
            "-m", "not slow and not performance",
            "--maxfail=3",
        ]

        return pytest.main(args)

    def run_regression_tests(self, verbose: bool = False) -> int:
        """Run regression tests against previous baselines."""
        print("📊 Running Regression Tests...")

        args = [
            "homodyne/tests/",
            "-v" if verbose else "-q",
            "--tb=short",
            "-m", "regression",
        ]

        return pytest.main(args)

    def run_smoke_tests(self, verbose: bool = False) -> int:
        """Run minimal smoke tests for basic functionality."""
        print("💨 Running Smoke Tests...")

        args = [
            "homodyne/tests/test_core_kernels.py::TestNumbaCoreKernels::test_sinc_squared_computation",
            "homodyne/tests/test_analysis_core.py::TestHomodyneAnalysisCore::test_initialization_with_config_dict",
            "homodyne/tests/test_config_management.py::TestConfigManager::test_config_manager_initialization",
            "-v" if verbose else "-q",
            "--tb=short",
        ]

        return pytest.main(args)

    def generate_test_report(self, output_file: str = "test_report.html") -> None:
        """Generate comprehensive test report."""
        print(f"📝 Generating test report: {output_file}")

        args = [
            "homodyne/tests/",
            "--html=" + output_file,
            "--self-contained-html",
            "--tb=short",
            "-q",
        ]

        pytest.main(args)

    def check_test_environment(self) -> dict[str, bool]:
        """Check test environment and dependencies."""
        print("🔍 Checking Test Environment...")

        dependencies = {
            "pytest": False,
            "numpy": False,
            "scipy": False,
            "numba": False,
            "cvxpy": False,
            "matplotlib": False,
        }

        for dep in dependencies:
            try:
                __import__(dep)
                dependencies[dep] = True
                print(f"  ✅ {dep}")
            except ImportError:
                print(f"  ❌ {dep} (optional)")

        return dependencies

    def run_with_profiling(self, test_pattern: str = "homodyne/tests/") -> int:
        """Run tests with performance profiling."""
        print("📊 Running Tests with Profiling...")

        try:
            import cProfile
            import pstats

            profiler = cProfile.Profile()
            profiler.enable()

            result = pytest.main([test_pattern, "-q", "--tb=short"])

            profiler.disable()

            # Save profiling results
            profiler.dump_stats("test_profile.prof")

            # Print top functions
            stats = pstats.Stats("test_profile.prof")
            stats.sort_stats("cumulative")
            stats.print_stats(20)

            return result

        except ImportError:
            print("⚠️  Profiling not available (cProfile not found)")
            return pytest.main([test_pattern, "-q", "--tb=short"])

    def memory_test(self, test_pattern: str = "homodyne/tests/") -> int:
        """Run tests with memory monitoring."""
        print("🧠 Running Tests with Memory Monitoring...")

        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            print(f"Initial memory usage: {initial_memory:.1f} MB")

            result = pytest.main([test_pattern, "-q", "--tb=short"])

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            print(f"Final memory usage: {final_memory:.1f} MB")
            print(f"Memory increase: {memory_increase:.1f} MB")

            if memory_increase > 500:  # 500 MB threshold
                print("⚠️  High memory usage detected")

            return result

        except ImportError:
            print("⚠️  Memory monitoring not available (psutil not found)")
            return pytest.main([test_pattern, "-q", "--tb=short"])


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(
        description="Advanced Test Runner for Homodyne Analysis Package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_runner.py --all                    # Run all tests
  python test_runner.py --unit --coverage        # Unit tests with coverage
  python test_runner.py --fast                   # Quick tests only
  python test_runner.py --performance --benchmark # Performance benchmarks
  python test_runner.py --scientific             # Scientific validation
  python test_runner.py --smoke                  # Minimal smoke tests
  python test_runner.py --check-env              # Check test environment
        """
    )

    # Test categories
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--security", action="store_true", help="Run security tests")
    parser.add_argument("--scientific", action="store_true", help="Run scientific validation tests")
    parser.add_argument("--fast", action="store_true", help="Run fast tests only")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests")
    parser.add_argument("--regression", action="store_true", help="Run regression tests")

    # Test options
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--profile", action="store_true", help="Run with profiling")
    parser.add_argument("--memory", action="store_true", help="Monitor memory usage")

    # Filtering
    parser.add_argument("--markers", nargs="+", help="Test markers to include")
    parser.add_argument("--pattern", help="Test file pattern")

    # Reporting
    parser.add_argument("--report", help="Generate HTML report file")
    parser.add_argument("--check-env", action="store_true", help="Check test environment")

    args = parser.parse_args()

    runner = TestRunner()

    # Check environment if requested
    if args.check_env:
        dependencies = runner.check_test_environment()
        if not dependencies["pytest"]:
            print("❌ pytest is required but not found")
            return 1
        return 0

    # Determine coverage setting
    coverage = args.coverage and not args.no_coverage
    if args.all or args.unit:
        coverage = True  # Default to coverage for comprehensive tests

    # Run requested tests
    result = 0

    if args.all:
        result = runner.run_all_tests(
            verbose=args.verbose,
            coverage=coverage,
            parallel=args.parallel,
            markers=args.markers
        )
    elif args.unit:
        result = runner.run_unit_tests(verbose=args.verbose, coverage=coverage)
    elif args.integration:
        result = runner.run_integration_tests(verbose=args.verbose)
    elif args.performance:
        result = runner.run_performance_tests(verbose=args.verbose, benchmark=args.benchmark)
    elif args.security:
        result = runner.run_security_tests(verbose=args.verbose)
    elif args.scientific:
        result = runner.run_scientific_tests(verbose=args.verbose)
    elif args.fast:
        result = runner.run_fast_tests(verbose=args.verbose)
    elif args.smoke:
        result = runner.run_smoke_tests(verbose=args.verbose)
    elif args.regression:
        result = runner.run_regression_tests(verbose=args.verbose)
    elif args.profile:
        pattern = args.pattern or "homodyne/tests/"
        result = runner.run_with_profiling(pattern)
    elif args.memory:
        pattern = args.pattern or "homodyne/tests/"
        result = runner.memory_test(pattern)
    elif args.pattern:
        result = pytest.main([args.pattern, "-v" if args.verbose else "-q"])
    else:
        # Default: run smoke tests
        print("No specific test category specified. Running smoke tests...")
        result = runner.run_smoke_tests(verbose=args.verbose)

    # Generate report if requested
    if args.report:
        runner.generate_test_report(args.report)

    # Print summary
    if result == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code: {result}")

    return result


if __name__ == "__main__":
    sys.exit(main())
