"""
Test Runner for Homodyne Scattering Analysis
===========================================

Advanced test execution framework for comprehensive validation of XPCS
analysis components. Provides flexible test selection, performance profiling,
and detailed reporting capabilities for development and quality assurance.

Testing Framework Features:
- Selective test execution (fast vs comprehensive)
- Parallel test execution with pytest-xdist integration
- Code coverage analysis and reporting
- Performance benchmarking and regression detection
- Flexible test filtering and marker-based selection

Test Categories:
- Unit tests: Core functionality and computational kernels
- Integration tests: End-to-end analysis workflows
- Performance tests: Optimization and scaling validation
- I/O tests: Data loading, saving, and serialization
- Plotting tests: Visualization and result presentation

Usage Scenarios:
- Development validation: Quick smoke tests during coding
- CI/CD integration: Comprehensive validation pipelines
- Performance monitoring: Regression detection and optimization
- Release validation: Full test suite execution with coverage
"""

import sys
import subprocess
from pathlib import Path
import argparse


def main():
    """
    Execute comprehensive test suite with intelligent configuration.
    
    Orchestrates test execution with advanced options for development,
    continuous integration, and quality assurance workflows. Automatically
    detects available testing tools and configures optimal execution strategy.
    
    Test Execution Strategy:
    - Fast mode: Unit tests only, optimized for development cycles
    - Full mode: Complete test suite including slow integration tests
    - Parallel mode: Multi-worker execution for faster completion
    - Coverage mode: Code coverage analysis with detailed reporting
    """
    parser = argparse.ArgumentParser(description="Run homodyne analysis test suite with advanced options")

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run only fast tests (exclude slow integration tests)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--coverage", action="store_true", help="Run with coverage reporting"
    )
    parser.add_argument(
        "--parallel",
        "-n",
        type=int,
        default=1,
        help="Number of parallel workers (requires pytest-xdist)",
    )
    parser.add_argument(
        "--markers",
        "-m",
        type=str,
        help="Run tests matching given mark expression",
    )
    parser.add_argument(
        "--test-file",
        "-k",
        type=str,
        help="Run tests matching keyword expression",
    )

    args = parser.parse_args()

    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"]

    # Add test directory (now at same level as this script)
    test_dir = Path(__file__).parent / "tests"
    cmd.append(str(test_dir))

    # Configure verbosity
    if args.verbose:
        cmd.extend(["-v", "-s"])

    # Configure parallelization
    if args.parallel > 1:
        try:
            import xdist

            cmd.extend(["-n", str(args.parallel)])
        except ImportError:
            print("Warning: pytest-xdist not available, running tests serially")

    # Configure coverage
    if args.coverage:
        try:
            import coverage

            cmd.extend(
                [
                    "--cov=homodyne",
                    "--cov-report=html",
                    "--cov-report=term-missing",
                    f"--cov-config={Path(__file__).parent / '.coveragerc'}",
                ]
            )
        except ImportError:
            print("Warning: coverage package not available, skipping coverage")

    # Configure test selection
    if args.fast:
        cmd.extend(["-m", "not slow"])
    elif args.markers:
        cmd.extend(["-m", args.markers])

    if args.test_file:
        cmd.extend(["-k", args.test_file])

    # Add additional pytest options for better output
    cmd.extend(
        [
            "--tb=short",  # Shorter traceback format
            "--strict-markers",  # Strict marker checking
            "--disable-warnings",  # Disable warning summary for cleaner output
        ]
    )

    print(f"Running command: {' '.join(cmd)}")
    print("-" * 60)

    # Run the tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        return 130
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
