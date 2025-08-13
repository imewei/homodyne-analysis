"""
Test Runner for Rheo-SAXS-XPCS Analysis
======================================

Convenience script to run the test suite with appropriate options.
"""

import sys
import subprocess
from pathlib import Path
import argparse


def main():
    """Run the test suite with configurable options."""
    parser = argparse.ArgumentParser(
        description="Run Rheo-SAXS-XPCS test suite"
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run only fast tests (exclude slow integration tests)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
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
            print(
                "Warning: pytest-xdist not available, running tests serially"
            )

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
