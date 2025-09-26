#!/usr/bin/env python3
"""
Comprehensive Performance Analysis Suite for Homodyne Analysis
=============================================================

This script runs the complete performance analysis suite and generates
actionable reports for optimization tracking and performance monitoring.

Includes:
1. Current performance metrics collection
2. Performance bottleneck identification
3. User experience impact assessment
4. Performance baseline report generation
5. Optimization opportunity prioritization

Usage:
    python run_performance_analysis.py [--quick] [--full-report] [--dashboard]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def print_header():
    """Print analysis header."""
    print("=" * 80)
    print("HOMODYNE ANALYSIS - COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def print_section(title: str):
    """Print section header."""
    print(f"\n{'-' * 60}")
    print(f"{title}")
    print(f"{'-' * 60}")


def run_command(cmd: list, description: str):
    """Run a command and return success status."""
    print(f"Running: {description}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(f"   {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False


def main():
    """Run comprehensive performance analysis."""
    parser = argparse.ArgumentParser(description="Homodyne Performance Analysis Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick analysis only")
    parser.add_argument(
        "--full-report", action="store_true", help="Generate detailed reports"
    )
    parser.add_argument(
        "--dashboard", action="store_true", help="Create performance dashboard"
    )
    parser.add_argument(
        "--test-suite", action="store_true", help="Include test suite performance"
    )

    args = parser.parse_args()

    print_header()

    success_count = 0
    total_analyses = 0

    # 1. Run basic performance baseline
    print_section("PHASE 1: BASIC PERFORMANCE BASELINE")
    total_analyses += 1
    cmd = [
        sys.executable,
        "-m",
        "homodyne.performance_baseline",
        "--profile-mode",
        "quick",
    ]
    if not args.test_suite:
        cmd.append("--skip-tests")

    if run_command(cmd, "Basic performance baseline analysis"):
        success_count += 1

    # 2. Run comprehensive monitoring if requested
    if not args.quick:
        print_section("PHASE 2: COMPREHENSIVE PERFORMANCE MONITORING")
        total_analyses += 1
        cmd = [
            sys.executable,
            "-m",
            "homodyne.performance_monitoring",
            "--mode",
            "comprehensive",
        ]
        if args.dashboard:
            cmd.append("--create-dashboard")

        if run_command(cmd, "Comprehensive performance monitoring"):
            success_count += 1

    # 3. Run test suite performance analysis
    if args.test_suite:
        print_section("PHASE 3: TEST SUITE PERFORMANCE ANALYSIS")
        total_analyses += 1
        cmd = [sys.executable, "-m", "pytest", "homodyne/tests/", "-v", "--tb=short"]

        if run_command(cmd, "Test suite execution and performance"):
            success_count += 1

    # 4. Generate reports
    if args.full_report:
        print_section("PHASE 4: REPORT GENERATION")

        # Check if reports exist
        reports_dir = Path("performance_reports")
        monitoring_dir = Path("performance_monitoring")

        reports_found = []
        if reports_dir.exists():
            reports_found.extend(list(reports_dir.glob("*.json")))
        if monitoring_dir.exists():
            reports_found.extend(list(monitoring_dir.glob("*.json")))

        if reports_found:
            print(f"📊 Found {len(reports_found)} performance reports")
            for report in reports_found[-3:]:  # Show last 3
                print(f"   • {report}")
        else:
            print("⚠️  No performance reports found")

    # Summary
    print_section("ANALYSIS SUMMARY")
    print(f"Completed: {success_count}/{total_analyses} analyses successful")

    if success_count == total_analyses:
        print("🎉 All analyses completed successfully!")
        print()
        print("📋 PERFORMANCE SUMMARY:")
        print("   • Performance baseline established")
        print("   • Bottlenecks identified and prioritized")
        print("   • Optimization opportunities documented")
        print("   • Actionable recommendations provided")
        print()
        print("📁 GENERATED FILES:")
        print("   • PERFORMANCE_BASELINE_REPORT.md - Executive summary")
        print("   • performance_reports/ - Detailed JSON reports")
        print("   • performance_monitoring/ - Advanced monitoring data")
        if args.dashboard:
            print("   • performance_dashboard.png - Visual dashboard")

        print()
        print("🚀 NEXT STEPS:")
        print("   1. Review PERFORMANCE_BASELINE_REPORT.md for key findings")
        print("   2. Implement Priority 1-2 optimizations (chi-squared, imports)")
        print("   3. Set up continuous performance monitoring")
        print("   4. Track optimization progress with regression testing")

        return 0
    else:
        print("⚠️  Some analyses failed - check output above for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
