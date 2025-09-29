#!/usr/bin/env python3
"""
Performance Baseline Establishment Script
=========================================

Script to establish performance baselines for import timing and startup benchmarks.
Run this script to create baseline measurements for regression testing.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any


def create_comprehensive_baselines() -> dict[str, Any]:
    """Create comprehensive performance baselines."""
    print("Establishing performance baselines...")
    print("====================================")

    suite = StartupBenchmarkSuite()

    # Core modules to benchmark
    core_modules = [
        "homodyne",
        "homodyne.core.config",
        "homodyne.core.kernels",
        "homodyne.analysis.core",
        "homodyne.optimization.classical",
        "homodyne.optimization.robust",
        "homodyne.visualization.plotting",
    ]

    baselines = {
        "metadata": {
            "created_at": time.time(),
            "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
        },
        "individual_modules": {},
        "progressive_loading": {},
        "lazy_loading": {},
        "conditional_imports": {},
        "memory_usage": {},
    }

    # Individual module baselines
    print("\n1. Individual module import baselines:")
    for module_name in core_modules:
        print(f"   Measuring {module_name}...")

        # Cold import measurement
        cold_result = suite.measure_cold_import(module_name)

        # Warm import measurement
        warm_result = suite.measure_warm_import(module_name, iterations=3)

        baselines["individual_modules"][module_name] = {
            "cold_import": cold_result,
            "warm_import": warm_result,
        }

        if "error" not in cold_result:
            import_time = cold_result.get("import_time", 0)
            memory_usage = cold_result.get("memory_delta_mb", 0)
            print(f"     Cold: {import_time:.3f}s, {memory_usage:.1f}MB")

    # Progressive loading baselines
    print("\n2. Progressive loading baselines:")
    progressive_results = suite.benchmark_progressive_loading()
    baselines["progressive_loading"] = progressive_results

    total_time = 0
    for stage, metrics in progressive_results.items():
        if "error" not in metrics:
            stage_time = metrics.get("import_time", 0)
            total_time = metrics.get("cumulative_time", total_time)
            print(f"   {stage}: {stage_time:.3f}s (cumulative: {total_time:.3f}s)")

    # Lazy loading effectiveness
    print("\n3. Lazy loading effectiveness:")
    lazy_results = suite.benchmark_lazy_loading_effectiveness()
    baselines["lazy_loading"] = lazy_results

    basic_time = lazy_results.get("basic_import_time", 0)
    lazy_overhead = lazy_results.get("lazy_loading_overhead", 0)
    print(f"   Basic import: {basic_time:.3f}s")
    print(f"   Lazy loading overhead: {lazy_overhead:.1f}x")

    # Conditional imports
    print("\n4. Conditional imports:")
    conditional_results = suite.benchmark_conditional_imports()
    baselines["conditional_imports"] = conditional_results

    for test_name, metrics in conditional_results.items():
        overhead = metrics.get("dependency_overhead", 1.0)
        print(f"   {test_name}: {overhead:.1f}x dependency overhead")

    # Memory usage patterns
    print("\n5. Memory usage patterns:")
    memory_patterns = {}

    for module_name in core_modules[:3]:  # Test a few key modules
        result = suite.measure_cold_import(module_name)
        if "error" not in result:
            memory_patterns[module_name] = {
                "import_memory_mb": result.get("memory_delta_mb", 0),
                "peak_memory_mb": result.get("peak_memory_mb", 0),
                "memory_percent": result.get("memory_percent", 0),
            }

    baselines["memory_usage"] = memory_patterns

    # Calculate summary statistics
    valid_times = []
    valid_memory = []

    for module_data in baselines["individual_modules"].values():
        cold_result = module_data.get("cold_import", {})
        if "error" not in cold_result:
            import_time = cold_result.get("import_time")
            memory_usage = cold_result.get("memory_delta_mb")

            if import_time is not None:
                valid_times.append(import_time)
            if memory_usage is not None:
                valid_memory.append(memory_usage)

    if valid_times:
        baselines["summary"] = {
            "avg_import_time": sum(valid_times) / len(valid_times),
            "max_import_time": max(valid_times),
            "total_import_time": sum(valid_times),
            "avg_memory_usage": (
                sum(valid_memory) / len(valid_memory) if valid_memory else 0
            ),
            "max_memory_usage": max(valid_memory) if valid_memory else 0,
            "total_modules_tested": len(valid_times),
        }

    return baselines


def save_baselines(baselines: dict[str, Any], output_file: Path):
    """Save baselines to JSON file."""
    with open(output_file, "w") as f:
        json.dump(baselines, f, indent=2, default=str)

    print(f"\nBaselines saved to: {output_file}")


def print_baseline_summary(baselines: dict[str, Any]):
    """Print a summary of the established baselines."""
    summary = baselines.get("summary", {})

    print("\nBaseline Summary:")
    print("================")
    print(f"Modules tested: {summary.get('total_modules_tested', 0)}")
    print(f"Average import time: {summary.get('avg_import_time', 0):.3f}s")
    print(f"Maximum import time: {summary.get('max_import_time', 0):.3f}s")
    print(f"Total import time: {summary.get('total_import_time', 0):.3f}s")
    print(f"Average memory usage: {summary.get('avg_memory_usage', 0):.1f}MB")
    print(f"Maximum memory usage: {summary.get('max_memory_usage', 0):.1f}MB")

    # Identify potential issues
    issues = []
    if summary.get("max_import_time", 0) > 3.0:
        issues.append(f"Slow import detected: {summary['max_import_time']:.3f}s")

    if summary.get("max_memory_usage", 0) > 100.0:
        issues.append(f"High memory usage: {summary['max_memory_usage']:.1f}MB")

    lazy_overhead = baselines.get("lazy_loading", {}).get("lazy_loading_overhead", 0)
    if lazy_overhead > 10.0:
        issues.append(f"High lazy loading overhead: {lazy_overhead:.1f}x")

    if issues:
        print("\nPotential Issues Detected:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("\n✅ All performance metrics within expected ranges")


def compare_with_existing_baselines(new_baselines: dict[str, Any], existing_file: Path):
    """Compare new baselines with existing ones."""
    if not existing_file.exists():
        print("\nNo existing baselines found for comparison.")
        return

    with open(existing_file) as f:
        existing_baselines = json.load(f)

    print("\nComparison with existing baselines:")
    print("===================================")

    # Compare individual modules
    for module_name in new_baselines.get("individual_modules", {}):
        if module_name in existing_baselines.get("individual_modules", {}):
            new_cold = new_baselines["individual_modules"][module_name].get(
                "cold_import", {}
            )
            old_cold = existing_baselines["individual_modules"][module_name].get(
                "cold_import", {}
            )

            new_time = new_cold.get("import_time")
            old_time = old_cold.get("import_time")

            if new_time is not None and old_time is not None:
                change_ratio = new_time / old_time
                change_percent = (change_ratio - 1) * 100

                status = (
                    "📈" if change_ratio > 1.1 else "📉" if change_ratio < 0.9 else "📊"
                )
                print(
                    f"  {status} {module_name}: {old_time:.3f}s → {new_time:.3f}s ({change_percent:+.1f}%)"
                )

    # Compare summary metrics
    new_summary = new_baselines.get("summary", {})
    old_summary = existing_baselines.get("summary", {})

    for metric in ["avg_import_time", "max_import_time", "avg_memory_usage"]:
        new_value = new_summary.get(metric, 0)
        old_value = old_summary.get(metric, 0)

        if old_value > 0:
            change_percent = ((new_value / old_value) - 1) * 100
            status = (
                "📈" if change_percent > 10 else "📉" if change_percent < -10 else "📊"
            )
            print(
                f"  {status} {metric}: {old_value:.3f} → {new_value:.3f} ({change_percent:+.1f}%)"
            )


def main():
    """Main function to establish baselines."""
    import sys

    # Determine output file
    baseline_file = Path(__file__).parent / "startup_baselines.json"

    print("Performance Baseline Establishment")
    print("=================================")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Output file: {baseline_file}")

    # Check if baselines already exist
    existing_baselines = None
    if baseline_file.exists():
        print("\nExisting baselines found. Creating new baselines for comparison.")
        existing_baselines = baseline_file

    # Create new baselines
    try:
        new_baselines = create_comprehensive_baselines()

        # Compare with existing if available
        if existing_baselines:
            compare_with_existing_baselines(new_baselines, existing_baselines)

        # Print summary
        print_baseline_summary(new_baselines)

        # Save new baselines
        save_baselines(new_baselines, baseline_file)

        print("\n✅ Baseline establishment completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during baseline establishment: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
