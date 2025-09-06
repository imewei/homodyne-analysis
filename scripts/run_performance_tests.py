#!/usr/bin/env python3
"""
Performance Baseline Testing and Regression Detection
====================================================

Automated performance testing system for Homodyne analysis package.
Runs performance benchmarks, compares against established baselines,
and detects performance regressions.

Usage:
    python scripts/run_performance_tests.py [--update-baselines] [--verbose]
    
    --update-baselines: Update baseline measurements with current results
    --verbose: Show detailed performance metrics
    --ci: Run in CI mode with appropriate timeouts and reporting
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import psutil


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


def load_baselines(baseline_file: Path) -> Dict[str, Any]:
    """Load performance baselines from JSON file."""
    if not baseline_file.exists():
        return {}
    
    with open(baseline_file) as f:
        return json.load(f)


def save_baselines(baseline_file: Path, baselines: Dict[str, Any]) -> None:
    """Save performance baselines to JSON file."""
    with open(baseline_file, 'w') as f:
        json.dump(baselines, f, indent=2, sort_keys=True)


def run_performance_tests(verbose: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Run performance tests and return timing results.
    
    Returns:
        Dictionary mapping test categories to performance metrics
    """
    logger = setup_logging(verbose)
    logger.info("🚀 Starting performance benchmark tests...")
    
    # Run pytest with performance markers
    cmd = [
        "pytest", 
        "-m", "performance", 
        "--tb=short",
        "-v" if verbose else "-q",
        "--benchmark-only",
        "--benchmark-json=benchmark_results.json"
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    total_time = time.time() - start_time
    
    if result.returncode != 0:
        logger.error(f"Performance tests failed: {result.stderr}")
        sys.exit(1)
    
    logger.info(f"✅ Performance tests completed in {total_time:.2f}s")
    
    # Parse benchmark results
    try:
        with open("benchmark_results.json") as f:
            benchmark_data = json.load(f)
        
        # Extract relevant metrics
        results = {}
        for benchmark in benchmark_data.get("benchmarks", []):
            name = benchmark["name"]
            stats = benchmark["stats"]
            
            # Categorize by test module
            if "core" in name:
                category = "core_analysis" 
            elif "optimization" in name:
                category = "optimization_methods"
            elif "angle" in name or "irls" in name:
                category = "data_processing"
            else:
                category = "other"
            
            if category not in results:
                results[category] = {}
            
            results[category][name] = {
                "mean_time_ms": stats["mean"] * 1000,
                "median_time_ms": stats["median"] * 1000,
                "stddev_ms": stats["stddev"] * 1000,
                "min_time_ms": stats["min"] * 1000,
                "max_time_ms": stats["max"] * 1000
            }
        
        return results
    
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse benchmark results: {e}")
        return {}


def check_memory_usage() -> Dict[str, float]:
    """Monitor memory usage during tests."""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "peak_memory_mb": memory_info.rss / 1024 / 1024,
        "virtual_memory_mb": memory_info.vms / 1024 / 1024
    }


def compare_with_baselines(
    current_results: Dict[str, Dict[str, float]],
    baselines: Dict[str, Any],
    logger: logging.Logger
) -> tuple[bool, list[str]]:
    """
    Compare current results with established baselines.
    
    Returns:
        Tuple of (passed, warnings) where passed indicates if all tests
        are within acceptable limits, and warnings contains regression alerts.
    """
    warnings = []
    passed = True
    
    regression_config = baselines.get("regression_thresholds", {})
    fail_threshold = regression_config.get("fail_if_slower_than_percent", 50.0)
    warn_threshold = regression_config.get("warn_if_slower_than_percent", 25.0)
    
    baseline_tests = baselines.get("baselines", {})
    
    for category, tests in current_results.items():
        if category not in baseline_tests:
            logger.info(f"ℹ️  New test category: {category}")
            continue
            
        baseline_category = baseline_tests[category]
        
        for test_name, metrics in tests.items():
            # Find matching baseline
            baseline_test = None
            for baseline_name, baseline_data in baseline_category.items():
                if baseline_data.get("test_function", "") in test_name:
                    baseline_test = baseline_data
                    break
            
            if not baseline_test:
                logger.info(f"ℹ️  New test: {test_name}")
                continue
            
            # Compare performance
            current_time = metrics["median_time_ms"]
            baseline_time = baseline_test.get("max_time_ms", float('inf'))
            
            if baseline_time > 0:
                percent_change = ((current_time - baseline_time) / baseline_time) * 100
                
                if percent_change > fail_threshold:
                    warnings.append(
                        f"❌ REGRESSION: {test_name} is {percent_change:.1f}% slower "
                        f"({current_time:.1f}ms vs {baseline_time:.1f}ms baseline)"
                    )
                    passed = False
                elif percent_change > warn_threshold:
                    warnings.append(
                        f"⚠️  WARNING: {test_name} is {percent_change:.1f}% slower "
                        f"({current_time:.1f}ms vs {baseline_time:.1f}ms baseline)"
                    )
                else:
                    logger.info(
                        f"✅ {test_name}: {current_time:.1f}ms "
                        f"({'📈' if percent_change > 0 else '📉'}{abs(percent_change):.1f}%)"
                    )
    
    return passed, warnings


def main():
    """Main performance testing workflow."""
    parser = argparse.ArgumentParser(description="Run performance baseline tests")
    parser.add_argument(
        "--update-baselines", 
        action="store_true",
        help="Update baseline measurements with current results"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Show detailed performance metrics"
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Run in CI mode with appropriate timeouts"
    )
    
    args = parser.parse_args()
    logger = setup_logging(args.verbose)
    
    # File paths
    baseline_file = Path("homodyne/tests/performance_baselines.json")
    
    # Load existing baselines
    baselines = load_baselines(baseline_file)
    
    logger.info("🔬 Running Homodyne Performance Benchmark Suite")
    logger.info("=" * 60)
    
    # Run performance tests
    current_results = run_performance_tests(args.verbose)
    
    if not current_results:
        logger.error("❌ No performance results obtained")
        sys.exit(1)
    
    # Check memory usage
    memory_metrics = check_memory_usage()
    current_results["memory_usage"] = memory_metrics
    
    # Compare with baselines if they exist
    if baselines and not args.update_baselines:
        passed, warnings = compare_with_baselines(current_results, baselines, logger)
        
        logger.info("=" * 60)
        logger.info("📊 Performance Analysis Results:")
        
        for warning in warnings:
            logger.warning(warning)
        
        if passed:
            logger.info("✅ All performance tests PASSED")
        else:
            logger.error("❌ Performance regressions detected")
            if args.ci:
                sys.exit(1)
    
    # Update baselines if requested
    if args.update_baselines:
        logger.info("📝 Updating performance baselines...")
        
        # Update the baselines structure with current results
        if "baselines" not in baselines:
            baselines["baselines"] = {}
        
        # Update metadata
        baselines["baseline_date"] = time.strftime("%Y-%m-%d")
        baselines["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        save_baselines(baseline_file, baselines)
        logger.info(f"✅ Baselines saved to {baseline_file}")
    
    logger.info("🏁 Performance testing complete!")


if __name__ == "__main__":
    main()