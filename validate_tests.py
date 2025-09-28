#!/usr/bin/env python3
"""
Quick Test Validation Script
============================

Validates that the generated test suite is properly configured and can run basic tests.
"""

import sys
import subprocess
import os

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"🔍 {description}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print(f"  ✅ Success")
            return True
        else:
            print(f"  ❌ Failed: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def check_imports():
    """Check that test modules can be imported."""
    print("🔍 Checking test module imports...")

    modules = [
        "homodyne.tests.conftest",
        "homodyne.core.kernels",
        "homodyne.core.config",
    ]

    success = True
    for module in modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            success = False

    return success

def check_test_structure():
    """Check that test files are properly structured."""
    print("🔍 Checking test file structure...")

    test_files = [
        "homodyne/tests/conftest.py",
        "homodyne/tests/test_core_kernels.py",
        "homodyne/tests/test_analysis_core.py",
        "homodyne/tests/test_config_management.py",
        "homodyne/tests/test_cli_integration.py",
        "homodyne/tests/test_optimization_performance.py",
        "homodyne/tests/test_security_validation.py",
        "homodyne/tests/test_scientific_validation.py",
        "homodyne/tests/pytest.ini",
        "homodyne/tests/requirements-test.txt",
        "homodyne/tests/test_runner.py",
    ]

    success = True
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"  ✅ {test_file}")
        else:
            print(f"  ❌ Missing: {test_file}")
            success = False

    return success

def run_basic_tests():
    """Run basic pytest functionality tests."""
    print("🔍 Running basic pytest functionality tests...")

    # Test pytest collection
    if not run_command("python -m pytest homodyne/tests/ --collect-only -q",
                      "Collecting tests"):
        return False

    # Test conftest.py loading
    if not run_command("python -c 'import homodyne.tests.conftest; print(\"conftest loaded\")'",
                      "Loading conftest.py"):
        return False

    return True

def check_dependencies():
    """Check required testing dependencies."""
    print("🔍 Checking testing dependencies...")

    required = ["pytest", "numpy"]
    optional = ["scipy", "numba", "matplotlib"]

    success = True
    for dep in required:
        try:
            __import__(dep)
            print(f"  ✅ {dep} (required)")
        except ImportError:
            print(f"  ❌ {dep} (required) - MISSING")
            success = False

    for dep in optional:
        try:
            __import__(dep)
            print(f"  ✅ {dep} (optional)")
        except ImportError:
            print(f"  ⚠️  {dep} (optional) - missing")

    return success

def main():
    """Main validation function."""
    print("🚀 Homodyne Test Suite Validation")
    print("=" * 50)

    all_checks = [
        check_dependencies,
        check_test_structure,
        check_imports,
        run_basic_tests,
    ]

    results = []
    for check in all_checks:
        try:
            result = check()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Check failed with error: {e}")
            results.append(False)
            print()

    # Summary
    print("📊 Validation Summary")
    print("-" * 30)

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✅ All {total} validation checks passed!")
        print("\n🎉 Test suite is ready for use!")
        print("\nQuick start commands:")
        print("  python homodyne/tests/test_runner.py --smoke")
        print("  python -m pytest homodyne/tests/ -v")
        return 0
    else:
        print(f"❌ {total - passed} out of {total} validation checks failed")
        print("\n⚠️  Test suite may have issues. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())