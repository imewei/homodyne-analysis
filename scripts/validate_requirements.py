#!/usr/bin/env python3
"""
Requirements Validation Script for Homodyne Scattering Analysis

This script validates that all requirements files are consistent with pyproject.toml
and that there are no conflicting dependencies or missing packages.

Usage:
    python scripts/validate_requirements.py
"""

import sys
from pathlib import Path
import toml
import re
from typing import Dict, List, Set, Tuple

def parse_requirements_file(file_path: Path) -> Set[str]:
    """Parse a requirements file and return set of package names."""
    packages = set()

    if not file_path.exists():
        print(f"⚠️  Requirements file not found: {file_path}")
        return packages

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip comments, empty lines, and -r includes
            if not line or line.startswith('#') or line.startswith('-r'):
                continue

            # Extract package name (before version specifiers)
            package_match = re.match(r'^([a-zA-Z0-9_-]+)', line)
            if package_match:
                packages.add(package_match.group(1).lower())

    return packages

def get_pyproject_dependencies() -> Dict[str, Set[str]]:
    """Extract dependencies from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")

    if not pyproject_path.exists():
        print("❌ pyproject.toml not found!")
        return {}

    with open(pyproject_path, 'r') as f:
        pyproject = toml.load(f)

    deps = {}

    # Core dependencies
    core_deps = set()
    for dep in pyproject.get('project', {}).get('dependencies', []):
        package_match = re.match(r'^([a-zA-Z0-9_-]+)', dep)
        if package_match:
            core_deps.add(package_match.group(1).lower())
    deps['core'] = core_deps

    # Optional dependencies
    optional_deps = pyproject.get('project', {}).get('optional-dependencies', {})
    for group, dep_list in optional_deps.items():
        group_deps = set()
        for dep in dep_list:
            # Skip self-references like "homodyne-analysis[test,docs]"
            if 'homodyne-analysis[' in dep:
                continue
            package_match = re.match(r'^([a-zA-Z0-9_-]+)', dep)
            if package_match:
                group_deps.add(package_match.group(1).lower())
        deps[group] = group_deps

    return deps

def validate_requirements():
    """Validate all requirements files against pyproject.toml."""
    print("🚀 VALIDATING REQUIREMENTS FILES")
    print("=" * 50)

    # Get pyproject.toml dependencies
    pyproject_deps = get_pyproject_dependencies()

    if not pyproject_deps:
        print("❌ Failed to parse pyproject.toml")
        return False

    print(f"✅ Found {len(pyproject_deps)} dependency groups in pyproject.toml")

    # Requirements files to validate
    req_files = [
        ("requirements.txt", "core"),
        ("requirements-jax.txt", "jax"),
    ]

    validation_results = []

    for req_file, expected_group in req_files:
        print(f"\n🧪 Validating {req_file}...")

        req_path = Path(req_file)
        req_packages = parse_requirements_file(req_path)

        if not req_packages:
            print(f"⚠️  No packages found in {req_file}")
            validation_results.append(False)
            continue

        expected_packages = pyproject_deps.get(expected_group, set())

        # Check for missing packages
        missing = expected_packages - req_packages
        if missing:
            print(f"❌ Missing packages in {req_file}: {sorted(missing)}")
            validation_results.append(False)
        else:
            print(f"✅ All expected packages present in {req_file}")
            validation_results.append(True)

        # Check for extra packages (informational)
        extra = req_packages - expected_packages
        if extra:
            print(f"ℹ️  Extra packages in {req_file}: {sorted(extra)}")

        print(f"📊 {req_file}: {len(req_packages)} packages")

    # Validate requirements-optional.txt (comprehensive)
    print(f"\n🧪 Validating requirements-optional.txt...")
    opt_req_path = Path("requirements-optional.txt")
    opt_packages = parse_requirements_file(opt_req_path)

    # Expected packages from multiple groups
    expected_optional = set()
    for group in ['data', 'performance', 'jax', 'robust', 'gurobi', 'completion']:
        expected_optional.update(pyproject_deps.get(group, set()))

    missing_opt = expected_optional - opt_packages
    if missing_opt:
        print(f"❌ Missing packages in requirements-optional.txt: {sorted(missing_opt)}")
        validation_results.append(False)
    else:
        print(f"✅ All expected optional packages present")
        validation_results.append(True)

    print(f"📊 requirements-optional.txt: {len(opt_packages)} packages")

    # Validate requirements-dev.txt (comprehensive)
    print(f"\n🧪 Validating requirements-dev.txt...")
    dev_req_path = Path("requirements-dev.txt")
    dev_packages = parse_requirements_file(dev_req_path)

    # Expected packages from dev-related groups
    expected_dev = set()
    for group in ['test', 'docs', 'quality', 'typing', 'completion']:
        expected_dev.update(pyproject_deps.get(group, set()))

    missing_dev = expected_dev - dev_packages
    if missing_dev:
        print(f"❌ Missing packages in requirements-dev.txt: {sorted(missing_dev)}")
        validation_results.append(False)
    else:
        print(f"✅ All expected development packages present")
        validation_results.append(True)

    print(f"📊 requirements-dev.txt: {len(dev_packages)} packages")

    # Check for MCMC-related packages (should be absent)
    print(f"\n🧪 Checking for deprecated MCMC packages...")
    mcmc_packages = {'pymc', 'arviz', 'pytensor', 'corner'}

    all_req_files = [
        "requirements.txt",
        "requirements-jax.txt",
        "requirements-optional.txt",
        "requirements-dev.txt"
    ]

    mcmc_found = False
    for req_file in all_req_files:
        req_packages = parse_requirements_file(Path(req_file))
        found_mcmc = req_packages.intersection(mcmc_packages)
        if found_mcmc:
            print(f"❌ Found MCMC packages in {req_file}: {sorted(found_mcmc)}")
            mcmc_found = True

    if not mcmc_found:
        print("✅ No deprecated MCMC packages found in requirements files")
        validation_results.append(True)
    else:
        validation_results.append(False)

    # Summary
    print("\n" + "=" * 50)
    print("📊 REQUIREMENTS VALIDATION SUMMARY")
    passed = sum(validation_results)
    total = len(validation_results)
    print(f"✅ Validations passed: {passed}/{total}")

    if passed == total:
        print("🎉 ALL REQUIREMENTS FILES ARE VALID!")
        print("✅ Dependencies consistent with pyproject.toml")
        print("✅ No deprecated MCMC packages found")
        print("✅ All required packages present")
    else:
        print(f"❌ {total - passed} validation(s) failed")
        print("🔧 Please review and update requirements files")

    return passed == total

if __name__ == "__main__":
    success = validate_requirements()
    sys.exit(0 if success else 1)