#!/usr/bin/env python3
"""
Version Consistency Validation Script
=====================================

Validates version consistency across the homodyne codebase to prevent drift.
Run this script as part of CI/CD or before releases to ensure all version
references are aligned.

Usage:
    python scripts/validate_version_consistency.py

Exit codes:
    0: All versions are consistent
    1: Version inconsistencies found
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def extract_python_version_from_pyproject() -> str:
    """Extract Python version from pyproject.toml mypy configuration."""
    pyproject_file = Path("pyproject.toml")
    if not pyproject_file.exists():
        raise FileNotFoundError("pyproject.toml not found")
    
    content = pyproject_file.read_text()
    
    # Extract MyPy Python version
    mypy_match = re.search(r'python_version = "(\d+\.\d+)"', content)
    if mypy_match:
        return mypy_match.group(1)
    
    raise ValueError("Could not find python_version in pyproject.toml [tool.mypy]")


def extract_target_versions_from_pyproject() -> List[str]:
    """Extract target versions from Black and Ruff configurations."""
    pyproject_file = Path("pyproject.toml")
    content = pyproject_file.read_text()
    
    versions = []
    
    # Extract Black target-version
    black_match = re.search(r'target-version = \[\s*"(py\d+)",?\s*"?(py\d+)?"?\s*\]', content)
    if black_match:
        versions.extend([v for v in black_match.groups() if v])
    
    # Extract Ruff target-version
    ruff_match = re.search(r'target-version = "(py\d+)"', content)
    if ruff_match:
        versions.append(ruff_match.group(1))
    
    return list(set(versions))


def check_workflow_python_versions() -> List[Tuple[str, str]]:
    """Check Python versions in GitHub workflows."""
    workflow_dir = Path(".github/workflows")
    if not workflow_dir.exists():
        return []
    
    versions = []
    for workflow_file in workflow_dir.glob("*.yml"):
        content = workflow_file.read_text()
        
        # Find python-version specifications
        python_version_matches = re.findall(r'python-version:\s*[\'"]?([\d.]+)[\'"]?', content)
        for version in python_version_matches:
            versions.append((workflow_file.name, version))
        
        # Find PYTHON_VERSION environment variables
        env_matches = re.findall(r'PYTHON_VERSION:\s*[\'"]?([\d.]+)[\'"]?', content)
        for version in env_matches:
            versions.append((workflow_file.name, version))
    
    return versions


def check_test_python_versions() -> List[Tuple[str, str]]:
    """Check hardcoded Python versions in test files."""
    test_dir = Path("homodyne/tests")
    if not test_dir.exists():
        return []
    
    versions = []
    for test_file in test_dir.rglob("*.py"):
        try:
            content = test_file.read_text()
            
            # Find hardcoded python_version references
            version_matches = re.findall(r'"python_version":\s*"([\d.]+)"', content)
            for version in version_matches:
                try:
                    relative_path = str(test_file.relative_to(Path.cwd()))
                    versions.append((relative_path, version))
                except ValueError:
                    # File is not in current working directory subpath
                    versions.append((str(test_file), version))
                
        except UnicodeDecodeError:
            # Skip binary files
            continue
    
    return versions


def check_performance_baselines() -> List[Tuple[str, str]]:
    """Check Python versions in performance baseline files."""
    baseline_files = [
        "homodyne/tests/performance_baselines.json"
    ]
    
    versions = []
    for baseline_file in baseline_files:
        baseline_path = Path(baseline_file)
        if baseline_path.exists():
            try:
                with open(baseline_path) as f:
                    data = json.load(f)
                    if "python_version" in data:
                        versions.append((baseline_file, data["python_version"]))
            except (json.JSONDecodeError, KeyError):
                continue
    
    return versions


def main() -> int:
    """Main validation function."""
    print("🔍 Validating version consistency across homodyne codebase...")
    
    errors = []
    
    try:
        # Get expected Python version from pyproject.toml
        expected_python_version = extract_python_version_from_pyproject()
        print(f"📋 Expected Python version: {expected_python_version}")
        
        # Check target versions
        target_versions = extract_target_versions_from_pyproject()
        expected_py_version = f"py{expected_python_version.replace('.', '')}"
        
        if expected_py_version not in target_versions:
            errors.append(f"Target version {expected_py_version} not found in Black/Ruff configs")
        
        if target_versions and target_versions[0] != expected_py_version:
            errors.append(f"Primary target version should be {expected_py_version}, found {target_versions[0]}")
        
        # Check workflow versions
        print("🔧 Checking GitHub workflows...")
        workflow_versions = check_workflow_python_versions()
        for workflow, version in workflow_versions:
            if version != expected_python_version:
                errors.append(f"Workflow {workflow} uses Python {version}, expected {expected_python_version}")
        
        # Check test versions
        print("🧪 Checking test files...")
        test_versions = check_test_python_versions()
        for test_file, version in test_versions:
            if version != expected_python_version:
                errors.append(f"Test file {test_file} has hardcoded Python {version}, expected {expected_python_version}")
        
        # Check performance baselines
        print("📊 Checking performance baselines...")
        baseline_versions = check_performance_baselines()
        for baseline_file, version in baseline_versions:
            # Allow more flexible matching for baselines (e.g., "3.13" vs "3.13.5")
            if not version.startswith(expected_python_version):
                errors.append(f"Baseline file {baseline_file} has Python {version}, expected {expected_python_version}.*")
        
        # Report results
        if errors:
            print("❌ Version consistency validation FAILED:")
            for error in errors:
                print(f"   • {error}")
            return 1
        else:
            print("✅ All version references are consistent!")
            print(f"   • Python version: {expected_python_version}")
            print(f"   • Target versions: {', '.join(target_versions)}")
            print(f"   • Workflows checked: {len(set(w for w, _ in workflow_versions))}")
            print(f"   • Test files checked: {len(test_versions)}")
            print(f"   • Baseline files checked: {len(baseline_versions)}")
            return 0
            
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())