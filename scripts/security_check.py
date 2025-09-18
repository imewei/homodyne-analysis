#!/usr/bin/env python3
"""
Security check script for GitHub Actions
Ensures proper configuration is used for all security tools
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n🔍 {description}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} - PASSED")
        if result.stdout.strip():
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def main():
    """Run all security checks with proper configuration"""

    # Change to project root
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    print("🛡️  Running Homodyne Security Checks")
    print(f"📁 Project root: {project_root}")

    # List of security checks to run
    checks = [
        {
            "cmd": ["bandit", "-r", "homodyne/", "--configfile", "pyproject.toml"],
            "description": "Bandit security scanning with pyproject.toml config",
        },
        {
            "cmd": ["pip-audit", "--requirement", "requirements.txt", "--desc"],
            "description": "pip-audit dependency vulnerability scanning",
        },
    ]

    failed_checks = []

    for check in checks:
        if not run_command(check["cmd"], check["description"]):
            failed_checks.append(check["description"])

    # Summary
    print(f"\n📊 Security Check Summary")
    print(f"Total checks: {len(checks)}")
    print(f"Passed: {len(checks) - len(failed_checks)}")
    print(f"Failed: {len(failed_checks)}")

    if failed_checks:
        print(f"\n❌ Failed checks:")
        for check in failed_checks:
            print(f"  - {check}")
        sys.exit(1)
    else:
        print(f"\n✅ All security checks passed!")
        print("🎉 No security issues found!")
        sys.exit(0)


if __name__ == "__main__":
    main()
