#!/usr/bin/env python3
"""
Homodyne System Validator
==========================

Comprehensive testing and validation system for shell completion,
GPU acceleration, and overall system health.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a system validation check."""

    name: str
    success: bool
    message: str
    details: dict[str, Any] | None = None
    execution_time: float = 0.0
    warnings: list[str] | None = None


class SystemValidator:
    """Comprehensive system validation for homodyne installation."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: list[ValidationResult] = []
        self.environment_info: dict[str, Any] = {}

    def log(self, message: str, level: str = "info") -> None:
        """Log message if verbose."""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {level.upper()}: {message}")

    def run_command(self, cmd: list[str], timeout: int = 30) -> tuple[bool, str, str]:
        """Run shell command and return success, stdout, stderr."""
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return False, "", str(e)

    def test_environment_detection(self) -> ValidationResult:
        """Test environment detection and basic setup."""
        start_time = time.perf_counter()

        try:
            # Gather environment info
            self.environment_info = {
                "platform": os.uname().sysname,
                "python_version": sys.version.split()[0],
                "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
                "virtual_env": os.environ.get("VIRTUAL_ENV"),
                "shell": os.environ.get("SHELL", "").split("/")[-1],
                "cuda_home": os.environ.get("CUDA_HOME"),
                "path_dirs": len(os.environ.get("PATH", "").split(":")),
            }

            # Check if in virtual environment
            is_venv = (
                hasattr(sys, "real_prefix")
                or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
                or os.environ.get("CONDA_DEFAULT_ENV") is not None
            )

            warnings = []
            if not is_venv:
                warnings.append("Not running in a virtual environment")

            if self.environment_info["platform"] != "Linux":
                warnings.append("GPU acceleration only available on Linux")

            execution_time = time.perf_counter() - start_time

            return ValidationResult(
                name="Environment Detection",
                success=True,
                message=f"Detected: {self.environment_info['platform']}, "
                f"Python {self.environment_info['python_version']}, "
                f"Shell: {self.environment_info['shell']}",
                details=self.environment_info,
                execution_time=execution_time,
                warnings=warnings,
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                name="Environment Detection",
                success=False,
                message=f"Failed to detect environment: {e}",
                execution_time=execution_time,
            )

    def test_homodyne_installation(self) -> ValidationResult:
        """Test homodyne package installation."""
        start_time = time.perf_counter()

        try:
            # Check if homodyne commands exist
            commands = [
                "homodyne",
                "homodyne-config",
                "homodyne-gpu",
                "homodyne-post-install",
                "homodyne-cleanup",
            ]
            found_commands = []
            missing_commands = []

            for cmd in commands:
                if shutil.which(cmd):
                    found_commands.append(cmd)
                else:
                    missing_commands.append(cmd)

            # Test basic command execution
            success, stdout, stderr = self.run_command(["homodyne", "--help"])
            if not success:
                execution_time = time.perf_counter() - start_time
                return ValidationResult(
                    name="Homodyne Installation",
                    success=False,
                    message="homodyne --help failed",
                    details={"stdout": stdout, "stderr": stderr},
                    execution_time=execution_time,
                )

            # Check if help output looks correct
            if "homodyne scattering analysis" not in stdout.lower():
                execution_time = time.perf_counter() - start_time
                return ValidationResult(
                    name="Homodyne Installation",
                    success=False,
                    message="homodyne help output doesn't look correct",
                    details={"help_output": stdout[:200]},
                    execution_time=execution_time,
                )

            execution_time = time.perf_counter() - start_time
            warnings = []
            if missing_commands:
                warnings.append(f"Missing commands: {', '.join(missing_commands)}")

            return ValidationResult(
                name="Homodyne Installation",
                success=True,
                message=f"Found {len(found_commands)}/{len(commands)} commands",
                details={
                    "found_commands": found_commands,
                    "missing_commands": missing_commands,
                },
                execution_time=execution_time,
                warnings=warnings,
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                name="Homodyne Installation",
                success=False,
                message=f"Installation test failed: {e}",
                execution_time=execution_time,
            )

    def test_shell_completion(self) -> ValidationResult:
        """Test shell completion system."""
        start_time = time.perf_counter()

        try:
            venv_path = Path(sys.prefix)
            completion_files = []
            missing_files = []

            # Check for completion files
            expected_files = [
                venv_path / "etc" / "conda" / "activate.d" / "homodyne-completion.sh",
                venv_path / "etc" / "zsh" / "homodyne-completion.zsh",
                venv_path / "share" / "fish" / "vendor_completions.d" / "homodyne.fish",
            ]

            for file_path in expected_files:
                if file_path.exists():
                    completion_files.append(str(file_path))
                else:
                    missing_files.append(str(file_path))

            # Test if activation scripts work
            bash_test_passed = False
            if completion_files:
                # Try to source a completion file (safely)
                try:
                    # Create a test script to source completion
                    test_script = f"""
#!/bin/bash
source {completion_files[0]} 2>/dev/null || exit 1
# Test if alias was created
alias hm >/dev/null 2>&1 && echo "alias_works" || echo "alias_missing"
"""
                    success, stdout, stderr = self.run_command(
                        ["bash", "-c", test_script.strip()]
                    )
                    bash_test_passed = "alias_works" in stdout
                except Exception:
                    pass

            execution_time = time.perf_counter() - start_time
            warnings = []
            if missing_files:
                warnings.append(f"Missing completion files: {len(missing_files)} files")
            if not bash_test_passed:
                warnings.append("Shell aliases may not be working")

            success = len(completion_files) > 0
            message = f"Found {len(completion_files)} completion files"
            if bash_test_passed:
                message += " (aliases working)"

            return ValidationResult(
                name="Shell Completion",
                success=success,
                message=message,
                details={
                    "found_files": completion_files,
                    "missing_files": missing_files,
                    "alias_test_passed": bash_test_passed,
                },
                execution_time=execution_time,
                warnings=warnings,
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                name="Shell Completion",
                success=False,
                message=f"Shell completion test failed: {e}",
                execution_time=execution_time,
            )

    def test_gpu_setup(self) -> ValidationResult:
        """Test GPU setup and acceleration."""
        start_time = time.perf_counter()

        try:
            if self.environment_info.get("platform") != "Linux":
                execution_time = time.perf_counter() - start_time
                return ValidationResult(
                    name="GPU Setup",
                    success=True,
                    message="GPU not available on non-Linux platforms",
                    execution_time=execution_time,
                )

            # Check GPU files
            venv_path = Path(sys.prefix)
            gpu_files = []
            missing_files = []

            expected_files = [
                venv_path / "etc" / "conda" / "activate.d" / "homodyne-gpu.sh",
                venv_path / "etc" / "homodyne" / "gpu" / "gpu_activation.sh",
            ]

            for file_path in expected_files:
                if file_path.exists():
                    gpu_files.append(str(file_path))
                else:
                    missing_files.append(str(file_path))

            # Check NVIDIA GPU availability
            nvidia_available = shutil.which("nvidia-smi") is not None
            if nvidia_available:
                success, stdout, stderr = self.run_command(["nvidia-smi", "-L"])
                gpu_count = len([line for line in stdout.split("\n") if "GPU" in line])
            else:
                gpu_count = 0

            # Test JAX GPU support
            jax_gpu_available = False
            jax_error = None
            try:
                import jax

                devices = jax.devices()
                jax_gpu_available = any(
                    "gpu" in str(d).lower() or "cuda" in str(d).lower() for d in devices
                )
            except ImportError:
                jax_error = "JAX not installed"
            except Exception as e:
                jax_error = str(e)

            execution_time = time.perf_counter() - start_time
            warnings = []

            if missing_files:
                warnings.append(f"Missing GPU files: {len(missing_files)} files")
            if not nvidia_available:
                warnings.append("NVIDIA drivers not available")
            if jax_error:
                warnings.append(f"JAX issue: {jax_error}")

            # Determine success
            has_gpu_files = len(gpu_files) > 0
            success = has_gpu_files  # Success if GPU files are present

            details = {
                "gpu_files": gpu_files,
                "missing_files": missing_files,
                "nvidia_available": nvidia_available,
                "gpu_count": gpu_count,
                "jax_gpu_available": jax_gpu_available,
                "jax_error": jax_error,
            }

            if success and nvidia_available and jax_gpu_available:
                message = f"GPU ready: {gpu_count} GPU(s) with JAX support"
            elif success:
                message = f"GPU files installed ({len(gpu_files)} files)"
            else:
                message = "GPU setup not found"

            return ValidationResult(
                name="GPU Setup",
                success=success,
                message=message,
                details=details,
                execution_time=execution_time,
                warnings=warnings,
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                name="GPU Setup",
                success=False,
                message=f"GPU test failed: {e}",
                execution_time=execution_time,
            )

    def test_integration(self) -> ValidationResult:
        """Test integration between components."""
        start_time = time.perf_counter()

        try:
            # Test if post-install script works
            success, stdout, stderr = self.run_command(
                ["python", "-c", "from homodyne.post_install import main"]
            )

            post_install_works = success

            # Test cleanup script
            success, stdout, stderr = self.run_command(
                ["python", "-c", "from homodyne.uninstall_scripts import main"]
            )

            cleanup_works = success

            # Test import of main modules
            import_tests: dict[str, bool | str] = {}
            modules = [
                "homodyne.run_homodyne",
                "homodyne.create_config",
                "homodyne.post_install",
                "homodyne.uninstall_scripts",
            ]

            for module in modules:
                try:
                    __import__(module)
                    import_tests[module] = True
                except Exception as e:
                    import_tests[module] = f"Import failed: {e}"

            execution_time = time.perf_counter() - start_time

            success_count = sum(1 for v in import_tests.values() if v is True)
            total_count = len(import_tests)

            success = (
                post_install_works and cleanup_works and success_count == total_count
            )

            warnings = []
            if not post_install_works:
                warnings.append("Post-install script has issues")
            if not cleanup_works:
                warnings.append("Cleanup script has issues")
            if success_count < total_count:
                failed = [k for k, v in import_tests.items() if v is not True]
                warnings.append(f"Module import failures: {len(failed)}")

            return ValidationResult(
                name="Integration",
                success=success,
                message=f"Module imports: {success_count}/{total_count}",
                details={
                    "post_install_works": post_install_works,
                    "cleanup_works": cleanup_works,
                    "import_tests": import_tests,
                },
                execution_time=execution_time,
                warnings=warnings,
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                name="Integration",
                success=False,
                message=f"Integration test failed: {e}",
                execution_time=execution_time,
            )

    def run_all_tests(self) -> dict[str, ValidationResult]:
        """Run all system tests."""
        tests = [
            self.test_environment_detection,
            self.test_homodyne_installation,
            self.test_shell_completion,
            self.test_gpu_setup,
            self.test_integration,
        ]

        results = {}

        for test_func in tests:
            self.log(f"Running {test_func.__name__}...")
            result = test_func()
            results[result.name] = result
            self.results.append(result)

            status = "✅ PASS" if result.success else "❌ FAIL"
            self.log(f"{status}: {result.name} - {result.message}")

            if result.warnings:
                for warning in result.warnings:
                    self.log(f"⚠️  WARNING: {warning}")

        return results

    def generate_report(self) -> str:
        """Generate comprehensive validation report."""
        if not self.results:
            return "No tests have been run."

        report = []

        # Header
        report.append("=" * 80)
        report.append("🔍 HOMODYNE SYSTEM VALIDATION REPORT")
        report.append("=" * 80)

        # Summary
        passed = sum(1 for r in self.results if r.success)
        total = len(self.results)
        report.append(f"\n📊 Summary: {passed}/{total} tests passed")

        if passed == total:
            report.append("🎉 All systems operational!")
        else:
            report.append("⚠️  Some issues detected - see details below")

        # Environment info
        if self.environment_info:
            report.append("\n🖥️  Environment:")
            for key, value in self.environment_info.items():
                report.append(f"   {key}: {value}")

        # Test results
        report.append("\n📋 Test Results:")
        report.append("-" * 40)

        for result in self.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            report.append(f"\n{status} {result.name}")
            report.append(f"   Message: {result.message}")
            report.append(f"   Time: {result.execution_time:.3f}s")

            if result.warnings:
                report.append("   Warnings:")
                for warning in result.warnings:
                    report.append(f"     ⚠️  {warning}")

            if result.details and self.verbose:
                report.append("   Details:")
                for key, value in result.details.items():
                    if isinstance(value, list | dict):
                        report.append(
                            f"     {key}: {len(value) if isinstance(value, list) else 'dict'} items"
                        )
                    else:
                        report.append(f"     {key}: {value}")

        # Recommendations
        report.append("\n💡 Recommendations:")

        failed_tests = [r for r in self.results if not r.success]
        if failed_tests:
            report.append("   🔧 Fix failed tests:")
            for test in failed_tests:
                report.append(f"     • {test.name}: {test.message}")

        warnings_count = sum(len(r.warnings or []) for r in self.results)
        if warnings_count > 0:
            report.append(
                f"   ⚠️  Address {warnings_count} warnings for optimal performance"
            )

        if passed == total:
            report.append("   🚀 Your homodyne installation is ready!")
            report.append("   📖 Check documentation for usage examples")

        report.append("\n" + "=" * 80)

        return "\n".join(report)


def main():
    """Main function for system validation CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Homodyne System Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  homodyne-validate              # Quick validation
  homodyne-validate --verbose    # Detailed output
  homodyne-validate --json       # JSON output for automation
        """,
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "--test",
        choices=["env", "install", "completion", "gpu", "integration"],
        help="Run specific test only",
    )

    args = parser.parse_args()

    validator = SystemValidator(verbose=args.verbose)

    if args.test:
        # Run specific test
        test_map = {
            "env": validator.test_environment_detection,
            "install": validator.test_homodyne_installation,
            "completion": validator.test_shell_completion,
            "gpu": validator.test_gpu_setup,
            "integration": validator.test_integration,
        }

        result = test_map[args.test]()

        if args.json:
            print(json.dumps(asdict(result), indent=2))
        else:
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"{status} {result.name}: {result.message}")
            if result.warnings:
                for warning in result.warnings:
                    print(f"⚠️  {warning}")

        sys.exit(0 if result.success else 1)
    else:
        # Run all tests
        results = validator.run_all_tests()

        if args.json:
            json_results = {name: asdict(result) for name, result in results.items()}
            print(json.dumps(json_results, indent=2))
        else:
            print(validator.generate_report())

        # Exit with error code if any test failed
        sys.exit(0 if all(r.success for r in results.values()) else 1)


if __name__ == "__main__":
    main()
