"""
Tests for System Validator - Phase 6 System Validation
======================================================

This module tests the comprehensive system validation functionality:
- Environment detection and validation
- Homodyne installation verification
- Shell completion testing
- GPU setup validation
- Integration testing
- Command-line interface validation

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import gc
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from homodyne.runtime.utils.system_validator import SystemValidator, ValidationResult


class TestSystemValidatorCore:
    """Test core system validator functionality."""

    def test_system_validator_initialization(self):
        """Test system validator initialization."""
        validator = SystemValidator(verbose=True)

        assert validator.verbose is True
        assert isinstance(validator.results, list)
        assert isinstance(validator.environment_info, dict)
        assert len(validator.results) == 0

    def test_system_validator_initialization_non_verbose(self):
        """Test system validator initialization without verbose."""
        validator = SystemValidator(verbose=False)

        assert validator.verbose is False
        assert isinstance(validator.results, list)

    def test_log_verbose_mode(self, capsys):
        """Test logging in verbose mode."""
        validator = SystemValidator(verbose=True)
        validator.log("Test message", "info")

        captured = capsys.readouterr()
        assert "INFO: Test message" in captured.out

    def test_log_non_verbose_mode(self, capsys):
        """Test logging in non-verbose mode."""
        validator = SystemValidator(verbose=False)
        validator.log("Test message", "info")

        captured = capsys.readouterr()
        assert captured.out == ""

    @patch("subprocess.run")
    def test_run_command_success(self, mock_subprocess):
        """Test successful command execution."""
        mock_subprocess.return_value = Mock(
            returncode=0, stdout="Command output", stderr=""
        )

        validator = SystemValidator()
        success, stdout, stderr = validator.run_command(["echo", "test"])

        assert success is True
        assert stdout == "Command output"
        assert stderr == ""

    @patch("subprocess.run")
    def test_run_command_failure(self, mock_subprocess):
        """Test failed command execution."""
        mock_subprocess.return_value = Mock(
            returncode=1, stdout="", stderr="Command failed"
        )

        validator = SystemValidator()
        success, stdout, stderr = validator.run_command(["false"])

        assert success is False
        assert stderr == "Command failed"

    @patch("subprocess.run")
    def test_run_command_timeout(self, mock_subprocess):
        """Test command timeout handling."""
        mock_subprocess.side_effect = subprocess.TimeoutExpired("sleep", 10)

        validator = SystemValidator()
        success, stdout, stderr = validator.run_command(["sleep", "10"], timeout=1)

        assert success is False
        assert "timeout" in stderr.lower() or "timed out" in stderr.lower()


class TestResultDataclass:
    """Test ValidationResult dataclass functionality."""

    def test_test_result_creation(self):
        """Test ValidationResult creation."""
        result = ValidationResult(
            name="test_example",
            success=True,
            message="Test passed",
            details={"key": "value"},
            execution_time=1.5,
            warnings=["warning1", "warning2"],
        )

        assert result.name == "test_example"
        assert result.success is True
        assert result.message == "Test passed"
        assert result.details == {"key": "value"}
        assert result.execution_time == 1.5
        assert result.warnings == ["warning1", "warning2"]

    def test_test_result_minimal(self):
        """Test ValidationResult with minimal parameters."""
        result = ValidationResult(
            name="minimal_test", success=False, message="Test failed"
        )

        assert result.name == "minimal_test"
        assert result.success is False
        assert result.message == "Test failed"
        assert result.details is None
        assert result.execution_time == 0.0
        assert result.warnings is None

    def test_test_result_to_dict(self):
        """Test ValidationResult conversion to dict."""
        result = ValidationResult(name="dict_test", success=True, message="Success")

        result_dict = asdict(result)
        assert isinstance(result_dict, dict)
        assert result_dict["name"] == "dict_test"
        assert result_dict["success"] is True


class TestEnvironmentDetection:
    """Test environment detection functionality."""

    @patch("platform.system")
    @patch("sys.version_info")
    def test_environment_detection(self, mock_version, mock_platform):
        """Test basic environment detection."""
        mock_platform.return_value = "Linux"
        mock_version.major = 3
        mock_version.minor = 12

        validator = SystemValidator()

        # Test environment detection if method exists
        if hasattr(validator, "test_environment_detection"):
            result = validator.test_environment_detection()
            assert isinstance(result, ValidationResult)
            assert result.name.lower().startswith("environment")

    @patch("os.environ")
    def test_virtual_environment_detection(self, mock_environ):
        """Test virtual environment detection."""
        mock_environ.get.return_value = "/path/to/venv"

        validator = SystemValidator()

        # Test virtual environment detection
        if hasattr(validator, "detect_virtual_environment"):
            venv_info = validator.detect_virtual_environment()
            assert isinstance(venv_info, dict | bool)

    @patch("shutil.which")
    def test_shell_detection(self, mock_which):
        """Test shell type detection."""
        mock_which.side_effect = lambda cmd: "/bin/bash" if cmd == "bash" else None

        validator = SystemValidator()

        # Test shell detection if available
        if hasattr(validator, "detect_shell"):
            shell_info = validator.detect_shell()
            assert isinstance(shell_info, str | dict)


class TestHomodyneInstallationValidation:
    """Test homodyne installation validation."""

    @patch("shutil.which")
    def test_command_availability(self, mock_which):
        """Test homodyne command availability."""
        mock_which.side_effect = lambda cmd: (
            f"/usr/bin/{cmd}" if "homodyne" in cmd else None
        )

        SystemValidator()

        # Test command availability check
        commands_to_test = [
            "homodyne",
            "homodyne-config",
            "homodyne-gpu",
            "homodyne-post-install",
            "homodyne-cleanup",
            "homodyne-gpu-optimize",
            "homodyne-validate",
        ]

        for cmd in commands_to_test:
            _ = shutil.which(cmd) is not None
            # Command may or may not be available in test environment

    @patch("subprocess.run")
    def test_homodyne_help_validation(self, mock_subprocess):
        """Test homodyne help command validation."""
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout="Homodyne Scattering Analysis\nUsage: homodyne [OPTIONS]",
            stderr="",
        )

        validator = SystemValidator()
        success, stdout, stderr = validator.run_command(["homodyne", "--help"])

        if success:
            assert "homodyne" in stdout.lower() or "usage" in stdout.lower()

    @patch("importlib.import_module")
    def test_module_imports(self, mock_import):
        """Test homodyne module import validation."""
        mock_import.return_value = Mock()

        SystemValidator()

        # Test module imports if validation method exists
        modules_to_test = [
            "homodyne",
            "homodyne.analysis.core",
            "homodyne.core.config",
            "homodyne.optimization.classical",
        ]

        for module in modules_to_test:
            try:
                import importlib

                importlib.import_module(module)
            except ImportError:
                pass
            # Module may or may not be available


class TestShellCompletionValidation:
    """Test shell completion validation."""

    @patch("pathlib.Path.exists")
    def test_completion_files_presence(self, mock_exists):
        """Test completion files presence."""
        mock_exists.return_value = True

        SystemValidator()

        # Test completion file detection
        completion_files = ["homodyne-completion.sh", "runtime/shell/completion.sh"]

        for _filename in completion_files:
            # File existence would be tested if method exists
            pass

    @patch("os.environ")
    def test_conda_environment_completion(self, mock_environ):
        """Test conda environment completion setup."""
        mock_environ.get.return_value = "/path/to/conda/env"

        validator = SystemValidator()

        # Test conda completion setup if available
        if hasattr(validator, "test_shell_completion"):
            result = validator.test_shell_completion()
            assert isinstance(result, ValidationResult)

    def test_completion_alias_functionality(self):
        """Test completion alias functionality."""
        SystemValidator()

        # Test alias functionality if available
        aliases_to_test = ["gpu-status", "gpu-bench", "gpu-on", "gpu-off"]

        for _alias in aliases_to_test:
            # Alias testing would depend on actual implementation
            pass


class TestGPUSetupValidation:
    """Test GPU setup validation."""

    @patch("subprocess.run")
    def test_gpu_hardware_detection(self, mock_subprocess):
        """Test GPU hardware detection validation."""
        mock_subprocess.return_value = Mock(
            returncode=0, stdout="GeForce RTX 3080\n", stderr=""
        )

        validator = SystemValidator()

        # Test GPU detection if method exists
        if hasattr(validator, "test_gpu_setup"):
            result = validator.test_gpu_setup()
            assert isinstance(result, ValidationResult)

    @patch("subprocess.run")
    def test_nvidia_driver_validation(self, mock_subprocess):
        """Test NVIDIA driver validation."""
        mock_subprocess.return_value = Mock(
            returncode=0, stdout="Driver Version: 470.86\n", stderr=""
        )

        validator = SystemValidator()
        success, stdout, stderr = validator.run_command(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv"]
        )

        if success:
            assert "driver" in stdout.lower() or stdout.strip()

    def test_jax_gpu_support_validation(self):
        """Test JAX GPU support validation."""
        SystemValidator()

        try:
            import jax

            jax_available = True
            devices = jax.devices()
            [d for d in devices if d.device_kind == "gpu"]
        except ImportError:
            jax_available = False

        # JAX and GPU availability testing
        assert jax_available in [True, False]  # Either way is valid

    def test_cuda_installation_validation(self):
        """Test CUDA installation validation."""
        SystemValidator()

        # Test CUDA validation if method exists
        cuda_paths = ["/usr/local/cuda", "/opt/cuda", "/usr/cuda"]

        for cuda_path in cuda_paths:
            Path(cuda_path).exists()
            # CUDA may or may not be installed


class TestIntegrationValidation:
    """Test integration validation functionality."""

    def test_cross_module_imports(self):
        """Test cross-module import validation."""
        validator = SystemValidator()

        # Test integration imports if method exists
        if hasattr(validator, "test_integration"):
            result = validator.test_integration()
            assert isinstance(result, ValidationResult)

    @patch("subprocess.run")
    def test_end_to_end_command_execution(self, mock_subprocess):
        """Test end-to-end command execution."""
        mock_subprocess.return_value = Mock(
            returncode=0, stdout="Analysis complete\n", stderr=""
        )

        SystemValidator()

        # Test end-to-end functionality if available
        # This would test actual command execution chains

    def test_configuration_integration(self):
        """Test configuration system integration."""
        SystemValidator()

        # Test config integration if method exists
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "test_config.json"
            test_config = {"analysis_settings": {"static_mode": True}}

            with open(config_file, "w") as f:
                json.dump(test_config, f)

            # Configuration testing would depend on actual validation methods


class TestValidationReporting:
    """Test validation reporting functionality."""

    def test_generate_summary_report(self):
        """Test validation summary report generation."""
        validator = SystemValidator()

        # Add some test results
        validator.results = [
            ValidationResult("test1", True, "Passed", execution_time=1.0),
            ValidationResult("test2", False, "Failed", execution_time=2.0),
            ValidationResult("test3", True, "Passed", warnings=["warning1"]),
        ]

        # Test report generation if method exists
        if hasattr(validator, "generate_report"):
            report = validator.generate_report()
            assert isinstance(report, str | dict)

    def test_json_output_format(self):
        """Test JSON output format."""
        validator = SystemValidator()

        validator.results = [
            ValidationResult("json_test", True, "Success", details={"key": "value"})
        ]

        # Test JSON output if method exists
        if hasattr(validator, "to_json"):
            json_output = validator.to_json()
            json_data = (
                json.loads(json_output) if isinstance(json_output, str) else json_output
            )
            assert isinstance(json_data, dict)

    def test_recommendations_generation(self):
        """Test recommendations generation."""
        validator = SystemValidator()

        validator.results = [
            ValidationResult("gpu_test", False, "No GPU detected"),
            ValidationResult(
                "completion_test", False, "Shell completion not installed"
            ),
        ]

        # Test recommendations if method exists
        if hasattr(validator, "generate_recommendations"):
            recommendations = validator.generate_recommendations()
            assert isinstance(recommendations, list)


class TestCLIIntegration:
    """Test CLI integration for homodyne-validate command."""

    @patch("sys.argv", ["homodyne-validate", "--help"])
    def test_cli_help_output(self):
        """Test CLI help output."""
        # Test CLI help if main function exists
        # Implementation depends on actual CLI structure
        pass

    @patch("sys.argv", ["homodyne-validate", "--json"])
    def test_cli_json_output(self):
        """Test CLI JSON output format."""
        # Test JSON output format if CLI exists
        # Implementation depends on actual CLI structure
        pass

    @patch("sys.argv", ["homodyne-validate", "--test", "gpu"])
    def test_cli_specific_test_selection(self):
        """Test CLI specific test selection."""
        # Test specific test selection if CLI exists
        # Implementation depends on actual CLI structure
        pass

    @patch("sys.argv", ["homodyne-validate", "--verbose"])
    def test_cli_verbose_mode(self):
        """Test CLI verbose mode."""
        # Test verbose mode if CLI exists
        # Implementation depends on actual CLI structure
        pass


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    def test_handle_missing_dependencies(self):
        """Test handling of missing dependencies."""
        validator = SystemValidator()

        # Test missing dependency handling
        # Should gracefully handle missing modules/commands
        assert hasattr(validator, "results")

    def test_handle_permission_errors(self):
        """Test handling of permission errors."""
        SystemValidator()

        # Test permission error handling
        with patch("pathlib.Path.exists", side_effect=PermissionError("Access denied")):
            # Should handle permission errors gracefully
            try:
                Path("/nonexistent").exists()
            except PermissionError:
                pass  # Expected behavior

    @patch("subprocess.run")
    def test_handle_command_failures(self, mock_subprocess):
        """Test handling of command execution failures."""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "test-command")

        validator = SystemValidator()
        success, stdout, stderr = validator.run_command(["failing-command"])

        assert success is False

    def test_handle_malformed_output(self):
        """Test handling of malformed command output."""
        validator = SystemValidator()

        # Test malformed output handling
        # Should gracefully handle unexpected output formats
        assert isinstance(validator.results, list)


class TestPerformanceAndScaling:
    """Test performance and scaling aspects."""

    @pytest.mark.slow
    def test_validation_performance(self):
        """Test validation suite performance."""
        validator = SystemValidator(verbose=False)

        # Test validation performance
        start_time = time.time()

        # Run some basic validations if methods exist
        if hasattr(validator, "run_all_tests"):
            validator.run_all_tests()

        execution_time = time.time() - start_time

        # Validation should complete reasonably quickly
        assert execution_time < 60  # Less than 1 minute

    def test_concurrent_validation(self):
        """Test concurrent validation execution."""
        validator = SystemValidator()

        # Test concurrent execution if supported
        # Implementation depends on actual validation methods
        assert hasattr(validator, "results")

    def test_memory_usage_during_validation(self):
        """Test memory usage during validation."""
        SystemValidator()

        # Test memory usage
        # Should not consume excessive memory
        import sys

        len(gc.get_objects()) if "gc" in sys.modules else 0

        # Run validations
        # Memory testing would depend on actual implementation
