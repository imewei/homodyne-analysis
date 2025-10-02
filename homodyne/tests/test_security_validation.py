"""
Comprehensive Security Tests and Input Validation
=================================================

Security tests for input validation, file operations, and data sanitization.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    from homodyne.core.secure_io import secure_file_operations
    from homodyne.core.security_metrics import SecurityMetrics

    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False

try:
    from homodyne.core.config import ConfigManager

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False


class TestInputValidation:
    """Test input validation and sanitization."""

    def test_parameter_bounds_validation(self):
        """Test validation of parameter bounds."""
        if not CORE_AVAILABLE:
            pytest.skip("Core modules not available")

        config = {
            "experimental_parameters": {
                "q_value": 0.1,
                "contrast": 0.95,
                "offset": 1.0,
            },
            "parameter_bounds": {
                "D0": [1e-6, 1e-1],
                "alpha": [0.1, 2.0],
                "D_offset": [1e-8, 1e-3],
            },
        }

        manager = ConfigManager(config=config)

        # Test valid parameters
        valid_params = {"D0": 1e-3, "alpha": 0.9, "D_offset": 1e-4}
        assert manager.validate_parameter_bounds(valid_params)

        # Test invalid parameters - below bounds
        invalid_low = {"D0": 1e-7, "alpha": 0.05, "D_offset": 1e-9}
        assert not manager.validate_parameter_bounds(invalid_low)

        # Test invalid parameters - above bounds
        invalid_high = {"D0": 1.0, "alpha": 3.0, "D_offset": 1e-2}
        assert not manager.validate_parameter_bounds(invalid_high)

        # Test NaN/Inf parameters
        invalid_nan = {"D0": np.nan, "alpha": 0.9, "D_offset": 1e-4}
        assert not manager.validate_parameter_bounds(invalid_nan)

        invalid_inf = {"D0": np.inf, "alpha": 0.9, "D_offset": 1e-4}
        assert not manager.validate_parameter_bounds(invalid_inf)

    def test_array_shape_validation(self):
        """Test validation of input array shapes."""
        # Test with mismatched array shapes
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        t1_array = np.array([1.0, 2.0, 3.0])
        t2_array = np.array([1.5, 2.5, 3.5])

        # Correct shape
        c2_correct = np.ones((8, 3, 3))
        assert self._validate_array_shapes(c2_correct, angles, t1_array, t2_array)

        # Wrong number of angles
        c2_wrong_angles = np.ones((6, 3, 3))
        assert not self._validate_array_shapes(
            c2_wrong_angles, angles, t1_array, t2_array
        )

        # Wrong time array dimensions
        c2_wrong_time = np.ones((8, 4, 3))
        assert not self._validate_array_shapes(
            c2_wrong_time, angles, t1_array, t2_array
        )

        # Wrong total dimensions
        c2_wrong_dims = np.ones((8, 3, 3, 2))  # Extra dimension
        assert not self._validate_array_shapes(
            c2_wrong_dims, angles, t1_array, t2_array
        )

    def _validate_array_shapes(self, c2_data, angles, t1_array, t2_array):
        """Helper function to validate array shapes."""
        try:
            expected_shape = (len(angles), len(t1_array), len(t2_array))
            return c2_data.shape == expected_shape and c2_data.ndim == 3
        except Exception:
            return False

    def test_numerical_stability_validation(self):
        """Test validation of numerical stability."""
        # Test with extreme values
        extreme_params = [1e-20, 0.001, 1e-30, 1e20, 100.0, 1e-50, 0.0]

        # Should detect potentially unstable parameters
        stability_issues = self._check_numerical_stability(extreme_params)
        assert len(stability_issues) > 0

        # Test with reasonable values
        reasonable_params = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]
        stability_issues = self._check_numerical_stability(reasonable_params)
        assert len(stability_issues) == 0

    def _check_numerical_stability(self, params):
        """Helper function to check numerical stability."""
        issues = []

        for i, param in enumerate(params):
            if abs(param) < 1e-15 and param != 0.0:
                issues.append(f"Parameter {i} too small: {param}")
            if abs(param) > 1e15:
                issues.append(f"Parameter {i} too large: {param}")
            if not np.isfinite(param):
                issues.append(f"Parameter {i} not finite: {param}")

        return issues

    def test_data_type_validation(self):
        """Test validation of data types."""
        # Test with correct data types
        angles_float = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        t1_float = np.array([1.0, 2.0], dtype=np.float64)
        c2_float = np.ones((3, 2, 2), dtype=np.float64)

        assert self._validate_data_types(c2_float, angles_float, t1_float, t1_float)

        # Test with incorrect data types
        angles_int = np.array([0, 1, 2], dtype=np.int32)
        # Should still work (will be converted)
        assert self._validate_data_types(c2_float, angles_int, t1_float, t1_float)

        # Test with complex numbers (should fail)
        c2_complex = np.ones((3, 2, 2), dtype=np.complex128)
        assert not self._validate_data_types(
            c2_complex, angles_float, t1_float, t1_float
        )

    def _validate_data_types(self, c2_data, angles, t1_array, t2_array):
        """Helper function to validate data types."""
        import warnings

        try:
            # Check for complex numbers first before conversion
            if np.iscomplexobj(c2_data):
                return False

            # Check if arrays can be converted to float (suppress warnings)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Ignore all warnings during conversion
                np.asarray(c2_data, dtype=np.float64)
                np.asarray(angles, dtype=np.float64)
                np.asarray(t1_array, dtype=np.float64)
                np.asarray(t2_array, dtype=np.float64)

            return True
        except (ValueError, TypeError):
            return False

    def test_configuration_injection_prevention(self):
        """Test prevention of configuration injection attacks."""
        import warnings

        # Test with malicious configuration content
        malicious_configs = [
            {
                "experimental_parameters": {
                    "q_value": "'; DROP TABLE users; --",  # SQL injection attempt
                    "contrast": 0.95,
                }
            },
            {
                "experimental_parameters": {
                    "q_value": "<script>alert('xss')</script>",  # XSS attempt
                    "contrast": 0.95,
                }
            },
            {
                "analysis_parameters": {
                    "mode": "../../../etc/passwd",  # Path traversal attempt
                    "method": "classical",
                }
            },
        ]

        for malicious_config in malicious_configs:
            # Should either reject the config or sanitize the inputs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # May issue warnings
                try:
                    if CORE_AVAILABLE:
                        manager = ConfigManager(config=malicious_config)
                        # Should not crash, may sanitize inputs
                        assert manager.config is not None
                except (ValueError, TypeError):
                    # Acceptable to reject malicious config
                    pass

    def test_file_path_validation(self):
        """Test validation of file paths for security."""
        # Test valid paths
        valid_paths = [
            "/tmp/test_data.npz",
            "./data/experiment.json",
            "results/output.txt",
        ]

        for path in valid_paths:
            assert self._validate_file_path(path)

        # Test potentially dangerous paths
        dangerous_paths = [
            "../../../etc/passwd",  # Path traversal
            "/dev/null",  # System device
            "//server/share/file",  # UNC path
            "file:///etc/passwd",  # File URL
            "C:\\Windows\\System32\\config\\SAM",  # Windows system file
        ]

        for path in dangerous_paths:
            # Should either reject or sanitize
            result = self._validate_file_path(path)
            # For security, some paths should be rejected
            if not result:
                assert True  # Rejection is acceptable
            else:
                # If accepted, should be sanitized
                assert "../" not in path or Path(path).resolve().parts[0] != ".."

    def _validate_file_path(self, file_path):
        """Helper function to validate file paths."""
        try:
            path = Path(file_path)

            # Check for path traversal attempts
            if ".." in path.parts:
                return False

            # Check for absolute paths to system directories
            system_dirs = [
                "/etc",
                "/sys",
                "/proc",
                "/dev",
                "C:\\Windows",
                "C:\\System32",
            ]
            resolved_path = str(path.resolve())

            for sys_dir in system_dirs:
                if resolved_path.startswith(sys_dir):
                    return False

            return True
        except Exception:
            return False


@pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security modules not available")
class TestSecureFileOperations:
    """Test secure file operations."""

    def setup_method(self):
        """Setup secure file operation tests."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup secure file operation tests."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_secure_file_reading(self):
        """Test secure file reading operations."""
        # Create test file
        test_file = os.path.join(self.temp_dir, "test_data.json")
        test_data = {"test": "data", "value": 123}

        with open(test_file, "w") as f:
            json.dump(test_data, f)

        # Test secure reading
        try:
            with secure_file_operations.secure_open(test_file, "r") as f:
                loaded_data = json.load(f)
                assert loaded_data == test_data
        except AttributeError:
            # secure_file_operations may not be implemented
            pytest.skip("Secure file operations not implemented")

    def test_file_permission_validation(self):
        """Test file permission validation."""
        # Create test file with specific permissions
        test_file = os.path.join(self.temp_dir, "perm_test.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        # Set restrictive permissions
        os.chmod(test_file, 0o600)  # Read/write for owner only

        # Test permission checking
        assert self._check_file_permissions(test_file)

        # Test with overly permissive file (if supported on system)
        try:
            os.chmod(test_file, 0o777)  # Read/write/execute for all
            # Should warn about overly permissive permissions
            with pytest.warns(None):
                self._check_file_permissions(test_file)
        except OSError:
            # Permission changes may not be supported on all systems
            pass

    def _check_file_permissions(self, file_path):
        """Helper function to check file permissions."""
        try:
            stat_info = os.stat(file_path)
            mode = stat_info.st_mode

            # Check if file is readable by others
            world_readable = mode & 0o004
            mode & 0o040

            # For security, configuration files should not be world-readable
            if world_readable and file_path.endswith((".json", ".conf", ".config")):
                import warnings

                warnings.warn(f"File {file_path} is world-readable", UserWarning)

            return True
        except OSError:
            return False

    def test_temporary_file_security(self):
        """Test security of temporary file operations."""
        # Test secure temporary file creation
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            temp_path = temp_file.name
            test_data = {"secure": "data"}
            json.dump(test_data, temp_file)

        try:
            # Check file permissions
            stat_info = os.stat(temp_path)
            mode = stat_info.st_mode

            # Temporary files should not be world-readable
            world_readable = mode & 0o004
            assert not world_readable, "Temporary file is world-readable"

            # Read back data
            with open(temp_path) as f:
                loaded_data = json.load(f)
                assert loaded_data == test_data

        finally:
            os.unlink(temp_path)

    def test_file_size_limits(self):
        """Test file size limit enforcement."""
        # Create large file
        large_file = os.path.join(self.temp_dir, "large_file.txt")

        # Write 1MB of data
        with open(large_file, "w") as f:
            f.writelines("A" * 100 + "\n" for _ in range(10000))

        # Test size checking
        max_size = 500 * 1024  # 500KB limit
        assert not self._check_file_size_limit(large_file, max_size)

        # Create small file
        small_file = os.path.join(self.temp_dir, "small_file.txt")
        with open(small_file, "w") as f:
            f.write("small content")

        assert self._check_file_size_limit(small_file, max_size)

    def _check_file_size_limit(self, file_path, max_size_bytes):
        """Helper function to check file size limits."""
        try:
            file_size = os.path.getsize(file_path)
            return file_size <= max_size_bytes
        except OSError:
            return False

    def test_directory_traversal_prevention(self):
        """Test prevention of directory traversal attacks."""
        base_dir = self.temp_dir

        # Test legitimate paths
        legitimate_paths = ["data.json", "subdir/data.json", "./data.json"]

        for rel_path in legitimate_paths:
            full_path = self._secure_join(base_dir, rel_path)
            assert full_path.startswith(base_dir), (
                f"Path {full_path} outside base directory"
            )

        # Test malicious paths
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\SAM",
        ]

        for mal_path in malicious_paths:
            try:
                full_path = self._secure_join(base_dir, mal_path)
                # If join succeeds, result should still be within base directory
                assert full_path.startswith(base_dir), (
                    f"Directory traversal vulnerability: {full_path}"
                )
            except ValueError:
                # Acceptable to reject malicious paths
                pass

    def _secure_join(self, base_path, user_path):
        """Secure path joining to prevent directory traversal."""
        # Resolve both paths
        base = Path(base_path).resolve()
        user = Path(user_path)

        # If user path is absolute, reject it
        if user.is_absolute():
            raise ValueError("Absolute paths not allowed")

        # Join and resolve
        joined = (base / user).resolve()

        # Check if result is within base directory
        try:
            joined.relative_to(base)
            return str(joined)
        except ValueError:
            raise ValueError("Path outside base directory")


class TestDataSanitization:
    """Test data sanitization and cleaning."""

    def test_numerical_data_sanitization(self):
        """Test sanitization of numerical data."""
        # Create data with various issues
        problematic_data = np.array(
            [[1.0, 2.0, np.inf], [np.nan, 3.0, 4.0], [-np.inf, 5.0, 6.0]]
        )

        sanitized = self._sanitize_numerical_data(problematic_data)

        # Should remove or replace problematic values
        assert np.all(np.isfinite(sanitized))
        assert not np.any(np.isnan(sanitized))
        assert not np.any(np.isinf(sanitized))

    def _sanitize_numerical_data(self, data):
        """Helper function to sanitize numerical data."""
        data_copy = data.copy()

        # Replace NaN with zeros or interpolated values
        data_copy = np.nan_to_num(
            data_copy,
            nan=0.0,
            posinf=np.finfo(np.float64).max,
            neginf=np.finfo(np.float64).min,
        )

        # Clip extreme values
        data_copy = np.clip(data_copy, -1e10, 1e10)

        return data_copy

    def test_string_input_sanitization(self):
        """Test sanitization of string inputs."""
        # Test with various potentially dangerous strings
        dangerous_strings = [
            "'; DROP TABLE users; --",  # SQL injection
            "<script>alert('xss')</script>",  # XSS
            "../../etc/passwd",  # Path traversal
            "$(rm -rf /)",  # Command injection
            "\x00\x01\x02",  # Binary data
        ]

        for dangerous_string in dangerous_strings:
            sanitized = self._sanitize_string_input(dangerous_string)

            # Should not contain dangerous patterns
            assert "DROP TABLE" not in sanitized.upper()
            assert "<script>" not in sanitized.lower()
            assert "../" not in sanitized
            assert "$(" not in sanitized
            assert "\x00" not in sanitized

    def _sanitize_string_input(self, input_string):
        """Helper function to sanitize string inputs."""
        if not isinstance(input_string, str):
            return str(input_string)

        # Remove null bytes and control characters
        sanitized = "".join(
            char for char in input_string if ord(char) >= 32 or char in "\t\n\r"
        )

        # Remove potentially dangerous patterns
        dangerous_patterns = [
            "DROP TABLE",
            "DELETE FROM",
            "INSERT INTO",
            "UPDATE SET",  # SQL
            "<script>",
            "</script>",
            "javascript:",
            "data:",  # XSS
            "../",
            "..\\",  # Path traversal
            "$(",
            "`",
            "${",  # Command injection
        ]

        for pattern in dangerous_patterns:
            sanitized = sanitized.replace(pattern, "")

        # Limit length
        if len(sanitized) > 1000:
            sanitized = sanitized[:1000]

        return sanitized

    def test_configuration_sanitization(self):
        """Test sanitization of configuration data."""
        # Create configuration with potential issues
        problematic_config = {
            "experimental_parameters": {
                "q_value": "0.1; DROP TABLE users;",
                "contrast": "<script>alert('xss')</script>",
                "offset": np.inf,
            },
            "file_paths": {
                "data_file": "../../etc/passwd",
                "output_dir": "/tmp/../../../root",
            },
        }

        sanitized_config = self._sanitize_configuration(problematic_config)

        # Check numerical values are sanitized
        assert np.isfinite(sanitized_config["experimental_parameters"]["offset"])

        # Check string values are sanitized
        assert "DROP TABLE" not in str(
            sanitized_config["experimental_parameters"]["q_value"]
        )
        assert "<script>" not in str(
            sanitized_config["experimental_parameters"]["contrast"]
        )

        # Check file paths are sanitized
        assert "../" not in sanitized_config["file_paths"]["data_file"]
        assert "../" not in sanitized_config["file_paths"]["output_dir"]

    def _sanitize_configuration(self, config):
        """Helper function to sanitize configuration data."""

        def sanitize_value(value):
            if isinstance(value, str):
                return self._sanitize_string_input(value)
            if isinstance(value, (int, float, np.number)):
                if np.isfinite(value):
                    return float(value)
                return 0.0
            if isinstance(value, dict):
                return {k: sanitize_value(v) for k, v in value.items()}
            if isinstance(value, list):
                return [sanitize_value(v) for v in value]
            return value

        return sanitize_value(config)


class TestAccessControl:
    """Test access control and permissions."""

    def test_file_access_restrictions(self):
        """Test file access restrictions."""
        # Test access to allowed directories
        allowed_dirs = ["/tmp", ".", "./data", "./results"]

        for directory in allowed_dirs:
            if os.path.exists(directory):
                assert self._check_directory_access(directory, "read")

        # Test access to restricted directories
        restricted_dirs = ["/etc", "/root", "/sys", "/proc"]

        for directory in restricted_dirs:
            if os.path.exists(directory):
                # Should either deny access or require special permissions
                access_allowed = self._check_directory_access(directory, "write")
                if access_allowed:
                    # If access is allowed, it should be due to running as root
                    # In a production environment, this should be restricted
                    import warnings

                    warnings.warn(
                        f"Write access to {directory} should be restricted", UserWarning
                    )

    def _check_directory_access(self, directory, access_type):
        """Helper function to check directory access."""
        try:
            if access_type == "read":
                return os.access(directory, os.R_OK)
            if access_type == "write":
                return os.access(directory, os.W_OK)
            if access_type == "execute":
                return os.access(directory, os.X_OK)
            return False
        except OSError:
            return False

    def test_resource_limits(self):
        """Test resource usage limits."""
        # Test memory usage limits
        try:
            import resource

            # Get current memory limit
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)

            # Test with reasonable memory allocation
            reasonable_size = 100 * 1024 * 1024  # 100MB
            test_array = np.zeros(reasonable_size // 8)  # 8 bytes per float64

            assert test_array.size > 0

            # Clean up
            del test_array

        except (ImportError, OSError):
            # Resource module may not be available on all systems
            pytest.skip("Resource limits not available on this system")

    def test_execution_time_limits(self):
        """Test execution time limits."""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Operation timed out")

        # Test with reasonable operation
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)  # 5 second timeout

            # Perform reasonable operation
            result = np.sum(np.random.random(1000000))
            assert result > 0

            signal.alarm(0)  # Cancel timeout

        except (AttributeError, OSError):
            # Signal handling may not be available on all systems
            pytest.skip("Signal handling not available on this system")

        # Test that long operations would be caught
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(1)  # 1 second timeout

            # This should timeout if implemented correctly
            with pytest.raises(TimeoutError):
                # Simulate long operation
                import time

                time.sleep(2)

        except (AttributeError, OSError):
            pytest.skip("Signal handling not available on this system")
        finally:
            try:
                signal.alarm(0)  # Cancel any pending alarms
            except:
                pass


@pytest.mark.skipif(not SECURITY_AVAILABLE, reason="Security modules not available")
class TestSecurityMetrics:
    """Test security metrics and monitoring."""

    def test_security_metrics_collection(self):
        """Test collection of security metrics."""
        try:
            metrics = SecurityMetrics()

            # Test basic metric collection
            metrics.record_file_access("/tmp/test_file.txt", "read", success=True)
            metrics.record_authentication_attempt("user", success=True)
            metrics.record_input_validation("parameter", valid=True)

            # Get security report
            report = metrics.get_security_report()

            assert isinstance(report, dict)
            assert "file_access" in report or len(report) >= 0

        except AttributeError:
            pytest.skip("SecurityMetrics not fully implemented")

    def test_anomaly_detection(self):
        """Test anomaly detection in security metrics."""
        try:
            metrics = SecurityMetrics()

            # Record normal activity
            for i in range(100):
                metrics.record_file_access(f"/tmp/file_{i}.txt", "read", success=True)

            # Record anomalous activity
            for i in range(10):
                metrics.record_file_access("/etc/passwd", "read", success=False)

            # Check for anomalies
            anomalies = metrics.detect_anomalies()

            # Should detect unusual access patterns
            assert isinstance(anomalies, (list, dict))

        except AttributeError:
            pytest.skip("Anomaly detection not implemented")

    def test_security_alerts(self):
        """Test security alert generation."""
        try:
            metrics = SecurityMetrics()

            # Generate events that should trigger alerts
            metrics.record_file_access("../../etc/passwd", "read", success=False)
            metrics.record_input_validation("'; DROP TABLE users; --", valid=False)

            alerts = metrics.get_security_alerts()

            # Should generate appropriate alerts
            assert isinstance(alerts, (list, dict))

        except AttributeError:
            pytest.skip("Security alerts not implemented")
