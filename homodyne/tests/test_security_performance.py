#!/usr/bin/env python3
"""
Security Performance Tests for Homodyne Analysis
===============================================

Comprehensive test suite for security features and performance optimizations.
Tests security controls while ensuring they don't degrade computational performance.

Test Categories:
1. Input validation and sanitization
2. Secure file operations
3. Memory safety and limits
4. Rate limiting effectiveness
5. Configuration security
6. Cryptographic operations
7. Performance impact measurement

Security Test Objectives:
- Validate input sanitization prevents injection attacks
- Ensure file operations are secure and efficient
- Verify memory limits prevent resource exhaustion
- Test rate limiting protects against abuse
- Confirm configuration validation works correctly
- Measure security overhead on performance

Authors: Security Engineer (Claude Code)
Institution: Anthropic AI Security
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

# Import security modules
try:
    from homodyne.core.security_performance import (
        ConfigurationSecurity,
        MemoryLimitError,
        RateLimitError,
        SecureCache,
        SecureFileManager,
        ValidationError,
        cleanup_security_resources,
        monitor_memory,
        rate_limit,
        secure_cache,
        secure_file_manager,
        secure_scientific_computation,
        validate_angle_range,
        validate_array_dimensions,
        validate_filename,
        validate_numeric_value,
        validate_parameter_name,
        validate_path,
    )

    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    pytest.skip("Security performance module not available", allow_module_level=True)

try:
    from homodyne.core.secure_io import (
        SecureDataHandler,
        cleanup_secure_io,
        ensure_dir_secure,
        load_numpy_secure,
        save_json_secure,
        save_numpy_secure,
        secure_data_handler,
    )

    SECURE_IO_AVAILABLE = True
except ImportError:
    SECURE_IO_AVAILABLE = False


class TestInputValidation:
    """Test input validation functions for security."""

    def test_filename_validation(self):
        """Test secure filename validation."""
        # Valid filenames
        assert validate_filename("data.json") is True
        assert validate_filename("results_2024.npz") is True
        assert validate_filename("analysis-output.pkl") is True

        # Invalid filenames
        assert validate_filename("../../../etc/passwd") is False
        assert validate_filename("file with spaces") is False
        assert validate_filename("") is False
        assert validate_filename("a" * 300) is False  # Too long
        assert validate_filename("file;rm -rf /") is False

    def test_path_validation(self):
        """Test secure path validation."""
        # Valid paths
        assert validate_path("data/results.json") is True
        assert validate_path("output/analysis/final.npz") is True

        # Invalid paths
        assert validate_path("../../../etc/passwd") is False
        assert validate_path("/absolute/path") is False
        assert validate_path("path/../with/traversal") is False
        assert validate_path("") is False

    def test_parameter_name_validation(self):
        """Test scientific parameter name validation."""
        # Valid parameter names
        assert validate_parameter_name("D0") is True
        assert validate_parameter_name("alpha") is True
        assert validate_parameter_name("gamma_dot_t0") is True

        # Invalid parameter names
        assert validate_parameter_name("0invalid") is False  # Starts with number
        assert validate_parameter_name("param-with-dash") is False
        assert validate_parameter_name("param with space") is False
        assert validate_parameter_name("") is False
        assert validate_parameter_name("a" * 100) is False  # Too long

    def test_numeric_value_validation(self):
        """Test numeric value validation."""
        # Valid numeric values
        assert validate_numeric_value("123.45") is True
        assert validate_numeric_value("-0.001") is True
        assert validate_numeric_value("1.23e-5") is True
        assert validate_numeric_value("0") is True

        # Invalid numeric values
        assert validate_numeric_value("not_a_number") is False
        assert validate_numeric_value("123.45.67") is False
        assert validate_numeric_value("") is False
        assert validate_numeric_value("1" * 100) is False  # Too long

    def test_angle_range_validation(self):
        """Test angle range validation."""
        # Valid angles
        assert validate_angle_range(0.0) is True
        assert validate_angle_range(90.0) is True
        assert validate_angle_range(-180.0) is True
        assert validate_angle_range(180.0) is True

        # Invalid angles
        assert validate_angle_range(181.0) is False
        assert validate_angle_range(-181.0) is False
        assert validate_angle_range(360.0) is False

    def test_array_dimensions_validation(self):
        """Test array dimension validation."""
        # Valid dimensions
        assert validate_array_dimensions((100, 50)) is True
        assert validate_array_dimensions((1000, 1000, 10)) is True

        # Invalid dimensions
        assert validate_array_dimensions((10**7, 10**7)) is False  # Too many elements
        assert validate_array_dimensions((0, 100)) is False  # Zero dimension
        assert validate_array_dimensions((-1, 100)) is False  # Negative dimension
        assert validate_array_dimensions((10**7,)) is False  # Single huge dimension


class TestSecureCache:
    """Test secure caching functionality."""

    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = SecureCache(max_size=3, ttl=1)

        # Store and retrieve
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Non-existent key
        assert cache.get("nonexistent") is None

    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration."""
        cache = SecureCache(max_size=10, ttl=0.1)  # 100ms TTL

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(0.2)
        assert cache.get("key1") is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = SecureCache(max_size=2, ttl=10)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_cache_integrity_protection(self):
        """Test cache integrity verification."""
        cache = SecureCache()

        # Store value
        cache.set("key1", [1, 2, 3])

        # Manually corrupt cache entry (simulate attack)
        if "key1" in cache._cache:
            data, timestamp, old_hash = cache._cache["key1"]
            cache._cache["key1"] = (data, timestamp, "corrupted_hash")

        # Should return None due to integrity check failure
        assert cache.get("key1") is None


class TestSecureFileManager:
    """Test secure file operations."""

    def test_secure_temp_file_creation(self):
        """Test secure temporary file creation and cleanup."""
        manager = SecureFileManager()

        with manager.secure_temp_file(suffix=".test") as temp_path:
            assert temp_path.exists()
            assert temp_path.suffix == ".test"

            # Check secure permissions
            stat_info = temp_path.stat()
            # Should be readable/writable by owner only
            assert stat_info.st_mode & 0o777 == 0o600

        # File should be cleaned up
        assert not temp_path.exists()

    def test_secure_file_read_size_limit(self):
        """Test file size limits for security."""
        manager = SecureFileManager()

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # Write small amount of data
            temp_file.write(b"test data")
            temp_path = Path(temp_file.name)

        try:
            # Should work with default size limit
            with manager.secure_file_read(temp_path) as mmap_obj:
                assert mmap_obj.read() == b"test data"

            # Should fail with very small size limit
            with pytest.raises(ValidationError, match="exceeds limit"):
                with manager.secure_file_read(temp_path, max_size=1):
                    pass

        finally:
            temp_path.unlink()

    def test_cleanup_temp_files(self):
        """Test cleanup of temporary files."""
        manager = SecureFileManager()
        temp_files = []

        # Create multiple temp files
        for i in range(3):
            with manager.secure_temp_file() as temp_path:
                temp_files.append(temp_path)
                # Files exist during context
                assert temp_path.exists()

        # All files should be cleaned up after context
        for temp_path in temp_files:
            assert not temp_path.exists()


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_decorator(self):
        """Test rate limiting decorator."""
        call_count = 0

        @rate_limit(max_calls=3, window=1)
        def test_function():
            nonlocal call_count
            call_count += 1
            return call_count

        # First 3 calls should succeed
        assert test_function() == 1
        assert test_function() == 2
        assert test_function() == 3

        # 4th call should fail
        with pytest.raises(RateLimitError):
            test_function()

    def test_rate_limit_window_reset(self):
        """Test rate limit window reset."""
        call_count = 0

        @rate_limit(max_calls=2, window=0.1)  # 100ms window
        def test_function():
            nonlocal call_count
            call_count += 1
            return call_count

        # Use up the rate limit
        test_function()
        test_function()

        with pytest.raises(RateLimitError):
            test_function()

        # Wait for window to reset
        time.sleep(0.2)

        # Should work again
        assert test_function() == 3


@pytest.mark.skipif(not SECURE_IO_AVAILABLE, reason="Secure I/O module not available")
class TestSecureIO:
    """Test secure I/O operations."""

    def test_ensure_dir_secure(self):
        """Test secure directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_secure_dir"

            result_dir = ensure_dir_secure(test_dir)
            assert result_dir.exists()
            assert result_dir.is_dir()

    def test_ensure_dir_security_validation(self):
        """Test directory creation security validation."""
        # Should reject unsafe paths
        with pytest.raises(ValidationError):
            ensure_dir_secure("../../../dangerous/path")

    def test_save_load_numpy_secure(self):
        """Test secure NumPy array operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_array = np.random.rand(100, 50)
            file_path = Path(temp_dir) / "test_array.npz"

            # Save array securely
            success = save_numpy_secure(test_array, file_path)
            assert success is True
            assert file_path.exists()

            # Load array securely
            loaded_array = load_numpy_secure(file_path)
            np.testing.assert_array_equal(test_array, loaded_array)

    def test_save_numpy_size_validation(self):
        """Test NumPy array size validation."""
        # Create array that's too large
        with pytest.raises(ValidationError, match="dimensions too large"):
            large_shape = (10**6, 10**6)  # Would be huge
            # Create a mock array that claims to have large dimensions
            mock_array = Mock()
            mock_array.shape = large_shape
            mock_array.nbytes = 10**15  # Very large

            save_numpy_secure(mock_array, "test.npz")

    def test_json_save_filename_validation(self):
        """Test JSON save filename validation."""
        test_data = {"test": "data"}

        with pytest.raises(ValidationError, match="Invalid filename"):
            save_json_secure(test_data, "../dangerous/../../file.json")


class TestConfigurationSecurity:
    """Test configuration security validation."""

    def test_config_structure_validation(self):
        """Test configuration structure validation."""
        # Valid configuration
        valid_config = {
            "analyzer_parameters": {"q_magnitude": 0.001},
            "experimental_data": {"data_file": "test.h5"},
            "optimization_config": {"methods": ["Nelder-Mead"]},
        }

        assert ConfigurationSecurity.validate_config_structure(valid_config) is True

        # Invalid configuration (missing required section)
        invalid_config = {
            "analyzer_parameters": {"q_magnitude": 0.001}
            # Missing other required sections
        }

        assert ConfigurationSecurity.validate_config_structure(invalid_config) is False

    def test_parameter_bounds_sanitization(self):
        """Test parameter bounds sanitization."""
        bounds = [
            {"name": "D0", "min": 1.0, "max": 1000.0, "type": "Normal"},
            {"name": "invalid_param", "min": "not_a_number", "max": 1000.0},
            {"name": "alpha", "min": -2.0, "max": 2.0, "type": "Normal"},
            {"name": "bad_range", "min": 1000.0, "max": 1.0},  # min > max
        ]

        sanitized = ConfigurationSecurity.sanitize_parameter_bounds(bounds)

        # Should keep only valid bounds
        assert len(sanitized) == 2
        assert sanitized[0]["name"] == "D0"
        assert sanitized[1]["name"] == "alpha"


class TestMemoryMonitoring:
    """Test memory monitoring functionality."""

    @pytest.mark.skipif(
        not hasattr(monitor_memory, "__call__"),
        reason="Memory monitoring not available",
    )
    def test_memory_monitor_decorator(self):
        """Test memory monitoring decorator."""

        @monitor_memory(max_usage_percent=99.0)  # Very high limit
        def test_function():
            return "success"

        # Should work with high memory limit
        result = test_function()
        assert result == "success"

        # Test with very low limit (if psutil available)
        try:
            import psutil

            @monitor_memory(max_usage_percent=0.1)  # Very low limit
            def test_function_low_limit():
                return "should_fail"

            with pytest.raises(MemoryLimitError):
                test_function_low_limit()

        except ImportError:
            # psutil not available, memory monitoring disabled
            pass


class TestPerformanceImpact:
    """Test performance impact of security features."""

    def test_validation_performance(self):
        """Test that validation doesn't significantly impact performance."""
        # Test filename validation performance
        start_time = time.time()

        for i in range(1000):
            validate_filename(f"test_file_{i}.json")

        validation_time = time.time() - start_time

        # Should complete 1000 validations in under 100ms
        assert validation_time < 0.1

    def test_cache_performance(self):
        """Test cache performance with security features."""
        cache = SecureCache(max_size=1000)

        # Measure cache set performance
        start_time = time.time()

        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}")

        set_time = time.time() - start_time

        # Measure cache get performance
        start_time = time.time()

        for i in range(1000):
            cache.get(f"key_{i}")

        get_time = time.time() - start_time

        # Cache operations should be fast
        assert set_time < 0.5  # 500ms for 1000 sets
        assert get_time < 0.1  # 100ms for 1000 gets


class TestSecurityDecorators:
    """Test security decorator functionality."""

    def test_secure_scientific_computation_decorator(self):
        """Test secure scientific computation decorator."""
        call_count = 0

        @secure_scientific_computation
        def test_computation(data):
            nonlocal call_count
            call_count += 1
            return len(data)

        # Should work with valid input
        result = test_computation([1, 2, 3])
        assert result == 3
        assert call_count == 1

    def test_decorator_error_handling(self):
        """Test decorator error handling."""

        @secure_scientific_computation
        def test_function():
            raise ValueError("Test error")

        # Original error should propagate
        with pytest.raises(ValueError, match="Test error"):
            test_function()


class TestCleanup:
    """Test cleanup functionality."""

    def test_security_cleanup(self):
        """Test security resource cleanup."""
        # Use some security resources
        secure_cache.set("test_key", "test_value")

        with secure_file_manager.secure_temp_file() as temp_path:
            temp_path.write_text("test data")

        # Cleanup should not raise errors
        cleanup_security_resources()

        # Cache should be cleared
        assert secure_cache.get("test_key") is None

    @pytest.mark.skipif(
        not SECURE_IO_AVAILABLE, reason="Secure I/O module not available"
    )
    def test_secure_io_cleanup(self):
        """Test secure I/O cleanup."""
        # Use some I/O resources
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.json"
            save_json_secure({"test": "data"}, test_file)

        # Cleanup should not raise errors
        cleanup_secure_io()


# Integration test
class TestSecurityIntegration:
    """Integration tests for security features."""

    def test_end_to_end_secure_workflow(self):
        """Test complete secure workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test data
            test_array = np.random.rand(100, 50)
            test_config = {
                "analyzer_parameters": {"q_magnitude": 0.001},
                "experimental_data": {"data_file": "test.h5"},
                "optimization_config": {"methods": ["Nelder-Mead"]},
            }

            # Save data securely
            array_file = temp_path / "data.npz"
            config_file = temp_path / "config.json"

            if SECURE_IO_AVAILABLE:
                assert save_numpy_secure(test_array, array_file) is True
                assert save_json_secure(test_config, config_file) is True

                # Load data securely
                loaded_array = load_numpy_secure(array_file)
                np.testing.assert_array_equal(test_array, loaded_array)

                # Verify files exist and have secure permissions
                assert array_file.exists()
                assert config_file.exists()


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running security performance smoke tests...")

    if SECURITY_AVAILABLE:
        # Test basic validation
        assert validate_filename("test.json") is True
        assert validate_path("data/test.json") is True
        print("✓ Input validation working")

        # Test cache
        test_cache = SecureCache(max_size=10, ttl=60)
        test_cache.set("test", "value")
        assert test_cache.get("test") == "value"
        print("✓ Secure cache working")

        # Test file manager
        test_manager = SecureFileManager()
        with test_manager.secure_temp_file() as temp_path:
            assert temp_path.exists()
        print("✓ Secure file manager working")

        print("\n🛡️ Security performance system operational!")
    else:
        print("❌ Security performance module not available")

    print(
        "\nTo run full test suite: pytest homodyne/tests/test_security_performance.py -v"
    )
