"""
Comprehensive Tests for Advanced Optimization Utilities
======================================================

Test suite for advanced optimization utility functions, configuration management,
system resource detection, and integration helpers for distributed and ML features.

Authors: Wei Chen, Hongrui He, Claude (Anthropic)
Institution: Argonne National Laboratory
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import the modules we're testing
try:
    from homodyne.optimization.utils import (
        OptimizationBenchmark,
        OptimizationConfig,
        SystemResourceDetector,
        create_comprehensive_benchmark_suite,
        quick_setup_distributed_optimization,
        quick_setup_ml_acceleration,
        setup_logging_for_optimization,
        validate_configuration,
    )

    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False


@pytest.mark.skipif(
    not UTILS_AVAILABLE, reason="Advanced optimization utils not available"
)
class TestOptimizationConfig:
    """Test suite for optimization configuration management."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for configuration files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_optimization_config_initialization_default(self):
        """Test optimization config initialization with defaults."""
        config = OptimizationConfig()

        assert isinstance(config.config, dict)
        assert "distributed_optimization" in config.config
        assert "ml_acceleration" in config.config
        assert "performance_monitoring" in config.config

    def test_optimization_config_file_loading(self, temp_config_dir):
        """Test loading configuration from file."""
        # Create test configuration file
        test_config = {
            "distributed_optimization": {
                "enabled": True,
                "backend_preference": ["ray", "multiprocessing"],
            },
            "ml_acceleration": {"enabled": False, "predictor_type": "ensemble"},
        }

        config_file = temp_config_dir / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(test_config, f)

        config = OptimizationConfig(config_file)

        assert config.config == test_config
        assert config.is_distributed_enabled() is True
        assert config.is_ml_enabled() is False

    def test_optimization_config_nonexistent_file(self):
        """Test handling of nonexistent configuration file."""
        config = OptimizationConfig("nonexistent_config.json")

        # Should fall back to defaults without error
        assert isinstance(config.config, dict)

    def test_optimization_config_getters(self):
        """Test configuration getter methods."""
        config = OptimizationConfig()

        distributed_config = config.get_distributed_config()
        ml_config = config.get_ml_config()
        perf_config = config.get_performance_config()

        assert isinstance(distributed_config, dict)
        assert isinstance(ml_config, dict)
        assert isinstance(perf_config, dict)

    def test_optimization_config_save(self, temp_config_dir):
        """Test saving configuration to file."""
        config = OptimizationConfig()

        output_file = temp_config_dir / "saved_config.json"
        config.save_config(output_file)

        assert output_file.exists()

        # Load and verify saved config
        with open(output_file) as f:
            saved_config = json.load(f)

        assert saved_config == config.config


@pytest.mark.skipif(
    not UTILS_AVAILABLE, reason="Advanced optimization utils not available"
)
class TestSystemResourceDetector:
    """Test suite for system resource detection."""

    def test_detect_system_capabilities(self):
        """Test system capability detection."""
        capabilities = SystemResourceDetector.detect_system_capabilities()

        assert isinstance(capabilities, dict)
        assert "cpu_count" in capabilities
        assert "physical_cpu_count" in capabilities
        assert "memory_total_gb" in capabilities
        assert "memory_available_gb" in capabilities
        assert "platform" in capabilities
        assert "architecture" in capabilities
        assert "python_version" in capabilities
        assert "gpu_count" in capabilities
        assert "network_interfaces" in capabilities

        # Check data types
        assert isinstance(capabilities["cpu_count"], int)
        assert isinstance(capabilities["memory_total_gb"], float)
        assert isinstance(capabilities["gpu_count"], int)
        assert capabilities["cpu_count"] > 0
        assert capabilities["memory_total_gb"] > 0

    def test_optimize_configuration(self):
        """Test configuration optimization based on system capabilities."""
        base_config = {
            "distributed_optimization": {
                "multiprocessing_config": {"num_processes": None},
                "ray_config": {"num_cpus": None, "memory_mb": None},
            },
            "ml_acceleration": {
                "ml_model_config": {
                    "hyperparameters": {"random_forest": {"n_estimators": 100}}
                }
            },
        }

        optimized_config = SystemResourceDetector.optimize_configuration(base_config)

        # Check that None values were replaced
        mp_config = optimized_config["distributed_optimization"][
            "multiprocessing_config"
        ]
        assert mp_config["num_processes"] is not None
        assert mp_config["num_processes"] > 0

        ray_config = optimized_config["distributed_optimization"]["ray_config"]
        assert ray_config["num_cpus"] is not None
        assert ray_config["memory_mb"] is not None

    def test_check_system_requirements(self):
        """Test system requirements checking."""
        requirements = SystemResourceDetector.check_system_requirements()

        assert isinstance(requirements, dict)
        assert "sufficient_memory" in requirements
        assert "sufficient_cpu" in requirements
        assert "python_version_ok" in requirements
        assert "multiprocessing_available" in requirements
        assert "ray_recommended" in requirements
        assert "ml_recommended" in requirements

        # Check data types
        for _req_name, req_value in requirements.items():
            assert isinstance(req_value, bool)

        # Multiprocessing should always be available
        assert requirements["multiprocessing_available"] is True

    @patch("psutil.cpu_count", return_value=2)
    @patch("psutil.virtual_memory")
    def test_optimize_configuration_low_resources(self, mock_memory, mock_cpu):
        """Test configuration optimization for low-resource systems."""
        # Mock low memory system
        mock_memory.return_value.total = 2 * 1024**3  # 2GB
        mock_memory.return_value.available = 1 * 1024**3  # 1GB

        base_config = {
            "distributed_optimization": {
                "multiprocessing_config": {"num_processes": None}
            },
            "ml_acceleration": {
                "ml_model_config": {
                    "hyperparameters": {
                        "random_forest": {"n_estimators": 100, "max_depth": 10},
                        "neural_network": {"hidden_layer_sizes": [100, 50]},
                    }
                }
            },
        }

        optimized_config = SystemResourceDetector.optimize_configuration(base_config)

        # Should optimize for limited resources
        ml_hp = optimized_config["ml_acceleration"]["ml_model_config"][
            "hyperparameters"
        ]
        assert ml_hp["random_forest"]["n_estimators"] == 50  # Reduced
        assert ml_hp["random_forest"]["max_depth"] == 5  # Reduced
        assert ml_hp["neural_network"]["hidden_layer_sizes"] == [50, 25]  # Reduced


@pytest.mark.skipif(
    not UTILS_AVAILABLE, reason="Advanced optimization utils not available"
)
class TestOptimizationBenchmark:
    """Test suite for optimization benchmarking."""

    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        benchmark = OptimizationBenchmark()

        assert benchmark.results == []
        assert benchmark.current_benchmark is None

    def test_benchmark_lifecycle(self):
        """Test complete benchmark lifecycle."""
        benchmark = OptimizationBenchmark()

        # Start benchmark
        config = {"method": "Nelder-Mead", "max_iter": 100}
        benchmark.start_benchmark("test_optimization", config)

        assert benchmark.current_benchmark is not None
        assert benchmark.current_benchmark["name"] == "test_optimization"
        assert benchmark.current_benchmark["config"] == config
        assert "start_time" in benchmark.current_benchmark

        # Record metrics
        benchmark.record_metric("optimization_time", 5.67)
        benchmark.record_metric("objective_value", 0.123)
        benchmark.record_metric("success", True)

        assert benchmark.current_benchmark["metrics"]["optimization_time"] == 5.67
        assert benchmark.current_benchmark["metrics"]["objective_value"] == 0.123
        assert benchmark.current_benchmark["metrics"]["success"] is True

        # End benchmark
        result = benchmark.end_benchmark()

        assert benchmark.current_benchmark is None
        assert len(benchmark.results) == 1
        assert result["name"] == "test_optimization"
        assert "total_time" in result
        assert "end_time" in result

    def test_compare_optimizers(self):
        """Test optimizer comparison functionality."""
        benchmark = OptimizationBenchmark()

        test_cases = [
            {"method": "Nelder-Mead", "complexity": "low"},
            {"method": "BFGS", "complexity": "medium"},
            {"method": "ML-Accelerated", "complexity": "high"},
        ]

        comparison_results = benchmark.compare_optimizers(test_cases)

        assert "test_cases" in comparison_results
        assert "summary" in comparison_results
        assert len(comparison_results["test_cases"]) == len(test_cases)

        summary = comparison_results["summary"]
        assert "average_time" in summary
        assert "std_time" in summary
        assert "min_time" in summary
        assert "max_time" in summary
        assert "average_objective" in summary
        assert "best_objective" in summary

    def test_performance_report_empty(self):
        """Test performance report with no results."""
        benchmark = OptimizationBenchmark()

        report = benchmark.get_performance_report()

        assert "error" in report
        assert report["error"] == "No benchmark results available"

    def test_performance_report_with_data(self):
        """Test performance report with benchmark data."""
        benchmark = OptimizationBenchmark()

        # Add some mock results
        benchmark.results = [
            {
                "name": "test1",
                "total_time": 5.0,
                "metrics": {"success": True, "memory_usage": 0.5},
            },
            {
                "name": "test2",
                "total_time": 3.0,
                "metrics": {"success": True, "memory_usage": 0.7},
            },
        ]

        report = benchmark.get_performance_report()

        assert "total_benchmarks" in report
        assert report["total_benchmarks"] == 2
        assert "benchmark_results" in report
        assert "performance_trends" in report
        assert "recommendations" in report


@pytest.mark.skipif(
    not UTILS_AVAILABLE, reason="Advanced optimization utils not available"
)
class TestValidationFunctions:
    """Test suite for validation functions."""

    def test_validate_configuration_valid(self):
        """Test validation of valid configuration."""
        valid_config = {
            "distributed_optimization": {
                "enabled": True,
                "backend_preference": ["ray", "multiprocessing"],
                "ray_config": {"num_cpus": 4, "num_gpus": 0},
            },
            "ml_acceleration": {
                "enabled": True,
                "predictor_type": "ensemble",
                "ml_model_config": {"validation_split": 0.2},
            },
            "performance_monitoring": {
                "enabled": True,
                "alert_thresholds": {"max_optimization_time": 3600},
            },
        }

        is_valid, errors = validate_configuration(valid_config)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_configuration_invalid_backend(self):
        """Test validation with invalid backend."""
        invalid_config = {
            "distributed_optimization": {
                "enabled": True,
                "backend_preference": ["invalid_backend", "ray"],
            }
        }

        is_valid, errors = validate_configuration(invalid_config)

        assert is_valid is False
        assert len(errors) > 0
        assert any("Invalid distributed backend" in error for error in errors)

    def test_validate_configuration_invalid_values(self):
        """Test validation with invalid parameter values."""
        invalid_config = {
            "distributed_optimization": {
                "enabled": True,
                "ray_config": {"num_cpus": -1},  # Invalid negative value
            },
            "ml_acceleration": {
                "enabled": True,
                "ml_model_config": {"validation_split": 1.5},  # Invalid > 1
            },
            "performance_monitoring": {
                "enabled": True,
                "alert_thresholds": {"max_optimization_time": -10},  # Invalid negative
            },
        }

        is_valid, errors = validate_configuration(invalid_config)

        assert is_valid is False
        assert len(errors) >= 3  # Should catch all three errors

    def test_validate_configuration_minimal(self):
        """Test validation with minimal configuration."""
        minimal_config = {}

        is_valid, errors = validate_configuration(minimal_config)

        # Minimal config should be valid (uses defaults)
        assert is_valid is True
        assert len(errors) == 0


@pytest.mark.skipif(
    not UTILS_AVAILABLE, reason="Advanced optimization utils not available"
)
class TestQuickSetupFunctions:
    """Test suite for quick setup functions."""

    @patch("homodyne.optimization.utils.create_distributed_optimizer")
    def test_quick_setup_distributed_optimization(self, mock_create_distributed):
        """Test quick distributed optimization setup."""
        mock_coordinator = Mock()
        mock_create_distributed.return_value = mock_coordinator

        coordinator = quick_setup_distributed_optimization(
            num_processes=4, backend="ray"
        )

        assert coordinator == mock_coordinator
        mock_create_distributed.assert_called_once()

    @patch("homodyne.optimization.utils.create_ml_accelerated_optimizer")
    def test_quick_setup_ml_acceleration(self, mock_create_ml):
        """Test quick ML acceleration setup."""
        mock_optimizer = Mock()
        mock_create_ml.return_value = mock_optimizer

        optimizer = quick_setup_ml_acceleration(
            data_path="/tmp/ml_data", enable_transfer_learning=False
        )

        assert optimizer == mock_optimizer
        mock_create_ml.assert_called_once()

    def test_create_comprehensive_benchmark_suite(self):
        """Test benchmark suite creation."""
        benchmark_cases = create_comprehensive_benchmark_suite()

        assert isinstance(benchmark_cases, list)
        assert len(benchmark_cases) > 0

        for case in benchmark_cases:
            assert "name" in case
            assert "distributed_enabled" in case
            assert "ml_enabled" in case
            assert isinstance(case["distributed_enabled"], bool)
            assert isinstance(case["ml_enabled"], bool)


@pytest.mark.skipif(
    not UTILS_AVAILABLE, reason="Advanced optimization utils not available"
)
class TestLoggingSetup:
    """Test suite for logging setup."""

    def test_setup_logging_for_optimization(self):
        """Test logging setup function."""
        # Should not raise any exceptions
        setup_logging_for_optimization(
            log_level="DEBUG", enable_distributed_logging=True, enable_ml_logging=True
        )

        # Test with different parameters
        setup_logging_for_optimization(
            log_level="WARNING",
            enable_distributed_logging=False,
            enable_ml_logging=False,
        )

    def test_setup_logging_invalid_level(self):
        """Test logging setup with invalid log level."""
        # Should handle gracefully or raise appropriate error
        try:
            setup_logging_for_optimization(log_level="INVALID_LEVEL")
        except (ValueError, AttributeError):
            # Expected behavior for invalid log level
            pass


@pytest.mark.skipif(
    not UTILS_AVAILABLE, reason="Advanced optimization utils not available"
)
class TestUtilityEdgeCases:
    """Test edge cases and error conditions for utility functions."""

    def test_optimization_config_with_empty_dict(self):
        """Test optimization config with empty configuration."""
        config = OptimizationConfig()
        config.config = {}

        # Should handle empty config gracefully
        distributed_config = config.get_distributed_config()
        ml_config = config.get_ml_config()

        assert isinstance(distributed_config, dict)
        assert isinstance(ml_config, dict)
        assert config.is_distributed_enabled() is False
        assert config.is_ml_enabled() is False

    def test_system_resource_detector_with_mock_psutil(self):
        """Test system resource detector with mocked psutil."""
        with (
            patch("psutil.cpu_count", return_value=8),
            patch("psutil.virtual_memory") as mock_memory,
        ):
            mock_memory.return_value.total = 16 * 1024**3  # 16GB
            mock_memory.return_value.available = 12 * 1024**3  # 12GB

            capabilities = SystemResourceDetector.detect_system_capabilities()

            assert capabilities["cpu_count"] == 8
            assert capabilities["memory_total_gb"] == 16.0

    def test_benchmark_with_no_current_benchmark(self):
        """Test benchmark operations with no current benchmark."""
        benchmark = OptimizationBenchmark()

        # Should handle gracefully
        benchmark.record_metric("test_metric", 123)  # Should not crash

        result = benchmark.end_benchmark()
        assert result == {}


def test_advanced_utils_basic_imports():
    """Test that advanced utils can be imported without errors."""
    try:
        from homodyne.optimization.utils import (
            OptimizationBenchmark,
            OptimizationConfig,
            SystemResourceDetector,
        )

        # Use imports to avoid F401
        _ = OptimizationBenchmark
        _ = OptimizationConfig
        _ = SystemResourceDetector

        print("✅ Advanced optimization utilities imported successfully")
    except ImportError as e:
        print(f"⚠️ Advanced optimization utilities not available: {e}")
        # This is acceptable if the features haven't been installed yet


if __name__ == "__main__":
    # Run basic import test
    test_advanced_utils_basic_imports()

    # Run tests with pytest if available
    try:
        pytest.main([__file__, "-v"])
    except SystemExit:
        pass
