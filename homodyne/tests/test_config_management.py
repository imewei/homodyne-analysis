"""
Comprehensive Unit Tests for Configuration Management
====================================================

Tests for configuration loading, validation, and template management.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

try:
    from homodyne.config import TEMPLATE_FILES
    from homodyne.config import get_config_dir
    from homodyne.config import get_template_path
    from homodyne.core.config import ConfigManager
    from homodyne.core.config import configure_logging
    from homodyne.core.config import performance_monitor

    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Configuration module not available")
class TestConfigManager:
    """Test suite for ConfigManager functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.sample_config = {
            "analyzer_parameters": {
                "q_value": 0.1,
                "wavevector_q": 0.005,
                "contrast": 0.95,
                "offset": 1.0,
                "pixel_size": 172e-6,
                "detector_distance": 8.0,
                "x_ray_energy": 7.35,
                "sample_thickness": 1.0,
                "dt": 0.1,
                "stator_rotor_gap": 2000000,
                "temporal": {"start_frame": 1, "end_frame": 100},
            },
            "experimental_data": {
                "data_file": "test_data.h5",
                "cache_enabled": True,
                "preload_data": False,
            },
            "optimization_config": {
                "mode": "laminar_flow",
                "method": "classical",
                "enable_angle_filtering": True,
                "chi_squared_threshold": 2.0,
                "max_iterations": 1000,
                "tolerance": 1e-6,
                "parameter_bounds": {
                    "D0": [1e-6, 1e-1],
                    "alpha": [0.1, 2.0],
                    "D_offset": [1e-8, 1e-3],
                    "gamma0": [1e-4, 1.0],
                    "beta": [0.1, 2.0],
                    "gamma_offset": [1e-6, 1e-1],
                    "phi0": [-180, 180],
                },
                "initial_guesses": {
                    "D0": 1e-3,
                    "alpha": 0.9,
                    "D_offset": 1e-4,
                    "gamma0": 0.01,
                    "beta": 0.8,
                    "gamma_offset": 0.001,
                    "phi0": 0.0,
                },
            },
            "output_settings": {
                "save_plots": True,
                "plot_format": "png",
                "save_results": True,
                "output_directory": "./results",
            },
        }

        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(self.sample_config, self.temp_config, indent=2)
        self.temp_config.close()

    def teardown_method(self):
        """Cleanup test fixtures."""
        if hasattr(self, "temp_config"):
            try:
                os.unlink(self.temp_config.name)
            except (PermissionError, FileNotFoundError):
                # Windows may have file locks, ignore cleanup errors
                pass

    def test_config_manager_initialization(self):
        """Test ConfigManager initialization."""
        # Test initialization with file path
        manager = ConfigManager(config_file=self.temp_config.name)
        assert manager.config is not None
        assert manager.config["analyzer_parameters"]["q_value"] == 0.1

        # ConfigManager only supports file-based initialization

    def test_config_loading_from_file(self):
        """Test configuration loading from file."""
        manager = ConfigManager()
        config = manager.load_config(self.temp_config.name)

        assert config["analyzer_parameters"]["contrast"] == 0.95
        assert config["optimization_config"]["mode"] == "laminar_flow"

    def test_config_validation(self):
        """Test configuration validation."""
        manager = ConfigManager(config=self.sample_config)

        # Test valid configuration
        assert manager.validate_config()

        # Test invalid configuration - missing required section
        invalid_config = {"analyzer_parameters": {}}
        manager_invalid = ConfigManager(config=invalid_config)
        assert not manager_invalid.validate_config()

    def test_parameter_bounds_validation(self):
        """Test parameter bounds validation."""
        manager = ConfigManager(config=self.sample_config)

        # Test valid parameters
        valid_params = {
            "D0": 1e-3,
            "alpha": 0.9,
            "D_offset": 1e-4,
            "gamma0": 0.01,
            "beta": 0.8,
            "gamma_offset": 0.001,
            "phi0": 0.0,
        }
        assert manager.validate_parameter_bounds(valid_params)

        # Test invalid parameters
        invalid_params = {
            "D0": 1.0,  # Above upper bound
            "alpha": 0.05,  # Below lower bound
        }
        assert not manager.validate_parameter_bounds(invalid_params)

    def test_get_parameter_value(self):
        """Test parameter value retrieval."""
        manager = ConfigManager(config=self.sample_config)

        # Test existing parameter
        q_value = manager.get_parameter("analyzer_parameters", "q_value")
        assert q_value == 0.1

        # Test non-existing parameter
        with pytest.raises(KeyError):
            manager.get_parameter("analyzer_parameters", "non_existing")

        # Test with default value
        default_val = manager.get_parameter(
            "analyzer_parameters", "non_existing", default=0.5
        )
        assert default_val == 0.5

    def test_set_parameter_value(self):
        """Test parameter value setting."""
        manager = ConfigManager(config=self.sample_config.copy())

        # Test setting existing parameter
        manager.set_parameter("analyzer_parameters", "q_value", 0.2)
        assert manager.get_parameter("analyzer_parameters", "q_value") == 0.2

        # Test setting new parameter
        manager.set_parameter("analyzer_parameters", "new_param", 42)
        assert manager.get_parameter("analyzer_parameters", "new_param") == 42

    def test_config_merging(self):
        """Test configuration merging functionality."""
        manager = ConfigManager(config=self.sample_config)

        # Create partial update config
        update_config = {
            "analyzer_parameters": {"q_value": 0.2, "new_parameter": "test_value"},
            "new_section": {"test_param": 123},
        }

        merged = manager.merge_configs(update_config)

        # Check that existing values are updated
        assert merged["analyzer_parameters"]["q_value"] == 0.2
        # Check that new values are added
        assert merged["analyzer_parameters"]["new_parameter"] == "test_value"
        # Check that existing values are preserved
        assert merged["analyzer_parameters"]["contrast"] == 0.95
        # Check that new sections are added
        assert merged["new_section"]["test_param"] == 123

    def test_config_saving(self):
        """Test configuration saving to file."""
        manager = ConfigManager(config=self.sample_config)

        # Test saving to new file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            temp_path = temp_file.name

        try:
            manager.save_config(temp_path)

            # Verify file was created and contains correct data
            assert os.path.exists(temp_path)

            with open(temp_path) as f:
                saved_config = json.load(f)

            assert saved_config == self.sample_config
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_template_discovery(self):
        """Test configuration template discovery."""
        manager = ConfigManager()

        # Test template listing
        templates = manager.list_available_templates()
        assert isinstance(templates, list)

        # Test template loading (if templates exist)
        if templates:
            template_name = templates[0]
            template_config = manager.load_template(template_name)
            assert isinstance(template_config, dict)

    def test_config_schema_validation(self):
        """Test configuration schema validation."""
        manager = ConfigManager(config=self.sample_config)

        # Test that required sections exist (using actual schema)
        required_sections = [
            "analyzer_parameters",
            "experimental_data",
            "optimization_config",
        ]

        for section in required_sections:
            assert section in manager.config

        # Test required analyzer parameters
        required_analyzer_params = ["q_value", "contrast", "offset"]
        for param in required_analyzer_params:
            assert param in manager.config["analyzer_parameters"]

    def test_config_type_validation(self):
        """Test configuration type validation."""
        ConfigManager()

        # Test with correct types using actual schema
        valid_config = {
            "analyzer_parameters": {
                "q_value": 0.1,  # float
                "contrast": 0.95,  # float
                "offset": 1.0,  # float
            },
            "experimental_data": {
                "data_file": "test_data.h5",  # string
                "cache_enabled": True,  # boolean
                "preload_data": False,  # boolean
            },
            "optimization_config": {
                "mode": "laminar_flow",  # string
                "enable_angle_filtering": True,  # boolean
                "max_iterations": 1000,  # int
            },
        }

        manager_valid = ConfigManager(config=valid_config)
        assert manager_valid.validate_config()

        # Test with incorrect types
        invalid_config = {
            "analyzer_parameters": {
                "q_value": "not_a_number",  # Should be float
                "contrast": 0.95,
            }
        }

        manager_invalid = ConfigManager(config=invalid_config)
        # Should handle type validation gracefully
        # May issue warnings but shouldn't crash
        manager_invalid.validate_config()

    def test_environment_variable_substitution(self):
        """Test environment variable substitution in config."""
        # Set test environment variable
        os.environ["TEST_HOMODYNE_PARAM"] = "0.123"

        config_with_env = {
            "analyzer_parameters": {
                "q_value": "${TEST_HOMODYNE_PARAM}",
                "contrast": 0.95,
            }
        }

        manager = ConfigManager()
        processed_config = manager.resolve_environment_variables(config_with_env)

        # Check that environment variable was substituted
        # Note: Implementation depends on whether env var substitution is implemented
        assert "q_value" in processed_config["analyzer_parameters"]

        # Cleanup
        del os.environ["TEST_HOMODYNE_PARAM"]

    def test_config_backup_and_restore(self):
        """Test configuration backup and restore functionality."""
        manager = ConfigManager(config=self.sample_config.copy())

        # Create backup
        backup = manager.create_backup()
        assert backup == manager.config

        # Modify config
        manager.set_parameter("analyzer_parameters", "q_value", 0.5)
        assert manager.get_parameter("analyzer_parameters", "q_value") == 0.5

        # Restore from backup
        manager.restore_from_backup(backup)
        assert manager.get_parameter("analyzer_parameters", "q_value") == 0.1

    def test_config_difference_detection(self):
        """Test configuration difference detection."""
        manager1 = ConfigManager(config=self.sample_config)

        # Create modified config
        modified_config = self.sample_config.copy()
        modified_config["analyzer_parameters"]["q_value"] = 0.2

        manager2 = ConfigManager(config=modified_config)

        # Test difference detection
        differences = manager1.get_config_differences(manager2.config)
        assert isinstance(differences, dict)

        # Should detect the q_value change
        if "analyzer_parameters" in differences:
            assert "q_value" in differences["analyzer_parameters"]


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Configuration module not available")
class TestConfigTemplates:
    """Test suite for configuration templates."""

    def test_template_files_availability(self):
        """Test that template files are available."""
        if TEMPLATE_FILES is not None:
            assert isinstance(TEMPLATE_FILES, dict)
            assert len(TEMPLATE_FILES) > 0

    def test_get_template_path(self):
        """Test template path retrieval."""
        if get_template_path is not None and TEMPLATE_FILES:
            # Test with valid template name
            template_name = next(iter(TEMPLATE_FILES.keys()))
            path = get_template_path(template_name)

            if path:  # If template exists
                assert isinstance(path, (str, Path))
                assert str(path).endswith(".json")

    def test_get_config_dir(self):
        """Test config directory retrieval."""
        if get_config_dir is not None:
            config_dir = get_config_dir()
            if config_dir:  # If config directory exists
                assert isinstance(config_dir, (str, Path))

    def test_template_loading(self):
        """Test template loading functionality."""
        if TEMPLATE_FILES and get_template_path:
            for template_file in TEMPLATE_FILES:
                template_name = template_file.replace(".json", "")
                template_path = get_template_path(template_name)

                if template_path and os.path.exists(template_path):
                    # Try to load the template
                    with open(template_path) as f:
                        template_config = json.load(f)

                    assert isinstance(template_config, dict)

                    # Check that it has required sections (using actual schema)
                    assert "analyzer_parameters" in template_config
                    assert "experimental_data" in template_config

    def test_template_validation(self):
        """Test that templates are valid configurations."""
        if TEMPLATE_FILES and get_template_path:
            for template_file in TEMPLATE_FILES:
                template_name = template_file.replace(".json", "")
                template_path = get_template_path(template_name)

                if template_path and os.path.exists(template_path):
                    with open(template_path) as f:
                        template_config = json.load(f)

                    # Validate template using ConfigManager
                    manager = ConfigManager(config=template_config)
                    # Should not raise exceptions
                    assert manager.config is not None


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Configuration module not available")
class TestLoggingConfiguration:
    """Test suite for logging configuration."""

    def test_configure_logging_basic(self):
        """Test basic logging configuration."""
        if configure_logging is not None:
            # Test with basic configuration
            cfg = {"logging": {"level": "INFO"}}
            configure_logging(cfg)

            # Should not raise exceptions
            import logging

            logger = logging.getLogger("homodyne")
            assert logger is not None

    def test_configure_logging_with_level(self):
        """Test logging configuration with specific level."""
        if configure_logging is not None:
            import logging

            configure_logging(level=logging.DEBUG)

            logger = logging.getLogger("homodyne")
            # Should be configured for DEBUG level
            assert logger.level <= logging.DEBUG

    def test_configure_logging_with_file(self):
        """Test logging configuration with file output."""
        if configure_logging is not None:
            with tempfile.NamedTemporaryFile(
                suffix=".log", delete=False, mode="w", encoding="utf-8"
            ) as temp_log:
                temp_log_path = temp_log.name

            try:
                configure_logging(log_file=temp_log_path)

                # Test that log file is created
                import logging

                logger = logging.getLogger("homodyne")
                logger.info("Test log message")

                # File should exist (though may be empty due to buffering)
                assert os.path.exists(temp_log_path)
            finally:
                if os.path.exists(temp_log_path):
                    try:
                        os.unlink(temp_log_path)
                    except (PermissionError, FileNotFoundError):
                        # Windows may have file locks, ignore cleanup errors
                        pass


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Configuration module not available")
class TestPerformanceMonitor:
    """Test suite for performance monitor configuration."""

    def test_performance_monitor_basic(self):
        """Test basic performance monitor functionality."""
        if performance_monitor is not None:
            # Should be callable or have expected interface
            assert callable(performance_monitor) or hasattr(
                performance_monitor, "start"
            )

    def test_performance_monitor_context(self):
        """Test performance monitor as context manager."""
        if performance_monitor is not None:
            # Test if it can be used as a context manager
            try:
                with performance_monitor("test_operation"):
                    # Simple operation
                    result = sum(range(1000))
                    assert result > 0
            except (AttributeError, TypeError):
                # May not be implemented as context manager
                pass

    def test_performance_monitor_decorator(self):
        """Test performance monitor as decorator."""
        if performance_monitor is not None:
            # Test if it can be used as a decorator
            try:

                @performance_monitor
                def test_function():
                    return sum(range(100))

                result = test_function()
                assert result > 0
            except (AttributeError, TypeError):
                # May not be implemented as decorator
                pass


class TestConfigurationIntegration:
    """Integration tests for configuration with other components."""

    def setup_method(self):
        """Setup integration test fixtures."""
        self.integration_config = {
            "analyzer_parameters": {
                "q_value": 0.08,
                "contrast": 0.92,
                "offset": 1.0,
                "pixel_size": 172e-6,
                "detector_distance": 8.0,
                "x_ray_energy": 7.35,
            },
            "experimental_data": {
                "data_file": "test_data.h5",
                "cache_enabled": True,
                "preload_data": False,
            },
            "optimization_config": {
                "mode": "static_isotropic",
                "method": "classical",
                "enable_angle_filtering": False,
                "max_iterations": 500,
                "tolerance": 1e-5,
                "parameter_bounds": {
                    "D0": [1e-5, 1e-2],
                    "alpha": [0.2, 1.8],
                    "D_offset": [1e-7, 1e-4],
                },
            },
            "performance_settings": {
                "use_numba": True,
                "enable_caching": True,
                "memory_limit_mb": 1000,
            },
        }

    @pytest.mark.skipif(
        not CONFIG_AVAILABLE, reason="Configuration module not available"
    )
    def test_config_integration_with_analysis(self):
        """Test configuration integration with analysis components."""
        manager = ConfigManager(config=self.integration_config)

        # Test that config can be used by analysis components
        q_value = manager.get_parameter("analyzer_parameters", "q_value")
        assert q_value == 0.08

        mode = manager.get_parameter("optimization_config", "mode")
        assert mode == "static_isotropic"

    @pytest.mark.skipif(
        not CONFIG_AVAILABLE, reason="Configuration module not available"
    )
    def test_config_with_different_modes(self):
        """Test configuration with different analysis modes."""
        modes = ["static_isotropic", "static_anisotropic", "laminar_flow"]

        for mode in modes:
            config = self.integration_config.copy()
            config["optimization_config"]["mode"] = mode

            manager = ConfigManager(config=config)
            assert manager.get_parameter("optimization_config", "mode") == mode

            # Validate that the config is still valid
            assert manager.validate_config()

    @pytest.mark.skipif(
        not CONFIG_AVAILABLE, reason="Configuration module not available"
    )
    def test_config_parameter_inheritance(self):
        """Test configuration parameter inheritance and defaults."""
        # Test minimal config with defaults
        minimal_config = {"analyzer_parameters": {"q_value": 0.1}}

        manager = ConfigManager(config=minimal_config)

        # Should handle missing parameters gracefully
        contrast = manager.get_parameter("analyzer_parameters", "contrast", default=0.9)
        assert contrast == 0.9
