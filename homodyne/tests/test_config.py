"""
Tests for Configuration Management Module
========================================

Tests configuration loading, validation, and access methods.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from homodyne.core.config import ConfigManager


class TestConfigManager:
    """Test suite for ConfigManager class."""

    @pytest.fixture
    def minimal_config(self):
        """Minimal valid configuration."""
        return {
            "metadata": {"config_version": "1.0"},
            "analyzer_parameters": {"start_frame": 1, "end_frame": 100},
            "experimental_data": {"data_directory": "test"},
            "optimization_config": {"method": "test"},
        }

    def test_config_manager_init_with_file(self, temp_directory, minimal_config):
        """Test initialization with config file."""
        config_file = temp_directory / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(minimal_config, f)

        manager = ConfigManager(str(config_file))
        assert manager.get("metadata", "config_version") == "1.0"

    def test_config_manager_get_nested(self, temp_directory, minimal_config):
        """Test getting nested configuration values."""
        # Add nested structure
        minimal_config["analyzer_parameters"]["nested"] = {"value": "test"}

        config_file = temp_directory / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(minimal_config, f)

        manager = ConfigManager(str(config_file))
        assert manager.get("analyzer_parameters", "nested", "value") == "test"
        assert manager.get("analyzer_parameters", "nested") == {"value": "test"}
        assert manager.get("nonexistent", default="default") == "default"

    def test_config_manager_get_with_default(self, temp_directory, minimal_config):
        """Test getting values with default fallback."""
        config_file = temp_directory / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(minimal_config, f)

        manager = ConfigManager(str(config_file))
        assert manager.get("analyzer_parameters", "start_frame") == 1
        assert manager.get("nonexistent", default="default") == "default"
        assert manager.get("nonexistent") is None

    def test_config_manager_invalid_file(self):
        """Test handling of invalid config file."""
        # ConfigManager should load default config when file doesn't exist
        manager = ConfigManager("/nonexistent/file.json")
        # The manager should have loaded default config
        assert manager.config is not None
        assert "metadata" in manager.config

    def test_config_manager_invalid_json(self, temp_directory):
        """Test handling of invalid JSON file."""
        config_file = temp_directory / "invalid.json"
        config_file.write_text("invalid json content")

        # Should fall back to default config instead of raising
        manager = ConfigManager(str(config_file))
        assert manager.config is not None  # Should have default config

    def test_config_manager_missing_required_sections(self, temp_directory):
        """Test validation with missing required sections."""
        incomplete_config = {
            "metadata": {"config_version": "1.0"}
            # Missing required sections
        }

        config_file = temp_directory / "incomplete.json"
        with open(config_file, "w") as f:
            json.dump(incomplete_config, f)

        with pytest.raises(ValueError, match="Missing required sections"):
            ConfigManager(str(config_file))

    def test_config_manager_invalid_frame_range(self, temp_directory, minimal_config):
        """Test validation with invalid frame range."""
        minimal_config["analyzer_parameters"]["start_frame"] = 100
        # Invalid: start > end
        minimal_config["analyzer_parameters"]["end_frame"] = 50

        config_file = temp_directory / "invalid_range.json"
        with open(config_file, "w") as f:
            json.dump(minimal_config, f)

        with pytest.raises(ValueError, match="Invalid frame range"):
            ConfigManager(str(config_file))

    def test_config_manager_get_methods(self, temp_directory, minimal_config):
        """Test various get method patterns."""
        config_file = temp_directory / "test.json"
        with open(config_file, "w") as f:
            json.dump(minimal_config, f)

        manager = ConfigManager(str(config_file))

        # Single key
        assert manager.get("metadata") == {"config_version": "1.0"}

        # Multiple keys
        assert manager.get("analyzer_parameters", "start_frame") == 1

        # Non-existent with default
        assert manager.get("missing", default="default") == "default"

        # Non-existent without default
        assert manager.get("missing") is None

    def test_static_mode_methods(self, temp_directory, minimal_config):
        """Test static mode configuration methods."""
        # Add analysis_settings to minimal config
        minimal_config["analysis_settings"] = {"static_mode": True}

        config_file = temp_directory / "static_test.json"
        with open(config_file, "w") as f:
            json.dump(minimal_config, f)

        manager = ConfigManager(str(config_file))

        # Test static mode detection - defaults to anisotropic when submode not
        # specified
        assert manager.is_static_mode_enabled() is True
        assert manager.get_analysis_mode() == "static_anisotropic"
        assert manager.get_effective_parameter_count() == 3

        # Test analysis settings
        settings = manager.get_analysis_settings()
        assert settings["static_mode"] is True
        assert "model_description" in settings

    def test_laminar_flow_mode_methods(self, temp_directory, minimal_config):
        """Test laminar flow mode configuration methods."""
        # Add analysis_settings to minimal config
        minimal_config["analysis_settings"] = {"static_mode": False}

        config_file = temp_directory / "flow_test.json"
        with open(config_file, "w") as f:
            json.dump(minimal_config, f)

        manager = ConfigManager(str(config_file))

        # Test flow mode detection
        assert manager.is_static_mode_enabled() is False
        assert manager.get_analysis_mode() == "laminar_flow"
        assert manager.get_effective_parameter_count() == 7

        # Test analysis settings with defaults
        settings = manager.get_analysis_settings()
        assert settings["static_mode"] is False
        assert "model_description" in settings

    def test_default_analysis_settings(self, temp_directory, minimal_config):
        """Test default analysis settings when not specified in config."""
        # Don't add analysis_settings - should use defaults
        config_file = temp_directory / "default_test.json"
        with open(config_file, "w") as f:
            json.dump(minimal_config, f)

        manager = ConfigManager(str(config_file))

        # Should default to laminar flow mode
        assert manager.is_static_mode_enabled() is False
        assert manager.get_analysis_mode() == "laminar_flow"
        assert manager.get_effective_parameter_count() == 7

        # Should provide default settings
        settings = manager.get_analysis_settings()
        assert "static_mode" in settings
        assert "model_description" in settings


def test_config_manager_default_config():
    """Test that default configuration is loaded when file is missing."""
    import tempfile

    # Use a non-existent file in temp directory to trigger default config loading
    temp_dir = tempfile.gettempdir()
    nonexistent_file = os.path.join(temp_dir, "definitely_nonexistent_file.json")
    manager = ConfigManager(nonexistent_file)

    # Should have loaded default config
    assert manager.config is not None
    assert "metadata" in manager.config
    assert "analyzer_parameters" in manager.config


def test_config_manager_real_config_file():
    """Test with the actual homodyne_config.json file if it exists."""
    # Check both project root and tests directory for config file
    project_root_path = Path(__file__).parent.parent.parent / "homodyne_config.json"
    tests_dir_path = Path(__file__).parent / "homodyne_config.json"

    config_path = None
    if project_root_path.exists():
        config_path = project_root_path
    elif tests_dir_path.exists():
        config_path = tests_dir_path

    if config_path:
        manager = ConfigManager(str(config_path))
        assert manager.config is not None
        assert manager.get("metadata") is not None
    else:
        pytest.skip("Real config file not found")


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "metadata": {
            "config_version": "1.0",
            "description": "Test configuration",
        },
        "analyzer_parameters": {
            "start_frame": 1,
            "end_frame": 100,
            "num_threads": 4,
        },
        "experimental_data": {
            "data_directory": "test_data",
            "file_pattern": "*.npz",
        },
        "optimization_config": {
            "method": "classical",
            "max_iterations": 1000,
            "tolerance": 1e-6,
        },
    }


def test_config_manager_comprehensive(sample_config, temp_directory):
    """Comprehensive test of ConfigManager functionality."""
    config_file = temp_directory / "comprehensive.json"
    with open(config_file, "w") as f:
        json.dump(sample_config, f)

    manager = ConfigManager(str(config_file))

    # Test various access patterns
    assert manager.get("metadata", "config_version") == "1.0"
    assert manager.get("analyzer_parameters", "start_frame") == 1
    assert manager.get("optimization_config", "max_iterations") == 1000

    # Test missing keys
    assert manager.get("nonexistent") is None
    assert manager.get("nonexistent", default="default") == "default"

    # Test nested missing
    assert manager.get("metadata", "missing", default="default") == "default"


class TestConfigManagerAngleFiltering:
    """Test suite for ConfigManager angle filtering methods."""

    @pytest.fixture
    def config_with_angle_filtering(self):
        """Configuration with complete angle filtering section."""
        return {
            "metadata": {"config_version": "5.1"},
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 100}
            },
            "experimental_data": {"data_folder_path": "./test_data/"},
            "optimization_config": {
                "angle_filtering": {
                    "enabled": True,
                    "target_ranges": [
                        {"min_angle": -10.0, "max_angle": 10.0},
                        {"min_angle": 170.0, "max_angle": 190.0},
                    ],
                    "fallback_to_all_angles": True,
                }
            },
        }

    @pytest.fixture
    def config_without_angle_filtering(self):
        """Minimal configuration without angle filtering section."""
        return {
            "metadata": {"config_version": "5.1"},
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 100}
            },
            "experimental_data": {"data_folder_path": "./test_data/"},
            "optimization_config": {},
        }

    @pytest.fixture
    def config_with_custom_angle_filtering(self):
        """Configuration with custom angle filtering settings."""
        return {
            "metadata": {"config_version": "5.1"},
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 100}
            },
            "experimental_data": {"data_folder_path": "./test_data/"},
            "optimization_config": {
                "angle_filtering": {
                    "enabled": False,
                    "target_ranges": [
                        {"min_angle": -5.0, "max_angle": 5.0},
                        {"min_angle": 175.0, "max_angle": 185.0},
                        {"min_angle": 45.0, "max_angle": 55.0},
                    ],
                    "fallback_to_all_angles": False,
                }
            },
        }

    def test_angle_filtering_enabled_with_config(
        self, temp_directory, config_with_angle_filtering
    ):
        """Test is_angle_filtering_enabled with complete configuration."""
        config_file = temp_directory / "angle_filtering.json"
        with open(config_file, "w") as f:
            json.dump(config_with_angle_filtering, f)

        manager = ConfigManager(str(config_file))
        assert manager.is_angle_filtering_enabled()

    def test_angle_filtering_enabled_without_config(
        self, temp_directory, config_without_angle_filtering
    ):
        """Test is_angle_filtering_enabled with missing configuration (should default to True)."""
        config_file = temp_directory / "no_angle_filtering.json"
        with open(config_file, "w") as f:
            json.dump(config_without_angle_filtering, f)

        manager = ConfigManager(str(config_file))
        assert manager.is_angle_filtering_enabled()  # Default value

    def test_angle_filtering_enabled_custom(
        self, temp_directory, config_with_custom_angle_filtering
    ):
        """Test is_angle_filtering_enabled with custom configuration."""
        config_file = temp_directory / "custom_angle_filtering.json"
        with open(config_file, "w") as f:
            json.dump(config_with_custom_angle_filtering, f)

        manager = ConfigManager(str(config_file))
        assert manager.is_angle_filtering_enabled() is False

    def test_get_target_angle_ranges_default(
        self, temp_directory, config_with_angle_filtering
    ):
        """Test get_target_angle_ranges with default configuration."""
        config_file = temp_directory / "angle_filtering.json"
        with open(config_file, "w") as f:
            json.dump(config_with_angle_filtering, f)

        manager = ConfigManager(str(config_file))
        target_ranges = manager.get_target_angle_ranges()

        expected = [(-10.0, 10.0), (170.0, 190.0)]
        assert target_ranges == expected

    def test_get_target_angle_ranges_missing_config(
        self, temp_directory, config_without_angle_filtering
    ):
        """Test get_target_angle_ranges with missing configuration (should use defaults)."""
        config_file = temp_directory / "no_angle_filtering.json"
        with open(config_file, "w") as f:
            json.dump(config_without_angle_filtering, f)

        manager = ConfigManager(str(config_file))
        target_ranges = manager.get_target_angle_ranges()

        expected = [(-10.0, 10.0), (170.0, 190.0)]  # Default values
        assert target_ranges == expected

    def test_get_target_angle_ranges_custom(
        self, temp_directory, config_with_custom_angle_filtering
    ):
        """Test get_target_angle_ranges with custom configuration."""
        config_file = temp_directory / "custom_angle_filtering.json"
        with open(config_file, "w") as f:
            json.dump(config_with_custom_angle_filtering, f)

        manager = ConfigManager(str(config_file))
        target_ranges = manager.get_target_angle_ranges()

        expected = [(-5.0, 5.0), (175.0, 185.0), (45.0, 55.0)]
        assert target_ranges == expected

    def test_should_fallback_to_all_angles_default(
        self, temp_directory, config_with_angle_filtering
    ):
        """Test should_fallback_to_all_angles with default configuration."""
        config_file = temp_directory / "angle_filtering.json"
        with open(config_file, "w") as f:
            json.dump(config_with_angle_filtering, f)

        manager = ConfigManager(str(config_file))
        assert manager.should_fallback_to_all_angles()

    def test_should_fallback_to_all_angles_custom(
        self, temp_directory, config_with_custom_angle_filtering
    ):
        """Test should_fallback_to_all_angles with custom configuration."""
        config_file = temp_directory / "custom_angle_filtering.json"
        with open(config_file, "w") as f:
            json.dump(config_with_custom_angle_filtering, f)

        manager = ConfigManager(str(config_file))
        assert manager.should_fallback_to_all_angles() is False

    def test_get_angle_filtering_config_complete(
        self, temp_directory, config_with_angle_filtering
    ):
        """Test get_angle_filtering_config with complete configuration."""
        config_file = temp_directory / "angle_filtering.json"
        with open(config_file, "w") as f:
            json.dump(config_with_angle_filtering, f)

        manager = ConfigManager(str(config_file))
        angle_config = manager.get_angle_filtering_config()

        # Check required keys are present
        assert "enabled" in angle_config
        assert "target_ranges" in angle_config
        assert "fallback_to_all_angles" in angle_config

        # Check values
        assert angle_config["enabled"]
        assert len(angle_config["target_ranges"]) == 2
        assert angle_config["fallback_to_all_angles"]

    def test_get_angle_filtering_config_defaults(
        self, temp_directory, config_without_angle_filtering
    ):
        """Test get_angle_filtering_config with missing configuration (should provide defaults)."""
        config_file = temp_directory / "no_angle_filtering.json"
        with open(config_file, "w") as f:
            json.dump(config_without_angle_filtering, f)

        manager = ConfigManager(str(config_file))
        angle_config = manager.get_angle_filtering_config()

        # Should provide defaults
        assert angle_config["enabled"]
        assert len(angle_config["target_ranges"]) == 2
        assert angle_config["fallback_to_all_angles"]

        # Check default ranges
        expected_ranges = [
            {"min_angle": -10.0, "max_angle": 10.0},
            {"min_angle": 170.0, "max_angle": 190.0},
        ]
        assert angle_config["target_ranges"] == expected_ranges

    def test_angle_filtering_config_validation(self, temp_directory):
        """Test angle filtering configuration validation with invalid ranges."""
        invalid_config = {
            "metadata": {"config_version": "5.1"},
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 100}
            },
            "experimental_data": {"data_folder_path": "./test_data/"},
            "optimization_config": {
                "angle_filtering": {
                    "enabled": True,
                    "target_ranges": [
                        {"min_angle": -10.0},  # Missing max_angle
                        {"max_angle": 190.0},  # Missing min_angle
                        "invalid_range",  # Invalid format
                        {
                            "min_angle": 175.0,
                            "max_angle": 185.0,
                        },  # Valid range
                    ],
                    "fallback_to_all_angles": True,
                }
            },
        }

        config_file = temp_directory / "invalid_angle_filtering.json"
        with open(config_file, "w") as f:
            json.dump(invalid_config, f)

        manager = ConfigManager(str(config_file))
        angle_config = manager.get_angle_filtering_config()

        # Should filter out invalid ranges and keep only valid ones
        valid_ranges = angle_config["target_ranges"]
        assert len(valid_ranges) == 1  # Only one valid range
        assert valid_ranges[0] == {"min_angle": 175.0, "max_angle": 185.0}

    def test_angle_filtering_empty_ranges(self, temp_directory):
        """Test behavior with empty target ranges."""
        empty_ranges_config = {
            "metadata": {"config_version": "5.1"},
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 100}
            },
            "experimental_data": {"data_folder_path": "./test_data/"},
            "optimization_config": {
                "angle_filtering": {
                    "enabled": True,
                    "target_ranges": [],  # Empty ranges
                    "fallback_to_all_angles": True,
                }
            },
        }

        config_file = temp_directory / "empty_ranges.json"
        with open(config_file, "w") as f:
            json.dump(empty_ranges_config, f)

        manager = ConfigManager(str(config_file))
        target_ranges = manager.get_target_angle_ranges()

        # Should return empty list
        assert target_ranges == []

    def test_angle_filtering_integration_with_real_configs(self):
        """Test angle filtering methods with real configuration files."""
        # Test with my_config.json if it exists
        my_config_path = Path(__file__).parent.parent.parent / "my_config.json"
        if my_config_path.exists():
            manager = ConfigManager(str(my_config_path))

            # Should not raise exceptions
            enabled = manager.is_angle_filtering_enabled()
            ranges = manager.get_target_angle_ranges()
            fallback = manager.should_fallback_to_all_angles()

            # Should return valid types
            assert isinstance(enabled, bool)
            assert isinstance(ranges, list)
            assert isinstance(fallback, bool)

            # Ranges should be tuples of floats
            for range_tuple in ranges:
                assert isinstance(range_tuple, tuple)
                assert len(range_tuple) == 2
                assert isinstance(range_tuple[0], (int, float))
                assert isinstance(range_tuple[1], (int, float))

        # Test with template config
        template_path = Path(__file__).parent.parent / "config_template.json"
        if template_path.exists():
            manager = ConfigManager(str(template_path))

            # Should not raise exceptions
            enabled = manager.is_angle_filtering_enabled()
            ranges = manager.get_target_angle_ranges()
            fallback = manager.should_fallback_to_all_angles()

            # Should return valid types
            assert isinstance(enabled, bool)
            assert isinstance(ranges, list)
            assert isinstance(fallback, bool)


class TestPlottingConfigurationConsistency:
    """Test plotting configuration consistency and validation."""

    @pytest.fixture
    def plotting_config(self):
        """Configuration with plotting settings."""
        return {
            "metadata": {"config_version": "1.0"},
            "analyzer_parameters": {"start_frame": 1, "end_frame": 100},
            "experimental_data": {"data_directory": "test"},
            "optimization_config": {"method": "test"},
            "initial_parameters": {
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_t0",
                    "beta",
                    "gamma_dot_t_offset",
                    "phi0",
                ],
                "values": [1000, -0.5, 100, 0.001, 0.2, 0.0001, 5.0],
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 1.0, "max": 10000.0, "unit": "Å²/s"},
                    {
                        "name": "alpha",
                        "min": -2.0,
                        "max": 0.0,
                        "unit": "dimensionless",
                    },
                    {
                        "name": "D_offset",
                        "min": 0.0,
                        "max": 1000.0,
                        "unit": "Å²/s",
                    },
                    {
                        "name": "gamma_dot_t0",
                        "min": 1e-5,
                        "max": 0.1,
                        "unit": "s⁻¹",
                    },
                    {
                        "name": "beta",
                        "min": -1.0,
                        "max": 1.0,
                        "unit": "dimensionless",
                    },
                    {
                        "name": "gamma_dot_t_offset",
                        "min": 0.0,
                        "max": 0.01,
                        "unit": "s⁻¹",
                    },
                    {
                        "name": "phi0",
                        "min": 0.0,
                        "max": 360.0,
                        "unit": "degrees",
                    },
                ]
            },
            "output_settings": {
                "reporting": {
                    "generate_plots": True,
                    "plot_formats": ["png", "pdf"],
                },
                "plotting": {
                    "plot_format": "png",
                    "dpi": 150,
                    "figure_size": [8, 6],
                },
            },
        }

    def test_parameter_name_consistency(self, temp_directory, plotting_config):
        """Test that parameter names are consistent across configuration sections."""
        config_file = temp_directory / "plotting_test.json"
        with open(config_file, "w") as f:
            json.dump(plotting_config, f)

        manager = ConfigManager(str(config_file))

        # Get parameter names from different sections
        param_names = manager.get("initial_parameters", "parameter_names")
        bounds = manager.get("parameter_space", "bounds")
        bound_names = [bound["name"] for bound in bounds]

        # Should be identical
        assert param_names == bound_names
        assert len(param_names) == len(bound_names) == 7

    def test_parameter_count_consistency(self, temp_directory, plotting_config):
        """Test that parameter counts are consistent across all sections."""
        config_file = temp_directory / "count_test.json"
        with open(config_file, "w") as f:
            json.dump(plotting_config, f)

        manager = ConfigManager(str(config_file))

        # Get counts from different sections
        param_names = manager.get("initial_parameters", "parameter_names")
        param_values = manager.get("initial_parameters", "values")
        bounds = manager.get("parameter_space", "bounds")

        # All should have same count
        assert len(param_names) == len(param_values) == len(bounds)

        # Should match effective parameter count
        effective_count = manager.get_effective_parameter_count()
        assert len(param_names) == effective_count

    def test_plotting_configuration_validation(self, temp_directory, plotting_config):
        """Test plotting configuration validation."""
        config_file = temp_directory / "plot_config_test.json"
        with open(config_file, "w") as f:
            json.dump(plotting_config, f)

        manager = ConfigManager(str(config_file))

        # Test plotting settings
        generate_plots = manager.get("output_settings", "reporting", "generate_plots")
        plot_formats = manager.get("output_settings", "reporting", "plot_formats")
        plot_format = manager.get("output_settings", "plotting", "plot_format")

        assert isinstance(generate_plots, bool)
        assert isinstance(plot_formats, list)
        assert isinstance(plot_format, str)

        # Validate format consistency
        valid_formats = ["png", "pdf", "svg", "eps"]
        assert plot_format in valid_formats
        assert all(fmt in valid_formats for fmt in plot_formats)
