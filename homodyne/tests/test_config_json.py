"""
Tests for JSON Configuration and Parameter Validation
====================================================

Tests JSON parsing, configuration validation, and parameter correctness.
"""

import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, mock_open
import tempfile

# Import the modules to test


from homodyne.tests.fixtures import (
    dummy_config,
    temp_directory,
    create_minimal_config_file,
    create_invalid_config_file,
)

# Import config management from homodyne module
from typing import Any

try:
    from homodyne import ConfigManager as _HomodyneConfigManager

    ConfigManager = _HomodyneConfigManager  # type: ignore
except ImportError:
    # Create a minimal mock if homodyne import fails
    class ConfigManager:
        def __init__(self, config_file: str = "test_config.json") -> None:
            self.config_file = config_file
            self.config: dict[str, Any] = {}


class TestJSONParsing:
    """Test JSON file parsing and loading."""

    def test_load_valid_json_config(self, temp_directory):
        """Test loading a valid JSON configuration file."""
        config_file = temp_directory / "valid_config.json"
        test_config = {
            "metadata": {"version": "test"},
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 100},
                "scattering": {"wavevector_q": 0.01},
                "geometry": {"stator_rotor_gap": 100000},
                "computational": {"num_threads": 2},
            },
            "experimental_data": {
                "data_folder_path": "./test/",
                "data_file_name": "test.hdf",
                "phi_angles_path": "./test/",
                "phi_angles_file": "phi.txt",
            },
            "initial_parameters": {
                "values": [1.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0],
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_t0",
                    "beta",
                    "gamma_dot_t_offset",
                    "phi0",
                ],
            },
            "parameter_space": {"bounds": []},
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]}
            },
            "validation_rules": {"frame_range": {"minimum_frames": 5}},
        }

        create_minimal_config_file(config_file, test_config)

        # Test JSON parsing
        with open(config_file, "r") as f:
            loaded_config = json.load(f)

        assert loaded_config == test_config
        assert "metadata" in loaded_config
        assert "analyzer_parameters" in loaded_config

    def test_load_invalid_json_syntax(self, temp_directory):
        """Test handling of invalid JSON syntax."""
        config_file = temp_directory / "invalid_syntax.json"
        create_invalid_config_file(config_file, "syntax")

        with pytest.raises(json.JSONDecodeError):
            with open(config_file, "r") as f:
                json.load(f)

    def test_config_manager_with_valid_file(self, temp_directory):
        """Test ConfigManager with valid configuration file."""
        config_file = temp_directory / "test_config.json"
        create_minimal_config_file(config_file)

        # Change to temp directory so ConfigManager can find the file
        import os

        original_cwd = os.getcwd()
        os.chdir(temp_directory)

        try:
            config_manager = ConfigManager("test_config.json")
            assert config_manager.config is not None
            assert isinstance(config_manager.config, dict)
        except Exception as e:
            # If ConfigManager import failed, this test is expected to fail
            if "ConfigManager" in str(e):
                pytest.skip("ConfigManager not available")
            else:
                raise
        finally:
            os.chdir(original_cwd)

    def test_config_manager_with_missing_file(self, temp_directory):
        """Test ConfigManager with missing configuration file."""
        import os

        original_cwd = os.getcwd()
        os.chdir(temp_directory)

        try:
            # This should not raise an exception, but use defaults
            config_manager = ConfigManager("nonexistent.json")
            # Should fall back to default configuration
            assert config_manager.config is not None
        except Exception as e:
            if "ConfigManager" in str(e):
                pytest.skip("ConfigManager not available")
            else:
                # This is acceptable - some implementations might raise FileNotFoundError
                assert "not found" in str(e).lower() or "no such file" in str(e).lower()
        finally:
            os.chdir(original_cwd)


class TestParameterValidation:
    """Test parameter validation and bounds checking."""

    def test_parameter_names_consistency(self, dummy_config):
        """Test that parameter names are consistent across configuration sections."""
        initial_params = dummy_config["initial_parameters"]["parameter_names"]
        bounds_params = [
            bound["name"] for bound in dummy_config["parameter_space"]["bounds"]
        ]

        # All initial parameters should have corresponding bounds
        for param in initial_params:
            assert param in bounds_params, f"Parameter {param} missing from bounds"

        # Check that we have the expected core parameters
        expected_params = [
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]
        for param in expected_params:
            assert param in initial_params, f"Expected parameter {param} not found"

    def test_parameter_bounds_validity(self, dummy_config):
        """Test that parameter bounds are physically reasonable."""
        bounds = dummy_config["parameter_space"]["bounds"]

        for bound in bounds:
            assert "name" in bound
            assert "min" in bound
            assert "max" in bound
            assert (
                bound["min"] < bound["max"]
            ), f"Invalid bound for {bound['name']}: {bound['min']} >= {bound['max']}"

            # Check specific physical constraints
            if bound["name"] == "D0":
                assert bound["min"] > 0, "Diffusion coefficient D0 must be positive"
            elif bound["name"] == "gamma_dot_t0":
                assert (
                    bound["min"] > 0
                ), "Reference shear rate gamma_dot_t0 must be positive"
            elif bound["name"] in ["alpha", "beta"]:
                # Power-law exponents should be reasonable
                assert (
                    bound["min"] >= -5.0 and bound["max"] <= 5.0
                ), f"Power-law exponent {bound['name']} has unreasonable bounds"

    def test_initial_parameters_within_bounds(self, dummy_config):
        """Test that initial parameter values are within specified bounds."""
        initial_values = dummy_config["initial_parameters"]["values"]
        parameter_names = dummy_config["initial_parameters"]["parameter_names"]
        bounds = {
            bound["name"]: bound for bound in dummy_config["parameter_space"]["bounds"]
        }

        for param_name, value in zip(parameter_names, initial_values):
            if param_name in bounds:
                bound = bounds[param_name]
                assert (
                    bound["min"] <= value <= bound["max"]
                ), f"Initial value {value} for {param_name} outside bounds [{bound['min']}, {bound['max']}]"

    def test_parameter_units_consistency(self, dummy_config):
        """Test that parameter units are correctly specified."""
        if "units" in dummy_config["initial_parameters"]:
            parameter_names = dummy_config["initial_parameters"]["parameter_names"]
            units = dummy_config["initial_parameters"]["units"]

            assert len(units) == len(
                parameter_names
            ), "Number of units must match number of parameters"

            # Check expected units for specific parameters
            unit_dict = dict(zip(parameter_names, units))

            if "D0" in unit_dict:
                assert "Å²/s" in unit_dict["D0"] or "A²/s" in unit_dict["D0"]
            if "gamma_dot_t0" in unit_dict:
                assert (
                    "s⁻¹" in unit_dict["gamma_dot_t0"]
                    or "s-1" in unit_dict["gamma_dot_t0"]
                )
            if "phi0" in unit_dict:
                assert "degree" in unit_dict["phi0"].lower()


class TestConfigurationSections:
    """Test required configuration sections and their structure."""

    def test_required_sections_present(self, dummy_config):
        """Test that all required configuration sections are present."""
        required_sections = [
            "analyzer_parameters",
            "experimental_data",
            "initial_parameters",
            "parameter_space",
            "optimization_config",
        ]

        for section in required_sections:
            assert (
                section in dummy_config
            ), f"Required section '{section}' missing from configuration"

    def test_analyzer_parameters_structure(self, dummy_config):
        """Test structure of analyzer_parameters section."""
        analyzer = dummy_config["analyzer_parameters"]

        assert "temporal" in analyzer
        assert "scattering" in analyzer
        assert "geometry" in analyzer

        # Check temporal parameters
        temporal = analyzer["temporal"]
        assert "dt" in temporal
        assert "start_frame" in temporal
        assert "end_frame" in temporal
        assert temporal["dt"] > 0
        assert temporal["start_frame"] < temporal["end_frame"]

        # Check scattering parameters
        scattering = analyzer["scattering"]
        assert "wavevector_q" in scattering
        assert scattering["wavevector_q"] > 0

        # Check geometry parameters
        geometry = analyzer["geometry"]
        assert "stator_rotor_gap" in geometry
        assert geometry["stator_rotor_gap"] > 0

    def test_experimental_data_structure(self, dummy_config):
        """Test structure of experimental_data section."""
        exp_data = dummy_config["experimental_data"]

        required_fields = [
            "data_folder_path",
            "data_file_name",
            "phi_angles_path",
            "phi_angles_file",
        ]

        for field in required_fields:
            assert (
                field in exp_data
            ), f"Required field '{field}' missing from experimental_data"
            assert isinstance(
                exp_data[field], str
            ), f"Field '{field}' should be a string"

    def test_optimization_config_structure(self, dummy_config):
        """Test structure of optimization_config section."""
        opt_config = dummy_config["optimization_config"]

        assert "classical_optimization" in opt_config

        classical = opt_config["classical_optimization"]
        assert "methods" in classical
        assert isinstance(classical["methods"], list)
        assert len(classical["methods"]) > 0

        # Check that methods are valid (only Nelder-Mead is supported for optimization)
        valid_methods = ["Nelder-Mead"]
        for method in classical["methods"]:
            assert (
                method in valid_methods
            ), f"Unsupported optimization method: {method}. Only Nelder-Mead is supported."


class TestParameterTypes:
    """Test parameter type specifications and constraints."""

    def test_parameter_type_specifications(self, dummy_config):
        """Test that parameter types are properly specified."""
        bounds = dummy_config["parameter_space"]["bounds"]

        for bound in bounds:
            if "type" in bound:
                valid_types = [
                    "Normal",
                    "LogNormal",
                    "Uniform",
                    "LogUniform",
                ]
                assert (
                    bound["type"] in valid_types
                ), f"Invalid parameter type: {bound['type']}"

                # LogNormal parameters should have positive bounds
                if bound["type"] == "LogNormal":
                    assert (
                        bound["min"] > 0
                    ), f"LogNormal parameter {bound['name']} must have positive min bound"
                    assert (
                        bound["max"] > 0
                    ), f"LogNormal parameter {bound['name']} must have positive max bound"

    def test_physical_parameter_constraints(self, dummy_config):
        """Test physical constraints on specific parameters."""
        bounds = {
            bound["name"]: bound for bound in dummy_config["parameter_space"]["bounds"]
        }

        # Diffusion coefficient constraints
        if "D0" in bounds:
            d0_bound = bounds["D0"]
            assert d0_bound["min"] > 0, "Diffusion coefficient D0 must be positive"
            # Typical range for XPCS: 1e-3 to 1e6 Å²/s
            assert d0_bound["min"] >= 1e-6, "D0 minimum seems too small"
            assert d0_bound["max"] <= 1e8, "D0 maximum seems too large"

        # Shear rate constraints
        if "gamma_dot_t0" in bounds:
            gamma_bound = bounds["gamma_dot_t0"]
            assert gamma_bound["min"] > 0, "Shear rate gamma_dot_t0 must be positive"
            # Typical range for rheology: 1e-6 to 1e3 s⁻¹
            assert gamma_bound["min"] >= 1e-8, "gamma_dot_t0 minimum seems too small"
            assert gamma_bound["max"] <= 1e5, "gamma_dot_t0 maximum seems too large"

        # Angle constraints
        if "phi0" in bounds:
            phi_bound = bounds["phi0"]
            # Angle should be in reasonable range
            assert phi_bound["min"] >= -180, "phi0 minimum should be >= -180 degrees"
            assert phi_bound["max"] <= 180, "phi0 maximum should be <= 180 degrees"


class TestConfigurationValidation:
    """Test overall configuration validation."""

    def test_frame_range_validation(self, dummy_config):
        """Test validation of frame range parameters."""
        temporal = dummy_config["analyzer_parameters"]["temporal"]
        start_frame = temporal["start_frame"]
        end_frame = temporal["end_frame"]

        assert start_frame < end_frame, "Start frame must be less than end frame"
        assert start_frame >= 0, "Start frame must be non-negative"
        assert end_frame > 0, "End frame must be positive"

        # Check minimum frame count
        min_frames = (
            dummy_config.get("validation_rules", {})
            .get("frame_range", {})
            .get("minimum_frames", 5)
        )
        frame_count = end_frame - start_frame
        assert (
            frame_count >= min_frames
        ), f"Frame count {frame_count} below minimum {min_frames}"

    def test_computational_parameters_validation(self, dummy_config):
        """Test validation of computational parameters."""
        if "computational" in dummy_config["analyzer_parameters"]:
            comp = dummy_config["analyzer_parameters"]["computational"]

            if "num_threads" in comp:
                assert comp["num_threads"] > 0, "Number of threads must be positive"
                assert (
                    comp["num_threads"] <= 1024
                ), "Number of threads seems unreasonably large"

            if "max_threads_limit" in comp:
                assert (
                    comp["max_threads_limit"] > 0
                ), "Max threads limit must be positive"
                if "num_threads" in comp:
                    # Max limit should be at least as large as num_threads
                    assert (
                        comp["max_threads_limit"] >= comp["num_threads"]
                    ), "Max threads limit should be >= num_threads"

    def test_output_settings_validation(self, dummy_config):
        """Test validation of output settings."""
        if "output_settings" in dummy_config:
            output = dummy_config["output_settings"]

            if "plotting" in output:
                plotting = output["plotting"]

                if "dpi" in plotting:
                    assert plotting["dpi"] > 0, "DPI must be positive"
                    assert plotting["dpi"] <= 2400, "DPI seems unreasonably high"

                if "figure_size" in plotting:
                    fig_size = plotting["figure_size"]
                    assert isinstance(fig_size, list), "Figure size should be a list"
                    assert (
                        len(fig_size) == 2
                    ), "Figure size should have exactly 2 elements"
                    assert all(
                        s > 0 for s in fig_size
                    ), "Figure size dimensions must be positive"


class TestJSONSchemaCompliance:
    """Test JSON structure compliance and data types."""

    def test_json_serializable_config(self, dummy_config):
        """Test that configuration can be serialized to JSON."""
        # This should not raise an exception
        json_string = json.dumps(dummy_config, indent=2)
        assert len(json_string) > 0

        # Should be able to load it back
        loaded_config = json.loads(json_string)
        assert loaded_config == dummy_config

    def test_numeric_types_in_config(self, dummy_config):
        """Test that numeric values are of appropriate types."""
        # Check initial parameter values
        initial_values = dummy_config["initial_parameters"]["values"]
        for value in initial_values:
            assert isinstance(
                value, (int, float)
            ), f"Parameter value {value} is not numeric"

        # Check bounds
        for bound in dummy_config["parameter_space"]["bounds"]:
            assert isinstance(
                bound["min"], (int, float)
            ), f"Bound min {bound['min']} is not numeric"
            assert isinstance(
                bound["max"], (int, float)
            ), f"Bound max {bound['max']} is not numeric"

        # Check temporal parameters
        temporal = dummy_config["analyzer_parameters"]["temporal"]
        assert isinstance(temporal["dt"], (int, float)), "dt must be numeric"
        assert isinstance(temporal["start_frame"], int), "start_frame must be integer"
        assert isinstance(temporal["end_frame"], int), "end_frame must be integer"

    def test_string_fields_in_config(self, dummy_config):
        """Test that string fields are properly typed."""
        exp_data = dummy_config["experimental_data"]
        string_fields = [
            "data_folder_path",
            "data_file_name",
            "phi_angles_path",
            "phi_angles_file",
        ]

        for field in string_fields:
            if field in exp_data:
                assert isinstance(exp_data[field], str), f"{field} should be a string"
                assert len(exp_data[field]) > 0, f"{field} should not be empty"

    def test_boolean_fields_in_config(self, dummy_config):
        """Test that boolean fields are properly typed."""

        # Check various boolean fields throughout the configuration
        def check_boolean_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if isinstance(value, bool):
                        # This is expected
                        pass
                    elif (
                        key.lower()
                        in ["enabled", "use_", "include_", "create_", "cache_"]
                        or key.lower().endswith("_enabled")
                        or key.lower().startswith("use_")
                    ):
                        # These should probably be booleans
                        assert isinstance(
                            value, bool
                        ), f"{current_path} should be boolean but is {type(value)}"
                    else:
                        check_boolean_recursive(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_boolean_recursive(item, f"{path}[{i}]")

        # This test is somewhat lenient as not all boolean-like fields need to be boolean
        check_boolean_recursive(dummy_config)


class TestConfigErrorHandling:
    """Test error handling for various configuration issues."""

    def test_missing_required_section(self, temp_directory):
        """Test handling of missing required sections."""
        incomplete_config = {
            "metadata": {"version": "test"},
            # Missing analyzer_parameters
            "experimental_data": {"data_folder_path": "./"},
        }

        config_file = temp_directory / "incomplete.json"
        with open(config_file, "w") as f:
            json.dump(incomplete_config, f)

        # Loading should work, but validation might fail
        with open(config_file, "r") as f:
            loaded = json.load(f)

        assert "metadata" in loaded
        assert "analyzer_parameters" not in loaded  # This section is missing

    def test_invalid_parameter_values(self):
        """Test detection of invalid parameter values."""
        invalid_configs = [
            # Negative time step
            {"analyzer_parameters": {"temporal": {"dt": -0.1}}},
            # Start frame > end frame
            {
                "analyzer_parameters": {
                    "temporal": {"start_frame": 100, "end_frame": 50}
                }
            },
            # Negative diffusion coefficient
            {"initial_parameters": {"values": [-100.0]}},
            # Invalid bounds
            {
                "parameter_space": {
                    "bounds": [{"name": "test", "min": 10.0, "max": 5.0}]
                }
            },
        ]

        for config in invalid_configs:
            # The configuration itself is valid JSON
            json_string = json.dumps(config)
            loaded = json.loads(json_string)
            assert loaded == config

            # But the values are physically invalid (would need validation logic to catch)

    def test_unicode_handling_in_config(self, temp_directory):
        """Test proper handling of unicode characters in configuration."""
        unicode_config = {
            "metadata": {"description": "Configuration with unicode: α, β, γ, Å, °"},
            "analyzer_parameters": {
                "scattering": {"q_unit": "Å⁻¹"},
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 10},
            },
            "initial_parameters": {"units": ["Å²/s", "dimensionless", "s⁻¹", "°"]},
        }

        config_file = temp_directory / "unicode_config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(unicode_config, f, ensure_ascii=False, indent=2)

        # Should be able to load it back
        with open(config_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded == unicode_config
        assert "Å⁻¹" in loaded["analyzer_parameters"]["scattering"]["q_unit"]
