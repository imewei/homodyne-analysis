"""
Tests for Static Mode Analysis Functionality
==========================================

Tests the static mode implementation in the homodyne scattering analysis,
including parameter handling, correlation calculations, and mode switching.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

from homodyne.analysis.core import HomodyneAnalysisCore
from homodyne.core.config import ConfigManager


class TestStaticModeAnalysis:
    """Test suite for static mode analysis functionality."""

    @pytest.fixture
    def base_config(self):
        """Base configuration for testing."""
        return {
            "metadata": {"config_version": "5.1"},
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 50},
                "scattering": {"wavevector_q": 0.01},
                "geometry": {"stator_rotor_gap": 2000000},
                "computational": {
                    "num_threads": 1,
                    "auto_detect_cores": False,
                },
            },
            "experimental_data": {
                "data_folder_path": "./data/test/",
                "data_file_name": "test.hdf",
                "phi_angles_path": "./data/test/",
                "phi_angles_file": "phi_list.txt",
                "exchange_key": "exchange",
                "cache_file_path": "./data/test/",
                "cache_filename_template": ("test_cache_{start_frame}_{end_frame}.npz"),
            },
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]}
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 1e-3, "max": 1e6},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": -5000, "max": 5000},
                    {"name": "gamma_dot_t0", "min": 1e-6, "max": 1.0},
                    {"name": "beta", "min": -2.0, "max": 2.0},
                    {"name": "gamma_dot_t_offset", "min": -0.1, "max": 0.1},
                    {"name": "phi0", "min": -15.0, "max": 15.0},
                ]
            },
            "initial_parameters": {
                "values": [1000.0, 0.0, 100.0, 0.001, -0.5, 0.0001, 5.0]
            },
            "advanced_settings": {
                "data_loading": {"use_diagonal_correction": False},
                "chi_squared_calculation": {
                    "_scaling_optimization_note": "Scaling optimization is always enabled: g₂ = offset + contrast × g₁",
                    "validity_check": {
                        "check_positive_D0": True,
                        "check_positive_gamma_dot_t0": True,
                        "check_parameter_bounds": True,
                    },
                },
            },
            "performance_settings": {"parallel_execution": False},
        }

    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_static_mode_detection_by_config(self, temp_directory, base_config):
        """Test static mode detection via configuration flag."""
        # Test static mode enabled
        static_config = base_config.copy()
        static_config["analysis_settings"] = {"static_mode": True}

        config_file = temp_directory / "static_config.json"
        with open(config_file, "w") as f:
            json.dump(static_config, f)

        core = HomodyneAnalysisCore(str(config_file))

        assert core.is_static_mode() is True
        assert core.get_effective_parameter_count() == 3

        # Test flow mode (default)
        flow_config = base_config.copy()
        flow_config["analysis_settings"] = {"static_mode": False}

        config_file = temp_directory / "flow_config.json"
        with open(config_file, "w") as f:
            json.dump(flow_config, f)

        core = HomodyneAnalysisCore(str(config_file))

        assert core.is_static_mode() is False
        assert core.get_effective_parameter_count() == 7

    def test_static_mode_detection_by_parameters(self, temp_directory, base_config):
        """Test static mode detection via parameter values."""
        config_file = temp_directory / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(base_config, f)

        core = HomodyneAnalysisCore(str(config_file))

        # Zero shear parameters (static)
        static_shear_params = np.array([0.0, 0.0, 0.0])
        assert core.is_static_parameters(static_shear_params) is True

        # Non-zero shear parameters (flow)
        flow_shear_params = np.array([0.001, -0.5, 0.0001])
        assert core.is_static_parameters(flow_shear_params) is False

        # Nearly zero parameters (should be detected as static)
        nearly_zero_params = np.array([1e-12, 1e-15, 1e-11])
        assert core.is_static_parameters(nearly_zero_params) is True

    def test_effective_parameter_handling(self, temp_directory, base_config):
        """Test effective parameter extraction for different modes."""
        # Static mode configuration
        static_config = base_config.copy()
        static_config["analysis_settings"] = {"static_mode": True}

        config_file = temp_directory / "static_config.json"
        with open(config_file, "w") as f:
            json.dump(static_config, f)

        core = HomodyneAnalysisCore(str(config_file))

        # Test parameter processing
        original_params = np.array([1000.0, -0.2, 100.0, 0.001, -0.5, 0.0001, 30.0])
        effective_params = core.get_effective_parameters(original_params)

        # In static mode, shear and phi0 should be zeroed
        expected = np.array([1000.0, -0.2, 100.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(effective_params, expected)

        # Test flow mode
        flow_config = base_config.copy()
        flow_config["analysis_settings"] = {"static_mode": False}

        config_file = temp_directory / "flow_config.json"
        with open(config_file, "w") as f:
            json.dump(flow_config, f)

        core = HomodyneAnalysisCore(str(config_file))

        effective_params = core.get_effective_parameters(original_params)
        # In flow mode, all parameters should be preserved
        np.testing.assert_array_equal(effective_params, original_params)

    def test_correlation_calculation_static_vs_flow(self, temp_directory, base_config):
        """Test that correlation calculations differ between static and flow modes."""
        # Create test parameters
        diffusion_params = [1000.0, 0.0, 100.0]
        shear_params = [0.001, -0.5, 0.0001]  # Non-zero shear
        phi0 = 30.0
        params = np.array(diffusion_params + shear_params + [phi0])
        phi_angle = 0.0

        # Static mode calculation
        static_config = base_config.copy()
        static_config["analysis_settings"] = {"static_mode": True}

        config_file = temp_directory / "static_config.json"
        with open(config_file, "w") as f:
            json.dump(static_config, f)

        core_static = HomodyneAnalysisCore(str(config_file))
        c2_static = core_static.calculate_c2_single_angle_optimized(params, phi_angle)

        # Flow mode calculation
        flow_config = base_config.copy()
        flow_config["analysis_settings"] = {"static_mode": False}

        config_file = temp_directory / "flow_config.json"
        with open(config_file, "w") as f:
            json.dump(flow_config, f)

        core_flow = HomodyneAnalysisCore(str(config_file))
        c2_flow = core_flow.calculate_c2_single_angle_optimized(params, phi_angle)

        # Results should be different (static ignores shear, flow includes it)
        max_diff = np.abs(c2_static - c2_flow).max()
        assert max_diff > 1e-6, "Static and flow mode should produce different results"

        # Both should have same shape
        assert c2_static.shape == c2_flow.shape

        # Both should be positive and normalized
        assert np.all(c2_static >= 0) and np.all(c2_static <= 1)
        assert np.all(c2_flow >= 0) and np.all(c2_flow <= 1)

    def test_phi0_irrelevance_in_static_mode(self, temp_directory, base_config):
        """Test that phi0 is irrelevant in static mode."""
        static_config = base_config.copy()
        static_config["analysis_settings"] = {"static_mode": True}

        config_file = temp_directory / "static_config.json"
        with open(config_file, "w") as f:
            json.dump(static_config, f)

        core = HomodyneAnalysisCore(str(config_file))

        # Test parameters with different phi0 values
        base_params = [1000.0, 0.0, 100.0, 0.001, -0.5, 0.0001]
        params_phi0_0 = np.array(base_params + [0.0])
        params_phi0_45 = np.array(base_params + [45.0])

        phi_angle = 0.0

        # Both should give identical results in static mode
        c2_phi0_0 = core.calculate_c2_single_angle_optimized(params_phi0_0, phi_angle)
        c2_phi0_45 = core.calculate_c2_single_angle_optimized(params_phi0_45, phi_angle)

        # Results should be identical (within numerical precision)
        max_diff = np.abs(c2_phi0_0 - c2_phi0_45).max()
        assert max_diff < 1e-15, "phi0 should be irrelevant in static mode"


class TestConfigManagerStaticMode:
    """Test ConfigManager static mode methods."""

    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_config_manager_static_mode_methods(self, temp_directory):
        """Test ConfigManager static mode detection methods."""
        # Test with static mode enabled
        static_config = {
            "metadata": {"config_version": "5.1"},
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 50}
            },
            "experimental_data": {"data_folder_path": "./test/"},
            "optimization_config": {"classical_optimization": {"methods": ["test"]}},
            "analysis_settings": {"static_mode": True},
        }

        config_file = temp_directory / "static_test.json"
        with open(config_file, "w") as f:
            json.dump(static_config, f)

        manager = ConfigManager(str(config_file))

        assert manager.is_static_mode_enabled() is True
        assert manager.get_analysis_mode() == "static_anisotropic"
        assert manager.get_effective_parameter_count() == 3

        settings = manager.get_analysis_settings()
        assert settings["static_mode"] is True
        assert "model_description" in settings

        # Test with static mode disabled
        static_config["analysis_settings"]["static_mode"] = False

        config_file = temp_directory / "flow_test.json"
        with open(config_file, "w") as f:
            json.dump(static_config, f)

        manager = ConfigManager(str(config_file))

        assert manager.is_static_mode_enabled() is False
        assert manager.get_analysis_mode() == "laminar_flow"
        assert manager.get_effective_parameter_count() == 7


def test_static_mode_integration():
    """Integration test using default configuration files."""
    try:
        # Test with template config (should default to flow mode)
        core = HomodyneAnalysisCore("homodyne/config_template.json")

        assert core.is_static_mode() is False
        assert core.get_effective_parameter_count() == 7

        # Test with config override for static mode
        config_override = {"analysis_settings": {"static_mode": True}}
        core_static = HomodyneAnalysisCore(
            "homodyne/config_template.json", config_override
        )

        assert core_static.is_static_mode() is True
        assert core_static.get_effective_parameter_count() == 3

        # Test parameter detection
        zero_shear = np.array([0.0, 0.0, 0.0])
        non_zero_shear = np.array([0.001, -0.5, 0.0001])

        assert core.is_static_parameters(zero_shear) is True
        assert core.is_static_parameters(non_zero_shear) is False

    except FileNotFoundError:
        pytest.skip("Template config file not found - skipping integration test")
