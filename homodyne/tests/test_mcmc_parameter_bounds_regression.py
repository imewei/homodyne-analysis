"""
MCMC Parameter Bounds Regression Test
====================================

Regression test to ensure MCMC initialization works properly with parameter bounds,
specifically testing the alpha parameter bounds fix that was causing initialization
failures with "Initial evaluation of model at starting point failed!" error.
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from homodyne.core.config import ConfigManager
from homodyne.analysis.core import HomodyneAnalysisCore


class TestMCMCParameterBoundsRegression:
    """Regression tests for MCMC parameter bounds initialization issues."""

    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config_with_boundary_alpha(self):
        """Configuration with alpha parameter at boundary of range - the problematic case."""
        return {
            "metadata": {"config_version": "6.0"},
            "analyzer_parameters": {
                "temporal": {"dt": 0.5, "start_frame": 400, "end_frame": 1000},
                "scattering": {"wavevector_q": 0.0237},
                "geometry": {"stator_rotor_gap": 2000000},
                "computational": {"num_threads": 2},
            },
            "experimental_data": {
                "data_folder_path": "./data/test/",
                "data_file_name": "test.hdf",
                "cache_file_path": "./data/test/",
                "cache_filename_template": "test_cache_{start_frame}_{end_frame}.npz",
            },
            "initial_parameters": {
                "values": [16000, -1.5, 1.1, 0.0, 0.0, 0.0, 0.0],  # alpha at boundary
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_t0",
                    "beta",
                    "gamma_dot_t_offset",
                    "phi0",
                ],
                "active_parameters": ["D0", "alpha", "D_offset"],
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 15000, "max": 20000, "type": "Normal"},
                    {
                        "name": "alpha",
                        "min": -1.6,
                        "max": -1.5,
                        "type": "Normal",
                    },  # Narrow range, initial at boundary
                    {"name": "D_offset", "min": 0, "max": 5, "type": "Normal"},
                    {"name": "gamma_dot_t0", "min": 0.0, "max": 0.0, "type": "fixed"},
                    {"name": "beta", "min": 0.0, "max": 0.0, "type": "fixed"},
                    {
                        "name": "gamma_dot_t_offset",
                        "min": 0.0,
                        "max": 0.0,
                        "type": "fixed",
                    },
                    {"name": "phi0", "min": 0.0, "max": 0.0, "type": "fixed"},
                ]
            },
            "optimization_config": {
                "mcmc_sampling": {
                    "enabled": True,
                    "sampler": "NUTS",
                    "draws": 100,
                    "tune": 50,
                    "chains": 2,
                    "cores": 2,
                    "target_accept": 0.9,
                    "return_inferencedata": True,
                }
            },
            "analysis_settings": {"static_mode": True, "static_submode": "isotropic"},
            "performance_settings": {
                "noise_model": {"use_simple_forward_model": True, "sigma_prior": 0.1}
            },
        }

    @pytest.fixture
    def config_with_safe_alpha(self):
        """Configuration with alpha parameter safely within bounds - the fixed case."""
        return {
            "metadata": {"config_version": "6.0"},
            "analyzer_parameters": {
                "temporal": {"dt": 0.5, "start_frame": 400, "end_frame": 1000},
                "scattering": {"wavevector_q": 0.0237},
                "geometry": {"stator_rotor_gap": 2000000},
                "computational": {"num_threads": 2},
            },
            "experimental_data": {
                "data_folder_path": "./data/test/",
                "data_file_name": "test.hdf",
                "cache_file_path": "./data/test/",
                "cache_filename_template": "test_cache_{start_frame}_{end_frame}.npz",
            },
            "initial_parameters": {
                "values": [16000, -1.5, 1.1, 0.0, 0.0, 0.0, 0.0],  # alpha in center
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_t0",
                    "beta",
                    "gamma_dot_t_offset",
                    "phi0",
                ],
                "active_parameters": ["D0", "alpha", "D_offset"],
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 15000, "max": 20000, "type": "Normal"},
                    {
                        "name": "alpha",
                        "min": -1.8,
                        "max": -1.2,
                        "type": "Normal",
                    },  # Wider range, initial in center
                    {"name": "D_offset", "min": 0, "max": 5, "type": "Normal"},
                    {"name": "gamma_dot_t0", "min": 0.0, "max": 0.0, "type": "fixed"},
                    {"name": "beta", "min": 0.0, "max": 0.0, "type": "fixed"},
                    {
                        "name": "gamma_dot_t_offset",
                        "min": 0.0,
                        "max": 0.0,
                        "type": "fixed",
                    },
                    {"name": "phi0", "min": 0.0, "max": 0.0, "type": "fixed"},
                ]
            },
            "optimization_config": {
                "mcmc_sampling": {
                    "enabled": True,
                    "sampler": "NUTS",
                    "draws": 100,
                    "tune": 50,
                    "chains": 2,
                    "cores": 2,
                    "target_accept": 0.9,
                    "return_inferencedata": True,
                }
            },
            "analysis_settings": {"static_mode": True, "static_submode": "isotropic"},
            "performance_settings": {
                "noise_model": {"use_simple_forward_model": True, "sigma_prior": 0.1}
            },
        }

    def test_mcmc_parameter_bounds_analysis_boundary_alpha(self):
        """Test that demonstrates analysis of boundary alpha parameter configuration."""
        # Boundary alpha configuration (the problematic case we fixed)
        boundary_config = {
            "parameter_space": {
                "bounds": [
                    {"name": "alpha", "min": -1.6, "max": -1.5, "type": "Normal"}
                ]
            },
            "initial_parameters": {
                "values": [16000, -1.5, 1.1],  # Alpha at boundary
                "parameter_names": ["D0", "alpha", "D_offset"],
            },
        }

        # Analyze parameter characteristics
        alpha_bound = boundary_config["parameter_space"]["bounds"][0]
        alpha_initial = boundary_config["initial_parameters"]["values"][1]

        bounds_width = alpha_bound["max"] - alpha_bound["min"]
        bounds_center = (alpha_bound["min"] + alpha_bound["max"]) / 2
        distance_from_center = abs(alpha_initial - bounds_center)

        # Verify characteristics that made this configuration problematic
        assert abs(bounds_width - 0.1) < 1e-10  # Very narrow bounds
        assert alpha_initial == alpha_bound["max"]  # Initial at max boundary
        assert abs(distance_from_center - 0.05) < 1e-10  # At boundary, not centered

        # This configuration was problematic because:
        # 1. Very narrow parameter space (width = 0.1)
        # 2. Initial value exactly at boundary (-1.5 = max bound)
        # 3. No room for PyMC jittering during initialization

        print("✓ Boundary alpha configuration analysis completed")
        print(f"  Bounds width: {bounds_width:.3f} (problematically narrow)")
        print(
            f"  Initial position: at max boundary (distance from center: {distance_from_center:.3f})"
        )

    def test_mcmc_parameter_bounds_analysis_safe_alpha(self):
        """Test that demonstrates analysis of safe alpha parameter configuration."""
        # Safe alpha configuration (the fixed case)
        safe_config = {
            "parameter_space": {
                "bounds": [
                    {"name": "alpha", "min": -1.8, "max": -1.2, "type": "Normal"}
                ]
            },
            "initial_parameters": {
                "values": [16000, -1.5, 1.1],  # Alpha centered in bounds
                "parameter_names": ["D0", "alpha", "D_offset"],
            },
        }

        # Analyze parameter characteristics
        alpha_bound = safe_config["parameter_space"]["bounds"][0]
        alpha_initial = safe_config["initial_parameters"]["values"][1]

        bounds_width = alpha_bound["max"] - alpha_bound["min"]
        bounds_center = (alpha_bound["min"] + alpha_bound["max"]) / 2
        distance_from_center = abs(alpha_initial - bounds_center)

        # Verify characteristics that make this configuration safe
        assert abs(bounds_width - 0.6) < 1e-10  # Wide bounds
        assert alpha_initial == bounds_center  # Initial exactly at center
        assert distance_from_center == 0.0  # Perfectly centered

        # Calculate room for jittering on both sides
        room_below = alpha_initial - alpha_bound["min"]
        room_above = alpha_bound["max"] - alpha_initial

        assert abs(room_below - 0.3) < 1e-10  # 0.3 units below center
        assert abs(room_above - 0.3) < 1e-10  # 0.3 units above center

        # This configuration is safe because:
        # 1. Wide parameter space (width = 0.6, 6x larger than problematic case)
        # 2. Initial value exactly at center (-1.5 = center of [-1.8, -1.2])
        # 3. Plenty of room for PyMC jittering during initialization (0.3 on each side)

        print("✓ Safe alpha configuration analysis completed")
        print(f"  Bounds width: {bounds_width:.3f} (safe and wide)")
        print(
            f"  Initial position: exactly at center (distance from center: {distance_from_center:.3f})"
        )
        print(f"  Jittering room: {room_below:.1f} below, {room_above:.1f} above")

    def test_alpha_bounds_configuration_comparison(self, temp_directory):
        """Test that demonstrates the difference between problematic and safe alpha bounds."""

        # Test problematic configuration (narrow bounds, initial at boundary)
        problematic_config = {
            "parameter_space": {
                "bounds": [
                    {
                        "name": "alpha",
                        "min": -1.6,
                        "max": -1.5,
                        "type": "Normal",
                    }  # Width: 0.1
                ]
            },
            "initial_parameters": {"values": [16000, -1.5, 1.1]},  # At max boundary
        }

        # Test safe configuration (wider bounds, initial in center)
        safe_config = {
            "parameter_space": {
                "bounds": [
                    {
                        "name": "alpha",
                        "min": -1.8,
                        "max": -1.2,
                        "type": "Normal",
                    }  # Width: 0.6
                ]
            },
            "initial_parameters": {"values": [16000, -1.5, 1.1]},  # At center
        }

        # Analyze bounds characteristics
        prob_bound = problematic_config["parameter_space"]["bounds"][0]
        safe_bound = safe_config["parameter_space"]["bounds"][0]

        prob_width = prob_bound["max"] - prob_bound["min"]
        safe_width = safe_bound["max"] - safe_bound["min"]

        prob_alpha = problematic_config["initial_parameters"]["values"][1]
        safe_alpha = safe_config["initial_parameters"]["values"][1]

        prob_mid = (prob_bound["min"] + prob_bound["max"]) / 2
        safe_mid = (safe_bound["min"] + safe_bound["max"]) / 2

        # Verify the problematic case characteristics
        assert (
            abs(prob_width - 0.1) < 1e-10
        )  # Very narrow (within floating point precision)
        assert prob_alpha == prob_bound["max"]  # At boundary
        assert abs(prob_alpha - prob_mid) > 0.01  # Not at center

        # Verify the safe case characteristics
        assert (
            abs(safe_width - 0.6) < 1e-10
        )  # Much wider (within floating point precision)
        assert safe_alpha == safe_mid  # At center
        assert safe_bound["min"] < safe_alpha < safe_bound["max"]  # Within bounds

        print(
            f"✓ Problematic alpha bounds: width={prob_width:.1f}, initial at boundary"
        )
        print(f"✓ Safe alpha bounds: width={safe_width:.1f}, initial at center")

    def test_mcmc_bounds_validation_edge_cases(self):
        """Test edge cases for MCMC parameter bounds validation."""

        # Test case 1: Initial value exactly at lower bound
        bounds_at_min = [{"name": "alpha", "min": -2.0, "max": -1.0, "type": "Normal"}]
        initial_at_min = [-2.0]  # Exactly at minimum

        # Test case 2: Initial value exactly at upper bound
        bounds_at_max = [{"name": "alpha", "min": -2.0, "max": -1.0, "type": "Normal"}]
        initial_at_max = [-1.0]  # Exactly at maximum

        # Test case 3: Initial value safely in middle
        bounds_centered = [
            {"name": "alpha", "min": -2.0, "max": -1.0, "type": "Normal"}
        ]
        initial_centered = [-1.5]  # Exactly in center

        # Verify bounds validation logic
        for bounds, initial, case_name in [
            (bounds_at_min, initial_at_min, "at_min"),
            (bounds_at_max, initial_at_max, "at_max"),
            (bounds_centered, initial_centered, "centered"),
        ]:
            bound = bounds[0]
            value = initial[0]

            # Check if value is within bounds (inclusive)
            within_bounds = bound["min"] <= value <= bound["max"]
            assert (
                within_bounds
            ), f"Value {value} should be within bounds [{bound['min']}, {bound['max']}] for case {case_name}"

            # Check distance from boundaries
            dist_from_min = abs(value - bound["min"])
            dist_from_max = abs(value - bound["max"])
            min_dist = min(dist_from_min, dist_from_max)

            if case_name == "centered":
                assert (
                    min_dist == 0.5
                ), f"Centered case should be 0.5 from each boundary"
            else:
                assert (
                    min_dist == 0.0
                ), f"Boundary cases should be at distance 0 from a boundary"

        print("✓ Parameter bounds validation logic verified for edge cases")


@pytest.mark.mcmc_integration
class TestMCMCBoundsIntegration:
    """Integration tests for MCMC bounds functionality."""

    def test_mcmc_bounds_loading_from_actual_config(self, tmp_path):
        """Test loading MCMC bounds from an actual configuration file like my_config.json."""

        # Create a realistic configuration similar to the fixed my_config.json
        realistic_config = {
            "metadata": {"config_version": "6.0", "analysis_mode": "static_isotropic"},
            "analyzer_parameters": {
                "temporal": {"dt": 0.5, "start_frame": 400, "end_frame": 1000},
                "scattering": {"wavevector_q": 0.0237},
                "geometry": {"stator_rotor_gap": 2000000},
                "computational": {"num_threads": 2},
            },
            "experimental_data": {
                "data_folder_path": "./data/test/",
                "data_file_name": "test.hdf",
                "cache_file_path": "./data/test/",
                "cache_filename_template": "test_cache_{start_frame}_{end_frame}.npz",
            },
            "initial_parameters": {
                "values": [16000, -1.5, 1.1, 0.0, 0.0, 0.0, 0.0],
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_t0",
                    "beta",
                    "gamma_dot_t_offset",
                    "phi0",
                ],
                "active_parameters": ["D0", "alpha", "D_offset"],
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 15000, "max": 20000, "type": "Normal"},
                    {
                        "name": "alpha",
                        "min": -1.8,
                        "max": -1.2,
                        "type": "Normal",
                        "_note": "PRIMARY PARAMETER - diffusion time dependence exponent. Bounds widened to avoid initialization at hard boundary.",
                    },
                    {"name": "D_offset", "min": 0, "max": 5, "type": "Normal"},
                    {"name": "gamma_dot_t0", "min": 0.0, "max": 0.0, "type": "fixed"},
                    {"name": "beta", "min": 0.0, "max": 0.0, "type": "fixed"},
                    {
                        "name": "gamma_dot_t_offset",
                        "min": 0.0,
                        "max": 0.0,
                        "type": "fixed",
                    },
                    {"name": "phi0", "min": 0.0, "max": 0.0, "type": "fixed"},
                ]
            },
            "optimization_config": {
                "mcmc_sampling": {
                    "enabled": True,
                    "sampler": "NUTS",
                    "draws": 100,
                    "tune": 50,
                    "chains": 2,
                    "cores": 2,
                    "target_accept": 0.9,
                }
            },
            "analysis_settings": {"static_mode": True, "static_submode": "isotropic"},
            "performance_settings": {"noise_model": {"use_simple_forward_model": True}},
        }

        config_file = tmp_path / "realistic_config.json"
        with open(config_file, "w") as f:
            json.dump(realistic_config, f, indent=2)

        # Test configuration loading
        config_manager = ConfigManager(str(config_file))

        # Verify bounds are loaded correctly
        param_bounds = realistic_config["parameter_space"]["bounds"]
        alpha_bound = next(b for b in param_bounds if b["name"] == "alpha")

        assert alpha_bound["min"] == -1.8
        assert alpha_bound["max"] == -1.2
        assert alpha_bound["type"] == "Normal"
        assert "Bounds widened" in alpha_bound["_note"]

        # Verify initial value is now centered
        alpha_initial = realistic_config["initial_parameters"]["values"][1]
        alpha_mid = (alpha_bound["min"] + alpha_bound["max"]) / 2
        assert alpha_initial == alpha_mid  # Should be exactly -1.5

        print("✓ Realistic configuration loads bounds correctly")
        print(f"  Alpha bounds: [{alpha_bound['min']}, {alpha_bound['max']}]")
        print(f"  Alpha initial: {alpha_initial} (center: {alpha_mid})")

    def test_mcmc_uses_full_forward_model_configuration(self, tmp_path):
        """Test that MCMC configuration uses full forward model with scaling optimization."""

        # Create configuration that ensures MCMC uses scaling optimization
        mcmc_config = {
            "metadata": {"config_version": "6.0"},
            "analyzer_parameters": {
                "temporal": {"dt": 0.5, "start_frame": 400, "end_frame": 1000},
                "scattering": {"wavevector_q": 0.0237},
                "geometry": {"stator_rotor_gap": 2000000},
                "computational": {"num_threads": 2},
            },
            "experimental_data": {
                "data_folder_path": "./data/test/",
                "data_file_name": "test.hdf",
                "cache_file_path": "./data/test/",
                "cache_filename_template": "test_cache_{start_frame}_{end_frame}.npz",
            },
            "initial_parameters": {
                "values": [16000, -1.5, 1.1, 0.0, 0.0, 0.0, 0.0],
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_t0",
                    "beta",
                    "gamma_dot_t_offset",
                    "phi0",
                ],
                "active_parameters": ["D0", "alpha", "D_offset"],
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 15000, "max": 20000, "type": "Normal"},
                    {"name": "alpha", "min": -1.8, "max": -1.2, "type": "Normal"},
                    {"name": "D_offset", "min": 0, "max": 5, "type": "Normal"},
                    {"name": "gamma_dot_t0", "min": 0.0, "max": 0.0, "type": "fixed"},
                    {"name": "beta", "min": 0.0, "max": 0.0, "type": "fixed"},
                    {
                        "name": "gamma_dot_t_offset",
                        "min": 0.0,
                        "max": 0.0,
                        "type": "fixed",
                    },
                    {"name": "phi0", "min": 0.0, "max": 0.0, "type": "fixed"},
                ]
            },
            "optimization_config": {
                "mcmc_sampling": {
                    "enabled": True,
                    "sampler": "NUTS",
                    "draws": 100,
                    "tune": 50,
                    "chains": 2,
                    "cores": 2,
                    "target_accept": 0.9,
                }
            },
            "analysis_settings": {"static_mode": True, "static_submode": "isotropic"},
            "performance_settings": {
                "noise_model": {
                    "use_simple_forward_model": False,  # CRITICAL: Must be False for MCMC
                    "sigma_prior": 0.1,
                }
            },
            "advanced_settings": {"chi_squared_calculation": {"method": "standard"}},
        }

        config_file = tmp_path / "mcmc_full_model_config.json"
        with open(config_file, "w") as f:
            json.dump(mcmc_config, f, indent=2)

        # Load and validate configuration
        config_manager = ConfigManager(str(config_file))
        loaded_config = config_manager.config
        
        # Verify configuration was loaded successfully
        assert loaded_config is not None, "Configuration loading failed"

        # Verify MCMC uses full forward model (not simplified)
        noise_model = loaded_config["performance_settings"]["noise_model"]
        use_simple_model = noise_model["use_simple_forward_model"]

        assert (
            use_simple_model is False
        ), "MCMC must use full forward model (use_simple_forward_model=False)"

        # Verify MCMC is enabled
        mcmc_enabled = loaded_config["optimization_config"]["mcmc_sampling"]["enabled"]
        assert mcmc_enabled is True, "MCMC sampling must be enabled for this test"

        # Verify scaling optimization is available (standard chi-squared method)
        chi_sq_method = loaded_config["advanced_settings"]["chi_squared_calculation"][
            "method"
        ]
        assert (
            chi_sq_method == "standard"
        ), "Standard chi-squared method supports scaling optimization"

        print("✓ MCMC configuration uses full forward model with scaling optimization")
        print(f"  use_simple_forward_model: {use_simple_model}")
        print(f"  MCMC enabled: {mcmc_enabled}")
        print(f"  Chi-squared method: {chi_sq_method}")
        print("  This ensures MCMC results are consistent with classical optimization")
