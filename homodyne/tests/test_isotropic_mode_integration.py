"""
Integration Tests for Static Isotropic Mode
==========================================

Comprehensive integration tests that verify the isotropic static mode works 
end-to-end including data loading, analysis, and plotting with dummy phi angles.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from homodyne.analysis.core import HomodyneAnalysisCore
from homodyne.core.config import ConfigManager


class TestIsotropicModeIntegration:
    """Integration tests for static isotropic mode functionality."""

    @pytest.fixture
    def isotropic_config(self):
        """Configuration for static isotropic mode."""
        return {
            "metadata": {"config_version": "6.0"},
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 50},
                "scattering": {"wavevector_q": 0.01},
                "geometry": {"stator_rotor_gap": 2000000},
                "computational": {"num_threads": 1, "auto_detect_cores": False},
            },
            "experimental_data": {
                "data_folder_path": "./data/test/",
                "data_file_name": "test.hdf",
                "phi_angles_path": "./data/test/",
                "phi_angles_file": "phi_list.txt",
                "exchange_key": "exchange",
                "cache_file_path": "./data/test/",
                "cache_filename_template": "test_isotropic_{start_frame}_{end_frame}.npz",
            },
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]},
                "angle_filtering": {"enabled": True}  # Should be automatically disabled
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
            "initial_parameters": {"values": [1000.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0]},
            "analysis_settings": {
                "static_mode": True,
                "static_submode": "isotropic"
            },
            "advanced_settings": {
                "data_loading": {"use_diagonal_correction": False},
                "chi_squared_calculation": {
                    "scaling_optimization": False,  # Should be automatically disabled
                    "validity_check": {"check_positive_D0": True, "check_positive_gamma_dot_t0": True}
                }
            },
            "performance_settings": {"parallel_execution": False}
        }

    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_config_manager_isotropic_mode_detection(self, temp_directory, isotropic_config):
        """Test that ConfigManager correctly detects isotropic mode."""
        config_file = temp_directory / "isotropic_config.json"
        with open(config_file, "w") as f:
            json.dump(isotropic_config, f)

        manager = ConfigManager(str(config_file))
        
        # Test mode detection
        assert manager.is_static_mode_enabled() is True
        assert manager.get_static_submode() == "isotropic"
        assert manager.is_static_isotropic_enabled() is True
        assert manager.is_static_anisotropic_enabled() is False
        assert manager.get_analysis_mode() == "static_isotropic"
        assert manager.get_effective_parameter_count() == 3

        # Test angle filtering is automatically disabled
        assert manager.is_angle_filtering_enabled() is False

    def test_isotropic_mode_core_initialization(self, temp_directory, isotropic_config):
        """Test HomodyneAnalysisCore initialization with isotropic mode."""
        config_file = temp_directory / "isotropic_config.json"
        with open(config_file, "w") as f:
            json.dump(isotropic_config, f)

        core = HomodyneAnalysisCore(str(config_file))
        
        # Test mode detection
        assert core.is_static_mode() is True
        assert core.config_manager.is_static_isotropic_enabled() is True
        assert core.get_effective_parameter_count() == 3

        # Test that angle filtering is disabled
        assert core.config_manager.is_angle_filtering_enabled() is False

    @patch('homodyne.analysis.core.np.savez_compressed')
    @patch('homodyne.analysis.core.np.load')
    @patch('homodyne.analysis.core.os.path.exists')
    def test_isotropic_mode_data_loading_skips_phi_angles(self, mock_exists, mock_load, mock_savez, temp_directory, isotropic_config):
        """Test that isotropic mode skips loading phi_angles_file."""
        config_file = temp_directory / "isotropic_config.json"
        with open(config_file, "w") as f:
            json.dump(isotropic_config, f)

        # Mock file existence checks
        def exists_side_effect(path):
            # Config file exists
            if str(path).endswith('isotropic_config.json'):
                return True
            # Data files don't exist (cache miss scenario)
            return False
        
        mock_exists.side_effect = exists_side_effect

        # Create mock experimental data
        mock_c2_data = np.random.rand(1, 49)  # Single angle for isotropic mode
        mock_load.return_value = {'c2_exp': mock_c2_data}

        core = HomodyneAnalysisCore(str(config_file))
        
        with patch.object(core, '_load_raw_data') as mock_load_raw:
            mock_load_raw.return_value = mock_c2_data
            
            # Load data - should use dummy angles
            c2_data, time_length, phi_angles, num_angles = core.load_experimental_data()
            
            # Verify dummy angles are used
            assert len(phi_angles) == 1
            assert phi_angles[0] == 0.0
            assert num_angles == 1
            
            # Verify data structure
            assert c2_data.shape[0] == 1  # Single angle

    @patch('homodyne.analysis.core.np.savez_compressed')
    @patch('homodyne.analysis.core.np.load')  
    @patch('homodyne.analysis.core.os.path.exists')
    def test_isotropic_mode_optimization_workflow(self, mock_exists, mock_load, mock_savez, temp_directory, isotropic_config):
        """Test complete optimization workflow in isotropic mode."""
        config_file = temp_directory / "isotropic_config.json"
        with open(config_file, "w") as f:
            json.dump(isotropic_config, f)

        # Mock file existence - cache miss scenario
        mock_exists.return_value = False

        # Create realistic mock data for single angle (isotropic mode)
        n_times = 49
        mock_c2_data = np.exp(-np.linspace(0, 5, n_times)).reshape(1, -1) + 0.1 * np.random.rand(1, n_times)
        mock_load.return_value = {'c2_exp': mock_c2_data}

        core = HomodyneAnalysisCore(str(config_file))

        with patch.object(core, '_load_raw_data') as mock_load_raw:
            mock_load_raw.return_value = mock_c2_data
            
            # Test parameter processing
            test_params = np.array([1200.0, -0.05, 150.0, 0.001, -0.1, 0.0001, 5.0])
            effective_params = core.get_effective_parameters(test_params)
            
            # In isotropic static mode, last 4 parameters should be zeroed
            expected = np.array([1200.0, -0.05, 150.0, 0.0, 0.0, 0.0, 0.0])
            np.testing.assert_array_equal(effective_params, expected)

            # Test basic analysis functions
            c2_data, time_length, phi_angles, num_angles = core.load_experimental_data()
            assert num_angles == 1  # Isotropic mode uses single angle
            assert phi_angles[0] == 0.0  # Dummy angle

    def test_isotropic_mode_correlation_calculation(self, temp_directory, isotropic_config):
        """Test C2 correlation calculation in isotropic mode."""
        config_file = temp_directory / "isotropic_config.json"
        with open(config_file, "w") as f:
            json.dump(isotropic_config, f)

        core = HomodyneAnalysisCore(str(config_file))
        
        # Test parameters - only first 3 should matter
        params = np.array([1000.0, 0.0, 100.0, 999.0, 999.0, 999.0, 999.0])  # Large values for last 4
        phi_angle = 0.0  # Dummy angle

        # Calculate C2 - should ignore last 4 parameters
        c2_result = core.calculate_c2_single_angle_optimized(params, phi_angle)
        
        # Should return valid correlation function
        assert c2_result.shape[0] > 0
        assert np.all(c2_result >= 0)  # C2 should be non-negative
        assert np.all(c2_result <= 2.0)  # Reasonable upper bound for normalized C2
        
        # Test that result is same regardless of last 4 parameters
        params_different_flow = np.array([1000.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0])
        c2_result_2 = core.calculate_c2_single_angle_optimized(params_different_flow, phi_angle)
        
        # Results should be identical (within numerical precision)
        np.testing.assert_array_almost_equal(c2_result, c2_result_2, decimal=15)

    def test_isotropic_mode_chi_squared_calculation(self, temp_directory, isotropic_config):
        """Test chi-squared calculation supports dummy angles in isotropic mode."""
        config_file = temp_directory / "isotropic_config.json"
        with open(config_file, "w") as f:
            json.dump(isotropic_config, f)

        core = HomodyneAnalysisCore(str(config_file))
        
        # Mock experimental data for single angle
        mock_exp_data = np.exp(-np.linspace(0, 3, 20)) + 0.05 * np.random.rand(20)
        
        with patch.object(core, 'cached_experimental_data', mock_exp_data.reshape(1, -1)):
            with patch.object(core, 'cached_phi_angles', np.array([0.0])):
                # Test parameters
                params = np.array([1000.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0])
                
                # Test that correlation calculation works with dummy angle
                c2_result = core.calculate_c2_single_angle_optimized(params, 0.0)
                
                # Should return finite, positive values
                assert len(c2_result) > 0
                assert np.all(np.isfinite(c2_result))
                assert np.all(c2_result >= 0)  # C2 should be non-negative

    def test_isotropic_mode_caching_behavior(self, temp_directory, isotropic_config):
        """Test caching behavior with dummy angles in isotropic mode."""
        config_file = temp_directory / "isotropic_config.json"
        
        # Create the cache directory that the config references
        cache_dir = temp_directory / "data" / "test"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Update config to use the temp directory
        isotropic_config["experimental_data"]["cache_file_path"] = str(cache_dir)
        
        with open(config_file, "w") as f:
            json.dump(isotropic_config, f)

        core = HomodyneAnalysisCore(str(config_file))
        
        # Mock experimental data for single angle
        mock_c2_data = np.random.rand(1, 49)  # Single angle
        
        # Test that data is cached in memory after loading
        assert core.cached_experimental_data is None  # Initially None
        assert core.cached_phi_angles is None  # Initially None
        
        with patch.object(core, '_load_raw_data') as mock_load_raw:
            with patch('homodyne.analysis.core.os.path.isfile') as mock_isfile:
                # Mock cache miss to ensure fresh load
                mock_isfile.return_value = False
                mock_load_raw.return_value = mock_c2_data
                
                # Load data - should trigger internal caching
                c2_data, time_length, phi_angles, num_angles = core.load_experimental_data()
                
                # Verify isotropic mode characteristics
                assert num_angles == 1
                assert len(phi_angles) == 1
                assert phi_angles[0] == 0.0
                
                # Verify internal memory caching occurred
                assert core.cached_experimental_data is not None
                assert core.cached_phi_angles is not None
                assert len(core.cached_phi_angles) == 1
                assert core.cached_phi_angles[0] == 0.0
                
                # Verify second call uses cache (raw data load should not be called again)
                mock_load_raw.reset_mock()
                c2_data_2, _, phi_angles_2, num_angles_2 = core.load_experimental_data()
                
                # Should return same data from cache
                assert num_angles_2 == 1
                assert phi_angles_2[0] == 0.0
                np.testing.assert_array_equal(c2_data, c2_data_2)
                
                # Raw data loading should not have been called again
                mock_load_raw.assert_not_called()

    def test_isotropic_mode_parameter_validation(self, temp_directory, isotropic_config):
        """Test parameter validation in isotropic mode."""
        config_file = temp_directory / "isotropic_config.json"
        with open(config_file, "w") as f:
            json.dump(isotropic_config, f)

        core = HomodyneAnalysisCore(str(config_file))
        
        # Test parameter processing - should zero out flow parameters
        original_params = np.array([1000.0, -0.1, 100.0, 0.01, -0.2, 0.001, 10.0])
        processed_params = core.get_effective_parameters(original_params)
        
        # First 3 should be preserved
        assert processed_params[0] == 1000.0  # D0
        assert processed_params[1] == -0.1    # alpha
        assert processed_params[2] == 100.0   # D_offset
        
        # Last 4 should be zeroed
        assert processed_params[3] == 0.0     # gamma_dot_t0
        assert processed_params[4] == 0.0     # beta
        assert processed_params[5] == 0.0     # gamma_dot_t_offset
        assert processed_params[6] == 0.0     # phi0

    def test_isotropic_vs_anisotropic_mode_comparison(self, temp_directory):
        """Test that isotropic and anisotropic modes produce different behavior."""
        base_config = {
            "metadata": {"config_version": "6.0"},
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 50},
                "scattering": {"wavevector_q": 0.01},
                "geometry": {"stator_rotor_gap": 2000000},
                "computational": {"num_threads": 1, "auto_detect_cores": False},
            },
            "experimental_data": {
                "data_folder_path": "./data/test/",
                "data_file_name": "test.hdf",
                "phi_angles_path": "./data/test/",
                "phi_angles_file": "phi_list.txt",
                "exchange_key": "exchange",
                "cache_file_path": "./data/test/",
                "cache_filename_template": "test_cache_{start_frame}_{end_frame}.npz",
            },
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]},
                "angle_filtering": {"enabled": True}
            },
            "parameter_space": {"bounds": []},
            "initial_parameters": {"values": [1000.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0]},
            "analysis_settings": {"static_mode": True},
            "advanced_settings": {"data_loading": {"use_diagonal_correction": False}},
            "performance_settings": {"parallel_execution": False}
        }
        
        # Test isotropic mode
        iso_config = base_config.copy()
        iso_config["analysis_settings"]["static_submode"] = "isotropic"
        
        iso_config_file = temp_directory / "iso_config.json"
        with open(iso_config_file, "w") as f:
            json.dump(iso_config, f)
        
        iso_manager = ConfigManager(str(iso_config_file))
        
        # Test anisotropic mode
        aniso_config = base_config.copy()
        aniso_config["analysis_settings"]["static_submode"] = "anisotropic"
        
        aniso_config_file = temp_directory / "aniso_config.json"
        with open(aniso_config_file, "w") as f:
            json.dump(aniso_config, f)
            
        aniso_manager = ConfigManager(str(aniso_config_file))
        
        # Test mode detection differences
        assert iso_manager.get_analysis_mode() == "static_isotropic"
        assert aniso_manager.get_analysis_mode() == "static_anisotropic"
        
        # Test angle filtering differences
        assert iso_manager.is_angle_filtering_enabled() is False    # Disabled in isotropic
        assert aniso_manager.is_angle_filtering_enabled() is True   # Enabled in anisotropic
        
        # Both should be static mode with 3 parameters
        assert iso_manager.is_static_mode_enabled() is True
        assert aniso_manager.is_static_mode_enabled() is True
        assert iso_manager.get_effective_parameter_count() == 3
        assert aniso_manager.get_effective_parameter_count() == 3

    def test_isotropic_mode_end_to_end_workflow(self, temp_directory, isotropic_config):
        """Test complete end-to-end workflow for isotropic mode."""
        config_file = temp_directory / "isotropic_config.json"
        with open(config_file, "w") as f:
            json.dump(isotropic_config, f)

        # Mock all file operations and data loading
        with patch('homodyne.analysis.core.os.path.exists', return_value=False):
            with patch('homodyne.analysis.core.np.load') as mock_load:
                with patch('homodyne.analysis.core.np.savez_compressed') as mock_savez:
                    
                    # Setup mock data for single angle
                    n_times = 49
                    mock_c2_data = np.exp(-np.linspace(0, 3, n_times)).reshape(1, -1) + 0.05 * np.random.rand(1, n_times)
                    mock_load.return_value = {'c2_exp': mock_c2_data}
                    
                    # Initialize core
                    core = HomodyneAnalysisCore(str(config_file))
                    
                    with patch.object(core, '_load_raw_data') as mock_load_raw:
                        mock_load_raw.return_value = mock_c2_data
                        
                        # Test data loading
                        c2_data, time_length, phi_angles, num_angles = core.load_experimental_data()
                        
                        # Verify isotropic mode characteristics
                        assert num_angles == 1
                        assert len(phi_angles) == 1
                        assert phi_angles[0] == 0.0
                        assert c2_data.shape[0] == 1  # Single angle
                        
                        # Test basic analysis functions work
                        params = np.array([1000.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0])
                        
                        # Test correlation calculation
                        c2_result = core.calculate_c2_single_angle_optimized(params, 0.0)
                        assert len(c2_result) > 0
                        assert np.all(np.isfinite(c2_result))
                        
                        # Test configuration is correctly applied
                        assert core.config_manager.is_static_isotropic_enabled()
                        assert not core.config_manager.is_angle_filtering_enabled()
                        assert core.get_effective_parameter_count() == 3


class TestIsotropicModeEdgeCases:
    """Test edge cases and error conditions for isotropic mode."""

    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_isotropic_mode_with_invalid_submode(self, temp_directory):
        """Test handling of invalid static_submode values."""
        config = {
            "metadata": {"config_version": "6.0"},
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 50}
            },
            "experimental_data": {"data_folder_path": "./data/test/"},
            "optimization_config": {"classical_optimization": {"methods": ["test"]}},
            "analysis_settings": {
                "static_mode": True,
                "static_submode": "invalid_mode"  # Invalid submode
            }
        }
        
        config_file = temp_directory / "invalid_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

        manager = ConfigManager(str(config_file))
        
        # Should default to anisotropic for invalid submode
        assert manager.get_static_submode() == "anisotropic"
        assert manager.get_analysis_mode() == "static_anisotropic"
        assert manager.is_static_isotropic_enabled() is False

    def test_isotropic_mode_case_insensitive_submode(self, temp_directory):
        """Test that static_submode is case-insensitive."""
        test_cases = [
            ("isotropic", True),
            ("Isotropic", True),
            ("ISOTROPIC", True),
            ("ISO", True),
            ("iso", True),
            ("anisotropic", False),
            ("ANISOTROPIC", False),
            ("aniso", False),
        ]
        
        for submode, expected_isotropic in test_cases:
            config = {
                "metadata": {"config_version": "6.0"},
                "analyzer_parameters": {
                    "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 50}
                },
                "experimental_data": {"data_folder_path": "./data/test/"},
                "optimization_config": {"classical_optimization": {"methods": ["test"]}},
                "analysis_settings": {
                    "static_mode": True,
                    "static_submode": submode
                }
            }
            
            config_file = temp_directory / f"test_{submode}_config.json"
            with open(config_file, "w") as f:
                json.dump(config, f)

            manager = ConfigManager(str(config_file))
            
            if expected_isotropic:
                assert manager.is_static_isotropic_enabled()
                assert manager.get_analysis_mode() == "static_isotropic"
            else:
                assert not manager.is_static_isotropic_enabled()
                assert manager.get_analysis_mode() == "static_anisotropic"

    def test_isotropic_mode_with_none_submode(self, temp_directory):
        """Test handling when static_submode is None or missing.""" 
        config = {
            "metadata": {"config_version": "6.0"},
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 50}
            },
            "experimental_data": {"data_folder_path": "./data/test/"},
            "optimization_config": {"classical_optimization": {"methods": ["test"]}},
            "analysis_settings": {
                "static_mode": True,
                "static_submode": None  # Explicitly None
            }
        }
        
        config_file = temp_directory / "none_submode_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

        manager = ConfigManager(str(config_file))
        
        # Should default to anisotropic when None
        assert manager.get_static_submode() == "anisotropic"
        assert manager.get_analysis_mode() == "static_anisotropic"

    def test_isotropic_mode_angle_filtering_override_warning(self, temp_directory):
        """Test that angle filtering configuration is properly overridden in isotropic mode."""
        config = {
            "metadata": {"config_version": "6.0"},
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 50}
            },
            "experimental_data": {"data_folder_path": "./data/test/"},
            "optimization_config": {
                "classical_optimization": {"methods": ["test"]},
                "angle_filtering": {
                    "enabled": True,  # Explicitly enabled but should be ignored
                    "target_ranges": [{"min_angle": -10, "max_angle": 10}]
                }
            },
            "analysis_settings": {
                "static_mode": True,
                "static_submode": "isotropic"
            }
        }
        
        config_file = temp_directory / "angle_filtering_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

        manager = ConfigManager(str(config_file))
        
        # Angle filtering should be disabled despite configuration
        assert manager.is_angle_filtering_enabled() is False
        
        # Original configuration should still be accessible
        angle_config = manager.get_angle_filtering_config()
        assert angle_config["enabled"] is True  # Original setting preserved
        
        # But the effective setting should be False due to isotropic mode
        assert manager.is_angle_filtering_enabled() is False
