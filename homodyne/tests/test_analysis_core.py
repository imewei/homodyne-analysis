"""
Comprehensive Unit Tests for Analysis Core Module
=================================================

Tests for the main homodyne analysis engine and chi-squared fitting functionality.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch
import tempfile
import json
import os

try:
    from homodyne.analysis.core import HomodyneAnalysisCore
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False


@pytest.mark.skipif(not ANALYSIS_AVAILABLE, reason="Analysis core not available")
class TestHomodyneAnalysisCore:
    """Test suite for HomodyneAnalysisCore functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create temporary config file
        self.config_data = {
            "experimental_data": {
                "q_value": 0.1,
                "contrast": 0.95,
                "offset": 1.0,
                "pixel_size": 172e-6,
                "detector_distance": 8.0,
                "x_ray_energy": 7.35,
                "sample_thickness": 1.0
            },
            "analyzer_parameters": {
                "mode": "laminar_flow",
                "method": "classical",
                "enable_angle_filtering": True,
                "chi_squared_threshold": 2.0,
                "max_iterations": 1000,
                "tolerance": 1e-6,
                "temporal": {
                    "dt": 0.1,
                    "start_frame": 1,
                    "end_frame": 100
                },
                "scattering": {
                    "wavevector_q": 0.005
                },
                "geometry": {
                    "stator_rotor_gap": 200000.0
                }
            },
            "optimization_config": {
                "method": "scipy_minimize",
                "max_iterations": 1000,
                "tolerance": 1e-6,
                "initial_step_size": 0.1
            },
            "parameter_bounds": {
                "D0": [1e-6, 1e-1],
                "alpha": [0.1, 2.0],
                "D_offset": [1e-8, 1e-3],
                "gamma0": [1e-4, 1.0],
                "beta": [0.1, 2.0],
                "gamma_offset": [1e-6, 1e-1],
                "phi0": [-180, 180]
            },
            "initial_guesses": {
                "D0": 1e-3,
                "alpha": 0.9,
                "D_offset": 1e-4,
                "gamma0": 0.01,
                "beta": 0.8,
                "gamma_offset": 0.001,
                "phi0": 0.0
            }
        }

        # Create mock data
        self.mock_angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        self.mock_t1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.mock_t2 = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

        # Create synthetic correlation data
        np.random.seed(42)  # For reproducible tests
        n_angles = len(self.mock_angles)
        n_times = len(self.mock_t1)
        self.mock_c2_data = 1.0 + 0.9 * np.random.exponential(0.1, (n_angles, n_times, n_times))

        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.config_data, self.temp_config, indent=2)
        self.temp_config.close()

    def teardown_method(self):
        """Cleanup test fixtures."""
        if hasattr(self, 'temp_config'):
            os.unlink(self.temp_config.name)

    def test_initialization_with_config_file(self):
        """Test initialization with configuration file."""
        analyzer = HomodyneAnalysisCore(config_path=self.temp_config.name)

        assert analyzer is not None
        assert hasattr(analyzer, 'config')
        assert analyzer.config['experimental_data']['q_value'] == 0.1

    def test_initialization_with_config_dict(self):
        """Test initialization with configuration dictionary."""
        analyzer = HomodyneAnalysisCore(config_override=self.config_data)

        assert analyzer is not None
        assert analyzer.config['experimental_data']['contrast'] == 0.95

    def test_parameter_extraction(self):
        """Test parameter extraction from configuration."""
        analyzer = HomodyneAnalysisCore(config_override=self.config_data)

        # Test experimental parameters
        q_value = analyzer._get_experimental_parameter('q_value')
        assert q_value == 0.1

        contrast = analyzer._get_experimental_parameter('contrast')
        assert contrast == 0.95

    def test_bounds_validation(self):
        """Test parameter bounds validation."""
        analyzer = HomodyneAnalysisCore(config_override=self.config_data)

        # Test valid parameters
        valid_params = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]
        is_valid = analyzer._validate_parameters(valid_params)
        assert is_valid

        # Test invalid parameters (D0 too large)
        invalid_params = [1.0, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]  # D0 = 1.0 > upper bound
        is_valid = analyzer._validate_parameters(invalid_params)
        assert not is_valid

    def test_mode_detection(self):
        """Test analysis mode detection."""
        # Test laminar flow mode
        analyzer = HomodyneAnalysisCore(config_override=self.config_data)
        assert analyzer._get_analysis_mode() == "laminar_flow"

        # Test static mode
        static_config = self.config_data.copy()
        static_config['analyzer_parameters']['mode'] = 'static_isotropic'
        analyzer_static = HomodyneAnalysisCore(config=static_config)
        assert analyzer_static._get_analysis_mode() == "static_isotropic"

    def test_chi_squared_calculation(self):
        """Test chi-squared calculation."""
        analyzer = HomodyneAnalysisCore(config_override=self.config_data)

        # Mock the correlation function calculation
        with patch.object(analyzer, '_calculate_theoretical_correlation') as mock_calc:
            mock_calc.return_value = np.ones_like(self.mock_c2_data)

            params = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]
            chi_squared = analyzer._calculate_chi_squared(params, self.mock_c2_data)

            assert isinstance(chi_squared, (float, np.floating))
            assert chi_squared >= 0.0

    def test_theoretical_correlation_calculation(self):
        """Test theoretical correlation function calculation."""
        analyzer = HomodyneAnalysisCore(config_override=self.config_data)

        params = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]

        # Mock the data arrays
        analyzer.angles = self.mock_angles
        analyzer.t1_array = self.mock_t1
        analyzer.t2_array = self.mock_t2

        theoretical = analyzer._calculate_theoretical_correlation(params)

        assert theoretical.shape == (len(self.mock_angles), len(self.mock_t1), len(self.mock_t2))
        assert np.all(theoretical >= 1.0)  # g2 should be >= 1

    def test_angle_filtering(self):
        """Test angle filtering functionality."""
        config_with_filtering = self.config_data.copy()
        config_with_filtering['analyzer_parameters']['enable_angle_filtering'] = True

        analyzer = HomodyneAnalysisCore(config=config_with_filtering)

        # Test that angle filtering is enabled
        assert analyzer._should_apply_angle_filtering()

    def test_fit_execution(self):
        """Test complete fit execution."""
        analyzer = HomodyneAnalysisCore(config_override=self.config_data)

        # Mock optimization method
        with patch.object(analyzer, '_run_optimization') as mock_opt:
            mock_result = Mock()
            mock_result.x = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]
            mock_result.fun = 0.5
            mock_result.success = True
            mock_opt.return_value = mock_result

            result = analyzer.fit(
                c2_data=self.mock_c2_data,
                angles=self.mock_angles,
                t1_array=self.mock_t1,
                t2_array=self.mock_t2
            )

            assert result is not None
            assert 'parameters' in result
            assert 'chi_squared' in result
            assert 'success' in result

    def test_error_handling_invalid_data(self):
        """Test error handling with invalid data."""
        analyzer = HomodyneAnalysisCore(config_override=self.config_data)

        # Test with mismatched array shapes
        invalid_c2 = np.ones((5, 3, 3))  # Wrong number of angles

        with pytest.raises((ValueError, IndexError)):
            analyzer.fit(
                c2_data=invalid_c2,
                angles=self.mock_angles,
                t1_array=self.mock_t1,
                t2_array=self.mock_t2
            )

    def test_parameter_scaling(self):
        """Test parameter scaling for optimization."""
        analyzer = HomodyneAnalysisCore(config_override=self.config_data)

        params = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]

        # Test scaling
        scaled = analyzer._scale_parameters(params)
        assert len(scaled) == len(params)

        # Test inverse scaling
        unscaled = analyzer._unscale_parameters(scaled)
        np.testing.assert_allclose(unscaled, params, rtol=1e-10)

    def test_result_validation(self):
        """Test result validation and post-processing."""
        analyzer = HomodyneAnalysisCore(config_override=self.config_data)

        # Mock optimization result
        mock_result = Mock()
        mock_result.x = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]
        mock_result.fun = 0.5
        mock_result.success = True

        processed = analyzer._process_optimization_result(mock_result)

        assert 'parameters' in processed
        assert 'chi_squared' in processed
        assert 'success' in processed
        assert processed['success'] == True

    def test_static_mode_parameter_handling(self):
        """Test parameter handling in static mode."""
        static_config = self.config_data.copy()
        # Ensure analysis_parameters key exists
        if 'analysis_parameters' not in static_config:
            static_config['analysis_parameters'] = {}
        static_config['analysis_parameters']['mode'] = 'static_isotropic'

        analyzer = HomodyneAnalysisCore(config=static_config)

        # In static mode, flow parameters should be fixed
        params = analyzer._get_initial_parameters()

        # Check that flow parameters are zeroed
        if len(params) >= 7:  # Full parameter set
            # gamma0, beta, gamma_offset should be minimal or zero
            assert params[3] <= 1e-10  # gamma0
            assert params[5] <= 1e-10  # gamma_offset

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test with invalid configuration
        invalid_config = {
            "experimental_parameters": {
                "q_value": -0.1,  # Invalid negative q
                "contrast": 1.5,  # Invalid contrast > 1
            }
        }

        with pytest.raises((ValueError, KeyError)):
            HomodyneAnalysisCore(config=invalid_config)

    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        analyzer = HomodyneAnalysisCore(config_override=self.config_data)

        # Create larger mock data
        large_angles = np.linspace(0, 2*np.pi, 32, endpoint=False)
        large_t1 = np.linspace(0.1, 10.0, 20)
        large_t2 = np.linspace(0.2, 10.1, 20)
        large_c2 = np.random.exponential(0.1, (32, 20, 20)) + 1.0

        # Should handle large data without memory errors
        params = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]

        with patch.object(analyzer, '_run_optimization') as mock_opt:
            mock_result = Mock()
            mock_result.x = params
            mock_result.fun = 0.5
            mock_result.success = True
            mock_opt.return_value = mock_result

            result = analyzer.fit(
                c2_data=large_c2,
                angles=large_angles,
                t1_array=large_t1,
                t2_array=large_t2
            )

            assert result is not None

    def test_numerical_stability(self):
        """Test numerical stability with edge case parameters."""
        analyzer = HomodyneAnalysisCore(config_override=self.config_data)

        # Test with extreme but valid parameters
        extreme_params = [1e-6, 0.1, 1e-8, 1e-4, 0.1, 1e-6, 0.0]

        analyzer.angles = self.mock_angles
        analyzer.t1_array = self.mock_t1
        analyzer.t2_array = self.mock_t2

        # Should not crash or produce NaN/Inf values
        theoretical = analyzer._calculate_theoretical_correlation(extreme_params)

        assert np.all(np.isfinite(theoretical))
        assert np.all(theoretical >= 1.0)

    def test_convergence_criteria(self):
        """Test convergence criteria handling."""
        analyzer = HomodyneAnalysisCore(config_override=self.config_data)

        # Test with tight tolerance
        tight_config = self.config_data.copy()
        # Ensure analysis_parameters key exists
        if 'analysis_parameters' not in tight_config:
            tight_config['analysis_parameters'] = {}
        tight_config['analysis_parameters']['tolerance'] = 1e-12

        analyzer_tight = HomodyneAnalysisCore(config=tight_config)

        # Verify tolerance is set correctly
        assert analyzer_tight.config['analysis_parameters']['tolerance'] == 1e-12


class TestAnalysisCoreIntegration:
    """Integration tests for analysis core with other modules."""

    def setup_method(self):
        """Setup integration test fixtures."""
        self.config_data = {
            "experimental_data": {
                "data_folder_path": "./data/test/",
                "data_file_name": "test_data.hdf",
                "phi_angles_path": "./data/phi_angles/",
                "phi_angles_file": "phi_list.txt",
                "exchange_key": "exchange"
            },
            "analyzer_parameters": {
                "temporal": {
                    "dt": 0.1,
                    "start_frame": 100,
                    "end_frame": 1000
                },
                "scattering": {
                    "wavevector_q": 0.05
                },
                "geometry": {
                    "stator_rotor_gap": 2000000
                },
                "computational": {
                    "num_threads": "auto",
                    "auto_detect_cores": True,
                    "max_threads_limit": 8
                }
            },
            "optimization_config": {
                "angle_filtering": {
                    "enabled": True
                },
                "classical_optimization": {
                    "methods": ["Nelder-Mead"],
                    "method_options": {
                        "Nelder-Mead": {
                            "maxiter": 100,
                            "xatol": 1e-6,
                            "fatol": 1e-6,
                            "adaptive": True
                        }
                    }
                },
                "scaling_parameters": {
                    "contrast": {
                        "min": 1e-4,
                        "max": 0.5,
                        "prior_mu": 0.9,
                        "prior_sigma": 0.01,
                        "type": "TruncatedNormal"
                    },
                    "offset": {
                        "min": 1.0,
                        "max": 1.5,
                        "prior_mu": 1.0,
                        "prior_sigma": 0.01,
                        "type": "TruncatedNormal"
                    }
                }
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 1e-5, "max": 1e-2, "type": "TruncatedNormal", "prior_mu": 1e-3, "prior_sigma": 1e-4, "unit": "Å²/s"},
                    {"name": "alpha", "min": 0.5, "max": 1.5, "type": "Normal", "prior_mu": 0.9, "prior_sigma": 0.1, "unit": "dimensionless"},
                    {"name": "D_offset", "min": 1e-7, "max": 1e-4, "type": "Normal", "prior_mu": 1e-5, "prior_sigma": 1e-6, "unit": "Å²/s"},
                    {"name": "gamma0", "min": 1e-3, "max": 0.1, "type": "TruncatedNormal", "prior_mu": 0.01, "prior_sigma": 0.001, "unit": "s⁻¹"},
                    {"name": "beta", "min": 0.5, "max": 1.5, "type": "Normal", "prior_mu": 0.8, "prior_sigma": 0.1, "unit": "dimensionless"},
                    {"name": "gamma_offset", "min": 1e-5, "max": 1e-2, "type": "Normal", "prior_mu": 0.001, "prior_sigma": 0.0001, "unit": "s⁻¹"},
                    {"name": "phi0", "min": -90, "max": 90, "type": "Normal", "prior_mu": 0.0, "prior_sigma": 5.0, "unit": "degrees"}
                ]
            },
            "analysis_settings": {
                "static_mode": False,
                "static_submode": None
            }
        }

    @pytest.mark.skipif(not ANALYSIS_AVAILABLE, reason="Analysis core not available")
    def test_end_to_end_analysis(self):
        """Test complete end-to-end analysis workflow."""
        # Create synthetic data with known parameters
        true_params = [2e-3, 0.8, 5e-5, 0.02, 0.7, 0.002, 10.0]

        analyzer = HomodyneAnalysisCore(config_override=self.config_data)

        # Generate synthetic correlation data
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        t1_array = np.array([0.5, 1.0, 1.5, 2.0])
        t2_array = np.array([1.0, 1.5, 2.0, 2.5])

        # Mock the fit to return known parameters
        with patch.object(analyzer, '_run_optimization') as mock_opt:
            mock_result = Mock()
            mock_result.x = true_params
            mock_result.fun = 0.1
            mock_result.success = True
            mock_opt.return_value = mock_result

            # Generate mock data
            np.random.seed(123)
            c2_data = 1.0 + 0.9 * np.random.exponential(0.05, (8, 4, 4))

            result = analyzer.fit(
                c2_data=c2_data,
                angles=angles,
                t1_array=t1_array,
                t2_array=t2_array
            )

            assert result['success']
            assert len(result['parameters']) == len(true_params)
            assert result['chi_squared'] >= 0.0

    @pytest.mark.skipif(not ANALYSIS_AVAILABLE, reason="Analysis core not available")
    def test_different_analysis_modes(self):
        """Test different analysis modes."""
        modes = ['static_isotropic', 'static_anisotropic', 'laminar_flow']

        for mode in modes:
            config = self.config_data.copy()
            # Ensure analysis_parameters key exists
            if 'analysis_parameters' not in config:
                config['analysis_parameters'] = {}
            config['analysis_parameters']['mode'] = mode

            analyzer = HomodyneAnalysisCore(config=config)
            assert analyzer._get_analysis_mode() == mode

            # Test parameter count for each mode
            params = analyzer._get_initial_parameters()
            if mode == 'static_isotropic':
                assert len(params) >= 3  # At least D0, alpha, D_offset
            elif mode == 'laminar_flow':
                assert len(params) == 7  # All parameters

    @pytest.mark.skipif(not ANALYSIS_AVAILABLE, reason="Analysis core not available")
    def test_robustness_to_noise(self):
        """Test robustness to noisy data."""
        analyzer = HomodyneAnalysisCore(config_override=self.config_data)

        # Create data with varying noise levels
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        t1_array = np.array([1.0, 2.0, 3.0])
        t2_array = np.array([1.5, 2.5, 3.5])

        # Base correlation data
        base_data = 1.0 + 0.9 * np.exp(-0.1 * np.random.random((6, 3, 3)))

        noise_levels = [0.01, 0.05, 0.1]

        for noise_level in noise_levels:
            np.random.seed(456)
            noisy_data = base_data + noise_level * np.random.randn(*base_data.shape)
            noisy_data = np.maximum(noisy_data, 1.0)  # Ensure g2 >= 1

            with patch.object(analyzer, '_run_optimization') as mock_opt:
                mock_result = Mock()
                mock_result.x = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]
                mock_result.fun = noise_level * 10  # Higher chi-squared for noisier data
                mock_result.success = True
                mock_opt.return_value = mock_result

                result = analyzer.fit(
                    c2_data=noisy_data,
                    angles=angles,
                    t1_array=t1_array,
                    t2_array=t2_array
                )

                # Should still converge
                assert result['success']
                assert np.isfinite(result['chi_squared'])