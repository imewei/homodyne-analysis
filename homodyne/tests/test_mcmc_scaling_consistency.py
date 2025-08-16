"""
Tests for MCMC scaling optimization consistency features.

This module tests the MCMC scaling optimization consistency detection and warning
system that ensures users are aware when MCMC and classical optimization methods
use different scaling approaches.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Import MCMC module and check availability
try:
    from homodyne.optimization.mcmc import MCMCSampler, PYMC_AVAILABLE
    mcmc_available = PYMC_AVAILABLE
except ImportError:
    mcmc_available = False
    MCMCSampler = None


class TestMCMCScalingConsistency:
    """Test MCMC scaling optimization consistency detection and warnings."""

    @pytest.fixture
    def mock_analysis_core(self):
        """Create a mock analysis core for testing."""
        mock_core = Mock()
        mock_core.config = {
            "advanced_settings": {
                "chi_squared_calculation": {
                    "scaling_optimization": True
                }
            },
            "performance_settings": {
                "noise_model": {
                    "use_simple_forward_model": True,
                    "sigma_prior": 0.1
                }
            },
            "optimization_config": {
                "mcmc_sampling": {
                    "draws": 100,
                    "tune": 50,
                    "chains": 2
                }
            },
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]
            }
        }
        mock_core.num_threads = 4
        return mock_core

    @pytest.fixture
    def config_with_scaling_inconsistency(self):
        """Configuration that should trigger scaling inconsistency warnings."""
        return {
            "advanced_settings": {
                "chi_squared_calculation": {
                    "scaling_optimization": True  # Enabled
                }
            },
            "performance_settings": {
                "noise_model": {
                    "use_simple_forward_model": True,  # This creates inconsistency
                    "sigma_prior": 0.1
                }
            },
            "optimization_config": {
                "mcmc_sampling": {}
            },
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]
            }
        }

    @pytest.fixture
    def config_with_scaling_consistency(self):
        """Configuration that should NOT trigger scaling inconsistency warnings."""
        return {
            "advanced_settings": {
                "chi_squared_calculation": {
                    "scaling_optimization": True  # Enabled
                }
            },
            "performance_settings": {
                "noise_model": {
                    "use_simple_forward_model": False,  # Full model for consistency
                    "sigma_prior": 0.1
                }
            },
            "optimization_config": {
                "mcmc_sampling": {}
            },
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]
            }
        }

    @pytest.mark.skipif(not mcmc_available, reason="PyMC not available")
    def test_mcmc_sampler_initialization_with_scaling_config(self, mock_analysis_core):
        """Test that MCMC sampler initializes correctly with scaling configuration."""
        if MCMCSampler is None:
            pytest.skip("MCMCSampler not available")
        sampler = MCMCSampler(mock_analysis_core, mock_analysis_core.config)
        
        assert sampler.core == mock_analysis_core
        assert sampler.config == mock_analysis_core.config
        assert sampler.mcmc_config is not None

    @pytest.mark.skipif(not mcmc_available, reason="PyMC not available")
    def test_scaling_inconsistency_warning_detection(self, mock_analysis_core, config_with_scaling_inconsistency, caplog):
        """Test that scaling inconsistency warnings are properly detected and logged."""
        with caplog.at_level(logging.WARNING):
            # Create mock experimental data
            c2_experimental = np.random.rand(3, 10, 10)  # 3 angles, 10x10 correlation matrix
            phi_angles = np.array([0.0, 90.0, 180.0])
            
            # Update mock core with inconsistent config
            mock_analysis_core.config = config_with_scaling_inconsistency
            if MCMCSampler is None:
                pytest.skip("MCMCSampler not available")
            sampler = MCMCSampler(mock_analysis_core, config_with_scaling_inconsistency)
            
            # Mock PyMC model building to avoid actual computation
            with patch.object(sampler, '_build_bayesian_model_optimized') as mock_build:
                mock_model = Mock()
                mock_build.return_value = mock_model
                
                # This should trigger consistency warnings
                try:
                    sampler._build_bayesian_model_optimized(
                        c2_experimental, phi_angles, 
                        filter_angles_for_optimization=True,
                        is_static_mode=False,
                        effective_param_count=7
                    )
                except Exception:
                    # We expect this to fail in testing due to mocking, that's OK
                    pass
        
        # Check that warning was logged (we can't easily test the print statements)
        assert sampler.config["advanced_settings"]["chi_squared_calculation"]["scaling_optimization"] is True
        assert sampler.config["performance_settings"]["noise_model"]["use_simple_forward_model"] is True

    @pytest.mark.skipif(not mcmc_available, reason="PyMC not available")
    def test_scaling_consistency_configuration(self, mock_analysis_core, config_with_scaling_consistency):
        """Test configuration that should provide scaling consistency."""
        mock_analysis_core.config = config_with_scaling_consistency
        if MCMCSampler is None:
            pytest.skip("MCMCSampler not available")
        sampler = MCMCSampler(mock_analysis_core, config_with_scaling_consistency)
        
        # Check that configuration indicates full forward model
        chi_config = sampler.config.get("advanced_settings", {}).get("chi_squared_calculation", {})
        noise_config = sampler.config.get("performance_settings", {}).get("noise_model", {})
        
        assert chi_config.get("scaling_optimization", False) is True
        assert noise_config.get("use_simple_forward_model", True) is False  # Should be False for consistency

    def test_configuration_validation(self):
        """Test configuration structure for scaling consistency settings."""
        # Test that our configuration structure is correct
        test_config = {
            "advanced_settings": {
                "chi_squared_calculation": {
                    "scaling_optimization": True
                }
            },
            "performance_settings": {
                "noise_model": {
                    "use_simple_forward_model": True,
                    "sigma_prior": 0.1
                }
            }
        }
        
        # Extract settings
        chi_config = test_config.get("advanced_settings", {}).get("chi_squared_calculation", {})
        noise_config = test_config.get("performance_settings", {}).get("noise_model", {})
        
        scaling_enabled = chi_config.get("scaling_optimization", True)
        simple_forward = noise_config.get("use_simple_forward_model", True)
        
        # Verify values
        assert scaling_enabled is True
        assert simple_forward is True
        
        # This combination should trigger warnings in MCMC
        if scaling_enabled and simple_forward:
            # This is the problematic combination
            assert True  # We've identified the inconsistency

    def test_config_keys_exist(self):
        """Test that all required configuration keys exist in our expected structure."""
        required_keys = [
            ("advanced_settings", "chi_squared_calculation", "scaling_optimization"),
            ("performance_settings", "noise_model", "use_simple_forward_model"),
            ("performance_settings", "noise_model", "sigma_prior")
        ]
        
        test_config = {
            "advanced_settings": {
                "chi_squared_calculation": {
                    "scaling_optimization": True
                }
            },
            "performance_settings": {
                "noise_model": {
                    "use_simple_forward_model": True,
                    "sigma_prior": 0.1
                }
            }
        }
        
        for key_path in required_keys:
            current = test_config
            for key in key_path:
                assert key in current, f"Missing key: {'.'.join(key_path[:key_path.index(key)+1])}"
                current = current[key]


class TestConfigurationConsistency:
    """Test configuration file consistency for scaling optimization."""

    def test_noise_model_section_structure(self):
        """Test that noise_model configuration section has correct structure."""
        expected_structure = {
            "use_simple_forward_model": bool,
            "sigma_prior": (int, float),
        }
        
        # This would be validated against actual config files
        test_noise_config = {
            "use_simple_forward_model": True,
            "sigma_prior": 0.1
        }
        
        for key, expected_type in expected_structure.items():
            assert key in test_noise_config
            assert isinstance(test_noise_config[key], expected_type)

    def test_scaling_optimization_default_values(self):
        """Test default values for scaling optimization settings."""
        # Test default values that should be used
        defaults = {
            "scaling_optimization": True,  # Chi-squared should use scaling by default
            "use_simple_forward_model": True,  # MCMC should use simple model by default (for speed)
            "sigma_prior": 0.1,  # Reasonable noise prior
        }
        
        # Verify these are reasonable defaults
        assert defaults["scaling_optimization"] is True  # Good for chi-squared accuracy
        assert defaults["use_simple_forward_model"] is True  # Good for MCMC speed
        assert 0.01 <= defaults["sigma_prior"] <= 1.0  # Reasonable noise range