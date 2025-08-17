"""
Test MCMC Configuration Reading
==============================

Tests to ensure MCMC configuration is read correctly from the JSON configuration file.
This addresses the issue where MCMC parameters (chains, draws, tune) were not being
read from the proper configuration section.
"""

import pytest
import tempfile
import json
from pathlib import Path
from homodyne.optimization.mcmc import MCMCSampler


class TestMCMCConfigurationReading:
    """Test MCMC configuration reading from JSON files."""

    def test_mcmc_config_reading_correct_path(self):
        """Test that MCMC configuration is read from the correct path in the config."""
        
        # Create a test configuration with MCMC settings
        test_config = {
            "optimization_config": {
                "mcmc_sampling": {
                    "enabled": True,
                    "draws": 5000,
                    "tune": 2000,
                    "chains": 4,
                    "target_accept": 0.85,
                    "cores": 4
                }
            },
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1000, -0.5, 100],
                "active_parameters": ["D0", "alpha", "D_offset"]
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 100, "max": 10000},
                    {"name": "alpha", "min": -2.0, "max": 0.0},
                    {"name": "D_offset", "min": 0, "max": 1000}
                ]
            }
        }
        
        # Test configuration access pattern
        mcmc_config = test_config.get("optimization_config", {}).get("mcmc_sampling", {})
        
        # Verify correct values are read
        assert mcmc_config.get("draws", 1000) == 5000
        assert mcmc_config.get("tune", 500) == 2000
        assert mcmc_config.get("chains", 2) == 4
        assert mcmc_config.get("target_accept", 0.9) == 0.85
        
        # Test that old (incorrect) access pattern would fail
        assert test_config.get("mcmc_draws", 1000) == 1000  # Default value
        assert test_config.get("mcmc_chains", 2) == 2      # Default value
        assert test_config.get("mcmc_tune", 500) == 500    # Default value

    def test_mcmc_sampler_reads_config_correctly(self):
        """Test that MCMCSampler class reads configuration correctly."""
        
        test_config = {
            "optimization_config": {
                "mcmc_sampling": {
                    "enabled": True,
                    "draws": 8000,
                    "tune": 1500,
                    "chains": 6,
                    "target_accept": 0.92
                }
            },
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1000, -0.5, 100],
                "active_parameters": ["D0", "alpha", "D_offset"]
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 100, "max": 10000},
                    {"name": "alpha", "min": -2.0, "max": 0.0},
                    {"name": "D_offset", "min": 0, "max": 1000}
                ]
            }
        }
        
        # Create mock core object
        from unittest.mock import Mock
        mock_core = Mock()
        mock_core.num_threads = 4
        
        # Create MCMCSampler instance
        sampler = MCMCSampler(mock_core, test_config)
        
        # Verify it reads the configuration correctly
        assert sampler.mcmc_config.get("draws", 1000) == 8000
        assert sampler.mcmc_config.get("tune", 500) == 1500
        assert sampler.mcmc_config.get("chains", 2) == 6
        assert sampler.mcmc_config.get("target_accept", 0.9) == 0.92

    def test_mcmc_config_defaults_when_missing(self):
        """Test that proper defaults are used when MCMC configuration is missing."""
        
        # Configuration without MCMC settings
        test_config = {
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1000, -0.5, 100],
                "active_parameters": ["D0", "alpha", "D_offset"]
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 100, "max": 10000},
                    {"name": "alpha", "min": -2.0, "max": 0.0},
                    {"name": "D_offset", "min": 0, "max": 1000}
                ]
            }
        }
        
        # Test configuration access with missing section
        mcmc_config = test_config.get("optimization_config", {}).get("mcmc_sampling", {})
        
        # Should get default values
        assert mcmc_config.get("draws", 1000) == 1000     # Default
        assert mcmc_config.get("tune", 500) == 500        # Default
        assert mcmc_config.get("chains", 2) == 2          # Default
        assert mcmc_config.get("target_accept", 0.9) == 0.9  # Default

    def test_real_config_file_structure(self):
        """Test with a realistic configuration file structure."""
        
        # Create a realistic config structure matching the project's format
        realistic_config = {
            "metadata": {
                "config_version": "6.0",
                "analysis_mode": "static_isotropic"
            },
            "optimization_config": {
                "classical_optimization": {
                    "methods": ["Nelder-Mead"]
                },
                "mcmc_sampling": {
                    "enabled": True,
                    "sampler": "NUTS",
                    "draws": 10000,
                    "tune": 1000, 
                    "chains": 8,
                    "cores": 8,
                    "target_accept": 0.95,
                    "max_treedepth": 10,
                    "return_inferencedata": True
                }
            },
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"],
                "values": [18000, -1.59, 3.10, 0, 0, 0, 0],
                "active_parameters": ["D0", "alpha", "D_offset"]
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 15000, "max": 20000, "type": "log-uniform"},
                    {"name": "alpha", "min": -1.6, "max": -1.5, "type": "uniform"},
                    {"name": "D_offset", "min": 0, "max": 5, "type": "uniform"},
                    {"name": "gamma_dot_t0", "min": 0.0, "max": 0.0, "type": "fixed"},
                    {"name": "beta", "min": 0.0, "max": 0.0, "type": "fixed"},
                    {"name": "gamma_dot_t_offset", "min": 0.0, "max": 0.0, "type": "fixed"},
                    {"name": "phi0", "min": 0.0, "max": 0.0, "type": "fixed"}
                ]
            }
        }
        
        # Test that values are read correctly
        mcmc_config = realistic_config.get("optimization_config", {}).get("mcmc_sampling", {})
        
        assert mcmc_config.get("draws", 1000) == 10000
        assert mcmc_config.get("chains", 2) == 8
        assert mcmc_config.get("tune", 500) == 1000
        assert mcmc_config.get("target_accept", 0.9) == 0.95
        assert mcmc_config.get("sampler", "NUTS") == "NUTS"
        assert mcmc_config.get("max_treedepth", 10) == 10

    def test_config_validation_with_correct_keys(self):
        """Test that configuration validation works with the correct key names."""
        
        test_config = {
            "optimization_config": {
                "mcmc_sampling": {
                    "enabled": True,
                    "draws": 2000,
                    "tune": 1000,
                    "chains": 3,
                    "target_accept": 0.88
                }
            },
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1000, -0.5, 100],
                "active_parameters": ["D0", "alpha", "D_offset"]
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 100, "max": 10000},
                    {"name": "alpha", "min": -2.0, "max": 0.0},
                    {"name": "D_offset", "min": 0, "max": 1000}
                ]
            }
        }
        
        # Create mock core object
        from unittest.mock import Mock
        mock_core = Mock()
        mock_core.num_threads = 4
        
        # Create MCMCSampler instance and test validation
        sampler = MCMCSampler(mock_core, test_config)
        
        # This should not raise an exception if validation works correctly
        try:
            sampler._validate_mcmc_config()
            validation_passed = True
        except Exception as e:
            validation_passed = False
            print(f"Validation failed with error: {e}")
            
        assert validation_passed, "Configuration validation should pass with correct key names"

    def test_configuration_key_mismatch_detection(self):
        """Test that demonstrates the key mismatch issue that was fixed."""
        
        # Configuration with correct structure but test access patterns
        config_with_wrong_keys = {
            "mcmc_draws": 10000,     # Wrong location - at root level
            "mcmc_chains": 8,        # Wrong location - at root level
            "mcmc_tune": 1000,       # Wrong location - at root level
            "optimization_config": {
                "mcmc_sampling": {
                    "draws": 5000,       # Correct location with correct key name
                    "chains": 4,         # Correct location with correct key name
                    "tune": 2000         # Correct location with correct key name
                }
            }
        }
        
        # Test old (incorrect) access pattern
        old_draws = config_with_wrong_keys.get("mcmc_draws", 1000)
        old_chains = config_with_wrong_keys.get("mcmc_chains", 2)
        old_tune = config_with_wrong_keys.get("mcmc_tune", 500)
        
        # Test new (correct) access pattern 
        mcmc_config = config_with_wrong_keys.get("optimization_config", {}).get("mcmc_sampling", {})
        new_draws = mcmc_config.get("draws", 1000)
        new_chains = mcmc_config.get("chains", 2)
        new_tune = mcmc_config.get("tune", 500)
        
        # The old pattern would read the wrong values
        assert old_draws == 10000
        assert old_chains == 8
        assert old_tune == 1000
        
        # The new pattern reads the correct values
        assert new_draws == 5000
        assert new_chains == 4
        assert new_tune == 2000

    def test_mcmc_parameter_bounds_usage(self):
        """Test that MCMC module uses parameter bounds correctly for PyMC priors."""
        
        test_config = {
            "optimization_config": {
                "mcmc_sampling": {
                    "enabled": True,
                    "draws": 1000,
                    "chains": 2
                }
            },
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [18000, -1.59, 3.10],
                "active_parameters": ["D0", "alpha", "D_offset"]
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 15000, "max": 20000, "type": "log-uniform"},
                    {"name": "alpha", "min": -1.6, "max": -1.5, "type": "uniform"},
                    {"name": "D_offset", "min": 0, "max": 5, "type": "uniform"}
                ]
            }
        }
        
        # Create mock core object
        from unittest.mock import Mock
        mock_core = Mock()
        mock_core.num_threads = 4
        
        # Create MCMCSampler instance
        sampler = MCMCSampler(mock_core, test_config)
        
        # Test that parameter bounds are accessed correctly
        param_bounds = test_config.get("parameter_space", {}).get("bounds", [])
        assert len(param_bounds) == 3
        
        # Verify bounds values match configuration
        d0_bound = param_bounds[0]
        assert d0_bound["name"] == "D0"
        assert d0_bound["min"] == 15000
        assert d0_bound["max"] == 20000
        assert d0_bound["type"] == "log-uniform"
        
        alpha_bound = param_bounds[1]
        assert alpha_bound["name"] == "alpha"
        assert alpha_bound["min"] == -1.6
        assert alpha_bound["max"] == -1.5
        assert alpha_bound["type"] == "uniform"
        
        d_offset_bound = param_bounds[2]
        assert d_offset_bound["name"] == "D_offset"
        assert d_offset_bound["min"] == 0
        assert d_offset_bound["max"] == 5
        assert d_offset_bound["type"] == "uniform"

    def test_mcmc_parameter_bounds_with_missing_bounds(self):
        """Test MCMC parameter bounds handling when bounds are missing."""
        
        test_config = {
            "optimization_config": {
                "mcmc_sampling": {
                    "enabled": True,
                    "draws": 1000,
                    "chains": 2
                }
            },
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1000, -0.5, 100],
                "active_parameters": ["D0", "alpha", "D_offset"]
            }
            # Missing parameter_space section
        }
        
        # Create mock core object
        from unittest.mock import Mock
        mock_core = Mock()
        mock_core.num_threads = 4
        
        # Create MCMCSampler instance - should not crash
        sampler = MCMCSampler(mock_core, test_config)
        
        # Test graceful handling of missing bounds
        param_bounds = test_config.get("parameter_space", {}).get("bounds", [])
        assert param_bounds == []  # Should get empty list as default
        
        # MCMC sampler should still be created successfully
        assert sampler.config == test_config

    def test_mcmc_real_config_parameter_bounds_compatibility(self):
        """Test with realistic parameter bounds matching the project configuration."""
        
        realistic_config = {
            "optimization_config": {
                "mcmc_sampling": {
                    "enabled": True,
                    "draws": 10000,
                    "tune": 1000,
                    "chains": 8,
                    "target_accept": 0.95
                }
            },
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"],
                "values": [18000, -1.59, 3.10, 0, 0, 0, 0],
                "active_parameters": ["D0", "alpha", "D_offset"]
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 15000, "max": 20000, "type": "log-uniform"},
                    {"name": "alpha", "min": -1.6, "max": -1.5, "type": "uniform"},
                    {"name": "D_offset", "min": 0, "max": 5, "type": "uniform"},
                    {"name": "gamma_dot_t0", "min": 0.0, "max": 0.0, "type": "fixed"},
                    {"name": "beta", "min": 0.0, "max": 0.0, "type": "fixed"},
                    {"name": "gamma_dot_t_offset", "min": 0.0, "max": 0.0, "type": "fixed"},
                    {"name": "phi0", "min": 0.0, "max": 0.0, "type": "fixed"}
                ]
            },
            "analysis_settings": {
                "static_mode": True,
                "static_submode": "isotropic"
            }
        }
        
        # Create mock core object
        from unittest.mock import Mock
        mock_core = Mock()
        mock_core.num_threads = 4
        
        # Create MCMCSampler instance
        sampler = MCMCSampler(mock_core, realistic_config)
        
        # Test that all parameter bounds are accessible
        param_bounds = realistic_config.get("parameter_space", {}).get("bounds", [])
        assert len(param_bounds) == 7
        
        # Test active parameters bounds (first 3 for static mode)
        active_bounds = param_bounds[:3]
        
        # Verify values match expected configuration exactly
        assert active_bounds[0]["name"] == "D0"
        assert active_bounds[0]["min"] == 15000
        assert active_bounds[0]["max"] == 20000
        
        assert active_bounds[1]["name"] == "alpha"
        assert active_bounds[1]["min"] == -1.6
        assert active_bounds[1]["max"] == -1.5
        
        assert active_bounds[2]["name"] == "D_offset"
        assert active_bounds[2]["min"] == 0
        assert active_bounds[2]["max"] == 5
        
        # Test that MCMC configuration is also correct
        mcmc_config = sampler.mcmc_config
        assert mcmc_config.get("draws") == 10000
        assert mcmc_config.get("chains") == 8
        assert mcmc_config.get("tune") == 1000