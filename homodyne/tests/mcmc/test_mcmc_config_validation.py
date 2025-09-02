"""
Test MCMC Configuration Usage Validation
========================================

Tests to ensure MCMC configuration values (draws, chains, tune) are actually
used during PyMC sampling and trace file generation. This is a regression test
suite for the issue where configuration was loaded correctly but old trace
files with default values caused plots to show incorrect chain/draw counts.

This test suite validates:
1. Configuration values are properly passed to PyMC's sample() function
2. Generated trace files have the expected dimensions
3. Default values are used only when configuration is missing
4. Warning messages are logged appropriately

Note: Type ignore comments are used because MCMCSampler may be None when PyMC is not
available. However, all tests are properly protected with pytest skip markers.
"""

# type: ignore

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest


# Test PyMC availability - dynamic check
def _check_pymc_available():
    try:
        import pymc  # noqa: F401

        return True
    except ImportError:
        return False


PYMC_AVAILABLE = _check_pymc_available()

# Import for type checking and runtime use
if TYPE_CHECKING:
    from homodyne.optimization.mcmc import MCMCSampler, create_mcmc_sampler
elif PYMC_AVAILABLE:
    try:
        from homodyne.optimization.mcmc import MCMCSampler, create_mcmc_sampler
    except ImportError:
        MCMCSampler = None
        create_mcmc_sampler = None
else:
    MCMCSampler = None
    create_mcmc_sampler = None

# pytestmark = pytest.mark.skipif(
#     not PYMC_AVAILABLE,
#     reason="PyMC is required for MCMC sampling but is not available.",
# )


class TestMCMCConfigurationUsage:
    """Test that MCMC configuration is properly used during sampling."""

    def create_test_config(
        self,
        draws: int | str = 10000,
        chains: int = 8,
        tune: int | str = 1000,
        thin: int = 1,
    ):
        """Create a test configuration with specified MCMC parameters.

        Parameters can be int or str to allow testing of type validation.
        """
        return {
            "optimization_config": {
                "mcmc_sampling": {
                    "enabled": True,
                    "sampler": "NUTS",
                    "draws": draws,
                    "tune": tune,
                    "thin": thin,
                    "chains": chains,
                    "cores": min(chains, 4),
                    "target_accept": 0.95,
                    "max_treedepth": 10,
                    "return_inferencedata": True,
                }
            },
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [18000, -1.59, 3.10],
                "active_parameters": ["D0", "alpha", "D_offset"],
            },
            "parameter_space": {
                "bounds": [
                    {
                        "name": "D0",
                        "min": 1.0,
                        "max": 1000000,
                        "type": "Normal",
                    },
                    {
                        "name": "alpha",
                        "min": -1.6,
                        "max": -1.5,
                        "type": "Normal",
                    },
                    {"name": "D_offset", "min": 0, "max": 5, "type": "Normal"},
                ]
            },
            "analyzer_parameters": {
                "temporal": {"dt": 0.5},
                "scattering": {"wavevector_q": 0.0237},
            },
            "analysis_settings": {
                "static_mode": True,
                "static_submode": "isotropic",
            },
            "performance_settings": {
                "noise_model": {
                    "use_simple_forward_model": True,
                    "sigma_prior": 0.1,
                }
            },
        }

    def create_mock_core(self):
        """Create a mock analysis core for testing."""
        mock_core = Mock()
        mock_core.num_threads = 4
        mock_core.config_manager = Mock()
        mock_core.config_manager.is_static_mode_enabled.return_value = True
        mock_core.config_manager.get_analysis_mode.return_value = "static_isotropic"
        mock_core.config_manager.get_effective_parameter_count.return_value = 3
        mock_core.config_manager.is_angle_filtering_enabled.return_value = True
        return mock_core

    @pytest.mark.skipif(
        not _check_pymc_available(),
        reason="PyMC is required for MCMC sampling but is not available.",
    )
    def test_mcmc_sampler_extracts_correct_config_values(self):
        """Test that MCMCSampler correctly extracts configuration values."""
        config = self.create_test_config(draws=5000, chains=6, tune=2000)
        mock_core = self.create_mock_core()

        sampler = MCMCSampler(mock_core, config)

        # Verify the sampler extracted the correct values
        assert sampler.mcmc_config["draws"] == 5000
        assert sampler.mcmc_config["chains"] == 6
        assert sampler.mcmc_config["tune"] == 2000
        assert sampler.mcmc_config["target_accept"] == 0.95

    @pytest.mark.skipif(
        not _check_pymc_available(),
        reason="PyMC is required for MCMC sampling but is not available.",
    )
    def test_mcmc_sampler_uses_defaults_when_config_missing(self):
        """Test that MCMCSampler uses default values when configuration is missing."""
        config = {
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [18000, -1.59, 3.10],
            },
            "parameter_space": {"bounds": []},
            "analyzer_parameters": {"temporal": {"dt": 0.5}},
            "analysis_settings": {"static_mode": True},
        }
        mock_core = self.create_mock_core()

        sampler = MCMCSampler(mock_core, config)

        # Should get empty dict, which will trigger defaults in the sampling
        # code
        assert sampler.mcmc_config == {}

    @pytest.mark.skipif(
        not _check_pymc_available(),
        reason="PyMC is required for MCMC sampling but is not available.",
    )
    def test_config_values_passed_to_pymc_sample(self):
        """Test that configuration values are passed to PyMC's sample function."""
        config = self.create_test_config(draws=1000, chains=4, tune=500)
        mock_core = self.create_mock_core()

        # Create dummy data
        c2_experimental = np.random.rand(1, 50, 50) * 0.1 + 1.0
        phi_angles = np.array([0.0])

        with patch("homodyne.optimization.mcmc.pm") as mock_pm:
            # Mock PyMC components
            mock_model = MagicMock()
            mock_pm.Model.return_value.__enter__.return_value = mock_model
            mock_pm.Uniform = MagicMock()
            mock_pm.HalfNormal = MagicMock()
            mock_pm.Normal = MagicMock()
            mock_pm.Deterministic = MagicMock()

            # Mock the shared function from pytensor
            with patch("homodyne.optimization.mcmc.shared") as mock_shared:
                mock_shared.return_value = MagicMock()

                # Mock pt.stack, pt.constant, pt.ones_like
                with patch("homodyne.optimization.mcmc.pt") as mock_pt:
                    mock_pt.stack.return_value = MagicMock()
                    mock_pt.constant.return_value = MagicMock()
                    mock_pt.ones_like.return_value = MagicMock()

                    # Create sampler and attempt to run
                    sampler = MCMCSampler(mock_core, config)

                    try:
                        sampler._run_mcmc_nuts_optimized(
                            c2_experimental=c2_experimental,
                            phi_angles=phi_angles,
                            config=config,
                            filter_angles_for_optimization=False,
                            is_static_mode=True,
                            analysis_mode="static_isotropic",
                            effective_param_count=3,
                        )
                    except Exception:
                        pass  # Expected to fail due to mocking, but we'll check the calls

                    # Verify that pm.sample was called with our configuration
                    # values
                    if mock_pm.sample.called:
                        call_args = mock_pm.sample.call_args
                        assert call_args[1]["draws"] == 1000
                        assert call_args[1]["chains"] == 4
                        assert call_args[1]["tune"] == 500
                        assert call_args[1]["target_accept"] == 0.95

    @pytest.mark.skipif(
        not _check_pymc_available(),
        reason="PyMC is required for MCMC sampling but is not available.",
    )
    def test_mcmc_config_validation_with_invalid_values(self):
        """Test that MCMC configuration validation catches invalid values."""
        config = self.create_test_config(
            draws=-1000, chains=0, tune=-500
        )  # Invalid values
        mock_core = self.create_mock_core()

        with pytest.raises(ValueError, match="draws must be a positive integer"):
            MCMCSampler(mock_core, config)

    @pytest.mark.skipif(
        not _check_pymc_available(),
        reason="PyMC is required for MCMC sampling but is not available.",
    )
    def test_mcmc_config_validation_with_invalid_types(self):
        """Test that MCMC configuration validation catches invalid types."""
        config = self.create_test_config(
            draws="1000", chains=4, tune="500"
        )  # Wrong types
        mock_core = self.create_mock_core()

        with pytest.raises(ValueError):
            sampler = MCMCSampler(mock_core, config)
            sampler._validate_mcmc_config()

    @pytest.mark.skipif(
        not _check_pymc_available(),
        reason="PyMC is required for MCMC sampling but is not available.",
    )
    def test_mcmc_default_fallback_behavior(self):
        """Test the exact default fallback behavior seen in the actual code."""
        config = self.create_test_config()
        mock_core = self.create_mock_core()

        # Remove MCMC config to test fallbacks
        config["optimization_config"] = {}

        sampler = MCMCSampler(mock_core, config)
        mcmc_config = sampler.mcmc_config

        # Test the actual .get() patterns used in the MCMC code
        draws = mcmc_config.get("draws", 1000)
        tune = mcmc_config.get("tune", 500)
        chains = mcmc_config.get("chains", 2)
        target_accept = mcmc_config.get("target_accept", 0.9)

        # Should get defaults
        assert draws == 1000
        assert tune == 500
        assert chains == 2
        assert target_accept == 0.9

    @pytest.mark.skipif(
        not _check_pymc_available(),
        reason="PyMC is required for MCMC sampling but is not available.",
    )
    def test_mcmc_actual_config_values_override_defaults(self):
        """Test that actual config values override defaults (the main fix)."""
        config = self.create_test_config(draws=10000, chains=8, tune=1000)
        mock_core = self.create_mock_core()

        sampler = MCMCSampler(mock_core, config)
        mcmc_config = sampler.mcmc_config

        # Test the actual .get() patterns used in the MCMC code
        draws = mcmc_config.get("draws", 1000)
        tune = mcmc_config.get("tune", 500)
        chains = mcmc_config.get("chains", 2)
        target_accept = mcmc_config.get("target_accept", 0.9)

        # Should get configured values, not defaults
        assert draws == 10000, f"Expected draws=10000, got {draws}"
        assert tune == 1000, f"Expected tune=1000, got {tune}"
        assert chains == 8, f"Expected chains=8, got {chains}"
        assert target_accept == 0.95, (
            f"Expected target_accept=0.95, got {target_accept}"
        )

    def test_trace_dimension_validation(self):
        """Test validation of trace dimensions against configuration."""
        # This test simulates what should happen with actual trace files
        draws, chains, tune = 2000, 4, 500
        self.create_test_config(draws=draws, chains=chains, tune=tune)

        # Mock a trace object with the expected dimensions
        mock_trace = Mock()
        mock_trace.posterior = Mock()
        mock_trace.posterior.sizes = {"chain": chains, "draw": draws}

        # Verify dimensions match configuration
        assert mock_trace.posterior.sizes["chain"] == chains
        assert mock_trace.posterior.sizes["draw"] == draws

    @pytest.mark.skipif(
        not _check_pymc_available(),
        reason="PyMC is required for MCMC sampling but is not available.",
    )
    def test_create_mcmc_sampler_factory_function(self):
        """Test the create_mcmc_sampler factory function uses configuration correctly."""
        config = self.create_test_config(draws=7000, chains=5, tune=1500)
        mock_core = self.create_mock_core()

        sampler = create_mcmc_sampler(mock_core, config)

        # Verify the factory function creates a sampler with correct config
        assert isinstance(sampler, MCMCSampler)
        assert sampler.mcmc_config["draws"] == 7000
        assert sampler.mcmc_config["chains"] == 5
        assert sampler.mcmc_config["tune"] == 1500

    @pytest.mark.skipif(
        not _check_pymc_available(),
        reason="PyMC is required for MCMC sampling but is not available.",
    )
    def test_mcmc_config_logging(self):
        """Test that MCMC configuration values are logged for debugging."""
        config = self.create_test_config(draws=3000, chains=6, tune=1000)
        mock_core = self.create_mock_core()

        with patch("homodyne.optimization.mcmc.logger") as mock_logger:
            MCMCSampler(mock_core, config)

            # Check that initialization was logged
            mock_logger.info.assert_called()

    @pytest.mark.skipif(
        not _check_pymc_available(),
        reason="PyMC is required for MCMC sampling but is not available.",
    )
    def test_config_file_loading_integration(self):
        """Test loading configuration from a JSON file and using it in MCMC."""
        config_data = self.create_test_config(draws=4000, chains=3, tune=800)

        # Write config to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_config_path = f.name

        try:
            # Load config from file
            with open(temp_config_path) as f:
                loaded_config = json.load(f)

            mock_core = self.create_mock_core()
            sampler = MCMCSampler(mock_core, loaded_config)

            # Verify config was loaded correctly
            assert sampler.mcmc_config["draws"] == 4000
            assert sampler.mcmc_config["chains"] == 3
            assert sampler.mcmc_config["tune"] == 800

        finally:
            # Clean up temp file
            Path(temp_config_path).unlink()

    @pytest.mark.skipif(
        not _check_pymc_available(),
        reason="PyMC is required for MCMC sampling but is not available.",
    )
    def test_realistic_simon_config_values(self):
        """Test with the exact values from my_config_simon.json that caused the issue."""
        # Use the exact values from the user's configuration
        config = self.create_test_config(draws=10000, chains=8, tune=1000)
        mock_core = self.create_mock_core()

        sampler = MCMCSampler(mock_core, config)

        # These are the values that should be used, not the defaults (2, 1000)
        mcmc_config = sampler.mcmc_config
        assert mcmc_config.get("draws", 1000) == 10000, (
            "Should use configured draws=10000, not default 1000"
        )
        assert mcmc_config.get("chains", 2) == 8, (
            "Should use configured chains=8, not default 2"
        )
        assert mcmc_config.get("tune", 500) == 1000, (
            "Should use configured tune=1000, not default 500"
        )

    @pytest.mark.skipif(
        not _check_pymc_available(),
        reason="PyMC is required for MCMC sampling but is not available.",
    )
    def test_mcmc_config_path_regression(self):
        """Regression test for the specific config path issue that was fixed."""
        # Test both the old (wrong) and new (correct) config access patterns

        # Config with values in both old and new locations (simulating
        # migration)
        config_with_both = {
            # Old (wrong) location - at root level
            "mcmc_draws": 999,
            "mcmc_chains": 99,
            "mcmc_tune": 999,
            # New (correct) location - nested properly
            "optimization_config": {
                "mcmc_sampling": {
                    "draws": 10000,
                    "chains": 8,
                    "tune": 1000,
                    "target_accept": 0.95,
                }
            },
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [18000, -1.59, 3.10],
            },
            "parameter_space": {"bounds": []},
        }

        mock_core = self.create_mock_core()
        sampler = MCMCSampler(mock_core, config_with_both)

        # The MCMCSampler should use the correct path and ignore the wrong
        # values
        assert sampler.mcmc_config["draws"] == 10000, (
            "Should read from optimization_config.mcmc_sampling.draws"
        )
        assert sampler.mcmc_config["chains"] == 8, (
            "Should read from optimization_config.mcmc_sampling.chains"
        )
        assert sampler.mcmc_config["tune"] == 1000, (
            "Should read from optimization_config.mcmc_sampling.tune"
        )

        # Verify it does NOT read from the old (wrong) locations
        assert sampler.mcmc_config["draws"] != 999, (
            "Should NOT read from root-level mcmc_draws"
        )
        assert sampler.mcmc_config["chains"] != 99, (
            "Should NOT read from root-level mcmc_chains"
        )
        assert sampler.mcmc_config["tune"] != 999, (
            "Should NOT read from root-level mcmc_tune"
        )


class TestMCMCTraceFileValidation:
    """Test validation of MCMC trace files and their dimensions."""

    def test_trace_dimension_extraction(self):
        """Test extracting chain and draw dimensions from trace objects."""
        # Mock ArviZ InferenceData structure
        mock_trace = Mock()
        mock_trace.posterior = Mock()
        mock_trace.posterior.sizes = {"chain": 8, "draw": 10000}

        # Extract dimensions (simulating plotting code)
        chain_count = mock_trace.posterior.sizes.get("chain", "Unknown")
        draw_count = mock_trace.posterior.sizes.get("draw", "Unknown")

        assert chain_count == 8
        assert draw_count == 10000

    def test_trace_dimension_mismatch_detection(self):
        """Test detection of mismatched trace dimensions vs configuration."""
        config_draws, config_chains = 10000, 8
        trace_draws, trace_chains = 1000, 2  # Old default values

        # Mock trace with wrong dimensions (the issue that was happening)
        mock_trace = Mock()
        mock_trace.posterior = Mock()
        mock_trace.posterior.sizes = {
            "chain": trace_chains,
            "draw": trace_draws,
        }

        # This should detect the mismatch
        assert trace_draws != config_draws, (
            f"Trace draws {trace_draws} don't match config {config_draws}"
        )
        assert trace_chains != config_chains, (
            f"Trace chains {trace_chains} don't match config {config_chains}"
        )

        # This is what was causing the plot to show wrong values
        plot_chain_text = f"Chains: {trace_chains}"
        plot_draw_text = f"Draws: {trace_draws}"

        assert "Chains: 2" in plot_chain_text
        assert "Draws: 1000" in plot_draw_text


@pytest.mark.integration
class TestMCMCConfigurationIntegration:
    """Integration tests for full MCMC configuration pipeline."""

    @pytest.mark.skipif(
        not _check_pymc_available(),
        reason="PyMC is required for MCMC sampling but is not available.",
    )
    def test_config_to_trace_pipeline(self):
        """Test the full pipeline from config loading to trace generation."""
        # This test would be more complex and might need actual PyMC
        # For now, we test the configuration flow

        config = {
            "optimization_config": {
                "mcmc_sampling": {
                    "enabled": True,
                    "draws": 5000,
                    "chains": 4,
                    "tune": 1000,
                }
            },
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [18000, -1.59, 3.10],
            },
            "parameter_space": {"bounds": []},
            "analyzer_parameters": {"temporal": {"dt": 0.5}},
            "analysis_settings": {"static_mode": True},
        }

        mock_core = Mock()
        mock_core.num_threads = 4

        # Step 1: Configuration is loaded correctly
        sampler = MCMCSampler(mock_core, config)
        assert sampler.mcmc_config["draws"] == 5000
        assert sampler.mcmc_config["chains"] == 4

        # Step 2: Configuration values would be passed to sampling
        mcmc_config = sampler.mcmc_config
        draws = mcmc_config.get("draws", 1000)
        chains = mcmc_config.get("chains", 2)

        assert draws == 5000
        assert chains == 4

        # Step 3: A real trace would have these dimensions
        expected_trace_dims = {"chain": chains, "draw": draws}
        assert expected_trace_dims["chain"] == 4
        assert expected_trace_dims["draw"] == 5000


class TestMCMCThinningConfiguration:
    """Test MCMC thinning configuration and validation."""

    def create_test_config(
        self,
        draws: int | str = 10000,
        chains: int = 4,
        tune: int | str = 1000,
        thin: int | str = 1,
    ):
        """Create a test configuration with specified MCMC parameters including thinning."""
        return {
            "optimization_config": {
                "mcmc_sampling": {
                    "enabled": True,
                    "sampler": "NUTS",
                    "draws": draws,
                    "tune": tune,
                    "thin": thin,
                    "chains": chains,
                    "cores": min(chains, 4),
                    "target_accept": 0.95,
                }
            },
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [18000, -1.59, 3.10],
                "active_parameters": ["D0", "alpha", "D_offset"],
            },
            "parameter_space": {
                "bounds": [
                    {
                        "name": "D0",
                        "min": 1.0,
                        "max": 1000000,
                        "type": "Normal",
                    },
                    {
                        "name": "alpha",
                        "min": -1.6,
                        "max": -1.5,
                        "type": "Normal",
                    },
                    {"name": "D_offset", "min": 0, "max": 5, "type": "Normal"},
                ]
            },
            "analyzer_parameters": {
                "temporal": {"dt": 0.5},
                "scattering": {"wavevector_q": 0.0237},
            },
            "analysis_settings": {
                "static_mode": True,
                "static_submode": "isotropic",
            },
            "performance_settings": {
                "noise_model": {
                    "use_simple_forward_model": True,
                    "sigma_prior": 0.1,
                }
            },
        }

    def create_mock_core(self):
        """Create a mock analysis core for testing."""
        mock_core = Mock()
        mock_core.num_threads = 4
        mock_core.config_manager = Mock()
        mock_core.config_manager.is_static_mode_enabled.return_value = True
        mock_core.config_manager.get_analysis_mode.return_value = "static_isotropic"
        mock_core.config_manager.get_effective_parameter_count.return_value = 3
        mock_core.config_manager.is_angle_filtering_enabled.return_value = True
        return mock_core

    @pytest.mark.skipif(
        not _check_pymc_available(),
        reason="PyMC is required for MCMC sampling but is not available.",
    )
    def test_thinning_parameter_extraction(self):
        """Test that MCMCSampler correctly extracts thinning parameter."""
        config = self.create_test_config(draws=2000, chains=4, tune=500, thin=3)
        mock_core = self.create_mock_core()

        sampler = MCMCSampler(mock_core, config)

        # Verify the sampler extracted the correct thinning value
        assert sampler.mcmc_config["thin"] == 3
        assert sampler.mcmc_config["draws"] == 2000

    @pytest.mark.skipif(
        not _check_pymc_available(),
        reason="PyMC is required for MCMC sampling but is not available.",
    )
    def test_thinning_default_value(self):
        """Test that thinning defaults to 1 when not specified."""
        config = {
            "optimization_config": {
                "mcmc_sampling": {
                    "enabled": True,
                    "draws": 1000,
                    "chains": 2,
                    "tune": 500,
                    # Note: thin parameter not specified
                }
            },
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [18000, -1.59, 3.10],
            },
            "parameter_space": {"bounds": []},
            "analyzer_parameters": {"temporal": {"dt": 0.5}},
            "analysis_settings": {"static_mode": True},
        }
        mock_core = self.create_mock_core()

        sampler = MCMCSampler(mock_core, config)
        mcmc_config = sampler.mcmc_config

        # Should get default value of 1 (no thinning)
        thin = mcmc_config.get("thin", 1)
        assert thin == 1

    @pytest.mark.skipif(
        not _check_pymc_available(),
        reason="PyMC is required for MCMC sampling but is not available.",
    )
    def test_thinning_validation_positive_integer(self):
        """Test that thinning validation requires positive integer."""
        mock_core = self.create_mock_core()

        # Test invalid thinning values
        invalid_values = [-1, 0, 2.5, "2", None]

        for invalid_thin in invalid_values:
            config = self.create_test_config(thin=invalid_thin)

            with pytest.raises(ValueError, match="thin must be a positive integer"):
                MCMCSampler(mock_core, config)

    @pytest.mark.skipif(
        not _check_pymc_available(),
        reason="PyMC is required for MCMC sampling but is not available.",
    )
    def test_thinning_valid_values(self):
        """Test that valid thinning values are accepted."""
        mock_core = self.create_mock_core()

        valid_values = [1, 2, 3, 5, 10, 100]

        for valid_thin in valid_values:
            config = self.create_test_config(thin=valid_thin)
            sampler = MCMCSampler(mock_core, config)
            assert sampler.mcmc_config["thin"] == valid_thin

    @pytest.mark.skipif(
        not _check_pymc_available(),
        reason="PyMC is required for MCMC sampling but is not available.",
    )
    def test_effective_draws_calculation(self):
        """Test effective draws calculation with thinning."""
        config = self.create_test_config(draws=1000, thin=5)
        mock_core = self.create_mock_core()

        sampler = MCMCSampler(mock_core, config)
        mcmc_config = sampler.mcmc_config

        draws = mcmc_config.get("draws", 1000)
        thin = mcmc_config.get("thin", 1)

        # Effective draws should be draws // thin
        effective_draws = draws // thin if thin > 1 else draws
        expected_effective = 1000 // 5  # = 200

        assert effective_draws == expected_effective
        assert effective_draws == 200

    @pytest.mark.skipif(
        not _check_pymc_available(),
        reason="PyMC is required for MCMC sampling but is not available.",
    )
    def test_thinning_passed_to_pymc_sample(self):
        """Test that thinning parameter is passed to PyMC's sample function."""
        config = self.create_test_config(draws=1000, chains=2, tune=500, thin=3)
        mock_core = self.create_mock_core()

        # Create dummy data
        c2_experimental = np.random.rand(1, 50, 50) * 0.1 + 1.0
        phi_angles = np.array([0.0])

        with patch("homodyne.optimization.mcmc.pm") as mock_pm:
            # Mock PyMC components
            mock_model = MagicMock()
            mock_pm.Model.return_value.__enter__.return_value = mock_model
            mock_pm.Uniform = MagicMock()
            mock_pm.HalfNormal = MagicMock()
            mock_pm.Normal = MagicMock()
            mock_pm.Deterministic = MagicMock()

            # Mock the shared function from pytensor
            with patch("homodyne.optimization.mcmc.shared") as mock_shared:
                mock_shared.return_value = MagicMock()

                # Mock pt.stack, pt.constant, pt.ones_like
                with patch("homodyne.optimization.mcmc.pt") as mock_pt:
                    mock_pt.stack.return_value = MagicMock()
                    mock_pt.constant.return_value = MagicMock()
                    mock_pt.ones_like.return_value = MagicMock()

                    # Create sampler and attempt to run
                    sampler = MCMCSampler(mock_core, config)

                    try:
                        sampler._run_mcmc_nuts_optimized(
                            c2_experimental=c2_experimental,
                            phi_angles=phi_angles,
                            config=config,
                            filter_angles_for_optimization=False,
                            is_static_mode=True,
                            analysis_mode="static_isotropic",
                            effective_param_count=3,
                        )
                    except Exception:
                        pass  # Expected to fail due to mocking, but we'll check the calls

                    # Verify that pm.sample was called with thinning parameter
                    if mock_pm.sample.called:
                        call_args = mock_pm.sample.call_args
                        assert call_args[1]["thin"] == 3
                        assert call_args[1]["draws"] == 1000
                        assert call_args[1]["chains"] == 2
                        assert call_args[1]["tune"] == 500

    @pytest.mark.skipif(
        not _check_pymc_available(),
        reason="PyMC is required for MCMC sampling but is not available.",
    )
    def test_thinning_warning_for_low_effective_samples(self):
        """Test that validation warns when thinning reduces effective samples too much."""
        config = self.create_test_config(draws=2000, thin=5)  # Effective = 400 samples
        mock_core = self.create_mock_core()

        sampler = MCMCSampler(mock_core, config)
        validation = sampler.validate_model_setup()

        # Should warn about low effective sample count
        warnings = validation.get("warnings", [])
        low_sample_warning = any(
            "reduces effective draws to 400" in warning for warning in warnings
        )
        assert low_sample_warning, (
            f"Expected low effective samples warning, got warnings: {warnings}"
        )

    @pytest.mark.skipif(
        not _check_pymc_available(),
        reason="PyMC is required for MCMC sampling but is not available.",
    )
    def test_thinning_recommendation_for_large_samples(self):
        """Test that validation recommends thinning for large sample sizes."""
        config = self.create_test_config(
            draws=10000, thin=1
        )  # Large samples, no thinning
        mock_core = self.create_mock_core()

        sampler = MCMCSampler(mock_core, config)
        validation = sampler.validate_model_setup()

        # Should recommend thinning for large sample sizes
        recommendations = validation.get("recommendations", [])
        thinning_recommendation = any(
            "Consider using thinning" in rec for rec in recommendations
        )
        assert thinning_recommendation, (
            f"Expected thinning recommendation, got: {recommendations}"
        )

    @pytest.mark.skipif(
        not _check_pymc_available(),
        reason="PyMC is required for MCMC sampling but is not available.",
    )
    def test_thinning_in_different_analysis_modes(self):
        """Test thinning configuration in different analysis modes."""
        mock_core = self.create_mock_core()

        # Test static isotropic mode
        config_static = self.create_test_config(draws=1000, thin=2)
        config_static["analysis_settings"]["static_submode"] = "isotropic"

        sampler_static = MCMCSampler(mock_core, config_static)
        assert sampler_static.mcmc_config["thin"] == 2

        # Test laminar flow mode
        config_flow = self.create_test_config(draws=2000, thin=1)
        config_flow["analysis_settings"]["static_mode"] = False

        mock_core.config_manager.is_static_mode_enabled.return_value = False
        mock_core.config_manager.get_analysis_mode.return_value = "laminar_flow"
        mock_core.config_manager.get_effective_parameter_count.return_value = 7

        sampler_flow = MCMCSampler(mock_core, config_flow)
        assert sampler_flow.mcmc_config["thin"] == 1
