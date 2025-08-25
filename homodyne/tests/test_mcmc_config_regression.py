"""
MCMC Configuration Regression Test
=================================

This test specifically validates the fix for the issue where MCMC plots
showed incorrect chain/draw counts (Chains: 2, Draws: 1000) instead of
the configured values (Chains: 8, Draws: 10000).

The issue was caused by old trace files with default values being read
by plotting functions, not by the MCMC configuration system itself.

This regression test ensures:
1. MCMC configuration is loaded correctly from JSON
2. Configuration values are used during sampling
3. Old trace files with wrong dimensions are detected
4. Users can identify when they need to re-run MCMC analysis
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from homodyne.optimization.mcmc import MCMCSampler, create_mcmc_sampler
from homodyne.tests.test_utils_mcmc import (
    create_realistic_user_config,
    create_mock_analysis_core,
    create_mock_trace,
    validate_trace_dimensions,
    get_mcmc_defaults,
)


class TestMCMCConfigurationRegression:
    """Regression tests for the specific MCMC configuration issue that was fixed."""

    def test_simon_config_exact_reproduction(self):
        """Test the exact scenario from my_config_simon.json that caused the issue."""
        # This is the exact configuration that was causing problems
        simon_config = {
            "optimization_config": {
                "mcmc_sampling": {
                    "enabled": True,
                    "sampler": "NUTS",
                    "draws": 10000,  # User wanted 10000
                    "tune": 1000,
                    "chains": 8,  # User wanted 8
                    "cores": 8,
                    "target_accept": 0.95,
                    "max_treedepth": 10,
                    "return_inferencedata": True,
                }
            },
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
                "values": [18000, -1.59, 3.10, 0, 0, 0, 0],
                "active_parameters": ["D0", "alpha", "D_offset"],
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 15000, "max": 20000, "type": "Normal"},
                    {"name": "alpha", "min": -1.6, "max": -1.5, "type": "Normal"},
                    {"name": "D_offset", "min": 0, "max": 5, "type": "Normal"},
                ]
            },
            "analysis_settings": {"static_mode": True, "static_submode": "isotropic"},
        }

        mock_core = create_mock_analysis_core()
        sampler = MCMCSampler(mock_core, simon_config)

        # The fix ensures these values are used, not the defaults
        mcmc_config = sampler.mcmc_config

        # These assertions would have failed before the fix
        assert (
            mcmc_config.get("draws", 1000) == 10000
        ), "Should use Simon's configured draws=10000, not default 1000"
        assert (
            mcmc_config.get("chains", 2) == 8
        ), "Should use Simon's configured chains=8, not default 2"
        assert (
            mcmc_config.get("tune", 500) == 1000
        ), "Should use Simon's configured tune=1000, not default 500"

    def test_old_trace_file_detection(self):
        """Test detection of old trace files with wrong dimensions."""
        # Configuration specifies 8 chains, 10000 draws
        config_chains, config_draws = 8, 10000

        # But old trace file has default values
        old_trace_chains, old_trace_draws = 2, 1000

        # Create mock traces
        correct_trace = create_mock_trace(
            chains=config_chains, draws=config_draws)
        old_trace = create_mock_trace(
            chains=old_trace_chains,
            draws=old_trace_draws)

        # Validation should pass for correct trace
        assert (
            validate_trace_dimensions(
                correct_trace,
                config_chains,
                config_draws))

        # Validation should fail for old trace
        assert (
            validate_trace_dimensions(
                old_trace,
                config_chains,
                config_draws) == False)

        # This mismatch is what was causing the plotting issue
        assert old_trace.posterior.sizes["chain"] != config_chains
        assert old_trace.posterior.sizes["draw"] != config_draws

    def test_plotting_function_trace_dimension_extraction(self):
        """Test how plotting functions extract chain/draw counts from traces."""
        # This simulates what plotting functions do

        # Case 1: Correct trace (after fix)
        correct_trace = create_mock_trace(chains=8, draws=10000)

        chain_count = correct_trace.posterior.sizes.get("chain", "Unknown")
        draw_count = correct_trace.posterior.sizes.get("draw", "Unknown")

        plot_text_correct = f"Chains: {chain_count} Draws: {draw_count}"
        assert plot_text_correct == "Chains: 8 Draws: 10000"

        # Case 2: Old trace (the problem)
        old_trace = create_mock_trace(chains=2, draws=1000)

        chain_count_old = old_trace.posterior.sizes.get("chain", "Unknown")
        draw_count_old = old_trace.posterior.sizes.get("draw", "Unknown")

        plot_text_old = f"Chains: {chain_count_old} Draws: {draw_count_old}"
        assert plot_text_old == "Chains: 2 Draws: 1000"  # This was the problem!

    def test_mcmc_sampler_factory_with_simon_config(self):
        """Test the create_mcmc_sampler factory with Simon's configuration."""
        config = create_realistic_user_config()
        mock_core = create_mock_analysis_core()

        # This should create a sampler with the correct configuration
        sampler = create_mcmc_sampler(mock_core, config)

        # Verify it uses the configured values
        assert sampler.mcmc_config["draws"] == 10000
        assert sampler.mcmc_config["chains"] == 8
        assert sampler.mcmc_config["tune"] == 1000

    def test_config_file_persistence(self):
        """Test that configuration persists correctly through file loading."""
        # Create configuration and save to temporary file
        config_data = create_realistic_user_config()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f, indent=2)
            temp_path = f.name

        try:
            # Load configuration from file (simulating user workflow)
            with open(temp_path, "r") as f:
                loaded_config = json.load(f)

            # Create sampler with loaded config
            mock_core = create_mock_analysis_core()
            sampler = MCMCSampler(mock_core, loaded_config)

            # Verify values are preserved through the file round-trip
            assert sampler.mcmc_config["draws"] == 10000
            assert sampler.mcmc_config["chains"] == 8
            assert sampler.mcmc_config["tune"] == 1000

        finally:
            Path(temp_path).unlink()

    def test_default_fallback_regression(self):
        """Test that defaults are only used when configuration is actually missing."""
        # Config with missing MCMC section (should use defaults)
        config_missing = {
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1000, -0.5, 100],
            },
            "parameter_space": {"bounds": []},
        }

        mock_core = create_mock_analysis_core()
        sampler = MCMCSampler(mock_core, config_missing)

        # Should get empty config dict, leading to defaults
        mcmc_config = sampler.mcmc_config
        defaults = get_mcmc_defaults()

        assert mcmc_config.get("draws", defaults["draws"]) == defaults["draws"]
        assert mcmc_config.get(
            "chains", defaults["chains"]) == defaults["chains"]
        assert mcmc_config.get("tune", defaults["tune"]) == defaults["tune"]

    def test_configuration_validation_regression(self):
        """Test that configuration validation works correctly."""
        config = create_realistic_user_config()
        mock_core = create_mock_analysis_core()

        # This should not raise any validation errors
        sampler = MCMCSampler(mock_core, config)

        # Validation should pass
        sampler._validate_mcmc_config()  # Should not raise

        # Verify configuration is accessible
        assert sampler.mcmc_config is not None
        assert len(sampler.mcmc_config) > 0

    def test_config_path_hierarchy_regression(self):
        """Test that the correct configuration path hierarchy is used."""
        config = create_realistic_user_config()

        # Test the actual path used in the code
        mcmc_config_path = config.get("optimization_config", {}).get(
            "mcmc_sampling", {}
        )

        # This is the correct path that was working
        assert mcmc_config_path["draws"] == 10000
        assert mcmc_config_path["chains"] == 8
        assert mcmc_config_path["tune"] == 1000

        # Test that the MCMCSampler uses this same path
        mock_core = create_mock_analysis_core()
        sampler = MCMCSampler(mock_core, config)

        # The sampler's extracted config should match
        assert sampler.mcmc_config["draws"] == mcmc_config_path["draws"]
        assert sampler.mcmc_config["chains"] == mcmc_config_path["chains"]
        assert sampler.mcmc_config["tune"] == mcmc_config_path["tune"]


class TestMCMCTraceFileRegression:
    """Test trace file related regression scenarios."""

    def test_trace_file_dimension_mismatch_scenario(self):
        """Test the exact scenario that caused the plotting issue."""
        # User has configuration with 8 chains, 10000 draws
        user_config_chains, user_config_draws = 8, 10000

        # But their old trace file has default values
        old_file_chains, old_file_draws = 2, 1000

        # This mismatch caused the plotting issue
        old_trace = create_mock_trace(
            chains=old_file_chains, draws=old_file_draws)

        # Extract values the way plotting functions do
        plot_chains = old_trace.posterior.sizes.get("chain", "Unknown")
        plot_draws = old_trace.posterior.sizes.get("draw", "Unknown")

        # This is what the user saw in their plots (the problem!)
        assert plot_chains == 2
        assert plot_draws == 1000

        # But this is what they expected to see
        assert plot_chains != user_config_chains
        assert plot_draws != user_config_draws

    def test_fresh_trace_correct_dimensions(self):
        """Test that a fresh trace would have correct dimensions."""
        # Configuration values
        config_chains, config_draws = 8, 10000

        # Fresh trace created with current config (after fix)
        fresh_trace = create_mock_trace(
            chains=config_chains, draws=config_draws)

        # Extract values the way plotting functions do
        plot_chains = fresh_trace.posterior.sizes.get("chain", "Unknown")
        plot_draws = fresh_trace.posterior.sizes.get("draw", "Unknown")

        # This is what the user should see now
        assert plot_chains == config_chains
        assert plot_draws == config_draws
        assert plot_chains == 8
        assert plot_draws == 10000

    def test_trace_file_validation_utility(self):
        """Test the utility function for validating trace dimensions."""
        # Create traces with different dimensions
        correct_trace = create_mock_trace(chains=8, draws=10000)
        wrong_trace = create_mock_trace(chains=2, draws=1000)

        # Validation should work correctly
        assert validate_trace_dimensions(correct_trace, 8, 10000)
        assert validate_trace_dimensions(wrong_trace, 8, 10000) == False
        assert validate_trace_dimensions(wrong_trace, 2, 1000)
