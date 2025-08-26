"""
MCMC Testing Utilities
======================

Utility functions and helpers for testing MCMC configuration and behavior.
These utilities help ensure consistent testing across different MCMC test files.
"""

from unittest.mock import Mock

import numpy as np
import pytest

# Test PyMC availability
try:
    import arviz as az
    import pymc as pm

    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    pm = None
    az = None


pytestmark = pytest.mark.skipif(
    not PYMC_AVAILABLE,
    reason="PyMC is required for MCMC sampling but is not available.",
)


def create_minimal_mcmc_config(draws=1000, chains=2, tune=500, **kwargs):
    """Create a minimal MCMC configuration for testing.

    Parameters
    ----------
    draws : int, optional
        Number of MCMC draws (default: 1000)
    chains : int, optional
        Number of MCMC chains (default: 2)
    tune : int, optional
        Number of tuning steps (default: 500)
    **kwargs : dict
        Additional MCMC configuration parameters

    Returns
    -------
    dict
        Complete configuration dictionary with MCMC settings
    """
    mcmc_config = {
        "enabled": True,
        "sampler": "NUTS",
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "cores": min(chains, 4),
        "target_accept": 0.9,
        "max_treedepth": 10,
        "return_inferencedata": True,
    }
    mcmc_config.update(kwargs)

    return {
        "optimization_config": {"mcmc_sampling": mcmc_config},
        "initial_parameters": {
            "parameter_names": ["D0", "alpha", "D_offset"],
            "values": [1000, -0.5, 100],
            "active_parameters": ["D0", "alpha", "D_offset"],
        },
        "parameter_space": {
            "bounds": [
                {"name": "D0", "min": 1.0, "max": 1000000, "type": "Normal"},
                {"name": "alpha", "min": -2.0, "max": 2.0, "type": "Normal"},
                {
                    "name": "D_offset",
                    "min": -100,
                    "max": 100,
                    "type": "Normal",
                },
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


def create_mock_analysis_core():
    """Create a mock HomodyneAnalysisCore for testing.

    Returns
    -------
    Mock
        Mock object configured to behave like HomodyneAnalysisCore
    """
    mock_core = Mock()
    mock_core.num_threads = 4

    # Mock config manager
    mock_core.config_manager = Mock()
    mock_core.config_manager.is_static_mode_enabled.return_value = True
    mock_core.config_manager.get_analysis_mode.return_value = (
        "static_isotropic"
    )
    mock_core.config_manager.get_effective_parameter_count.return_value = 3
    mock_core.config_manager.is_angle_filtering_enabled.return_value = True

    return mock_core


def create_dummy_experimental_data(n_angles=1, n_time=50):
    """Create dummy experimental data for MCMC testing.

    Parameters
    ----------
    n_angles : int, optional
        Number of scattering angles (default: 1)
    n_time : int, optional
        Number of time points (default: 50)

    Returns
    -------
    tuple
        (c2_experimental, phi_angles) where:
        - c2_experimental: np.ndarray of shape (n_angles, n_time, n_time)
        - phi_angles: np.ndarray of shape (n_angles,)
    """
    # Create dummy c2 data with realistic values
    c2_experimental = np.random.rand(n_angles, n_time, n_time) * 0.1 + 1.0

    # Create dummy angles
    if n_angles == 1:
        phi_angles = np.array([0.0])
    else:
        phi_angles = np.linspace(0, 180, n_angles)

    return c2_experimental, phi_angles


def assert_mcmc_config_values(
    sampler, expected_draws=None, expected_chains=None, expected_tune=None
):
    """Assert that an MCMCSampler has the expected configuration values.

    Parameters
    ----------
    sampler : MCMCSampler
        The sampler instance to check
    expected_draws : int, optional
        Expected number of draws
    expected_chains : int, optional
        Expected number of chains
    expected_tune : int, optional
        Expected number of tuning steps

    Raises
    ------
    AssertionError
        If any configuration value doesn't match the expected value
    """
    if expected_draws is not None:
        actual_draws = sampler.mcmc_config.get("draws", 1000)
        assert (
            actual_draws == expected_draws
        ), f"Expected draws={expected_draws}, got {actual_draws}"

    if expected_chains is not None:
        actual_chains = sampler.mcmc_config.get("chains", 2)
        assert (
            actual_chains == expected_chains
        ), f"Expected chains={expected_chains}, got {actual_chains}"

    if expected_tune is not None:
        actual_tune = sampler.mcmc_config.get("tune", 500)
        assert (
            actual_tune == expected_tune
        ), f"Expected tune={expected_tune}, got {actual_tune}"


def create_mock_trace(chains=2, draws=1000, parameters=None):
    """Create a mock MCMC trace object for testing.

    Parameters
    ----------
    chains : int, optional
        Number of chains in the trace (default: 2)
    draws : int, optional
        Number of draws per chain (default: 1000)
    parameters : list, optional
        List of parameter names (default: ["D0", "alpha", "D_offset"])

    Returns
    -------
    Mock
        Mock trace object with proper structure for ArviZ InferenceData
    """
    if parameters is None:
        parameters = ["D0", "alpha", "D_offset"]

    mock_trace = Mock()
    mock_trace.posterior = Mock()
    mock_trace.posterior.sizes = {"chain": chains, "draw": draws}

    # Mock parameter data
    mock_trace.posterior.data_vars = {}
    for param in parameters:
        mock_trace.posterior.data_vars[param] = Mock()

    return mock_trace


def validate_trace_dimensions(trace, expected_chains, expected_draws):
    """Validate that a trace has the expected dimensions.

    Parameters
    ----------
    trace : Mock or ArviZ InferenceData
        The trace object to validate
    expected_chains : int
        Expected number of chains
    expected_draws : int
        Expected number of draws

    Returns
    -------
    bool
        True if dimensions match, False otherwise
    """
    try:
        chain_count = trace.posterior.sizes.get("chain", None)
        draw_count = trace.posterior.sizes.get("draw", None)

        return chain_count == expected_chains and draw_count == expected_draws
    except AttributeError:
        return False


def create_config_with_wrong_location():
    """Create a configuration with MCMC settings in wrong locations for testing.

    This simulates the old (incorrect) configuration structure that caused
    the original issue where MCMC settings were at the root level instead
    of nested under optimization_config.mcmc_sampling.

    Returns
    -------
    dict
        Configuration with MCMC settings in wrong locations
    """
    return {
        # Wrong location - at root level (old approach)
        "mcmc_draws": 999,
        "mcmc_chains": 99,
        "mcmc_tune": 999,
        # Correct location but empty (should be used)
        "optimization_config": {"mcmc_sampling": {}},
        "initial_parameters": {
            "parameter_names": ["D0", "alpha", "D_offset"],
            "values": [1000, -0.5, 100],
        },
        "parameter_space": {"bounds": []},
    }


def create_realistic_user_config():
    """Create a realistic user configuration matching the actual issue scenario.

    This creates a configuration similar to my_config_simon.json that caused
    the original issue.

    Returns
    -------
    dict
        Realistic configuration with proper MCMC settings
    """
    return {
        "metadata": {
            "config_version": "6.0",
            "analysis_mode": "static_isotropic",
        },
        "optimization_config": {
            "mcmc_sampling": {
                "enabled": True,
                "sampler": "NUTS",
                "draws": 10000,
                "tune": 1000,
                "chains": 8,
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
                {"name": "D0", "min": 1.0, "max": 1000000, "type": "Normal"},
                {"name": "alpha", "min": -1.6, "max": -1.5, "type": "Normal"},
                {
                    "name": "D_offset",
                    "min": -100,
                    "max": 100,
                    "type": "Normal",
                },
                {
                    "name": "gamma_dot_t0",
                    "min": 0.0,
                    "max": 0.0,
                    "type": "fixed",
                },
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
        "analysis_settings": {
            "static_mode": True,
            "static_submode": "isotropic",
        },
        "analyzer_parameters": {
            "temporal": {"dt": 0.5, "start_frame": 400, "end_frame": 1000},
            "scattering": {"wavevector_q": 0.0237},
        },
    }


def get_mcmc_defaults():
    """Get the default MCMC configuration values used in the code.

    Returns
    -------
    dict
        Dictionary with default MCMC configuration values
    """
    return {
        "draws": 1000,
        "tune": 500,
        "chains": 2,
        "target_accept": 0.9,
        "cores": 1,
    }
