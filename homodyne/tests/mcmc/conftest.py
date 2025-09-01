"""
MCMC Test Configuration
=======================

Configuration for MCMC-specific tests.
"""

import numpy as np
import pytest


def pytest_collection_modifyitems(items):
    """Automatically mark all tests in MCMC directory."""
    for item in items:
        # Add MCMC marker to all tests in this directory
        item.add_marker(pytest.mark.mcmc)

        # MCMC tests are typically slow
        if not any(m.name == "fast" for m in item.iter_markers()):
            item.add_marker(pytest.mark.slow)


@pytest.fixture
def mcmc_test_data():
    """Generate test data for MCMC tests."""
    np.random.seed(42)
    n_angles = 20
    n_times = 100

    return {
        "angles": np.linspace(0, np.pi, n_angles),
        "times": np.logspace(-3, 2, n_times),
        "data": np.random.exponential(scale=1.0, size=(n_angles, n_times)),
        "errors": np.random.uniform(0.01, 0.1, size=(n_angles, n_times)),
    }


@pytest.fixture
def mcmc_config():
    """Provide MCMC configuration for tests."""
    return {
        "n_walkers": 10,
        "n_steps": 100,
        "burn_in": 50,
        "thin": 1,
        "parameters": {
            "beta": {"initial": 0.5, "bounds": [0.1, 1.0]},
            "gamma": {"initial": 1.0, "bounds": [0.5, 2.0]},
            "baseline": {"initial": 0.1, "bounds": [0.0, 1.0]},
        },
    }


@pytest.fixture
def skip_if_no_pymc(request):
    """Skip test if PyMC is not available."""
    try:
        import pymc
    except ImportError:
        pytest.skip("PyMC not available")
