"""
Unit Test Configuration
=======================

Configuration for fast, isolated unit tests.
"""

import pytest


def pytest_collection_modifyitems(items):
    """Automatically mark all tests in unit directory."""
    for item in items:
        # Add unit marker to all tests in this directory
        item.add_marker(pytest.mark.unit)

        # Most unit tests should be fast
        if not any(m.name == "slow" for m in item.iter_markers()):
            item.add_marker(pytest.mark.fast)

        # Add ci marker for tests that should run in CI (equivalent to "not slow and not integration and not mcmc")
        item.add_marker(pytest.mark.ci)


@pytest.fixture(autouse=True)
def fast_mode():
    """Ensure unit tests run in fast mode."""
    import os

    # Disable any slow features for unit tests
    os.environ["HOMODYNE_FAST_TESTS"] = "1"
    yield
    os.environ.pop("HOMODYNE_FAST_TESTS", None)
