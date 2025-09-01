"""
Regression Test Configuration
=============================

Configuration for regression tests.
"""

import pytest


def pytest_collection_modifyitems(items):
    """Automatically mark all tests in regression directory."""
    for item in items:
        # Add regression marker to all tests in this directory
        item.add_marker(pytest.mark.regression)
        # Regression tests are usually fast
        if not any(m.name == "slow" for m in item.iter_markers()):
            item.add_marker(pytest.mark.fast)