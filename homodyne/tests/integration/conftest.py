"""
Integration Test Configuration
==============================

Configuration for integration tests that combine multiple components.
"""

import tempfile
from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    """Automatically mark all tests in integration directory."""
    for item in items:
        # Add integration marker to all tests in this directory
        item.add_marker(pytest.mark.integration)


@pytest.fixture
def integration_temp_dir():
    """Provide a temporary directory for integration tests."""
    with tempfile.TemporaryDirectory(prefix="homodyne_integration_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def integration_config(integration_temp_dir):
    """Provide a test configuration for integration tests."""
    import json

    config_path = integration_temp_dir / "test_config.json"
    config_data = {
        "analysis_settings": {
            "static_mode": True,
            "chunk_size": 100,
            "use_numba": True,
        },
        "output": {
            "directory": str(integration_temp_dir / "output"),
            "save_plots": True,
        },
    }

    with open(config_path, "w") as f:
        json.dump(config_data, f)

    return config_path
