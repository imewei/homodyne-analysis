"""
System Test Configuration
=========================

Configuration for system-level tests including CLI, GPU, and installation.
"""

import os

import pytest


def pytest_collection_modifyitems(items):
    """Automatically mark all tests in system directory."""
    for item in items:
        # Add system marker to all tests in this directory
        item.add_marker(pytest.mark.system)

        # System tests are usually slower
        if not any(m.name == "fast" for m in item.iter_markers()):
            item.add_marker(pytest.mark.slow)


@pytest.fixture
def system_environment():
    """Setup system test environment."""
    original_env = os.environ.copy()

    # Set up test environment variables
    os.environ["HOMODYNE_TEST_MODE"] = "1"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_system_paths(tmp_path):
    """Mock system paths for installation tests."""
    return {
        "venv": tmp_path / "test_venv",
        "bin": tmp_path / "test_venv" / "bin",
        "etc": tmp_path / "test_venv" / "etc",
        "lib": tmp_path / "test_venv" / "lib",
    }
