"""
Test Fixtures Module
====================

Central location for all test fixtures and utilities.
"""

# Re-export all fixtures from data module for backward compatibility
from homodyne.tests.fixtures.data import (
    create_invalid_config_file,
    create_minimal_config_file,
    dummy_analysis_results,
    dummy_config,
    dummy_correlation_data,
    dummy_hdf5_data,
    dummy_phi_angles,
    dummy_theoretical_data,
    dummy_time_arrays,
    mock_optimization_result,
    test_output_directory,
)

__all__ = [
    "dummy_analysis_results",
    "dummy_config",
    "dummy_correlation_data",
    "dummy_hdf5_data",
    "dummy_phi_angles",
    "dummy_theoretical_data",
    "dummy_time_arrays",
    "mock_optimization_result",
    "test_output_directory",
    "create_minimal_config_file",
    "create_invalid_config_file",
]
