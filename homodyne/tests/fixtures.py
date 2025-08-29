"""
Test Fixtures for Rheo-SAXS-XPCS Tests
=====================================

Provides dummy datasets and configurations for unit and integration tests.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_output_directory():
    """Create a homodyne_results directory marked as a test artifact.

    This fixture creates a homodyne_results directory in the current working
    directory and marks it as a test artifact so it will be cleaned up safely.
    Use this instead of manually creating homodyne_results in tests.

    SAFETY: If a homodyne_results directory already exists (e.g., user data),
    this fixture will NOT mark it as a test artifact. It will only mark
    directories that it creates itself.
    """
    from pathlib import Path

    # Avoid circular import by importing inside function
    def mark_directory_as_test_artifact(directory_path):
        """Mark directory as test artifact (fallback implementation)."""
        # Ensure the directory exists before writing the marker file
        directory_path.mkdir(parents=True, exist_ok=True)
        marker_file = directory_path / ".test-artifact"
        marker_file.write_text(
            "This directory was created by tests and can be safely removed."
        )

    # Create homodyne_results directory in current directory
    test_dir = Path.cwd() / "homodyne_results"

    # SAFETY CHECK: Only mark as test artifact if we create it
    # If it already exists, assume it contains user data and preserve it
    directory_existed_before = test_dir.exists()

    if not directory_existed_before:
        # Directory doesn't exist - safe to create and mark as test artifact
        mark_directory_as_test_artifact(test_dir)
    else:
        # Directory already exists - preserve it, don't mark as test artifact
        # Just make sure it exists (it should already)
        test_dir.mkdir(exist_ok=True)

    yield test_dir

    # No manual cleanup needed - conftest.py will handle it
    # (but only if we marked it as a test artifact)


@pytest.fixture
def dummy_config():
    """Generate a minimal test configuration."""
    return {
        "metadata": {
            "config_version": "test-1.0",
            "description": "Test configuration for unit tests",
            "created_date": "2024-01-01",
        },
        "experimental_data": {
            "data_folder_path": "./test_data/",
            "data_file_name": "test_data.hdf",
            "phi_angles_path": "./test_data/",
            "phi_angles_file": "test_phi_list.txt",
            "exchange_key": "exchange",
            "cache_file_path": "./test_cache/",
            "cache_filename_template": "test_c2_{start_frame}_{end_frame}.npz",
            "cache_compression": True,
        },
        "analyzer_parameters": {
            "temporal": {
                "dt": 0.1,
                "start_frame": 10,
                "end_frame": 50,
                "frame_description": "Small test window",
            },
            "scattering": {"wavevector_q": 0.01, "q_unit": "Å⁻¹"},
            "geometry": {"stator_rotor_gap": 200000, "gap_unit": "Å"},
            "computational": {
                "num_threads": 2,
                "auto_detect_cores": False,
                "max_threads_limit": 4,
            },
        },
        "initial_parameters": {
            "values": [100.0, -0.1, 50.0, 0.001, -0.2, 0.0, 0.0],
            "parameter_names": [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_t0",
                "beta",
                "gamma_dot_t_offset",
                "phi0",
            ],
            "units": [
                "Å²/s",
                "dimensionless",
                "Å²/s",
                "s⁻¹",
                "dimensionless",
                "s⁻¹",
                "degrees",
            ],
        },
        "parameter_space": {
            "bounds": [
                {
                    "name": "D0",
                    "min": 1.0,
                    "max": 1000000,
                    "type": "Normal",
                    "unit": "Å²/s",
                },
                {
                    "name": "alpha",
                    "min": -2.0,
                    "max": 2.0,
                    "type": "Normal",
                    "unit": "dimensionless",
                },
                {
                    "name": "D_offset",
                    "min": -100.0,
                    "max": 100.0,
                    "type": "Normal",
                    "unit": "Å²/s",
                },
                {
                    "name": "gamma_dot_t0",
                    "min": 1e-6,
                    "max": 1.0,
                    "type": "Normal",
                    "unit": "s⁻¹",
                },
                {
                    "name": "beta",
                    "min": -2.0,
                    "max": 2.0,
                    "type": "Normal",
                    "unit": "dimensionless",
                },
                {
                    "name": "gamma_dot_t_offset",
                    "min": -1e-2,
                    "max": 1e-2,
                    "type": "Normal",
                    "unit": "s⁻¹",
                },
                {
                    "name": "phi0",
                    "min": -10.0,
                    "max": 10.0,
                    "type": "Normal",
                    "unit": "degrees",
                },
            ]
        },
        "optimization_config": {
            "classical_optimization": {
                "methods": ["Nelder-Mead"],
                "method_options": {
                    "Nelder-Mead": {
                        "maxiter": 10,
                        "xatol": 1e-3,
                        "fatol": 1e-3,
                    }
                },
            },
            "mcmc_sampling": {
                "enabled": False,
                "draws": 10,
                "tune": 5,
                "chains": 1,
            },
            "scaling_parameters": {
                "fitted_range": {"min": 1.0, "max": 2.0},
                "theory_range": {"min": 0.0, "max": 1.0},
                "contrast": {
                    "min": 0.05,
                    "max": 0.5,
                    "prior_mu": 0.3,
                    "prior_sigma": 0.1,
                    "type": "TruncatedNormal",
                },
                "offset": {
                    "min": 0.05,
                    "max": 1.95,
                    "prior_mu": 1.0,
                    "prior_sigma": 0.2,
                    "type": "TruncatedNormal",
                },
            },
        },
        "performance_settings": {
            "jit_compilation": {"use_numba": False, "warmup_numba": False},
            "parallelization": {
                "parallel_execution": False,
                "use_threading": False,
            },
            "caching": {"memory_efficient_cache_maxsize": 8},
        },
        "output_settings": {
            "results_directory": "./test_results",
            "file_naming": {
                "timestamp_format": "%Y%m%d_%H%M%S",
                "include_chi_squared": True,
                "include_config_name": True,
            },
            "plotting": {
                "create_plots": True,
                "plot_format": "png",
                "dpi": 100,
                "figure_size": [6, 4],
            },
        },
        "validation_rules": {
            "frame_range": {"minimum_frames": 5, "maximum_frames": 1000},
            "fit_quality": {
                "overall_chi_squared": {
                    "excellent_threshold": 2.0,
                    "acceptable_threshold": 5.0,
                    "warning_threshold": 10.0,
                    "critical_threshold": 20.0,
                },
                "per_angle_chi_squared": {
                    "excellent_threshold": 2.0,
                    "acceptable_threshold": 5.0,
                    "warning_threshold": 10.0,
                    "outlier_threshold_multiplier": 2.5,
                    "max_outlier_fraction": 0.25,
                    "min_good_angles": 3,
                },
            },
        },
        "logging": {
            "log_to_file": False,
            "log_to_console": False,
            "level": "ERROR",
        },
    }


@pytest.fixture
def dummy_correlation_data():
    """Generate synthetic correlation data for testing."""
    np.random.seed(42)  # Reproducible data

    n_angles = 3
    n_t2 = 20
    n_t1 = 30

    # Create realistic-looking correlation data
    # Start with exponential decay plus noise
    t2 = np.linspace(0.1, 2.0, n_t2)
    t1 = np.linspace(0.1, 3.0, n_t1)

    experimental_data = np.zeros((n_angles, n_t2, n_t1))

    for i in range(n_angles):
        for j, tau in enumerate(t2):
            for k, t in enumerate(t1):
                # Base correlation with exponential decay
                base_corr = 1.0 + 0.8 * np.exp(-tau / 1.0 - t / 2.0)
                # Add some angle dependence
                angle_factor = 1 + 0.2 * np.cos(i * np.pi / 3)
                # Add realistic noise
                noise = 0.05 * np.random.normal()
                experimental_data[i, j, k] = base_corr * angle_factor + noise

    return experimental_data


@pytest.fixture
def dummy_phi_angles():
    """Generate test phi angles."""
    return np.array([0.0, 45.0, 90.0])


@pytest.fixture
def dummy_time_arrays():
    """Generate time lag and delay time arrays."""
    t2 = np.linspace(0.1, 2.0, 20)
    t1 = np.linspace(0.1, 3.0, 30)
    return t2, t1


@pytest.fixture
def dummy_theoretical_data(dummy_correlation_data):
    """Generate theoretical correlation data (similar to experimental but slightly different)."""
    # Add small systematic difference to test residuals
    theoretical = dummy_correlation_data.copy()
    theoretical += 0.02 * np.random.normal(size=theoretical.shape)
    return theoretical


@pytest.fixture
def dummy_analysis_results(
    dummy_config,
    dummy_correlation_data,
    dummy_theoretical_data,
    dummy_phi_angles,
):
    """Generate complete analysis results for testing."""
    return {
        "config": dummy_config,
        "experimental_data": dummy_correlation_data,
        "theoretical_data": dummy_theoretical_data,
        "phi_angles": dummy_phi_angles,
        "best_parameters": {
            "D0": 123.45,
            "alpha": -0.123,
            "D_offset": 12.34,
            "gamma_dot_t0": 0.00123,
            "beta": -0.234,
            "gamma_dot_t_offset": 0.001,
            "phi0": 0.0,
        },
        "best_chi_squared": 1.234,
        "parameter_bounds": dummy_config["parameter_space"]["bounds"],
        "parameter_names": dummy_config["initial_parameters"]["parameter_names"],
        "parameter_units": dummy_config["initial_parameters"]["units"],
        "initial_parameters": dict(
            zip(
                dummy_config["initial_parameters"]["parameter_names"],
                dummy_config["initial_parameters"]["values"],
                strict=False,
            )
        ),
        "optimization_history": [
            {"chi_squared": 2.5, "iteration": 1},
            {"chi_squared": 1.8, "iteration": 2},
            {"chi_squared": 1.4, "iteration": 3},
            {"chi_squared": 1.234, "iteration": 4},
        ],
        "residuals": dummy_correlation_data - dummy_theoretical_data,
        "parameter_uncertainties": {
            "D0": 5.67,
            "alpha": 0.01,
            "D_offset": 2.34,
            "gamma_dot_t0": 0.0001,
            "beta": 0.02,
            "gamma_dot_t_offset": 0.0002,
            "phi0": 0.0,
        },
    }


@pytest.fixture
def dummy_hdf5_data(temp_directory):
    """Create a dummy HDF5-like data structure for testing data loading."""
    # This would normally create an actual HDF5 file, but for testing
    # we'll use a simple numpy array structure that mimics the expected format

    data_dir = temp_directory / "test_data"
    data_dir.mkdir(exist_ok=True)

    # Create dummy correlation data
    np.random.seed(42)
    correlation_data = np.random.exponential(1.0, (100, 50, 50))  # frames x t2 x t1

    # Save as npz for testing (simulating HDF5 structure)
    data_file = data_dir / "test_correlation_data.npz"
    np.savez_compressed(data_file, correlation_data=correlation_data)

    # Create phi angles file
    phi_file = data_dir / "test_phi_list.txt"
    phi_angles = [0.0, 22.5, 45.0, 67.5, 90.0]
    with open(phi_file, "w") as f:
        for angle in phi_angles:
            f.write(f"{angle}\n")

    return {
        "data_path": data_file,
        "phi_path": phi_file,
        "correlation_data": correlation_data,
        "phi_angles": np.array(phi_angles),
    }


def create_minimal_config_file(
    filepath: Path, config: dict[str, Any] | None = None
) -> Path:
    """Create a minimal JSON config file for testing."""
    if config is None:
        config = {
            "metadata": {"config_version": "test-minimal"},
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 10},
                "scattering": {"wavevector_q": 0.01},
                "geometry": {"stator_rotor_gap": 100000},
                "computational": {"num_threads": 1},
            },
            "experimental_data": {
                "data_folder_path": "./",
                "data_file_name": "test.hdf",
                "phi_angles_path": "./",
                "phi_angles_file": "phi.txt",
            },
            "initial_parameters": {
                "values": [1.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0],
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_t0",
                    "beta",
                    "gamma_dot_t_offset",
                    "phi0",
                ],
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
                        "min": -2.0,
                        "max": 2.0,
                        "type": "Normal",
                    },
                    {
                        "name": "D_offset",
                        "min": -100,
                        "max": 100,
                        "type": "Normal",
                    },
                    {
                        "name": "gamma_dot_t0",
                        "min": 1e-6,
                        "max": 1.0,
                        "type": "Normal",
                    },
                    {
                        "name": "beta",
                        "min": -2.0,
                        "max": 2.0,
                        "type": "Normal",
                    },
                    {
                        "name": "gamma_dot_t_offset",
                        "min": -1e-2,
                        "max": 1e-2,
                        "type": "Normal",
                    },
                    {
                        "name": "phi0",
                        "min": -10.0,
                        "max": 10.0,
                        "type": "Normal",
                    },
                ]
            },
            "optimization_config": {
                "classical_optimization": {"methods": ["Nelder-Mead"]}
            },
            "validation_rules": {"frame_range": {"minimum_frames": 5}},
        }

    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)

    return filepath


def create_invalid_config_file(filepath: Path, error_type: str = "syntax") -> Path:
    """Create an invalid JSON config file for testing error handling."""
    if error_type == "syntax":
        # Missing closing brace
        content = '{"metadata": {"version": "test"}, "data": {'
    elif error_type == "missing_required":
        # Missing required sections
        content = '{"metadata": {"version": "test"}}'
    elif error_type == "invalid_values":
        # Invalid parameter values
        content = """
        {
            "analyzer_parameters": {
                "temporal": {
                    "dt": -1.0,
                    "start_frame": 100,
                    "end_frame": 50
                }
            }
        }"""
    else:
        content = '{"invalid": "config"}'

    with open(filepath, "w") as f:
        f.write(content)

    return filepath


@pytest.fixture
def mock_optimization_result():
    """Create a mock optimization result for testing."""

    class MockOptimizeResult:
        def __init__(self):
            self.x = [100.0, -0.1, 10.0, 0.001, -0.2, 0.0, 1.0]
            self.fun = 1.234
            self.success = True
            self.nit = 15
            self.nfev = 150
            self.message = "Optimization terminated successfully"

    return MockOptimizeResult()
