"""
PyTest Configuration and Shared Fixtures
========================================

Centralized configuration and fixtures for the homodyne test suite.
"""

import json
import os
import tempfile

import numpy as np
import pytest


# Test markers for categorizing tests
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "scientific: marks tests as scientific validation tests"
    )
    config.addinivalue_line(
        "markers", "cli: marks tests as CLI tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU acceleration"
    )
    config.addinivalue_line(
        "markers", "distributed: marks tests for distributed computing"
    )


# Skip certain tests based on available dependencies
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on available dependencies."""
    # Check for optional dependencies
    skip_numba = pytest.mark.skip(reason="Numba not available")
    skip_cvxpy = pytest.mark.skip(reason="CVXPY not available")
    skip_scipy = pytest.mark.skip(reason="SciPy not available")

    try:
        numba_available = True
    except ImportError:
        numba_available = False

    try:
        cvxpy_available = True
    except ImportError:
        cvxpy_available = False

    try:
        scipy_available = True
    except ImportError:
        scipy_available = False

    for item in items:
        # Skip tests requiring Numba if not available
        if "numba" in item.nodeid.lower() and not numba_available:
            item.add_marker(skip_numba)

        # Skip robust optimization tests if CVXPY not available
        if "robust" in item.nodeid.lower() and not cvxpy_available:
            item.add_marker(skip_cvxpy)

        # Skip scientific tests if SciPy not available
        if "scientific" in item.nodeid.lower() and not scipy_available:
            item.add_marker(skip_scipy)


@pytest.fixture(scope="session")
def temp_directory():
    """Create a temporary directory for the test session."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "experimental_parameters": {
            "q_value": 0.1,
            "contrast": 0.95,
            "offset": 1.0,
            "pixel_size": 172e-6,
            "detector_distance": 8.0,
            "x_ray_energy": 7.35,
            "sample_thickness": 1.0
        },
        "analysis_parameters": {
            "mode": "laminar_flow",
            "method": "classical",
            "enable_angle_filtering": True,
            "chi_squared_threshold": 2.0,
            "max_iterations": 1000,
            "tolerance": 1e-6
        },
        "parameter_bounds": {
            "D0": [1e-6, 1e-1],
            "alpha": [0.1, 2.0],
            "D_offset": [1e-8, 1e-3],
            "gamma0": [1e-4, 1.0],
            "beta": [0.1, 2.0],
            "gamma_offset": [1e-6, 1e-1],
            "phi0": [-180, 180]
        },
        "initial_guesses": {
            "D0": 1e-3,
            "alpha": 0.9,
            "D_offset": 1e-4,
            "gamma0": 0.01,
            "beta": 0.8,
            "gamma_offset": 0.001,
            "phi0": 0.0
        },
        "output_settings": {
            "save_plots": False,  # Disable for testing
            "save_results": True,
            "output_directory": "./test_results"
        }
    }


@pytest.fixture
def static_config():
    """Configuration for static (non-flow) analysis."""
    return {
        "experimental_parameters": {
            "q_value": 0.08,
            "contrast": 0.92,
            "offset": 1.0
        },
        "analysis_parameters": {
            "mode": "static_isotropic",
            "method": "classical",
            "enable_angle_filtering": False,
            "max_iterations": 500,
            "tolerance": 1e-5
        },
        "parameter_bounds": {
            "D0": [1e-5, 1e-2],
            "alpha": [0.2, 1.8],
            "D_offset": [1e-7, 1e-4]
        }
    }


@pytest.fixture
def sample_data_small():
    """Small sample dataset for testing."""
    np.random.seed(42)  # For reproducibility

    # Small dataset for fast testing
    angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
    t1_array = np.array([0.5, 1.0, 1.5])
    t2_array = np.array([1.0, 1.5, 2.0])

    # Generate realistic correlation data
    n_angles = len(angles)
    n_t1 = len(t1_array)
    n_t2 = len(t2_array)

    c2_data = np.ones((n_angles, n_t1, n_t2))

    for i, angle in enumerate(angles):
        for j, t1 in enumerate(t1_array):
            for k, t2 in enumerate(t2_array):
                dt = abs(t2 - t1)
                # Realistic correlation with angular dependence
                correlation = 0.9 * np.exp(-0.1 * dt) * (1 + 0.1 * np.cos(2*angle))
                noise = 0.01 * np.random.randn()
                c2_data[i, j, k] = 1.0 + correlation + noise

    return {
        'c2_data': c2_data,
        'angles': angles,
        't1_array': t1_array,
        't2_array': t2_array
    }


@pytest.fixture
def sample_data_medium():
    """Medium-sized sample dataset for testing."""
    np.random.seed(123)

    # Medium dataset for comprehensive testing
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    t1_array = np.linspace(0.5, 5.0, 10)
    t2_array = np.linspace(1.0, 5.5, 10)

    n_angles = len(angles)
    n_t1 = len(t1_array)
    n_t2 = len(t2_array)

    c2_data = np.ones((n_angles, n_t1, n_t2))

    for i, angle in enumerate(angles):
        for j, t1 in enumerate(t1_array):
            for k, t2 in enumerate(t2_array):
                dt = abs(t2 - t1)
                # More complex correlation structure
                correlation = 0.95 * np.exp(-0.05 * dt**0.9) * (1 + 0.15 * np.cos(2*angle + 0.1))
                noise = 0.005 * np.random.randn()
                c2_data[i, j, k] = 1.0 + correlation + noise

    return {
        'c2_data': c2_data,
        'angles': angles,
        't1_array': t1_array,
        't2_array': t2_array
    }


@pytest.fixture
def noisy_data():
    """Dataset with outliers for robust optimization testing."""
    np.random.seed(456)

    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    t1_array = np.array([1.0, 2.0, 3.0, 4.0])
    t2_array = np.array([1.5, 2.5, 3.5, 4.5])

    n_angles = len(angles)
    n_t1 = len(t1_array)
    n_t2 = len(t2_array)

    c2_data = np.ones((n_angles, n_t1, n_t2))

    for i, angle in enumerate(angles):
        for j, t1 in enumerate(t1_array):
            for k, t2 in enumerate(t2_array):
                dt = abs(t2 - t1)
                correlation = 0.9 * np.exp(-0.08 * dt) * (1 + 0.1 * np.cos(2*angle))

                # Add normal noise
                noise = 0.02 * np.random.randn()

                # Add occasional outliers
                if np.random.random() < 0.1:  # 10% outliers
                    noise += 0.3 * np.random.randn()

                c2_data[i, j, k] = 1.0 + correlation + noise

    return {
        'c2_data': c2_data,
        'angles': angles,
        't1_array': t1_array,
        't2_array': t2_array
    }


@pytest.fixture
def config_file(temp_directory, sample_config):
    """Create a temporary configuration file."""
    config_path = os.path.join(temp_directory, "test_config.json")
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    return config_path


@pytest.fixture
def data_file(temp_directory, sample_data_small):
    """Create a temporary data file."""
    data_path = os.path.join(temp_directory, "test_data.npz")
    np.savez(data_path, **sample_data_small)
    return data_path


@pytest.fixture
def realistic_parameters():
    """Realistic physical parameters for testing."""
    return {
        'D0': 1e-3,      # Å²/s
        'alpha': 0.9,    # dimensionless
        'D_offset': 1e-4, # Å²/s
        'gamma0': 0.01,  # s⁻¹
        'beta': 0.8,     # dimensionless
        'gamma_offset': 0.001, # s⁻¹
        'phi0': 0.0      # radians
    }


@pytest.fixture
def extreme_parameters():
    """Extreme but valid parameters for stress testing."""
    return [
        # Very small diffusion
        {'D0': 1e-10, 'alpha': 0.9, 'D_offset': 1e-12, 'gamma0': 1e-6, 'beta': 0.8, 'gamma_offset': 1e-8, 'phi0': 0.0},
        # Very large diffusion
        {'D0': 1e-1, 'alpha': 0.9, 'D_offset': 1e-3, 'gamma0': 1e-1, 'beta': 0.8, 'gamma_offset': 1e-3, 'phi0': 0.0},
        # Extreme time dependencies
        {'D0': 1e-3, 'alpha': 0.1, 'D_offset': 1e-4, 'gamma0': 0.01, 'beta': 0.1, 'gamma_offset': 0.001, 'phi0': 0.0},
        {'D0': 1e-3, 'alpha': 1.9, 'D_offset': 1e-4, 'gamma0': 0.01, 'beta': 1.9, 'gamma_offset': 0.001, 'phi0': 0.0},
    ]


@pytest.fixture
def mock_optimization_result():
    """Mock optimization result for testing."""
    from unittest.mock import Mock

    result = Mock()
    result.x = [1e-3, 0.9, 1e-4, 0.01, 0.8, 0.001, 0.0]
    result.fun = 0.5
    result.success = True
    result.nit = 100
    result.message = "Optimization terminated successfully"

    return result


@pytest.fixture(scope="session")
def performance_baseline():
    """Performance baselines for regression testing."""
    return {
        'small_data_optimization': 0.1,     # seconds
        'medium_data_optimization': 0.5,    # seconds
        'chi_squared_calculation': 0.01,    # seconds
        'g1_correlation_single': 1e-5,      # seconds
        'memory_usage_mb': 100               # MB
    }


@pytest.fixture
def security_test_strings():
    """Potentially dangerous strings for security testing."""
    return [
        "'; DROP TABLE users; --",  # SQL injection
        "<script>alert('xss')</script>",  # XSS
        "../../etc/passwd",  # Path traversal
        "$(rm -rf /)",  # Command injection
        "\x00\x01\x02",  # Binary data
        "javascript:alert('xss')",  # JavaScript protocol
        "data:text/html,<script>alert('xss')</script>",  # Data URL
        "../../../../../windows/system32/config/sam",  # Windows path traversal
    ]


@pytest.fixture
def test_file_paths():
    """Various file paths for testing."""
    return {
        'valid': [
            "/tmp/test_data.npz",
            "./data/experiment.json",
            "results/output.txt",
            "subdir/config.json"
        ],
        'dangerous': [
            "../../../etc/passwd",
            "/dev/null",
            "//server/share/file",
            "file:///etc/passwd",
            "C:\\Windows\\System32\\config\\SAM",
            "/proc/self/mem"
        ]
    }


@pytest.fixture
def numerical_test_cases():
    """Numerical test cases for validation."""
    return {
        'sinc_values': {
            0.0: 1.0,
            np.pi: 0.0,
            np.pi/2: (2.0/np.pi)**2,
            2*np.pi: 0.0
        },
        'time_points': np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0]),
        'q_values': np.array([0.01, 0.05, 0.1, 0.2, 0.5]),
        'angles': np.linspace(0, 2*np.pi, 16, endpoint=False)
    }


# Custom pytest plugins for test organization
class TestCategories:
    """Constants for test categories."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    SCIENTIFIC = "scientific"
    CLI = "cli"


# Helper functions for test setup
def create_test_environment(temp_dir):
    """Create a complete test environment."""
    # Create necessary subdirectories
    subdirs = ['data', 'config', 'results', 'logs']
    for subdir in subdirs:
        os.makedirs(os.path.join(temp_dir, subdir), exist_ok=True)

    return {
        'data_dir': os.path.join(temp_dir, 'data'),
        'config_dir': os.path.join(temp_dir, 'config'),
        'results_dir': os.path.join(temp_dir, 'results'),
        'logs_dir': os.path.join(temp_dir, 'logs')
    }


def generate_synthetic_xpcs_data(n_angles=8, n_times=10, seed=42):
    """Generate synthetic XPCS data for testing."""
    np.random.seed(seed)

    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    t1_array = np.linspace(0.5, 5.0, n_times)
    t2_array = np.linspace(1.0, 5.5, n_times)

    c2_data = np.ones((n_angles, n_times, n_times))

    # Parameters for synthetic data
    D0, alpha, D_offset = 1e-3, 0.9, 1e-4
    gamma0, beta, gamma_offset = 0.02, 0.8, 0.002

    for i, angle in enumerate(angles):
        for j, t1 in enumerate(t1_array):
            for k, t2 in enumerate(t2_array):
                dt = abs(t2 - t1)

                # Diffusion contribution
                D_integral = D0 * (dt**(alpha + 1)) / (alpha + 1) + D_offset * dt

                # Shear contribution (simplified)
                shear_phase = gamma0 * (dt**(beta + 1)) / (beta + 1) + gamma_offset * dt
                shear_factor = np.sinc(0.1 * shear_phase * np.cos(angle))**2

                # Combined correlation
                g1 = np.exp(-0.05 * D_integral) * shear_factor
                g2 = 1.0 + 0.95 * g1**2

                # Add realistic noise
                noise = 0.01 * np.random.randn()
                c2_data[i, j, k] = g2 + noise

    return {
        'c2_data': c2_data,
        'angles': angles,
        't1_array': t1_array,
        't2_array': t2_array
    }


# Pytest hooks for custom test execution
def pytest_runtest_setup(item):
    """Setup for individual test runs."""
    # Check if test requires special setup
    if hasattr(item, 'get_closest_marker'):
        # Performance tests need clean environment
        if item.get_closest_marker('performance'):
            import gc
            gc.collect()

        # GPU tests disabled for CPU-only optimization
        if item.get_closest_marker('gpu'):
            pytest.skip("GPU tests disabled - CPU-only configuration")


def pytest_runtest_teardown(item, nextitem):
    """Cleanup after individual test runs."""
    # Clean up after performance tests
    if hasattr(item, 'get_closest_marker'):
        if item.get_closest_marker('performance'):
            import gc
            gc.collect()


# Custom assertions for scientific computing
def assert_physical_bounds(value, lower=None, upper=None, name="value"):
    """Assert that a value is within physical bounds."""
    assert np.isfinite(value), f"{name} is not finite: {value}"

    if lower is not None:
        assert value >= lower, f"{name} below physical bound: {value} < {lower}"

    if upper is not None:
        assert value <= upper, f"{name} above physical bound: {value} > {upper}"


def assert_correlation_properties(g1, g2=None, contrast=None, offset=None):
    """Assert that correlation functions have correct properties."""
    # g1 should be between 0 and 1
    assert_physical_bounds(g1, 0.0, 1.0, "g1 correlation")

    # If g2 provided, check Siegert relation
    if g2 is not None and contrast is not None and offset is not None:
        expected_g2 = offset + contrast * g1**2
        np.testing.assert_allclose(g2, expected_g2, rtol=1e-10,
                                  err_msg="Siegert relation violated")

    # g2 should be >= 1
    if g2 is not None:
        assert g2 >= 1.0, f"g2 correlation violates lower bound: {g2}"


def assert_monotonic_decay(values, tolerance=1e-10):
    """Assert that values show monotonic decay."""
    for i in range(len(values) - 1):
        assert values[i] >= values[i+1] - tolerance, \
            f"Non-monotonic behavior at index {i}: {values[i]} -> {values[i+1]}"


# Performance measurement utilities
class PerformanceTimer:
    """Context manager for measuring execution time."""

    def __init__(self, name="operation"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.perf_counter()

    @property
    def elapsed_time(self):
        """Get elapsed time in seconds."""
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time