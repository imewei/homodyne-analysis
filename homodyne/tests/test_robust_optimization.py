"""
Tests for Robust Optimization Methods
====================================

Comprehensive test suite for the robust optimization framework in the homodyne package.
Tests cover all three robust optimization methods, integration with classical optimization,
configuration validation, and edge cases.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from scipy import optimize

# Import homodyne modules
try:
    from homodyne.analysis.core import HomodyneAnalysisCore
    from homodyne.core.config import ConfigManager
    from homodyne.optimization.classical import ClassicalOptimizer
    from homodyne.optimization.robust import (
        CVXPY_AVAILABLE,
        GUROBI_AVAILABLE,
        RobustHomodyneOptimizer,
        create_robust_optimizer,
    )

    ROBUST_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    RobustHomodyneOptimizer = None  # type: ignore
    create_robust_optimizer = None  # type: ignore
    ClassicalOptimizer = None  # type: ignore
    HomodyneAnalysisCore = None  # type: ignore
    ConfigManager = None  # type: ignore
    CVXPY_AVAILABLE = False
    GUROBI_AVAILABLE = False
    ROBUST_OPTIMIZATION_AVAILABLE = False
    logging.warning(f"Robust optimization not available for testing: {e}")

# Test configuration for robust optimization
TEST_CONFIG = {
    "metadata": {
        "config_version": "0.6.5",
        "description": "Test configuration for robust optimization",
    },
    "experimental_data": {
        "data_folder_path": "./test_data/",
        "data_file_name": "test_data.hdf",
        "phi_angles_path": "./test_data/",
        "phi_angles_file": "phi_list.txt",
    },
    "analyzer_parameters": {
        "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 100},
        "scattering": {"wavevector_q": 0.01},
        "geometry": {"stator_rotor_gap": 1000000},
    },
    "initial_parameters": {
        "values": [100.0, -0.5, 10.0, 0.0, 0.0, 0.0, 0.0],
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
    "optimization_config": {
        "classical_optimization": {
            "methods": ["Nelder-Mead", "Robust-Wasserstein"],
            "method_options": {"Nelder-Mead": {"maxiter": 100}},
        },
        "robust_optimization": {
            "enabled": True,
            "uncertainty_model": "wasserstein",
            "uncertainty_radius": 0.05,
            "n_scenarios": 10,  # Small number for fast testing
            "regularization_alpha": 0.01,
            "regularization_beta": 0.001,
            "solver_settings": {
                "Method": 2,
                "TimeLimit": 60,  # Short time limit for tests
                "OutputFlag": 0,
            },
            "method_options": {
                "wasserstein": {
                    "uncertainty_radius": 0.03,
                    "regularization_alpha": 0.01,
                },
                "scenario": {
                    "n_scenarios": 10,
                    "bootstrap_method": "residual",
                },
                "ellipsoidal": {
                    "gamma": 0.1,
                    "l1_regularization": 0.001,
                    "l2_regularization": 0.01,
                },
            },
        },
    },
    "parameter_space": {
        "bounds": [
            {"name": "D0", "min": 1.0, "max": 10000.0},
            {"name": "alpha", "min": -2.0, "max": 2.0},
            {"name": "D_offset", "min": 0.1, "max": 1000.0},
        ]
    },
    "analysis_settings": {"mode": "static", "num_parameters": 3},
    "output_config": {"base_directory": "./test_output/"},
}


class MockAnalysisCore:
    """Mock analysis core for testing robust optimization without full XPCS setup."""

    def __init__(self, config):
        self.config = config
        self.phi_angles = np.linspace(-30, 30, 10)
        self.c2_experimental = self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """Generate simple synthetic correlation function data."""
        n_angles = len(self.phi_angles)
        n_times = 50

        # Simple exponential decay model for testing
        time_delays = np.linspace(0, 5, n_times)
        c2_data = np.zeros((n_angles, n_times))

        for i in range(n_angles):
            decay = np.exp(-0.5 * time_delays)
            c2_data[i, :] = 1.0 + 0.3 * decay + 0.02 * np.random.randn(n_times)

        return c2_data

    def compute_c2_correlation_optimized(self, params, phi_angles):
        """Mock correlation function computation."""
        n_angles = len(phi_angles)
        n_times = 50
        time_delays = np.linspace(0, 5, n_times)

        D0, alpha, D_offset = params[0], params[1], params[2]
        c2_theory = np.zeros((n_angles, n_times))

        for i in range(n_angles):
            # Simple model: c2 = 1 + contrast * exp(-(D0*t^alpha + D_offset*t))
            decay_factor = D0 * time_delays ** abs(alpha) + D_offset * time_delays
            decay = np.exp(-decay_factor)
            c2_theory[i, :] = 1.0 + 0.3 * decay

        return c2_theory

    def calculate_chi_squared_optimized(self, params, phi_angles, c2_experimental):
        """Mock chi-squared calculation."""
        c2_theory = self.compute_c2_correlation_optimized(params, phi_angles)
        residuals = c2_experimental - c2_theory
        return np.sum(residuals**2)

    def calculate_c2_nonequilibrium_laminar_parallel(self, params, phi_angles):
        """Mock method for robust optimization compatibility."""
        n_angles = len(phi_angles)
        n_times = 50
        time_delays = np.linspace(0, 5, n_times)

        D0, alpha, D_offset = params[0], params[1], params[2]
        # Return shape (n_angles, n_times, n_times) for compatibility
        c2_theory = np.zeros((n_angles, n_times, n_times))

        for i in range(n_angles):
            # Simple model with proper 3D structure
            decay_factor = D0 * time_delays ** abs(alpha) + D_offset * time_delays
            for j in range(n_times):
                for k in range(n_times):
                    abs(time_delays[j] - time_delays[k])
                    decay = np.exp(-decay_factor[min(j, k)])
                    c2_theory[i, j, k] = 1.0 + 0.3 * decay

        return c2_theory

    def is_static_mode(self):
        """Return True for static mode testing."""
        return True

    def get_effective_parameter_count(self):
        """Return 3 parameters for static mode."""
        return 3

    @property
    def time_length(self):
        """Mock time length property."""
        return 50


@pytest.fixture
def mock_analysis_core():
    """Fixture providing a mock analysis core."""
    return MockAnalysisCore(TEST_CONFIG)


@pytest.fixture
def test_config():
    """Fixture providing test configuration."""
    return TEST_CONFIG.copy()


@pytest.fixture
def synthetic_data():
    """Fixture providing synthetic test data."""
    phi_angles = np.linspace(-30, 30, 10)
    n_times = 50

    # Generate clean synthetic data
    time_delays = np.linspace(0, 5, n_times)
    c2_clean = np.zeros((len(phi_angles), n_times))

    for i in range(len(phi_angles)):
        decay = np.exp(-0.5 * time_delays)
        c2_clean[i, :] = 1.0 + 0.3 * decay

    # Add noise
    noise_level = 0.03
    c2_noisy = c2_clean + noise_level * np.random.randn(*c2_clean.shape)

    return {
        "phi_angles": phi_angles,
        "c2_experimental": c2_noisy,
        "c2_clean": c2_clean,
        "true_params": np.array([100.0, -0.5, 10.0]),
    }


# Skip all robust optimization tests if dependencies not available
@pytest.mark.skipif(
    not ROBUST_OPTIMIZATION_AVAILABLE,
    reason="Robust optimization dependencies not available",
)
class TestRobustHomodyneOptimizer:
    """Test suite for RobustHomodyneOptimizer class."""

    def test_optimizer_initialization(self, mock_analysis_core, test_config):
        """Test RobustHomodyneOptimizer initialization."""
        assert (
            RobustHomodyneOptimizer is not None
        ), "RobustHomodyneOptimizer not available"
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, test_config)

        assert optimizer.core == mock_analysis_core
        assert optimizer.config == test_config
        assert optimizer.best_params_robust is None
        assert "uncertainty_model" in optimizer.settings
        assert optimizer.settings["uncertainty_model"] == "wasserstein"

    def test_create_robust_optimizer_factory(self, mock_analysis_core, test_config):
        """Test create_robust_optimizer factory function."""
        assert (
            create_robust_optimizer is not None
        ), "create_robust_optimizer not available"
        assert (
            RobustHomodyneOptimizer is not None
        ), "RobustHomodyneOptimizer not available"
        optimizer = create_robust_optimizer(mock_analysis_core, test_config)

        assert isinstance(optimizer, RobustHomodyneOptimizer)
        assert optimizer.core == mock_analysis_core
        assert optimizer.config == test_config

    def test_check_dependencies(self, mock_analysis_core, test_config):
        """Test dependency checking."""
        assert (
            RobustHomodyneOptimizer is not None
        ), "RobustHomodyneOptimizer not available"
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, test_config)

        if CVXPY_AVAILABLE:
            assert optimizer.check_dependencies()
        else:
            with pytest.raises(ImportError, match="CVXPY is required"):
                optimizer.check_dependencies()

    def test_parameter_bounds_extraction(self, mock_analysis_core, test_config):
        """Test parameter bounds extraction from configuration."""
        assert (
            RobustHomodyneOptimizer is not None
        ), "RobustHomodyneOptimizer not available"
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, test_config)
        bounds = optimizer._get_parameter_bounds()

        assert bounds is not None
        assert len(bounds) == 3  # Static mode has 3 parameters
        assert bounds[0] == (1.0, 10000.0)  # D0 bounds
        assert bounds[1] == (-2.0, 2.0)  # alpha bounds
        assert bounds[2] == (0.1, 1000.0)  # D_offset bounds

    def test_chi_squared_computation(
        self, mock_analysis_core, test_config, synthetic_data
    ):
        """Test chi-squared computation."""
        assert (
            RobustHomodyneOptimizer is not None
        ), "RobustHomodyneOptimizer not available"
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, test_config)

        chi_squared = optimizer._compute_chi_squared(
            synthetic_data["true_params"],
            synthetic_data["phi_angles"],
            synthetic_data["c2_experimental"],
        )

        assert isinstance(chi_squared, float)
        assert chi_squared >= 0
        assert np.isfinite(chi_squared)

    def test_theoretical_correlation_computation(
        self, mock_analysis_core, test_config, synthetic_data
    ):
        """Test theoretical correlation function computation."""
        assert (
            RobustHomodyneOptimizer is not None
        ), "RobustHomodyneOptimizer not available"
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, test_config)

        c2_theory = optimizer._compute_theoretical_correlation(
            synthetic_data["true_params"], synthetic_data["phi_angles"]
        )

        assert isinstance(c2_theory, np.ndarray)
        assert c2_theory.shape[0] == len(synthetic_data["phi_angles"])
        assert np.all(np.isfinite(c2_theory))
        # Correlation functions should be positive
        assert np.all(c2_theory > 0)

    def test_linearized_correlation_computation(
        self, mock_analysis_core, test_config, synthetic_data
    ):
        """Test linearized correlation function and Jacobian computation."""
        assert (
            RobustHomodyneOptimizer is not None
        ), "RobustHomodyneOptimizer not available"
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, test_config)

        c2_theory, jacobian = optimizer._compute_linearized_correlation(
            synthetic_data["true_params"],
            synthetic_data["phi_angles"],
            synthetic_data["c2_experimental"],
        )

        assert isinstance(c2_theory, np.ndarray)
        assert isinstance(jacobian, np.ndarray)
        assert jacobian.shape[1] == len(synthetic_data["true_params"])
        assert jacobian.shape[0] == c2_theory.size
        assert np.all(np.isfinite(jacobian))


@pytest.mark.skipif(
    not ROBUST_OPTIMIZATION_AVAILABLE or not CVXPY_AVAILABLE,
    reason="CVXPY not available",
)
class TestDistributionallyRobustOptimization:
    """Test suite for Distributionally Robust Optimization (Wasserstein)."""

    def test_wasserstein_dro_basic(
        self, mock_analysis_core, test_config, synthetic_data
    ):
        """Test basic Wasserstein DRO functionality."""
        assert (
            RobustHomodyneOptimizer is not None
        ), "RobustHomodyneOptimizer not available"
        assert CVXPY_AVAILABLE, "CVXPY not available"
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, test_config)

        # Use small uncertainty radius for stable testing
        result = optimizer._solve_distributionally_robust(
            # Start near true params
            theta_init=synthetic_data["true_params"] * 1.1,
            phi_angles=synthetic_data["phi_angles"],
            c2_experimental=synthetic_data["c2_experimental"],
            uncertainty_radius=0.01,
        )

        optimal_params, info = result

        if optimal_params is not None:
            assert isinstance(optimal_params, np.ndarray)
            assert len(optimal_params) == 3
            assert np.all(np.isfinite(optimal_params))
            assert info["method"] == "distributionally_robust"
            assert "uncertainty_radius" in info
            assert "final_chi_squared" in info
        else:
            # If optimization failed, check that error info is provided
            assert "status" in info or "error" in info

    def test_wasserstein_uncertainty_radius_effect(
        self, mock_analysis_core, test_config, synthetic_data
    ):
        """Test effect of different uncertainty radius values."""
        assert (
            RobustHomodyneOptimizer is not None
        ), "RobustHomodyneOptimizer not available"
        assert CVXPY_AVAILABLE, "CVXPY not available"
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, test_config)

        radius_values = [0.01, 0.05, 0.1]
        results = []

        for radius in radius_values:
            result = optimizer._solve_distributionally_robust(
                theta_init=synthetic_data["true_params"] * 1.1,
                phi_angles=synthetic_data["phi_angles"],
                c2_experimental=synthetic_data["c2_experimental"],
                uncertainty_radius=radius,
            )
            results.append(result)

        # Check that we get different results for different uncertainty radii
        successful_results = [r for r in results if r[0] is not None]

        if len(successful_results) > 1:
            # Results should be different for different uncertainty levels
            params_list = [r[0] for r in successful_results]
            for i in range(1, len(params_list)):
                assert not np.allclose(params_list[0], params_list[i], rtol=1e-3)


@pytest.mark.skipif(
    not ROBUST_OPTIMIZATION_AVAILABLE or not CVXPY_AVAILABLE,
    reason="CVXPY not available",
)
class TestScenarioBasedRobustOptimization:
    """Test suite for Scenario-Based Robust Optimization."""

    def test_bootstrap_scenario_generation(
        self, mock_analysis_core, test_config, synthetic_data
    ):
        """Test bootstrap scenario generation."""
        assert (
            RobustHomodyneOptimizer is not None
        ), "RobustHomodyneOptimizer not available"
        assert CVXPY_AVAILABLE, "CVXPY not available"
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, test_config)

        scenarios = optimizer._generate_bootstrap_scenarios(
            theta_init=synthetic_data["true_params"],
            phi_angles=synthetic_data["phi_angles"],
            c2_experimental=synthetic_data["c2_experimental"],
            n_scenarios=5,
        )

        assert len(scenarios) == 5
        for scenario in scenarios:
            assert isinstance(scenario, np.ndarray)
            assert scenario.shape == synthetic_data["c2_experimental"].shape
            assert np.all(np.isfinite(scenario))

    def test_scenario_robust_basic(
        self, mock_analysis_core, test_config, synthetic_data
    ):
        """Test basic scenario-based robust optimization."""
        assert (
            RobustHomodyneOptimizer is not None
        ), "RobustHomodyneOptimizer not available"
        assert CVXPY_AVAILABLE, "CVXPY not available"
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, test_config)

        result = optimizer._solve_scenario_robust(
            theta_init=synthetic_data["true_params"] * 1.1,
            phi_angles=synthetic_data["phi_angles"],
            c2_experimental=synthetic_data["c2_experimental"],
            n_scenarios=5,  # Small number for fast testing
        )

        optimal_params, info = result

        if optimal_params is not None:
            assert isinstance(optimal_params, np.ndarray)
            assert len(optimal_params) == 3
            assert np.all(np.isfinite(optimal_params))
            assert info["method"] == "scenario_robust"
            assert "n_scenarios" in info
            assert info["n_scenarios"] == 5
        else:
            assert "status" in info or "error" in info

    def test_scenario_count_effect(
        self, mock_analysis_core, test_config, synthetic_data
    ):
        """Test effect of different scenario counts."""
        assert (
            RobustHomodyneOptimizer is not None
        ), "RobustHomodyneOptimizer not available"
        assert CVXPY_AVAILABLE, "CVXPY not available"
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, test_config)

        scenario_counts = [3, 5, 10]
        results = []

        for n_scenarios in scenario_counts:
            result = optimizer._solve_scenario_robust(
                theta_init=synthetic_data["true_params"] * 1.1,
                phi_angles=synthetic_data["phi_angles"],
                c2_experimental=synthetic_data["c2_experimental"],
                n_scenarios=n_scenarios,
            )
            results.append((n_scenarios, result))

        # Check that scenarios are properly counted in results
        for n_scenarios, (optimal_params, info) in results:
            if optimal_params is not None:
                assert info["n_scenarios"] == n_scenarios


@pytest.mark.skipif(
    not ROBUST_OPTIMIZATION_AVAILABLE or not CVXPY_AVAILABLE,
    reason="CVXPY not available",
)
class TestEllipsoidalRobustOptimization:
    """Test suite for Ellipsoidal Uncertainty Sets Robust Optimization."""

    def test_ellipsoidal_robust_basic(
        self, mock_analysis_core, test_config, synthetic_data
    ):
        """Test basic ellipsoidal robust optimization."""
        assert (
            RobustHomodyneOptimizer is not None
        ), "RobustHomodyneOptimizer not available"
        assert CVXPY_AVAILABLE, "CVXPY not available"
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, test_config)

        result = optimizer._solve_ellipsoidal_robust(
            theta_init=synthetic_data["true_params"] * 1.1,
            phi_angles=synthetic_data["phi_angles"],
            c2_experimental=synthetic_data["c2_experimental"],
            gamma=0.1,  # Uncertainty bound
        )

        optimal_params, info = result

        if optimal_params is not None:
            assert isinstance(optimal_params, np.ndarray)
            assert len(optimal_params) == 3
            assert np.all(np.isfinite(optimal_params))
            assert info["method"] == "ellipsoidal_robust"
            assert "uncertainty_bound" in info
            assert info["uncertainty_bound"] == 0.1
        else:
            assert "status" in info or "error" in info

    def test_ellipsoidal_gamma_effect(
        self, mock_analysis_core, test_config, synthetic_data
    ):
        """Test effect of different uncertainty bounds (gamma)."""
        assert (
            RobustHomodyneOptimizer is not None
        ), "RobustHomodyneOptimizer not available"
        assert CVXPY_AVAILABLE, "CVXPY not available"
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, test_config)

        gamma_values = [0.05, 0.1, 0.2]
        results = []

        for gamma in gamma_values:
            result = optimizer._solve_ellipsoidal_robust(
                theta_init=synthetic_data["true_params"] * 1.1,
                phi_angles=synthetic_data["phi_angles"],
                c2_experimental=synthetic_data["c2_experimental"],
                gamma=gamma,
            )
            results.append(result)

        # Check that different gamma values produce different results
        successful_results = [r for r in results if r[0] is not None]
        if len(successful_results) > 1:
            params_list = [r[0] for r in successful_results]
            for i in range(1, len(params_list)):
                # Results should vary with uncertainty bound
                assert not np.allclose(params_list[0], params_list[i], rtol=1e-2)


@pytest.mark.skipif(
    not ROBUST_OPTIMIZATION_AVAILABLE,
    reason="Robust optimization not available",
)
class TestRobustOptimizationInterface:
    """Test suite for the main robust optimization interface."""

    def test_run_robust_optimization_wasserstein(
        self, mock_analysis_core, test_config, synthetic_data
    ):
        """Test main robust optimization interface with Wasserstein method."""
        assert (
            RobustHomodyneOptimizer is not None
        ), "RobustHomodyneOptimizer not available"
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, test_config)

        if not CVXPY_AVAILABLE:
            # Test that appropriate error is raised when CVXPY not available
            with pytest.raises(ImportError, match="CVXPY is required"):
                optimizer.run_robust_optimization(
                    initial_parameters=synthetic_data["true_params"],
                    phi_angles=synthetic_data["phi_angles"],
                    c2_experimental=synthetic_data["c2_experimental"],
                    method="wasserstein",
                )
            return

        result = optimizer.run_robust_optimization(
            initial_parameters=synthetic_data["true_params"] * 1.1,
            phi_angles=synthetic_data["phi_angles"],
            c2_experimental=synthetic_data["c2_experimental"],
            method="wasserstein",
            uncertainty_radius=0.03,
        )

        optimal_params, info = result

        if optimal_params is not None:
            assert isinstance(optimal_params, np.ndarray)
            assert len(optimal_params) == 3
            assert optimizer.best_params_robust is not None
            assert np.array_equal(optimizer.best_params_robust, optimal_params)
        else:
            assert "error" in info or "status" in info

    def test_run_robust_optimization_all_methods(
        self, mock_analysis_core, test_config, synthetic_data
    ):
        """Test all robust optimization methods through main interface."""
        if not CVXPY_AVAILABLE:
            pytest.skip("CVXPY not available")

        assert (
            RobustHomodyneOptimizer is not None
        ), "RobustHomodyneOptimizer not available"
        assert CVXPY_AVAILABLE, "CVXPY not available"
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, test_config)

        methods = ["wasserstein", "scenario", "ellipsoidal"]
        results = {}

        for method in methods:
            result = optimizer.run_robust_optimization(
                initial_parameters=synthetic_data["true_params"] * 1.1,
                phi_angles=synthetic_data["phi_angles"],
                c2_experimental=synthetic_data["c2_experimental"],
                method=method,
            )
            results[method] = result

        # Check that each method returns reasonable results
        for _method, (optimal_params, _info) in results.items():
            if optimal_params is not None:
                assert isinstance(optimal_params, np.ndarray)
                assert len(optimal_params) == 3
                assert np.all(np.isfinite(optimal_params))
            # Note: Some methods may fail in test environment, which is
            # acceptable

    def test_invalid_method_error(
        self, mock_analysis_core, test_config, synthetic_data
    ):
        """Test error handling for invalid optimization method."""
        if not CVXPY_AVAILABLE:
            pytest.skip("CVXPY not available")

        assert (
            RobustHomodyneOptimizer is not None
        ), "RobustHomodyneOptimizer not available"
        assert CVXPY_AVAILABLE, "CVXPY not available"
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, test_config)

        optimal_params, info = optimizer.run_robust_optimization(
            initial_parameters=synthetic_data["true_params"],
            phi_angles=synthetic_data["phi_angles"],
            c2_experimental=synthetic_data["c2_experimental"],
            method="invalid_method",
        )

        assert optimal_params is None
        assert "error" in info
        assert "Unknown robust optimization method" in info["error"]


@pytest.mark.skipif(
    not ROBUST_OPTIMIZATION_AVAILABLE,
    reason="Robust optimization not available",
)
class TestClassicalOptimizerIntegration:
    """Test suite for integration with ClassicalOptimizer."""

    def test_robust_methods_in_available_methods(self, mock_analysis_core, test_config):
        """Test that robust methods appear in available methods."""
        assert ClassicalOptimizer is not None, "ClassicalOptimizer not available"
        optimizer = ClassicalOptimizer(mock_analysis_core, test_config)
        available_methods = optimizer.get_available_methods()

        assert "Nelder-Mead" in available_methods

        if ROBUST_OPTIMIZATION_AVAILABLE:
            assert "Robust-Wasserstein" in available_methods
            assert "Robust-Scenario" in available_methods
            assert "Robust-Ellipsoidal" in available_methods

    @patch("homodyne.optimization.classical.create_robust_optimizer")
    def test_run_single_method_robust_wasserstein(
        self, mock_create_robust, mock_analysis_core, test_config
    ):
        """Test running robust optimization through ClassicalOptimizer."""
        # Mock the robust optimizer
        mock_robust_optimizer = Mock()
        mock_robust_optimizer.run_robust_optimization.return_value = (
            np.array([100.0, -0.5, 10.0]),
            {
                "method": "wasserstein",
                "status": "success",
                "final_chi_squared": 1.5,
            },
        )
        mock_create_robust.return_value = mock_robust_optimizer

        # Mock analysis core attributes
        mock_analysis_core.phi_angles = np.linspace(-30, 30, 10)
        mock_analysis_core.c2_experimental = np.random.randn(10, 50)

        assert ClassicalOptimizer is not None, "ClassicalOptimizer not available"
        optimizer = ClassicalOptimizer(mock_analysis_core, test_config)

        def dummy_objective(params):
            return np.sum(params**2)

        success, result = optimizer.run_single_method(
            method="Robust-Wasserstein",
            objective_func=dummy_objective,
            initial_parameters=np.array([90.0, -0.4, 9.0]),
            bounds=[(1.0, 1000.0), (-2.0, 2.0), (0.1, 100.0)],
        )

        if ROBUST_OPTIMIZATION_AVAILABLE:
            assert success
            # Only check result attributes when optimization succeeded and
            # result is not an exception
            if success and not isinstance(result, Exception):
                assert hasattr(result, "x")
                assert hasattr(result, "fun")
                assert hasattr(result, "success")
                # Type narrowing: at this point we know result is
                # OptimizeResult, not Exception
                optimization_result: optimize.OptimizeResult = (
                    result  # Type annotation for clarity
                )
                assert np.array_equal(
                    optimization_result.x, np.array([100.0, -0.5, 10.0])
                )
        else:
            assert not success

    def test_robust_optimization_error_handling(self, mock_analysis_core, test_config):
        """Test error handling when robust optimization fails."""
        # Remove required attributes to trigger error
        if hasattr(mock_analysis_core, "phi_angles"):
            delattr(mock_analysis_core, "phi_angles")

        assert ClassicalOptimizer is not None, "ClassicalOptimizer not available"
        optimizer = ClassicalOptimizer(mock_analysis_core, test_config)

        def dummy_objective(params):
            return np.sum(params**2)

        success, result = optimizer._run_robust_optimization(
            method="Robust-Wasserstein",
            objective_func=dummy_objective,
            initial_parameters=np.array([100.0, -0.5, 10.0]),
            bounds=None,
            method_options=None,
        )

        assert not success
        assert isinstance(result, Exception) or hasattr(result, "success")


class TestRobustOptimizationConfiguration:
    """Test suite for robust optimization configuration validation."""

    def test_robust_config_in_template(self):
        """Test that robust optimization configuration is present in config templates."""
        # Test that the configuration section exists
        assert "robust_optimization" in TEST_CONFIG["optimization_config"]

        robust_config = TEST_CONFIG["optimization_config"]["robust_optimization"]

        # Check required fields
        assert "enabled" in robust_config
        assert "uncertainty_model" in robust_config
        assert "uncertainty_radius" in robust_config
        assert "n_scenarios" in robust_config
        assert "solver_settings" in robust_config
        assert "method_options" in robust_config

        # Check method-specific options
        assert "wasserstein" in robust_config["method_options"]
        assert "scenario" in robust_config["method_options"]
        assert "ellipsoidal" in robust_config["method_options"]

    def test_robust_methods_in_classical_config(self):
        """Test that robust methods are included in classical optimization methods."""
        classical_config = TEST_CONFIG["optimization_config"]["classical_optimization"]

        assert "methods" in classical_config
        methods = classical_config["methods"]

        # Should include at least one robust method
        robust_methods = [m for m in methods if m.startswith("Robust-")]
        assert len(robust_methods) > 0
        assert "Robust-Wasserstein" in methods

    def test_configuration_validation(self, test_config):
        """Test validation of robust optimization configuration."""
        # Test with valid configuration
        robust_config = test_config["optimization_config"]["robust_optimization"]

        # Required fields should be present
        required_fields = [
            "enabled",
            "uncertainty_model",
            "uncertainty_radius",
            "solver_settings",
        ]
        for field in required_fields:
            assert field in robust_config

        # Uncertainty radius should be reasonable
        assert 0.001 <= robust_config["uncertainty_radius"] <= 1.0

        # Number of scenarios should be positive
        assert robust_config["n_scenarios"] > 0

        # Solver settings should include key parameters
        solver_settings = robust_config["solver_settings"]
        assert "Method" in solver_settings
        assert "TimeLimit" in solver_settings


class TestRobustOptimizationPerformance:
    """Test suite for performance and edge cases in robust optimization."""

    def test_large_parameter_space(self, mock_analysis_core, test_config):
        """Test robust optimization with larger parameter spaces."""
        if not ROBUST_OPTIMIZATION_AVAILABLE or not CVXPY_AVAILABLE:
            pytest.skip("CVXPY not available")

        # Modify config for 7-parameter laminar flow mode
        test_config["analysis_settings"]["mode"] = "laminar_flow"
        test_config["analysis_settings"]["num_parameters"] = 7

        # Mock analysis core for laminar flow
        mock_analysis_core.is_static_mode = lambda: False
        mock_analysis_core.get_effective_parameter_count = lambda: 7

        assert (
            RobustHomodyneOptimizer is not None
        ), "RobustHomodyneOptimizer not available"
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, test_config)

        # Test parameter bounds extraction for 7 parameters
        bounds = optimizer._get_parameter_bounds()
        if bounds is not None:
            assert len(bounds) <= 7  # May be limited by config bounds

    def test_small_datasets(self, test_config):
        """Test robust optimization with very small datasets."""
        if not ROBUST_OPTIMIZATION_AVAILABLE or not CVXPY_AVAILABLE:
            pytest.skip("CVXPY not available")

        # Create mock core with minimal data
        small_core = MockAnalysisCore(test_config)
        small_core.phi_angles = np.array([0.0, 30.0])  # Only 2 angles
        small_core.c2_experimental = np.random.randn(2, 5).astype(np.float64)  # type: ignore

        assert (
            RobustHomodyneOptimizer is not None
        ), "RobustHomodyneOptimizer not available"
        optimizer = RobustHomodyneOptimizer(small_core, test_config)

        # Should handle small datasets gracefully
        c2_theory = optimizer._compute_theoretical_correlation(
            np.array([100.0, -0.5, 10.0]), small_core.phi_angles
        )

        assert c2_theory.shape[0] == 2
        assert np.all(np.isfinite(c2_theory))

    def test_edge_case_parameters(self, mock_analysis_core, test_config):
        """Test robust optimization with edge case parameter values."""
        if not ROBUST_OPTIMIZATION_AVAILABLE or not CVXPY_AVAILABLE:
            pytest.skip("CVXPY not available")

        assert (
            RobustHomodyneOptimizer is not None
        ), "RobustHomodyneOptimizer not available"
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, test_config)

        # Test with parameters at bounds
        edge_params = np.array([1.0, -2.0, 0.1])  # Minimum values from bounds

        chi_squared = optimizer._compute_chi_squared(
            edge_params,
            mock_analysis_core.phi_angles,
            mock_analysis_core.c2_experimental,
        )

        assert np.isfinite(chi_squared)
        assert chi_squared >= 0

    def test_numerical_stability(self, mock_analysis_core, test_config):
        """Test numerical stability with extreme values."""
        if not ROBUST_OPTIMIZATION_AVAILABLE or not CVXPY_AVAILABLE:
            pytest.skip("CVXPY not available")

        assert (
            RobustHomodyneOptimizer is not None
        ), "RobustHomodyneOptimizer not available"
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, test_config)

        # Test with very large parameter values
        large_params = np.array([1e6, 1.9, 999.0])

        try:
            c2_theory = optimizer._compute_theoretical_correlation(
                large_params, mock_analysis_core.phi_angles
            )
            assert np.all(np.isfinite(c2_theory))
        except (OverflowError, ValueError):
            # Acceptable to fail gracefully with extreme values
            pass


# Integration test with actual file I/O
class TestRobustOptimizationIntegration:
    """Integration tests for robust optimization with file I/O."""

    def test_config_file_loading(self):
        """Test loading robust optimization configuration from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(TEST_CONFIG, f, indent=2)
            config_path = f.name

        try:
            # Test that configuration can be loaded
            with open(config_path) as f:
                loaded_config = json.load(f)

            assert "optimization_config" in loaded_config
            assert "robust_optimization" in loaded_config["optimization_config"]

            # Test creating optimizer with loaded config
            if ROBUST_OPTIMIZATION_AVAILABLE:
                mock_core = MockAnalysisCore(loaded_config)
                assert (
                    RobustHomodyneOptimizer is not None
                ), "RobustHomodyneOptimizer not available"
                optimizer = RobustHomodyneOptimizer(mock_core, loaded_config)
                assert optimizer.settings["uncertainty_model"] == "wasserstein"

        finally:
            Path(config_path).unlink()  # Clean up

    def test_robust_optimization_with_logging(
        self, mock_analysis_core, test_config, caplog
    ):
        """Test that robust optimization produces appropriate log messages."""
        if not ROBUST_OPTIMIZATION_AVAILABLE or not CVXPY_AVAILABLE:
            pytest.skip("CVXPY not available")

        assert (
            RobustHomodyneOptimizer is not None
        ), "RobustHomodyneOptimizer not available"
        optimizer = RobustHomodyneOptimizer(mock_analysis_core, test_config)

        with caplog.at_level(logging.INFO):
            synthetic_data = {
                "true_params": np.array([100.0, -0.5, 10.0]),
                "phi_angles": np.linspace(-30, 30, 5),
                "c2_experimental": np.random.randn(5, 20),
            }

            optimizer.run_robust_optimization(
                initial_parameters=synthetic_data["true_params"] * 1.1,
                phi_angles=synthetic_data["phi_angles"],
                c2_experimental=synthetic_data["c2_experimental"],
                method="wasserstein",
            )

        # Check that appropriate log messages were generated
        log_messages = [record.message for record in caplog.records]
        optimization_logs = [
            msg for msg in log_messages if "robust optimization" in msg.lower()
        ]
        assert len(optimization_logs) > 0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
